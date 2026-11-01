//! Per-model usage metering: in-memory daily buckets flushed to the
//! store's `usage_daily` rollups. Recording is O(1) on the request
//! path (hash bump, no I/O); `flush` is the only writer, called from
//! the periodic task and opportunistically before reads. Token counts
//! come from each dialect's final usage JSON (OpenAI/Anthropic/ollama
//! shapes all land in the response tail), so a stream that never sends
//! usage still counts its request + generation window — never a fake
//! token number.

use std::collections::HashMap;
use std::sync::Mutex;

use blazar_core::BlazarDirs;
use blazar_core::store::{Store, UsageDayRow, UsageDelta};

/// Accumulator alias: the gateway bucket IS a core [`UsageDelta`].
#[allow(clippy::struct_field_names)] // token/column naming mirrors usage_daily
type UsageAcc = UsageDelta;

/// UTC day key (`yyyymmdd`). Usage days are UTC buckets — the same
/// convention the OpenAI/Anthropic usage APIs use for server-side
/// accounting; local-date bucketing would need a date dependency for
/// a cosmetic difference.
fn today_key() -> i64 {
    let secs = blazar_core::store::unix_now();
    let days = secs.div_euclid(86_400);
    // Civil-from-days (Howard Hinnant's algorithm): no date crate.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (yoe * 365 + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    y * 10_000 + m * 100 + d
}

/// Shared meter. `record` is synchronous and lock-scoped (cheap);
/// `flush` drains everything into the store in one pass. The store
/// handle lives behind a `Mutex<Option<..>>` and re-opens lazily per
/// flush — a transient open failure (locked db during a CLI write)
/// degrades one flush, never the request path, and self-heals on the
/// next tick. Same shape as the `AppState`'s cached store connection.
pub struct UsageMeter {
    buckets: Mutex<HashMap<(i64, String), UsageAcc>>,
    store: Mutex<Option<Store>>,
    dirs: BlazarDirs,
}

impl UsageMeter {
    #[must_use]
    pub fn new(dirs: &BlazarDirs) -> Self {
        Self {
            buckets: Mutex::new(HashMap::new()),
            store: Mutex::new(Store::open(dirs).ok()),
            dirs: dirs.clone(),
        }
    }

    /// One finished generation request. `ttft_ms` is `None` when the
    /// lane could not measure first-token latency honestly.
    pub fn record(
        &self,
        model: &str,
        in_tokens: i64,
        out_tokens: i64,
        cached_tokens: i64,
        gen_secs: f64,
        ttft_ms: Option<f64>,
    ) {
        let mut map = self.buckets.lock().expect("usage buckets poisoned");
        let acc = map.entry((today_key(), model.to_string())).or_default();
        acc.in_tokens += in_tokens;
        acc.out_tokens += out_tokens;
        acc.cached_tokens += cached_tokens;
        acc.requests += 1;
        acc.gen_secs += gen_secs.max(0.0);
        if let Some(ms) = ttft_ms.filter(|m| m.is_finite() && *m >= 0.0) {
            acc.ttft_count += 1;
            acc.ttft_ms_total += ms;
        }
    }

    /// Drain in-memory buckets into `usage_daily` in ONE store
    /// transaction. Returns rows written. Failure-safe by construction:
    /// the store is ensured BEFORE anything is drained (an unopenable
    /// database defers the flush with the batch untouched), and a
    /// failed batch write requeues the whole drained batch additively —
    /// `add_usage_deltas` is all-or-nothing, so a mid-batch failure
    /// persists nothing and the retry next tick re-offers exactly what
    /// was measured, plus whatever `record` accumulated meanwhile.
    /// Errors propagate (fail loud; the caller's periodic task logs
    /// them and tries again next tick).
    pub fn flush(&self) -> anyhow::Result<usize> {
        {
            let mut guard = self.store.lock().expect("usage store poisoned");
            if guard.is_none() {
                *guard = Store::open(&self.dirs).ok();
                if guard.is_none() {
                    tracing::warn!(
                        db = %self.dirs.db_file().display(),
                        "usage flush deferred: store open failed (batch kept for next tick)"
                    );
                    return Ok(0);
                }
            }
        }
        let drained: Vec<((i64, String), UsageAcc)> = {
            let mut map = self.buckets.lock().expect("usage buckets poisoned");
            map.drain().collect()
        };
        if drained.is_empty() {
            return Ok(0);
        }
        let n = drained.len();
        let batch: Vec<(i64, String, UsageDelta)> = drained
            .into_iter()
            .map(|((day, model), a)| (day, model, a))
            .collect();
        let guard = self.store.lock().expect("usage store poisoned");
        let store = guard.as_ref().expect("store ensured above");
        if let Err(e) = store.add_usage_deltas(&batch) {
            drop(guard);
            requeue_additively(&self.buckets, &batch);
            return Err(e.into());
        }
        Ok(n)
    }

    /// Read path: flush first so the answer includes the live window.
    pub fn snapshot(&self, days: i64) -> anyhow::Result<Vec<UsageDayRow>> {
        self.flush()?;
        let guard = self.store.lock().expect("usage store poisoned");
        let Some(store) = guard.as_ref() else {
            return Ok(Vec::new());
        };
        Ok(store.list_usage_days(days)?)
    }
}

/// Fold a failed flush-batch back into the live buckets, ADDITIVELY:
/// `record` calls that landed while the batch was being written must
/// survive — assignment would silently erase them. Field-by-field
/// because the accumulator type is core's [`UsageDelta`] and orphan
/// rules forbid an `AddAssign` impl here.
fn requeue_additively(
    buckets: &Mutex<HashMap<(i64, String), UsageAcc>>,
    batch: &[(i64, String, UsageDelta)],
) {
    let mut map = buckets.lock().expect("usage buckets poisoned");
    for (day, model, d) in batch {
        let acc = map.entry((*day, model.clone())).or_default();
        acc.in_tokens += d.in_tokens;
        acc.out_tokens += d.out_tokens;
        acc.cached_tokens += d.cached_tokens;
        acc.requests += d.requests;
        acc.gen_secs += d.gen_secs;
        acc.ttft_count += d.ttft_count;
        acc.ttft_ms_total += d.ttft_ms_total;
    }
}

/// Extract (prompt, completion, cached) tokens from a response tail
/// across dialects. Pure; `None` fields stay unknown.
pub fn sniff_usage_full(tail: &[u8]) -> (Option<i64>, Option<i64>, Option<i64>) {
    // Scan NDJSON/SSE tails from the END: the final usage chunk is the
    // authoritative one (streaming lanes repeat partial counts).
    let text = String::from_utf8_lossy(tail);
    let mut prompt = None;
    let mut completion = None;
    let mut cached = None;
    for line in text.lines().rev() {
        // `data: ` prefix for SSE; bare JSON for ollama NDJSON and
        // non-streaming bodies; anything else cannot hold usage.
        let json = line
            .strip_prefix("data: ")
            .unwrap_or(if line.starts_with('{') { line } else { "" });
        let Ok(v) = serde_json::from_str::<serde_json::Value>(json) else {
            continue;
        };
        // OpenAI/Anthropic nest usage under "usage"; ollama NDJSON
        // puts the same fields at the top level.
        let usage_obj = v.get("usage").unwrap_or(&v);
        let prompt_now = usage_obj
            .get("prompt_tokens")
            .or_else(|| usage_obj.get("input_tokens"))
            .or_else(|| usage_obj.get("prompt_eval_count"))
            .and_then(serde_json::Value::as_i64);
        let completion_now = usage_obj
            .get("completion_tokens")
            .or_else(|| usage_obj.get("output_tokens"))
            .or_else(|| usage_obj.get("eval_count"))
            .and_then(serde_json::Value::as_i64);
        let cached_now = usage_obj
            .get("prompt_tokens_details")
            .and_then(|details| details.get("cached_tokens"))
            .or_else(|| usage_obj.get("cache_read_input_tokens"))
            .and_then(serde_json::Value::as_i64);
        if prompt.is_none() {
            prompt = prompt_now;
        }
        if completion.is_none() {
            completion = completion_now;
        }
        if cached.is_none() {
            cached = cached_now;
        }
        if prompt.is_some() && completion.is_some() {
            break;
        }
    }
    (prompt, completion, cached)
}

/// Paths whose responses carry generation usage. Control/plane routes
/// (models, tags, capacity) are not generation and stay unmetered.
pub const USAGE_PATHS: &[&str] = &[
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/responses",
    "/v1/messages",
    "/api/chat",
    "/api/generate",
];

/// Trailing response-body window the usage tap retains while a stream
/// flows. Streaming lanes put the usage object in a small final
/// SSE/NDJSON line, but a NON-streaming reply serializes the whole
/// completion as ONE JSON line — long generations exceed 8 KiB, and a
/// front-truncated line no longer parses (`from_str` needs the opening
/// brace). 64 KiB covers non-stream replies up to ~16k output tokens
/// while staying trivial per in-flight request; beyond it the sniff
/// degrades to uncounted tokens (requests still counted).
pub const TAIL_LIMIT: usize = 64 * 1024;

/// `GET /api/usage?days=N` (default 7, clamped 1..=365): per-(day,
/// model) token/latency rollups. Flushing the in-memory buckets first
/// means the endpoint never shows stale-but-recorded traffic.
pub async fn usage_api(
    axum::extract::State(state): axum::extract::State<std::sync::Arc<crate::state::AppState>>,
    axum::extract::Query(q): axum::extract::Query<UsageQuery>,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    let days = q.days.unwrap_or(7).clamp(1, 365);
    let rows = match state.usage.snapshot(days) {
        Ok(r) => r,
        Err(e) => {
            return (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                format!("usage store: {e}"),
            )
                .into_response();
        }
    };
    axum::Json(serde_json::json!({
        "object": "blazar.usage.daily",
        "days_window": days,
        "days": rows,
    }))
    .into_response()
}

#[derive(serde::Deserialize)]
pub struct UsageQuery {
    pub days: Option<i64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(non_snake_case)] // repo test-name convention: unit__unit__scenario
    fn unit__sniff_usage_full__dialects_and_order() {
        // OpenAI SSE with stream_options.include_usage: final data
        // chunk carries the authoritative usage.
        let sse = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n",
            "data: {\"usage\":{\"prompt_tokens\":120,\"completion_tokens\":34,\"prompt_tokens_details\":{\"cached_tokens\":96}}}\n\n"
        );
        assert_eq!(
            sniff_usage_full(sse.as_bytes()),
            (Some(120), Some(34), Some(96))
        );
        // Anthropic: input/output + cache_read_input_tokens.
        let anthropic = concat!(
            "event: message_delta\n",
            "data: {\"usage\":{\"output_tokens\":21,\"input_tokens\":55,\"cache_read_input_tokens\":40}}\n\n"
        );
        assert_eq!(
            sniff_usage_full(anthropic.as_bytes()),
            (Some(55), Some(21), Some(40))
        );
        // ollama NDJSON final line.
        let ndjson = "{\"prompt_eval_count\":9,\"eval_count\":4}\n";
        assert_eq!(
            sniff_usage_full(ndjson.as_bytes()),
            (Some(9), Some(4), None)
        );
        // Non-stream single-line body larger than the historical 8 KiB
        // window: a long completion serializes as ONE JSON line, and a
        // front-truncated line cannot parse. The tap's TAIL_LIMIT must
        // keep the whole final line for this shape (regression pin for
        // the 8 KiB undercount bug).
        let filler = "x".repeat(9 * 1024);
        let big = format!(
            "{{\"message\":{{\"content\":\"{filler}\"}},\"prompt_eval_count\":77,\"eval_count\":31}}"
        );
        assert!(big.len() > 8 * 1024);
        assert!(
            big.len() < TAIL_LIMIT,
            "TAIL_LIMIT must cover >8 KiB single-line bodies (undercount bug)"
        );
        assert_eq!(sniff_usage_full(big.as_bytes()), (Some(77), Some(31), None));
        // No usage anywhere: all unknown, never zeros.
        let bare = "data: {\"choices\":[{\"delta\":{}}]}\n\n";
        assert_eq!(sniff_usage_full(bare.as_bytes()), (None, None, None));
    }

    #[test]
    #[allow(non_snake_case)] // repo test-name convention: unit__unit__scenario
    fn unit__usage_meter__record_flush_snapshot_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let dirs = BlazarDirs {
            config_dir: tmp.path().join("cfg"),
            data_dir: tmp.path().join("data"),
        };
        let m = UsageMeter::new(&dirs);
        m.record("m1", 100, 40, 80, 2.5, Some(120.0));
        m.record("m1", 50, 10, 0, 1.5, None);
        m.record("m2", 7, 3, 0, 1.0, Some(90.0));
        let n = m.flush().unwrap();
        assert_eq!(n, 2, "two (day, model) buckets");
        let rows = m.snapshot(365).unwrap();
        let m1 = rows.iter().find(|r| r.model == "m1").unwrap();
        assert_eq!((m1.in_tokens, m1.out_tokens, m1.requests), (150, 50, 2));
        assert!((m1.gen_secs - 4.0).abs() < 1e-9);
        // TTFT: only the measured sample counts.
        assert_eq!(m1.ttft_count, 1);
        assert!((m1.ttft_ms_total - 120.0).abs() < 1e-9);
    }

    /// Pin for the flush-ordering bug: the old flush drained buckets
    /// BEFORE ensuring the store, so an unopenable database silently
    /// discarded the whole in-memory batch (the "will retry next tick"
    /// warning lied — nothing remained to retry). Deferred-open must
    /// keep the batch, and the later successful flush must land it.
    #[test]
    #[allow(non_snake_case)] // repo test-name convention: integration__unit__scenario
    fn integration__usage_flush__open_failure_keeps_batch_and_lands_later() {
        let tmp = tempfile::tempdir().unwrap();
        let dirs = BlazarDirs {
            config_dir: tmp.path().join("cfg"),
            data_dir: tmp.path().join("data"),
        };
        // Block the data dir with a regular file: no db can be created
        // inside a file path, so every Store::open fails.
        std::fs::create_dir_all(tmp.path().join("cfg")).unwrap();
        std::fs::write(tmp.path().join("data"), b"not a directory").unwrap();
        let m = UsageMeter::new(&dirs); // eager open fails -> None
        m.record("m1", 100, 40, 80, 2.5, Some(120.0));
        assert_eq!(
            m.flush().unwrap(),
            0,
            "open failure defers the flush — the batch must survive"
        );
        // Unblock: the very same meter must land exactly what was
        // recorded, proving nothing was drained into the void.
        std::fs::remove_file(tmp.path().join("data")).unwrap();
        assert_eq!(m.flush().unwrap(), 1, "the deferred batch lands on retry");
        let rows = m.snapshot(365).unwrap();
        let m1 = rows
            .iter()
            .find(|r| r.model == "m1")
            .expect("deferred record must exist after recovery");
        assert_eq!((m1.in_tokens, m1.out_tokens, m1.requests), (100, 40, 1));
        assert_eq!(m.flush().unwrap(), 0, "exactly-once: nothing left to flush");
    }

    /// Pin for the mid-batch loss: the old per-row loop aborted on the
    /// first failing row, leaving a partially-written day and no
    /// requeue. The transactional batch must fail all-or-nothing and
    /// requeue the whole drained set; records arriving during the
    /// failure window must merge in additively (never be overwritten).
    #[test]
    #[allow(non_snake_case)] // repo test-name convention: integration__unit__scenario
    fn integration__usage_flush__midbatch_failure_requeues_exactly_once() {
        let tmp = tempfile::tempdir().unwrap();
        let dirs = BlazarDirs {
            config_dir: tmp.path().join("cfg"),
            data_dir: tmp.path().join("data"),
        };
        let m = UsageMeter::new(&dirs);
        m.record("m1", 100, 40, 80, 2.5, Some(120.0));
        m.record("m2", 7, 3, 0, 1.0, None);
        // Sabotage the rollup table from a second connection.
        {
            let s2 = Store::open(&dirs).unwrap();
            s2.conn().execute_batch("DROP TABLE usage_daily;").unwrap();
        }
        assert!(
            m.flush().is_err(),
            "batch write must fail loudly with the table gone"
        );
        // A record landing during the failure window must survive the
        // requeue merge (additive, not overwrite).
        m.record("m1", 10, 5, 0, 0.5, None);
        // Repair the table (reopen alone does not replay the schema at
        // a current user_version, by design).
        {
            let s3 = Store::open(&dirs).unwrap();
            s3.conn()
                .execute_batch(
                    "CREATE TABLE IF NOT EXISTS usage_daily (
                         day           INTEGER NOT NULL,
                         model         TEXT NOT NULL,
                         in_tokens     INTEGER NOT NULL DEFAULT 0,
                         out_tokens    INTEGER NOT NULL DEFAULT 0,
                         cached_tokens INTEGER NOT NULL DEFAULT 0,
                         requests      INTEGER NOT NULL DEFAULT 0,
                         gen_secs      REAL NOT NULL DEFAULT 0,
                         ttft_count    INTEGER NOT NULL DEFAULT 0,
                         ttft_ms_total REAL NOT NULL DEFAULT 0,
                         PRIMARY KEY (day, model)
                     );",
                )
                .unwrap();
        }
        assert_eq!(m.flush().unwrap(), 2, "requeued batch lands whole");
        let rows = m.snapshot(365).unwrap();
        let m1 = rows.iter().find(|r| r.model == "m1").unwrap();
        assert_eq!(
            (m1.in_tokens, m1.out_tokens, m1.requests),
            (110, 45, 2),
            "original + failure-window record, merged additively"
        );
        let m2 = rows.iter().find(|r| r.model == "m2").unwrap();
        assert_eq!((m2.in_tokens, m2.out_tokens), (7, 3));
        assert_eq!(m.flush().unwrap(), 0, "exactly-once: nothing left");
    }
}
