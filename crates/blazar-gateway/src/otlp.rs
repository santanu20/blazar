//! OTLP/HTTP-JSON trace export (default OFF, `otlp_endpoint = "..."`).
//!
//! One span per gateway request (method/path/status/duration/model/key),
//! shipped to an OpenTelemetry collector in batches. Bounded by design:
//! the hot path `try_send`s into a fixed channel and never blocks; an
//! unreachable collector drops batches with a throttled warning. This is
//! intentionally a hand-rolled minimal exporter - the full opentelemetry
//! crate family would dwarf blazar's gateway for a local-first server.

use blazar_core::Config;

/// One recorded span (plain data; JSON shaping happens in the flusher).
pub struct OtlpSpan {
    pub trace_id_hex: String,
    pub span_id_hex: String,
    /// The caller's operation when the request continued a trace.
    pub parent_span_id_hex: Option<String>,
    /// W3C `tracestate` that traveled with the request.
    pub trace_state: Option<String>,
    /// W3C trace flags of the continued trace; 0 for a new one.
    pub trace_flags: u8,
    pub name: String,
    pub start_unix_nano: u128,
    pub end_unix_nano: u128,
    pub http_status: u16,
    pub attributes: Vec<(String, String)>,
}

pub struct Otlp {
    enabled: bool,
    endpoint: String,
    service: String,
    tx: tokio::sync::mpsc::Sender<OtlpSpan>,
    rx: std::sync::Mutex<Option<tokio::sync::mpsc::Receiver<OtlpSpan>>>,
    /// Spans never exported (channel saturated or POST failed) -
    /// surfaced via `/metrics` as `blazar_otlp_dropped_total` (F81).
    pub dropped: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

const CHANNEL_CAP: usize = 1024;

impl Otlp {
    #[must_use]
    pub fn new(config: &Config) -> Self {
        let (tx, rx) = tokio::sync::mpsc::channel(CHANNEL_CAP);
        Self {
            enabled: !config.otlp_endpoint.is_empty(),
            endpoint: config.otlp_endpoint.clone(),
            service: if config.otlp_service.is_empty() {
                "blazar".to_string()
            } else {
                config.otlp_service.clone()
            },
            tx,
            rx: std::sync::Mutex::new(Some(rx)),
            dropped: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Hot path: never blocks, drops on overflow (observability must not
    /// become backpressure).
    pub fn record(&self, span: OtlpSpan) {
        if !self.enabled {
            return;
        }
        if self.tx.try_send(span).is_err() {
            self.dropped
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Background flusher: batch up to 256 spans / 5 s onto
    /// `{endpoint}/v1/traces`. Run only when enabled.
    pub async fn run(self: std::sync::Arc<Self>, http: reqwest::Client) {
        let Some(mut rx) = self.rx.lock().expect("otlp rx").take() else {
            return;
        };
        let url = format!("{}/v1/traces", self.endpoint.trim_end_matches('/'));
        let mut warned = false;
        loop {
            let mut batch = Vec::with_capacity(256);
            tokio::select! {
                maybe = rx.recv() => {
                    match maybe {
                        Some(s) => batch.push(s),
                        None => break,
                    }
                }
                () = tokio::time::sleep(std::time::Duration::from_secs(5)) => {}
            }
            // Drain whatever is immediately pending (up to cap).
            while batch.len() < 256 {
                match rx.try_recv() {
                    Ok(s) => batch.push(s),
                    Err(_) => break,
                }
            }
            if batch.is_empty() {
                continue;
            }
            let body = self.encode(&batch);
            match http.post(&url).json(&body).send().await {
                Ok(r) if r.status().is_success() => warned = false,
                Ok(r) => {
                    if !warned {
                        tracing::warn!(target: "blazar::otlp", "collector returned {}", r.status());
                        warned = true;
                    }
                    self.dropped.fetch_add(
                        u64::try_from(batch.len()).unwrap_or(u64::MAX),
                        std::sync::atomic::Ordering::Relaxed,
                    );
                }
                Err(e) => {
                    if !warned {
                        tracing::warn!(target: "blazar::otlp", "collector unreachable: {e}");
                        warned = true;
                    }
                    self.dropped.fetch_add(
                        u64::try_from(batch.len()).unwrap_or(u64::MAX),
                        std::sync::atomic::Ordering::Relaxed,
                    );
                }
            }
        }
    }

    /// OTLP/HTTP-JSON trace payload (protobuf-JSON mapping; strictly the
    /// `ExportTraceServiceRequest` shape - no extra fields).
    fn encode(&self, spans: &[OtlpSpan]) -> serde_json::Value {
        let otlp_spans: Vec<serde_json::Value> = spans
            .iter()
            .map(|s| {
                let error = (400..600).contains(&s.http_status);
                let mut span = serde_json::json!({
                    "traceId": s.trace_id_hex,
                    "spanId": s.span_id_hex,
                    "name": s.name,
                    "kind": 2, // SERVER
                    "startTimeUnixNano": s.start_unix_nano.to_string(),
                    "endTimeUnixNano": s.end_unix_nano.to_string(),
                    "status": if error {
                        serde_json::json!({"code": 2, "message": format!("HTTP {}", s.http_status)})
                    } else {
                        serde_json::json!({"code": 1})
                    },
                    "attributes": s.attributes.iter().map(|(k, v)| serde_json::json!({
                        "key": k, "value": {"stringValue": v}
                    })).collect::<Vec<_>>(),
                });
                // Only a continued trace carries these: a fresh trace
                // exports the same shape it always did.
                if let Some(parent) = &s.parent_span_id_hex {
                    span["parentSpanId"] = serde_json::json!(parent);
                }
                if let Some(state) = &s.trace_state {
                    span["traceState"] = serde_json::json!(state);
                }
                if s.trace_flags != 0 {
                    span["flags"] = serde_json::json!(s.trace_flags);
                }
                span
            })
            .collect();
        serde_json::json!({
            "resourceSpans": [{
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": self.service}},
                        {"key": "telemetry.sdk.name", "value": {"stringValue": "blazar-otlp"}},
                    ]
                },
                "scopeSpans": [{
                    "scope": {"name": "blazar-gateway"},
                    "spans": otlp_spans,
                }],
            }],
        })
    }
}

/// Stream-completion span state: the span's `end_unix_nano` moves
/// from response-header time to the moment the last body chunk is
/// consumed (or the stream errors, or the body is dropped without a
/// drain - client aborts included). Exactly-once via `Option::take`;
/// `Drop` is the backstop for bodies nobody reads.
struct SpanSt {
    otlp: std::sync::Arc<Otlp>,
    span: Option<OtlpSpan>,
    inner: axum::body::BodyDataStream,
}

impl SpanSt {
    fn finish(&mut self) {
        if let Some(mut span) = self.span.take() {
            span.end_unix_nano = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(span.start_unix_nano, |d| d.as_nanos());
            self.otlp.record(span);
        }
    }
}

impl Drop for SpanSt {
    fn drop(&mut self) {
        self.finish();
    }
}

/// Wrap `resp` so its OTLP span closes at stream completion instead
/// of header time: streaming lanes keep generating after the handler
/// returns, so exporting at header time under-reports their true
/// duration. Pass-through bytes, zero buffering.
pub(crate) fn spanned(
    otlp: std::sync::Arc<Otlp>,
    span: OtlpSpan,
    resp: axum::response::Response,
) -> axum::response::Response {
    let (parts, body) = resp.into_parts();
    let st = SpanSt {
        otlp,
        span: Some(span),
        inner: body.into_data_stream(),
    };
    let stream = futures::stream::unfold(st, |mut st| async move {
        use futures::StreamExt;
        match st.inner.next().await {
            Some(Ok(bytes)) => Some((Ok(bytes), st)),
            Some(Err(e)) => {
                st.finish();
                Some((Err(std::io::Error::other(e)), st))
            }
            None => {
                st.finish();
                None
            }
        }
    });
    axum::response::Response::from_parts(parts, axum::body::Body::from_stream(stream))
}

/// Random 32-hex trace id / 16-hex span id from OS entropy.
#[must_use]
pub fn new_ids() -> (String, String) {
    let mut buf = [0u8; 24];
    getrandom::fill(&mut buf).expect("OS entropy source");
    let mut hex = String::with_capacity(48);
    for b in buf {
        use std::fmt::Write as _;
        let _ = write!(hex, "{b:02x}");
    }
    (hex[..32].to_string(), hex[32..48].to_string())
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    fn span(status: u16) -> OtlpSpan {
        OtlpSpan {
            trace_id_hex: "a".repeat(32),
            span_id_hex: "b".repeat(16),
            parent_span_id_hex: None,
            trace_state: None,
            trace_flags: 0,
            name: "POST /v1/chat/completions".into(),
            start_unix_nano: 1,
            end_unix_nano: 5_000_000,
            http_status: status,
            attributes: vec![("blazar.trace_id".into(), "plm-x".into())],
        }
    }

    #[test]
    fn unit__encode__strict_export_shape() {
        let otlp = Otlp::new(&Config::default());
        let v = otlp.encode(&[span(200), span(500)]);
        let keys: Vec<&str> = v.as_object().unwrap().keys().map(String::as_str).collect();
        assert_eq!(
            keys,
            vec!["resourceSpans"],
            "no extra top-level fields: {keys:?}"
        );
        let sp = &v["resourceSpans"][0]["scopeSpans"][0]["spans"];
        assert_eq!(sp.as_array().unwrap().len(), 2);
        assert_eq!(sp[0]["status"]["code"], 1);
        assert_eq!(sp[1]["status"]["code"], 2);
        assert_eq!(sp[0]["traceId"].as_str().unwrap().len(), 32);
        assert_eq!(sp[0]["spanId"].as_str().unwrap().len(), 16);
        let svc = &v["resourceSpans"][0]["resource"]["attributes"][0]["value"]["stringValue"];
        assert_eq!(svc, "blazar");
        for absent in ["parentSpanId", "traceState", "flags"] {
            assert!(
                sp[0].get(absent).is_none(),
                "a fresh trace exports no {absent}"
            );
        }
    }

    #[test]
    fn unit__encode__a_continued_trace_names_its_parent_state_and_flags() {
        let otlp = Otlp::new(&Config::default());
        let continued = OtlpSpan {
            parent_span_id_hex: Some("c".repeat(16)),
            trace_state: Some("vendor=value".into()),
            trace_flags: 1,
            ..span(200)
        };
        let v = otlp.encode(&[continued]);
        let span = &v["resourceSpans"][0]["scopeSpans"][0]["spans"][0];
        assert_eq!(span["parentSpanId"], "c".repeat(16));
        assert_eq!(span["traceState"], "vendor=value");
        assert_eq!(span["flags"], 1);
    }

    #[tokio::test]
    async fn unit__spanned__records_exactly_one_span_at_eof() {
        let cfg = Config {
            otlp_endpoint: "http://127.0.0.1:9".into(),
            ..Config::default()
        };
        let otlp = std::sync::Arc::new(Otlp::new(&cfg));
        let mut rx = otlp.rx.lock().expect("otlp rx").take().expect("rx present");
        let chunks = futures::stream::iter([
            Ok::<axum::body::Bytes, std::io::Error>(axum::body::Bytes::from_static(b"he")),
            Ok(axum::body::Bytes::from_static(b"llo")),
        ]);
        let resp = spanned(
            std::sync::Arc::clone(&otlp),
            span(200),
            axum::response::Response::new(axum::body::Body::from_stream(chunks)),
        );
        let drained = axum::body::to_bytes(resp.into_body(), 1 << 20)
            .await
            .expect("body drains");
        assert_eq!(&drained[..], b"hello", "bytes pass through untouched");
        let s = rx.try_recv().expect("span recorded at EOF");
        assert!(
            s.end_unix_nano >= s.start_unix_nano,
            "end moves to completion"
        );
        assert_eq!(s.http_status, 200, "status preserved");
        assert_eq!(s.name, "POST /v1/chat/completions");
        assert!(
            rx.try_recv().is_err(),
            "exactly one span - Drop after EOF is a no-op"
        );
    }

    #[test]
    fn unit__spanned__drop_without_consume_still_records() {
        let cfg = Config {
            otlp_endpoint: "http://127.0.0.1:9".into(),
            ..Config::default()
        };
        let otlp = std::sync::Arc::new(Otlp::new(&cfg));
        let mut rx = otlp.rx.lock().expect("otlp rx").take().expect("rx present");
        // A body that is never polled: only the Drop backstop can close
        // the span (client-abort shape).
        let resp = spanned(
            std::sync::Arc::clone(&otlp),
            span(500),
            axum::response::Response::new(axum::body::Body::empty()),
        );
        drop(resp);
        let s = rx.try_recv().expect("drop backstop records the span");
        assert_eq!(s.http_status, 500);
    }

    #[tokio::test]
    #[allow(clippy::items_after_statements)] // io traits near their only use
    async fn unit__flusher__posts_batches_to_collector() {
        // Capture server: one HTTP POST, respond 200.
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
            .await
            .unwrap();
        let port = listener.local_addr().unwrap().port();
        let cfg = Config {
            otlp_endpoint: format!("http://127.0.0.1:{port}"),
            ..Config::default()
        };
        let otlp = std::sync::Arc::new(Otlp::new(&cfg));
        assert!(otlp.enabled());
        otlp.record(span(200));
        let task = tokio::spawn({
            let otlp = std::sync::Arc::clone(&otlp);
            async move {
                blazar_core::tls::ensure_tls_provider();
                otlp.run(reqwest::Client::new()).await;
            }
        });
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let (mut sock, _) = listener.accept().await.unwrap();
        let mut buf = vec![0u8; 64 * 1024];
        let n = sock.read(&mut buf).await.unwrap();
        let req = String::from_utf8_lossy(&buf[..n]).to_string();
        sock.write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 0\r\n\r\n")
            .await
            .unwrap();
        assert!(req.starts_with("POST /v1/traces "), "{req}");
        assert!(req.contains("\"service.name\""), "{req}");
        assert!(req.contains("plm-x"), "span attribute exported: {req}");
        task.abort();
    }
}
