//! OTLP/HTTP-JSON trace export (default OFF, `otlp_endpoint = "..."`).
//!
//! One span per gateway request (method/path/status/duration/model/key),
//! shipped to an OpenTelemetry collector in batches. Bounded by design:
//! the hot path `try_send`s into a fixed channel and never blocks; an
//! unreachable collector drops batches with a throttled warning. This is
//! intentionally a hand-rolled minimal exporter — the full opentelemetry
//! crate family would dwarf blazar's gateway for a local-first server.

use blazar_core::Config;

/// One recorded span (plain data; JSON shaping happens in the flusher).
pub struct OtlpSpan {
    pub trace_id_hex: String,
    pub span_id_hex: String,
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
    /// Spans never exported (channel saturated or POST failed) —
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
    /// `ExportTraceServiceRequest` shape — no extra fields).
    fn encode(&self, spans: &[OtlpSpan]) -> serde_json::Value {
        let otlp_spans: Vec<serde_json::Value> = spans
            .iter()
            .map(|s| {
                let error = (400..600).contains(&s.http_status);
                serde_json::json!({
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
                })
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
            async move { otlp.run(reqwest::Client::new()).await }
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
