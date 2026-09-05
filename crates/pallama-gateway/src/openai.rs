//! OpenAI-compatible surface: byte-faithful proxy to the child engine.
//! Pallama never parses or rewrites these bodies — tool calls, structured
//! output, logprobs, `stream_options` ride through untouched (complaint #1
//! fidelity argument).

use std::sync::Arc;

use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{HeaderMap, Method, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use serde_json::json;

use pallama_core::store::Store;

use crate::proxy::{admission_gate, ensure_with_admission, openai_error, path_and_query, proxy_request};
use crate::queue::Priority;
use crate::state::AppState;
use crate::TraceId;

use axum::Extension;

/// GET /v1/models — synthesized from the local store (any pulled model is
/// servable; the engine is hot-swapped underneath).
pub async fn models(State(state): State<Arc<AppState>>) -> Response {
    let store = match Store::open(&state.dirs) {
        Ok(s) => s,
        Err(e) => return openai_error(500, &e.to_string()),
    };
    let list = match store.list_models() {
        Ok(l) => l,
        Err(e) => return openai_error(500, &e.to_string()),
    };
    let data: Vec<serde_json::Value> = list
        .iter()
        .map(|m| {
            json!({
                "id": format!("{}:{}", m.name, m.quant.to_lowercase()),
                "object": "model",
                "owned_by": "pallama",
                "created": m.pulled_at,
            })
        })
        .collect();
    axum::Json(json!({"object": "list", "data": data})).into_response()
}

/// All POST /v1/* traffic: one handler, one proxy path, zero body
/// rewriting. `X-Pallama-Priority` orders admission under load.
pub async fn openai_proxy(
    State(state): State<Arc<AppState>>,
    trace_ext: Option<Extension<TraceId>>,
    uri: Uri,
    method: Method,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let model = extract_model(&body).or_else(|| {
        let ct = headers
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        if ct.starts_with("multipart/form-data") {
            extract_model_multipart(&body, ct)
        } else {
            None
        }
    });
    let Some(model) = model else {
        return openai_error(400, "missing `model` field in request body");
    };
    let priority = Priority::from_header(
        headers
            .get("x-pallama-priority")
            .and_then(|v| v.to_str().ok()),
    );

    // X-Pallama-Num-Ctx: per-request ctx on the OpenAI path — the
    // protocol has no such field, the extension header fills the gap
    // (same restart-once semantics as options.num_ctx on the ollama API).
    if let Some(raw) = headers.get("x-pallama-num-ctx").and_then(|v| v.to_str().ok()) {
        let want: i64 = match raw.parse() {
            Ok(w) => w,
            Err(_) => {
                return openai_error(400, &format!("X-Pallama-Num-Ctx: not a number: {raw:?}"))
            }
        };
        if want <= 0 {
            return openai_error(400, "X-Pallama-Num-Ctx must be positive");
        }
        if let Err(resp) = crate::ollama::apply_num_ctx(&state, &model, want).await {
            return resp;
        }
    }

    let (engine, load_ms) = match ensure_with_admission(&state, &model, priority).await {
        Ok(ok) => ok,
        Err(resp) => return resp,
    };
    let model_name = engine.name.clone();

    let guard = match admission_gate(&state, &model_name, priority).await {
        Ok(g) => g,
        Err(resp) => return resp,
    };
    proxy_request(
        &state,
        &engine,
        &model_name,
        &method,
        &path_and_query(&uri),
        &headers,
        body,
        load_ms,
        trace_ext.map(|Extension(t)| t.0),
        Some(guard),
    )
    .await
}

fn extract_model(body: &[u8]) -> Option<String> {
    let v: serde_json::Value = serde_json::from_slice(body).ok()?;
    v.get("model")?.as_str().map(str::to_string)
}

/// Extract the `model` field from a multipart/form-data body (audio
/// transcriptions upload audio bytes; the JSON path cannot apply).
/// Boundary-aware: only scans part HEADERS (bounded window after each
/// boundary), never audio payload bytes.
fn extract_model_multipart(body: &[u8], content_type: &str) -> Option<String> {
    let boundary = content_type
        .split(';')
        .map(str::trim)
        .find_map(|p| p.strip_prefix("boundary="))?
        .trim_matches('"');
    let delim = format!("--{boundary}");
    let mut pos = 0;
    while let Some(rel) = find_sub(&body[pos..], delim.as_bytes()) {
        let start = pos + rel + delim.len();
        // Part header ends at CRLFCRLF; scan only within it.
        let header_end = find_sub(&body[start..], b"\r\n\r\n")? + start;
        let headers = &body[start..header_end];
        let is_field = !headers
            .windows(9)
            .any(|w| w == b"filename=");
        if is_field && find_sub(headers, b"name=\"model\"").is_some() {
            // value starts after the blank line, ends at next CRLF
            let vstart = header_end + 4;
            let vend = find_sub(&body[vstart..], b"\r\n")? + vstart;
            if vend > vstart {
                return Some(String::from_utf8_lossy(&body[vstart..vend]).into_owned());
            }
        }
        pos = header_end;
    }
    None
}

fn find_sub(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || hay.len() < needle.len() {
        return None;
    }
    hay.windows(needle.len()).position(|w| w == needle)
}

/// GET|POST /v1/adapters -> child /lora-adapters (route verified in
/// upstream server.cpp).
pub async fn lora_adapters(
    State(state): State<Arc<AppState>>,
    uri: Uri,
    method: Method,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let store = match Store::open(&state.dirs) {
        Ok(s) => s,
        Err(e) => return openai_error(500, &e.to_string()),
    };
    let first = match store.list_models() {
        Ok(l) => l.into_iter().next(),
        Err(_) => None,
    };
    let Some(m) = first else {
        return openai_error(404, "no models pulled; adapters apply to a running engine");
    };
    let (engine, load_ms) = match ensure_with_admission(&state, &m.name, Priority::Normal).await {
        Ok(ok) => ok,
        Err(resp) => return resp,
    };
    let url = format!(
        "/lora-adapters{}",
        path_and_query(&uri).trim_start_matches("/v1/adapters")
    );
    let name = engine.name.clone();
    let guard = admission_gate(&state, &name, Priority::Normal).await;
    match guard {
        Ok(g) => proxy_request(&state, &engine, &name, &method, &url, &headers, body, load_ms, None, Some(g)).await,
        Err(resp) => resp,
    }
}

/// GET /healthz — liveness of the gateway itself (children not required).
pub async fn healthz() -> impl IntoResponse {
    (StatusCode::OK, "ok")
}

/// Type alias to satisfy the router signature expectations.
pub type BodyBytes = Bytes;
pub type StreamBody = Body;

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    const CT: &str = "multipart/form-data; boundary=pallama-b";

    #[test]
    fn unit__multipart_extract__model_field_after_file_part() {
        let body = b"--pallama-b\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\nContent-Type: audio/wav\r\n\r\nRIFF--pallama-b-fake-bytes\r\n--pallama-b\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nm1\r\n--pallama-b--\r\n";
        assert_eq!(extract_model_multipart(body, CT).as_deref(), Some("m1"));
    }

    #[test]
    fn unit__multipart_extract__model_field_before_file_part() {
        let body = b"--pallama-b\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nqwen3-0.6b\r\n--pallama-b\r\nContent-Disposition: form-data; name=\"file\"; filename=\"b.mp3\"\r\n\r\nID3-bytes\r\n--pallama-b--\r\n";
        assert_eq!(
            extract_model_multipart(body, CT).as_deref(),
            Some("qwen3-0.6b")
        );
    }

    #[test]
    fn unit__multipart_extract__binary_bytes_never_misread_as_model() {
        // hostile/odd audio that contains a decoy header-looking sequence
        let body = b"--pallama-b\r\nContent-Disposition: form-data; name=\"file\"; filename=\"x.wav\"\r\n\r\nname=\"model\"\r\n\r\ndecoy\r\n--pallama-b--\r\n";
        assert_eq!(extract_model_multipart(body, CT), None);
    }

    #[test]
    fn unit__multipart_extract__no_boundary_header_is_none() {
        assert_eq!(extract_model_multipart(b"whatever", "application/json"), None);
    }
}
