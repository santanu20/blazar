//! Remote engine instances: `[[remotes]]` entries name external
//! OpenAI-compatible servers (another pallama, vLLM, an MLX server,
//! llamactl — anything speaking `/v1/*`). Requests for
//! `<remote-name>:<model>` route there instead of spawning a local
//! child; nothing local is loaded, quotas still apply at the gateway.

use axum::body::Body;
use axum::response::{IntoResponse, Response};

use pallama_core::{Config, Remote};

use crate::state::AppState;

/// Split `"<remote>:<model>"` when the prefix names a configured
/// remote. Returns (remote, stripped-model).
#[must_use]
pub fn split_remote<'a>(model: &'a str, cfg: &'a Config) -> Option<(&'a Remote, &'a str)> {
    let (name, rest) = model.split_once(':')?;
    let remote = cfg.remotes.iter().find(|r| r.name == name)?;
    Some((remote, rest))
}

/// Forward an OpenAI-shaped request to a remote: rewrite `model` to the
/// stripped name, attach the remote's bearer key, stream the response
/// back byte-for-byte (the zero-tax contract holds — the remote is
/// someone else's engine).
pub async fn forward_openai(
    state: &AppState,
    remote: &Remote,
    remote_model: &str,
    method: &axum::http::Method,
    path_query: &str,
    headers: &axum::http::HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    // Rewrite the model field (JSON bodies on the /v1 lane).
    let body = rewrite_model(body, remote_model);
    let url = format!("{}{}", remote.url.trim_end_matches('/'), path_query);
    let mut req = state.http.request(method.clone(), &url);
    for (name, value) in headers {
        if !crate::proxy::STRIP_REQUEST.contains(&name.as_str()) {
            req = req.header(name, value);
        }
    }
    if !remote.key.is_empty() {
        req = req.bearer_auth(&remote.key);
    }
    let resp = match req
        .body(reqwest::Body::wrap_stream(futures::stream::once(
            async move { Ok::<_, std::io::Error>(body) },
        )))
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            tracing::error!(target: "pallama::remotes", remote = %remote.name, "forward failed: {e:#}");
            return crate::proxy::openai_error(
                502,
                &format!("remote {:?} unreachable: {e}", remote.name),
            );
        }
    };
    let status = axum::http::StatusCode::from_u16(resp.status().as_u16())
        .unwrap_or(axum::http::StatusCode::BAD_GATEWAY);
    let mut builder = Response::builder().status(status);
    for (name, value) in resp.headers() {
        if !crate::proxy::STRIP_RESPONSE.contains(&name.as_str()) {
            builder = builder.header(name, value);
        }
    }
    let stream = futures::StreamExt::map(resp.bytes_stream(), |r| {
        r.map_err(|e| std::io::Error::other(e.to_string()))
    });
    builder.body(Body::from_stream(stream)).unwrap_or_else(|e| {
        crate::proxy::openai_error(500, &format!("remote body: {e}")).into_response()
    })
}

/// Best-effort model rewrite on a JSON body; non-JSON (multipart)
/// passes through untouched (remote gets the prefixed name — its owner
/// can name the model that way if they want).
fn rewrite_model(body: axum::body::Bytes, remote_model: &str) -> axum::body::Bytes {
    match serde_json::from_slice::<serde_json::Value>(&body) {
        Ok(mut v) => {
            if let Some(obj) = v.as_object_mut() {
                obj.insert("model".into(), serde_json::json!(remote_model));
            }
            serde_json::to_vec(&v).map_or(body, axum::body::Bytes::from)
        }
        Err(_) => body,
    }
}

/// Health probe for `pallama ps`: one GET /v1/models with a short
/// timeout; returns (ok, model-count-or-error).
pub async fn probe(state: &AppState, remote: &Remote) -> (bool, String) {
    let url = format!("{}/v1/models", remote.url.trim_end_matches('/'));
    let mut req = state.http.get(&url);
    if !remote.key.is_empty() {
        req = req.bearer_auth(&remote.key);
    }
    match req.timeout(std::time::Duration::from_secs(3)).send().await {
        Ok(r) if r.status().is_success() => match r.json::<serde_json::Value>().await {
            Ok(v) => {
                let n = v["data"].as_array().map_or(0, std::vec::Vec::len);
                (true, format!("{n} models"))
            }
            Err(_) => (true, "ok (unparsable /v1/models)".to_string()),
        },
        Ok(r) => (false, format!("HTTP {}", r.status())),
        Err(e) => (false, e.to_string()),
    }
}

/// `/api/chat` against a remote: ollama body -> `OpenAI` -> remote ->
/// ollama shape back (non-stream JSON; stream = SSE->NDJSON with the
/// same translate helpers the local path uses).
#[allow(clippy::too_many_lines, clippy::items_after_statements)] // mirrors the local translate path 1:1
pub async fn ollama_chat_remote(
    state: &AppState,
    remote: &Remote,
    remote_model: &str,
    req: &serde_json::Value,
) -> Response {
    let (mut openai_req, _num_ctx) = match crate::translate::chat_to_openai(req) {
        Ok(r) => r,
        Err(e) => return crate::proxy::openai_error(400, &e),
    };
    if let Some(obj) = openai_req.as_object_mut() {
        obj.insert("model".into(), serde_json::json!(remote_model));
    }
    let stream = req
        .get("stream")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(true);
    if stream {
        if let Some(obj) = openai_req.as_object_mut() {
            obj.insert(
                "stream_options".into(),
                serde_json::json!({"include_usage": true}),
            );
        }
    }
    let url = format!("{}/v1/chat/completions", remote.url.trim_end_matches('/'));
    let mut http = state.http.post(&url).json(&openai_req);
    if !remote.key.is_empty() {
        http = http.bearer_auth(&remote.key);
    }
    let resp = match http.send().await {
        Ok(r) => r,
        Err(e) => {
            return crate::proxy::openai_error(
                502,
                &format!("remote {:?} unreachable: {e}", remote.name),
            );
        }
    };
    if !resp.status().is_success() {
        let status = resp.status().as_u16();
        let text = resp.text().await.unwrap_or_default();
        return crate::proxy::openai_error(status, &format!("remote error: {text}"));
    }
    if !stream {
        let openai: serde_json::Value = match resp.json().await {
            Ok(v) => v,
            Err(e) => return crate::proxy::openai_error(502, &format!("bad remote response: {e}")),
        };
        return axum::Json(crate::translate::openai_chat_to_ollama(
            remote_model,
            &openai,
        ))
        .into_response();
    }
    // SSE -> ollama NDJSON (same incremental translate as the local path).
    use futures::StreamExt as _;
    let upstream = resp.bytes_stream();
    let model_c = remote_model.to_string();
    let ndjson = futures::stream::unfold(
        (
            upstream.boxed(),
            String::new(),
            model_c,
            false,
            None::<serde_json::Value>,
            None::<String>,
            false,
        ),
        |(mut stream, mut buf, model, mut done, mut usage, mut finish, mut usage_sent)| async move {
            loop {
                if done && !usage_sent {
                    let final_chunk = crate::translate::ollama_final_chunk(
                        &model,
                        usage.as_ref(),
                        finish.as_deref(),
                    );
                    usage_sent = true;
                    return Some((
                        Ok(axum::body::Bytes::from(format!("{final_chunk}\n"))),
                        (stream, buf, model, done, usage, finish, usage_sent),
                    ));
                }
                match stream.next().await {
                    Some(Ok(bytes)) => {
                        buf.push_str(&String::from_utf8_lossy(&bytes));
                        let (events, saw_done, consumed) = crate::translate::parse_sse(&buf);
                        buf.drain(..consumed);
                        for ev in &events {
                            if let Some(u) = ev.get("usage").filter(|u| !u.is_null()) {
                                usage = Some(u.clone());
                            }
                            if let Some(fr) = ev["choices"][0]["finish_reason"].as_str() {
                                finish = Some(fr.to_string());
                            }
                        }
                        if saw_done {
                            done = true;
                        }
                        let lines: Vec<String> = events
                            .iter()
                            .flat_map(|ev| crate::translate::openai_chunk_to_ollama(&model, ev))
                            .map(|v| format!("{v}\n"))
                            .collect();
                        if lines.is_empty() {
                            continue;
                        }
                        return Some((
                            Ok(axum::body::Bytes::from(lines.join(""))),
                            (stream, buf, model, done, usage, finish, usage_sent),
                        ));
                    }
                    Some(Err(e)) => {
                        return Some((
                            Err(std::io::Error::other(e.to_string())),
                            (stream, buf, model, done, usage, finish, usage_sent),
                        ));
                    }
                    None => {
                        done = true;
                        if usage_sent {
                            return None;
                        }
                    }
                }
            }
        },
    );
    Response::builder()
        .status(200)
        .header("content-type", "application/x-ndjson")
        .body(Body::from_stream(ndjson))
        .unwrap_or_else(|e| {
            crate::proxy::openai_error(500, &format!("remote stream: {e}")).into_response()
        })
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    #[test]
    fn unit__split_remote__prefix_must_name_a_remote() {
        let cfg = Config {
            remotes: vec![Remote {
                name: "vllm".into(),
                url: "http://10.0.0.4:8000".into(),
                key: String::new(),
            }],
            ..Config::default()
        };
        let (r, m) = split_remote("vllm:qwen3-72b", &cfg).unwrap();
        assert_eq!(r.name, "vllm");
        assert_eq!(m, "qwen3-72b");
        assert!(split_remote("localmodel", &cfg).is_none());
        assert!(split_remote("nope:x", &cfg).is_none());
        // A model whose name contains a colon but matches nothing local
        // must not silently route — split_remote only fires on exact
        // remote-name prefixes.
        assert!(split_remote("localhost:11434", &cfg).is_none());
    }

    #[test]
    fn unit__rewrite_model__json_only() {
        let b = rewrite_model(
            axum::body::Bytes::from(r#"{"model":"vllm:x","messages":[]}"#),
            "x",
        );
        assert!(std::str::from_utf8(&b).unwrap().contains(r#""model":"x""#));
        let raw = rewrite_model(axum::body::Bytes::from_static(b"not-json"), "x");
        assert_eq!(&*raw, b"not-json");
    }
}
