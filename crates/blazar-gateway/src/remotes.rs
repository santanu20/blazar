//! Remote engine instances: `[[remotes]]` entries name external
//! OpenAI-compatible servers (another blazar, vLLM, an MLX server,
//! llamactl — anything speaking `/v1/*`). Requests for
//! `<remote-name>:<model>` route there instead of spawning a local
//! child; nothing local is loaded, quotas still apply at the gateway.

use axum::body::Body;
use axum::response::{IntoResponse, Response};

use blazar_core::{Config, Remote};

use crate::state::AppState;

/// Split `"<remote>:<model>"` when the prefix names a configured
/// remote. Returns (remote, stripped-model).
#[must_use]
pub fn split_remote<'a>(model: &'a str, cfg: &'a Config) -> Option<(&'a Remote, &'a str)> {
    let (name, rest) = model.split_once(':')?;
    let remote = cfg.remotes.iter().find(|r| r.name == name)?;
    Some((remote, rest))
}

/// Consecutive failures before a remote is marked down (circuit open).
const REMOTE_MARK_DOWN_FAILS: u32 = 3;
/// How long a marked-down remote is skipped before a half-open probe.
const REMOTE_COOLDOWN: std::time::Duration = std::time::Duration::from_secs(30);

/// Circuit + load state for one remote pool member (C2/C3).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct RemoteHealth {
    pub consec_failures: u32,
    /// `Some(t)` = marked down until `t` (requests skip it, then one
    /// half-open probe passes through).
    pub down_until: Option<std::time::Instant>,
    pub in_flight: u32,
}

fn health_key(remote: &Remote) -> String {
    format!("{}|{}", remote.name, remote.url)
}

/// Bound on the C4 prefix→remote stickiness table: one entry per
/// distinct conversation prefix; arbitrary eviction keeps it bounded.
const REMOTE_AFFINITY_CAP: usize = 4096;

/// C4: fold pool name + conversation prefix into one affinity key.
fn affinity_key(pool_name: &str, prefix: &blazar_runtime::PrefixKey) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in pool_name
        .as_bytes()
        .iter()
        .chain(&prefix.sys.to_le_bytes())
        .chain(&prefix.convo.to_le_bytes())
    {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Remember which remote served this prefix (called on success).
fn bind_remote(
    map: &std::sync::Mutex<std::collections::HashMap<u64, String>>,
    key: u64,
    hkey: &str,
) {
    let mut m = map.lock().expect("remote_affinity lock poisoned");
    if m.len() >= REMOTE_AFFINITY_CAP && !m.contains_key(&key) {
        if let Some(evict) = m.keys().next().copied() {
            m.remove(&evict);
        }
    }
    m.insert(key, hkey.to_string());
}

/// Forget a prefix binding when it points at the failed remote.
fn unbind_remote(
    map: &std::sync::Mutex<std::collections::HashMap<u64, String>>,
    key: u64,
    hkey: &str,
) {
    let mut m = map.lock().expect("remote_affinity lock poisoned");
    if m.get(&key).is_some_and(|k| k == hkey) {
        m.remove(&key);
    }
}

/// RAII in-flight lease: bumped by `select_remote`, released on drop.
/// Released when the caller's response HEADERS are ready (forward
/// functions return at header time); body streaming continues after —
/// header-time is the meaningful queue signal for load balancing.
pub struct RemoteLease {
    map: std::sync::Arc<std::sync::Mutex<std::collections::HashMap<String, RemoteHealth>>>,
    key: String,
}

impl Drop for RemoteLease {
    fn drop(&mut self) {
        if let Ok(mut m) = self.map.lock() {
            if let Some(h) = m.get_mut(&self.key) {
                h.in_flight = h.in_flight.saturating_sub(1);
            }
        }
    }
}

/// Select the remote for a `<remote>:<model>` request: filter the
/// name-pool down to live members (marked-down members are skipped —
/// all-down returns a 503 teaching error with the retry window), then
/// pick the least-busy (C3). With a conversation `prefix` hint (C4),
/// prefer the live pool member that last served this prefix — its KV
/// cache holds the conversation — falling back to least-busy. Returns
/// the remote, the stripped model, an in-flight lease and the affinity
/// key to bind/unbind on the result.
#[allow(clippy::type_complexity, clippy::result_large_err)] // gateway error currency
pub fn select_remote<'a>(
    state: &'a AppState,
    model: &'a str,
    prefix: Option<&blazar_runtime::PrefixKey>,
) -> Result<(&'a Remote, &'a str, RemoteLease, Option<u64>), Response> {
    let Some((name, rest)) = model.split_once(':') else {
        return Err(crate::proxy::openai_error(
            400,
            "model has no remote prefix",
        ));
    };
    let now = std::time::Instant::now();
    let mut map = state
        .remote_health
        .lock()
        .expect("remote_health lock poisoned");
    let pool: Vec<&Remote> = state
        .config
        .remotes
        .iter()
        .filter(|r| r.name == name)
        .collect();
    if pool.is_empty() {
        return Err(crate::proxy::openai_error(
            400,
            &format!("unknown remote {name:?}; check [[remotes]] in config"),
        ));
    }
    let live: Vec<(&Remote, RemoteHealth)> = pool
        .iter()
        .filter_map(|r| {
            let h = map.get(&health_key(r)).copied().unwrap_or_default();
            let down = h.down_until.is_some_and(|t| t > now);
            (!down).then_some((*r, h))
        })
        .collect();
    if live.is_empty() {
        let soonest = pool
            .iter()
            .filter_map(|r| map.get(&health_key(r)).and_then(|h| h.down_until))
            .min()
            .unwrap_or(now);
        let secs = soonest.saturating_duration_since(now).as_secs().max(1);
        return Err(crate::proxy::openai_error(
            503,
            &format!(
                "remote {name:?} marked down (circuit open); retry in ~{secs}s or check the remote"
            ),
        ));
    }
    // C4: sticky prefix affinity — the live member holding this
    // conversation's KV wins; without a hint (or after eviction/circuit)
    // it is plain least-busy.
    let akey = prefix.map(|p| affinity_key(name, p));
    let sticky = akey.and_then(|k| {
        state
            .remote_affinity
            .lock()
            .expect("remote_affinity lock poisoned")
            .get(&k)
            .cloned()
    });
    let chosen = sticky.as_ref().and_then(|want| {
        live.iter()
            .find(|(r, _)| health_key(r) == *want)
            .map(|(r, _)| *r)
    });
    let remote = chosen.unwrap_or_else(|| {
        live.iter()
            .min_by_key(|(_, h)| h.in_flight)
            .map(|(r, _)| *r)
            .expect("live pool non-empty")
    });
    let key = health_key(remote);
    map.entry(key.clone()).or_default().in_flight += 1;
    let lease = RemoteLease {
        map: std::sync::Arc::clone(&state.remote_health),
        key,
    };
    Ok((remote, rest, lease, akey))
}

/// Record a forward result: success resets the circuit; a failure
/// (connect error or 5xx) increments, and `REMOTE_MARK_DOWN_FAILS` in a
/// row marks the remote down for the cooldown window.
pub fn note_remote_result(state: &AppState, remote: &Remote, ok: bool) {
    let key = health_key(remote);
    let mut map = state
        .remote_health
        .lock()
        .expect("remote_health lock poisoned");
    let h = map.entry(key).or_default();
    if ok {
        if h.consec_failures > 0 || h.down_until.is_some() {
            tracing::info!(target: "blazar::remotes", remote = %remote.name, "remote recovered");
        }
        h.consec_failures = 0;
        h.down_until = None;
        return;
    }
    h.consec_failures += 1;
    if h.consec_failures >= REMOTE_MARK_DOWN_FAILS {
        h.down_until = Some(std::time::Instant::now() + REMOTE_COOLDOWN);
        tracing::warn!(
            target: "blazar::remotes",
            remote = %remote.name,
            url = %remote.url,
            "remote marked down for {}s after {fails} consecutive failures",
            REMOTE_COOLDOWN.as_secs(),
            fails = h.consec_failures
        );
    }
}

/// Result bookkeeping shared by every remote lane: circuit note, C4
/// affinity bind/unbind, and the `x-blazar-remote` observability
/// header naming the member that served the request.
pub fn tag_remote_result(
    state: &AppState,
    remote: &Remote,
    akey: Option<u64>,
    mut resp: Response,
) -> Response {
    let hkey = health_key(remote);
    let ok = resp.status().as_u16() < 500;
    note_remote_result(state, remote, ok);
    if let Some(k) = akey {
        if ok {
            bind_remote(&state.remote_affinity, k, &hkey);
        } else {
            unbind_remote(&state.remote_affinity, k, &hkey);
        }
    }
    if let Ok(v) = axum::http::HeaderValue::from_str(&hkey) {
        resp.headers_mut().insert("x-blazar-remote", v);
    }
    resp
}

/// `forward_openai` + circuit bookkeeping + C4 affinity: select
/// (prefix-sticky LB + mark-down filter), forward, note the result by
/// response class (<500 = ok). Success binds the conversation prefix to
/// the remote that served it; failure unbinds.
pub async fn forward_with_health(
    state: &AppState,
    model: &str,
    method: &axum::http::Method,
    path_query: &str,
    headers: &axum::http::HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    // Affinity hash over the ORIGINAL body (forward_openai rewrites the
    // model field — the prefix identity lives in the prompt fields).
    let prefix = crate::proxy::affinity_hash_bytes(&body);
    let (remote, remote_model, _lease, akey) = match select_remote(state, model, prefix.as_ref()) {
        Ok(x) => x,
        Err(resp) => return resp,
    };
    let resp = forward_openai(
        state,
        remote,
        remote_model,
        method,
        path_query,
        headers,
        body,
    )
    .await;
    tag_remote_result(state, remote, akey, resp)
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
            tracing::error!(target: "blazar::remotes", remote = %remote.name, "forward failed: {e:#}");
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

/// Health probe for `blazar ps`: one GET /v1/models with a short
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
    // Shared clock: stamp first/last-byte offsets like the local lane so the
    // final NDJSON line carries gateway-measured timings (remotes report
    // none of their own).
    let t0 = std::time::Instant::now();
    let clock = std::sync::Arc::new(std::sync::Mutex::new((None::<u64>, 0u64)));
    let clock_map = std::sync::Arc::clone(&clock);
    let upstream = resp
        .bytes_stream()
        .map(move |chunk| {
            let elapsed = u64::try_from(t0.elapsed().as_nanos()).unwrap_or(u64::MAX);
            {
                let mut c = clock_map.lock().unwrap();
                if c.0.is_none() {
                    c.0 = Some(elapsed);
                }
                c.1 = elapsed;
            }
            chunk
        })
        .boxed();
    let model_c = remote_model.to_string();
    // F70: boundary-safe decode — chunk-split UTF-8 chars stay raw until
    // their final byte arrives.
    let ndjson = futures::stream::unfold(
        (
            upstream,
            String::new(),
            crate::translate::LineBuffer::new(),
            model_c,
            false,
            None::<serde_json::Value>,
            None::<String>,
            false,
            std::sync::Arc::clone(&clock),
            crate::translate::ToolCallAccum::default(),
        ),
        |(
            mut stream,
            mut buf,
            mut lines,
            model,
            mut done,
            mut usage,
            mut finish,
            mut usage_sent,
            clock,
            mut tool_accum,
        )| async move {
            loop {
                if done && !usage_sent {
                    // Decode window = last - first byte; total = full wall.
                    let (eval_ns, total_ns) = {
                        let c = clock.lock().unwrap();
                        match c.0 {
                            Some(f) => (Some(c.1.saturating_sub(f)), Some(c.1)),
                            None => (None, (c.1 > 0).then_some(c.1)),
                        }
                    };
                    let final_chunk = crate::translate::ollama_final_chunk(
                        &model,
                        usage.as_ref(),
                        finish.as_deref(),
                        eval_ns,
                        total_ns,
                    );
                    usage_sent = true;
                    let flush_prefix = tool_accum.flush_line(&model).unwrap_or_default();
                    return Some((
                        Ok(axum::body::Bytes::from(format!(
                            "{flush_prefix}{final_chunk}\n"
                        ))),
                        (
                            stream, buf, lines, model, done, usage, finish, usage_sent, clock,
                            tool_accum,
                        ),
                    ));
                }
                match stream.next().await {
                    Some(Ok(bytes)) => {
                        buf.push_str(&lines.feed(&bytes));
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
                        let ndjson_lines: Vec<String> = events
                            .iter()
                            .flat_map(|ev| {
                                crate::translate::openai_chunk_to_ollama(
                                    &mut tool_accum,
                                    &model,
                                    ev,
                                )
                            })
                            .map(|v| format!("{v}\n"))
                            .collect();
                        if ndjson_lines.is_empty() {
                            continue;
                        }
                        return Some((
                            Ok(axum::body::Bytes::from(ndjson_lines.join(""))),
                            (
                                stream, buf, lines, model, done, usage, finish, usage_sent, clock,
                                tool_accum,
                            ),
                        ));
                    }
                    Some(Err(e)) => {
                        return Some((
                            Err(std::io::Error::other(e.to_string())),
                            (
                                stream, buf, lines, model, done, usage, finish, usage_sent, clock,
                                tool_accum,
                            ),
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
    fn unit__remote_lease__drop_releases_in_flight_slot() {
        let map: std::sync::Arc<std::sync::Mutex<std::collections::HashMap<String, RemoteHealth>>> =
            std::sync::Arc::default();
        {
            let lease = RemoteLease {
                map: std::sync::Arc::clone(&map),
                key: "r|http://10.0.0.4:8000".into(),
            };
            // select_remote bumps in_flight when it hands out the lease.
            map.lock()
                .unwrap()
                .entry(lease.key.clone())
                .or_default()
                .in_flight += 1;
            assert_eq!(map.lock().unwrap()["r|http://10.0.0.4:8000"].in_flight, 1);
        }
        // Lease dropped at header time: the LB slot is back.
        assert_eq!(map.lock().unwrap()["r|http://10.0.0.4:8000"].in_flight, 0);
    }

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

    #[test]
    fn unit__remote_affinity__key_discriminates_pool_and_convo() {
        let p = blazar_runtime::PrefixKey { sys: 1, convo: 2 };
        let a = affinity_key("vllm", &p);
        assert_eq!(a, affinity_key("vllm", &p), "stable for same inputs");
        assert_ne!(
            a,
            affinity_key("vllm", &blazar_runtime::PrefixKey { sys: 9, convo: 2 }),
            "different system prompt = different key"
        );
        assert_ne!(a, affinity_key("mlx", &p), "different pool = different key");
    }

    #[test]
    fn unit__remote_affinity__bind_sticks_unbind_only_owns() {
        let map = std::sync::Mutex::new(std::collections::HashMap::new());
        bind_remote(&map, 7, "vllm|http://a:1");
        assert_eq!(
            map.lock().unwrap().get(&7).map(String::as_str),
            Some("vllm|http://a:1")
        );
        // Unbind with a DIFFERENT remote's key must not steal the entry.
        unbind_remote(&map, 7, "vllm|http://b:2");
        assert!(
            map.lock().unwrap().contains_key(&7),
            "foreign unbind no-ops"
        );
        // Own unbind removes it.
        unbind_remote(&map, 7, "vllm|http://a:1");
        assert!(!map.lock().unwrap().contains_key(&7));
    }

    #[test]
    fn unit__remote_affinity__bounded_at_cap() {
        let map = std::sync::Mutex::new(std::collections::HashMap::new());
        for i in 0..=REMOTE_AFFINITY_CAP {
            bind_remote(&map, i as u64, "r|u");
        }
        assert!(
            map.lock().unwrap().len() <= REMOTE_AFFINITY_CAP,
            "affinity table stays bounded: {}",
            map.lock().unwrap().len()
        );
    }
}
