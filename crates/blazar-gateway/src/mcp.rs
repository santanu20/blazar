//! Gateway-side MCP (Model Context Protocol) tool catalog: registered
//! stdio servers contribute tools that any dialect lane can inject on
//! request; the gateway executes the tool calls the model makes and
//! feeds results back — every engine gets tool use with zero
//! child-side MCP support.
//!
//! Opt-in per request: body field `"mcp": "all" | "<server>"` or header
//! `x-blazar-mcp` (header wins). The field is stripped before the body
//! is forwarded. Tools are namespaced `mcp__<server>__<tool>` so they
//! can never collide with caller-provided tools; only calls carrying
//! that prefix are gateway-executed.
//!
//! Protocol: JSON-RPC 2.0 over stdio (initialize → initialized →
//! tools/list with cursor pagination → tools/call), per the MCP
//! 2025-06-18 specification. Shutdown follows the spec order: close
//! stdin, wait, `SIGTERM`, `SIGKILL`.
//!
//! Bounds: `MCP_MAX_ROUNDS` tool rounds per request, per-server
//! `timeout_secs` on every call, streaming requests are refused with a
//! teaching 400 (the loop is buffered by design in this release).

use std::collections::HashMap;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt};

use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use blazar_core::config::McpServer;
use blazar_runtime::EngineRef;
use serde_json::{Value, json};

use crate::state::AppState;

pub const MCP_TOOL_PREFIX: &str = "mcp__";
const MCP_MAX_ROUNDS: usize = 4;
const INIT_TIMEOUT_SECS: u64 = 20;

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

/// One live stdio session with an MCP server (or a remembered death,
/// respawned lazily on the next request).
struct Session {
    child: tokio::process::Child,
    stdin: tokio::process::ChildStdin,
    stdout: tokio::io::BufReader<tokio::process::ChildStdout>,
    next_id: u64,
    tools: Vec<McpTool>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct McpTool {
    pub server: String,
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

struct Entry {
    cfg: McpServer,
    session: tokio::sync::Mutex<Option<Session>>,
}

/// Process registry for the configured `[[mcp]]` servers.
pub struct Registry {
    entries: HashMap<String, Arc<Entry>>,
}

impl Registry {
    #[must_use]
    pub fn from_config(servers: &[McpServer]) -> Self {
        let entries = servers
            .iter()
            .cloned()
            .map(|cfg| {
                (
                    cfg.name.clone(),
                    Arc::new(Entry {
                        cfg,
                        session: tokio::sync::Mutex::new(None),
                    }),
                )
            })
            .collect();
        Self { entries }
    }

    #[must_use]
    pub fn configured_names(&self) -> Vec<String> {
        self.entries.keys().cloned().collect()
    }

    /// Spec-ordered teardown for every live session: close stdin,
    /// wait, SIGTERM, SIGKILL. Idempotent.
    pub async fn shutdown_all(&self) {
        for entry in self.entries.values() {
            let mut guard = entry.session.lock().await;
            if let Some(Session {
                mut child, stdin, ..
            }) = guard.take()
            {
                drop(stdin); // close stdin first, per the spec shutdown order
                if tokio::time::timeout(std::time::Duration::from_secs(5), child.wait())
                    .await
                    .is_err()
                {
                    let _ = child.start_kill();
                }
            }
        }
    }

    /// Resolve the tools of every selected server, spawning (or
    /// respawning) sessions as needed.
    async fn catalog(&self, sel: &str) -> Result<Vec<McpTool>, String> {
        let names: Vec<String> = if sel == "all" {
            let mut n = self.configured_names();
            n.sort();
            n
        } else {
            vec![sel.to_string()]
        };
        let mut tools = Vec::new();
        for name in names {
            let entry = self
                .entries
                .get(&name)
                .ok_or_else(|| unknown_server(&name, &self.configured_names()))?;
            let mut guard = entry.session.lock().await;
            if guard.is_none() {
                *guard = Some(spawn_session(&entry.cfg).await.map_err(|e| {
                    format!("mcp server '{}' failed to start: {e}", entry.cfg.name)
                })?);
            }
            let session = guard.as_ref().expect("just spawned");
            tools.extend(session.tools.iter().cloned());
        }
        Ok(tools)
    }

    /// Execute `mcp__<server>__<tool>`; returns the text content (or
    /// the tool's `isError` text, prefixed) for the model to read.
    async fn execute(&self, server: &str, tool: &str, args: &Value) -> Result<String, String> {
        let entry = self
            .entries
            .get(server)
            .ok_or_else(|| format!("mcp server '{server}' is no longer configured"))?;
        let mut guard = entry.session.lock().await;
        // Dead session (server exited mid-flight): respawn once.
        if guard.is_none() {
            *guard = Some(
                spawn_session(&entry.cfg)
                    .await
                    .map_err(|e| format!("mcp server '{server}' failed to restart: {e}"))?,
            );
        }
        let timeout = std::time::Duration::from_secs(entry.cfg.timeout_secs);
        let call = async {
            let session = guard.as_mut().expect("just spawned");
            let id = session.next_id;
            session.next_id += 1;
            rpc_call(
                session,
                id,
                "tools/call",
                &json!({"name": tool, "arguments": args}),
            )
            .await
        };
        match tokio::time::timeout(timeout, call).await {
            Ok(Ok(result)) => Ok(render_content(&result)),
            Ok(Err(e)) => {
                // Transport broke: mark dead so the next call respawns.
                *guard = None;
                Err(format!("mcp server '{server}' call failed: {e}"))
            }
            Err(_) => Err(format!(
                "mcp server '{server}' tool '{tool}' timed out after {}s",
                entry.cfg.timeout_secs
            )),
        }
    }

    /// `/api/mcp` status plane: configured servers, liveness, tool
    /// names. Never spawns (report-only).
    pub async fn status_json(&self) -> Value {
        let mut servers = Vec::new();
        for (name, entry) in &self.entries {
            let alive = entry.session.lock().await.is_some();
            servers.push(json!({
                "name": name,
                "command": entry.cfg.command,
                "alive": alive,
                "timeout_secs": entry.cfg.timeout_secs,
            }));
        }
        servers.sort_by_key(|s| s["name"].as_str().unwrap_or_default().to_string());
        json!({ "servers": servers })
    }
}

// ---------------------------------------------------------------------------
// stdio transport
// ---------------------------------------------------------------------------

async fn spawn_session(cfg: &McpServer) -> Result<Session, String> {
    if cfg.command.is_empty() {
        return Err("empty command".into());
    }
    let mut cmd = tokio::process::Command::new(&cfg.command[0]);
    cmd.args(&cfg.command[1..])
        .envs(&cfg.env)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null());
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("spawn {}: {e}", cfg.command[0]))?;
    let stdin = child.stdin.take().ok_or("no stdin pipe")?;
    let stdout = child.stdout.take().ok_or("no stdout pipe")?;
    let mut session = Session {
        child,
        stdin,
        stdout: tokio::io::BufReader::new(stdout),
        next_id: 1,
        tools: Vec::new(),
    };
    let init = async {
        let id = session.next_id;
        session.next_id += 1;
        rpc_call(
            &mut session,
            id,
            "initialize",
            &json!({
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "blazar", "version": env!("CARGO_PKG_VERSION")},
            }),
        )
        .await?;
        // Initialized notification (no id, no response expected).
        let note = json!({"jsonrpc": "2.0", "method": "notifications/initialized"});
        let line = serde_json::to_string(&note).map_err(|e| e.to_string())?;
        session
            .stdin
            .write_all(format!("{line}\n").as_bytes())
            .await
            .map_err(|e| format!("write initialized: {e}"))?;
        // tools/list with cursor pagination (cycle-guarded).
        let mut cursor: Option<String> = None;
        let mut seen = std::collections::HashSet::new();
        loop {
            let id = session.next_id;
            session.next_id += 1;
            let mut params = json!({});
            if let Some(c) = &cursor {
                params["cursor"] = json!(c);
            }
            let page = rpc_call(&mut session, id, "tools/list", &params).await?;
            for t in page["tools"].as_array().cloned().unwrap_or_default() {
                let name = t["name"].as_str().unwrap_or_default().to_string();
                if name.is_empty() {
                    continue;
                }
                session.tools.push(McpTool {
                    server: cfg.name.clone(),
                    name,
                    description: t["description"].as_str().unwrap_or_default().to_string(),
                    input_schema: t["inputSchema"].clone(),
                });
            }
            cursor = page["nextCursor"].as_str().map(str::to_string);
            match cursor {
                Some(ref c) if !c.is_empty() && seen.insert(c.clone()) => {}
                _ => break,
            }
        }
        if session.tools.is_empty() {
            return Err("server exposes no tools".into());
        }
        Ok::<(), String>(())
    };
    match tokio::time::timeout(std::time::Duration::from_secs(INIT_TIMEOUT_SECS), init).await {
        Ok(Ok(())) => Ok(session),
        Ok(Err(e)) => Err(e),
        Err(_) => Err(format!("initialize timed out after {INIT_TIMEOUT_SECS}s")),
    }
}

/// One JSON-RPC request over the session, skipping server-initiated
/// notifications (no matching id) until the answer arrives.
async fn rpc_call(
    session: &mut Session,
    id: u64,
    method: &str,
    params: &Value,
) -> Result<Value, String> {
    let req = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
    let line = serde_json::to_string(&req).map_err(|e| e.to_string())?;
    session
        .stdin
        .write_all(format!("{line}\n").as_bytes())
        .await
        .map_err(|e| format!("write {method}: {e}"))?;
    let mut buf = String::new();
    loop {
        buf.clear();
        let n = session
            .stdout
            .read_line(&mut buf)
            .await
            .map_err(|e| format!("read {method}: {e}"))?;
        if n == 0 {
            return Err(format!("server closed stdout during {method}"));
        }
        let Ok(v) = serde_json::from_str::<Value>(&buf) else {
            continue;
        };
        if v["id"].as_u64() != Some(id) {
            continue; // notification or stale response
        }
        if let Some(err) = v.get("error") {
            return Err(format!(
                "{}: {}",
                err["code"].as_i64().unwrap_or_default(),
                err["message"].as_str().unwrap_or("unknown error")
            ));
        }
        return Ok(v["result"].clone());
    }
}

fn render_content(result: &Value) -> String {
    let mut text = String::new();
    for part in result["content"].as_array().cloned().unwrap_or_default() {
        if part["type"] == "text"
            && let Some(t) = part["text"].as_str()
        {
            if !text.is_empty() {
                text.push('\n');
            }
            text.push_str(t);
        }
    }
    if result["isError"].as_bool() == Some(true) {
        format!("tool error: {text}")
    } else if text.is_empty() {
        "(tool returned no text content)".into()
    } else {
        text
    }
}

fn unknown_server(name: &str, known: &[String]) -> String {
    format!(
        "unknown mcp server '{name}' — configured: [{}] (see [[mcp]] in docs/7.SETUP.md)",
        known.join(", ")
    )
}

// ---------------------------------------------------------------------------
// Request-side surface
// ---------------------------------------------------------------------------

/// Read the per-request selector: header `x-blazar-mcp` wins, else the
/// top-level body field `mcp` (which is stripped when read). A present
/// but non-string body field is an error (never silently ignored).
pub(crate) fn selector_from(
    headers: &HeaderMap,
    body: Option<&mut Value>,
) -> Result<Option<String>, String> {
    if let Some(v) = headers.get("x-blazar-mcp").and_then(|v| v.to_str().ok()) {
        if v.is_empty() {
            return Err("x-blazar-mcp header must not be empty".into());
        }
        return Ok(Some(v.to_string()));
    }
    let Some(body) = body else { return Ok(None) };
    let Some(raw) = body.get("mcp").cloned() else {
        return Ok(None);
    };
    if let Some(obj) = body.as_object_mut() {
        obj.remove("mcp");
    }
    let Some(sel) = raw.as_str() else {
        return Err(format!(
            "body field 'mcp' must be a string (\"all\" or a server name), got {raw}"
        ));
    };
    if sel.is_empty() {
        return Err("body field 'mcp' must not be empty".into());
    }
    Ok(Some(sel.to_string()))
}

/// Read-only variant for lanes whose translated body already dropped
/// the extension field (anthropic, ollama): no strip needed.
pub(crate) fn selector_read(headers: &HeaderMap, body: &Value) -> Result<Option<String>, String> {
    if let Some(v) = headers.get("x-blazar-mcp").and_then(|v| v.to_str().ok()) {
        if v.is_empty() {
            return Err("x-blazar-mcp header must not be empty".into());
        }
        return Ok(Some(v.to_string()));
    }
    match body.get("mcp") {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(sel)) if !sel.is_empty() => Ok(Some(sel.clone())),
        Some(other) => Err(format!(
            "body field 'mcp' must be a string (\"all\" or a server name), got {other}"
        )),
    }
}

/// Split `mcp__<server>__<tool>` into its parts (exact, no
/// reassembly surprises — the pieces feed `execute` directly).
pub(crate) fn split_tool_name(name: &str) -> Option<(String, String)> {
    let rest = name.strip_prefix(MCP_TOOL_PREFIX)?;
    let (server, tool) = rest.split_once("__")?;
    (!server.is_empty() && !tool.is_empty() && !tool.contains("__"))
        .then(|| (server.to_string(), tool.to_string()))
}

/// Catalog tools as `OpenAI` function definitions, namespaced.
pub(crate) fn tools_to_openai(tools: &[McpTool]) -> Vec<Value> {
    tools
        .iter()
        .map(|t| {
            json!({
                "type": "function",
                "function": {
                    "name": format!("{MCP_TOOL_PREFIX}{}__{}", t.server, t.name),
                    "description": format!("[mcp:{}] {}", t.server, t.description),
                    "parameters": if t.input_schema.is_object()
                        && t.input_schema.get("type") == Some(&json!("object"))
                    {
                        t.input_schema.clone()
                    } else {
                        json!({"type": "object"})
                    },
                }
            })
        })
        .collect()
}

/// The tool-result messages appended after a round, one per call.
pub(crate) fn tool_result_messages(
    calls: &[Value],
    results: &[Result<String, String>],
) -> Vec<Value> {
    calls
        .iter()
        .zip(results)
        .map(|(call, res)| {
            let content = match res {
                Ok(text) => text.clone(),
                Err(e) => format!("(gateway could not execute tool: {e})"),
            };
            json!({
                "role": "tool",
                "tool_call_id": call["id"].clone(),
                "content": content,
            })
        })
        .collect()
}

fn teaching_error(status: StatusCode, msg: &str) -> Response {
    let body = json!({"error": {"message": msg, "type": "blazar_error"}});
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap_or_default()))
        .unwrap_or_else(|_| Response::new(Body::empty()))
}

/// Gateway-mediated chat loop: inject catalog tools, drive tool rounds
/// through `child_send`, return the final child response (rebuilt with
/// the receipt header). `openai_body` must be the OpenAI-dialect chat
/// body with `stream == false` (callers gate streaming requests).
#[allow(clippy::result_large_err, clippy::too_many_lines)] // Response is the handlers' currency; one cohesive mediated loop
pub(crate) async fn chat_via_mcp(
    state: &Arc<AppState>,
    engine: &EngineRef,
    openai_body: &mut Value,
    sel: &str,
) -> Result<reqwest::Response, Response> {
    let tools = state
        .mcp
        .catalog(sel)
        .await
        .map_err(|e| teaching_error(StatusCode::BAD_REQUEST, &e))?;
    if tools.is_empty() {
        return Err(teaching_error(
            StatusCode::BAD_REQUEST,
            "mcp selection exposes no tools",
        ));
    }
    // Merge with caller-provided tools (namespace prevents collisions).
    let injected = tools_to_openai(&tools);
    match openai_body["tools"].as_array_mut() {
        Some(arr) => arr.extend(injected),
        None => {
            openai_body["tools"] = Value::Array(injected);
        }
    }

    let url = format!(
        "{}/v1/chat/completions",
        crate::proxy::child_base(&engine.endpoint)
    );
    let client = crate::state::child_client(state, &engine.endpoint);
    let mut rounds = 0usize;
    loop {
        let bytes = serde_json::to_vec(openai_body).unwrap_or_default();
        let send = crate::proxy::child_auth(
            client
                .post(&url)
                .header("content-type", "application/json")
                .body(bytes),
            engine,
        );
        let resp = match crate::proxy::child_send(state, engine, send.send()).await {
            Ok(r) => r,
            Err(e) => {
                return Err(teaching_error(
                    StatusCode::BAD_GATEWAY,
                    &format!("engine unreachable during mcp round: {e}"),
                ));
            }
        };
        let status = resp.status();
        let headers = resp.headers().clone();
        let body_bytes = resp
            .bytes()
            .await
            .map_err(|e| teaching_error(StatusCode::BAD_GATEWAY, &format!("engine body: {e}")))?;
        if !status.is_success() {
            return Ok(rebuild(status, &headers, body_bytes, rounds));
        }
        let parsed: Value = serde_json::from_slice(&body_bytes).unwrap_or(Value::Null);
        let calls = parsed
            .pointer("/choices/0/message/tool_calls")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let mcp_calls: Vec<Value> = calls
            .iter()
            .filter(|c| {
                c.pointer("/function/name")
                    .and_then(Value::as_str)
                    .is_some_and(|n| n.starts_with(MCP_TOOL_PREFIX))
            })
            .cloned()
            .collect();
        if mcp_calls.is_empty() {
            // No gateway tool requested: this is the final answer (or a
            // caller-tool call the client executes as usual).
            return Ok(rebuild(status, &headers, body_bytes, rounds));
        }
        rounds += 1;
        if rounds > MCP_MAX_ROUNDS {
            return Err(teaching_error(
                StatusCode::BAD_GATEWAY,
                &format!(
                    "mcp loop exceeded {MCP_MAX_ROUNDS} tool rounds — tighten the prompt or raise the bound"
                ),
            ));
        }
        // Execute every gateway-namespaced call (parallel calls in one
        // response are served sequentially; per-server timeouts bound it).
        let mut results = Vec::with_capacity(mcp_calls.len());
        for call in &mcp_calls {
            let fname = call
                .pointer("/function/name")
                .and_then(Value::as_str)
                .unwrap_or_default();
            match split_tool_name(fname) {
                Some((server, tool)) => {
                    let args = call
                        .pointer("/function/arguments")
                        .and_then(Value::as_str)
                        .and_then(|a| serde_json::from_str::<Value>(a).ok())
                        .unwrap_or_else(|| json!({}));
                    let t0 = std::time::Instant::now();
                    let res = state.mcp.execute(&server, &tool, &args).await;
                    tracing::info!(
                        target: "blazar::mcp",
                        server = %server, tool = %tool,
                        ms = u64::try_from(t0.elapsed().as_millis()).unwrap_or(u64::MAX),
                        ok = res.is_ok(),
                        "mcp tool call"
                    );
                    results.push(res);
                }
                None => results.push(Err(format!("malformed tool name '{fname}'"))),
            }
        }
        // Append the assistant turn verbatim, then the tool results.
        if let Some(msgs) = openai_body["messages"].as_array_mut() {
            if let Some(assistant) = parsed.pointer("/choices/0/message") {
                let mut entry = assistant.clone();
                entry["role"] = json!("assistant");
                msgs.push(entry);
            }
            msgs.extend(tool_result_messages(&mcp_calls, &results));
        } else {
            return Err(teaching_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "mcp loop lost the message list",
            ));
        }
    }
}

/// Rebuild a `reqwest::Response` from buffered parts, stamping the
/// receipt header on the way out.
fn rebuild(
    status: reqwest::StatusCode,
    headers: &reqwest::header::HeaderMap,
    body: axum::body::Bytes,
    rounds: usize,
) -> reqwest::Response {
    let mut builder = axum::http::Response::builder().status(
        axum::http::StatusCode::from_u16(status.as_u16())
            .unwrap_or(axum::http::StatusCode::INTERNAL_SERVER_ERROR),
    );
    for (k, v) in headers {
        if k == reqwest::header::CONTENT_LENGTH {
            continue; // recomputed below
        }
        builder = builder.header(k, v);
    }
    builder = builder.header("x-blazar-mcp", format!("rounds={rounds}"));
    let http_resp = builder
        .body(body)
        .unwrap_or_else(|_| axum::http::Response::new(axum::body::Bytes::new()));
    reqwest::Response::from(http_resp)
}

// ---------------------------------------------------------------------------
// Admin plane
// ---------------------------------------------------------------------------

pub async fn mcp_status(State(state): State<Arc<AppState>>) -> Response {
    let status = state.mcp.status_json().await;
    (
        StatusCode::OK,
        [(axum::http::header::CONTENT_TYPE, "application/json")],
        serde_json::to_vec(&status).unwrap_or_default(),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(non_snake_case)]
    #[test]
    fn unit__split_tool_name__parses_namespaced_calls() {
        assert_eq!(
            split_tool_name("mcp__fetch__search"),
            Some(("fetch".into(), "search".into()))
        );
        assert_eq!(split_tool_name("mcp__a__b__c"), None); // tool must not contain __
        assert_eq!(split_tool_name("get_weather"), None);
        assert_eq!(split_tool_name("mcp__srv__"), None);
        assert_eq!(split_tool_name("mcp____tool"), None);
    }

    #[allow(non_snake_case)]
    #[test]
    fn unit__tools_to_openai__namespace_and_schema_defaults() {
        let tools = vec![McpTool {
            server: "fetch".into(),
            name: "search".into(),
            description: "Search the web".into(),
            input_schema: json!({"type": "object", "properties": {"q": {"type": "string"}}}),
        }];
        let out = tools_to_openai(&tools);
        assert_eq!(out[0]["function"]["name"], "mcp__fetch__search");
        assert!(
            out[0]["function"]["description"]
                .as_str()
                .unwrap()
                .starts_with("[mcp:fetch]")
        );
        assert_eq!(
            out[0]["function"]["parameters"]["properties"]["q"]["type"],
            "string"
        );
        // Non-object schemas fall back to a bare object.
        let weird = vec![McpTool {
            server: "x".into(),
            name: "y".into(),
            description: String::new(),
            input_schema: json!("bogus"),
        }];
        assert_eq!(
            tools_to_openai(&weird)[0]["function"]["parameters"],
            json!({"type": "object"})
        );
    }

    #[allow(non_snake_case)]
    #[test]
    fn unit__tool_result_messages__zip_calls_and_errors() {
        let calls = vec![json!({"id": "call_1"}), json!({"id": "call_2"})];
        let results = vec![Ok("22C".into()), Err("timeout".into())];
        let msgs = tool_result_messages(&calls, &results);
        assert_eq!(msgs[0]["role"], "tool");
        assert_eq!(msgs[0]["tool_call_id"], "call_1");
        assert_eq!(msgs[0]["content"], "22C");
        assert!(
            msgs[1]["content"]
                .as_str()
                .unwrap()
                .contains("could not execute")
        );
    }

    #[allow(non_snake_case)]
    #[test]
    fn unit__selector_from__header_wins_and_body_field_stripped() {
        let mut body = json!({"model": "m", "mcp": "all"});
        let mut headers = HeaderMap::new();
        assert_eq!(
            selector_from(&headers, Some(&mut body)),
            Ok(Some("all".into()))
        );
        assert!(body.get("mcp").is_none(), "field must be stripped");
        headers.insert("x-blazar-mcp", "fetch".parse().unwrap());
        let mut body2 = json!({"mcp": "all"});
        assert_eq!(
            selector_from(&headers, Some(&mut body2)),
            Ok(Some("fetch".into()))
        );
        assert_eq!(selector_from(&headers, None), Ok(Some("fetch".into())));
        let mut empty = json!({"model": "m"});
        assert_eq!(selector_from(&HeaderMap::new(), Some(&mut empty)), Ok(None));
        let mut bad = json!({"mcp": true});
        assert!(selector_from(&HeaderMap::new(), Some(&mut bad)).is_err());
    }

    #[allow(non_snake_case)]
    #[test]
    fn unit__render_content__concat_and_iserror() {
        let ok = json!({"content": [{"type": "text", "text": "a"}, {"type": "image"}, {"type": "text", "text": "b"}]});
        assert_eq!(render_content(&ok), "a\nb");
        let err = json!({"content": [{"type": "text", "text": "boom"}], "isError": true});
        assert_eq!(render_content(&err), "tool error: boom");
        let none = json!({"content": []});
        assert_eq!(render_content(&none), "(tool returned no text content)");
    }
}
