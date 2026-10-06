//! Gateway-side MCP (Model Context Protocol) tool catalog: registered
//! stdio or streamable-HTTP servers contribute tools that any dialect
//! lane can inject on request; the gateway executes the tool calls the
//! model makes and feeds results back — every engine gets tool use with
//! zero child-side MCP support.
//!
//! Opt-in per request: body field `"mcp": "all" | "<server>"` or header
//! `x-blazar-mcp` (header wins). The field is stripped before the body
//! is forwarded. Tools are namespaced `mcp__<server>__<tool>` so they
//! can never collide with caller-provided tools; only calls carrying
//! that prefix are gateway-executed.
//!
//! Protocol: JSON-RPC 2.0 per the MCP 2025-06-18 specification, over
//! stdio (newline-delimited; shutdown: close stdin, wait, `SIGTERM`,
//! `SIGKILL`) or Streamable HTTP (POST with `Accept:
//! application/json, text/event-stream`; JSON or SSE-framed responses;
//! `Mcp-Session-Id` carried on every request after initialize).
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

/// Streamable-HTTP session state (MCP 2025-06-18): the server-assigned
/// session id and negotiated protocol version ride every request after
/// initialize; the tool list is fetched once per session.
struct HttpState {
    session_id: Option<String>,
    protocol_version: Option<String>,
    next_id: u64,
    tools: Vec<McpTool>,
    inited: bool,
}

struct HttpSession {
    endpoint: String,
    client: reqwest::Client,
    state: tokio::sync::Mutex<HttpState>,
}

// Boxed: the stdio Mutex dwarfs HttpSession, and an unboxed enum this
// size penalizes every registry entry (clippy::large_enum_variant).
enum Transport {
    Stdio(Box<tokio::sync::Mutex<Option<Session>>>),
    Http(HttpSession),
}

struct Entry {
    cfg: McpServer,
    transport: Transport,
}

/// Process registry for the configured `[[mcp]]` servers.
pub struct Registry {
    entries: HashMap<String, Arc<Entry>>,
    /// Config-level default (`mcp_default`) applied when a request carries
    /// no selector of its own. `"none"`/absent keeps requests MCP-free.
    default_sel: Option<String>,
}

impl Registry {
    #[must_use]
    pub fn from_config(servers: &[McpServer], default_sel: Option<String>) -> Self {
        let entries = servers
            .iter()
            .cloned()
            .map(|cfg| {
                let transport = match cfg.url.clone() {
                    Some(endpoint) => Transport::Http(HttpSession {
                        endpoint,
                        client: reqwest::Client::new(),
                        state: tokio::sync::Mutex::new(HttpState {
                            session_id: None,
                            protocol_version: None,
                            next_id: 1,
                            tools: Vec::new(),
                            inited: false,
                        }),
                    }),
                    None => Transport::Stdio(Box::new(tokio::sync::Mutex::new(None))),
                };
                (cfg.name.clone(), Arc::new(Entry { cfg, transport }))
            })
            .collect();
        Self {
            entries,
            default_sel,
        }
    }

    #[must_use]
    pub fn configured_names(&self) -> Vec<String> {
        self.entries.keys().cloned().collect()
    }

    /// Layer the config default under a request-level selector result:
    /// header/body win (with `"none"` as the explicit opt-out), then the
    /// config default, then no MCP. Errors from the request selector
    /// pass straight through — invalid input must teach, not fall back.
    pub(crate) fn resolve(
        &self,
        request: Result<Option<String>, String>,
    ) -> Result<Option<String>, String> {
        match request? {
            Some(sel) if sel == "none" => Ok(None),
            Some(sel) => Ok(Some(sel)),
            None => Ok(self.default_sel.clone().filter(|d| d != "none")),
        }
    }

    /// Spec-ordered teardown for every live session: stdio closes
    /// stdin, waits, `SIGTERM`, `SIGKILL`; HTTP sends a best-effort
    /// session `DELETE`. Idempotent.
    pub async fn shutdown_all(&self) {
        for entry in self.entries.values() {
            match &entry.transport {
                Transport::Stdio(m) => {
                    let mut guard = m.lock().await;
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
                Transport::Http(h) => {
                    let sid = h.state.lock().await.session_id.clone();
                    if let Some(sid) = sid {
                        let _ = h
                            .client
                            .delete(&h.endpoint)
                            .header("Mcp-Session-Id", sid)
                            .send()
                            .await;
                    }
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
            match &entry.transport {
                Transport::Stdio(m) => {
                    let mut guard = m.lock().await;
                    if guard.is_none() {
                        *guard = Some(spawn_session(&entry.cfg).await.map_err(|e| {
                            format!("mcp server '{}' failed to start: {e}", entry.cfg.name)
                        })?);
                    }
                    let session = guard.as_ref().expect("just spawned");
                    tools.extend(session.tools.iter().cloned());
                }
                Transport::Http(h) => {
                    let mut st = h.state.lock().await;
                    if !st.inited {
                        *st = http_init(&entry.cfg, h).await.map_err(|e| {
                            format!("mcp server '{}' failed to initialize: {e}", entry.cfg.name)
                        })?;
                    }
                    tools.extend(st.tools.iter().cloned());
                }
            }
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
        match &entry.transport {
            Transport::Stdio(m) => {
                let mut guard = m.lock().await;
                // Dead session (server exited mid-flight): respawn once.
                if guard.is_none() {
                    *guard =
                        Some(spawn_session(&entry.cfg).await.map_err(|e| {
                            format!("mcp server '{server}' failed to restart: {e}")
                        })?);
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
            Transport::Http(h) => {
                let timeout = std::time::Duration::from_secs(entry.cfg.timeout_secs);
                let call = async {
                    // Session marked dead (404/transport error): re-init once.
                    let mut st = h.state.lock().await;
                    if !st.inited {
                        *st = http_init(&entry.cfg, h).await.map_err(|e| {
                            format!("mcp server '{server}' failed to initialize: {e}")
                        })?;
                    }
                    let id = st.next_id;
                    st.next_id += 1;
                    http_rpc(
                        h,
                        Some(id),
                        "tools/call",
                        &json!({"name": tool, "arguments": args}),
                        st.session_id.as_deref(),
                        st.protocol_version.as_deref(),
                        timeout,
                    )
                    .await
                    .map(|(result, sid)| {
                        if let Some(s) = sid {
                            st.session_id = Some(s);
                        }
                        result
                    })
                };
                match tokio::time::timeout(timeout, call).await {
                    Ok(Ok(result)) => Ok(render_content(&result)),
                    Ok(Err(e)) => {
                        // Session presumed stale or expired: re-init next call.
                        h.state.lock().await.inited = false;
                        Err(format!("mcp server '{server}' call failed: {e}"))
                    }
                    Err(_) => Err(format!(
                        "mcp server '{server}' tool '{tool}' timed out after {}s",
                        entry.cfg.timeout_secs
                    )),
                }
            }
        }
    }

    /// `/api/mcp` status plane: configured servers, liveness, tool
    /// names. Never spawns (report-only).
    pub async fn status_json(&self) -> Value {
        let mut servers = Vec::new();
        for (name, entry) in &self.entries {
            let row = match &entry.transport {
                Transport::Stdio(m) => json!({
                    "name": name,
                    "transport": "stdio",
                    "command": entry.cfg.command,
                    "alive": m.lock().await.is_some(),
                    "timeout_secs": entry.cfg.timeout_secs,
                }),
                Transport::Http(h) => json!({
                    "name": name,
                    "transport": "http",
                    "url": entry.cfg.url,
                    "alive": h.state.lock().await.inited,
                    "timeout_secs": entry.cfg.timeout_secs,
                }),
            };
            servers.push(row);
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

// ---------------------------------------------------------------------------
// streamable-HTTP transport
// ---------------------------------------------------------------------------

/// POST one JSON-RPC message to the endpoint per the Streamable HTTP
/// transport: `Accept: application/json, text/event-stream`, response
/// either a single JSON object or SSE `data:` frames (first frame
/// carrying the matching id wins; interleaved notifications are
/// skipped). Returns `(result, Mcp-Session-Id)` — the session id only
/// appears on the initialize response.
async fn http_rpc(
    hs: &HttpSession,
    id: Option<u64>,
    method: &str,
    params: &Value,
    session_hdr: Option<&str>,
    protocol_hdr: Option<&str>,
    timeout: std::time::Duration,
) -> Result<(Value, Option<String>), String> {
    let mut msg = json!({"jsonrpc": "2.0", "method": method, "params": params});
    if let Some(id) = id {
        msg["id"] = json!(id);
    }
    let body = serde_json::to_vec(&msg).map_err(|e| format!("encode {method}: {e}"))?;
    let mut req = hs
        .client
        .post(&hs.endpoint)
        .timeout(timeout)
        .header("content-type", "application/json")
        .header("accept", "application/json, text/event-stream")
        .body(body);
    if let Some(s) = session_hdr {
        req = req.header("Mcp-Session-Id", s);
    }
    if let Some(p) = protocol_hdr {
        req = req.header("MCP-Protocol-Version", p);
    }
    let resp = req
        .send()
        .await
        .map_err(|e| format!("http {method}: {e}"))?;
    let sid = resp
        .headers()
        .get("Mcp-Session-Id")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        let excerpt: String = body.chars().take(200).collect();
        return Err(format!("http {method}: status {status}: {excerpt}"));
    }
    let Some(id) = id else {
        return Ok((Value::Null, sid)); // notification: 202, no body
    };
    let ctype = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_ascii_lowercase();
    let body = resp
        .text()
        .await
        .map_err(|e| format!("http {method} body: {e}"))?;
    let v = if ctype.contains("text/event-stream") {
        sse_response(&body, id).ok_or_else(|| {
            format!("http {method}: sse stream closed without a matching response")
        })?
    } else {
        serde_json::from_str(&body).map_err(|e| format!("http {method} json body: {e}"))?
    };
    if let Some(err) = v.get("error") {
        return Err(format!(
            "{}: {}",
            err["code"].as_i64().unwrap_or_default(),
            err["message"].as_str().unwrap_or("unknown error")
        ));
    }
    Ok((v.get("result").cloned().unwrap_or(Value::Null), sid))
}

/// Scan SSE frames for the JSON-RPC response carrying `id`, skipping
/// interleaved notifications and server-initiated requests.
fn sse_response(body: &str, id: u64) -> Option<Value> {
    for line in body.lines() {
        let Some(data) = line.strip_prefix("data:") else {
            continue;
        };
        let Ok(v) = serde_json::from_str::<Value>(data.trim()) else {
            continue;
        };
        if v.get("id").is_some() && v["id"].as_u64() == Some(id) {
            return Some(v);
        }
    }
    None
}

/// Fresh HTTP initialize: negotiate protocol version, capture the
/// server-assigned session id, announce `notifications/initialized`,
/// then page through `tools/list` (cycle-guarded, mirroring stdio).
async fn http_init(cfg: &McpServer, hs: &HttpSession) -> Result<HttpState, String> {
    let timeout = std::time::Duration::from_secs(INIT_TIMEOUT_SECS);
    let (result, sid) = http_rpc(
        hs,
        Some(1),
        "initialize",
        &json!({
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "blazar", "version": env!("CARGO_PKG_VERSION")},
        }),
        None, // fresh session — never a stale header
        None,
        timeout,
    )
    .await?;
    let protocol_version = result["protocolVersion"]
        .as_str()
        .unwrap_or("2025-06-18")
        .to_string();
    // Initialized notification: 202 Accepted, no body to read.
    let _ = http_rpc(
        hs,
        None,
        "notifications/initialized",
        &json!({}),
        sid.as_deref(),
        Some(&protocol_version),
        timeout,
    )
    .await;
    let mut state = HttpState {
        session_id: sid,
        protocol_version: Some(protocol_version),
        next_id: 2,
        tools: Vec::new(),
        inited: true,
    };
    let mut cursor: Option<String> = None;
    let mut seen = std::collections::HashSet::new();
    loop {
        let id = state.next_id;
        state.next_id += 1;
        let mut params = json!({});
        if let Some(c) = &cursor {
            params["cursor"] = json!(c);
        }
        let (page, _) = http_rpc(
            hs,
            Some(id),
            "tools/list",
            &params,
            state.session_id.as_deref(),
            state.protocol_version.as_deref(),
            timeout,
        )
        .await?;
        for t in page["tools"].as_array().cloned().unwrap_or_default() {
            let name = t["name"].as_str().unwrap_or_default().to_string();
            if name.is_empty() {
                continue;
            }
            state.tools.push(McpTool {
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
    Ok(state)
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
            "body field 'mcp' must be a string (\"all\", a server name, or \"none\"), got {raw}"
        ));
    };
    if sel.is_empty() {
        return Err("body field 'mcp' must not be empty".into());
    }
    Ok(Some(sel.to_string()))
}

/// Re-stamp the mediation receipt on a lane-rebuilt response: the child
/// carries `x-blazar-mcp`, but the ollama/anthropic lanes rebuild the
/// dialect response around it and would drop the header otherwise.
pub(crate) fn stamp(resp: &mut axum::response::Response, hdr: Option<&str>) {
    if let Some(h) = hdr
        && let Ok(v) = axum::http::HeaderValue::from_str(h)
    {
        resp.headers_mut().insert("x-blazar-mcp", v);
    }
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
            "body field 'mcp' must be a string (\"all\", a server name, or \"none\"), got {other}"
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

/// Tool results for caller-owned tool calls that landed in a mixed
/// turn (caller tools alongside gateway tools): the gateway cannot
/// execute them — the client owns them — so they get an honest
/// placeholder result instead of being silently dropped, which would
/// leave a dangling `tool_call` and violate the chat protocol on the
/// next turn.
pub(crate) fn caller_tool_results(all_calls: &[Value], mcp_calls: &[Value]) -> Vec<Value> {
    all_calls
        .iter()
        .filter(|c| !mcp_calls.contains(*c))
        .map(|call| {
            let name = call
                .pointer("/function/name")
                .and_then(Value::as_str)
                .unwrap_or("unknown");
            json!({
                "role": "tool",
                "tool_call_id": call["id"].clone(),
                "content": format!(
                    "(gateway mcp note: caller tool '{name}' runs on the client, not in this mediated turn — answer from the mcp results, or re-issue it on a follow-up request)"
                ),
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
    // Computed once: the constraint receipt the proxy lane would stamp.
    let so_kind = crate::proxy::structured_output_kind(Some(openai_body));
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
            crate::metadata_card::persist_from_response(
                state,
                openai_body
                    .get("model")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
                Some(openai_body),
                &body_bytes,
            );
            return Ok(rebuild(status, &headers, body_bytes, rounds, so_kind));
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
            crate::metadata_card::persist_from_response(
                state,
                openai_body
                    .get("model")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
                Some(openai_body),
                &body_bytes,
            );
            return Ok(rebuild(status, &headers, body_bytes, rounds, so_kind));
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
            // Mixed turns: backfill results for caller-owned calls so
            // every tool_call in the appended assistant turn is
            // answered (protocol-complete round 2).
            msgs.extend(caller_tool_results(&calls, &mcp_calls));
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
    so_kind: Option<&'static str>,
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
    // The mediated path never passes through the proxy response builder,
    // so the structured-output receipt rides here instead.
    if let Some(kind) = so_kind {
        builder = builder.header("x-blazar-structured-output", kind);
    }
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

    #[allow(non_snake_case)]
    #[test]
    fn unit__stamp__receipt_survives_lane_rebuild() {
        let mut resp = axum::response::Response::new(axum::body::Body::empty());
        stamp(&mut resp, Some("rounds=2"));
        assert_eq!(
            resp.headers()
                .get("x-blazar-mcp")
                .and_then(|v| v.to_str().ok()),
            Some("rounds=2")
        );
        stamp(&mut resp, None);
        assert_eq!(
            resp.headers()
                .get("x-blazar-mcp")
                .and_then(|v| v.to_str().ok()),
            // Some overwrites a prior None-stamp path; None never clears.
            Some("rounds=2")
        );
        let mut bare = axum::response::Response::new(axum::body::Body::empty());
        stamp(&mut bare, None);
        assert!(bare.headers().get("x-blazar-mcp").is_none());
    }

    #[allow(non_snake_case)]
    #[test]
    fn unit__rebuild__stamps_structured_output_receipt() {
        // The mediated lane must carry the same constraint receipt the
        // proxy lane stamps, so constrained requests stay provable.
        let resp = rebuild(
            reqwest::StatusCode::OK,
            &reqwest::header::HeaderMap::new(),
            axum::body::Bytes::new(),
            1,
            Some("json_schema"),
        );
        assert_eq!(
            resp.headers()
                .get("x-blazar-structured-output")
                .and_then(|v| v.to_str().ok()),
            Some("json_schema")
        );
        assert_eq!(
            resp.headers()
                .get("x-blazar-mcp")
                .and_then(|v| v.to_str().ok()),
            Some("rounds=1")
        );
        let bare = rebuild(
            reqwest::StatusCode::OK,
            &reqwest::header::HeaderMap::new(),
            axum::body::Bytes::new(),
            0,
            None,
        );
        assert!(bare.headers().get("x-blazar-structured-output").is_none());
    }

    #[allow(non_snake_case)]
    #[test]
    fn unit__resolve__request_wins_default_layers_none_opts_out() {
        let with_default = Registry::from_config(&[], Some("all".into()));
        // No request selector -> config default applies.
        assert_eq!(with_default.resolve(Ok(None)), Ok(Some("all".into())));
        // Request selector overrides the default.
        assert_eq!(
            with_default.resolve(Ok(Some("fetch".into()))),
            Ok(Some("fetch".into()))
        );
        // "none" is the explicit opt-out even with a default configured.
        assert_eq!(with_default.resolve(Ok(Some("none".into()))), Ok(None));
        // Request errors teach, never fall back to the default.
        assert!(with_default.resolve(Err("bad selector".into())).is_err());

        let off = Registry::from_config(&[], Some("none".into()));
        assert_eq!(off.resolve(Ok(None)), Ok(None));
        let bare = Registry::from_config(&[], None);
        assert_eq!(bare.resolve(Ok(None)), Ok(None));
        assert_eq!(
            bare.resolve(Ok(Some("fetch".into()))),
            Ok(Some("fetch".into()))
        );
    }

    #[allow(non_snake_case)]
    #[test]
    fn unit__sse_response__finds_matching_id_among_frames() {
        let body = concat!(
            "event: message\n",
            "data: {\"jsonrpc\":\"2.0\",\"method\":\"notifications/progress\"}\n\n",
            "data: {\"jsonrpc\":\"2.0\",\"id\":7,\"result\":{\"tools\":[]}}\n\n",
            "data: {\"jsonrpc\":\"2.0\",\"id\":8,\"result\":{}}\n\n",
        );
        let hit = sse_response(body, 7).expect("frame for id 7 present");
        assert!(hit.pointer("/result/tools").is_some());
        assert!(sse_response(body, 9).is_none(), "no frame for id 9");
        assert!(sse_response("", 7).is_none(), "empty stream");
    }

    #[allow(non_snake_case)]
    #[tokio::test]
    async fn unit__http_rpc__attaches_jsonrpc_body_and_reads_json_reply() {
        // One-shot TCP server: pins the wire contract — the POST must
        // carry the JSON-RPC envelope as its body (a regression once
        // sent headers only, which lenient servers masked as 202s).
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let srv = std::thread::spawn(move || {
            let (mut sock, _) = listener.accept().unwrap();
            let mut buf = [0u8; 4096];
            let n = sock.read(&mut buf).unwrap();
            let req = String::from_utf8_lossy(&buf[..n]).to_string();
            let clen = req
                .split("content-length:")
                .nth(1)
                .and_then(|s| s.split_whitespace().next())
                .and_then(|v| v.parse::<usize>().ok())
                .expect("content-length header present");
            let body = req.split("\r\n\r\n").nth(1).unwrap_or_default();
            assert!(clen > 0, "request must carry a body");
            let parsed: Value = serde_json::from_str(body).expect("body is the JSON-RPC envelope");
            assert_eq!(parsed["jsonrpc"], "2.0");
            assert_eq!(parsed["method"], "tools/list");
            assert_eq!(parsed["id"], 5);
            let reply = json!({"jsonrpc": "2.0", "id": 5, "result": {"tools": []}});
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                reply.to_string().len(),
                reply
            );
            sock.write_all(resp.as_bytes()).unwrap();
        });
        let hs = HttpSession {
            endpoint: format!("http://127.0.0.1:{port}/mcp"),
            client: {
                blazar_core::tls::ensure_tls_provider();
                reqwest::Client::new()
            },
            state: tokio::sync::Mutex::new(HttpState {
                session_id: None,
                protocol_version: None,
                next_id: 1,
                tools: Vec::new(),
                inited: false,
            }),
        };
        let (result, sid) = http_rpc(
            &hs,
            Some(5),
            "tools/list",
            &json!({}),
            None,
            None,
            std::time::Duration::from_secs(5),
        )
        .await
        .expect("rpc over the one-shot server");
        assert!(result["tools"].is_array());
        assert!(sid.is_none(), "no session id on a plain reply");
        srv.join().unwrap();
    }

    #[allow(non_snake_case)]
    #[test]
    fn unit__caller_tool_results__mixed_turn_backfills_non_mcp_calls() {
        let caller = |id: &str, name: &str| {
            json!({
                "id": id,
                "type": "function",
                "function": {"name": name, "arguments": "{}"},
            })
        };
        let all = vec![
            caller("call_1", "get_weather"),
            caller("call_2", "mcp__demo__add"),
            caller("call_3", "search_docs"),
        ];
        let mcp = vec![all[1].clone()];
        let out = caller_tool_results(&all, &mcp);
        assert_eq!(out.len(), 2, "only the two caller tools get results");
        assert_eq!(out[0]["tool_call_id"], json!("call_1"));
        assert_eq!(out[1]["tool_call_id"], json!("call_3"));
        let note = out[0]["content"].as_str().unwrap_or_default();
        assert!(
            note.contains("get_weather") && note.contains("client"),
            "placeholder names the tool and who runs it: {note}"
        );
        assert!(caller_tool_results(&all, &all).is_empty(), "no mixed calls");
    }
}
