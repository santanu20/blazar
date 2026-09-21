//! Gateway-side chat prompt recipes.
//!
//! `child` (default) hands the structured chat request to the engine and
//! lets its embedded template render the prompt. `ollama_compat` moves that
//! responsibility into the gateway: the full conversation is rendered to
//! a finished prompt string here (chatml wrapping plus a JSON tool-call
//! grammar system block) and sent to the engine through the raw
//! completion lane; the completion text is then parsed back into
//! content and structured tool calls.
//!
//! Why own a recipe at all: live A/B on identical weights showed the
//! baked GGUF template (XML tool grammar, forced reasoning opener)
//! measurably under-performs on error-path and multi-turn tool flows —
//! the same model emits a clean immediate tool call under a JSON
//! grammar recipe and a long refusal text under the XML one. The recipe
//! is a blazar-native wording informed by widely-used tool-preamble
//! conventions; it is deliberately NOT a byte-copy of any other
//! server's prompt.

use serde_json::{json, Value};

/// Recipe name selecting gateway-side pre-rendering (see
/// `Config::prompt_recipe`). Anything else means the child renders.
pub const OLLAMA_COMPAT: &str = "ollama_compat";

const TOOL_CALL_OPEN: &str = "<tool_call>";
const TOOL_CALL_CLOSE: &str = "</tool_call>";

const TOOL_BLOCK_HEADER: &str = "\n\n## Tools\n\nYou have access to the following tools:\n\n";
const TOOL_BLOCK_TEACHING: &str = concat!(
    "\n\nTo call a tool, reply ONLY with the call and no other text:\n",
    "<tool_call>\n",
    "{\"name\": \"tool-name\", \"arguments\": {\"arg\": \"value\"}}\n",
    "</tool_call>\n",
    "\n",
    "String and scalar arguments are written as-is; lists and objects use JSON. ",
    "Issue at most one tool call per reply and wait for its result before continuing."
);

/// Render an OpenAI-shaped chat request (messages, optional tools) into
/// a finished chatml-style prompt ending at the assistant opener —
/// no reasoning/think tag: the recipe, not the template, decides.
///
/// Tool-role messages (results) render as `tool` blocks in order;
/// assistant history that carries `tool_calls` re-renders them in the
/// same `<tool_call>` grammar so the model sees its own prior shape.
#[must_use]
pub fn render_chat_prompt(body: &Value) -> String {
    let mut out = String::with_capacity(4096);
    let messages = body
        .get("messages")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();

    // System first: chatml-family prompts put the system block (with
    // the tool roster appended) before the conversation.
    let mut system = String::new();
    for m in &messages {
        if m.get("role").and_then(Value::as_str) == Some("system") {
            if !system.is_empty() {
                system.push('\n');
            }
            system.push_str(&message_text(m));
        }
    }
    system.push_str(&tools_block(body));
    if !system.is_empty() {
        out.push_str("<|im_start|>system\n");
        out.push_str(&system);
        out.push_str("<|im_end|>\n");
    }

    for m in &messages {
        let role = m.get("role").and_then(Value::as_str).unwrap_or("user");
        if role == "system" {
            continue;
        }
        out.push_str("<|im_start|>");
        out.push_str(role);
        out.push('\n');
        out.push_str(&message_text(m));
        if let Some(calls) = m.get("tool_calls").and_then(Value::as_array) {
            for call in calls {
                if let Some(func) = call.get("function") {
                    out.push('\n');
                    out.push_str(TOOL_CALL_OPEN);
                    out.push('\n');
                    out.push_str(&serde_json::to_string(func).unwrap_or_default());
                    out.push('\n');
                    out.push_str(TOOL_CALL_CLOSE);
                }
            }
        }
        out.push_str("<|im_end|>\n");
    }
    out.push_str("<|im_start|>assistant\n");
    out
}

/// Stop sequences for the raw-completion lane: the chatml turn end (and
/// its bare opener, which some models emit before the tag).
#[must_use]
pub fn completion_stops() -> Vec<String> {
    vec!["<|im_end|>".to_string(), "<|im_start|>".to_string()]
}

/// A tool call parsed out of a completion: the ollama-dialect shape
/// (`function.name` + `function.arguments`).
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedToolCall {
    pub name: String,
    pub arguments: Value,
}

/// Split a raw completion into prose content and structured tool calls.
///
/// `None` inner JSON (or a non-object) surfaces as no call: malformed
/// output is honest visible content, never a fabricated call.
#[must_use]
pub fn parse_completion(text: &str) -> (String, Vec<ParsedToolCall>) {
    let mut content = String::with_capacity(text.len());
    let mut calls = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find(TOOL_CALL_OPEN) {
        content.push_str(&rest[..start]);
        let after_open = &rest[start + TOOL_CALL_OPEN.len()..];
        if let Some(end) = after_open.find(TOOL_CALL_CLOSE) {
            let inner = after_open[..end].trim();
            let parsed = serde_json::from_str::<Value>(inner).ok().filter(|v| {
                let name_ok = v
                    .get("name")
                    .and_then(Value::as_str)
                    .is_some_and(|n| !n.is_empty());
                name_ok && v.get("arguments").is_some_and(Value::is_object)
            });
            if let Some(v) = parsed {
                calls.push(ParsedToolCall {
                    name: v
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string(),
                    arguments: v.get("arguments").cloned().unwrap_or(Value::Null),
                });
            } else {
                // Malformed call: never fabricated, never silently
                // dropped — the raw block stays visible content so
                // the client (and logs) see exactly what happened.
                let raw_end = rest.len() - after_open.len() + end + TOOL_CALL_CLOSE.len();
                content.push_str(&rest[start..raw_end]);
            }
            rest = &after_open[end + TOOL_CALL_CLOSE.len()..];
        } else {
            // Unterminated call tag: treat as trailing content (the
            // model ran out of tokens mid-call; the client sees the
            // raw text and can decide).
            content.push_str(&rest[start..]);
            return (content.trim().to_string(), calls);
        }
    }
    content.push_str(rest);
    (content.trim().to_string(), calls)
}

/// Longest suffix of `accumulated` that could grow into the tool-call
/// opening tag — streaming holds this many bytes back so a tag split
/// across SSE deltas never leaks into the content channel.
#[must_use]
pub fn partial_tag_holdback(accumulated_tail: &str) -> usize {
    let bytes = accumulated_tail.as_bytes();
    let tag = TOOL_CALL_OPEN.as_bytes();
    let mut best = 0;
    for len in 1..=bytes.len().min(tag.len() - 1) {
        if bytes[bytes.len() - len..] == tag[..len] {
            best = len;
        }
    }
    best
}

fn tools_block(body: &Value) -> String {
    let Some(tools) = body.get("tools").and_then(Value::as_array) else {
        return String::new();
    };
    if tools.is_empty() {
        return String::new();
    }
    let mut out = String::from(TOOL_BLOCK_HEADER);
    for t in tools {
        // Tools arrive in OpenAI shape; pass the function descriptor
        // through verbatim so schema fidelity is never lost in
        // re-wording.
        out.push_str(&serde_json::to_string(t).unwrap_or_default());
        out.push('\n');
    }
    out.push_str(TOOL_BLOCK_TEACHING);
    out
}

fn message_text(m: &Value) -> String {
    match m.get("content") {
        Some(Value::String(s)) => s.clone(),
        Some(other) if !other.is_null() => other.to_string(),
        _ => String::new(),
    }
}

/// Does the OpenAI-shaped chat body carry images anywhere? The
/// `ollama_compat` recipe is text-only; image requests stay on the child's
/// native multimodal template lane.
#[must_use]
pub fn has_images(body: &Value) -> bool {
    body.get("messages")
        .and_then(Value::as_array)
        .is_some_and(|ms| {
            ms.iter().any(|m| {
                m.get("images")
                    .is_some_and(|v| v.as_array().is_some_and(|a| !a.is_empty()))
                    || m.get("content")
                        .and_then(Value::as_array)
                        .is_some_and(|parts| {
                            parts
                                .iter()
                                .any(|p| p.get("type").and_then(Value::as_str) == Some("image_url"))
                        })
            })
        })
}

/// Build the raw-completion request for the `ollama_compat` lane: the
/// rendered prompt, the recipe's stops, and a verbatim passthrough of
/// every sampler knob the chat request carried (fidelity to the
/// caller's temperature/seed/etc. is the recipe's contract).
/// Unbounded-generation guard for the raw completion lane. Chat-lane
/// Raw-lane generations are server-capped: the pre-rendered prompt
/// style (prose-first, JSON tool grammar) can ramble past where the
/// child-lane chat template would emit EOS — unbounded runs walked to
/// the context ceiling (~24k tokens, a 300s+ request observed live).
/// `server_cap` (config `raw_lane_max_tokens`, 0 = off) bounds both
/// the omitted and the absurd caller cap. Scheduling note: 2048 tokens
/// at ~80 tok/s holds a single-slot lane ~26s per generation; the
/// structural lever for that class is concurrent slots, not a smaller
/// cap — shrinking it truncates honest generations.
#[must_use]
/// Tool names from an OpenAI-shaped chat body, `None` when any name
/// cannot be embedded safely in a grammar (quotes/backslashes/control
/// characters) or the tools array is not shaped as expected — the
/// caller then stays on free decode rather than shipping a broken
/// constraint.
pub(crate) fn extract_tool_names(body: &Value) -> Option<Vec<String>> {
    let tools = body.get("tools")?.as_array()?;
    if tools.is_empty() {
        return None;
    }
    let mut names = Vec::with_capacity(tools.len());
    for t in tools {
        let name = t.get("function")?.get("name")?.as_str()?;
        if name.is_empty()
            || !name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.')
        {
            return None;
        }
        names.push(name.to_string());
    }
    Some(names)
}

/// GBNF grammar for strict tool decode: one-or-more `<tool_call>`
/// envelopes, each a JSON object whose `name` must be one of the
/// request's tools and whose `arguments` is an arbitrary JSON object.
/// Verified against the llama.cpp fork's grammar field live (the same
/// fork build's `json_schema` converter rejects standard schemas — the
/// GBNF grammar for strict tool decode: one-or-more `<tool_call>`
/// envelopes, each a JSON object whose `name` must be one of the
/// request's tools and whose `arguments` is an arbitrary JSON object.
/// Rule names are `plm`-prefixed on purpose: the llama.cpp OAI
/// completion endpoint carries preloaded grammar rules (`string`,
/// `object`, `value`, ...) that silently shadow same-named user rules
/// — colliding names were the difference between the grammar being
/// followed exactly and not applying at all (verified live). `root`
/// stays unprefixed: the endpoint requires it.
fn build_tool_grammar(names: &[String]) -> String {
    let alternation = names
        .iter()
        .map(|n| format!("\"{n}\""))
        .collect::<Vec<_>>()
        .join(" | ");
    format!(
        r#"root ::= plmcall (plmws plmcall)*
plmcall ::= "<tool_call>" plmws "{{" plmws "\"name\"" plmws ":" plmws "\"" plmident plmws "\"" plmws "," plmws "\"arguments\"" plmws ":" plmws plmobject plmws "}}" plmws "</tool_call>"
plmident ::= {alternation}
plmvalue ::= plmobject | plmarray | plmstring | plmnumber | plmboolean | plmnull
plmobject ::= "{{" plmws (plmmember (plmws "," plmws plmmember)*)? plmws "}}"
plmmember ::= plmstring plmws ":" plmws plmvalue
plmarray ::= "[" plmws (plmvalue (plmws "," plmws plmvalue)*)? plmws "]"
plmstring ::= "\"" ([^"\\\x00-\x1f] | "\\" [\"\\/bfnrt] | "\\u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])* "\""
plmnumber ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)?
plmboolean ::= "true" | "false"
plmnull ::= "null"
plmws ::= [ \t\n\r]*"#
    )
}

pub fn to_completion_request(
    body: &Value,
    stream: bool,
    server_cap: u64,
    strict_tools: bool,
) -> Value {
    let mut req = json!({
        "prompt": render_chat_prompt(body),
        "stop": completion_stops(),
        "stream": stream,
        "stream_options": {"include_usage": true},
    });
    // `strict` decode policy: confine the completion to one-or-more
    // <tool_call> envelopes with a whitelisted name — decoder-level
    // grammar, not a prompt instruction. Only ever attached alongside
    // tools; prose-only requests stay free.
    if strict_tools {
        if let Some(names) = extract_tool_names(body) {
            req["grammar"] = json!(build_tool_grammar(&names));
        }
    }
    for key in [
        "max_tokens",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "seed",
        "repeat_penalty",
        "presence_penalty",
        "frequency_penalty",
        "n",
    ] {
        if let Some(v) = body.get(key) {
            if !v.is_null() {
                req[key] = v.clone();
            }
        }
    }
    // Server-side ceiling: the raw lane bounds generations the caller
    // left open (or asked absurdly large) — min(caller, cap) with cap 0
    // meaning "no ceiling".
    if server_cap > 0 {
        let effective = match req.get("max_tokens").and_then(serde_json::Value::as_i64) {
            Some(caller) if caller > 0 => caller.min(i64::try_from(server_cap).unwrap_or(i64::MAX)),
            _ => i64::try_from(server_cap).unwrap_or(i64::MAX),
        };
        req["max_tokens"] = json!(effective);
    }
    req
}

/// Adapt a raw-completion response body into the chat-completion shape
/// the translation layer expects: prose stays content, parsed
/// `<tool_call>` blocks become structured `tool_calls`, usage rides
/// through untouched.
#[must_use]
pub fn raw_json_to_chat(raw: &Value) -> Value {
    let text = raw
        .pointer("/choices/0/text")
        .and_then(Value::as_str)
        .unwrap_or("");
    let (content, calls) = parse_completion(text);
    let message = if calls.is_empty() {
        json!({"role": "assistant", "content": content})
    } else {
        json!({
            "role": "assistant",
            "content": content,
            "tool_calls": calls
                .iter()
                .enumerate()
                .map(|(i, c)| {
                    json!({
                        "id": format!("call-{i}"),
                        "type": "function",
                        "function": {"name": c.name, "arguments": c.arguments},
                    })
                })
                .collect::<Vec<_>>(),
        })
    };
    let finish = if calls.is_empty() {
        "stop"
    } else {
        "tool_calls"
    };
    let mut chat = json!({
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
    });
    if let Some(u) = raw.get("usage") {
        chat["usage"] = u.clone();
    }
    chat
}

/// Streaming adapter: child raw-completion SSE in, chat-shaped SSE
/// out. Child frames are PARSED (`data: {choices:[{text}...]}` lines),
/// the text deltas forwarded as chat `delta.content` chunks minus a
/// split-tag holdback; at end-of-stream the accumulated text is parsed
/// and tool calls surface as one closing delta chunk — the exact shape
/// a native tool-streaming child emits, so every downstream concern
/// (sentinel, timings, NDJSON translation) reuses the child lane
/// unchanged. All state rides the stream (no globals).
struct Adapt {
    /// Upstream BYTES not yet split into complete SSE lines. Bytes,
    /// never a String: a multibyte UTF-8 char may straddle TCP
    /// chunk boundaries, and per-chunk lossy decoding would corrupt
    /// it into replacement chars. Decoding happens per COMPLETE
    /// line only ('\n' can not appear inside a multibyte sequence).
    recv: Vec<u8>,
    /// Adapted SSE lines ready to yield (each `data: ...\n`).
    ready: Vec<String>,
    /// Bytes of `whole` already forwarded as content: everything
    /// from an unclosed tool-call tag onward withholds to stream
    /// end (the block may span any number of deltas).
    emitted: usize,
    /// Whole completion text for the final tool-call parse.
    whole: String,
    /// Usage from the child's final frame, forwarded so the
    /// downstream accounting stays intact.
    usage: Option<Value>,
    done: bool,
}

fn process_frame(st: &mut Adapt, frame: &Value) {
    if frame
        .get("usage")
        .is_some_and(|u| u.as_object().is_some_and(|o| !o.is_empty()))
    {
        st.usage = frame.get("usage").cloned();
    }
    let text = frame
        .pointer("/choices/0/text")
        .and_then(Value::as_str)
        .unwrap_or("");
    if !text.is_empty() {
        st.whole.push_str(text);
        // Safe-to-forward prefix: everything before the FIRST
        // tool-call tag. Raw index — blocks (closed or open) never
        // leak mid-stream; the holdback guards a tag split across
        // deltas.
        let hold = partial_tag_holdback(&st.whole);
        let first_tag = st.whole.find(TOOL_CALL_OPEN).unwrap_or(st.whole.len());
        let safe_end = first_tag.min(st.whole.len() - hold);
        if safe_end > st.emitted {
            let out_text = st.whole[st.emitted..safe_end].to_string();
            st.emitted = safe_end;
            st.ready.push(sse_content_chunk(&out_text));
        }
    }
}
fn process_payload(st: &mut Adapt, payload: &str) {
    if payload == "[DONE]" {
        finish(st);
    } else if let Ok(frame) = serde_json::from_str::<Value>(payload) {
        process_frame(st, &frame);
    }
}
fn finish(st: &mut Adapt) {
    let (cleaned, calls) = parse_completion(&st.whole);
    // Trailing prose (or malformed-block text the parser keeps
    // visible) — the cleaned string's prefix below `emitted` was
    // already streamed (identical bytes: nothing before the first
    // tag is ever rewritten).
    if cleaned.len() > st.emitted {
        let line = sse_content_chunk(&cleaned[st.emitted..]);
        st.ready.push(line);
    }
    if !calls.is_empty() {
        st.ready.push(sse_tool_calls_chunk(&calls));
    }
    if let Some(u) = st.usage.clone() {
        st.ready.push(format!(
            "data: {}\n\n",
            serde_json::json!({"choices": [], "usage": u})
        ));
    }
    st.ready.push("data: [DONE]\n\n".to_string());
    st.done = true;
}

pub fn raw_sse_to_chat_sse<S>(
    upstream: S,
) -> impl futures::Stream<Item = std::io::Result<axum::body::Bytes>>
where
    S: futures::Stream<Item = std::io::Result<axum::body::Bytes>> + Unpin,
{
    use futures::StreamExt;

    let mut st = Adapt {
        recv: Vec::new(),
        ready: Vec::new(),
        emitted: 0,
        whole: String::new(),
        usage: None,
        done: false,
    };
    let mut upstream = upstream;
    // One child frame: capture usage, forward the safe text prefix.
    // Shared by the live event loop and the EOF flush so a final frame
    // without a trailing newline lands identically.
    // One wire payload (`data: ...` prefix stripped): the live loop
    // gets parsed events from translate::parse_sse; only the EOF tail
    // flush feeds raw payloads through here.
    // End-of-stream: flush the held tail (unless the whole text
    // carries a parsed tool call — its surrounding prose may pass, a
    // raw unsplit tag may not), emit the tool_calls chunk when
    // present, then [DONE].
    futures::stream::poll_fn(move |cx| {
        loop {
            if let Some(line) = st.ready.first() {
                let out = line.clone();
                st.ready.remove(0);
                return std::task::Poll::Ready(Some(Ok(axum::body::Bytes::from(out))));
            }
            if st.done {
                return std::task::Poll::Ready(None);
            }
            // Drain every COMPLETE SSE line the buffer holds. The
            // translation path's parser (translate::parse_sse) already
            // handles boundary splits and the [DONE] sentinel — one
            // framing implementation, not a bespoke second one. Its
            // `consumed` counts decoded-string bytes; the decoded
            // string matches the original bytes exactly for every
            // complete line, and the (possibly lossy) partial tail
            // past `consumed` is discarded with the drain remainder's
            // bytes staying in `recv`.
            let decoded = String::from_utf8_lossy(&st.recv).into_owned();
            let (events, wire_done, consumed) = crate::translate::parse_sse(&decoded);
            if consumed > 0 {
                st.recv.drain(..consumed);
            }
            for frame in &events {
                process_frame(&mut st, frame);
            }
            if wire_done {
                finish(&mut st);
                continue;
            }
            if !events.is_empty() {
                continue;
            }
            match upstream.poll_next_unpin(cx) {
                std::task::Poll::Ready(Some(Ok(chunk))) => {
                    st.recv.extend_from_slice(&chunk);
                }
                std::task::Poll::Ready(Some(Err(e))) => {
                    // Mid-body upstream failure (child connection reset
                    // mid-generation): the response headers are already
                    // out, so a raw `Err` here aborts the chunked body
                    // ILLEGALLY — the client dies on an incomplete read
                    // (observed live: a tool-flow stream dropped at 10s
                    // surfaced as RemoteProtocolError). Streaming
                    // discipline (vLLM/SGLang close path): ship what
                    // arrived, mark the truncation observably, close
                    // legally.
                    tracing::warn!(
                        target: "blazar::proxy",
                        "raw-lane upstream stream failed mid-body: {e} — closing legally with arrived bytes"
                    );
                    st.ready.push(sse_truncated_chunk(&e));
                    finish(&mut st);
                }
                std::task::Poll::Ready(None) => {
                    // Stream ended without [DONE]: a final frame the
                    // child sent WITHOUT a trailing newline still sits
                    // unparsed in `recv` (usage / text / tool call).
                    // Flush it through the same frame path, then close
                    // out — EOF is a transport fact, not completion.
                    if !st.recv.is_empty() {
                        let tail = String::from_utf8_lossy(&st.recv).trim().to_string();
                        if let Some(payload) = tail.strip_prefix("data: ") {
                            process_payload(&mut st, payload);
                        }
                        st.recv.clear();
                    }
                    if !st.done {
                        finish(&mut st);
                    }
                }
                std::task::Poll::Pending => return std::task::Poll::Pending,
            }
        }
    })
}

fn sse_content_chunk(text: &str) -> String {
    format!(
        "data: {}\n\n",
        serde_json::json!({"choices": [{"index": 0, "delta": {"content": text}}]})
    )
}

fn sse_tool_calls_chunk(calls: &[ParsedToolCall]) -> String {
    let chunks: Vec<serde_json::Value> = calls
        .iter()
        .enumerate()
        .map(|(i, c)| {
            serde_json::json!({
                "index": i,
                "function": {
                    "name": c.name,
                    "arguments": serde_json::to_string(&c.arguments)
                        .unwrap_or_else(|_| "{}".into()),
                },
            })
        })
        .collect();
    format!(
        "data: {}\n\n",
        serde_json::json!({"choices": [{"index": 0, "delta": {"tool_calls": chunks}, "finish_reason": "tool_calls"}]})
    )
}

/// Terminal marker for a stream the child cut short mid-body: the
/// content is truncated and the cause rides the chunk so downstream
/// consumers see WHY instead of a silently short answer.
fn sse_truncated_chunk(cause: &dyn std::fmt::Display) -> String {
    format!(
        "data: {}\n\n",
        serde_json::json!({
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "error",
                "blazar": {"stream_truncated": true, "cause": format!("{cause}")}
            }]
        })
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use serde_json::json;

    async fn collect_sse(frames: Vec<std::io::Result<axum::body::Bytes>>) -> (String, bool) {
        let mut out = raw_sse_to_chat_sse(futures::stream::iter(frames));
        let mut items = Vec::new();
        while let Some(item) = out.next().await {
            items.push(item);
        }
        let all_ok = items.iter().all(Result::is_ok);
        let joined = items
            .iter()
            .map(|i| String::from_utf8_lossy(i.as_ref().unwrap()).to_string())
            .collect::<String>();
        (joined, all_ok)
    }

    #[tokio::test]
    #[allow(non_snake_case)]
    async fn unit__sse_adapter__upstream_mid_body_error_closes_legally() {
        // Live incident pin: a child connection reset mid-generation used
        // to propagate as a raw stream `Err` AFTER response headers were
        // out — axum aborts the body, the client dies on an incomplete
        // chunked read (RemoteProtocolError). The adapter must instead
        // ship the arrived bytes, mark the truncation observably, and
        // end with a legal terminator.
        let frames = futures::stream::iter(vec![
            Ok::<_, std::io::Error>(axum::body::Bytes::from(
                "data: {\"choices\":[{\"text\":\"partial answer\"}]}\n\n",
            )),
            Err(std::io::Error::new(
                std::io::ErrorKind::ConnectionReset,
                "connection reset mid-body",
            )),
        ]);
        let mut out = raw_sse_to_chat_sse(frames);
        let mut items = Vec::new();
        while let Some(item) = out.next().await {
            items.push(item);
        }
        // No raw Err may escape once headers are committed.
        assert!(
            items.iter().all(Result::is_ok),
            "adapter must close legally, got an Err item"
        );
        let joined = items
            .iter()
            .map(|i| String::from_utf8_lossy(i.as_ref().unwrap()).to_string())
            .collect::<String>();
        assert!(
            joined.contains("partial answer"),
            "arrived bytes ship: {joined}"
        );
        assert!(
            joined.contains("stream_truncated"),
            "truncation is observable: {joined}"
        );
        assert!(
            joined.trim_end().ends_with("data: [DONE]"),
            "legal terminator: {joined}"
        );
    }

    #[tokio::test]
    #[allow(non_snake_case)]
    async fn unit__raw_sse__eof_flushes_unterminated_line() {
        // An upstream body ending mid-line (no trailing '\n') still owns
        // a complete SSE frame: `data: {...}` without the terminator
        // must be parsed at EOF, not dropped on the floor.
        let (joined, all_ok) = collect_sse(vec![
            Ok(axum::body::Bytes::from_static(
                b"data: {\"choices\":[{\"text\":\"head \"}]}\n\n",
            )),
            // Unterminated final frame — no newline after it.
            Ok(axum::body::Bytes::from_static(
                b"data: {\"choices\":[{\"text\":\"tail\"}]}",
            )),
        ])
        .await;
        assert!(all_ok, "no raw Err items");
        assert!(joined.contains("head "), "{joined}");
        assert!(
            joined.contains("tail"),
            "EOF frame parsed, not dropped: {joined}"
        );
        assert!(joined.trim_end().ends_with("data: [DONE]"), "{joined}");
    }

    #[tokio::test]
    #[allow(non_snake_case)]
    async fn unit__raw_sse__utf8_split_across_chunks() {
        // '€' is E2 82 AC — split across two TCP chunks, per-chunk lossy
        // decoding would emit U+FFFD twice. The byte buffer must keep
        // it intact.
        let evil = b"data: {\"choices\":[{\"text\":\"a\xe2".to_vec();
        let rest = b"\x82\xac\"}]}\n\n".to_vec();
        let (joined, all_ok) = collect_sse(vec![
            Ok(axum::body::Bytes::from(evil.clone())),
            Ok(axum::body::Bytes::from(rest.clone())),
        ])
        .await;
        assert!(all_ok);
        assert!(
            joined.contains("a\u{20ac}"),
            "multibyte char intact: {joined}"
        );
        assert!(
            !joined.contains('\u{fffd}'),
            "no replacement chars: {joined}"
        );
    }

    #[tokio::test]
    #[allow(non_snake_case)]
    async fn unit__raw_sse__tool_call_at_eof() {
        // The final unterminated frame carries a complete tool-call
        // block: EOF flush must surface it as a tool_calls chunk, never
        // leak the raw <tool_call> text as content.
        let payload = r#"data: {"choices":[{"text":"<tool_call>{\"name\":\"compute\",\"arguments\":{\"x\":1}}</tool_call>"}]}"#;
        let payload: &'static str = payload;
        let (joined, all_ok) =
            collect_sse(vec![Ok(axum::body::Bytes::from_static(payload.as_bytes()))]).await;
        assert!(all_ok);
        assert!(
            joined.contains("\"function\"") && joined.contains("compute"),
            "tool_calls chunk emitted: {joined}"
        );
        assert!(
            !joined.contains("a<tool_call>") && !joined.contains(": <tool_call>"),
            "no raw tag leak: {joined}"
        );
        assert!(joined.trim_end().ends_with("data: [DONE]"), "{joined}");
    }

    #[tokio::test]
    #[allow(non_snake_case)]
    async fn unit__raw_sse__usage_at_eof() {
        // Final unterminated frame is a bare usage frame: the EOF flush
        // must capture and re-emit it as the adapter's usage chunk.
        let (joined, all_ok) = collect_sse(vec![Ok(axum::body::Bytes::from_static(
            b"data: {\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5}}",
        ))])
        .await;
        assert!(all_ok);
        assert!(
            joined.contains("\"prompt_tokens\":10"),
            "usage forwarded: {joined}"
        );
        assert!(joined.trim_end().ends_with("data: [DONE]"), "{joined}");
    }

    #[tokio::test]
    #[allow(non_snake_case)]
    async fn unit__raw_sse__done_not_required_for_clean_eof() {
        // Some children close the body without ever sending
        // `data: [DONE]` — EOF itself must complete the stream legally
        // (terminal chunks + [DONE]), not surface as an error.
        let (joined, all_ok) = collect_sse(vec![Ok(axum::body::Bytes::from_static(
            b"data: {\"choices\":[{\"text\":\"done-less child\"}]}\n\n",
        ))])
        .await;
        assert!(all_ok, "clean EOF is never an error: {joined}");
        assert!(joined.contains("done-less child"), "{joined}");
        assert!(
            joined.trim_end().ends_with("data: [DONE]"),
            "adapter terminates: {joined}"
        );
    }

    fn body(messages: &Value, tools: Option<Value>) -> Value {
        let mut b = json!({ "messages": messages });
        if let Some(t) = tools {
            b["tools"] = t;
        }
        b
    }

    #[test]
    #[allow(non_snake_case)]
    fn unit__render__system_tools_and_opener() {
        let tools = json!([{
            "type": "function",
            "function": {
                "name": "compute",
                "description": "math",
                "parameters": {"type": "object", "properties": {"x": {"type": "number"}}}
            }
        }]);
        let b = body(
            &json!([
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "hi"}
            ]),
            Some(tools),
        );
        let p = render_chat_prompt(&b);
        assert!(
            p.starts_with("<|im_start|>system\nBe terse.\n\n## Tools"),
            "{p}"
        );
        assert!(p.contains("\"name\":\"compute\"") || p.contains("\"name\": \"compute\""));
        assert!(p.contains(TOOL_CALL_OPEN));
        assert!(p.contains("<|im_start|>user\nhi<|im_end|>"));
        assert!(p.ends_with("<|im_start|>assistant\n"), "{p}");
        assert!(!p.contains("<think>"), "recipe owns the opener");
    }

    #[test]
    #[allow(non_snake_case)]
    fn unit__render__no_tools_no_block() {
        let b = body(&json!([{"role": "user", "content": "hi"}]), None);
        let p = render_chat_prompt(&b);
        assert!(!p.contains("## Tools"));
        assert_eq!(p, "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n");
    }

    #[test]
    #[allow(non_snake_case)]
    fn unit__render__tool_history_renders_own_grammar() {
        let b = body(
            &json!([
                {"role": "user", "content": "go"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"function": {"name": "compute", "arguments": {"x": 1}}}]},
                {"role": "tool", "content": "{\"result\": 42}"},
            ]),
            None,
        );
        let p = render_chat_prompt(&b);
        assert!(p.contains("<tool_call>"), "{p}");
        assert!(
            p.contains("\"name\":\"compute\"") || p.contains("\"name\": \"compute\""),
            "{p}"
        );
        assert!(p.contains("\"x\":1") || p.contains("\"x\": 1"), "{p}");
        assert!(p.contains("</tool_call><|im_end|>"), "{p}");
        assert!(
            p.contains("<|im_start|>tool\n{\"result\": 42}<|im_end|>"),
            "{p}"
        );
    }

    #[test]
    #[allow(non_snake_case)]
    fn unit__parse__clean_call_with_preamble() {
        let (c, calls) = parse_completion(
            "Let me compute that.\n<tool_call>\n{\"name\": \"compute\", \"arguments\": {\"x\": 1}}\n</tool_call>\n",
        );
        assert_eq!(c, "Let me compute that.");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "compute");
        assert_eq!(calls[0].arguments, json!({"x": 1}));
    }

    #[test]
    #[allow(non_snake_case)]
    fn unit__parse__plain_text_no_calls() {
        let (c, calls) = parse_completion("The answer is 42.");
        assert_eq!(c, "The answer is 42.");
        assert!(calls.is_empty());
    }

    #[test]
    #[allow(non_snake_case)]
    fn unit__parse__malformed_inner_json_is_content_not_fabrication() {
        let (c, calls) = parse_completion("<tool_call>\n{broken\n</tool_call>");
        assert!(calls.is_empty(), "never fabricate a call");
        assert!(c.contains("{broken"), "raw text stays visible: {c}");
    }

    #[test]
    #[allow(non_snake_case)]
    fn unit__parse__unterminated_tag_is_trailing_content() {
        let (c, calls) = parse_completion("ok<tool_call>\n{\"name\": \"x\"");
        assert!(calls.is_empty());
        assert!(c.contains("<tool_call>"));
    }

    #[test]
    #[allow(non_snake_case)]
    fn unit__holdback__split_tag_across_deltas() {
        assert_eq!(partial_tag_holdback("answer<tool"), 5);
        assert_eq!(partial_tag_holdback("answer."), 0);
        assert_eq!(partial_tag_holdback("<tool_call> full"), 0);
    }

    #[test]
    #[allow(non_snake_case)]
    fn unit__completion_request__server_cap_guards_raw_lane() {
        // No caller cap -> the server ceiling applies; a smaller caller
        // cap stays sovereign; an ABSURD caller cap is clamped by the
        // server ceiling; ceiling 0 = no guard at all.
        let body = json!({"messages": [{"role": "user", "content": "hi"}]});
        let req = to_completion_request(&body, false, 2048, false);
        assert_eq!(req["max_tokens"], json!(2048));
        let body = json!({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 512});
        let req = to_completion_request(&body, false, 2048, false);
        assert_eq!(req["max_tokens"], json!(512));
        let body = json!({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 50000});
        let req = to_completion_request(&body, false, 2048, false);
        assert_eq!(req["max_tokens"], json!(2048));
        let body = json!({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 50000});
        let req = to_completion_request(&body, false, 0, false);
        assert_eq!(req["max_tokens"], json!(50000));
    }

    #[test]
    #[allow(non_snake_case)]
    fn unit__tool_grammar__enumerates_names_and_requires_envelope() {
        let g = build_tool_grammar(&["compute_geochem".into(), "read_data".into()]);
        assert!(
            g.contains(r#"plmident ::= "compute_geochem" | "read_data""#)
                && g.contains(r#"plmws ":" plmws "\"" plmident"#),
            "{g}"
        );
        // The completion MUST be one-or-more tool_call envelopes.
        assert!(g.contains("root ::= plmcall (plmws plmcall)*"), "{g}");
        assert!(g.contains("<tool_call>"), "{g}");
        // Full JSON sub-grammar so arguments stay arbitrary objects.
        assert!(
            g.contains(
                "plmvalue ::= plmobject | plmarray | plmstring | plmnumber | plmboolean | plmnull"
            ),
            "{g}"
        );
    }

    #[test]
    #[allow(non_snake_case)]
    fn unit__tool_grammar__unsafe_names_rejected() {
        // A quote/backslash/control-laden name cannot be embedded
        // safely: extract returns None and the caller stays free.
        let body = json!({"tools": [{"type": "function", "function": {"name": "bad\"name"}}]});
        assert!(extract_tool_names(&body).is_none());
        let body = json!({"tools": [{"type": "function", "function": {"name": ""}}]});
        assert!(extract_tool_names(&body).is_none());
        let body = json!({"messages": []});
        assert!(extract_tool_names(&body).is_none());
    }

    #[test]
    #[allow(non_snake_case)]
    fn unit__completion_request__strict_attaches_grammar_only_with_tools() {
        let tools = json!({"messages": [{"role": "user", "content": "go"}],
                           "tools": [{"type": "function",
                                      "function": {"name": "compute_geochem",
                                                   "parameters": {"type": "object"}}}]});
        let req = to_completion_request(&tools, false, 0, true);
        assert!(req
            .get("grammar")
            .is_some_and(|g| g.as_str().unwrap().contains("compute_geochem")));
        // free policy: no grammar even with tools.
        let req = to_completion_request(&tools, false, 0, false);
        assert!(req.get("grammar").is_none());
        // strict but prose-only request: stays free.
        let plain = json!({"messages": [{"role": "user", "content": "hi"}]});
        let req = to_completion_request(&plain, false, 0, true);
        assert!(req.get("grammar").is_none());
    }
}
