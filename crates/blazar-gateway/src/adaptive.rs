//! Adaptive control for the gateway-mediated MCP execution path.
//!
//! This module deliberately contains a cheap, deterministic policy: it
//! profiles request shape once, then allocates bounded tool rounds/calls.
//! It does not ask a second model to decide whether another model call is
//! worthwhile. The configured MCP request deadline remains the wall-clock
//! hard stop, and existing API-key MCP scopes remain authoritative.

use axum::http::HeaderMap;
use serde_json::Value;

pub(crate) const HEADER: &str = "x-blazar-adaptive";
const MAX_TOOL_ROUNDS: usize = 6;
const MAX_TOOL_CALLS: usize = 20;
const MAX_SCHEMA_RETRIES: usize = 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Mode {
    Auto,
    Deep,
}

impl Mode {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Deep => "deep",
        }
    }
}

/// Resolve the opt-in header. A missing or "off" header preserves legacy behavior.
pub(crate) fn mode_from_headers(headers: &HeaderMap) -> Result<Option<Mode>, String> {
    let Some(raw) = headers.get(HEADER) else {
        return Ok(None);
    };
    let value = raw
        .to_str()
        .map_err(|_| format!("{HEADER} must be valid ASCII: off, auto, or deep"))?;
    match value {
        "off" => Ok(None),
        "auto" => Ok(Some(Mode::Auto)),
        "deep" => Ok(Some(Mode::Deep)),
        _ => Err(format!(
            "{HEADER} must be one of off, auto, or deep; got {value:?}"
        )),
    }
}

/// Per-request execution limits. The MCP registry's configured aggregate
/// deadline and every server's configured timeout remain independent hard
/// ceilings; these limits only bound how much work the model may request.
#[derive(Clone, Debug)]
pub(crate) struct Plan {
    pub mode: Mode,
    pub complexity: u8,
    pub max_tool_rounds: usize,
    pub max_tool_calls: usize,
    pub max_schema_retries: usize,
    pub verify_schema: bool,
}

impl Plan {
    pub(crate) fn for_request(mode: Mode, request: &Value) -> Self {
        let complexity = complexity_score(request);
        let (max_tool_rounds, max_tool_calls) = match (mode, complexity) {
            (Mode::Auto, 0..=1) => (2, 4),
            (Mode::Auto, 2..=3) => (3, 8),
            (Mode::Auto, _) => (4, 12),
            (Mode::Deep, 0..=1) => (3, 8),
            (Mode::Deep, 2..=3) => (5, 16),
            (Mode::Deep, _) => (MAX_TOOL_ROUNDS, MAX_TOOL_CALLS),
        };
        let verify_schema = request
            .pointer("/response_format/json_schema/schema")
            .is_some_and(Value::is_object);
        Self {
            mode,
            complexity,
            max_tool_rounds,
            max_tool_calls,
            max_schema_retries: if verify_schema {
                MAX_SCHEMA_RETRIES
            } else {
                0
            },
            verify_schema,
        }
    }

    /// Safe, low-cardinality receipt; never includes prompts, tool arguments,
    /// model reasoning, tool results, or validation details.
    pub(crate) fn receipt(
        &self,
        tool_rounds: usize,
        tool_calls: usize,
        schema_verified: Option<bool>,
        schema_retries: usize,
    ) -> String {
        let schema = match (self.verify_schema, schema_verified) {
            (false, _) => "not_requested",
            (true, Some(true)) => "pass",
            (true, Some(false)) => "fail",
            (true, None) => "not_checked",
        };
        format!(
            "mode={} complexity={} max_rounds={} max_tool_calls={} rounds={} calls={} schema={} repairs={}",
            self.mode.as_str(),
            self.complexity,
            self.max_tool_rounds,
            self.max_tool_calls,
            tool_rounds,
            tool_calls,
            schema,
            schema_retries
        )
    }
}

fn complexity_score(request: &Value) -> u8 {
    let messages = request.get("messages").and_then(Value::as_array);
    let message_count = messages.map_or(0, Vec::len);
    let message_chars = messages.map_or(0, |rows| {
        rows.iter().map(recursive_text_len).sum()
    });
    let text_chars = message_chars
        + request.get("input").map_or(0, recursive_text_len)
        + request.get("instructions").map_or(0, recursive_text_len);
    let tool_count = request
        .get("tools")
        .and_then(Value::as_array)
        .map_or(0, Vec::len);
    let has_schema = request
        .pointer("/response_format/json_schema/schema")
        .is_some_and(Value::is_object);

    let mut score = 0_u8;
    if text_chars > 12_000 {
        score += 2;
    } else if text_chars > 3_000 {
        score += 1;
    }
    if message_count >= 8 {
        score += 1;
    }
    if tool_count >= 8 {
        score += 2;
    } else if tool_count >= 3 {
        score += 1;
    }
    if has_schema {
        score += 1;
    }
    score.min(6)
}

fn recursive_text_len(value: &Value) -> usize {
    match value {
        Value::String(text) => text.len(),
        Value::Array(items) => items.iter().map(recursive_text_len).sum(),
        Value::Object(fields) => fields.values().map(recursive_text_len).sum(),
        Value::Null | Value::Bool(_) | Value::Number(_) => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn unit__mode_from_headers__explicit_opt_in_only() {
        let mut headers = HeaderMap::new();
        assert_eq!(mode_from_headers(&headers), Ok(None));
        headers.insert(HEADER, "off".parse().unwrap());
        assert_eq!(mode_from_headers(&headers), Ok(None));
        headers.insert(HEADER, "auto".parse().unwrap());
        assert_eq!(mode_from_headers(&headers), Ok(Some(Mode::Auto)));
        headers.insert(HEADER, "deep".parse().unwrap());
        assert_eq!(mode_from_headers(&headers), Ok(Some(Mode::Deep)));
        headers.insert(HEADER, "swarm".parse().unwrap());
        assert!(mode_from_headers(&headers).is_err());
    }

    #[test]
    fn unit__auto_plan__easy_requests_get_small_budget() {
        let request = json!({
            "messages": [{"role": "user", "content": "What is 2 + 2?"}],
            "tools": [{"type": "function", "function": {"name": "lookup"}}]
        });
        let plan = Plan::for_request(Mode::Auto, &request);
        assert_eq!(plan.max_tool_rounds, 2);
        assert_eq!(plan.max_tool_calls, 4);
        assert_eq!(plan.max_schema_retries, 0);
        assert!(!plan.verify_schema);
    }

    #[test]
    fn unit__auto_plan__complex_requests_receive_more_compute() {
        let tools: Vec<Value> = (0..9)
            .map(|i| json!({"type": "function", "function": {"name": format!("tool_{i}")}}))
            .collect();
        let request = json!({
            "messages": [{"role": "user", "content": "x".repeat(13_000)}],
            "tools": tools,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"schema": {"type": "object"}}
            }
        });
        let plan = Plan::for_request(Mode::Auto, &request);
        assert_eq!(plan.max_tool_rounds, 4);
        assert_eq!(plan.max_tool_calls, 12);
        assert_eq!(plan.max_schema_retries, 1);
        assert!(plan.verify_schema);
    }

    #[test]
    fn unit__deep_plan__hard_caps_never_exceed_constants() {
        let request = json!({
            "messages": [{"role": "user", "content": "x".repeat(20_000)}],
            "tools": (0..30).map(|i| json!({"name": format!("tool_{i}")})).collect::<Vec<_>>(),
            "response_format": {
                "type": "json_schema",
                "json_schema": {"schema": {"type": "object"}}
            }
        });
        let plan = Plan::for_request(Mode::Deep, &request);
        assert_eq!(plan.max_tool_rounds, MAX_TOOL_ROUNDS);
        assert_eq!(plan.max_tool_calls, MAX_TOOL_CALLS);
        assert!(plan.max_schema_retries <= MAX_SCHEMA_RETRIES);
    }

    #[test]
    fn unit__receipt__contains_no_request_content() {
        let plan = Plan::for_request(Mode::Auto, &json!({
            "messages": [{"role": "user", "content": "private prompt"}]
        }));
        let receipt = plan.receipt(1, 2, None, 0);
        assert!(receipt.contains("mode=auto"));
        assert!(!receipt.contains("private prompt"));
    }
}
