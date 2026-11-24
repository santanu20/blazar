//! Gateway-wide adaptive inference control for normalized chat requests.
//!
//! The shared controller runs after protocol normalization and before the
//! resolved model backend. It can allocate bounded inference revisions,
//! deterministic schema verification, and (on supported local lanes) parallel
//! candidates. MCP is optional: when selected, its independent tool-round,
//! server-timeout, and per-key authorization limits remain authoritative.
//!
//! The initial policy is deterministic and request-shape driven. It does not
//! inspect hidden reasoning or spend an extra model call deciding whether to
//! spend another model call. Its thresholds require calibration against
//! held-out workloads before any quality-optimality claim is justified.

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
            (Mode::Auto, 2..=3) | (Mode::Deep, 0..=1) => (3, 8),
            (Mode::Auto, _) => (4, 12),
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
                if mode == Mode::Deep {
                    MAX_SCHEMA_RETRIES + 1
                } else {
                    MAX_SCHEMA_RETRIES
                }
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
    let message_chars = messages.map_or(0, |rows| rows.iter().map(recursive_text_len).sum());
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

/// Execution strategy selected by the gateway controller.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Strategy {
    Direct,
    SequentialRevision,
    ParallelVerifiedCandidates,
}

impl Strategy {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Direct => "direct",
            Self::SequentialRevision => "sequential_revision",
            Self::ParallelVerifiedCandidates => "parallel_verified_candidates",
        }
    }
}

impl Plan {
    pub(crate) fn strategy(&self, request: &Value) -> Strategy {
        let has_caller_tools = request
            .get("tools")
            .and_then(Value::as_array)
            .is_some_and(|tools| !tools.is_empty());
        let requests_multiple_choices = request
            .get("n")
            .and_then(Value::as_u64)
            .is_some_and(|choices| choices > 1);
        if has_caller_tools || requests_multiple_choices {
            Strategy::Direct
        } else if self.mode == Mode::Deep && self.verify_schema {
            Strategy::ParallelVerifiedCandidates
        } else if self.verify_schema || self.mode == Mode::Deep || self.complexity >= 3 {
            // Auto must retain one repair call for invalid structured output,
            // even when the request is otherwise classified as simple.
            Strategy::SequentialRevision
        } else {
            Strategy::Direct
        }
    }

    /// Hard call ceiling for this plan. Deep mode grants additional
    /// evaluator-driven repairs, not unconditional self-rewrites.
    pub(crate) const fn max_model_calls(&self, strategy: Strategy) -> usize {
        match (strategy, self.mode) {
            (Strategy::Direct, _) => 1,
            (Strategy::SequentialRevision, Mode::Auto) => 2,
            (Strategy::SequentialRevision, Mode::Deep) => 3,
            (Strategy::ParallelVerifiedCandidates, Mode::Auto) => 3,
            (Strategy::ParallelVerifiedCandidates, Mode::Deep) => 4,
        }
    }

    pub(crate) const fn max_candidates(strategy: Strategy) -> u64 {
        match strategy {
            Strategy::ParallelVerifiedCandidates => 2,
            Strategy::Direct | Strategy::SequentialRevision => 1,
        }
    }

    pub(crate) fn execution_receipt(
        &self,
        strategy: Strategy,
        model_calls: usize,
        candidates: usize,
        schema_verified: Option<bool>,
        revisions: usize,
        degraded: Option<&str>,
    ) -> String {
        let schema = match (self.verify_schema, schema_verified) {
            (false, _) => "not_requested",
            (true, Some(true)) => "pass",
            (true, Some(false)) => "fail",
            (true, None) => "not_checked",
        };
        let mut receipt = format!(
            "mode={} complexity={} strategy={} calls={} candidates={} schema={} revisions={}",
            self.mode.as_str(),
            self.complexity,
            strategy.as_str(),
            model_calls,
            candidates,
            schema,
            revisions
        );
        if let Some(reason) = degraded {
            let safe_reason: String = reason
                .chars()
                .filter(|ch| ch.is_ascii_alphanumeric() || *ch == '-' || *ch == '_')
                .take(32)
                .collect();
            if !safe_reason.is_empty() {
                receipt.push_str(" degraded=");
                receipt.push_str(&safe_reason);
            }
        }
        receipt
    }
}

const ADAPTIVE_MAX_REQUEST_SECS: u64 = 120;
const MAX_ADAPTIVE_RESPONSE_BYTES: usize = 32 * 1024 * 1024;

struct BufferedResponse {
    status: reqwest::StatusCode,
    headers: reqwest::header::HeaderMap,
    body: axum::body::Bytes,
}

fn rebuild_response(response: BufferedResponse, receipt: &str) -> reqwest::Response {
    let mut builder = axum::http::Response::builder().status(
        axum::http::StatusCode::from_u16(response.status.as_u16())
            .unwrap_or(axum::http::StatusCode::INTERNAL_SERVER_ERROR),
    );
    for (name, value) in &response.headers {
        if name == reqwest::header::CONTENT_LENGTH
            || name == reqwest::header::TRANSFER_ENCODING
            || name == reqwest::header::CONNECTION
            || name.as_str() == "x-blazar-adaptive-result"
        {
            continue;
        }
        builder = builder.header(name, value);
    }
    builder = builder.header("x-blazar-adaptive-result", receipt);
    let http_response = builder
        .body(response.body)
        .unwrap_or_else(|_| axum::http::Response::new(axum::body::Bytes::new()));
    reqwest::Response::from(http_response)
}

fn usage_from_body(body: &Value) -> Option<Value> {
    body.get("usage").filter(|usage| usage.is_object()).cloned()
}

fn merge_json_counters(total: &mut Value, next: &Value) {
    if let (Some(left), Some(right)) = (total.as_u64(), next.as_u64()) {
        *total = Value::from(left.saturating_add(right));
        return;
    }
    if let (Some(total_fields), Some(next_fields)) = (total.as_object_mut(), next.as_object()) {
        for (key, value) in next_fields {
            match total_fields.get_mut(key) {
                Some(existing) => merge_json_counters(existing, value),
                None => {
                    total_fields.insert(key.clone(), value.clone());
                }
            }
        }
    }
}

fn merge_usage(total: &mut Option<Value>, body: &Value) {
    let Some(next) = usage_from_body(body) else {
        return;
    };
    match total {
        Some(current) => merge_json_counters(current, &next),
        None => *total = Some(next),
    }
}

fn patch_usage(body: &mut Value, usage: Option<Value>) {
    if let Some(mut usage) = usage {
        let prompt = usage.get("prompt_tokens").and_then(Value::as_u64);
        let completion = usage.get("completion_tokens").and_then(Value::as_u64);
        if let (Some(prompt), Some(completion)) = (prompt, completion) {
            usage["total_tokens"] = Value::from(prompt.saturating_add(completion));
        }
        body["usage"] = usage;
    }
}

fn message_has_tool_calls(body: &Value) -> bool {
    body.pointer("/choices/0/message/tool_calls")
        .and_then(Value::as_array)
        .is_some_and(|calls| !calls.is_empty())
        || body.pointer("/choices/0/message/function_call").is_some()
        || body
            .pointer("/choices/0/finish_reason")
            .and_then(Value::as_str)
            .is_some_and(|reason| reason == "tool_calls" || reason == "function_call")
}

pub(crate) fn stamp(response: &mut axum::response::Response, receipt: Option<&str>) {
    if let Some(value) = receipt.and_then(|v| axum::http::HeaderValue::from_str(v).ok()) {
        response
            .headers_mut()
            .insert("x-blazar-adaptive-result", value);
    }
}

fn response_content(body: &Value) -> Option<&str> {
    body.pointer("/choices/0/message/content")
        .and_then(Value::as_str)
}

fn append_revision_prompt(
    request: &mut Value,
    assistant_message: &Value,
    schema_feedback: Option<&str>,
) -> Result<(), String> {
    let messages = request
        .get_mut("messages")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| "adaptive chat request lost its messages array".to_string())?;
    let mut assistant = assistant_message.clone();
    assistant["role"] = Value::String("assistant".into());
    messages.push(assistant);
    let prompt = if let Some(feedback) = schema_feedback {
        format!(
            "The previous candidate failed an objective response-contract check. \\
             Validation feedback: {feedback} Address this specific defect while preserving \\
             all already-correct information. Return only the complete final answer."
        )
    } else {
        "The previous candidate was incomplete or empty according to the gateway's \\
         response-completion checks. Produce a complete answer to the original request. \\
         Preserve correct information and do not add unsupported claims. Return only \\
         the final answer."
            .to_string()
    };
    messages.push(serde_json::json!({"role": "user", "content": prompt}));
    Ok(())
}

fn schema_violation(
    state: &crate::state::AppState,
    plan: &Plan,
    schema: Option<&Value>,
    response: &Value,
) -> Option<String> {
    if !plan.verify_schema {
        return None;
    }
    let schema = schema?;
    match response_content(response) {
        Some(content) => match serde_json::from_str::<Value>(content) {
            Ok(candidate) => state.sentinel.schema_check(schema, &candidate),
            Err(error) => Some(format!("response content is not valid JSON: {error}")),
        },
        None => Some("response did not contain JSON text at choices[0].message.content".into()),
    }
}

/// Contract-level assessment for an answer candidate. This deliberately
/// does not claim factual or semantic correctness: without a task-specific
/// independent verifier, Blazar can prove transport, response-envelope,
/// completion and requested-schema properties only.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CandidateQuality {
    http_success: bool,
    parseable: bool,
    schema_valid: Option<bool>,
    complete: bool,
    non_empty: bool,
}

impl CandidateQuality {
    fn rank(self) -> (u8, u8, u8, u8, u8) {
        (
            u8::from(self.http_success),
            u8::from(self.schema_valid != Some(false)),
            u8::from(self.complete),
            u8::from(self.non_empty),
            u8::from(self.parseable),
        )
    }

    fn is_better_than(self, incumbent: Self) -> bool {
        self.rank() > incumbent.rank()
    }

    fn is_satisfied(self) -> bool {
        self.http_success
            && self.parseable
            && self.schema_valid != Some(false)
            && self.complete
            && self.non_empty
    }

    fn needs_revision(self) -> bool {
        self.http_success && (self.schema_valid == Some(false) || !self.complete || !self.non_empty)
    }
}

fn assess_candidate(
    state: &crate::state::AppState,
    plan: &Plan,
    schema: Option<&Value>,
    status: reqwest::StatusCode,
    body: &Value,
) -> CandidateQuality {
    let schema_valid = if plan.verify_schema {
        Some(schema_violation(state, plan, schema, body).is_none())
    } else {
        None
    };
    let content = response_content(body).unwrap_or_default();
    let finish_reason = body
        .pointer("/choices/0/finish_reason")
        .and_then(Value::as_str);
    let complete = match finish_reason {
        Some("length") => false,
        Some("stop" | "content_filter") | None => true,
        Some(_) => false,
    };
    CandidateQuality {
        http_success: status.is_success(),
        parseable: true,
        schema_valid,
        complete,
        non_empty: !content.trim().is_empty(),
    }
}

fn revision_feedback(quality: CandidateQuality, schema_feedback: Option<&str>) -> Option<String> {
    if let Some(feedback) = schema_feedback {
        return Some(feedback.to_string());
    }
    if !quality.needs_revision() {
        return None;
    }
    if !quality.non_empty {
        return Some("response content is empty; provide a complete answer".into());
    }
    if !quality.complete {
        return Some(
            "response ended before a normal completion (finish_reason is not stop); complete the answer without dropping required content".into()
        );
    }
    None
}

fn loop_receipt(
    plan: &Plan,
    strategy: Strategy,
    model_calls: usize,
    candidates: usize,
    schema_verified: Option<bool>,
    revisions: usize,
    degraded: Option<&str>,
    accepted: usize,
    rejected: usize,
    stop_reason: &str,
) -> String {
    let base = plan.execution_receipt(
        strategy,
        model_calls,
        candidates,
        schema_verified,
        revisions,
        degraded,
    );
    format!("{base} accepted={accepted} rejected={rejected} stop={stop_reason}")
}

/// Execute one internal local model round through Blazar's normal proxy
/// pipeline. Every revision/candidate therefore participates in key budget
/// reservation, usage sniffing, Sentinel and local service telemetry.
#[allow(clippy::result_large_err)]
async fn local_proxy_round(
    state: &std::sync::Arc<crate::state::AppState>,
    engine: &blazar_runtime::EngineRef,
    request: &Value,
    headers: &HeaderMap,
    key: Option<&crate::keys::KeyCtx>,
    candidates: Option<u64>,
) -> Result<BufferedResponse, axum::response::Response> {
    let mut child_headers = headers.clone();
    child_headers.remove(HEADER);
    child_headers.remove("x-blazar-mcp");
    child_headers.remove("x-blazar-adaptive-result");
    let body = axum::body::Bytes::from(serde_json::to_vec(request).unwrap_or_default());
    let sf_gate = candidates.map_or(
        crate::proxy::SfGate::AdaptiveDirect,
        crate::proxy::SfGate::AdaptiveCandidates,
    );
    let response = crate::proxy::proxy_request(
        state,
        engine,
        &engine.name,
        &axum::http::Method::POST,
        "/v1/chat/completions",
        &child_headers,
        body,
        0,
        None,
        None,
        key.cloned(),
        None, // Intermediate drafts must not create completion metadata cards.
        sf_gate,
        None,
    )
    .await;
    let (parts, body) = response.into_parts();
    let bytes = axum::body::to_bytes(body, MAX_ADAPTIVE_RESPONSE_BYTES)
        .await
        .map_err(|error| {
            crate::mcp::teaching_error(
                axum::http::StatusCode::BAD_GATEWAY,
                &format!("adaptive proxy response body failed: {error}"),
            )
        })?;
    let status = reqwest::StatusCode::from_u16(parts.status.as_u16())
        .unwrap_or(reqwest::StatusCode::BAD_GATEWAY);
    Ok(BufferedResponse {
        status,
        headers: parts.headers,
        body: bytes,
    })
}

fn candidate_usage(headers: &reqwest::header::HeaderMap, asked: usize) -> (usize, Option<String>) {
    let Some(receipt) = headers
        .get("x-blazar-adaptive-candidates")
        .and_then(|value| value.to_str().ok())
    else {
        return (1, None);
    };
    let used = receipt
        .split_whitespace()
        .find_map(|part| part.strip_prefix("used="))
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1)
        .clamp(1, asked.max(1));
    let reason = receipt
        .split_whitespace()
        .find_map(|part| part.strip_prefix("reason="))
        .map(str::to_string);
    (used, reason)
}

#[allow(clippy::result_large_err)] // Admission failures use the gateway's standard Axum response.
async fn adaptive_admission_guard(
    state: &std::sync::Arc<crate::state::AppState>,
    engine: &blazar_runtime::EngineRef,
    request: &Value,
    headers: &HeaderMap,
    key: Option<&crate::keys::KeyCtx>,
) -> Result<crate::proxy::InFlightGuard, axum::response::Response> {
    let body = serde_json::to_vec(request).unwrap_or_default();
    let has_tools = request
        .get("tools")
        .and_then(Value::as_array)
        .is_some_and(|tools| !tools.is_empty());
    let class = crate::workload::classify_for(
        state,
        &engine.name,
        crate::workload::WorkloadSurface::Chat,
        has_tools,
        false,
    );
    let priority = crate::queue::Priority::from_header(
        headers
            .get("x-blazar-priority")
            .and_then(|value| value.to_str().ok()),
    );
    let deadline_ms = headers
        .get("x-blazar-deadline-ms")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok());
    let wfq = key.map(|key| (key.name.as_str(), key.weight));
    let exclusive = crate::proxy::exclusive_intent_bytes(state, &engine.name, &body);
    crate::proxy::admission_gate_slo(
        state,
        &engine.name,
        priority,
        class,
        deadline_ms,
        body.len(),
        wfq,
        exclusive,
    )
    .await
    .map_err(|response| *response)
}

/// Run an adaptive chat task on any locally resolved Blazar engine. All
/// local engines use the same OpenAI-compatible child chat endpoint.
///
/// The request argument is a private working copy. Intermediate drafts,
/// evaluator prompts and tool transcripts are never committed to the
/// caller's conversation store. Caller-owned tool calls remain handoffs;
/// only an authorized MCP selector can enable gateway-side tool execution.
#[allow(clippy::result_large_err)] // Axum Response is the gateway handler's error currency.
pub(crate) async fn run_local_chat(
    state: &std::sync::Arc<crate::state::AppState>,
    engine: &blazar_runtime::EngineRef,
    openai_body: &Value,
    mode: Mode,
    headers: &HeaderMap,
    key: Option<&crate::keys::KeyCtx>,
) -> Result<reqwest::Response, axum::response::Response> {
    let request = openai_body.clone();
    let budget = std::time::Duration::from_secs(ADAPTIVE_MAX_REQUEST_SECS);
    match tokio::time::timeout(
        budget,
        run_local_chat_inner(state, engine, request, mode, headers, key),
    )
    .await
    {
        Ok(result) => result,
        Err(_) => Err(crate::mcp::teaching_error(
            axum::http::StatusCode::GATEWAY_TIMEOUT,
            &format!(
                "adaptive inference exceeded its {ADAPTIVE_MAX_REQUEST_SECS}-second wall-clock budget; intermediate context was discarded"
            ),
        )),
    }
}

#[allow(clippy::too_many_lines)]
#[allow(clippy::result_large_err)] // Axum Response is the gateway handler's error currency.
async fn run_local_chat_inner(
    state: &std::sync::Arc<crate::state::AppState>,
    engine: &blazar_runtime::EngineRef,
    mut request: Value,
    mode: Mode,
    headers: &HeaderMap,
    key: Option<&crate::keys::KeyCtx>,
) -> Result<reqwest::Response, axum::response::Response> {
    let original_request = request.clone();
    let plan = Plan::for_request(mode, &request);
    let strategy = plan.strategy(&request);
    let schema = request
        .pointer("/response_format/json_schema/schema")
        .filter(|schema| schema.is_object())
        .cloned();
    let so_kind = crate::proxy::structured_output_kind(Some(&request));
    let model = request
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or(&engine.name)
        .to_string();
    let mut revisions = 0usize;
    let mut accepted = 0usize;
    let mut rejected = 0usize;
    let mut schema_repair_attempts = 0usize;
    let mut usage_total = None;
    let mut stop_reason: String;

    // All local calls stay on the normal proxy/accounting path.
    let asked_candidates =
        (strategy == Strategy::ParallelVerifiedCandidates).then(|| Plan::max_candidates(strategy));
    let mut current =
        local_proxy_round(state, engine, &request, headers, key, asked_candidates).await?;
    let (candidates, mut degraded, mut model_calls) = if let Some(asked) = asked_candidates {
        let (used, reason) = candidate_usage(&current.headers, usize::try_from(asked).unwrap_or(1));
        (used, reason, used)
    } else {
        (1, None, 1)
    };

    let mut current_body = match serde_json::from_slice::<Value>(&current.body) {
        Ok(body) => body,
        Err(_) => {
            stop_reason = "unparseable_initial_response".into();
            return Ok(rebuild_response(
                current,
                &loop_receipt(
                    &plan,
                    strategy,
                    model_calls,
                    candidates,
                    None,
                    revisions,
                    degraded.as_deref(),
                    accepted,
                    rejected,
                    &stop_reason,
                ),
            ));
        }
    };
    merge_usage(&mut usage_total, &current_body);
    let mut best_quality =
        assess_candidate(state, &plan, schema.as_ref(), current.status, &current_body);
    let mut schema_verified = best_quality.schema_valid;

    if !current.status.is_success() {
        stop_reason = "upstream_error".into();
    } else if message_has_tool_calls(&current_body) {
        // Caller-owned function calls remain handoffs; the controller does
        // not execute them or mistake them for a final natural-language answer.
        stop_reason = "tool_handoff".into();
    } else {
        loop {
            let schema_detail = if best_quality.schema_valid == Some(false) {
                schema_violation(state, &plan, schema.as_ref(), &current_body)
            } else {
                None
            };
            let feedback = revision_feedback(best_quality, schema_detail.as_deref());
            let Some(feedback) = feedback else {
                stop_reason = "quality_gate_pass".into();
                break;
            };
            if model_calls >= plan.max_model_calls(strategy) {
                stop_reason = "model_call_budget".into();
                break;
            }
            if best_quality.schema_valid == Some(false)
                && schema_repair_attempts >= plan.max_schema_retries
            {
                stop_reason = "schema_repair_budget".into();
                break;
            }

            let assistant = current_body
                .pointer("/choices/0/message")
                .cloned()
                .unwrap_or_else(|| serde_json::json!({"role": "assistant", "content": ""}));
            let detail: String = feedback.chars().take(512).collect();
            append_revision_prompt(&mut request, &assistant, Some(&detail)).map_err(|error| {
                crate::mcp::teaching_error(axum::http::StatusCode::INTERNAL_SERVER_ERROR, &error)
            })?;
            if best_quality.schema_valid == Some(false) {
                schema_repair_attempts += 1;
            }

            let revised = match local_proxy_round(state, engine, &request, headers, key, None).await
            {
                Ok(response) => response,
                Err(_error) => {
                    stop_reason = "revision_error".into();
                    if degraded.is_none() {
                        degraded = Some("revision_error".into());
                    }
                    tracing::warn!(model = %model, "adaptive revision failed; retaining the best prior candidate");
                    break;
                }
            };
            model_calls += 1;
            revisions += 1;

            let revised_body = match serde_json::from_slice::<Value>(&revised.body) {
                Ok(body) => body,
                Err(_) => {
                    rejected += 1;
                    stop_reason = "unparseable_revision".into();
                    break;
                }
            };
            merge_usage(&mut usage_total, &revised_body);
            let revised_quality =
                assess_candidate(state, &plan, schema.as_ref(), revised.status, &revised_body);
            if !revised_quality.is_better_than(best_quality) {
                rejected += 1;
                stop_reason = "no_objective_improvement".into();
                break;
            }

            // Accept only a strict improvement. Equal-quality rewrites and
            // regressions never replace the incumbent response.
            current = revised;
            current_body = revised_body;
            best_quality = revised_quality;
            schema_verified = best_quality.schema_valid;
            accepted += 1;

            if message_has_tool_calls(&current_body) {
                stop_reason = "tool_handoff".into();
                break;
            }
            if best_quality.is_satisfied() {
                stop_reason = "quality_gate_pass".into();
                break;
            }
        }
    }

    if let Ok(mut final_body) = serde_json::from_slice::<Value>(&current.body) {
        patch_usage(&mut final_body, usage_total);
        current.body = axum::body::Bytes::from(
            serde_json::to_vec(&final_body).unwrap_or_else(|_| current.body.to_vec()),
        );
        crate::metadata_card::persist_from_response(
            state,
            &model,
            Some(&original_request),
            &current.body,
        );
    }
    if let Some(kind) = so_kind
        && let Ok(value) = axum::http::HeaderValue::from_str(kind)
    {
        current.headers.insert("x-blazar-structured-output", value);
    }
    Ok(rebuild_response(
        current,
        &loop_receipt(
            &plan,
            strategy,
            model_calls,
            candidates,
            schema_verified,
            revisions,
            degraded.as_deref(),
            accepted,
            rejected,
            &stop_reason,
        ),
    ))
}

async fn remote_round(
    state: &std::sync::Arc<crate::state::AppState>,
    model: &str,
    method: &axum::http::Method,
    path_query: &str,
    headers: &axum::http::HeaderMap,
    request: &Value,
) -> Result<BufferedResponse, axum::response::Response> {
    let mut outbound = headers.clone();
    outbound.remove(HEADER);
    outbound.remove("x-blazar-mcp");
    let bytes = serde_json::to_vec(request).unwrap_or_default();
    let response = crate::remotes::forward_with_health(
        state,
        model,
        method,
        path_query,
        &outbound,
        axum::body::Bytes::from(bytes),
    )
    .await;
    let (parts, body) = response.into_parts();
    let bytes = axum::body::to_bytes(body, MAX_ADAPTIVE_RESPONSE_BYTES)
        .await
        .map_err(|error| {
            crate::mcp::teaching_error(
                axum::http::StatusCode::BAD_GATEWAY,
                &format!("remote adaptive response body failed: {error}"),
            )
        })?;
    let status = reqwest::StatusCode::from_u16(parts.status.as_u16())
        .unwrap_or(reqwest::StatusCode::BAD_GATEWAY);
    Ok(BufferedResponse {
        status,
        headers: parts.headers,
        body: bytes,
    })
}

fn rebuild_axum_response(response: BufferedResponse, receipt: &str) -> axum::response::Response {
    let mut builder = axum::http::Response::builder().status(
        axum::http::StatusCode::from_u16(response.status.as_u16())
            .unwrap_or(axum::http::StatusCode::INTERNAL_SERVER_ERROR),
    );
    for (name, value) in &response.headers {
        if name == reqwest::header::CONTENT_LENGTH
            || name == reqwest::header::TRANSFER_ENCODING
            || name == reqwest::header::CONNECTION
            || name.as_str() == "x-blazar-adaptive-result"
        {
            continue;
        }
        builder = builder.header(name, value);
    }
    builder = builder.header("x-blazar-adaptive-result", receipt);
    builder
        .body(axum::body::Body::from(response.body))
        .unwrap_or_else(|_| axum::response::Response::new(axum::body::Body::empty()))
}

/// Adaptive execution for an explicitly configured OpenAI-compatible remote.
/// Local engines and remotes therefore share the same request profiler,
/// verification and revision policy; remote selection/health/affinity remain
/// owned by Blazar's existing remote forwarder.
pub(crate) async fn run_remote_chat(
    state: &std::sync::Arc<crate::state::AppState>,
    model: &str,
    method: &axum::http::Method,
    path_query: &str,
    headers: &axum::http::HeaderMap,
    openai_body: &Value,
    mode: Mode,
) -> axum::response::Response {
    let mut selector_body = openai_body.clone();
    let selector = match state
        .mcp
        .resolve(crate::mcp::selector_from(headers, Some(&mut selector_body)))
    {
        Ok(selector) => selector,
        Err(message) => {
            return crate::mcp::teaching_error(axum::http::StatusCode::BAD_REQUEST, &message);
        }
    };
    if selector.is_some() {
        return crate::mcp::teaching_error(
            axum::http::StatusCode::BAD_REQUEST,
            "gateway MCP execution is available on local engine lanes; adaptive inference itself supports OpenAI-compatible remotes",
        );
    }
    let request = selector_body;
    let budget = std::time::Duration::from_secs(ADAPTIVE_MAX_REQUEST_SECS);
    match tokio::time::timeout(
        budget,
        run_remote_chat_inner(state, model, method, path_query, headers, request, mode),
    )
    .await
    {
        Ok(result) => result.unwrap_or_else(std::convert::identity),
        Err(_) => crate::mcp::teaching_error(
            axum::http::StatusCode::GATEWAY_TIMEOUT,
            &format!(
                "adaptive remote inference exceeded its {ADAPTIVE_MAX_REQUEST_SECS}-second wall-clock budget; intermediate context was discarded"
            ),
        ),
    }
}

#[allow(clippy::result_large_err)] // Axum Response is the gateway handler's error currency.
async fn run_remote_chat_inner(
    state: &std::sync::Arc<crate::state::AppState>,
    model: &str,
    method: &axum::http::Method,
    path_query: &str,
    headers: &axum::http::HeaderMap,
    mut request: Value,
    mode: Mode,
) -> Result<axum::response::Response, axum::response::Response> {
    let original_request = request.clone();
    let plan = Plan::for_request(mode, &request);
    // Remote fan-out is not exposed by the current remote admission contract.
    // Keep candidate evaluation and repair semantics identical, using serial calls.
    let proposed = plan.strategy(&request);
    let strategy = if proposed == Strategy::ParallelVerifiedCandidates {
        Strategy::SequentialRevision
    } else {
        proposed
    };
    let schema = request
        .pointer("/response_format/json_schema/schema")
        .filter(|schema| schema.is_object())
        .cloned();
    let model_name = request
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or(model)
        .to_string();
    let mut model_calls = 0usize;
    let mut revisions = 0usize;
    let mut accepted = 0usize;
    let mut rejected = 0usize;
    let mut schema_repair_attempts = 0usize;
    let mut schema_verified = None;
    let mut usage_total = None;
    let mut stop_reason = "quality_gate_pass".to_string();

    let first = remote_round(state, model, method, path_query, headers, &request).await?;
    model_calls += 1;
    let mut current = first;
    if !current.status.is_success() {
        stop_reason = "upstream_error".into();
        return Ok(rebuild_axum_response(
            current,
            &loop_receipt(
                &plan,
                strategy,
                model_calls,
                1,
                None,
                revisions,
                None,
                accepted,
                rejected,
                &stop_reason,
            ),
        ));
    }
    let mut current_body = match serde_json::from_slice::<Value>(&current.body) {
        Ok(body) => body,
        Err(_) => {
            stop_reason = "unparseable_initial_response".into();
            return Ok(rebuild_axum_response(
                current,
                &loop_receipt(
                    &plan,
                    strategy,
                    model_calls,
                    1,
                    None,
                    revisions,
                    None,
                    accepted,
                    rejected,
                    &stop_reason,
                ),
            ));
        }
    };
    merge_usage(&mut usage_total, &current_body);
    let mut best_quality =
        assess_candidate(state, &plan, schema.as_ref(), current.status, &current_body);
    schema_verified = best_quality.schema_valid;

    if message_has_tool_calls(&current_body) {
        stop_reason = "tool_handoff".into();
    } else {
        loop {
            let schema_detail = if best_quality.schema_valid == Some(false) {
                schema_violation(state, &plan, schema.as_ref(), &current_body)
            } else {
                None
            };
            let feedback = revision_feedback(best_quality, schema_detail.as_deref());
            let Some(feedback) = feedback else {
                stop_reason = "quality_gate_pass".into();
                break;
            };
            if model_calls >= plan.max_model_calls(strategy) {
                stop_reason = "model_call_budget".into();
                break;
            }
            if best_quality.schema_valid == Some(false)
                && schema_repair_attempts >= plan.max_schema_retries
            {
                stop_reason = "schema_repair_budget".into();
                break;
            }

            let assistant = current_body
                .pointer("/choices/0/message")
                .cloned()
                .unwrap_or_else(|| serde_json::json!({"role": "assistant", "content": ""}));
            let detail: String = feedback.chars().take(512).collect();
            append_revision_prompt(&mut request, &assistant, Some(&detail)).map_err(|error| {
                crate::mcp::teaching_error(axum::http::StatusCode::INTERNAL_SERVER_ERROR, &error)
            })?;
            if best_quality.schema_valid == Some(false) {
                schema_repair_attempts += 1;
            }

            let revised = match remote_round(state, model, method, path_query, headers, &request)
                .await
            {
                Ok(response) => response,
                Err(error) => {
                    stop_reason = "revision_error".into();
                    tracing::warn!(model = %model_name, "adaptive remote revision failed; retaining the best prior candidate");
                    let _ = error;
                    break;
                }
            };
            model_calls += 1;
            revisions += 1;
            let revised_body = match serde_json::from_slice::<Value>(&revised.body) {
                Ok(body) => body,
                Err(_) => {
                    rejected += 1;
                    stop_reason = "unparseable_revision".into();
                    break;
                }
            };
            merge_usage(&mut usage_total, &revised_body);
            let revised_quality =
                assess_candidate(state, &plan, schema.as_ref(), revised.status, &revised_body);
            if !revised_quality.is_better_than(best_quality) {
                rejected += 1;
                stop_reason = "no_objective_improvement".into();
                break;
            }
            current = revised;
            current_body = revised_body;
            best_quality = revised_quality;
            schema_verified = best_quality.schema_valid;
            accepted += 1;
            if message_has_tool_calls(&current_body) {
                stop_reason = "tool_handoff".into();
                break;
            }
            if best_quality.is_satisfied() {
                stop_reason = "quality_gate_pass".into();
                break;
            }
        }
    }

    if let Ok(mut final_body) = serde_json::from_slice::<Value>(&current.body) {
        patch_usage(&mut final_body, usage_total);
        current.body = axum::body::Bytes::from(
            serde_json::to_vec(&final_body).unwrap_or_else(|_| current.body.to_vec()),
        );
        crate::metadata_card::persist_from_response(
            state,
            &model_name,
            Some(&original_request),
            &current.body,
        );
    }
    Ok(rebuild_axum_response(
        current,
        &loop_receipt(
            &plan,
            strategy,
            model_calls,
            1,
            schema_verified,
            revisions,
            None,
            accepted,
            rejected,
            &stop_reason,
        ),
    ))
}

/// Per-adapter inputs shared by the normalized gateway execution entry point.
pub(crate) struct ChatExecutionContext<'a> {
    pub(crate) mcp_selector: Option<&'a str>,
    pub(crate) key: Option<&'a crate::keys::KeyCtx>,
    pub(crate) headers: &'a HeaderMap,
    /// `OpenAI` enters before the SLO gate. Anthropic and Ollama arrive with
    /// their protocol-boundary admission guard already held.
    pub(crate) acquire_slo_guard: bool,
}

/// Shared protocol-adapter entry point. MCP is an optional tool provider;
/// ordinary inference uses the same controller without requiring MCP.
#[allow(clippy::result_large_err)] // Axum Response is the gateway handler's error currency.
pub(crate) async fn execute_chat(
    state: &std::sync::Arc<crate::state::AppState>,
    engine: &blazar_runtime::EngineRef,
    openai_body: &mut Value,
    mode: Mode,
    context: ChatExecutionContext<'_>,
) -> Result<reqwest::Response, axum::response::Response> {
    let _inflight = if context.acquire_slo_guard {
        Some(
            adaptive_admission_guard(state, engine, openai_body, context.headers, context.key)
                .await?,
        )
    } else {
        None
    };
    if let Some(selector) = context.mcp_selector {
        crate::mcp::chat_via_mcp(
            state,
            engine,
            openai_body,
            selector,
            context.key,
            Some(mode),
        )
        .await
    } else {
        run_local_chat(
            state,
            engine,
            openai_body,
            mode,
            context.headers,
            context.key,
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]
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
    fn unit__strategy__direct_revision_and_parallel_candidates() {
        let simple = json!({
            "messages": [{"role": "user", "content": "What is 2 + 2?"}]
        });
        assert_eq!(
            Plan::for_request(Mode::Auto, &simple).strategy(&simple),
            Strategy::Direct
        );
        let caller_tools = json!({
            "messages": [{"role": "user", "content": "Use a tool."}],
            "tools": [{"type": "function", "function": {"name": "search"}}]
        });
        assert_eq!(
            Plan::for_request(Mode::Deep, &caller_tools).strategy(&caller_tools),
            Strategy::Direct
        );
        let multiple_choices = json!({
            "messages": [{"role": "user", "content": "Give alternatives."}],
            "n": 2,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"schema": {"type": "object"}}
            }
        });
        assert_eq!(
            Plan::for_request(Mode::Deep, &multiple_choices).strategy(&multiple_choices),
            Strategy::Direct
        );
        let mut complex_messages = vec![json!({"role": "user", "content": "x".repeat(13_000)})];
        complex_messages.extend((1..8).map(|_| json!({"role": "user", "content": "context"})));
        let complex = json!({"messages": complex_messages});
        assert_eq!(
            Plan::for_request(Mode::Auto, &complex).strategy(&complex),
            Strategy::SequentialRevision
        );
        let structured = json!({
            "messages": [{"role": "user", "content": "Return a JSON object."}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"schema": {"type": "object"}}
            }
        });
        assert_eq!(
            Plan::for_request(Mode::Deep, &structured).strategy(&structured),
            Strategy::ParallelVerifiedCandidates
        );
    }

    #[test]
    fn unit__usage_merge__adds_nested_token_counters() {
        let mut total = Some(json!({
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "prompt_tokens_details": {"cached_tokens": 2}
        }));
        let next = json!({
            "prompt_tokens": 20,
            "completion_tokens": 7,
            "total_tokens": 27,
            "prompt_tokens_details": {"cached_tokens": 3}
        });
        merge_usage(&mut total, &json!({"usage": next}));
        let usage = total.expect("usage counters should be present");
        assert_eq!(usage["prompt_tokens"], 30);
        assert_eq!(usage["completion_tokens"], 12);
        assert_eq!(usage["total_tokens"], 42);
        assert_eq!(usage["prompt_tokens_details"]["cached_tokens"], 5);

        // Best-of-N sums prompt and completion tokens but may leave the
        // winner's original total_tokens untouched. Normalize after merging.
        let mut response = json!({});
        patch_usage(
            &mut response,
            Some(json!({
                "prompt_tokens": 30,
                "completion_tokens": 12,
                "total_tokens": 15
            })),
        );
        assert_eq!(response["usage"]["total_tokens"], 42);
    }

    #[test]
    fn unit__tool_handoff__recognizes_calls_and_finish_reasons() {
        assert!(message_has_tool_calls(&json!({
            "choices": [{
                "message": {
                    "tool_calls": [{"id": "call_1", "type": "function"}]
                }
            }]
        })));
        assert!(message_has_tool_calls(&json!({
            "choices": [{
                "message": {"content": null},
                "finish_reason": "tool_calls"
            }]
        })));
        assert!(message_has_tool_calls(&json!({
            "choices": [{
                "message": {"function_call": {"name": "lookup"}},
                "finish_reason": "function_call"
            }]
        })));
        assert!(!message_has_tool_calls(&json!({
            "choices": [{
                "message": {"content": "finished"},
                "finish_reason": "stop"
            }]
        })));
    }

    #[test]
    fn unit__revision_context__stays_in_the_working_copy() {
        let mut working = json!({
            "messages": [{"role": "user", "content": "Return valid JSON."}]
        });
        let assistant = json!({"role": "assistant", "content": "not json"});
        append_revision_prompt(
            &mut working,
            &assistant,
            Some("response content is not valid JSON"),
        )
        .expect("working context should accept a repair turn");
        assert_eq!(working["messages"].as_array().unwrap().len(), 3);
        assert_eq!(working["messages"][1]["role"], "assistant");
        assert_eq!(working["messages"][2]["role"], "user");
        assert!(
            working["messages"][2]["content"]
                .as_str()
                .unwrap()
                .contains("Validation feedback")
        );
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
    fn unit__candidate_quality__schema_validity_beats_completion_only() {
        let invalid_but_complete = CandidateQuality {
            http_success: true,
            parseable: true,
            schema_valid: Some(false),
            complete: true,
            non_empty: true,
        };
        let valid_but_truncated = CandidateQuality {
            http_success: true,
            parseable: true,
            schema_valid: Some(true),
            complete: false,
            non_empty: true,
        };
        assert!(valid_but_truncated.is_better_than(invalid_but_complete));
    }

    #[test]
    fn unit__candidate_quality__completion_beats_truncation() {
        let truncated = CandidateQuality {
            http_success: true,
            parseable: true,
            schema_valid: None,
            complete: false,
            non_empty: true,
        };
        let complete = CandidateQuality {
            complete: true,
            ..truncated
        };
        assert!(complete.is_better_than(truncated));
        assert!(!truncated.is_better_than(complete));
    }

    #[test]
    fn unit__candidate_quality__ties_preserve_incumbent() {
        let incumbent = CandidateQuality {
            http_success: true,
            parseable: true,
            schema_valid: None,
            complete: true,
            non_empty: true,
        };
        assert!(!incumbent.is_better_than(incumbent));
    }

    #[test]
    fn unit__candidate_quality__empty_and_failed_candidates_need_repair() {
        let empty = CandidateQuality {
            http_success: true,
            parseable: true,
            schema_valid: None,
            complete: true,
            non_empty: false,
        };
        let failed = CandidateQuality {
            http_success: false,
            ..empty
        };
        assert!(empty.needs_revision());
        assert!(!failed.needs_revision());
    }

    #[test]
    fn unit__auto_schema__gets_one_repair_opportunity() {
        let request = json!({
            "messages": [{"role": "user", "content": "Extract one field."}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"schema": {"type": "object"}}
            }
        });
        let plan = Plan::for_request(Mode::Auto, &request);
        assert_eq!(plan.strategy(&request), Strategy::SequentialRevision);
        assert_eq!(plan.max_schema_retries, 1);
        assert_eq!(plan.max_model_calls(Strategy::SequentialRevision), 2);
    }

    #[test]
    fn unit__deep_mode__allocates_bounded_iterative_repairs() {
        let request = json!({
            "messages": [{"role": "user", "content": "Return valid JSON."}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"schema": {"type": "object"}}
            }
        });
        let plan = Plan::for_request(Mode::Deep, &request);
        assert_eq!(plan.max_schema_retries, 2);
        assert_eq!(plan.max_model_calls(Strategy::SequentialRevision), 3);
        assert_eq!(
            plan.max_model_calls(Strategy::ParallelVerifiedCandidates),
            4
        );
    }

    #[test]
    fn unit__receipt__contains_no_request_content() {
        let plan = Plan::for_request(
            Mode::Auto,
            &json!({
                "messages": [{"role": "user", "content": "private prompt"}]
            }),
        );
        let receipt = plan.receipt(1, 2, None, 0);
        assert!(receipt.contains("mode=auto"));
        assert!(!receipt.contains("private prompt"));
    }
}
