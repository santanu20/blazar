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

use std::collections::HashMap;
use std::fmt::Write as _;
use std::fmt::Write as _;

use axum::http::HeaderMap;
use serde_json::Value;

pub(crate) const HEADER: &str = "x-blazar-adaptive";
const MAX_TOOL_ROUNDS: usize = 6;
const MAX_TOOL_CALLS: usize = 20;
const MAX_SCHEMA_RETRIES: usize = 1;
const MAX_DEEP_SCHEMA_RETRIES: usize = 3;

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
    pub task_profile: crate::reasoning::TaskProfile,
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
                    MAX_DEEP_SCHEMA_RETRIES
                } else {
                    MAX_SCHEMA_RETRIES
                }
            } else {
                0
            },
            verify_schema,
            task_profile: crate::reasoning::TaskProfile::from_request(request),
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
        } else if self.mode == Mode::Deep
            && request.get("seed").is_none()
            && request.get("temperature").and_then(Value::as_f64) != Some(0.0)
        {
            // Deep mode explores independent trajectories for ordinary chat,
            // not only JSON Schema tasks. Explicit deterministic sampling retains
            // a single path.
            Strategy::ParallelVerifiedCandidates
        } else if self.verify_schema
            || self.task_profile.has_deterministic_verifier()
            || self.mode == Mode::Deep
            || self.complexity >= 3
        {
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
            (Strategy::SequentialRevision, Mode::Deep) => 4,
            (Strategy::ParallelVerifiedCandidates, Mode::Auto) => 3,
            // Three initial candidates plus up to four verifier-guided revisions.
            (Strategy::ParallelVerifiedCandidates, Mode::Deep) => 7,
        }
    }

    pub(crate) const fn max_candidates(&self, strategy: Strategy) -> u64 {
        match (strategy, self.mode) {
            (Strategy::ParallelVerifiedCandidates, Mode::Deep) => 3,
            (Strategy::ParallelVerifiedCandidates, Mode::Auto) => 2,
            (Strategy::Direct | Strategy::SequentialRevision, _) => 1,
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

/// Request-local vote accounting. Parallel fan-out supplies the winning
/// answer's initial vote count; later revisions extend the bounded vote map.
#[derive(Clone, Debug, Default)]
struct CandidateSummary {
    candidates: usize,
    fanout: bool,
    votes: HashMap<u64, usize>,
}

impl CandidateSummary {
    fn from_response(headers: &reqwest::header::HeaderMap, body: &Value) -> Self {
        let mut summary = Self {
            candidates: 1,
            ..Self::default()
        };
        if let Some(receipt) = headers
            .get("x-blazar-adaptive-candidates")
            .and_then(|value| value.to_str().ok())
        {
            let field = |prefix: &str| {
                receipt
                    .split_whitespace()
                    .find_map(|part| part.strip_prefix(prefix))
            };
            summary.candidates = field("used=")
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(1)
                .clamp(1, 4);
            summary.fanout = summary.candidates > 1;
            let key = field("vote_key")
                .and_then(|value| u64::from_str_radix(value, 16).ok())
                .or_else(|| {
                    crate::reasoning::answer_hash(response_content(body).unwrap_or_default())
                });
            if let Some(tallies) = field("vote_tallies=") {
                for entry in tallies.split(',') {
                    let Some((hash, count)) = entry.split_once(':') else {
                        continue;
                    };
                    let (Some(hash), Some(count)) = (
                        u64::from_str_radix(hash, 16).ok(),
                        count.parse::<usize>().ok(),
                    ) else {
                        continue;
                    };
                    summary.votes.insert(hash, count.min(summary.candidates));
                }
            }
            if summary.votes.is_empty()
                && let Some(key) = key
            {
                let count = field("agreement=")
                    .and_then(|value| value.split('/').next())
                    .and_then(|value| value.parse::<usize>().ok())
                    .unwrap_or(1)
                    .min(summary.candidates);
                summary.votes.insert(key, count);
            }
        } else if let Some(key) =
            crate::reasoning::answer_hash(response_content(body).unwrap_or_default())
        {
            summary.votes.insert(key, 1);
        }
        summary
    }

    fn add_revision(&mut self, body: &Value) {
        let hash = crate::reasoning::answer_hash(&response_content(body).unwrap_or_default());
        if !self.fanout {
            self.candidates = 1;
            self.votes.clear();
            if let Some(hash) = hash {
                self.votes.insert(hash, 1);
            }
            return;
        }
        self.candidates = self.candidates.saturating_add(1);
        if let Some(hash) = hash {
            *self.votes.entry(hash).or_default() += 1;
        }
    }

    fn agreement_for(&self, body: &Value) -> usize {
        crate::reasoning::answer_hash(&response_content(body).unwrap_or_default())
            .and_then(|hash| self.votes.get(&hash).copied())
            .unwrap_or(0)
    }
}

/// Contract-level assessment distinguishes deterministic checks (schema and
/// recognized arithmetic) from answer agreement, which is only a consistency
/// signal and never proof of factual correctness.
#[allow(clippy::struct_excessive_bools)] // Independent response-contract gates.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct CandidateQuality {
    http_success: bool,
    parseable: bool,
    schema_valid: Option<bool>,
    task_valid: Option<bool>,
    complete: bool,
    non_empty: bool,
    agreement: usize,
    candidates: usize,
}

impl CandidateQuality {
    fn consensus_score(self) -> usize {
        self.agreement.saturating_mul(1000) / self.candidates.max(1)
    }

    fn has_majority(self) -> bool {
        self.candidates <= 1
            || self.task_valid == Some(true)
            || self.agreement.saturating_mul(2) > self.candidates
    }

    fn rank(self) -> (u8, u8, u8, u8, u8, usize, u8, u8) {
        let schema_ok = u8::from(self.schema_valid != Some(false));
        let math_ok = u8::from(self.task_valid != Some(false));
        (
            u8::from(self.http_success),
            schema_ok + math_ok,
            schema_ok,
            math_ok,
            u8::from(self.complete),
            self.consensus_score(),
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
            && self.task_valid != Some(false)
            && self.complete
            && self.non_empty
            && self.has_majority()
    }

    fn needs_revision(self) -> bool {
        self.http_success
            && (self.schema_valid == Some(false)
                || self.task_valid == Some(false)
                || !self.complete
                || !self.non_empty
                || !self.has_majority())
    }
}

fn assess_candidate(
    state: &crate::state::AppState,
    plan: &Plan,
    schema: Option<&Value>,
    status: reqwest::StatusCode,
    body: &Value,
    summary: &CandidateSummary,
) -> CandidateQuality {
    let schema_valid = if plan.verify_schema {
        Some(schema_violation(state, plan, schema, body).is_none())
    } else {
        None
    };
    let content = response_content(body).unwrap_or_default();
    let task_valid = plan.task_profile.task_valid(content);
    let finish_reason = body
        .pointer("/choices/0/finish_reason")
        .and_then(Value::as_str);
    let complete = matches!(finish_reason, Some("stop" | "content_filter") | None);
    CandidateQuality {
        http_success: status.is_success(),
        parseable: true,
        schema_valid,
        task_valid,
        complete,
        non_empty: !content.trim().is_empty(),
        agreement: summary.agreement_for(body),
        candidates: summary.candidates.max(1),
    }
}

fn reasoning_receipt(
    receipt: &str,
    plan: &Plan,
    summary: &CandidateSummary,
    body: &Value,
) -> String {
    let judge = if plan.task_profile.math_expected().is_some() {
        "math_exact"
    } else if plan.task_profile.table_label().is_some() {
        "table_lookup_exact"
    } else if plan.verify_schema {
        "json_schema+answer_consensus"
    } else if summary.fanout {
        "answer_consensus"
    } else {
        "response_contract"
    };
    format!(
        "{receipt} judge={judge} agreement={}/{}",
        summary.agreement_for(body),
        summary.candidates.max(1)
    )
}

fn revision_feedback(
    quality: CandidateQuality,
    schema_feedback: Option<&str>,
    task_feedback: Option<&str>,
) -> Option<String> {
    if quality.task_valid == Some(false) {
        return task_feedback.map(str::to_string).or_else(|| {
            Some("The deterministic task verifier rejected the answer. Re-evaluate the original task and return the verified result.".into())
        });
    }
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
    if !quality.has_majority() {
        return Some(
            "Independent candidate answers disagreed. Re-evaluate the original problem, check the key steps, and provide one clearly stated final answer.".into()
        );
    }
    None
}

#[allow(clippy::too_many_arguments)] // Arguments map directly to the compact machine-readable receipt.
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

fn append_search_expansion_prompt(
    request: &mut Value,
    partial_state: &str,
    branch_limit: usize,
) -> Result<(), String> {
    let messages = request
        .get_mut("messages")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| "reasoning search request lost its messages array".to_string())?;
    let state = if partial_state.trim().is_empty() {
        "No partial solution has been generated yet.".to_string()
    } else {
        partial_state.chars().take(8_000).collect()
    };
    messages.push(serde_json::json!({
        "role": "user",
        "content": format!(
            "Continue solving the original task by expanding this partial solution state:\n\n{state}\n\n             Return a JSON object with a branches array containing at most {branch_limit} distinct next states.              Each branch must have: step (a concise, checkable progress summary, not private chain-of-thought),              prior (0..1 estimate of how promising this branch is), and final_answer (a complete final answer string              only when this branch is ready, otherwise null). Explore meaningfully different approaches when possible.              Do not repeat an existing state. If no valid continuation exists, return an empty branches array. JSON only."
        )
    }));
    request["response_format"] = serde_json::json!({"type": "json_object"});
    // Limit each internal expansion response. The aggregate model-call and
    // request deadline budgets remain the authoritative compute ceilings.
    request["max_tokens"] = Value::from(512_u64);
    if let Some(object) = request.as_object_mut() {
        for field in ["max_completion_tokens", "max_output_tokens", "n", "logprobs"] {
            object.remove(field);
        }
    }
    Ok(())
}

fn verify_search_answer(
    state: &crate::state::AppState,
    plan: &Plan,
    schema: Option<&Value>,
    answer: &str,
) -> crate::reasoning_search::Verification {
    let mut applicable = false;
    let mut valid = true;
    if let Some(task_valid) = plan.task_profile.task_valid(answer) {
        applicable = true;
        valid &= task_valid;
    }
    if plan.verify_schema {
        applicable = true;
        let schema_valid = schema
            .and_then(|schema| {
                serde_json::from_str::<Value>(answer)
                    .ok()
                    .map(|value| state.sentinel.schema_check(schema, &value).is_none())
            })
            .unwrap_or(false);
        valid &= schema_valid;
    }
    if !applicable {
        crate::reasoning_search::Verification::Unknown
    } else if valid {
        crate::reasoning_search::Verification::Passed
    } else {
        crate::reasoning_search::Verification::Failed
    }
}

fn parse_search_expansion(
    state: &crate::state::AppState,
    plan: &Plan,
    schema: Option<&Value>,
    value_model: &dyn crate::reasoning_search::ProcessRewardModel,
    parent_state: &str,
    response: &Value,
    branch_limit: usize,
) -> Vec<crate::reasoning_search::SearchProposal> {
    let Some(content) = response_content(response) else {
        return Vec::new();
    };
    let content = content
        .trim()
        .strip_prefix("```json")
        .or_else(|| content.trim().strip_prefix("```"))
        .unwrap_or(content.trim())
        .trim_end_matches('`')
        .trim();
    let Ok(parsed) = serde_json::from_str::<Value>(content) else {
        return Vec::new();
    };
    let Some(branches) = parsed.get("branches").and_then(Value::as_array) else {
        return Vec::new();
    };

    branches
        .iter()
        .take(branch_limit)
        .filter_map(|branch| {
            let step = branch.get("step").and_then(Value::as_str)?.trim();
            if step.is_empty() || step.len() > 4_096 {
                return None;
            }
            let answer = branch
                .get("final_answer")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|answer| !answer.is_empty() && answer.len() <= 16_384)
                .map(str::to_string);
            let mut state_text = String::with_capacity(parent_state.len() + step.len() + 8);
            if !parent_state.is_empty() {
                state_text.push_str(parent_state);
                state_text.push_str("\n");
            }
            state_text.push_str(step);
            if state_text.len() > 16_384 {
                return None;
            }

            let verification = answer.as_deref().map_or(
                crate::reasoning_search::Verification::Unknown,
                |final_answer| verify_search_answer(state, plan, schema, final_answer),
            );
            // Score the partial prefix through the independent value-model
            // interface. The default adapter is explicitly uncalibrated; it
            // never treats the generator's own confidence as a PRM score.
            let prior = branch.get("prior").and_then(Value::as_f64).unwrap_or(0.5);
            let value = value_model.evaluate(&state_text);
            let estimated_state_tokens = step
                .len()
                .saturating_add(answer.as_ref().map_or(0, String::len))
                .div_ceil(4)
                .max(1);
            Some(crate::reasoning_search::SearchProposal {
                state: state_text.clone(),
                answer,
                prior,
                value,
                verification,
                estimated_state_tokens,
            })
        })
        .collect()
}

fn install_search_answer(body: &mut Value, answer: &str) -> bool {
    let Some(message) = body.pointer_mut("/choices/0/message") else {
        return false;
    };
    message["role"] = Value::String("assistant".to_string());
    message["content"] = Value::String(answer.to_string());
    if let Some(object) = message.as_object_mut() {
        object.remove("tool_calls");
        object.remove("function_call");
        object.remove("reasoning_content");
    }
    if let Some(choice) = body.pointer_mut("/choices/0") {
        choice["finish_reason"] = Value::String("stop".to_string());
    }
    true
}

/// Deep mode expands partial reasoning states with budgeted PUCT search.
/// Model-provided value hints are explicitly uncalibrated; only exact task
/// verifiers and requested schemas can mark an answer as independently verified.
#[allow(clippy::too_many_lines, clippy::result_large_err)]
async fn run_local_search_inner(
    state: &std::sync::Arc<crate::state::AppState>,
    engine: &blazar_runtime::EngineRef,
    request: Value,
    mode: Mode,
    headers: &HeaderMap,
    key: Option<&crate::keys::KeyCtx>,
) -> Result<reqwest::Response, axum::response::Response> {
    let plan = Plan::for_request(mode, &request);
    let strategy = plan.strategy(&request);
    let schema = request
        .pointer("/response_format/json_schema/schema")
        .filter(|schema| schema.is_object())
        .cloned();
    let original_request = request.clone();
    let model = request
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or(&engine.name)
        .to_string();
    let max_calls = plan.max_model_calls(strategy);
    let expansion_budget = max_calls.saturating_sub(1).max(1);
    let mut tree = crate::reasoning_search::SearchTree::new(
        "root",
        crate::reasoning_search::SearchConfig {
            max_nodes: 40,
            max_depth: 8,
            max_expansions: expansion_budget,
            max_tokens: expansion_budget.saturating_mul(768),
            branch_factor: 3,
            exploration: 1.6,
            uncertainty_bonus: 0.45,
        },
    );
    let value_model = crate::reasoning_search::UncalibratedValueModel;
    let started = std::time::Instant::now();
    let deadline = std::time::Duration::from_secs(ADAPTIVE_MAX_REQUEST_SECS);
    let mut current: Option<BufferedResponse> = None;
    let mut usage_total = None;
    let mut model_calls = 0usize;
    let mut revisions = 0usize;
    let mut accepted = 0usize;
    let mut rejected = 0usize;
    let mut stop_reason = "search_budget".to_string();

    while model_calls < expansion_budget && started.elapsed() < deadline {
        let Some(leaf) = tree.select_leaf() else {
            stop_reason = "no_expandable_state".to_string();
            break;
        };
        let parent_state = tree.nodes()[leaf].state.clone();
        let branch_limit = tree.current_branch_limit();
        let mut expansion_request = original_request.clone();
        append_search_expansion_prompt(&mut expansion_request, &parent_state, branch_limit)
            .map_err(|error| {
                crate::mcp::teaching_error(axum::http::StatusCode::INTERNAL_SERVER_ERROR, &error)
            })?;
        let response =
            match local_proxy_round(state, engine, &expansion_request, headers, key, None).await {
                Ok(response) => response,
                Err(_error) => {
                    let _ = tree.expand(leaf, Vec::new());
                    rejected = rejected.saturating_add(1);
                    stop_reason = "expansion_error".to_string();
                    continue;
                }
            };
        model_calls = model_calls.saturating_add(1);
        revisions = revisions.saturating_add(1);
        let Ok(expansion_body) = serde_json::from_slice::<Value>(&response.body) else {
            let _ = tree.expand(leaf, Vec::new());
            rejected = rejected.saturating_add(1);
            current = Some(response);
            stop_reason = "unparseable_expansion".to_string();
            continue;
        };
        merge_usage(&mut usage_total, &expansion_body);
        let proposals = parse_search_expansion(
            state,
            &plan,
            schema.as_ref(),
            &value_model,
            &parent_state,
            &expansion_body,
            branch_limit,
        );
        let added = tree.expand(leaf, proposals).unwrap_or_default();
        let added_accepted = added
            .iter()
            .filter(|id| tree.nodes()[**id].status != crate::reasoning_search::NodeStatus::Rejected)
            .count();
        let added_rejected = added.len().saturating_sub(added_accepted);
        accepted = accepted.saturating_add(added_accepted);
        rejected = rejected.saturating_add(added_rejected);
        current = Some(response);

        if tree.stats().verified_terminals > 0 {
            stop_reason = "verified_answer".to_string();
            break;
        }
        if added.is_empty() && tree.select_leaf().is_none() {
            stop_reason = "no_expandable_state".to_string();
            break;
        }
    }

    let best = tree
        .best_answer()
        .map(|(id, answer, verification)| (id, answer.to_string(), verification));
    let (mut current, selected_answer, selected_verification) = if let Some((
        _,
        answer,
        verification,
    )) = best
    {
        let Some(response) = current else {
            return Err(crate::mcp::teaching_error(
                axum::http::StatusCode::BAD_GATEWAY,
                "reasoning search produced an answer without a response envelope",
            ));
        };
        (response, Some(answer), Some(verification))
    } else {
        // Reserve one call for a conventional final answer if search could not
        // produce a terminal state. It is explicitly unverified without an
        // independent verifier; search never returns an internal JSON trace.
        if model_calls < max_calls && started.elapsed() < deadline {
            let response =
                local_proxy_round(state, engine, &original_request, headers, key, None).await?;
            model_calls = model_calls.saturating_add(1);
            if let Ok(body) = serde_json::from_slice::<Value>(&response.body) {
                merge_usage(&mut usage_total, &body);
            }
            stop_reason = "search_fallback".to_string();
            let verification = serde_json::from_slice::<Value>(&response.body)
                .ok()
                .and_then(|body| {
                    response_content(&body)
                        .map(|answer| verify_search_answer(state, &plan, schema.as_ref(), answer))
                })
                .unwrap_or(crate::reasoning_search::Verification::Unknown);
            (response, None, Some(verification))
        } else {
            return Err(crate::mcp::teaching_error(
                axum::http::StatusCode::GATEWAY_TIMEOUT,
                "reasoning search exhausted its compute deadline before producing a user-facing answer",
            ));
        }
    };

    let Ok(mut final_body) = serde_json::from_slice::<Value>(&current.body) else {
        return Ok(rebuild_response(
            current,
            &format!("mode=deep strategy=prefix_puct stop=unparseable_final"),
        ));
    };
    if let Some(answer) = selected_answer.as_deref() {
        if !install_search_answer(&mut final_body, answer) {
            return Err(crate::mcp::teaching_error(
                axum::http::StatusCode::BAD_GATEWAY,
                "reasoning search response was missing the OpenAI chat response envelope",
            ));
        }
    }
    patch_usage(&mut final_body, usage_total);
    current.body = axum::body::Bytes::from(
        serde_json::to_vec(&final_body).unwrap_or_else(|_| current.body.to_vec()),
    );
    let schema_verified = plan
        .verify_schema
        .then(|| schema_violation(state, &plan, schema.as_ref(), &final_body).is_none());
    let stats = tree.stats();
    let verified = selected_verification == Some(crate::reasoning_search::Verification::Passed);
    let search_receipt = format!(
        "algorithm=prefix_puct value_source=uncalibrated_process_reward_fallback prior_source=generator_proposal_heuristic verified={verified} nodes={} expansions={} depth={} estimated_state_tokens={} transpositions={} prm_calibrated_nodes={} best_value={:.3} uncertainty={:.3} stop={stop_reason}",
        stats.nodes,
        stats.expansions,
        stats.max_depth_reached,
        stats.estimated_state_tokens,
        stats.transpositions,
        stats.prm_calibrated_nodes,
        stats.best_value,
        stats.best_uncertainty,
    );
    if let Ok(value) = axum::http::HeaderValue::from_str(&search_receipt) {
        current.headers.insert("x-blazar-adaptive-search", value);
    }
    let receipt = loop_receipt(
        &plan,
        strategy,
        model_calls,
        stats.nodes.saturating_sub(1),
        schema_verified,
        revisions,
        None,
        accepted,
        rejected,
        &stop_reason,
    );
    crate::metadata_card::persist_from_response(
        state,
        &model,
        Some(&original_request),
        &current.body,
    );
    Ok(rebuild_response(
        current,
        &format!("{receipt} {search_receipt}"),
    ))
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
    if mode == Mode::Deep && strategy == Strategy::ParallelVerifiedCandidates {
        return run_local_search_inner(state, engine, request, mode, headers, key).await;
    }
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
        (strategy == Strategy::ParallelVerifiedCandidates).then(|| plan.max_candidates(strategy));
    let mut current =
        local_proxy_round(state, engine, &request, headers, key, asked_candidates).await?;
    let (candidates, mut degraded, mut model_calls) = if let Some(asked) = asked_candidates {
        let (used, reason) = candidate_usage(&current.headers, usize::try_from(asked).unwrap_or(1));
        (used, reason, used)
    } else {
        (1, None, 1)
    };

    let Ok(mut current_body) = serde_json::from_slice::<Value>(&current.body) else {
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
    };
    merge_usage(&mut usage_total, &current_body);
    let mut candidate_summary = CandidateSummary::from_response(&current.headers, &current_body);
    let mut best_quality = assess_candidate(
        state,
        &plan,
        schema.as_ref(),
        current.status,
        &current_body,
        &candidate_summary,
    );
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
            let feedback = revision_feedback(
                best_quality,
                schema_detail.as_deref(),
                plan.task_profile.task_feedback().as_deref(),
            );
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

            let Ok(revised_body) = serde_json::from_slice::<Value>(&revised.body) else {
                rejected += 1;
                stop_reason = "unparseable_revision".into();
                break;
            };
            merge_usage(&mut usage_total, &revised_body);
            candidate_summary.add_revision(&revised_body);
            let revised_quality = assess_candidate(
                state,
                &plan,
                schema.as_ref(),
                revised.status,
                &revised_body,
                &candidate_summary,
            );
            if !revised_quality.is_better_than(best_quality) {
                rejected += 1;
                stop_reason = "no_objective_improvement".into();
                // When candidate voting is still inconclusive, a rejected
                // sample contributes evidence and the incumbent is retained.
                if model_calls < plan.max_model_calls(strategy)
                    && best_quality.needs_revision()
                    && (best_quality.schema_valid != Some(false)
                        || schema_repair_attempts < plan.max_schema_retries)
                {
                    continue;
                }
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
    let receipt = loop_receipt(
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
    );
    let receipt = reasoning_receipt(&receipt, &plan, &candidate_summary, &current_body);
    Ok(rebuild_response(current, &receipt))
}

#[allow(clippy::result_large_err)] // Axum Response is the gateway handler's error currency.
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
    // Never trust provider-supplied Blazar control-plane receipts. The
    // gateway creates its own candidate receipt after judging this response.
    let mut response_headers = parts.headers;
    response_headers.remove("x-blazar-adaptive-candidates");
    response_headers.remove("x-blazar-adaptive-result");
    Ok(BufferedResponse {
        status,
        headers: response_headers,
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

/// Run bounded remote candidates concurrently. Partial transport failures
/// reduce the sample count; a complete failure returns the first upstream error.
#[allow(clippy::too_many_lines, clippy::result_large_err)]
async fn remote_candidate_rounds(
    state: &std::sync::Arc<crate::state::AppState>,
    model: &str,
    method: &axum::http::Method,
    path_query: &str,
    headers: &axum::http::HeaderMap,
    request: &Value,
    want: u64,
) -> Result<BufferedResponse, axum::response::Response> {
    let sends = (0..want).map(|_| remote_round(state, model, method, path_query, headers, request));
    let results = futures::future::join_all(sends).await;
    let mut responses = Vec::new();
    let mut first_error = None;
    for result in results {
        match result {
            Ok(response) => responses.push(response),
            Err(error) => {
                if first_error.is_none() {
                    first_error = Some(error);
                }
            }
        }
    }
    if responses.is_empty() {
        return Err(first_error.unwrap_or_else(|| {
            crate::mcp::teaching_error(
                axum::http::StatusCode::BAD_GATEWAY,
                "all adaptive remote candidates failed",
            )
        }));
    }

    let mut parsed = Vec::with_capacity(responses.len());
    let mut usage_total = None;
    for response in &responses {
        let value = serde_json::from_slice::<Value>(&response.body).unwrap_or(Value::Null);
        merge_usage(&mut usage_total, &value);
        parsed.push((response.status.as_u16(), value));
    }
    let schema = request
        .pointer("/response_format/json_schema/schema")
        .filter(|value| value.is_object());
    let decision = crate::reasoning::pick_candidate(&state.sentinel, &parsed, schema, request);
    let mut winner = responses.swap_remove(decision.winner_index);
    if let Ok(mut body) = serde_json::from_slice::<Value>(&winner.body) {
        patch_usage(&mut body, usage_total);
        winner.body = axum::body::Bytes::from(
            serde_json::to_vec(&body).unwrap_or_else(|_| winner.body.to_vec()),
        );
        let mut receipt = format!(
            "asked={want} used={} judge={} agreement={}/{}",
            parsed.len(),
            decision.verifier,
            decision.agreement,
            decision.candidates
        );
        if let Some(hash) = decision.answer_hash {
            let _ = write!(receipt, " vote_key={hash:016x}");
        }
        let vote_tallies = decision
            .vote_tallies
            .iter()
            .map(|(hash, count)| format!("{hash:016x}:{count}"))
            .collect::<Vec<_>>()
            .join(",");
        if !vote_tallies.is_empty() {
            let _ = write!(receipt, " vote_tallies={vote_tallies}");
        }
        if let Ok(value) = axum::http::HeaderValue::from_str(&receipt) {
            winner.headers.insert("x-blazar-adaptive-candidates", value);
        }
    }
    Ok(winner)
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

#[allow(clippy::result_large_err, clippy::too_many_lines)] // One bounded remote evaluate/revise/accept state machine.
async fn run_remote_search_inner(
    state: &std::sync::Arc<crate::state::AppState>,
    model: &str,
    method: &axum::http::Method,
    path_query: &str,
    headers: &HeaderMap,
    request: Value,
    mode: Mode,
) -> Result<axum::response::Response, axum::response::Response> {
    let plan = Plan::for_request(mode, &request);
    let strategy = plan.strategy(&request);
    let schema = request
        .pointer("/response_format/json_schema/schema")
        .filter(|schema| schema.is_object())
        .cloned();
    let original_request = request.clone();
    let max_calls = plan.max_model_calls(strategy);
    let expansion_budget = max_calls.saturating_sub(1).max(1);
    let mut tree = crate::reasoning_search::SearchTree::new(
        "root",
        crate::reasoning_search::SearchConfig {
            max_nodes: 40,
            max_depth: 8,
            max_expansions: expansion_budget,
            max_tokens: expansion_budget.saturating_mul(768),
            branch_factor: 3,
            exploration: 1.6,
            uncertainty_bonus: 0.45,
        },
    );
    let value_model = crate::reasoning_search::UncalibratedValueModel;
    let started = std::time::Instant::now();
    let deadline = std::time::Duration::from_secs(ADAPTIVE_MAX_REQUEST_SECS);
    let mut current: Option<BufferedResponse> = None;
    let mut usage_total = None;
    let mut model_calls = 0usize;
    let mut revisions = 0usize;
    let mut accepted = 0usize;
    let mut rejected = 0usize;
    let mut stop_reason = "search_budget".to_string();

    while model_calls < expansion_budget && started.elapsed() < deadline {
        let Some(leaf) = tree.select_leaf() else {
            stop_reason = "no_expandable_state".to_string();
            break;
        };
        let parent_state = tree.nodes()[leaf].state.clone();
        let branch_limit = tree.current_branch_limit();
        let mut expansion_request = original_request.clone();
        append_search_expansion_prompt(&mut expansion_request, &parent_state, branch_limit)
            .map_err(|error| {
                crate::mcp::teaching_error(axum::http::StatusCode::INTERNAL_SERVER_ERROR, &error)
            })?;
        let response = match remote_round(
            state,
            model,
            method,
            path_query,
            headers,
            &expansion_request,
        )
        .await
        {
            Ok(response) => response,
            Err(_error) => {
                let _ = tree.expand(leaf, Vec::new());
                rejected = rejected.saturating_add(1);
                stop_reason = "expansion_error".to_string();
                continue;
            }
        };
        model_calls = model_calls.saturating_add(1);
        revisions = revisions.saturating_add(1);
        let Ok(expansion_body) = serde_json::from_slice::<Value>(&response.body) else {
            let _ = tree.expand(leaf, Vec::new());
            rejected = rejected.saturating_add(1);
            current = Some(response);
            stop_reason = "unparseable_expansion".to_string();
            continue;
        };
        merge_usage(&mut usage_total, &expansion_body);
        let proposals = parse_search_expansion(
            state,
            &plan,
            schema.as_ref(),
            &value_model,
            &parent_state,
            &expansion_body,
            branch_limit,
        );
        let added = tree.expand(leaf, proposals).unwrap_or_default();
        let added_accepted = added
            .iter()
            .filter(|id| tree.nodes()[**id].status != crate::reasoning_search::NodeStatus::Rejected)
            .count();
        let added_rejected = added.len().saturating_sub(added_accepted);
        accepted = accepted.saturating_add(added_accepted);
        rejected = rejected.saturating_add(added_rejected);
        current = Some(response);

        if tree.stats().verified_terminals > 0 {
            stop_reason = "verified_answer".to_string();
            break;
        }
        if added.is_empty() && tree.select_leaf().is_none() {
            stop_reason = "no_expandable_state".to_string();
            break;
        }
    }

    let best = tree
        .best_answer()
        .map(|(id, answer, verification)| (id, answer.to_string(), verification));
    let (mut current, selected_answer, selected_verification) = if let Some((
        _,
        answer,
        verification,
    )) = best
    {
        let Some(response) = current else {
            return Err(crate::mcp::teaching_error(
                axum::http::StatusCode::BAD_GATEWAY,
                "reasoning search produced an answer without a response envelope",
            ));
        };
        (response, Some(answer), Some(verification))
    } else {
        // Reserve one call for a conventional final answer if search could not
        // produce a terminal state. It is explicitly unverified without an
        // independent verifier; search never returns an internal JSON trace.
        if model_calls < max_calls && started.elapsed() < deadline {
            let response =
                remote_round(state, model, method, path_query, headers, &original_request).await?;
            model_calls = model_calls.saturating_add(1);
            if let Ok(body) = serde_json::from_slice::<Value>(&response.body) {
                merge_usage(&mut usage_total, &body);
            }
            stop_reason = "search_fallback".to_string();
            let verification = serde_json::from_slice::<Value>(&response.body)
                .ok()
                .and_then(|body| {
                    response_content(&body)
                        .map(|answer| verify_search_answer(state, &plan, schema.as_ref(), answer))
                })
                .unwrap_or(crate::reasoning_search::Verification::Unknown);
            (response, None, Some(verification))
        } else {
            return Err(crate::mcp::teaching_error(
                axum::http::StatusCode::GATEWAY_TIMEOUT,
                "reasoning search exhausted its compute deadline before producing a user-facing answer",
            ));
        }
    };

    let Ok(mut final_body) = serde_json::from_slice::<Value>(&current.body) else {
        return Ok(rebuild_axum_response(
            current,
            &format!("mode=deep strategy=prefix_puct stop=unparseable_final"),
        ));
    };
    if let Some(answer) = selected_answer.as_deref() {
        if !install_search_answer(&mut final_body, answer) {
            return Err(crate::mcp::teaching_error(
                axum::http::StatusCode::BAD_GATEWAY,
                "reasoning search response was missing the OpenAI chat response envelope",
            ));
        }
    }
    patch_usage(&mut final_body, usage_total);
    current.body = axum::body::Bytes::from(
        serde_json::to_vec(&final_body).unwrap_or_else(|_| current.body.to_vec()),
    );
    let schema_verified = plan
        .verify_schema
        .then(|| schema_violation(state, &plan, schema.as_ref(), &final_body).is_none());
    let stats = tree.stats();
    let verified = selected_verification == Some(crate::reasoning_search::Verification::Passed);
    let search_receipt = format!(
        "algorithm=prefix_puct value_source=uncalibrated_process_reward_fallback prior_source=generator_proposal_heuristic verified={verified} nodes={} expansions={} depth={} estimated_state_tokens={} transpositions={} prm_calibrated_nodes={} best_value={:.3} uncertainty={:.3} stop={stop_reason}",
        stats.nodes,
        stats.expansions,
        stats.max_depth_reached,
        stats.estimated_state_tokens,
        stats.transpositions,
        stats.prm_calibrated_nodes,
        stats.best_value,
        stats.best_uncertainty,
    );
    if let Ok(value) = axum::http::HeaderValue::from_str(&search_receipt) {
        current.headers.insert("x-blazar-adaptive-search", value);
    }
    let receipt = loop_receipt(
        &plan,
        strategy,
        model_calls,
        stats.nodes.saturating_sub(1),
        schema_verified,
        revisions,
        None,
        accepted,
        rejected,
        &stop_reason,
    );
    Ok(rebuild_axum_response(
        current,
        &format!("{receipt} {search_receipt}"),
    ))
}

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
    let strategy = plan.strategy(&request);
    if mode == Mode::Deep && strategy == Strategy::ParallelVerifiedCandidates {
        return run_remote_search_inner(state, model, method, path_query, headers, request, mode)
            .await;
    }
    let schema = request
        .pointer("/response_format/json_schema/schema")
        .filter(|schema| schema.is_object())
        .cloned();
    let model_name = request
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or(model)
        .to_string();
    let mut model_calls;
    let mut revisions = 0usize;
    let mut accepted = 0usize;
    let mut rejected = 0usize;
    let mut schema_repair_attempts = 0usize;
    let mut usage_total = None;
    let mut stop_reason: String;

    let asked_candidates =
        (strategy == Strategy::ParallelVerifiedCandidates).then(|| plan.max_candidates(strategy));
    let mut current = if let Some(want) = asked_candidates {
        remote_candidate_rounds(state, model, method, path_query, headers, &request, want).await?
    } else {
        remote_round(state, model, method, path_query, headers, &request).await?
    };
    model_calls = current
        .headers
        .get("x-blazar-adaptive-candidates")
        .and_then(|value| value.to_str().ok())
        .and_then(|receipt| {
            receipt
                .split_whitespace()
                .find_map(|part| part.strip_prefix("used="))
        })
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1);
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
    let Ok(mut current_body) = serde_json::from_slice::<Value>(&current.body) else {
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
    };
    merge_usage(&mut usage_total, &current_body);
    let mut candidate_summary = CandidateSummary::from_response(&current.headers, &current_body);
    let mut best_quality = assess_candidate(
        state,
        &plan,
        schema.as_ref(),
        current.status,
        &current_body,
        &candidate_summary,
    );
    let mut schema_verified = best_quality.schema_valid;

    if message_has_tool_calls(&current_body) {
        stop_reason = "tool_handoff".into();
    } else {
        loop {
            let schema_detail = if best_quality.schema_valid == Some(false) {
                schema_violation(state, &plan, schema.as_ref(), &current_body)
            } else {
                None
            };
            let feedback = revision_feedback(
                best_quality,
                schema_detail.as_deref(),
                plan.task_profile.task_feedback().as_deref(),
            );
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
                Err(_error) => {
                    stop_reason = "revision_error".into();
                    tracing::warn!(model = %model_name, "adaptive remote revision failed; retaining the best prior candidate");
                    break;
                }
            };
            model_calls += 1;
            revisions += 1;
            let Ok(revised_body) = serde_json::from_slice::<Value>(&revised.body) else {
                rejected += 1;
                stop_reason = "unparseable_revision".into();
                break;
            };
            merge_usage(&mut usage_total, &revised_body);
            candidate_summary.add_revision(&revised_body);
            let revised_quality = assess_candidate(
                state,
                &plan,
                schema.as_ref(),
                revised.status,
                &revised_body,
                &candidate_summary,
            );
            if !revised_quality.is_better_than(best_quality) {
                rejected += 1;
                stop_reason = "no_objective_improvement".into();
                if model_calls < plan.max_model_calls(strategy)
                    && best_quality.needs_revision()
                    && (best_quality.schema_valid != Some(false)
                        || schema_repair_attempts < plan.max_schema_retries)
                {
                    continue;
                }
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
    let receipt = loop_receipt(
        &plan,
        strategy,
        model_calls,
        candidate_summary.candidates,
        schema_verified,
        revisions,
        None,
        accepted,
        rejected,
        &stop_reason,
    );
    let receipt = reasoning_receipt(&receipt, &plan, &candidate_summary, &current_body);
    Ok(rebuild_axum_response(current, &receipt))
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
        assert!(plan.max_schema_retries <= MAX_DEEP_SCHEMA_RETRIES);
    }

    #[test]
    fn unit__auto_math__gets_a_bounded_verifier_repair() {
        let request = json!({
            "messages": [{
                "role": "user",
                "content": "Calculate 17 * (23 - 8) + 144 / 12. Return only the number."
            }]
        });
        let plan = Plan::for_request(Mode::Auto, &request);
        assert_eq!(plan.task_profile.math_expected(), Some(267.0));
        assert_eq!(plan.strategy(&request), Strategy::SequentialRevision);
        assert_eq!(plan.max_model_calls(Strategy::SequentialRevision), 2);
    }

    #[test]
    fn unit__auto_table__gets_a_bounded_verifier_repair() {
        let request = json!({
            "messages": [{
                "role": "user",
                "content": "Oak=12, Pine=19, Birch=26. Return the highest label and value."
            }]
        });
        let plan = Plan::for_request(Mode::Auto, &request);
        assert_eq!(plan.task_profile.table_label(), Some("Birch"));
        assert_eq!(plan.strategy(&request), Strategy::SequentialRevision);
        assert_eq!(plan.max_model_calls(Strategy::SequentialRevision), 2);
    }

    #[test]
    fn unit__candidate_quality__schema_validity_beats_completion_only() {
        let invalid_but_complete = CandidateQuality {
            http_success: true,
            parseable: true,
            schema_valid: Some(false),
            complete: true,
            non_empty: true,
            ..CandidateQuality::default()
        };
        let valid_but_truncated = CandidateQuality {
            http_success: true,
            parseable: true,
            schema_valid: Some(true),
            complete: false,
            non_empty: true,
            ..CandidateQuality::default()
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
            ..CandidateQuality::default()
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
            ..CandidateQuality::default()
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
            ..CandidateQuality::default()
        };
        let failed = CandidateQuality {
            http_success: false,
            ..empty
        };
        assert!(empty.needs_revision());
        assert!(!failed.needs_revision());
    }

    #[test]
    fn unit__candidate_quality__exact_math_overrides_wrong_answer_consensus() {
        let wrong_consensus = CandidateQuality {
            http_success: true,
            parseable: true,
            schema_valid: None,
            task_valid: Some(false),
            complete: true,
            non_empty: true,
            agreement: 3,
            candidates: 3,
        };
        let verified_answer = CandidateQuality {
            task_valid: Some(true),
            agreement: 1,
            ..wrong_consensus
        };
        assert!(verified_answer.is_better_than(wrong_consensus));
        assert!(verified_answer.is_satisfied());
    }

    #[test]
    fn unit__candidate_summary__revisions_accumulate_votes() {
        let first = json!({"choices":[{"message":{"content":"Paris"},"finish_reason":"stop"}]});
        let mut summary = CandidateSummary {
            candidates: 3,
            fanout: true,
            votes: HashMap::from([(crate::reasoning::answer_hash("Paris").unwrap(), 1)]),
        };
        summary.add_revision(&first);
        assert_eq!(summary.candidates, 4);
        assert_eq!(summary.agreement_for(&first), 2);
        summary.add_revision(&first);
        assert_eq!(summary.candidates, 5);
        assert_eq!(summary.agreement_for(&first), 3);
        assert!(summary.agreement_for(&first).saturating_mul(2) > summary.candidates);
    }

    #[test]
    fn unit__deep_mode__parallelizes_general_chat_but_respects_seed() {
        let ordinary = json!({
            "messages": [{"role": "user", "content": "Explain the difference between a mutex and a semaphore."}]
        });
        assert_eq!(
            Plan::for_request(Mode::Deep, &ordinary).strategy(&ordinary),
            Strategy::ParallelVerifiedCandidates
        );
        let seeded = json!({
            "seed": 7,
            "messages": [{"role": "user", "content": "Explain the difference between a mutex and a semaphore."}]
        });
        assert_eq!(
            Plan::for_request(Mode::Deep, &seeded).strategy(&seeded),
            Strategy::SequentialRevision
        );
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
        assert_eq!(plan.max_schema_retries, MAX_DEEP_SCHEMA_RETRIES);
        assert_eq!(plan.max_model_calls(Strategy::SequentialRevision), 4);
        assert_eq!(
            plan.max_model_calls(Strategy::ParallelVerifiedCandidates),
            7
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
