//! Sentinel: warn-only response-semantics observation (the semantic
//! reliability layer). The gateway proves the pipe works; the sentinel
//! checks whether the *semantic operation* succeeded — truncation,
//! tool-call validity, schema violations, empty replies, stalls — and
//! surfaces each with a fix hint. It never alters bytes: request and
//! response bodies pass through untouched, observation happens on a
//! side-channel (bounded, drop-with-degraded-flag), so it can neither
//! backpressure nor fail inference. Kill-switch: `sentinel = false`.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;
use std::time::Duration;

use serde_json::Value;
use tokio::sync::mpsc;

use crate::translate as tr;

const RING_CAP: usize = 256;
const CHANNEL_CAP: usize = 64;
/// Accumulation caps: beyond these the record is marked degraded and the
/// analyzer stops accumulating (memory is bounded by construction).
const MAX_ACCUM_BYTES: usize = 4 << 20;
const MAX_CARRY_BYTES: usize = 1 << 20;
/// Crude bound for compiled schema validators: full -> clear (diagnostics
/// only; re-compiling a hot schema is off the request path).
const SCHEMA_CACHE_CAP: usize = 32;
/// Enforce path: non-stream bodies larger than this pass through
/// unenforced (loudly) — buffering unbounded bodies is never worth it.
pub const ENFORCE_BODY_CAP: usize = 4 << 20;
/// Violations `sentinel_enforce` may hard-fail (422). Deliberately
/// excludes the empty-response class: a weird-but-valid completion is
/// not a protocol violation.
const HARD_CODES: &[Code] = &[
    Code::ToolArgsInvalidJson,
    Code::ToolNameUnknown,
    Code::SchemaViolation,
];
/// Persistence: JSONL cap; on crossing, rewrite keeping the newest rows.
const PERSIST_CAP_BYTES: u64 = 1 << 20;
const PERSIST_KEEP: usize = 128;
/// `prompt_tokens` above this fraction of the serving ctx warns
/// (integer form: p * 10 > ctx * `NEAR_LIMIT_TENTHS`).
const NEAR_LIMIT_TENTHS: u64 = 9;

/// Detection vocabulary. `as_str` is the machine code (response headers,
/// `/api/why` consumers); `hint` is the human fix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Code {
    CtxTruncated,
    CtxNearLimit,
    ToolArgsInvalidJson,
    ToolNameUnknown,
    SchemaViolation,
    EmptyResponse,
    ReasoningNoAnswer,
    StalledStream,
    TemplateNoTools,
}

impl Code {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CtxTruncated => "ctx_truncated",
            Self::CtxNearLimit => "ctx_near_limit",
            Self::ToolArgsInvalidJson => "tool_args_invalid_json",
            Self::ToolNameUnknown => "tool_name_unknown",
            Self::SchemaViolation => "schema_violation",
            Self::EmptyResponse => "empty_response",
            Self::ReasoningNoAnswer => "reasoning_no_answer",
            Self::StalledStream => "stalled_stream",
            Self::TemplateNoTools => "template_no_tools",
        }
    }

    #[must_use]
    pub fn hint(self) -> &'static str {
        match self {
            Self::CtxTruncated => "generation hit the context ceiling: raise default_ctx / [model_overrides].ctx, or pass options.num_ctx (ollama API)",
            Self::CtxNearLimit => "prompt is near the serving context; the next requests will truncate",
            Self::ToolArgsInvalidJson => "model emitted tool-call arguments that are not valid JSON",
            Self::ToolNameUnknown => "model called a tool absent from the request's tools list (hallucinated tool)",
            Self::SchemaViolation => "response violated the requested JSON schema / json_object format",
            Self::EmptyResponse => "200 with no content, no tool calls, no reasoning",
            Self::ReasoningNoAnswer => "model produced reasoning but never answered",
            Self::StalledStream => "stream alive but no chunks for the stall threshold (swap thrash / CPU fallback?)",
            Self::TemplateNoTools => "request carries tools but the model's chat template has no tool support — expect plain text instead of tool calls",
        }
    }

    /// The exact retry policy per code — what an agent loop should DO
    /// next, not just what went wrong.
    #[must_use]
    pub fn retry_hint(self) -> &'static str {
        match self {
            Self::CtxTruncated => "retry: raise ctx (X-Pallama-Num-Ctx / options.num_ctx / [model_overrides].ctx) and resend the same request",
            Self::CtxNearLimit => "retry: trim history or raise ctx before sending the next request",
            Self::ToolArgsInvalidJson => "retry: one repair round-trip — resend with the parse error quoted back to the model",
            Self::ToolNameUnknown => "retry: one repair round-trip — name the valid tools in the repair prompt",
            Self::SchemaViolation => "retry: one repair round-trip — include the schema error and the schema in the prompt",
            Self::EmptyResponse => "retry: resend once; persistent empties point at ctx/template config — `pallama doctor`",
            Self::ReasoningNoAnswer => "retry: resend (reasoning consumed the budget); persistent -> raise max tokens or switch reasoning_format",
            Self::StalledStream => "retry after checking `pallama ps` (swap thrash / CPU fallback); the request may still complete late",
            Self::TemplateNoTools => "no retry will help: pull a tool-capable model (template with tool markers) for tool work",
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Detection {
    pub code: Code,
    pub detail: String,
}

/// One observed request: what the model returned and what was wrong with
/// it. Bounded in-memory ring + bounded JSONL under `run/` (persistence:
/// `pallama why` survives daemon restarts). Metadata only — never content.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SentinelRecord {
    pub trace: String,
    pub ts: u64,
    pub route: String,
    pub model: String,
    pub status: u16,
    pub stream: bool,
    /// Engine `finish_reason` (stop / length / `tool_calls` …). `length` on a
    /// small completion = client budget cap; near ctx = real truncation —
    /// one glance separates the two in `why`.
    pub finish: Option<String>,
    pub detections: Vec<Detection>,
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub ctx: Option<u32>,
    pub degraded: bool,
    pub ms: u128,
    /// Response confidence (R2): mean/min token logprob when the client
    /// requested logprobs. None = not requested (display skips).
    pub logprob_mean: Option<f64>,
    pub logprob_min: Option<f64>,
    pub logprob_tokens: Option<u64>,
}

impl SentinelRecord {
    #[must_use]
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "trace": self.trace,
            "ts": self.ts,
            "route": self.route,
            "model": self.model,
            "status": self.status,
            "stream": self.stream,
            "finish": self.finish,
            "detections": self.detections.iter().map(|d| serde_json::json!({
                "code": d.code.as_str(),
                "detail": d.detail,
                "hint": d.code.hint(),
                "retry": d.code.retry_hint(),
            })).collect::<Vec<_>>(),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "ctx": self.ctx,
            "degraded": self.degraded,
            "ms": self.ms,
            "logprob_mean": self.logprob_mean,
            "logprob_min": self.logprob_min,
            "logprob_tokens": self.logprob_tokens,
        })
    }
}

/// Verdict of the model-template capability heuristic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemplateSupport {
    Tools,
    NoTools,
    Missing,
}

/// Request context the analyzer needs, parsed off the hot path.
#[derive(Debug, Clone, Default)]
pub struct RequestCtx {
    pub trace: String,
    pub route: String,
    pub model: String,
    pub tool_names: Vec<String>,
    pub tool_schemas: HashMap<String, Value>,
    pub response_format: Option<Value>,
    pub stream: bool,
    /// Serving ctx from the live instance row (unknown = skip ctx checks).
    pub ctx: Option<u32>,
    pub template: Option<TemplateSupport>,
}

/// Parse tool names/schemas + `response_format` + stream flag from an
/// OpenAI-shaped request body. Pure; shared by both API paths.
#[must_use]
pub fn parse_request_ctx(
    body: &[u8],
) -> (Vec<String>, HashMap<String, Value>, Option<Value>, bool) {
    let Ok(v) = serde_json::from_slice::<Value>(body) else {
        return (Vec::new(), HashMap::new(), None, false);
    };
    let mut names = Vec::new();
    let mut schemas = HashMap::new();
    if let Some(tools) = v.get("tools").and_then(Value::as_array) {
        for t in tools {
            if let Some(name) = t["function"]["name"].as_str() {
                names.push(name.to_string());
                if let Some(p) = t["function"].get("parameters").filter(|p| p.is_object()) {
                    schemas.insert(name.to_string(), p.clone());
                }
            }
        }
    }
    let rf = v.get("response_format").filter(|r| r.is_object()).cloned();
    let stream = v.get("stream").and_then(Value::as_bool).unwrap_or(false);
    (names, schemas, rf, stream)
}

/// Strict tool-definition lint (`OpenAI` strict subset): a `strict: true`
/// function def must be an object schema with `additionalProperties:
/// false` and EVERY property listed in `required` — and every declared
/// schema (strict or not) must compile. Returns the first teaching
/// error; None = clean or no tools. This is request-side: catch a broken
/// def BEFORE the model wastes a turn producing calls nothing accepts.
#[must_use]
pub fn strict_tool_def_error(body: &Value) -> Option<String> {
    let tools = body.get("tools").and_then(Value::as_array)?;
    for (i, t) in tools.iter().enumerate() {
        let f = t.get("function").unwrap_or(t);
        let Some(name) = f.get("name").and_then(Value::as_str) else {
            continue;
        };
        let params = f.get("parameters").cloned().unwrap_or(Value::Null);
        if !params.is_object() {
            if f.get("strict").and_then(Value::as_bool) == Some(true) {
                return Some(format!(
                    "tools[{i}] {name:?}: strict:true requires a parameters object schema"
                ));
            }
            continue;
        }
        // Any declared schema must compile.
        if let Err(e) = jsonschema::validator_for(&params) {
            return Some(format!(
                "tools[{i}] {name:?}: parameters is not a valid JSON Schema: {e}"
            ));
        }
        if f.get("strict").and_then(Value::as_bool) == Some(true) {
            let ap_false =
                params.get("additionalProperties").and_then(Value::as_bool) == Some(false);
            if !ap_false {
                return Some(format!(
                    "tools[{i}] {name:?}: strict:true requires additionalProperties = false"
                ));
            }
            let props = params.get("properties").and_then(Value::as_object);
            let required = params.get("required").and_then(Value::as_array);
            if let Some(props) = props {
                for key in props.keys() {
                    let listed = required
                        .is_some_and(|r| r.iter().any(|v| v.as_str() == Some(key.as_str())));
                    if !listed {
                        return Some(format!(
                            "tools[{i}] {name:?}: strict:true requires every property in required (missing {key:?})"
                        ));
                    }
                }
            }
        }
    }
    None
}

/// Raw GBNF grammar size cap the gateway will forward (256 KiB — far
/// beyond any sane grammar, small enough to bound request memory).
pub const MAX_GRAMMAR_BYTES: usize = 256 * 1024;

/// Structured-output request lint (R9): top-level sanity for ollama
/// `format`, raw `grammar`, and a client-sent `response_format` BEFORE
/// admission — a malformed schema or an over-sized grammar fails fast
/// with a teaching error instead of loading a model to die at the child
/// (or being silently ignored). Deliberately TOP-LEVEL ONLY: no
/// recursion, no `$ref` resolution — anything the child's converter
/// accepts deeper in the schema passes untouched.
#[must_use]
pub fn structured_output_error(body: &Value) -> Option<String> {
    // Raw GBNF grammar: forwarded verbatim downstream — type and cap.
    let mut grammar_set = false;
    if let Some(g) = body.get("grammar").filter(|v| !v.is_null()) {
        match g.as_str() {
            Some("") => {} // empty = absent
            Some(s) => {
                if s.len() > MAX_GRAMMAR_BYTES {
                    return Some(format!(
                        "grammar is {} bytes; the gateway cap is {MAX_GRAMMAR_BYTES} bytes",
                        s.len()
                    ));
                }
                grammar_set = true;
            }
            None => return Some("grammar must be a string (GBNF)".into()),
        }
    }
    // ollama `format`: absent/null = unconstrained, "json" = JSON mode,
    // object = structured schema. Anything else was silently ignored
    // before — now it fails fast.
    if let Some(fmt) = body.get("format").filter(|v| !v.is_null()) {
        match fmt {
            Value::String(s) if s == "json" => {}
            Value::String(other) => {
                return Some(format!(
                    "format must be \"json\" or a JSON schema object, got {other:?}"
                ));
            }
            Value::Object(_) => {
                // Mirrors the child's json_schema+grammar mutual exclusion.
                if grammar_set {
                    return Some(
                        "cannot use both format (schema) and grammar — the engine rejects the pair"
                            .into(),
                    );
                }
                if let Some(e) = schema_shape_error(fmt) {
                    return Some(e);
                }
            }
            _ => return Some("format must be \"json\" or a JSON schema object".into()),
        }
    }
    // Client-sent `response_format` (OpenAI shape) is NOT consumed on the
    // ollama lane — but if present and malformed, say so instead of
    // silently dropping it. Mirrors the child's accepted type set
    // (server-common.cpp:1168-1180: json_object / json_schema / empty).
    if let Some(rf) = body.get("response_format").filter(|v| !v.is_null()) {
        let Some(obj) = rf.as_object() else {
            return Some("response_format must be an object".into());
        };
        if let Some(t) = obj.get("type") {
            let ok = match t {
                Value::String(s) => s.is_empty() || s == "json_object" || s == "json_schema",
                _ => false,
            };
            if !ok {
                return Some(
                    "response_format.type must be \"json_object\" or \"json_schema\"".into(),
                );
            }
        }
        if let Some(js) = obj.get("json_schema").filter(|v| !v.is_null()) {
            if !js.is_object() {
                return Some("response_format.json_schema must be an object".into());
            }
            if let Some(sch) = js.get("schema").filter(|v| !v.is_null()) {
                if !sch.is_object() {
                    return Some("response_format.json_schema.schema must be an object".into());
                }
            }
        }
    }
    None
}

/// Top-level shape sanity for a schema object carried in ollama `format`.
fn schema_shape_error(schema: &Value) -> Option<String> {
    if let Some(p) = schema.get("properties").filter(|v| !v.is_null()) {
        if !p.is_object() {
            return Some("format.properties must be an object".into());
        }
    }
    if let Some(r) = schema.get("required").filter(|v| !v.is_null()) {
        let all_strings = r.as_array().is_some_and(|a| a.iter().all(Value::is_string));
        if !all_strings {
            return Some("format.required must be an array of strings".into());
        }
    }
    if let Some(t) = schema.get("type").filter(|v| !v.is_null()) {
        let ok = match t {
            Value::String(_) => true,
            Value::Array(a) => !a.is_empty() && a.iter().all(Value::is_string),
            _ => false,
        };
        if !ok {
            return Some("format.type must be a string or an array of strings".into());
        }
    }
    None
}

/// Single-flight hash: FNV-1a over model + non-stream body. Stream
/// flag is stripped first so stream/non-stream twins do not collide.
#[must_use]
pub fn singleflight_key(model: &str, body: &[u8], stream: bool) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    // Hash model + body, then fold the stream bit apart.
    for b in model.as_bytes().iter().chain(body) {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    if stream {
        h ^= 0xffff;
    }
    h
}

/// Marker-scan heuristic for tool support in a chat template.
/// Lowercase substring: catches `{%- if tools %}`, `<tool_call>`, hermes
/// and qwen styles. False negatives merely suppress the precheck (safe
/// direction); warn-only means a stray false positive costs one log line.
#[must_use]
pub fn template_has_tools(template: &str) -> bool {
    let lower = template.to_lowercase();
    lower.contains("tools") || lower.contains("tool_call") || lower.contains("<tool")
}

enum FeedEvent {
    /// Raw response bytes (SSE or JSON body; grammar fixed at `begin`).
    Bytes(Vec<u8>),
    /// Pre-parsed OpenAI-shaped chunk/response (ollama-compat path).
    Value(Value),
    End,
}

/// Hot-path handle: clone bytes + `try_send`, nothing else. Any send
/// failure (channel full, analyzer gone) only marks the record degraded.
/// `Drop` sends `End`, covering both clean drain and client abort.
pub struct SentinelFeed {
    tx: Option<mpsc::Sender<FeedEvent>>,
    degraded: Arc<AtomicBool>,
}

impl SentinelFeed {
    pub fn bytes(&self, b: &[u8]) {
        if let Some(tx) = &self.tx {
            if tx.try_send(FeedEvent::Bytes(b.to_vec())).is_err() {
                self.degraded.store(true, Ordering::Relaxed);
            }
        }
    }

    pub fn value(&self, v: Value) {
        if let Some(tx) = &self.tx {
            if tx.try_send(FeedEvent::Value(v)).is_err() {
                self.degraded.store(true, Ordering::Relaxed);
            }
        }
    }

    pub fn end(&self) {
        if let Some(tx) = &self.tx {
            let _ = tx.try_send(FeedEvent::End);
        }
    }

    /// A feed that observes nothing (sentinel off / unobserved route).
    #[must_use]
    pub fn inert() -> Self {
        Self {
            tx: None,
            degraded: Arc::new(AtomicBool::new(false)),
        }
    }

    #[must_use]
    pub fn is_observing(&self) -> bool {
        self.tx.is_some()
    }
}

impl Drop for SentinelFeed {
    fn drop(&mut self) {
        self.end();
    }
}

/// Build the request context (parse, precheck, serving-ctx capture) for
/// BOTH API paths. Returns the context plus precheck warning codes (known
/// before inference, so callers can surface them as response headers).
#[must_use]
pub fn request_ctx(
    state: &crate::state::AppState,
    route: &'static str,
    model: &str,
    body: &[u8],
    trace: Option<String>,
    sse: bool,
) -> (RequestCtx, Vec<&'static str>) {
    let (tool_names, tool_schemas, response_format, stream) = parse_request_ctx(body);
    // Precheck: tools on a template without tool support — the
    // plain-text-instead-of-tool-calls root cause, known before inference.
    let template = if tool_names.is_empty() {
        None
    } else {
        state
            .with_store(|s| s.get_model(model).ok().flatten())
            .flatten()
            .and_then(|row| {
                state
                    .sentinel
                    .template_support(model, std::path::Path::new(&row.path))
            })
    };
    let mut warnings = Vec::new();
    if matches!(
        template,
        Some(TemplateSupport::NoTools | TemplateSupport::Missing)
    ) {
        warnings.push(Code::TemplateNoTools.as_str());
    }
    let ctx = RequestCtx {
        trace: trace.unwrap_or_default(),
        route: route.to_string(),
        model: model.to_string(),
        tool_names,
        tool_schemas,
        response_format,
        stream: stream || sse,
        ctx: state
            .sup
            .ps()
            .into_iter()
            .find(|p| p.name == model)
            .map(|p| p.ctx),
        template,
    };
    (ctx, warnings)
}

/// Shared entry for both API paths: build the context and begin
/// observation. Returns the hot-path feed plus precheck warning codes.
#[must_use]
pub fn begin_chat_observation(
    state: &crate::state::AppState,
    route: &'static str,
    model: &str,
    body: &[u8],
    trace: Option<String>,
    status: u16,
    sse: bool,
) -> (SentinelFeed, Vec<&'static str>) {
    if !state.config.sentinel {
        return (SentinelFeed::inert(), Vec::new());
    }
    let (ctx, warnings) = request_ctx(state, route, model, body, trace, sse);
    (state.sentinel.begin(ctx, status, sse), warnings)
}

/// Open append handle + bytes written (rotation accounting).
struct PersistState {
    file: std::fs::File,
    written: u64,
}

/// D6: bounded LRU caching verdicts of the pure request-side lints —
/// the same schema/grammar bytes always produce the same verdict, and
/// `jsonschema::validator_for` (the expensive part) should run once per
/// distinct schema, not once per request. No TTL: the inputs are pure.
struct VerdictCache {
    order: VecDeque<u64>,
    map: HashMap<u64, Option<String>>,
    hits: u64,
    misses: u64,
}

const VERDICT_CAP: usize = 512;

impl Default for VerdictCache {
    fn default() -> Self {
        Self {
            order: VecDeque::with_capacity(VERDICT_CAP),
            map: HashMap::with_capacity(VERDICT_CAP),
            hits: 0,
            misses: 0,
        }
    }
}

impl VerdictCache {
    fn get(&mut self, k: u64) -> Option<&Option<String>> {
        if self.map.contains_key(&k) {
            if let Some(pos) = self.order.iter().position(|&x| x == k) {
                self.order.remove(pos);
                self.order.push_back(k);
            }
            self.hits += 1;
            self.map.get(&k)
        } else {
            self.misses += 1;
            None
        }
    }

    fn put(&mut self, k: u64, v: Option<String>) {
        if self.map.len() >= VERDICT_CAP && !self.map.contains_key(&k) {
            if let Some(old) = self.order.pop_front() {
                self.map.remove(&old);
            }
        }
        if !self.map.contains_key(&k) {
            self.order.push_back(k);
        }
        self.map.insert(k, v);
    }
}

/// FNV-1a over the compact serialization of exactly the fields a lint
/// reads — two bodies identical in those fields share a verdict.
fn verdict_key(tag: u64, fields: &[&str], body: &Value) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ tag;
    for f in fields {
        for b in f.as_bytes() {
            h ^= u64::from(*b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        let v = body.get(*f).unwrap_or(&Value::Null);
        match serde_json::to_vec(v) {
            Ok(bytes) => {
                for b in bytes {
                    h ^= u64::from(b);
                    h = h.wrapping_mul(0x0000_0100_0000_01b3);
                }
            }
            Err(_) => h ^= 0xdead_beef, // non-serializable = unique
        }
    }
    h
}

/// The observation layer: ring + bounded caches + per-request analyzers.
pub struct Sentinel {
    enabled: bool,
    stall: Duration,
    ring: Mutex<VecDeque<SentinelRecord>>,
    schemas: Mutex<HashMap<String, Option<Arc<jsonschema::Validator>>>>,
    templates: Mutex<HashMap<String, TemplateSupport>>,
    /// D6: lint verdicts (see [`VerdictCache`]).
    verdicts: Mutex<VerdictCache>,
    /// Bounded JSONL at `<run_dir>/sentinel.jsonl`; `None` = memory-only
    /// (no run dir, or open failed — IO problems never take the daemon).
    persist: Option<Mutex<PersistState>>,
    persist_path: Option<std::path::PathBuf>,
    /// Live tail for `pallama watch` / `GET /api/watch`: every committed
    /// record is broadcast; slow consumers lag (resync note), history
    /// stays in the ring for `why`.
    watch_tx: tokio::sync::broadcast::Sender<SentinelRecord>,
    /// J5: stall-triggered evictions (None until the owner wires it).
    evict_hook: Mutex<Option<EvictHook>>,
}

/// Wrapper so the struct stays Clone-cheap and Debug-clean.
#[derive(Clone)]
struct EvictHook(tokio::sync::mpsc::UnboundedSender<String>);

impl Sentinel {
    /// J5 hook: where stalled-model evictions are requested. The owner
    /// (`AppState::new`) installs a channel consumed by a task calling the
    /// supervisor; the analyzer never awaits evicts itself.
    pub fn set_evict_channel(&self, tx: tokio::sync::mpsc::UnboundedSender<String>) {
        *self.evict_hook.lock() = Some(EvictHook(tx));
    }

    pub fn new(enabled: bool, stall_secs: u64, run_dir: Option<&std::path::Path>) -> Arc<Self> {
        let mut ring = VecDeque::with_capacity(RING_CAP);
        let mut persist = None;
        let mut persist_path = None;
        if let Some(dir) = run_dir {
            let path = dir.join("sentinel.jsonl");
            // Load history (best effort): corrupt lines are skipped, never
            // fatal — one bad line must not erase the rest.
            if let Ok(raw) = std::fs::read_to_string(&path) {
                for line in raw.lines().filter(|l| !l.trim().is_empty()) {
                    if let Ok(r) = serde_json::from_str::<SentinelRecord>(line) {
                        if ring.len() == RING_CAP {
                            ring.pop_front();
                        }
                        ring.push_back(r);
                    } else {
                        tracing::debug!(
                            target: "pallama::sentinel",
                            "skip corrupt line in {}", path.display()
                        );
                    }
                }
            }
            if let Ok(file) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
            {
                let written = std::fs::metadata(&path).map_or(0, |m| m.len());
                persist = Some(Mutex::new(PersistState { file, written }));
                persist_path = Some(path);
            }
        }
        let (watch_tx, _) = tokio::sync::broadcast::channel(128);
        Arc::new(Self {
            enabled,
            stall: Duration::from_secs(stall_secs),
            ring: Mutex::new(ring),
            schemas: Mutex::new(HashMap::new()),
            templates: Mutex::new(HashMap::new()),
            verdicts: Mutex::new(VerdictCache::default()),
            persist,
            persist_path,
            watch_tx,
            evict_hook: Mutex::new(None),
        })
    }

    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// D6: cached structured-output lint — identical
    /// (`grammar`, `format`, `response_format`) triples hit the verdict LRU.
    #[must_use]
    pub fn structured_output_error_cached(&self, body: &Value) -> Option<String> {
        let key = verdict_key(1, &["grammar", "format", "response_format"], body);
        if let Some(hit) = self.verdicts.lock().get(key) {
            return hit.clone();
        }
        let verdict = structured_output_error(body);
        self.verdicts.lock().put(key, verdict.clone());
        verdict
    }

    /// D6: cached strict-tool-def lint — identical `tools` arrays hit
    /// the verdict LRU (the `jsonschema::validator_for` compile runs
    /// once per distinct schema).
    #[must_use]
    pub fn strict_tool_def_error_cached(&self, body: &Value) -> Option<String> {
        let key = verdict_key(2, &["tools"], body);
        if let Some(hit) = self.verdicts.lock().get(key) {
            return hit.clone();
        }
        let verdict = strict_tool_def_error(body);
        self.verdicts.lock().put(key, verdict.clone());
        verdict
    }

    /// D6 counters for `/metrics` and tests: (hits, misses).
    #[must_use]
    pub fn verdict_stats(&self) -> (u64, u64) {
        let v = self.verdicts.lock();
        (v.hits, v.misses)
    }

    /// Begin observing a request: spawns the analyzer and returns the
    /// hot-path feed. Error statuses are recorded nowhere — the gateway
    /// already surfaces transport failures loudly.
    #[must_use]
    pub fn begin(self: &Arc<Self>, ctx: RequestCtx, status: u16, sse: bool) -> SentinelFeed {
        let degraded = Arc::new(AtomicBool::new(false));
        if !self.enabled || status >= 400 {
            return SentinelFeed { tx: None, degraded };
        }
        // (inert feeds are also constructed via `SentinelFeed::inert()`)
        let (tx, rx) = mpsc::channel(CHANNEL_CAP);
        let sentinel = Arc::clone(self);
        let feed_degraded = Arc::clone(&degraded);
        tokio::spawn(async move {
            sentinel.analyze(ctx, status, sse, feed_degraded, rx).await;
        });
        SentinelFeed {
            tx: Some(tx),
            degraded,
        }
    }

    /// Capability precheck: does this model's embedded template render
    /// tools? Cached per model; reads the GGUF header once.
    /// `None` = no verdict (unreadable GGUF — doctor flags those).
    #[must_use]
    pub fn template_support(&self, model: &str, gguf_path: &Path) -> Option<TemplateSupport> {
        {
            let map = self.templates.lock();
            if let Some(v) = map.get(model) {
                return Some(*v);
            }
        }
        let meta = pallama_core::read_metadata_file(gguf_path).ok()?;
        let verdict = match meta.chat_template.as_deref() {
            None => TemplateSupport::Missing,
            Some(t) if template_has_tools(t) => TemplateSupport::Tools,
            Some(_) => TemplateSupport::NoTools,
        };
        self.templates.lock().insert(model.to_string(), verdict);
        Some(verdict)
    }

    /// Ring query for `/api/why` + `pallama why`: exact trace match, or
    /// the most recent `limit` records.
    #[must_use]
    pub fn why(&self, trace: Option<&str>, limit: usize) -> Vec<SentinelRecord> {
        let ring = self.ring.lock();
        match trace {
            Some(t) => ring
                .iter()
                .rev()
                .filter(|r| r.trace == t)
                .cloned()
                .collect(),
            None => ring.iter().rev().take(limit).cloned().collect(),
        }
    }

    async fn analyze(
        &self,
        ctx: RequestCtx,
        status: u16,
        sse: bool,
        degraded: Arc<AtomicBool>,
        mut rx: mpsc::Receiver<FeedEvent>,
    ) {
        let started = std::time::Instant::now();
        let responses = ctx.route == "openai-responses";
        let mut acc = Accum::default();
        let mut carry = String::new();
        let mut lines = tr::LineBuffer::new();
        let mut json_buf: Vec<u8> = Vec::new();
        let mut stalled = false;
        loop {
            let ev = if self.stall.as_secs() > 0 {
                let Ok(ev) = tokio::time::timeout(self.stall, rx.recv()).await else {
                    if !stalled {
                        stalled = true;
                        tracing::warn!(
                            target: "pallama::sentinel",
                            trace = %ctx.trace,
                            model = %ctx.model,
                            code = Code::StalledStream.as_str(),
                            "{}", Code::StalledStream.hint()
                        );
                    }
                    continue;
                };
                ev
            } else {
                rx.recv().await
            };
            match ev {
                Some(FeedEvent::Bytes(b)) => {
                    if sse {
                        carry.push_str(&lines.feed(&b));
                        if carry.len() > MAX_CARRY_BYTES {
                            acc.degraded = true;
                            carry.clear();
                            continue;
                        }
                        let (events, _done, consumed) = tr::parse_sse(&carry);
                        carry.drain(..consumed);
                        for ev in &events {
                            if responses {
                                acc.apply_responses(ev);
                            } else {
                                acc.apply(ev);
                            }
                        }
                    } else if json_buf.len() + b.len() <= MAX_ACCUM_BYTES {
                        json_buf.extend_from_slice(&b);
                    } else {
                        acc.degraded = true;
                    }
                }
                Some(FeedEvent::Value(v)) => {
                    if responses {
                        acc.apply_responses(&v);
                    } else {
                        acc.apply(&v);
                    }
                }
                Some(FeedEvent::End) | None => break,
            }
        }
        if !json_buf.is_empty() {
            if let Ok(v) = serde_json::from_slice::<Value>(&json_buf) {
                if responses {
                    acc.apply_responses(&v);
                } else {
                    acc.apply(&v);
                }
            }
        }
        let mut detections = self.finalize(&ctx, &acc, status);
        if stalled {
            // The live warn above fired DURING the stall; the record makes
            // it answerable via `why` after the fact.
            detections.push(Detection {
                code: Code::StalledStream,
                detail: format!(
                    "no chunks for {}s while the stream stayed open",
                    self.stall.as_secs()
                ),
            });
        }
        let record = build_record(
            &ctx,
            &acc,
            detections,
            status,
            sse,
            degraded.load(Ordering::Relaxed) || acc.degraded,
            started,
        );
        self.commit(&record);
    }

    /// Warn, ring-insert, and persist one finalized record.
    fn commit(&self, record: &SentinelRecord) {
        for d in &record.detections {
            tracing::warn!(
                target: "pallama::sentinel",
                trace = %record.trace,
                model = %record.model,
                code = d.code.as_str(),
                detail = %d.detail,
                "{}", d.code.hint()
            );
            // J5: a stalled stream means the child is wedged (swap
            // thrash / driver hang) — ask the supervisor to reap it so
            // the next request respawns clean instead of queueing
            // behind a zombie. Best-effort: the evict itself is async.
            if d.code == Code::StalledStream {
                if let Some(hook) = self.evict_hook.lock().clone() {
                    let _ = hook.0.send(record.model.clone());
                    tracing::info!(target: "pallama::sentinel", model = %record.model, "stalled child — eviction requested");
                }
            }
        }
        {
            let mut ring = self.ring.lock();
            if ring.len() == RING_CAP {
                ring.pop_front();
            }
            ring.push_back(record.clone());
        }
        // No receivers = no error worth hearing about.
        let _ = self.watch_tx.send(record.clone());
        self.persist_record(record);
    }

    /// Subscribe to the live record stream (`pallama watch`).
    #[must_use]
    pub fn watch(&self) -> tokio::sync::broadcast::Receiver<SentinelRecord> {
        self.watch_tx.subscribe()
    }

    /// Append one JSON line; rotate (rewrite with the newest rows) when
    /// the file crosses its cap. IO failures log and continue — the ring
    /// in memory is always the source of truth for the current session.
    fn persist_record(&self, record: &SentinelRecord) {
        use std::io::Write as _;
        let Some(path) = &self.persist_path else {
            return;
        };
        let Some(mut p) = self.persist.as_ref().map(|p| p.lock()) else {
            return;
        };
        let Ok(line) = serde_json::to_string(record) else {
            return;
        };
        let line_len = u64::try_from(line.len()).unwrap_or(u64::MAX);
        if p.written.saturating_add(line_len).saturating_add(1) > PERSIST_CAP_BYTES {
            let keep: Vec<String> = {
                let ring = self.ring.lock();
                ring.iter()
                    .rev()
                    .take(PERSIST_KEEP)
                    .rev()
                    .filter_map(|r| serde_json::to_string(r).ok())
                    .collect()
            };
            if let Ok(mut f) = std::fs::File::create(path) {
                let mut written = 0_u64;
                for l in keep {
                    let _ = writeln!(f, "{l}");
                    written =
                        written.saturating_add(u64::try_from(l.len()).unwrap_or(u64::MAX) + 1);
                }
                p.file = f;
                p.written = written;
            }
        }
        if let Err(e) = writeln!(p.file, "{line}") {
            tracing::debug!(target: "pallama::sentinel", "persist append: {e}");
        } else {
            p.written = p.written.saturating_add(line_len + 1);
        }
    }

    /// Detection predicates over the accumulated response.
    fn finalize(&self, ctx: &RequestCtx, acc: &Accum, status: u16) -> Vec<Detection> {
        let mut out = Vec::new();

        if !ctx.tool_names.is_empty() {
            match ctx.template {
                Some(TemplateSupport::Missing) => out.push(Detection {
                    code: Code::TemplateNoTools,
                    detail: format!(
                        "{} has no embedded chat template; tools cannot be rendered",
                        ctx.model
                    ),
                }),
                Some(TemplateSupport::NoTools) => out.push(Detection {
                    code: Code::TemplateNoTools,
                    detail: format!("{}'s chat template contains no tool markers", ctx.model),
                }),
                _ => {}
            }
        }

        if acc.finish.as_deref() == Some("length") {
            // `length` covers BOTH a real ctx-ceiling hit and a client-set
            // max_tokens budget cap. Only call it ctx truncation when the
            // token counts prove the ceiling was approached; a small budget
            // capping generation is the client's own request, not a fault
            // (ReasoningNoAnswer still catches the pathological pairing).
            // Unverifiable (usage or ctx unknown) keeps the legacy flag.
            let ctx_exhausted = match (acc.usage_prompt, acc.usage_completion, ctx.ctx) {
                (Some(p), Some(c), Some(window)) => {
                    (p + c) * 10 >= u64::from(window) * NEAR_LIMIT_TENTHS
                }
                _ => true,
            };
            if ctx_exhausted {
                out.push(Detection {
                    code: Code::CtxTruncated,
                    detail: format!(
                        "prompt {:?} + completion {:?} tokens hit the ctx ceiling{}",
                        acc.usage_prompt,
                        acc.usage_completion,
                        ctx.ctx.map(|c| format!(" ({c})")).unwrap_or_default()
                    ),
                });
            }
        }
        if let (Some(p), Some(c)) = (acc.usage_prompt, ctx.ctx) {
            if p * 10 > u64::from(c) * NEAR_LIMIT_TENTHS {
                out.push(Detection {
                    code: Code::CtxNearLimit,
                    detail: format!("prompt {p} tokens vs ctx {c}"),
                });
            }
        }

        self.tool_detections(ctx, acc, &mut out);
        self.format_detections(ctx, acc, &mut out);

        if acc.saw_any_choice {
            let has_content = !acc.content.trim().is_empty() || !acc.text.trim().is_empty();
            let has_tools = !acc.tools.is_empty();
            let has_reasoning = acc.has_reasoning || !acc.reasoning.trim().is_empty();
            if !has_content && !has_tools {
                if has_reasoning {
                    out.push(Detection {
                        code: Code::ReasoningNoAnswer,
                        detail: format!(
                            "{} reasoning chars, zero answer content",
                            acc.reasoning.len()
                        ),
                    });
                } else {
                    out.push(Detection {
                        code: Code::EmptyResponse,
                        detail: "no content, tool calls, or reasoning in the response".into(),
                    });
                }
            }
        } else if status < 400 {
            out.push(Detection {
                code: Code::EmptyResponse,
                detail: "response carried no choices".into(),
            });
        }
        out
    }

    /// Tool-call validity: unknown names, non-JSON arguments, and
    /// arguments violating the tool's own `parameters` schema.
    fn tool_detections(&self, ctx: &RequestCtx, acc: &Accum, out: &mut Vec<Detection>) {
        for frag in acc.tools.values() {
            if frag.name.is_empty() {
                continue;
            }
            if !ctx.tool_names.is_empty() && !ctx.tool_names.contains(&frag.name) {
                out.push(Detection {
                    code: Code::ToolNameUnknown,
                    detail: format!(
                        "`{}` not in request tools [{}]",
                        frag.name,
                        ctx.tool_names.join(", ")
                    ),
                });
            }
            // Empty arguments string: lenient reading as "no arguments"
            // (some templates emit it); anything else must parse.
            if frag.args.trim().is_empty() {
                continue;
            }
            match serde_json::from_str::<Value>(&frag.args) {
                Err(e) => out.push(Detection {
                    code: Code::ToolArgsInvalidJson,
                    detail: format!(
                        "tool `{}` args not valid JSON: {e}: {}",
                        frag.name,
                        preview(&frag.args, 120)
                    ),
                }),
                Ok(v) => {
                    if let Some(schema) = ctx.tool_schemas.get(&frag.name) {
                        if let Some(e) = self.schema_violation(schema, &v) {
                            out.push(Detection {
                                code: Code::SchemaViolation,
                                detail: format!("tool `{}` args: {e}", frag.name),
                            });
                        }
                    }
                }
            }
        }
    }

    /// Structured-output validity: `json_object` must parse;
    /// `json_schema` must also validate against the requested schema.
    fn format_detections(&self, ctx: &RequestCtx, acc: &Accum, out: &mut Vec<Detection>) {
        let Some(rf) = &ctx.response_format else {
            return;
        };
        let ty = rf.get("type").and_then(Value::as_str).unwrap_or_default();
        if ty != "json_object" && ty != "json_schema" {
            return;
        }
        let content = if acc.content.is_empty() {
            &acc.text
        } else {
            &acc.content
        };
        match serde_json::from_str::<Value>(content) {
            Err(e) => out.push(Detection {
                code: Code::SchemaViolation,
                detail: format!("response is not valid JSON: {e}"),
            }),
            Ok(v) => {
                if ty == "json_schema" {
                    if let Some(schema) = rf.get("schema") {
                        if let Some(e) = self.schema_violation(schema, &v) {
                            out.push(Detection {
                                code: Code::SchemaViolation,
                                detail: format!("response: {e}"),
                            });
                        }
                    }
                }
            }
        }
    }

    /// Synchronous judgement for the enforce path (non-streaming, body
    /// already buffered): runs the SHARED detection predicates (no drift
    /// from warn-only), records the outcome via `commit`, and returns the
    /// hard violations (empty = pass).
    pub fn judge(&self, ctx: &RequestCtx, body: &[u8], status: u16) -> Vec<Detection> {
        let started = std::time::Instant::now();
        let mut acc = Accum::default();
        if let Ok(v) = serde_json::from_slice::<Value>(body) {
            if ctx.route == "openai-responses" {
                acc.apply_responses(&v);
            } else {
                acc.apply(&v);
            }
        }
        let detections = self.finalize(ctx, &acc, status);
        let hard: Vec<Detection> = detections
            .iter()
            .filter(|d| HARD_CODES.contains(&d.code))
            .cloned()
            .collect();
        let record = build_record(ctx, &acc, detections, status, false, acc.degraded, started);
        self.commit(&record);
        hard
    }

    /// Validate `instance` against `schema`; `None` = clean or
    /// unjudgeable (uncompilable schema is not the response's fault).
    fn schema_violation(&self, schema: &Value, instance: &Value) -> Option<String> {
        let key = serde_json::to_string(schema).ok()?;
        let validator = {
            let mut cache = self.schemas.lock();
            if cache.len() >= SCHEMA_CACHE_CAP {
                cache.clear();
            }
            cache
                .entry(key)
                .or_insert_with(|| jsonschema::validator_for(schema).ok().map(Arc::new))
                .clone()
        };
        let validator = validator?;
        if validator.is_valid(instance) {
            return None;
        }
        let mut msg = String::from("schema violation");
        if let Some(e) = validator.iter_errors(instance).next() {
            let text: String = e.to_string().chars().take(160).collect();
            msg = format!("{text} (at {})", e.instance_path());
        }
        Some(msg)
    }
}

/// Assemble the finalized record (shared by the async analyzer and the
/// synchronous enforce judge — one shape, no drift).
/// True request latency: trace ids are minted at request start
/// (`plm-<millis-hex>-seq`, from `request_log`), so the request's wall
/// time decodes straight out of the id — no plumbing through handlers.
/// Fallback: the analyzer's own span (tests, extension-less calls).
fn now_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}

fn request_ms(trace: &str, fallback: std::time::Instant) -> u128 {
    let decoded = trace
        .split('-')
        .nth(1)
        .and_then(|hex| u64::from_str_radix(hex, 16).ok())
        .map(|start_ms| {
            let now_ms = now_millis();
            u128::from(now_ms.saturating_sub(start_ms))
        });
    decoded.unwrap_or_else(|| fallback.elapsed().as_millis())
}

#[allow(clippy::cast_precision_loss)] // token count -> mean divisor
fn build_record(
    ctx: &RequestCtx,
    acc: &Accum,
    detections: Vec<Detection>,
    status: u16,
    sse: bool,
    degraded: bool,
    started: std::time::Instant,
) -> SentinelRecord {
    SentinelRecord {
        trace: ctx.trace.clone(),
        ts: now_secs(),
        route: ctx.route.clone(),
        model: ctx.model.clone(),
        status,
        stream: ctx.stream || sse,
        finish: acc.finish.clone(),
        detections,
        prompt_tokens: acc.usage_prompt,
        completion_tokens: acc.usage_completion,
        ctx: ctx.ctx,
        degraded,
        ms: request_ms(&ctx.trace, started),
        logprob_mean: (acc.lp_tokens > 0).then(|| acc.lp_sum / acc.lp_tokens as f64),
        logprob_min: acc.lp_min,
        logprob_tokens: (acc.lp_tokens > 0).then_some(acc.lp_tokens),
    }
}

/// Resolve enforce for a request: an explicit `X-Pallama-Enforce` header
/// wins over the config default (`1`/`0`; absent = config value).
#[must_use]
pub fn enforce_enabled(config: &pallama_core::Config, headers: &axum::http::HeaderMap) -> bool {
    match headers
        .get("x-pallama-enforce")
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
    {
        Some(v) if v == "1" || v.eq_ignore_ascii_case("true") => true,
        Some(v) if v == "0" || v.eq_ignore_ascii_case("false") => false,
        _ => config.sentinel_enforce,
    }
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

fn preview(s: &str, n: usize) -> String {
    let t = s.trim();
    if t.chars().count() <= n {
        t.to_string()
    } else {
        format!("{}…", t.chars().take(n).collect::<String>())
    }
}

/// Accumulated response state. Handles both streamed deltas and complete
/// messages (`delta` / `message` shapes), plus legacy `text` completions.
#[derive(Default)]
struct Accum {
    content: String,
    reasoning: String,
    text: String,
    tools: BTreeMap<u64, ToolFrag>,
    finish: Option<String>,
    usage_prompt: Option<u64>,
    usage_completion: Option<u64>,
    saw_any_choice: bool,
    degraded: bool,
    /// Responses grammar: key of the most recent `function_call` item
    /// (argument deltas append to it).
    last_tool: Option<u64>,
    /// Reasoning seen at all (Responses grammar reports items, not text).
    has_reasoning: bool,
    /// Logprob confidence (R2): per-token logprobs from
    /// choices[].logprobs.content[] when the client asked for them.
    /// Display-only — no detection fires on these.
    lp_sum: f64,
    lp_min: Option<f64>,
    lp_tokens: u64,
}

#[derive(Default, Clone)]
struct ToolFrag {
    name: String,
    args: String,
}

impl Accum {
    fn apply(&mut self, ev: &Value) {
        if let Some(u) = ev.get("usage").filter(|u| u.is_object()) {
            self.usage_prompt = u
                .get("prompt_tokens")
                .and_then(Value::as_u64)
                .or(self.usage_prompt);
            self.usage_completion = u
                .get("completion_tokens")
                .and_then(Value::as_u64)
                .or(self.usage_completion);
        }
        let Some(choices) = ev.get("choices").and_then(Value::as_array) else {
            return;
        };
        if !choices.is_empty() {
            self.saw_any_choice = true;
        }
        for choice in choices {
            if let Some(fr) = choice.get("finish_reason").and_then(Value::as_str) {
                self.finish = Some(fr.to_string());
            }
            // Logprob confidence (R2): same shape on stream chunks and
            // complete bodies — content[].logprob per token.
            if let Some(lps) = choice
                .pointer("/logprobs/content")
                .and_then(Value::as_array)
            {
                for lp in lps {
                    if let Some(v) = lp.get("logprob").and_then(Value::as_f64) {
                        self.lp_sum += v;
                        self.lp_tokens += 1;
                        self.lp_min = Some(match self.lp_min {
                            Some(m) if m <= v => m,
                            _ => v,
                        });
                    }
                }
            }
            let carrier = choice
                .get("delta")
                .or_else(|| choice.get("message"))
                .unwrap_or(&Value::Null);
            if let Some(s) = carrier.get("content").and_then(Value::as_str) {
                Self::push_bounded(&mut self.content, &mut self.degraded, s);
            }
            if let Some(s) = carrier.get("reasoning_content").and_then(Value::as_str) {
                self.has_reasoning = true;
                Self::push_bounded(&mut self.reasoning, &mut self.degraded, s);
            }
            // Legacy completions grammar puts `text` directly on the
            // choice (no delta/message carrier); both keys are read so
            // /api/generate + /v1/completions traffic accumulates.
            let text = carrier
                .get("text")
                .or_else(|| choice.get("text"))
                .and_then(Value::as_str);
            if let Some(s) = text {
                Self::push_bounded(&mut self.text, &mut self.degraded, s);
            }
            if let Some(tcs) = carrier.get("tool_calls").and_then(Value::as_array) {
                for (i, tc) in tcs.iter().enumerate() {
                    let key = tc
                        .get("index")
                        .and_then(Value::as_u64)
                        .unwrap_or_else(|| u64::try_from(i).unwrap_or(0));
                    let entry = self.tools.entry(key).or_default();
                    if let Some(n) = tc["function"]["name"].as_str() {
                        entry.name = n.to_string();
                    }
                    if let Some(a) = tc["function"]["arguments"].as_str() {
                        if entry.args.len() + a.len() < MAX_ACCUM_BYTES {
                            entry.args.push_str(a);
                        } else {
                            self.degraded = true;
                        }
                    }
                }
            }
        }
    }

    /// Responses-API grammar: streamed events carry `type`; the final
    /// non-stream response object does not — both land here. Unknown
    /// event types are ignored by design (the surface evolves upstream).
    fn apply_responses(&mut self, ev: &Value) {
        if let Some(ty) = ev.get("type").and_then(Value::as_str) {
            match ty {
                "response.output_text.delta" => {
                    self.saw_any_choice = true;
                    if let Some(d) = ev.get("delta").and_then(Value::as_str) {
                        Self::push_bounded(&mut self.content, &mut self.degraded, d);
                    }
                }
                "response.reasoning_text.delta" | "response.reasoning_summary_text.delta" => {
                    self.has_reasoning = true;
                    if let Some(d) = ev.get("delta").and_then(Value::as_str) {
                        Self::push_bounded(&mut self.reasoning, &mut self.degraded, d);
                    }
                }
                "response.output_item.added" | "response.output_item.done" => {
                    let item = ev.get("item").unwrap_or(&Value::Null);
                    if item.get("type").and_then(Value::as_str) == Some("function_call") {
                        self.saw_any_choice = true;
                        let key = ev.get("output_index").and_then(Value::as_u64).unwrap_or(0);
                        let entry = self.tools.entry(key).or_default();
                        if let Some(n) = item.get("name").and_then(Value::as_str) {
                            entry.name = n.to_string();
                        }
                        // `done` carries the complete item: only take its
                        // arguments when the deltas never fed this frag
                        // (double-append would corrupt the JSON check).
                        if ty.ends_with("done") {
                            if let Some(a) = item.get("arguments").and_then(Value::as_str) {
                                // F61: bound like every sibling accumulator
                                if entry.args.is_empty()
                                    && !a.is_empty()
                                    && entry.args.len() + a.len() < MAX_ACCUM_BYTES
                                {
                                    entry.args.push_str(a);
                                }
                            }
                        }
                        self.last_tool = Some(key);
                    } else {
                        self.saw_any_choice = true;
                    }
                }
                "response.function_call_arguments.delta" => {
                    if let (Some(key), Some(d)) =
                        (self.last_tool, ev.get("delta").and_then(Value::as_str))
                    {
                        if let Some(entry) = self.tools.get_mut(&key) {
                            if entry.args.len() + d.len() < MAX_ACCUM_BYTES {
                                entry.args.push_str(d);
                            } else {
                                self.degraded = true;
                            }
                        }
                    }
                }
                "response.completed" | "response.incomplete" | "response.failed" => {
                    self.apply_responses_object(ev.get("response").unwrap_or(&Value::Null));
                }
                _ => {}
            }
            return;
        }
        self.apply_responses_object(ev);
    }

    /// Terminal states: `status`/`incomplete_details` + output items + usage.
    fn apply_responses_object(&mut self, obj: &Value) {
        if let Some(status) = obj.get("status").and_then(Value::as_str) {
            match status {
                "incomplete" => {
                    let reason = obj
                        .pointer("/incomplete_details/reason")
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    self.finish = Some(
                        if reason == "max_output_tokens" {
                            "length"
                        } else {
                            "stop"
                        }
                        .into(),
                    );
                }
                "failed" => self.finish = Some("stop".into()),
                _ => {}
            }
        }
        if let Some(items) = obj.get("output").and_then(Value::as_array) {
            if !items.is_empty() {
                self.saw_any_choice = true;
            }
            for (i, item) in items.iter().enumerate() {
                match item.get("type").and_then(Value::as_str) {
                    Some("function_call") => {
                        let key = u64::try_from(i).unwrap_or(0);
                        let entry = self.tools.entry(key).or_default();
                        if let Some(n) = item.get("name").and_then(Value::as_str) {
                            entry.name = n.to_string();
                        }
                        // F61: bound like every sibling accumulator — a
                        // body-limit-sized `arguments` string must not
                        // mirror into memory unbounded.
                        if let Some(a) = item.get("arguments").and_then(Value::as_str) {
                            if entry.args.len() + a.len() < MAX_ACCUM_BYTES {
                                entry.args.push_str(a);
                            }
                        }
                    }
                    Some("message") => {
                        if let Some(c) = item.get("content") {
                            if let Some(s) = c.as_str() {
                                Self::push_bounded(&mut self.content, &mut self.degraded, s);
                            } else if let Some(parts) = c.as_array() {
                                for p in parts {
                                    if let Some(t) = p.get("text").and_then(Value::as_str) {
                                        Self::push_bounded(
                                            &mut self.content,
                                            &mut self.degraded,
                                            t,
                                        );
                                    }
                                }
                            }
                        }
                    }
                    Some("reasoning") => {
                        self.has_reasoning = true;
                    }
                    _ => {}
                }
            }
        }
        if let Some(u) = obj.get("usage").filter(|u| u.is_object()) {
            self.usage_prompt = u
                .get("input_tokens")
                .and_then(Value::as_u64)
                .or(self.usage_prompt);
            self.usage_completion = u
                .get("output_tokens")
                .and_then(Value::as_u64)
                .or(self.usage_completion);
        }
    }

    fn push_bounded(dst: &mut String, degraded: &mut bool, s: &str) {
        if dst.len() + s.len() < MAX_ACCUM_BYTES {
            dst.push_str(s);
        } else {
            *degraded = true;
        }
    }
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;
    use serde_json::json;

    fn ctx_with(tools: &[&str]) -> RequestCtx {
        let (names, schemas, _, _) = parse_request_ctx(
            serde_json::to_vec(&json!({
                "tools": tools.iter().map(|n| json!({"function": {"name": n, "parameters": {"type": "object"}}})).collect::<Vec<_>>()
            }))
            .unwrap_or_default()
            .as_slice(),
        );
        RequestCtx {
            tool_names: names,
            tool_schemas: schemas,
            ..RequestCtx::default()
        }
    }

    #[tokio::test]
    async fn integration__commit_broadcasts_to_watch_subscribers() {
        let s = Sentinel::new(true, 0, None);
        let mut rx = s.watch();
        let ctx = RequestCtx {
            trace: "plm-watch-1".into(),
            route: "openai-chat".into(),
            model: "m".into(),
            ..Default::default()
        };
        let body = serde_json::to_vec(&serde_json::json!({
            "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]
        }))
        .unwrap();
        let _ = s.judge(&ctx, &body, 200);
        let got = rx.recv().await.expect("record broadcast");
        assert_eq!(got.trace, "plm-watch-1");
    }

    #[test]
    fn unit__request_ms__decodes_trace_age_falls_back_clean() {
        // Fresh trace (now): decodes to a tiny non-zero-ish value >= 0.
        let now_ms = now_millis();
        let fresh = format!("plm-{now_ms:x}-7");
        let v = request_ms(&fresh, std::time::Instant::now());
        assert!(v < 5_000, "fresh trace decoded to {v}ms");
        // Old trace: roughly the age.
        let old = format!("plm-{:x}-1", now_ms - 60_000);
        let v = request_ms(&old, std::time::Instant::now());
        assert!(
            (59_000..=61_500).contains(&v),
            "aged trace decoded to {v}ms"
        );
        // Garbage / absent trace: falls back to the analyzer span.
        assert_eq!(request_ms("garbage", std::time::Instant::now()), 0);
    }

    #[test]
    fn unit__retry_hints__cover_every_code_nonempty() {
        let all = [
            Code::CtxTruncated,
            Code::CtxNearLimit,
            Code::ToolArgsInvalidJson,
            Code::ToolNameUnknown,
            Code::SchemaViolation,
            Code::EmptyResponse,
            Code::ReasoningNoAnswer,
            Code::StalledStream,
            Code::TemplateNoTools,
        ];
        for c in all {
            assert!(c.retry_hint().len() > 20, "{c:?} retry hint too thin");
        }
        // The JSON surface carries it.
        let rec = SentinelRecord {
            trace: "t".into(),
            ts: 1,
            route: "openai-chat".into(),
            model: "m".into(),
            status: 200,
            stream: false,
            finish: None,
            detections: vec![Detection {
                code: Code::CtxTruncated,
                detail: "d".into(),
            }],
            prompt_tokens: None,
            completion_tokens: None,
            ctx: None,
            degraded: false,
            logprob_mean: None,
            logprob_min: None,
            logprob_tokens: None,
            ms: 1,
        };
        assert!(rec.to_json()["detections"][0]["retry"]
            .as_str()
            .is_some_and(|s| !s.is_empty()));
    }

    #[test]
    fn unit__verdict_cache__hits_misses_and_purity() {
        let s = Sentinel::new(false, 30, None);
        let body = serde_json::json!({
            "model": "m",
            "format": "jsom" // typo'd format -> verdict Some(..)
        });
        let v1 = s.structured_output_error_cached(&body);
        assert!(v1.is_some(), "typo format must lint");
        // Same read-fields -> cache hit, identical verdict; unrelated
        // field differences (model) must NOT miss.
        let body2 = serde_json::json!({
            "model": "other",
            "stream": true,
            "format": "jsom"
        });
        let v2 = s.structured_output_error_cached(&body2);
        assert_eq!(v1, v2);
        let (hits, misses) = s.verdict_stats();
        assert_eq!((hits, misses), (1, 1), "second call hits the LRU");
        // Different format value -> miss -> clean verdict.
        let clean = serde_json::json!({"format": "json"});
        assert!(s.structured_output_error_cached(&clean).is_none());
        let (hits, misses) = s.verdict_stats();
        assert_eq!((hits, misses), (1, 2));
    }

    #[test]
    fn unit__verdict_cache__strict_tools_cached_across_callers() {
        let s = Sentinel::new(false, 30, None);
        let broken = serde_json::json!({
            "tools": [{"function": {"name": "f", "strict": true,
                "parameters": {"type": "object", "properties": {"x": {}}, "required": ["x"]}}}]
        });
        let a = s.strict_tool_def_error_cached(&broken);
        let b = s.strict_tool_def_error_cached(&broken);
        assert!(a.is_some() && a == b);
        assert_eq!(s.verdict_stats(), (1, 1));
    }

    #[test]
    fn unit__verdict_cache__lru_bounded_and_fresh_after_eviction() {
        let s = Sentinel::new(false, 30, None);
        // VERDICT_CAP + 1 distinct clean bodies evict the first key.
        for i in 0..=VERDICT_CAP {
            let b = serde_json::json!({"format": "json", "stream": i});
            let _ = s.structured_output_error_cached(&b);
        }
        assert!(
            s.verdicts.lock().map.len() <= VERDICT_CAP,
            "cache stays bounded"
        );
        // Distinct grammar bytes hash to distinct keys (no aliasing).
        let g1 = serde_json::json!({"grammar": "root ::= \"a\""});
        let g2 = serde_json::json!({"grammar": "root ::= \"b\""});
        assert_ne!(
            verdict_key(1, &["grammar", "format", "response_format"], &g1),
            verdict_key(1, &["grammar", "format", "response_format"], &g2)
        );
    }

    #[test]
    fn unit__strict_tool_def__openai_strict_subset() {
        // Clean strict def (OpenAI chat + Responses internal-tag shapes).
        let ok = serde_json::json!({
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "strict": true,
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                        "additionalProperties": false
                    }
                }
            }]
        });
        assert!(strict_tool_def_error(&ok).is_none());
        let responses_shape = serde_json::json!({
            "tools": [{
                "type": "function", "name": "f", "strict": true,
                "parameters": {
                    "type": "object", "properties": {"x": {"type": "integer"}},
                    "required": ["x"], "additionalProperties": false
                }
            }]
        });
        assert!(strict_tool_def_error(&responses_shape).is_none());

        // additionalProperties missing.
        let ap = serde_json::json!({
            "tools": [{"function": {"name": "f", "strict": true,
                "parameters": {"type": "object", "properties": {"x": {}}, "required": ["x"]}}}]
        });
        assert!(strict_tool_def_error(&ap)
            .unwrap()
            .contains("additionalProperties"));
        // Property not in required.
        let req_missing = serde_json::json!({
            "tools": [{"function": {"name": "f", "strict": true,
                "parameters": {"type": "object", "properties": {"x": {}, "y": {}},
                    "required": ["x"], "additionalProperties": false}}}]
        });
        assert!(strict_tool_def_error(&req_missing).unwrap().contains('y'));
        // Non-compiling schema (any tool).
        let bad_schema = serde_json::json!({
            "tools": [{"function": {"name": "f",
                "parameters": {"type": "not-a-type"}}}]
        });
        assert!(strict_tool_def_error(&bad_schema)
            .unwrap()
            .contains("JSON Schema"));
        // No tools / no strict = None.
        assert!(strict_tool_def_error(&serde_json::json!({"model": "m"})).is_none());
    }

    #[test]
    fn unit__structured_output__happy_paths_pass() {
        // Absent / null / "json" / clean schema / empty grammar / clean
        // response_format — all pass untouched.
        for body in [
            serde_json::json!({"model": "m"}),
            serde_json::json!({"model": "m", "format": null, "grammar": null}),
            serde_json::json!({"model": "m", "format": "json"}),
            serde_json::json!({"model": "m", "format": {"type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]}}),
            serde_json::json!({"model": "m", "grammar": ""}),
            serde_json::json!({"model": "m", "grammar": "root ::= \"yes\" | \"no\""}),
            serde_json::json!({"model": "m", "response_format": {"type": "json_object"}}),
            serde_json::json!({"model": "m", "response_format": {"type": "json_schema",
                "json_schema": {"schema": {"type": "object"}}}}),
        ] {
            assert_eq!(structured_output_error(&body), None, "body: {body}");
        }
    }

    #[test]
    fn unit__structured_output__garbage_format_fails_fast() {
        // Unknown string / number / bool / array formats were silently
        // ignored before R9 — now they 400.
        for fmt in [
            serde_json::json!("nonsense"),
            serde_json::json!(5),
            serde_json::json!(true),
            serde_json::json!(["object"]),
        ] {
            let body = serde_json::json!({"model": "m", "format": fmt});
            let err = structured_output_error(&body).expect("must fail");
            assert!(err.contains("format"), "err: {err}");
        }
    }

    #[test]
    fn unit__structured_output__grammar_shape_and_cap() {
        // Non-string grammar.
        let body = serde_json::json!({"model": "m", "grammar": 42});
        assert!(structured_output_error(&body).unwrap().contains("string"));
        // Over-cap grammar.
        let huge = "a".repeat(MAX_GRAMMAR_BYTES + 1);
        let body = serde_json::json!({"model": "m", "grammar": huge});
        let err = structured_output_error(&body).unwrap();
        assert!(err.contains("cap"), "err: {err}");
        // Exactly at cap passes.
        let edge = "a".repeat(MAX_GRAMMAR_BYTES);
        let body = serde_json::json!({"model": "m", "grammar": edge});
        assert_eq!(structured_output_error(&body), None);
    }

    #[test]
    fn unit__structured_output__format_and_grammar_mutually_exclusive() {
        let body = serde_json::json!({
            "model": "m",
            "format": {"type": "object"},
            "grammar": "root ::= \"x\""
        });
        let err = structured_output_error(&body).expect("must fail");
        assert!(err.contains("both"), "err: {err}");
        // format:"json" + grammar is resolved by the child, not us.
        let body = serde_json::json!({"model": "m", "format": "json", "grammar": "root ::= \"x\""});
        assert_eq!(structured_output_error(&body), None);
    }

    #[test]
    fn unit__structured_output__schema_top_level_shape() {
        let bad_props = serde_json::json!({"model": "m", "format": {"properties": 5}});
        assert!(structured_output_error(&bad_props)
            .unwrap()
            .contains("properties"));
        let bad_required = serde_json::json!({"model": "m", "format": {"required": ["a", 5]}});
        assert!(structured_output_error(&bad_required)
            .unwrap()
            .contains("required"));
        let bad_required2 = serde_json::json!({"model": "m", "format": {"required": "a"}});
        assert!(structured_output_error(&bad_required2)
            .unwrap()
            .contains("required"));
        let bad_type = serde_json::json!({"model": "m", "format": {"type": 7}});
        assert!(structured_output_error(&bad_type).unwrap().contains("type"));
        // String-array type is legal.
        let ok = serde_json::json!({"model": "m", "format": {"type": ["object", "null"]}});
        assert_eq!(structured_output_error(&ok), None);
    }

    #[test]
    fn unit__structured_output__exotic_valid_schemas_never_rejected() {
        // Deep/compound keywords are the child converter's business —
        // the gateway lint never recurses into them.
        let exotic = serde_json::json!({
            "model": "m",
            "format": {
                "oneOf": [{"type": "string"}, {"type": "integer"}],
                "allOf": [{"type": "object"}],
                "anyOf": [{"type": "null"}],
                "$ref": "#/definitions/x",
                "patternProperties": {"^S_": {"type": "string"}},
                "definitions": {"x": {"type": "string"}},
                "nested": {"deep": {"deeper": {"required": [5]}}}
            }
        });
        assert_eq!(structured_output_error(&exotic), None);
    }

    #[test]
    fn unit__structured_output__response_format_mirrors_child_type_set() {
        // Unknown non-empty type → 400 (child: server-common.cpp:1180).
        let body = serde_json::json!({"model": "m", "response_format": {"type": "yaml"}});
        assert!(structured_output_error(&body)
            .unwrap()
            .contains("response_format.type"));
        // Non-string type.
        let body = serde_json::json!({"model": "m", "response_format": {"type": 4}});
        assert!(structured_output_error(&body)
            .unwrap()
            .contains("response_format.type"));
        // Non-object response_format.
        let body = serde_json::json!({"model": "m", "response_format": "json"});
        assert!(structured_output_error(&body)
            .unwrap()
            .contains("response_format must be"));
        // json_schema.schema present-non-object.
        let body = serde_json::json!({"model": "m",
            "response_format": {"type": "json_schema", "json_schema": {"schema": "oops"}}});
        assert!(structured_output_error(&body)
            .unwrap()
            .contains("schema must be an object"));
        // Absent schema inside json_schema = pass (child defaults {}).
        let body = serde_json::json!({"model": "m",
            "response_format": {"type": "json_schema", "json_schema": {}}});
        assert_eq!(structured_output_error(&body), None);
    }

    #[test]
    fn unit__singleflight_key__stream_bit_folds_apart() {
        let body = br#"{"messages":[{"role":"user","content":"hi"}]}"#;
        let a = singleflight_key("m1", body, false);
        let b = singleflight_key("m1", body, true);
        assert_ne!(a, b, "stream twins never share a slot");
        let other = singleflight_key("m2", body, false);
        assert_ne!(a, other, "model is part of the key");
        let same = singleflight_key("m1", body, false);
        assert_eq!(a, same);
    }

    #[test]
    fn unit__template_has_tools__markers() {
        assert!(template_has_tools(
            "{%- if tools %}{{ tool_calls }}{%- endif %}"
        ));
        assert!(template_has_tools("Hermes: <tool_call>"));
        assert!(!template_has_tools("You are a helpful assistant."));
    }

    #[test]
    fn unit__parse_request_ctx__tools_and_format() {
        let body = serde_json::to_vec(&json!({
            "tools": [
                {"function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}},
                {"function": {"name": "no_schema"}}
            ],
            "response_format": {"type": "json_schema", "schema": {"type": "object"}}
        }))
        .unwrap();
        let (names, schemas, rf, _stream) = parse_request_ctx(&body);
        assert_eq!(names, ["get_weather", "no_schema"]);
        assert!(schemas.contains_key("get_weather"));
        assert!(!schemas.contains_key("no_schema"));
        assert_eq!(rf.unwrap()["type"], "json_schema");
        // Non-JSON body: empty, no panic.
        let (n, s, r, st) = parse_request_ctx(b"not json");
        assert!(n.is_empty() && s.is_empty() && r.is_none() && !st);
    }

    #[test]
    fn unit__finalize__truncation_and_near_limit() {
        let s = Sentinel::new(true, 0, None);
        let acc = Accum {
            finish: Some("length".into()),
            usage_prompt: Some(120),
            usage_completion: Some(28),
            saw_any_choice: true,
            content: "partial".into(),
            ..Accum::default()
        };
        let ctx = RequestCtx {
            ctx: Some(128),
            ..RequestCtx::default()
        };
        let d = s.finalize(&ctx, &acc, 200);
        assert!(d.iter().any(|x| x.code == Code::CtxTruncated), "{d:?}");
        assert!(d.iter().any(|x| x.code == Code::CtxNearLimit), "{d:?}");
    }

    #[test]
    fn unit__accum__responses_object_args_bounded() {
        // F61: a body-limit-sized `arguments` string must not mirror into
        // memory unbounded — the accumulator caps like every sibling.
        let mut acc = Accum::default();
        let huge = "x".repeat(MAX_ACCUM_BYTES + 1024);
        let ev = serde_json::json!({
            "output": [
                {"type": "function_call", "name": "f", "arguments": huge}
            ]
        });
        acc.apply_responses_object(&ev);
        assert!(
            acc.tools
                .get(&0)
                .is_none_or(|t| t.args.len() < MAX_ACCUM_BYTES),
            "args must be bounded"
        );
        // Small arguments still accumulate normally.
        let mut acc = Accum::default();
        acc.apply_responses_object(&serde_json::json!({
            "output": [
                {"type": "function_call", "name": "f", "arguments": "{\"a\":1}"}
            ]
        }));
        assert_eq!(
            acc.tools.get(&0).map(|t| t.args.as_str()),
            Some("{\"a\":1}")
        );
    }

    #[test]
    fn unit__finalize__tool_fragment_checks() {
        let s = Sentinel::new(true, 0, None);
        let mut acc = Accum {
            saw_any_choice: true,
            ..Accum::default()
        };
        acc.tools.insert(
            0,
            ToolFrag {
                name: "get_weather".into(),
                args: "{\"city\"".into(),
            },
        );
        acc.tools.insert(
            1,
            ToolFrag {
                name: "hallucinated".into(),
                args: "{}".into(),
            },
        );
        let ctx = ctx_with(&["get_weather"]);
        let d = s.finalize(&ctx, &acc, 200);
        assert!(
            d.iter().any(|x| x.code == Code::ToolArgsInvalidJson),
            "{d:?}"
        );
        assert!(d.iter().any(|x| x.code == Code::ToolNameUnknown), "{d:?}");
    }

    #[test]
    fn unit__finalize__schema_violation_and_json_parse() {
        let s = Sentinel::new(true, 0, None);
        let acc = Accum {
            saw_any_choice: true,
            content: "{\"answer\": 42}".into(),
            ..Accum::default()
        };
        let ctx = RequestCtx {
            response_format: Some(
                json!({"type": "json_schema", "schema": {"type": "object", "properties": {"answer": {"type": "string"}}}}),
            ),
            ..RequestCtx::default()
        };
        let d = s.finalize(&ctx, &acc, 200);
        assert!(d.iter().any(|x| x.code == Code::SchemaViolation), "{d:?}");

        let acc2 = Accum {
            saw_any_choice: true,
            content: "not json at all".into(),
            ..Accum::default()
        };
        let d2 = s.finalize(&ctx, &acc2, 200);
        assert!(d2.iter().any(|x| x.code == Code::SchemaViolation), "{d2:?}");
    }

    #[test]
    fn unit__finalize__empty_vs_reasoning_no_answer() {
        let s = Sentinel::new(true, 0, None);
        let mut acc = Accum {
            saw_any_choice: true,
            ..Accum::default()
        };
        let d = s.finalize(&RequestCtx::default(), &acc, 200);
        assert!(d.iter().any(|x| x.code == Code::EmptyResponse), "{d:?}");

        acc.reasoning = "thinking hard".into();
        let d = s.finalize(&RequestCtx::default(), &acc, 200);
        assert!(d.iter().any(|x| x.code == Code::ReasoningNoAnswer), "{d:?}");
        assert!(!d.iter().any(|x| x.code == Code::EmptyResponse), "{d:?}");
    }

    #[test]
    fn unit__apply__completions_grammar_text_on_choice() {
        // Pin: the legacy completions grammar puts `text` directly on the
        // choice (never under delta/message). Dropping it made every
        // /api/generate + /v1/completions response look empty (observed:
        // 100% false EmptyResponse rate on both routes).
        let s = Sentinel::new(true, 0, None);
        let mut acc = Accum::default();
        acc.apply(&serde_json::json!({
            "choices": [{"text": "benchmark tokens", "finish_reason": "length"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2}
        }));
        assert_eq!(acc.text, "benchmark tokens");
        assert!(acc.saw_any_choice);
        let d = s.finalize(&RequestCtx::default(), &acc, 200);
        assert!(
            !d.iter().any(|x| x.code == Code::EmptyResponse),
            "completions text on choice must count as content: {d:?}"
        );

        // Genuinely empty completions response still flags.
        let mut empty = Accum::default();
        empty.apply(&serde_json::json!({
            "choices": [{"text": "", "finish_reason": "length"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2}
        }));
        let d2 = s.finalize(&RequestCtx::default(), &empty, 200);
        assert!(d2.iter().any(|x| x.code == Code::EmptyResponse), "{d2:?}");

        // Chat grammar (delta/message carriers) must keep working.
        let mut chat = Accum::default();
        chat.apply(&serde_json::json!({
            "choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}]
        }));
        assert_eq!(chat.content, "hi");
        let d3 = s.finalize(&RequestCtx::default(), &chat, 200);
        assert!(!d3.iter().any(|x| x.code == Code::EmptyResponse), "{d3:?}");
    }

    #[test]
    fn unit__accum__captures_every_upstream_grammar() {
        // Coverage pin: every response grammar any lane can feed the
        // analyzer must land in the accumulator — a grammar the parser
        // silently drops becomes a 100% false-flag route (the
        // EmptyResponse incident on ollama-generate/openai-completions).
        // Chat non-stream (message carrier).
        let mut chat_json = Accum::default();
        chat_json.apply(&serde_json::json!({
            "choices": [{"message": {"role": "assistant", "content": "a"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1}
        }));
        assert_eq!(chat_json.content, "a");
        assert_eq!(chat_json.finish.as_deref(), Some("stop"));
        assert_eq!(chat_json.usage_completion, Some(1));

        // Chat SSE delta chunks, content + reasoning.
        let mut chat_sse = Accum::default();
        chat_sse.apply(&serde_json::json!({"choices": [{"delta": {"reasoning_content": "th"}}]}));
        chat_sse.apply(&serde_json::json!({"choices": [{"delta": {"content": "b"}}]}));
        chat_sse.apply(&serde_json::json!({"choices": [{"delta": {}, "finish_reason": "stop"}]}));
        assert_eq!(chat_sse.reasoning, "th");
        assert_eq!(chat_sse.content, "b");

        // Completions non-stream + SSE chunk (text on the choice).
        let mut comp = Accum::default();
        comp.apply(&serde_json::json!({"choices": [{"text": "c"}]}));
        comp.apply(&serde_json::json!({"choices": [{"text": "d", "finish_reason": "length"}]}));
        assert_eq!(comp.text, "cd");

        // Streamed tool-call fragments reassemble by index.
        let mut tools = Accum::default();
        tools.apply(&serde_json::json!({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"name": "get_weather"}}
        ]}}]}));
        tools.apply(&serde_json::json!({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": "{\"city\":"}}
        ]}}]}));
        tools.apply(&serde_json::json!({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": "\"Oslo\"}"}}
        ]}}]}));
        let frag = tools.tools.get(&0).expect("fragment keyed by index");
        assert_eq!(frag.name, "get_weather");
        assert_eq!(frag.args, "{\"city\":\"Oslo\"}");

        // Each populated grammar alone must clear EmptyResponse.
        let s = Sentinel::new(true, 0, None);
        for acc in [chat_json, chat_sse, comp, tools] {
            let d = s.finalize(&RequestCtx::default(), &acc, 200);
            assert!(
                !d.iter().any(|x| x.code == Code::EmptyResponse),
                "grammar dropped by the accumulator: {d:?}"
            );
        }
    }

    #[test]
    fn unit__finalize__template_precheck_detection() {
        let s = Sentinel::new(true, 0, None);
        let acc = Accum {
            saw_any_choice: true,
            content: "I cannot call tools".into(),
            ..Accum::default()
        };
        let ctx = RequestCtx {
            template: Some(TemplateSupport::NoTools),
            ..ctx_with(&["get_weather"])
        };
        let d = s.finalize(&ctx, &acc, 200);
        assert!(d.iter().any(|x| x.code == Code::TemplateNoTools), "{d:?}");
    }

    #[test]
    fn unit__apply__streamed_tool_fragments_merge_by_index() {
        let mut acc = Accum::default();
        acc.apply(&json!({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"name": "f", "arguments": "{\"a\""}}
        ]}}]}));
        acc.apply(&json!({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": ": 1}"}}
        ]}}]}));
        acc.apply(&json!({"choices": [{"delta": {}, "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 9, "completion_tokens": 3}}));
        let frag = acc.tools.get(&0).cloned().unwrap_or_default();
        assert_eq!(frag.name, "f");
        assert_eq!(frag.args, "{\"a\": 1}");
        assert_eq!(acc.finish.as_deref(), Some("tool_calls"));
        assert_eq!(acc.usage_prompt, Some(9));
    }

    #[test]
    fn unit__apply__logprobs_accumulate_mean_min_tokens() {
        let mut acc = Accum::default();
        acc.apply(
            &json!({"choices": [{"delta": {"content": "he"}, "logprobs": {"content": [
                {"token": "he", "logprob": -0.2}, {"token": "llo", "logprob": -1.9}
            ]}}]}),
        );
        acc.apply(
            &json!({"choices": [{"delta": {"content": "llo"}, "logprobs": {"content": [
                {"token": "!", "logprob": -0.5}
            ]}}]}),
        );
        // A chunk without logprobs (client opted out mid-stream, or the
        // usage/final event) must not disturb the accumulation.
        acc.apply(&json!({"choices": [{"delta": {}, "finish_reason": "stop"}]}));
        assert_eq!(acc.lp_tokens, 3);
        assert!((acc.lp_sum - (-2.6)).abs() < 1e-9, "{}", acc.lp_sum);
        assert_eq!(acc.lp_min, Some(-1.9));
    }

    #[test]
    fn unit__build_record__logprob_fields_populated_or_none() {
        let mut acc = Accum::default();
        acc.apply(
            &json!({"choices": [{"delta": {"content": "x"}, "logprobs": {"content": [
                {"logprob": -0.5}, {"logprob": -1.5}
            ]}}]}),
        );
        let rec = build_record(
            &RequestCtx::default(),
            &acc,
            vec![],
            200,
            false,
            false,
            std::time::Instant::now(),
        );
        let mean = rec.logprob_mean.expect("mean");
        assert!((mean - (-1.0)).abs() < 1e-9, "{mean}");
        assert_eq!(rec.logprob_min, Some(-1.5));
        assert_eq!(rec.logprob_tokens, Some(2));
        // No logprobs requested -> all None, JSON stays lean.
        let bare = build_record(
            &RequestCtx::default(),
            &Accum::default(),
            vec![],
            200,
            false,
            false,
            std::time::Instant::now(),
        );
        assert_eq!(bare.logprob_mean, None);
        assert_eq!(bare.logprob_min, None);
        assert_eq!(bare.logprob_tokens, None);
        let j = bare.to_json().to_string();
        assert!(j.contains("\"logprob_mean\":null"), "{j}");
        let j2 = rec.to_json().to_string();
        assert!(j2.contains("\"logprob_min\":-1.5"), "{j2}");
        assert!(j2.contains("\"logprob_tokens\":2"), "{j2}");
    }

    #[tokio::test]
    async fn integration__analyze_sse_stream__logprob_confidence_recorded() {
        let s = Sentinel::new(true, 0, None);
        let ctx = RequestCtx {
            trace: "plm-lp-1".into(),
            model: "m".into(),
            route: "openai-chat".into(),
            ..RequestCtx::default()
        };
        let feed = s.begin(ctx, 200, true);
        feed.bytes(b"data: {\"choices\":[{\"delta\":{\"content\":\"he\"},\"logprobs\":{\"content\":[{\"logprob\":-0.2}]}}]}\n\n");
        feed.bytes(b"data: {\"choices\":[{\"delta\":{\"content\":\"llo\"},\"logprobs\":{\"content\":[{\"logprob\":-3.1}]}}]}\n\n");
        feed.bytes(b"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2}}\n\n");
        feed.bytes(b"data: [DONE]\n\n");
        feed.end();
        for _ in 0..50 {
            if !s.why(None, 10).is_empty() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        let recs = s.why(None, 10);
        assert_eq!(recs.len(), 1, "{recs:?}");
        let r = &recs[0];
        let mean = r.logprob_mean.expect("mean recorded");
        assert!((mean - (-1.65)).abs() < 1e-9, "{mean}");
        assert_eq!(r.logprob_min, Some(-3.1));
        assert_eq!(r.logprob_tokens, Some(2));
    }

    #[test]
    fn unit__why__ring_filter_and_limit() {
        let s = Sentinel::new(true, 0, None);
        let rec = |trace: &str| SentinelRecord {
            trace: trace.into(),
            ts: 1,
            route: "openai-chat".into(),
            model: "m".into(),
            status: 200,
            stream: true,
            finish: None,
            detections: vec![],
            prompt_tokens: None,
            completion_tokens: None,
            ctx: None,
            degraded: false,
            logprob_mean: None,
            logprob_min: None,
            logprob_tokens: None,
            ms: 5,
        };
        let mut ring = s.ring.lock();
        ring.push_back(rec("plm-a-1"));
        ring.push_back(rec("plm-b-2"));
        drop(ring);
        assert_eq!(s.why(Some("plm-b-2"), 10).len(), 1);
        assert_eq!(s.why(None, 1).len(), 1);
        assert_eq!(s.why(None, 10).len(), 2);
        assert_eq!(s.why(None, 10)[0].trace, "plm-b-2", "most recent first");
    }

    #[tokio::test]
    async fn integration__begin_disabled__feed_is_inert() {
        let s = Sentinel::new(false, 0, None);
        let feed = s.begin(RequestCtx::default(), 200, true);
        assert!(!feed.is_observing());
        feed.bytes(b"data: {}\n\n");
        feed.end();
        assert!(s.why(None, 10).is_empty());
    }

    #[tokio::test]
    async fn integration__analyze_sse_stream__records_truncation() {
        let s = Sentinel::new(true, 0, None);
        let ctx = RequestCtx {
            trace: "plm-t-1".into(),
            model: "m".into(),
            route: "openai-chat".into(),
            ctx: Some(64),
            ..RequestCtx::default()
        };
        let feed = s.begin(ctx, 200, true);
        feed.bytes(b"data: {\"choices\":[{\"delta\":{\"content\":\"he\"}}]}\n\n");
        feed.bytes(b"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}],\"usage\":{\"prompt_tokens\":80,\"completion_tokens\":2}}\n\n");
        feed.bytes(b"data: [DONE]\n\n");
        feed.end();
        // Analyzer runs async; poll the ring briefly.
        for _ in 0..50 {
            if !s.why(None, 10).is_empty() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        let recs = s.why(None, 10);
        assert_eq!(recs.len(), 1, "{recs:?}");
        let codes: Vec<&str> = recs[0].detections.iter().map(|d| d.code.as_str()).collect();
        assert!(codes.contains(&"ctx_truncated"), "{codes:?}");
        assert!(codes.contains(&"ctx_near_limit"), "{codes:?}");
    }

    #[tokio::test]
    async fn integration__analyze_sse_stream__budget_cap_not_ctx_truncated() {
        // Live incident shape (2026-09-06): finish_reason=length from a
        // client max_tokens=64 cap, 83 tokens against a 16384 ctx — the
        // budget is the client's own request, not a ctx-ceiling hit.
        let s = Sentinel::new(true, 0, None);
        let ctx = RequestCtx {
            trace: "plm-budget-1".into(),
            model: "m".into(),
            route: "openai-chat".into(),
            ctx: Some(16_384),
            ..RequestCtx::default()
        };
        let feed = s.begin(ctx, 200, true);
        feed.bytes(b"data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"thinking hard about the answer\"}}]}\n\n");
        feed.bytes(b"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}],\"usage\":{\"prompt_tokens\":19,\"completion_tokens\":64}}\n\n");
        feed.bytes(b"data: [DONE]\n\n");
        feed.end();
        for _ in 0..50 {
            if !s.why(None, 10).is_empty() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        let recs = s.why(None, 10);
        assert_eq!(recs.len(), 1, "{recs:?}");
        let codes: Vec<&str> = recs[0].detections.iter().map(|d| d.code.as_str()).collect();
        assert!(!codes.contains(&"ctx_truncated"), "{codes:?}");
        assert!(!codes.contains(&"ctx_near_limit"), "{codes:?}");
        assert!(codes.contains(&"reasoning_no_answer"), "{codes:?}");
    }

    #[tokio::test]
    async fn integration__analyze_sse_stream__usageless_length_still_flagged() {
        // Streaming without a usage chunk cannot prove either way — the
        // legacy conservative flag stands.
        let s = Sentinel::new(true, 0, None);
        let ctx = RequestCtx {
            trace: "plm-nousage-1".into(),
            model: "m".into(),
            route: "openai-chat".into(),
            ctx: Some(16_384),
            ..RequestCtx::default()
        };
        let feed = s.begin(ctx, 200, true);
        feed.bytes(b"data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n");
        feed.bytes(b"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}]}\n\n");
        feed.bytes(b"data: [DONE]\n\n");
        feed.end();
        for _ in 0..50 {
            if !s.why(None, 10).is_empty() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        let recs = s.why(None, 10);
        let codes: Vec<&str> = recs[0].detections.iter().map(|d| d.code.as_str()).collect();
        assert!(codes.contains(&"ctx_truncated"), "{codes:?}");
    }
}
