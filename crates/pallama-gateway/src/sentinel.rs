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
const HARD_CODES: &[Code] =
    &[Code::ToolArgsInvalidJson, Code::ToolNameUnknown, Code::SchemaViolation];
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
    pub detections: Vec<Detection>,
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub ctx: Option<u32>,
    pub degraded: bool,
    pub ms: u128,
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
        Self { tx: None, degraded: Arc::new(AtomicBool::new(false)) }
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
        pallama_core::store::Store::open(&state.dirs)
            .ok()
            .and_then(|s| s.get_model(model).ok().flatten())
            .and_then(|row| state.sentinel.template_support(model, std::path::Path::new(&row.path)))
    };
    let mut warnings = Vec::new();
    if matches!(template, Some(TemplateSupport::NoTools | TemplateSupport::Missing)) {
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
        ctx: state.sup.ps().into_iter().find(|p| p.name == model).map(|p| p.ctx),
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

/// The observation layer: ring + bounded caches + per-request analyzers.
pub struct Sentinel {
    enabled: bool,
    stall: Duration,
    ring: Mutex<VecDeque<SentinelRecord>>,
    schemas: Mutex<HashMap<String, Option<Arc<jsonschema::Validator>>>>,
    templates: Mutex<HashMap<String, TemplateSupport>>,
    /// Bounded JSONL at `<run_dir>/sentinel.jsonl`; `None` = memory-only
    /// (no run dir, or open failed — IO problems never take the daemon).
    persist: Option<Mutex<PersistState>>,
    persist_path: Option<std::path::PathBuf>,
}

impl Sentinel {
    #[must_use]
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
        Arc::new(Self {
            enabled,
            stall: Duration::from_secs(stall_secs),
            ring: Mutex::new(ring),
            schemas: Mutex::new(HashMap::new()),
            templates: Mutex::new(HashMap::new()),
            persist,
            persist_path,
        })
    }

    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.enabled
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
        SentinelFeed { tx: Some(tx), degraded }
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
            Some(t) => ring.iter().rev().filter(|r| r.trace == t).cloned().collect(),
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
                        carry.push_str(&String::from_utf8_lossy(&b));
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
                Some(FeedEvent::Value(v)) => acc.apply(&v),
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
        }
        {
            let mut ring = self.ring.lock();
            if ring.len() == RING_CAP {
                ring.pop_front();
            }
            ring.push_back(record.clone());
        }
        self.persist_record(record);
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
                    written = written.saturating_add(u64::try_from(l.len()).unwrap_or(u64::MAX) + 1);
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
                    detail: format!("{} has no embedded chat template; tools cannot be rendered", ctx.model),
                }),
                Some(TemplateSupport::NoTools) => out.push(Detection {
                    code: Code::TemplateNoTools,
                    detail: format!("{}'s chat template contains no tool markers", ctx.model),
                }),
                _ => {}
            }
        }

        if acc.finish.as_deref() == Some("length") {
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
                        detail: format!("{} reasoning chars, zero answer content", acc.reasoning.len()),
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
                    detail: format!("`{}` not in request tools [{}]", frag.name, ctx.tool_names.join(", ")),
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
                    detail: format!("tool `{}` args not valid JSON: {e}: {}", frag.name, preview(&frag.args, 120)),
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
        let content = if acc.content.is_empty() { &acc.text } else { &acc.content };
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
        detections,
        prompt_tokens: acc.usage_prompt,
        completion_tokens: acc.usage_completion,
        ctx: ctx.ctx,
        degraded,
        ms: request_ms(&ctx.trace, started),
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
            if let Some(s) = carrier.get("text").and_then(Value::as_str) {
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
                                if entry.args.is_empty() && !a.is_empty() {
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
                    self.finish =
                        Some(if reason == "max_output_tokens" { "length" } else { "stop" }.into());
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
                        if let Some(a) = item.get("arguments").and_then(Value::as_str) {
                            entry.args.push_str(a);
                        }
                    }
                    Some("message") => {
                        if let Some(c) = item.get("content") {
                            if let Some(s) = c.as_str() {
                                Self::push_bounded(&mut self.content, &mut self.degraded, s);
                            } else if let Some(parts) = c.as_array() {
                                for p in parts {
                                    if let Some(t) = p.get("text").and_then(Value::as_str) {
                                        Self::push_bounded(&mut self.content, &mut self.degraded, t);
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
        assert!((59_000..=61_500).contains(&v), "aged trace decoded to {v}ms");
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
            detections: vec![Detection { code: Code::CtxTruncated, detail: "d".into() }],
            prompt_tokens: None,
            completion_tokens: None,
            ctx: None,
            degraded: false,
            ms: 1,
        };
        assert!(rec.to_json()["detections"][0]["retry"].as_str().is_some_and(|s| !s.is_empty()));
    }

    #[test]
    fn unit__template_has_tools__markers() {
        assert!(template_has_tools("{%- if tools %}{{ tool_calls }}{%- endif %}"));
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
        let ctx = RequestCtx { ctx: Some(128), ..RequestCtx::default() };
        let d = s.finalize(&ctx, &acc, 200);
        assert!(d.iter().any(|x| x.code == Code::CtxTruncated), "{d:?}");
        assert!(d.iter().any(|x| x.code == Code::CtxNearLimit), "{d:?}");
    }

    #[test]
    fn unit__finalize__tool_fragment_checks() {
        let s = Sentinel::new(true, 0, None);
        let mut acc = Accum { saw_any_choice: true, ..Accum::default() };
        acc.tools.insert(0, ToolFrag { name: "get_weather".into(), args: "{\"city\"".into() });
        acc.tools.insert(1, ToolFrag { name: "hallucinated".into(), args: "{}".into() });
        let ctx = ctx_with(&["get_weather"]);
        let d = s.finalize(&ctx, &acc, 200);
        assert!(d.iter().any(|x| x.code == Code::ToolArgsInvalidJson), "{d:?}");
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
            response_format: Some(json!({"type": "json_schema", "schema": {"type": "object", "properties": {"answer": {"type": "string"}}}})),
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
        let mut acc = Accum { saw_any_choice: true, ..Accum::default() };
        let d = s.finalize(&RequestCtx::default(), &acc, 200);
        assert!(d.iter().any(|x| x.code == Code::EmptyResponse), "{d:?}");

        acc.reasoning = "thinking hard".into();
        let d = s.finalize(&RequestCtx::default(), &acc, 200);
        assert!(d.iter().any(|x| x.code == Code::ReasoningNoAnswer), "{d:?}");
        assert!(!d.iter().any(|x| x.code == Code::EmptyResponse), "{d:?}");
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
    fn unit__why__ring_filter_and_limit() {
        let s = Sentinel::new(true, 0, None);
        let rec = |trace: &str| SentinelRecord {
            trace: trace.into(),
            ts: 1,
            route: "openai-chat".into(),
            model: "m".into(),
            status: 200,
            stream: true,
            detections: vec![],
            prompt_tokens: None,
            completion_tokens: None,
            ctx: None,
            degraded: false,
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
}
