//! Engine-side tokenization, in each lane's own dialect.
//!
//! The gateway holds no tokenizer of its own - every model ships one
//! inside its engine. Lanes that expose it over HTTP get exact counts
//! (prompt-fit admission, `/v1/tokenize`, `/v1/detokenize` and the
//! ollama-compat `/api/tokenize`). The request and answer shapes differ
//! per engine, so this module is the one place that knows them:
//!
//! | lane       | tokenize request                 | token ids  | detokenize request                 | text      |
//! |------------|----------------------------------|------------|------------------------------------|-----------|
//! | `llama.cpp` | `POST /tokenize {content}`       | `tokens[]` | `POST /detokenize {tokens}`        | `content` |
//! | `vLLM`     | `POST /tokenize {model, prompt}` | `tokens[]` | `POST /detokenize {model, tokens}` | `prompt`  |
//! | `SGLang`   | `POST /tokenize {model, prompt}` | `tokens[]` | `POST /detokenize {model, tokens}` | `text`    |
//!
//! mlx-lm and mistral.rs serve no tokenize route (verified against the
//! installed mlx-lm 0.32.2 and mistral.rs 0.9.5): callers get a teaching
//! error there instead of a silent zero.

use axum::Json;
use axum::body::Bytes;
use axum::extract::State;
use axum::response::{IntoResponse, Response};
use blazar_core::engine_kind::EngineKind;
use blazar_runtime::EngineRef;
use serde_json::{Value, json};
use std::sync::Arc;

use crate::state::AppState;

/// Engines whose HTTP surface includes tokenize and detokenize routes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dialect {
    LlamaCpp,
    Vllm,
    Sglang,
}

impl Dialect {
    fn of(kind: EngineKind) -> Option<Self> {
        match kind {
            EngineKind::LlamaCpp => Some(Self::LlamaCpp),
            EngineKind::Vllm => Some(Self::Vllm),
            EngineKind::Sglang => Some(Self::Sglang),
            _ => None,
        }
    }
}

/// Why an exact count could not be had.
#[derive(Debug)]
pub enum TokenizeError {
    /// The lane's engine serves no tokenize route (names the engine).
    Unsupported(&'static str),
    /// The engine answered, but not with a usable result.
    Engine(String),
}

impl std::fmt::Display for TokenizeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported(engine) => write!(
                f,
                "the {engine} engine exposes no tokenize route - exact token \
                 counts are available on the llama.cpp, vLLM and SGLang lanes"
            ),
            Self::Engine(message) => write!(f, "engine tokenize failed: {message}"),
        }
    }
}

/// What is being converted: text into ids, or ids into text.
pub enum Input<'a> {
    Text(&'a str),
    Tokens(&'a [u64]),
}

/// The HTTP request a lane expects: its path and JSON body.
#[must_use]
pub fn build_request(dialect: Dialect, model: &str, input: &Input<'_>) -> (&'static str, Value) {
    match (dialect, input) {
        (Dialect::LlamaCpp, Input::Text(text)) => ("/tokenize", json!({ "content": text })),
        (Dialect::Vllm | Dialect::Sglang, Input::Text(text)) => {
            ("/tokenize", json!({ "model": model, "prompt": text }))
        }
        (Dialect::LlamaCpp, Input::Tokens(tokens)) => ("/detokenize", json!({ "tokens": tokens })),
        (Dialect::Vllm | Dialect::Sglang, Input::Tokens(tokens)) => {
            ("/detokenize", json!({ "model": model, "tokens": tokens }))
        }
    }
}

/// Token ids in a tokenize answer (`None` when the engine only counted).
#[must_use]
pub fn parse_tokens(answer: &Value) -> Option<Vec<u64>> {
    answer
        .get("tokens")
        .and_then(Value::as_array)?
        .iter()
        .map(Value::as_u64)
        .collect()
}

/// Token count in a tokenize answer: the id list's length, else the
/// engine's own `count` (vLLM and `SGLang` may omit the ids).
#[must_use]
pub fn parse_count(answer: &Value) -> Option<u64> {
    let ids = answer
        .get("tokens")
        .and_then(Value::as_array)
        .map(|ids| ids.iter().filter(|id| id.is_u64()).count() as u64);
    ids.or_else(|| answer.get("count").and_then(Value::as_u64))
}

/// The text in a detokenize answer, under the lane's own key.
#[must_use]
pub fn parse_text(dialect: Dialect, answer: &Value) -> Option<String> {
    let key = match dialect {
        Dialect::LlamaCpp => "content",
        Dialect::Vllm => "prompt",
        Dialect::Sglang => "text",
    };
    answer.get(key).and_then(Value::as_str).map(str::to_string)
}

fn dialect_of(engine: &EngineRef) -> Result<Dialect, TokenizeError> {
    Dialect::of(engine.kind).ok_or(TokenizeError::Unsupported(engine.kind.as_str()))
}

/// The model id a lane matches in request bodies: vLLM and `SGLang`
/// compare it with the name they were started with.
fn served_name(engine: &EngineRef) -> String {
    crate::proxy::child_model_stamp(engine).unwrap_or_else(|| engine.name.clone())
}

async fn call(
    state: &AppState,
    engine: &EngineRef,
    path: &str,
    payload: &Value,
) -> Result<Value, TokenizeError> {
    let base = crate::proxy::child_base(&engine.endpoint);
    let response = crate::proxy::child_auth(
        crate::state::child_client(state, &engine.endpoint).post(format!("{base}{path}")),
        engine,
    )
    .json(payload)
    .send()
    .await
    .map_err(|e| TokenizeError::Engine(format!("request failed: {e}")))?;
    let status = response.status();
    if !status.is_success() {
        let text = response.text().await.unwrap_or_default();
        return Err(TokenizeError::Engine(format!(
            "HTTP {status}: {}",
            text.chars().take(300).collect::<String>()
        )));
    }
    response
        .json()
        .await
        .map_err(|e| TokenizeError::Engine(format!("unreadable answer: {e}")))
}

/// The resident child serving `model`, when one is up.
#[must_use]
pub fn resident(state: &AppState, model: &str) -> Option<EngineRef> {
    state
        .sup
        .live_http_endpoints()
        .into_iter()
        .find(|engine| engine.name == model)
}

/// Exact token count of `text` under the model's own tokenizer.
pub async fn count_prompt(
    state: &AppState,
    engine: &EngineRef,
    text: &str,
) -> Result<u64, TokenizeError> {
    let dialect = dialect_of(engine)?;
    let (path, payload) = build_request(dialect, &served_name(engine), &Input::Text(text));
    let answer = call(state, engine, path, &payload).await?;
    parse_count(&answer).ok_or_else(|| TokenizeError::Engine("no token count in the answer".into()))
}

/// Token ids of `text` under the model's own tokenizer.
pub async fn tokenize_text(
    state: &AppState,
    engine: &EngineRef,
    text: &str,
) -> Result<Vec<u64>, TokenizeError> {
    let dialect = dialect_of(engine)?;
    let (path, payload) = build_request(dialect, &served_name(engine), &Input::Text(text));
    let answer = call(state, engine, path, &payload).await?;
    parse_tokens(&answer).ok_or_else(|| TokenizeError::Engine("no token ids in the answer".into()))
}

/// The text `tokens` decode to under the model's own tokenizer.
pub async fn detokenize_ids(
    state: &AppState,
    engine: &EngineRef,
    tokens: &[u64],
) -> Result<String, TokenizeError> {
    let dialect = dialect_of(engine)?;
    let (path, payload) = build_request(dialect, &served_name(engine), &Input::Tokens(tokens));
    let answer = call(state, engine, path, &payload).await?;
    parse_text(dialect, &answer)
        .ok_or_else(|| TokenizeError::Engine("no text in the answer".into()))
}

fn bad_request(message: impl Into<String>) -> Response {
    crate::proxy::openai_error(400, &message.into())
}

fn engine_error(error: TokenizeError) -> Response {
    match error {
        unsupported @ TokenizeError::Unsupported(_) => {
            crate::proxy::openai_error(501, &unsupported.to_string())
        }
        failed => crate::proxy::openai_error(502, &failed.to_string()),
    }
}

fn request_model(request: &Value) -> Option<&str> {
    request
        .get("model")
        .and_then(Value::as_str)
        .filter(|model| !model.is_empty())
}

fn request_input<'a>(request: &'a Value, name: &str) -> Option<&'a Value> {
    request.get(name).or_else(|| request.get("input"))
}

/// Resolve `model` and find its resident child: 404 for a name nothing
/// answers to, 503 when the model is known but not loaded (counting
/// through a cold model would silently pay its full load).
#[allow(clippy::result_large_err)] // Response is the handlers' currency here
fn resident_engine(state: &AppState, model: &str) -> Result<EngineRef, Response> {
    let row = match crate::proxy::resolve_model_for_handler(state, model) {
        Ok(Ok(row)) => row,
        Ok(Err(message)) => return Err(crate::proxy::openai_error(404, &message)),
        Err(message) => return Err(crate::proxy::openai_error(503, &message)),
    };
    match resident(state, &row.name) {
        Some(engine) => Ok(engine),
        None => Err(crate::proxy::openai_error(
            503,
            &format!(
                "model {:?} is not running - the tokenizer lives inside its engine; warm it \
                 first with a chat request or `blazar warm {}`",
                row.name, row.name
            ),
        )),
    }
}

/// `prompt` (or its `input` alias): one text or a batch. Also reports
/// whether the caller sent a batch, so the answer mirrors the shape.
fn texts_of(request: &Value) -> Result<(Vec<String>, bool), String> {
    let Some(input) = request_input(request, "prompt") else {
        return Err("missing `prompt` field (a string or an array of strings)".into());
    };
    match input {
        Value::String(text) => Ok((vec![text.clone()], false)),
        Value::Array(items) => items
            .iter()
            .map(|item| {
                item.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| "every `prompt` entry must be a string".to_string())
            })
            .collect::<Result<Vec<_>, _>>()
            .map(|texts| (texts, true)),
        _ => Err("`prompt` must be a string or an array of strings".into()),
    }
}

fn ids_of(request: &Value) -> Result<(Vec<Vec<u64>>, bool), String> {
    let Some(input) = request.get("tokens") else {
        return Err("missing `tokens` field (an array of token ids)".into());
    };
    let ids = |value: &Value| {
        value
            .as_array()
            .ok_or_else(|| "`tokens` must be an array of token ids".to_string())?
            .iter()
            .map(|id| {
                id.as_u64()
                    .ok_or_else(|| "every token id must be a non-negative integer".to_string())
            })
            .collect::<Result<Vec<_>, _>>()
    };
    match input {
        Value::Array(items) if items.iter().any(Value::is_array) => items
            .iter()
            .map(ids)
            .collect::<Result<Vec<_>, _>>()
            .map(|all| (all, true)),
        _ => ids(input).map(|one| (vec![one], false)),
    }
}

/// POST /v1/tokenize - token ids for text, from the live engine's own
/// tokenizer.
///
/// Body: `{"model": "...", "prompt": "..."}` with a string or an array of
/// strings (`input` is accepted as an alias). Answer:
/// `{"tokens": [id, ...], "count": N}`, or parallel arrays for a batch.
pub async fn v1_tokenize(State(state): State<Arc<AppState>>, body: Bytes) -> Response {
    let request: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return bad_request(format!("invalid JSON: {e}")),
    };
    let Some(model) = request_model(&request) else {
        return bad_request("missing `model` field");
    };
    let (texts, batch) = match texts_of(&request) {
        Ok(shape) => shape,
        Err(message) => return bad_request(message),
    };
    let engine = match resident_engine(&state, model) {
        Ok(engine) => engine,
        Err(response) => return response,
    };
    let mut all_ids = Vec::with_capacity(texts.len());
    for text in &texts {
        match tokenize_text(&state, &engine, text).await {
            Ok(ids) => all_ids.push(ids),
            Err(error) => return engine_error(error),
        }
    }
    let answer = if batch {
        let counts: Vec<usize> = all_ids.iter().map(Vec::len).collect();
        json!({ "tokens": all_ids, "count": counts })
    } else {
        let ids = all_ids.first().cloned().unwrap_or_default();
        let count = ids.len();
        json!({ "tokens": ids, "count": count })
    };
    Json(answer).into_response()
}

/// POST /v1/detokenize - the text behind token ids, from the live
/// engine's own tokenizer.
///
/// Body: `{"model": "...", "tokens": [id, ...]}` with one array or an
/// array of arrays. Answer: `{"prompt": "..."}`, or an array for a batch.
pub async fn v1_detokenize(State(state): State<Arc<AppState>>, body: Bytes) -> Response {
    let request: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return bad_request(format!("invalid JSON: {e}")),
    };
    let Some(model) = request_model(&request) else {
        return bad_request("missing `model` field");
    };
    let (all_ids, batch) = match ids_of(&request) {
        Ok(shape) => shape,
        Err(message) => return bad_request(message),
    };
    let engine = match resident_engine(&state, model) {
        Ok(engine) => engine,
        Err(response) => return response,
    };
    let mut texts = Vec::with_capacity(all_ids.len());
    for ids in &all_ids {
        match detokenize_ids(&state, &engine, ids).await {
            Ok(text) => texts.push(text),
            Err(error) => return engine_error(error),
        }
    }
    let answer = if batch {
        json!({ "prompt": texts })
    } else {
        json!({ "prompt": texts.first().cloned().unwrap_or_default() })
    };
    Json(answer).into_response()
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    fn ids(v: &[u64]) -> Vec<u64> {
        v.to_vec()
    }

    #[test]
    fn unit__build_request__speaks_each_lane_dialect() {
        assert_eq!(
            build_request(Dialect::LlamaCpp, "m", &Input::Text("hi")),
            ("/tokenize", json!({ "content": "hi" }))
        );
        assert_eq!(
            build_request(Dialect::Vllm, "served-name", &Input::Text("hi")),
            (
                "/tokenize",
                json!({ "model": "served-name", "prompt": "hi" })
            )
        );
        assert_eq!(
            build_request(Dialect::Sglang, "served-name", &Input::Text("hi")).1,
            json!({ "model": "served-name", "prompt": "hi" })
        );
        assert_eq!(
            build_request(Dialect::LlamaCpp, "m", &Input::Tokens(&ids(&[2, 3]))),
            ("/detokenize", json!({ "tokens": [2, 3] }))
        );
        assert_eq!(
            build_request(Dialect::Sglang, "served-name", &Input::Tokens(&ids(&[2]))),
            (
                "/detokenize",
                json!({ "model": "served-name", "tokens": [2] })
            )
        );
        // llama.cpp never hears a model field: it serves exactly one.
        assert!(
            build_request(Dialect::LlamaCpp, "ignored", &Input::Text("x"))
                .1
                .get("model")
                .is_none()
        );
    }

    #[test]
    fn unit__parse_answers__reads_ids_counts_and_each_text_key() {
        assert_eq!(parse_tokens(&json!({ "tokens": [7, 8] })), Some(vec![7, 8]));
        assert_eq!(parse_tokens(&json!({ "count": 2 })), None);
        assert_eq!(parse_tokens(&json!({ "tokens": [7, "x"] })), None);
        assert_eq!(parse_count(&json!({ "tokens": [7, 8, 9] })), Some(3));
        assert_eq!(parse_count(&json!({ "count": 12 })), Some(12));
        assert_eq!(parse_count(&json!({})), None);
        assert_eq!(
            parse_text(Dialect::LlamaCpp, &json!({ "content": "a bb" })),
            Some("a bb".to_string())
        );
        assert_eq!(
            parse_text(Dialect::Vllm, &json!({ "prompt": "a bb" })),
            Some("a bb".to_string())
        );
        assert_eq!(
            parse_text(Dialect::Sglang, &json!({ "text": "a bb" })),
            Some("a bb".to_string())
        );
        // A lane does not read another lane's key.
        assert_eq!(
            parse_text(Dialect::Sglang, &json!({ "content": "a" })),
            None
        );
    }

    #[test]
    fn unit__dialect_of__only_lanes_with_a_tokenize_route() {
        let engine = |kind| EngineRef {
            name: "m".into(),
            key: "m".into(),
            kind,
            endpoint: blazar_core::Endpoint::Tcp {
                host: "127.0.0.1".into(),
                port: 1,
            },
            auth: None,
            model_path: String::new(),
        };
        assert_eq!(
            dialect_of(&engine(EngineKind::LlamaCpp)).ok(),
            Some(Dialect::LlamaCpp)
        );
        assert_eq!(
            dialect_of(&engine(EngineKind::Vllm)).ok(),
            Some(Dialect::Vllm)
        );
        assert_eq!(
            dialect_of(&engine(EngineKind::Sglang)).ok(),
            Some(Dialect::Sglang)
        );
        assert!(dialect_of(&engine(EngineKind::Mlx)).is_err());
        assert!(dialect_of(&engine(EngineKind::MistralRs)).is_err());
        assert!(
            TokenizeError::Unsupported("mlx")
                .to_string()
                .contains("the mlx engine exposes no tokenize route")
        );
    }

    #[test]
    fn unit__request_shapes__accept_single_and_batch_and_name_malformed() {
        assert_eq!(
            texts_of(&json!({ "prompt": "a" })),
            Ok((vec!["a".to_string()], false))
        );
        assert_eq!(
            texts_of(&json!({ "input": ["a", "b"] })),
            Ok((vec!["a".to_string(), "b".to_string()], true))
        );
        assert_eq!(
            texts_of(&json!({ "prompt": [1] })),
            Err("every `prompt` entry must be a string".to_string())
        );
        assert_eq!(
            texts_of(&json!({})),
            Err("missing `prompt` field (a string or an array of strings)".to_string())
        );
        assert_eq!(
            ids_of(&json!({ "tokens": [1, 2] })),
            Ok((vec![vec![1, 2]], false))
        );
        assert_eq!(
            ids_of(&json!({ "tokens": [[1], [2, 3]] })),
            Ok((vec![vec![1], vec![2, 3]], true))
        );
        assert_eq!(
            ids_of(&json!({ "tokens": [1, -2] })),
            Err("every token id must be a non-negative integer".to_string())
        );
        assert_eq!(
            ids_of(&json!({})),
            Err("missing `tokens` field (an array of token ids)".to_string())
        );
        assert_eq!(request_model(&json!({ "model": "" })), None);
        assert_eq!(request_model(&json!({ "model": "m" })), Some("m"));
    }
}
