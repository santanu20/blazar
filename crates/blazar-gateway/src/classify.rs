//! `POST /v1/classify` — classification as a first-class lane.
//!
//! Decision models (modern-bert classifiers, e.g. Laya-class system-one
//! scorers) answer with a label distribution, not prose. Upstream's
//! surface is llama-server's `/v1/systemone` — a Q&A shape tied to one
//! engine dialect. This lane normalizes it behind a stable contract:
//! `{model, input, labels[]}` in, `{labels: [{label, probability}],
//! top, confidence}` out. Translation rides `openai_proxy`, so
//! admission, routing, the llamacpp-lane gate, child auth and retries
//! are all the proven pipeline — classify only reshapes the two ends.

use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, Method, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use serde_json::Value;

use crate::state::AppState;

/// Maximum labels in one request — the child renders every criterion
/// into the prompt; a bound keeps a typo'd unbounded array from
/// becoming a context bomb.
pub const CLASSIFY_MAX_LABELS: usize = 16;

/// Pure: `{model, input, labels[]}` -> a `/v1/systemone` body. `None`
/// when the shape is unusable (missing model/input, <2 or >16 labels,
/// duplicate/empty labels). Criteria keys carry the labels verbatim —
/// the child's response keys them back by the same strings.
#[must_use]
pub fn classify_request_to_systemone(v: &Value) -> Option<Value> {
    let model = v.get("model")?.as_str()?.to_string();
    if model.is_empty() {
        return None;
    }
    let input = v
        .get("input")
        .and_then(|i| i.as_str())
        .map_or_else(|| v.get("state").and_then(|s| s.as_str()), Some)?
        .to_string();
    let labels: Vec<String> = v
        .get("labels")?
        .as_array()?
        .iter()
        .map(|l| l.as_str().map(str::to_string))
        .collect::<Option<Vec<_>>>()?;
    if labels.len() < 2 || labels.len() > CLASSIFY_MAX_LABELS {
        return None;
    }
    if labels.iter().any(|l| l.trim().is_empty())
        || labels.len()
            != labels
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
    {
        return None;
    }
    let mut criteria = serde_json::Map::new();
    for l in &labels {
        criteria.insert(l.clone(), Value::String(l.clone()));
    }
    Some(serde_json::json!({
        "model": model,
        "state": input,
        "questions": {
            "q1": {
                "type": "choice",
                "instructions": "Classify the state into exactly one of the criteria labels.",
                "criteria": Value::Object(criteria),
            }
        }
    }))
}

/// Pure: a `/v1/systemone` response -> the normalized classify card.
/// `None` when the answer shape is not a q1 choice (passthrough then
/// shows the caller the engine's own answer verbatim).
#[must_use]
pub fn systemone_to_classify(v: &Value) -> Option<Value> {
    let answer = v.get("answers")?.get("q1")?;
    let choice = answer.get("choice")?.as_str()?.to_string();
    let mut rows: Vec<(String, f64)> = answer
        .get("probabilities")?
        .as_object()?
        .iter()
        .filter_map(|(k, p)| Some((k.clone(), p.as_f64()?)))
        .collect();
    rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let confidence = answer.get("confidence").and_then(Value::as_f64);
    let labels: Vec<Value> = rows
        .iter()
        .map(|(l, p)| serde_json::json!({"label": l, "probability": p}))
        .collect();
    Some(serde_json::json!({
        "object": "blazar.classify",
        "model": v.get("model").cloned().unwrap_or(Value::Null),
        "labels": labels,
        "top": choice,
        "confidence": confidence,
    }))
}

/// `POST /v1/classify` handler: translate in, ride the systemone
/// pipeline, normalize out. Non-200 child answers pass through
/// verbatim (the engine's teaching is better than a generic error).
pub async fn classify(State(state): State<Arc<AppState>>, body: Bytes) -> Response {
    let parsed: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => {
            return crate::error_response(400, &format!("invalid JSON: {e}"));
        }
    };
    let Some(systemone_body) = classify_request_to_systemone(&parsed) else {
        return crate::error_response(
            400,
            &format!(
                "classify needs model + input (or state) and 2..={CLASSIFY_MAX_LABELS} unique non-empty labels"
            ),
        );
    };
    let bytes = serde_json::to_vec(&systemone_body).unwrap_or_default();
    let mut headers = HeaderMap::new();
    headers.insert(
        "content-type",
        axum::http::HeaderValue::from_static("application/json"),
    );
    let resp = crate::openai::openai_proxy(
        State(state),
        None,
        None,
        Uri::from_static("/v1/systemone"),
        Method::POST,
        headers,
        Bytes::from(bytes),
    )
    .await;
    if resp.status() != StatusCode::OK {
        return resp;
    }
    let (parts, body) = resp.into_parts();
    let raw = axum::body::to_bytes(body, 8 * 1024 * 1024)
        .await
        .unwrap_or_default();
    match serde_json::from_slice::<Value>(&raw)
        .ok()
        .and_then(|v| systemone_to_classify(&v))
    {
        Some(card) => (parts.status, axum::Json(card)).into_response(),
        // Unrecognizable-but-200 answer: hand the raw engine payload
        // through instead of guessing at it.
        None => (parts.status, raw).into_response(),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    #[test]
    #[allow(non_snake_case)] // unit__<x>__<y> double-underscore convention
    fn unit__classify_request_to_systemone__maps_labels_to_criteria() {
        let body = json!({
            "model": "laya",
            "input": "the user asked for a summary",
            "labels": ["safe", "unsafe"]
        });
        let out = super::classify_request_to_systemone(&body).expect("2 labels map");
        assert_eq!(out["model"], "laya");
        assert_eq!(out["state"], "the user asked for a summary");
        let q1 = &out["questions"]["q1"];
        assert_eq!(q1["type"], "choice");
        assert_eq!(q1["criteria"]["safe"], "safe");
        assert_eq!(q1["criteria"]["unsafe"], "unsafe");

        // `state` accepted as the input alias (systemone vocabulary).
        let aliased = json!({"model": "m", "state": "s", "labels": ["a", "b"]});
        assert_eq!(
            super::classify_request_to_systemone(&aliased).unwrap()["state"],
            "s"
        );

        // Rejections: one label, 17 labels, duplicates, empties.
        assert!(
            super::classify_request_to_systemone(
                &json!({"model":"m","input":"s","labels":["only"]})
            )
            .is_none()
        );
        let many = json!({"model":"m","input":"s","labels": (0..17).map(|i| i.to_string()).collect::<Vec<_>>()});
        assert!(super::classify_request_to_systemone(&many).is_none());
        assert!(
            super::classify_request_to_systemone(
                &json!({"model":"m","input":"s","labels":["a","a"]})
            )
            .is_none()
        );
        assert!(
            super::classify_request_to_systemone(
                &json!({"model":"m","input":"s","labels":["a",""]})
            )
            .is_none()
        );
        assert!(
            super::classify_request_to_systemone(
                &json!({"model":"","input":"s","labels":["a","b"]})
            )
            .is_none()
        );
    }

    #[test]
    #[allow(non_snake_case)] // unit__<x>__<y> double-underscore convention
    fn unit__systemone_to_classify__normalizes_sorted_probabilities() {
        let live = json!({
            "model": "laya",
            "answers": {"q1": {"type": "choice", "choice": "no",
                               "probabilities": {"yes": 0.123, "no": 0.877},
                               "confidence": 0.75}},
            "usage": {"input_tokens": 38, "output_tokens": 0}
        });
        let card = super::systemone_to_classify(&live).expect("live shape normalizes");
        assert_eq!(card["object"], "blazar.classify");
        assert_eq!(card["top"], "no");
        assert_eq!(card["confidence"], 0.75);
        let labels = card["labels"].as_array().unwrap();
        assert_eq!(labels.len(), 2);
        // Descending order: the winner leads.
        assert_eq!(labels[0]["label"], "no");
        assert_eq!(labels[0]["probability"], 0.877);
        assert_eq!(labels[1]["label"], "yes");

        // Non-choice answers do not normalize (caller sees raw payload).
        assert!(super::systemone_to_classify(&json!({"answers": {}})).is_none());
        assert!(super::systemone_to_classify(&json!({})).is_none());
    }
}
