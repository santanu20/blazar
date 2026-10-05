//! Completion metadata cards — the id→metadata mapping behind
//! `POST /v1/chat/completions/{id}` (`OpenAI` spec v2.3.0 parity).
//!
//! Contract (documented in `docs/4.API_SPEC.md`):
//! - A create carrying a top-level `metadata` object stores a slim card
//!   (id, model, created, metadata) keyed by the child's completion id.
//! - Streaming completions cannot be carded: bytes are on the wire
//!   before the id exists, so an update for them 404s with teaching.
//! - Cards live in the v12 `completion_cards` table, bounded by a
//!   oldest-first cap (store side).

use crate::state::AppState;
use std::sync::Arc;

/// `OpenAI`'s own limits for completion metadata: at most 16 keys, every
/// value a string of at most 512 characters.
pub const METADATA_MAX_KEYS: usize = 16;
pub const METADATA_MAX_VALUE_CHARS: usize = 512;

/// Validate a request/update `metadata` object against the `OpenAI`
/// limits. Returns the teaching error on violation.
pub fn validate_metadata(v: &serde_json::Value) -> Result<(), String> {
    let Some(obj) = v.as_object() else {
        return Err("metadata must be a JSON object of string keys to string values".into());
    };
    if obj.len() > METADATA_MAX_KEYS {
        return Err(format!(
            "metadata carries {} keys; the limit is {METADATA_MAX_KEYS}",
            obj.len()
        ));
    }
    for (k, val) in obj {
        let Some(s) = val.as_str() else {
            return Err(format!(
                "metadata value for key {k:?} must be a string (numbers and nested objects are not stored)"
            ));
        };
        if s.chars().count() > METADATA_MAX_VALUE_CHARS {
            return Err(format!(
                "metadata value for key {k:?} is {} chars; the limit is {METADATA_MAX_VALUE_CHARS}",
                s.chars().count()
            ));
        }
    }
    Ok(())
}

/// Derive the slim card from the request parse + buffered response
/// body. `None` = no card (request carried no metadata, response was
/// unparseable, or the metadata shape is not storable).
#[must_use]
pub fn card_from_response(
    model: &str,
    req: Option<&serde_json::Value>,
    resp_body: &[u8],
) -> Option<blazar_core::CompletionCardRow> {
    let req = req?;
    let meta = req
        .get("metadata")
        .filter(|m| !m.as_object().is_some_and(serde_json::Map::is_empty))?;
    // Create-time validation normally rejected bad shapes at admission;
    // a card written by a lane that skipped that gate still must not
    // store unbounded junk.
    if let Err(e) = validate_metadata(meta) {
        tracing::warn!(target: "blazar::metadata_card", "not carding completion: {e}");
        return None;
    }
    let v: serde_json::Value = serde_json::from_slice(resp_body).ok()?;
    let id = v.get("id").and_then(serde_json::Value::as_str)?;
    if id.is_empty() {
        tracing::warn!(
            target: "blazar::metadata_card",
            "not carding completion: response carries no id"
        );
        return None;
    }
    Some(blazar_core::CompletionCardRow {
        id: id.to_string(),
        model: model.to_string(),
        created: v
            .get("created")
            .and_then(serde_json::Value::as_i64)
            .unwrap_or_else(|| crate::responses::unix_now().cast_signed()),
        metadata_json: meta.to_string(),
        updated_at: crate::responses::unix_now().cast_signed(),
    })
}

/// Persist a slim card from the buffered NON-STREAM chat completion
/// response, but only when the request asked for metadata. Called from
/// the proxy's buffered branch (same site as usage classification), so
/// the request parse and the response bytes are already in hand.
///
/// Failure policy: a malformed response body or a store error logs and
/// drops the card — the completion itself already succeeded, and
/// failing the client retroactively over a bookkeeping write would
/// charge tokens twice on retry (H1 applies to the request, not the
/// side record).
pub fn persist_from_response(
    state: &Arc<AppState>,
    model: &str,
    req: Option<&serde_json::Value>,
    resp_body: &[u8],
) {
    let Some(card) = card_from_response(model, req, resp_body) else {
        return;
    };
    let guard = state
        .store
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(store) = guard.as_ref() else {
        tracing::warn!(
            target: "blazar::metadata_card",
            "card skipped: store unavailable (blazar serve without a writable data dir)"
        );
        return;
    };
    if let Err(e) = store.put_completion_card(&card) {
        tracing::warn!(target: "blazar::metadata_card", "card write failed: {e}");
    }
}

/// The JSON body of a successful update: the stored card shape, the
/// same fields the create response carried.
#[must_use]
pub fn card_to_json(card: &blazar_core::CompletionCardRow) -> serde_json::Value {
    let metadata: serde_json::Value =
        serde_json::from_str(&card.metadata_json).unwrap_or(serde_json::json!({}));
    serde_json::json!({
        "id": card.id,
        "object": "chat.completion",
        "created": card.created,
        "model": card.model,
        "metadata": metadata,
    })
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]
    use super::*;

    #[test]
    fn unit__validate_metadata__accepts_flat_string_map() {
        assert!(validate_metadata(&serde_json::json!({"batch": "b-1", "run": "42"})).is_ok());
    }

    #[test]
    fn unit__validate_metadata__rejects_non_object_and_non_string_values() {
        let e = validate_metadata(&serde_json::json!("tagged"));
        assert!(e.is_err() && e.unwrap_err().contains("must be a JSON object"));
        let e = validate_metadata(&serde_json::json!({"n": 5}));
        assert!(e.is_err() && e.unwrap_err().contains("must be a string"));
        let e = validate_metadata(&serde_json::json!({"nested": {"a": 1}}));
        assert!(e.is_err() && e.unwrap_err().contains("must be a string"));
    }

    #[test]
    fn unit__validate_metadata__key_count_and_value_length_limits() {
        use serde_json::Value;
        let mut full = serde_json::Map::new();
        for i in 0..METADATA_MAX_KEYS {
            full.insert(format!("k{i}"), serde_json::json!("v"));
        }
        assert!(validate_metadata(&Value::Object(full.clone())).is_ok());
        full.insert("one-too-many".into(), serde_json::json!("v"));
        let e = validate_metadata(&Value::Object(full));
        assert!(e.is_err() && e.unwrap_err().contains("the limit is"));

        let long = "x".repeat(METADATA_MAX_VALUE_CHARS + 1);
        let e = validate_metadata(&serde_json::json!({"big": long}));
        assert!(e.is_err() && e.unwrap_err().contains("the limit is"));
    }

    #[test]
    fn unit__card_to_json__stored_shape() {
        let card = blazar_core::CompletionCardRow {
            id: "chatcmpl-1".into(),
            model: "m".into(),
            created: 7,
            metadata_json: r#"{"a":"b"}"#.into(),
            updated_at: 9,
        };
        let v = card_to_json(&card);
        assert_eq!(v["id"], "chatcmpl-1");
        assert_eq!(v["object"], "chat.completion");
        assert_eq!(v["metadata"]["a"], "b");
    }

    #[test]
    fn unit__card_from_response__no_metadata_no_card_and_bad_body_dropped() {
        // No request metadata: nothing happens — the common case is free.
        assert!(
            card_from_response(
                "m",
                Some(&serde_json::json!({"model": "m"})),
                br#"{"id":"c1"}"#
            )
            .is_none()
        );
        // Metadata present but body unparseable: card silently dropped
        // (the completion already succeeded; bookkeeping never fails it).
        assert!(
            card_from_response(
                "m",
                Some(&serde_json::json!({"model": "m", "metadata": {"t": "1"}})),
                b"not json"
            )
            .is_none()
        );
        // Metadata present, body has no id: dropped.
        assert!(
            card_from_response(
                "m",
                Some(&serde_json::json!({"model": "m", "metadata": {"t": "1"}})),
                br#"{"choices":[]}"#
            )
            .is_none()
        );
    }

    #[test]
    fn unit__card_from_response__full_shape() {
        let card = card_from_response(
            "qwen3:14b",
            Some(&serde_json::json!({"model": "qwen3:14b", "metadata": {"tag": "eval"}})),
            br#"{"id": "chatcmpl-9", "created": 12345, "choices": []}"#,
        )
        .unwrap();
        assert_eq!(card.id, "chatcmpl-9");
        assert_eq!(card.model, "qwen3:14b");
        assert_eq!(card.created, 12345);
        assert_eq!(card.metadata_json, r#"{"tag":"eval"}"#);
    }
}
