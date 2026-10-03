//! Responses API conversation state: `previous_response_id` chaining +
//! `store` semantics — pure gateway value-add (upstream llama-server has
//! no storage; verified absent in server.cpp). Bounded LRU, never grows
//! unbounded, metadata + items only.
//!
//! Durability (v0.15): every stored response is written through to the
//! SQLite ledger (`responses` table, store schema v8) and survives
//! gateway restarts. Memory stays the hot path; the ledger is only read
//! on a memory miss (`promote_from_store` re-checks the TTL — a row may
//! have aged past the window while the gateway was down).

use std::collections::VecDeque;

use blazar_core::Store;
use blazar_core::store::StoredResponseRow;
use serde_json::Value;

/// One stored response: enough to reconstruct the conversation prefix.
#[derive(Clone)]
pub struct StoredResponse {
    pub model: String,
    /// The request's `input` items (message list) as received.
    pub input_items: Value,
    /// The response's `output` items (assistant messages, tool calls).
    pub output_items: Value,
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub ts: u64,
    /// v10: conversation grouping key ("" when the request named none).
    /// Chained requests inherit the previous response's value so a whole
    /// thread lands under one listing handle.
    pub conversation: String,
    /// v10: the exact completed body JSON when the gateway kept it
    /// (background mode always). The RAM registry never reads it —
    /// only `to_row` carries it into the ledger for GET fidelity.
    pub body_json: Option<String>,
}

/// Retention window, memory and ledger alike: 24h.
pub const RESPONSES_TTL_SECS: u64 = 24 * 60 * 60;
/// Hard bound on the hot LRU; also the durable-ledger prune target.
pub const RESPONSES_CAP: usize = 256;

/// Bounded registry: newest at the back; overflow evicts the oldest.
/// `VecDeque` scan is fine at this cap (chaining touches one entry).
pub struct ResponsesRegistry {
    entries: VecDeque<(String, StoredResponse)>,
}

impl Default for ResponsesRegistry {
    fn default() -> Self {
        Self {
            entries: VecDeque::with_capacity(RESPONSES_CAP),
        }
    }
}

impl ResponsesRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn sweep(&mut self) {
        let now = unix_now();
        while let Some((_, e)) = self.entries.front() {
            if now.saturating_sub(e.ts) > RESPONSES_TTL_SECS {
                self.entries.pop_front();
            } else {
                break;
            }
        }
    }

    pub fn put(&mut self, id: String, r: StoredResponse) {
        self.sweep();
        if self.entries.len() >= RESPONSES_CAP {
            self.entries.pop_front();
        }
        // Idempotent re-put (retry-safe).
        if let Some(slot) = self.entries.iter_mut().find(|(i, _)| *i == id) {
            slot.1 = r;
            return;
        }
        self.entries.push_back((id, r));
    }

    pub fn get(&mut self, id: &str) -> Option<&StoredResponse> {
        self.sweep();
        self.entries.iter().find(|(i, _)| i == id).map(|(_, e)| e)
    }

    /// Reconstruct the chained input: stored input + stored output + the
    /// new request's input (Responses API conversation semantics).
    #[must_use]
    pub fn chain_input(stored: &StoredResponse, new_input: &Value) -> Value {
        let mut items = Vec::new();
        let push = |items: &mut Vec<Value>, v: &Value| match v {
            Value::Array(a) => items.extend(a.iter().cloned()),
            Value::String(s) => items.push(serde_json::json!({
                "role": "user", "content": s,
            })),
            Value::Null => {}
            other => items.push(other.clone()),
        };
        push(&mut items, &stored.input_items);
        push(&mut items, &stored.output_items);
        push(&mut items, new_input);
        Value::Array(items)
    }

    /// Restart durability: memory miss → ledger lookup. The row's TTL is
    /// re-checked before use (a response may have aged past the window
    /// while the gateway was down — expired means miss, not stale hit).
    /// A found row is promoted into the hot LRU so subsequent hits are
    /// memory-speed. Corrupt JSON degrades to `Null` (`chain_input` treats
    /// it as empty) rather than poisoning the conversation.
    pub fn promote_from_store(&mut self, store: &Store, id: &str) -> Option<StoredResponse> {
        let row = store.get_response(id).ok()??;
        let ts = u64::try_from(row.ts.max(0)).unwrap_or(0);
        if unix_now().saturating_sub(ts) > RESPONSES_TTL_SECS {
            return None;
        }
        let sr = StoredResponse {
            model: row.model,
            input_items: serde_json::from_str(&row.input_json).unwrap_or(Value::Null),
            output_items: serde_json::from_str(&row.output_json).unwrap_or(Value::Null),
            input_tokens: row
                .input_tokens
                .map(|v| u64::try_from(v).unwrap_or(u64::MAX)),
            output_tokens: row
                .output_tokens
                .map(|v| u64::try_from(v).unwrap_or(u64::MAX)),
            ts,
            // Inheritance matters: a chained request must land in the same
            // conversation listing even after a restart. The exact body is
            // deliberately NOT promoted — GET reads the row for fidelity,
            // chaining never needs it, and bodies would bloat the LRU.
            conversation: row.conversation,
            body_json: None,
        };
        let out = sr.clone();
        self.put(id.to_string(), sr);
        Some(out)
    }

    /// Ledger row for the write-through path (schema v8 `responses`).
    #[must_use]
    pub fn to_row(id: &str, r: &StoredResponse) -> StoredResponseRow {
        StoredResponseRow {
            id: id.to_string(),
            model: r.model.clone(),
            input_json: serde_json::to_string(&r.input_items).unwrap_or_default(),
            output_json: serde_json::to_string(&r.output_items).unwrap_or_default(),
            input_tokens: r.input_tokens.map(|v| i64::try_from(v).unwrap_or(i64::MAX)),
            output_tokens: r
                .output_tokens
                .map(|v| i64::try_from(v).unwrap_or(i64::MAX)),
            ts: i64::try_from(r.ts).unwrap_or(i64::MAX),
            conversation: r.conversation.clone(),
            body_json: r.body_json.clone(),
        }
    }
}

#[must_use]
pub fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// `resp_` + 24 hex chars of OS entropy (client-chainable id).
#[must_use]
pub fn new_response_id() -> String {
    let mut buf = [0u8; 12];
    getrandom::fill(&mut buf).expect("OS entropy source");
    let mut hex = String::with_capacity(24);
    for b in buf {
        use std::fmt::Write as _;
        let _ = write!(hex, "{b:02x}");
    }
    format!("resp_{hex}")
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    #[test]
    fn unit__registry__put_get_evict() {
        let mut r = ResponsesRegistry::new();
        for i in 0..300 {
            r.put(
                format!("resp_{i}"),
                StoredResponse {
                    model: "m".into(),
                    input_items: Value::Array(vec![]),
                    output_items: Value::Array(vec![]),
                    input_tokens: None,
                    output_tokens: None,
                    ts: unix_now(),
                    conversation: String::new(),
                    body_json: None,
                },
            );
        }
        assert!(r.get("resp_0").is_none(), "oldest evicted");
        assert!(r.get("resp_299").is_some(), "newest kept");
        assert!(r.entries.len() <= RESPONSES_CAP);
    }

    #[test]
    fn unit__chain_input__stored_then_new() {
        let stored = StoredResponse {
            model: "m".into(),
            input_items: serde_json::json!([{"role": "user", "content": "hi"}]),
            output_items: serde_json::json!([{"type": "message", "content": "hello"}]),
            input_tokens: None,
            output_tokens: None,
            ts: 0,
            conversation: String::new(),
            body_json: None,
        };
        let chained = ResponsesRegistry::chain_input(
            &stored,
            &serde_json::json!([{"role": "user", "content": "again"}]),
        );
        let arr = chained.as_array().unwrap();
        assert_eq!(arr.len(), 3);
        assert_eq!(arr[2]["content"], "again");
        // String shorthand input expands to a user message.
        let chained = ResponsesRegistry::chain_input(&stored, &serde_json::json!("plain"));
        assert_eq!(chained.as_array().unwrap().len(), 3);
    }

    fn tmp_store() -> (tempfile::TempDir, Store) {
        let tmp = tempfile::tempdir().unwrap();
        let dirs = blazar_core::BlazarDirs {
            config_dir: tmp.path().join("cfg"),
            data_dir: tmp.path().join("data"),
        };
        (tmp, Store::open(&dirs).unwrap())
    }

    fn sample(model: &str, ts: u64) -> StoredResponse {
        StoredResponse {
            model: model.into(),
            input_items: serde_json::json!([{"role": "user", "content": "hi"}]),
            output_items: serde_json::json!([{"type": "message", "content": "hello"}]),
            input_tokens: Some(1),
            output_tokens: Some(2),
            ts,
            conversation: String::new(),
            body_json: None,
        }
    }

    #[test]
    fn unit__responses__sqlite_roundtrip_after_restart() {
        // Restart simulation: registry A writes through to the ledger;
        // registry B (fresh process, empty LRU) + same ledger finds the
        // row, promotes it, and chains off it.
        let (tmp, store) = tmp_store();
        let sr = sample("m", unix_now());
        store
            .put_response(&ResponsesRegistry::to_row("resp_x", &sr))
            .unwrap();

        let mut b = ResponsesRegistry::new();
        assert!(b.get("resp_x").is_none(), "fresh registry: memory miss");
        let loaded = b.promote_from_store(&store, "resp_x").expect("promoted");
        assert_eq!(loaded.model, "m");
        assert_eq!(loaded.input_tokens, Some(1));
        assert_eq!(loaded.output_items[0]["content"], "hello");
        assert!(b.get("resp_x").is_some(), "row promoted into hot LRU");
        let chained = ResponsesRegistry::chain_input(
            &loaded,
            &serde_json::json!([{"role": "user", "content": "again"}]),
        );
        assert_eq!(chained.as_array().unwrap().len(), 3);
        drop(tmp);
    }

    #[test]
    fn unit__responses__promote_rejects_expired_row() {
        let (tmp, store) = tmp_store();
        let old = sample("m", unix_now().saturating_sub(RESPONSES_TTL_SECS + 5));
        store
            .put_response(&ResponsesRegistry::to_row("resp_old", &old))
            .unwrap();
        let mut r = ResponsesRegistry::new();
        assert!(r.promote_from_store(&store, "resp_old").is_none());
        assert!(r.get("resp_old").is_none(), "expired row not promoted");
        drop(tmp);
    }

    #[test]
    fn unit__responses__row_roundtrip_preserves_fields() {
        let sr = sample("qwen3-8b", unix_now());
        let row = ResponsesRegistry::to_row("resp_r", &sr);
        let back = StoredResponse {
            model: row.model,
            input_items: serde_json::from_str(&row.input_json).unwrap(),
            output_items: serde_json::from_str(&row.output_json).unwrap(),
            input_tokens: row
                .input_tokens
                .map(|v| u64::try_from(v).unwrap_or(u64::MAX)),
            output_tokens: row
                .output_tokens
                .map(|v| u64::try_from(v).unwrap_or(u64::MAX)),
            ts: u64::try_from(row.ts.max(0)).unwrap_or(0),
            conversation: String::new(),
            body_json: None,
        };
        assert_eq!(back.model, sr.model);
        assert_eq!(back.input_items, sr.input_items);
        assert_eq!(back.output_items, sr.output_items);
        assert_eq!(back.input_tokens, sr.input_tokens);
        assert_eq!(back.ts, sr.ts);
    }
}
