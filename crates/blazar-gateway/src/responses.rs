//! Responses API conversation state: `previous_response_id` chaining +
//! `store` semantics — pure gateway value-add (upstream llama-server has
//! no storage; verified absent in server.cpp). Bounded LRU, never grows
//! unbounded, metadata + items only.

use std::collections::VecDeque;

use serde_json::Value;

/// One stored response: enough to reconstruct the conversation prefix.
pub struct StoredResponse {
    pub model: String,
    /// The request's `input` items (message list) as received.
    pub input_items: Value,
    /// The response's `output` items (assistant messages, tool calls).
    pub output_items: Value,
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub ts: u64,
}

const CAP: usize = 256;
const TTL_SECS: u64 = 24 * 60 * 60;

/// Bounded registry: newest at the back; overflow evicts the oldest.
/// `VecDeque` scan is fine at this cap (chaining touches one entry).
pub struct ResponsesRegistry {
    entries: VecDeque<(String, StoredResponse)>,
}

impl Default for ResponsesRegistry {
    fn default() -> Self {
        Self {
            entries: VecDeque::with_capacity(CAP),
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
            if now.saturating_sub(e.ts) > TTL_SECS {
                self.entries.pop_front();
            } else {
                break;
            }
        }
    }

    pub fn put(&mut self, id: String, r: StoredResponse) {
        self.sweep();
        if self.entries.len() >= CAP {
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
                },
            );
        }
        assert!(r.get("resp_0").is_none(), "oldest evicted");
        assert!(r.get("resp_299").is_some(), "newest kept");
        assert!(r.entries.len() <= CAP);
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
}
