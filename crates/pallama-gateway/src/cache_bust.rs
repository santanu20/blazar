//! Cache-bust churn detector — the agent-client teaching layer ollama
//! shipped as v0.33.0 "trustworthy prefill restore points", inverted:
//! pallama's engine already resumes cancelled prefills natively (live
//! proven: 41 KB prompt aborted at 2 s, retry served 6,154 cached
//! tokens), so the remaining loss is CLIENT-side — chatty agent clients
//! (token counters, timestamps, per-turn tool list reshuffles inside the
//! system prompt) mutate the prefix every turn and silently re-prefill
//! from scratch. This module fingerprints system+tools per request and
//! flags a conversation whose fingerprint churns across consecutive
//! turns. Advisory only: never mutates the request, never errors, one
//! sentinel record per churn episode.

use std::collections::HashMap;
use std::sync::Mutex;

use crate::sentinel::{Code, Detection, Sentinel, SentinelRecord};

/// Models tracked before the map resets wholesale (review revision 2:
/// clear-all beats LRU bookkeeping for a telemetry-only structure).
const MAX_MODELS: usize = 64;

struct Entry {
    /// Last-seen system+tools fingerprint.
    sys_tools: u64,
    /// Conversation identity (`first_user_fp`). `None`
    /// = the request had no recognizable prompt (embeddings, tools-only
    /// transforms) — no episode tracking for it.
    convo: Option<u64>,
    /// Consecutive turns with a changed system+tools fingerprint on a
    /// stable conversation.
    streak: u32,
    /// Episode already reported (one sentinel record per streak).
    warned: bool,
}

/// Per-model churn state. Hashing happens outside the lock (review
/// revision 4); the critical section is a few map operations.
#[derive(Default)]
pub struct CacheBustTracker {
    inner: Mutex<HashMap<String, Entry>>,
}

impl CacheBustTracker {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one turn. Returns the streak length when an episode fires
    /// (>= 2 consecutive mutations, first emission only); `None` = silent.
    ///
    /// Episode semantics (review revision 3): a system-stable turn closes
    /// the episode — resets both the streak and the warned flag, so a
    /// later burst warns anew. A conversation change also resets (new
    /// conversation, fresh baseline).
    fn note(&self, model: &str, sys_tools_fp: u64, convo_fp: Option<u64>) -> Option<u32> {
        let mut map = self.inner.lock().expect("cache-bust tracker lock");
        if map.len() >= MAX_MODELS {
            // Telemetry-only state: wholesale reset beats per-entry LRU.
            map.clear();
        }
        let Some(convo) = convo_fp else {
            // No conversation identity: reset any episode for this model
            // so a tools-only interlude never poisons the next one.
            map.remove(model);
            return None;
        };
        let entry = map.entry(model.to_string()).or_insert(Entry {
            sys_tools: sys_tools_fp,
            convo: Some(convo),
            streak: 0,
            warned: false,
        });
        if entry.convo != Some(convo) {
            // Different conversation on the same model: fresh baseline.
            entry.convo = Some(convo);
            entry.sys_tools = sys_tools_fp;
            entry.streak = 0;
            entry.warned = false;
            return None;
        }
        if entry.sys_tools == sys_tools_fp {
            // Stable turn closes the episode.
            entry.streak = 0;
            entry.warned = false;
            return None;
        }
        entry.sys_tools = sys_tools_fp;
        entry.streak += 1;
        if entry.streak >= 2 && !entry.warned {
            entry.warned = true;
            return Some(entry.streak);
        }
        None
    }
}

/// Full-length fingerprint of everything a client puts BEFORE the
/// conversation: all system message contents concatenated plus the
/// canonical JSON of the tools array when present. Full length, not the
/// proxy's 256 B sys head — late mutations deep in a long system prompt
/// are exactly the cache busters this detector exists to catch.
/// `developer` rides with `system`: it is the o-series spelling of the
/// same prefix slot on the `OpenAI` dialect.
#[must_use]
pub fn system_tools_fp(req: &serde_json::Value) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    if let Some(messages) = req.get("messages").and_then(|m| m.as_array()) {
        for m in messages {
            let role = m.get("role").and_then(|r| r.as_str());
            if matches!(role, Some("system" | "developer")) {
                match m.get("content") {
                    Some(serde_json::Value::String(s)) => s.hash(&mut h),
                    Some(other) => other.to_string().hash(&mut h),
                    None => serde_json::Value::Null.to_string().hash(&mut h),
                }
            }
        }
    }
    if let Some(tools) = req.get("tools") {
        serde_json::to_string(tools)
            .unwrap_or_default()
            .hash(&mut h);
    }
    h.finish()
}

/// Conversation identity: the FIRST user message, capped at 1 KiB (the
/// proxy's affinity-hash discipline). Deliberately excludes the system
/// prompt — the very thing being fingerprinted for churn must not also
/// be part of the "same conversation" key, or every mutation would look
/// like a new conversation and no episode could ever form. The first
/// user message is the stable anchor of an agent-client conversation:
/// history appends after it, counters churn in the system above it.
#[must_use]
pub fn first_user_fp(req: &serde_json::Value) -> Option<u64> {
    use std::hash::{Hash, Hasher};
    let messages = req.get("messages")?.as_array()?;
    for m in messages {
        if m.get("role").and_then(|r| r.as_str()) == Some("user") {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            match m.get("content") {
                Some(serde_json::Value::String(s)) => {
                    let cap = &s[..s.len().min(1024)];
                    cap.hash(&mut h);
                }
                Some(other) => other.to_string().hash(&mut h),
                None => serde_json::Value::Null.to_string().hash(&mut h),
            }
            return Some(h.finish());
        }
    }
    None
}

/// One request observed on a chat route (dialect-agnostic: ollama and
/// `OpenAI` bodies both carry `messages` + `tools`). Never mutates `req`.
/// Fires at most once per churn episode per model+conversation. `trace`
/// is the request-scoped trace id so `pallama why` output correlates
/// with the response header the client saw.
pub fn note_request(
    sentinel: &Sentinel,
    tracker: &CacheBustTracker,
    model: &str,
    req: &serde_json::Value,
    route: &str,
    trace: &str,
) {
    let sys_fp = system_tools_fp(req);
    let convo_fp = first_user_fp(req);
    let Some(streak) = tracker.note(model, sys_fp, convo_fp) else {
        return;
    };
    let detail = format!(
        "system prompt/tools fingerprint changed {streak} turns in a row on a stable conversation"
    );
    let record = SentinelRecord {
        trace: trace.to_string(),
        ts: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs()),
        route: route.to_string(),
        model: model.to_string(),
        // Advisory observation, not a served response: there is no HTTP
        // status of our own to report, the request was forwarded fine.
        status: 0,
        stream: false,
        finish: None,
        detections: vec![Detection {
            code: Code::CacheBustSystem,
            detail,
        }],
        prompt_tokens: None,
        completion_tokens: None,
        ctx: None,
        degraded: false,
        ms: 0,
        logprob_mean: None,
        logprob_min: None,
        logprob_tokens: None,
    };
    sentinel.commit(&record);
}

#[cfg(test)]
#[allow(non_snake_case)] // unit__<x>__<y> double-underscore convention
mod tests {
    use super::*;

    fn tracker() -> CacheBustTracker {
        CacheBustTracker::new()
    }

    #[test]
    fn unit__note__first_seen_silent() {
        let t = tracker();
        assert_eq!(t.note("m", 1, Some(10)), None);
    }

    #[test]
    fn unit__note__single_mutation_silent_second_fires() {
        let t = tracker();
        t.note("m", 1, Some(10));
        assert_eq!(
            t.note("m", 2, Some(10)),
            None,
            "first mutation: teach later"
        );
        assert_eq!(
            t.note("m", 3, Some(10)),
            Some(2),
            "second consecutive fires"
        );
    }

    #[test]
    fn unit__note__third_mutation_silent_once_per_episode() {
        let t = tracker();
        t.note("m", 1, Some(10));
        t.note("m", 2, Some(10));
        assert_eq!(t.note("m", 3, Some(10)), Some(2));
        assert_eq!(
            t.note("m", 4, Some(10)),
            None,
            "already warned this episode"
        );
    }

    #[test]
    fn unit__note__stable_turn_closes_episode() {
        let t = tracker();
        t.note("m", 1, Some(10));
        t.note("m", 2, Some(10));
        t.note("m", 3, Some(10)); // fires
        t.note("m", 3, Some(10)); // stable: closes episode
        assert_eq!(
            t.note("m", 4, Some(10)),
            None,
            "first mutation of the new episode is silent"
        );
        assert_eq!(
            t.note("m", 5, Some(10)),
            Some(2),
            "post-closure burst warns anew"
        );
    }

    #[test]
    fn unit__note__convo_change_resets() {
        let t = tracker();
        t.note("m", 1, Some(10));
        t.note("m", 2, Some(10));
        assert_eq!(t.note("m", 3, Some(99)), None, "new conversation: baseline");
        assert_eq!(t.note("m", 4, Some(99)), None, "streak restarted at 1");
        assert_eq!(t.note("m", 5, Some(99)), Some(2));
    }

    #[test]
    fn unit__note__no_convo_identity_resets_state() {
        let t = tracker();
        t.note("m", 1, Some(10));
        t.note("m", 2, Some(10));
        assert_eq!(t.note("m", 3, None), None);
        assert_eq!(t.note("m", 4, Some(10)), None, "prior episode discarded");
        assert_eq!(t.note("m", 5, Some(10)), None);
        assert_eq!(t.note("m", 6, Some(10)), Some(2));
    }

    #[test]
    fn unit__system_tools_fp__stable_and_tools_sensitive() {
        let base = serde_json::json!({
            "messages": [
                {"role": "system", "content": "you are terse"},
                {"role": "user", "content": "hi"}
            ]
        });
        assert_eq!(system_tools_fp(&base), system_tools_fp(&base));
        // User-turn growth (the normal append-only pattern) must NOT
        // change the fingerprint.
        let grown = serde_json::json!({
            "messages": [
                {"role": "system", "content": "you are terse"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "more"}
            ]
        });
        assert_eq!(system_tools_fp(&base), system_tools_fp(&grown));
        // System mutation (the cache buster) must change it.
        let mutated = serde_json::json!({
            "messages": [
                {"role": "system", "content": "you are terse (turn 2)"},
                {"role": "user", "content": "hi"}
            ]
        });
        assert_ne!(system_tools_fp(&base), system_tools_fp(&mutated));
        // Tools presence/shape must change it.
        let with_tools = serde_json::json!({
            "messages": [
                {"role": "system", "content": "you are terse"},
                {"role": "user", "content": "hi"}
            ],
            "tools": [{"type": "function", "function": {"name": "ls"}}]
        });
        assert_ne!(system_tools_fp(&base), system_tools_fp(&with_tools));
        let tools2 = serde_json::json!({
            "messages": [
                {"role": "system", "content": "you are terse"},
                {"role": "user", "content": "hi"}
            ],
            "tools": [{"type": "function", "function": {"name": "pwd"}}]
        });
        assert_ne!(system_tools_fp(&with_tools), system_tools_fp(&tools2));
    }

    #[test]
    fn unit__note__cap_clears_all_state() {
        let t = tracker();
        t.note("m", 1, Some(10));
        for i in 0..MAX_MODELS {
            t.note(&format!("filler-{i}"), 1, Some(10));
        }
        // The original entry was cleared with the whole map: the full
        // episode (baseline, silent mutation, firing mutation) replays.
        assert_eq!(t.note("m", 2, Some(10)), None, "first-seen again");
        assert_eq!(t.note("m", 3, Some(10)), None, "first mutation silent");
        assert_eq!(t.note("m", 4, Some(10)), Some(2), "second fires");
    }

    #[test]
    fn unit__first_user_fp__stable_under_system_and_history_churn() {
        let base = serde_json::json!({
            "messages": [
                {"role": "system", "content": "sys v1"},
                {"role": "user", "content": "fix the bug in foo.rs"}
            ]
        });
        assert_eq!(first_user_fp(&base), first_user_fp(&base));
        let churned = serde_json::json!({
            "messages": [
                {"role": "system", "content": "sys v2 — tokens left: 8123"},
                {"role": "user", "content": "fix the bug in foo.rs"},
                {"role": "assistant", "content": "looking"},
                {"role": "user", "content": "any progress?"}
            ]
        });
        assert_eq!(
            first_user_fp(&base),
            first_user_fp(&churned),
            "system mutation + history growth must not change conversation identity"
        );
        let other_convo = serde_json::json!({
            "messages": [
                {"role": "system", "content": "sys v1"},
                {"role": "user", "content": "write a poem"}
            ]
        });
        assert_ne!(first_user_fp(&base), first_user_fp(&other_convo));
        // No user turn at all: no identity.
        let userless = serde_json::json!({
            "messages": [{"role": "system", "content": "sys"}]
        });
        assert_eq!(first_user_fp(&userless), None);
    }

    /// The full pipeline against a real Sentinel: the Claude-Code-style
    /// cache-buster pattern (stable first user turn, mutating system
    /// counter, append-only history) fires exactly ONE advisory record;
    /// a stable client stays silent; the request body is never mutated.
    #[test]
    fn unit__note_request__churning_client_fires_once_stable_client_silent() {
        let sentinel = Sentinel::new(true, 0, None);
        let tracker = CacheBustTracker::new();
        let mut rx = sentinel.watch();

        let turn = |n: u64| {
            serde_json::json!({
                "model": "agent-model",
                "messages": [
                    {"role": "system", "content": format!("You are an agent. Context window left: {} tokens.", 9000 - n * 750)},
                    {"role": "user", "content": "fix the failing test in bar.rs"},
                    {"role": "assistant", "content": format!("pass {n} done")},
                    {"role": "user", "content": format!("continue ({n})")}
                ]
            })
        };
        for n in 0..4_u64 {
            let req = turn(n);
            let before = serde_json::to_string(&req).unwrap();
            note_request(
                &sentinel,
                &tracker,
                "agent-model",
                &req,
                "api/chat",
                "plm-cb-1",
            );
            assert_eq!(
                serde_json::to_string(&req).unwrap(),
                before,
                "detector must never mutate the request"
            );
        }
        let rec = rx
            .try_recv()
            .expect("exactly one record after 4 churning turns");
        assert_eq!(rec.trace, "plm-cb-1");
        assert_eq!(rec.route, "api/chat");
        assert_eq!(rec.model, "agent-model");
        assert_eq!(rec.detections.len(), 1);
        assert_eq!(rec.detections[0].code.as_str(), "cache_bust_system");
        assert!(rx.try_recv().is_err(), "no second record this episode");

        // Stable system from here on: the episode closes, silence again.
        let stable = turn(3);
        note_request(
            &sentinel,
            &tracker,
            "agent-model",
            &stable,
            "api/chat",
            "plm-cb-2",
        );
        note_request(
            &sentinel,
            &tracker,
            "agent-model",
            &stable,
            "api/chat",
            "plm-cb-3",
        );
        assert!(rx.try_recv().is_err(), "stable client must stay silent");
    }
}
