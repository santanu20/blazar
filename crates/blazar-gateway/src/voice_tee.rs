//! Inline voice-out for chat replies ("speak the answer").
//!
//! A request carrying `"voice": "<installed-voice-id>"` gets its reply
//! synthesized through the piper lane and returned alongside the text:
//! streamed requests receive audio NDJSON lines interleaved after the
//! text deltas that formed each sentence; buffered requests receive one
//! `"audio"` object on the response. The text path stays byte-identical
//! when the field is absent (the tee is `None` end to end).
//!
//! Sentence boundaries follow the streaming TTS lane's rules: a `.`, `!`
//! or `?` followed by whitespace or end-of-text ends a sentence, decimal
//! digits never split (`3.14` stays glued), and an overlong run without
//! any terminator hard-splits at the same cap the TTS lane uses. A
//! per-request sentence budget bounds piper spawns on looping output.

use axum::body::Bytes;
use axum::extract::State;
use std::sync::Arc;

/// Same hard-split cap as the streaming TTS lane (`tts::STREAM_MAX_CHUNK_CHARS`).
pub const HARD_SPLIT_CHARS: usize = 600;
/// Bounds piper spawns per request: a looping model must not synthesize
/// forever (audio is additive; the text stream itself is capped by the
/// engine's own token budget).
pub const MAX_SENTENCES_PER_REQUEST: usize = 64;

/// Incremental sentence extractor for one voiced request.
#[derive(Debug, Default)]
pub struct VoiceTee {
    pending: String,
    spoken: usize,
    degraded: bool,
}

impl VoiceTee {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed one text delta; returns the sentences now complete and ready
    /// to synthesize (in order). Split across SSE chunks mid-sentence is
    /// the normal case - the partial stays buffered until its terminator
    /// lands.
    pub fn feed(&mut self, delta: &str) -> Vec<String> {
        if self.degraded {
            return Vec::new();
        }
        self.pending.push_str(delta);
        let mut ready = Vec::new();
        loop {
            if self.spoken + ready.len() >= MAX_SENTENCES_PER_REQUEST {
                self.degraded = true;
                self.pending.clear();
                return ready;
            }
            let Some((sent, rest)) = split_front(&self.pending) else {
                break;
            };
            self.pending = rest;
            ready.push(sent);
        }
        ready
    }

    /// Terminal flush: whatever non-blank text remains is one last
    /// sentence. `None` when nothing is left to speak (or the tee stood
    /// down).
    pub fn finish(&mut self) -> Option<String> {
        if self.degraded {
            return None;
        }
        let rest = self.pending.trim().to_string();
        self.pending.clear();
        if rest.is_empty() || self.spoken >= MAX_SENTENCES_PER_REQUEST {
            if !rest.is_empty() {
                self.degraded = true;
            }
            return None;
        }
        Some(rest)
    }

    /// TTS failed mid-stream: stand down cleanly. The text stream is the
    /// primary product; audio is additive and simply stops.
    pub fn degrade(&mut self) {
        self.degraded = true;
        self.pending.clear();
    }

    #[must_use]
    pub fn degraded(&self) -> bool {
        self.degraded
    }

    #[must_use]
    pub fn spoken(&self) -> usize {
        self.spoken
    }

    /// Record that a sentence was actually emitted (post-synthesis).
    pub fn note_spoken(&mut self) {
        self.spoken += 1;
    }
}

/// Longest complete sentence at the front of `text`, with the remainder.
/// Boundary rules mirror the streaming TTS lane: `[.!?]` followed by an
/// actual whitespace character ends a sentence (a terminator at the very
/// end of the buffer is held back - the next delta decides whether a
/// digit follows, so `3.` + `14` never splits); a dot between digits
/// (decimals) never splits; a terminator-less stretch beyond the cap
/// hard-splits (lossless).
fn split_front(text: &str) -> Option<(String, String)> {
    let trimmed = text.trim_start();
    let lead = text.len() - trimmed.len();
    let chars: Vec<char> = trimmed.chars().collect();
    let mut hard = 0;
    for i in 0..chars.len() {
        hard += chars[i].len_utf8();
        let c = chars[i];
        if matches!(c, '.' | '!' | '?') {
            let prev_digit = i > 0 && chars[i - 1].is_ascii_digit();
            let Some(next) = chars.get(i + 1) else {
                break; // terminator at buffer end: wait for more text
            };
            let next_digit = next.is_ascii_digit();
            if next.is_whitespace() && !(prev_digit && next_digit) {
                // The sentence keeps its trailing whitespace so emitted
                // sentences + remainder always re-concatenate to the input.
                let mut j = i + 1;
                while j < chars.len() && chars[j].is_whitespace() {
                    j += 1;
                }
                let end: String = chars[..j].iter().collect();
                let rest: String = chars[j..].iter().collect();
                let sent = text[..lead].to_string() + &end;
                return Some((sent, rest));
            }
        }
        if hard >= HARD_SPLIT_CHARS {
            let head: String = chars[..=i].iter().collect();
            let rest: String = chars[i + 1..].iter().collect();
            return Some((text[..lead].to_string() + &head, rest));
        }
    }
    None
}

/// Synthesize one sentence to WAV bytes through the gateway's own piper
/// lane (same internal-call pattern as the realtime session lane, so
/// admission, voices and accounting all behave identically).
pub async fn synthesize_wav(
    state: &Arc<crate::state::AppState>,
    key: Option<&crate::keys::KeyCtx>,
    voice: &str,
    text: &str,
) -> Result<Vec<u8>, String> {
    let tts_body = serde_json::to_vec(&serde_json::json!({
        "model": voice,
        "input": text,
        "response_format": "wav",
    }))
    .unwrap_or_default();
    let mut tts_headers = axum::http::HeaderMap::new();
    tts_headers.insert(
        axum::http::header::CONTENT_TYPE,
        axum::http::HeaderValue::from_static("application/json"),
    );
    let tts = crate::tts::audio_speech(
        State(state.clone()),
        key.cloned().map(axum::Extension),
        axum::http::Uri::from_static("/v1/audio/speech"),
        axum::http::Method::POST,
        tts_headers,
        Bytes::from(tts_body),
    )
    .await;
    let (parts, body) = tts.into_parts();
    let bytes = axum::body::to_bytes(body, 64 * 1024 * 1024)
        .await
        .map_err(|e| format!("tts body read failed: {e}"))?
        .to_vec();
    if !parts.status.is_success() {
        return Err(format!(
            "piper lane returned {} for voice '{voice}' - audio stops, text continues",
            parts.status.as_u16()
        ));
    }
    Ok(bytes)
}

/// One ollama-dialect NDJSON audio line (interleaved after the text
/// deltas that formed the sentence).
#[must_use]
pub fn ollama_audio_line(model: &str, seq: usize, wav: &[u8]) -> String {
    let b64 = crate::realtime::base64_encode(wav);
    format!(
        "{{\"model\":{model:?},\"audio\":{{\"data\":\"{b64}\",\"format\":\"wav\",\"seq\":{seq}}},\"done\":false}}\n"
    )
}

/// WAV bytes to the `"audio"` object buffered responses carry.
#[must_use]
pub fn audio_field(wav: &[u8]) -> serde_json::Value {
    serde_json::json!({
        "data": crate::realtime::base64_encode(wav),
        "format": "wav",
    })
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)] // unit__<x>__<y> double-underscore convention
    use super::*;

    #[test]
    fn unit__split_front__sentence_boundary_and_remainder() {
        let (sent, rest) = split_front("Hello there. Next one!").unwrap();
        assert_eq!(sent, "Hello there. ");
        assert_eq!(rest, "Next one!");
    }

    #[test]
    fn unit__split_front__decimals_never_split() {
        // 3.14: digit dot digit - the dot is not a boundary.
        assert!(split_front("Pi is 3.14").is_none());
        let (sent, _) = split_front("Pi is 3.14159 rounded. And more.").unwrap();
        assert!(sent.contains("3.14159"));
    }

    #[test]
    fn unit__split_front__decimal_at_chunk_edge_is_held() {
        // "3." at buffer end: the next delta may deliver "14" - held.
        assert!(split_front("Pi is 3.").is_none());
        let (sent, _) = split_front("Pi is 3.14 approx. Next.").unwrap();
        assert!(sent.contains("3.14"));
    }

    #[test]
    fn unit__split_front__hard_split_is_lossless_and_bounded() {
        let blob = "w".repeat(HARD_SPLIT_CHARS + 50);
        let (head, rest) = split_front(&blob).unwrap();
        assert!(head.chars().count() <= HARD_SPLIT_CHARS + 1);
        assert_eq!(
            head.chars().count() + rest.chars().count(),
            blob.chars().count()
        );
    }

    #[test]
    fn unit__voice_tee__feeds_across_chunk_splits_then_finishes() {
        let mut tee = VoiceTee::new();
        // Sentence split across two deltas: nothing completes until the
        // terminator + following space land.
        assert!(tee.feed("The sea ").is_empty());
        let ready = tee.feed("is salty. Because erosion. Done.");
        assert_eq!(ready.len(), 2, "two complete sentences: {ready:?}");
        assert_eq!(ready[0], "The sea is salty. ");
        assert_eq!(ready[1], "Because erosion. ");
        let tail = tee.finish().unwrap();
        assert_eq!(tail, "Done.");
    }

    #[test]
    fn unit__voice_tee__budget_stands_down_cleanly() {
        let mut tee = VoiceTee::new();
        let mut total = 0;
        loop {
            let ready = tee.feed("Sentence. ");
            if ready.is_empty() && tee.degraded() {
                break;
            }
            total += ready.len();
            for _ in ready {
                tee.note_spoken();
            }
            assert!(total <= MAX_SENTENCES_PER_REQUEST);
        }
        assert_eq!(tee.spoken(), MAX_SENTENCES_PER_REQUEST);
        assert!(tee.finish().is_none());
    }

    #[test]
    fn unit__voice_tee__degrade_stops_everything() {
        let mut tee = VoiceTee::new();
        tee.feed("Hello. ");
        tee.degrade();
        assert!(tee.feed("More text. ").is_empty());
        assert!(tee.finish().is_none());
    }

    #[test]
    fn unit__ollama_audio_line__shape() {
        let line = ollama_audio_line("m", 1, b"RIFF");
        let v: serde_json::Value = serde_json::from_str(line.trim()).unwrap();
        assert_eq!(v["model"], "m");
        assert_eq!(v["audio"]["format"], "wav");
        assert_eq!(v["audio"]["seq"], 1);
        assert_eq!(v["done"], false);
    }
}
