//! `/v1/audio/transcriptions` + `/v1/audio/translations`: local
//! whisper.cpp lane (H8).
//!
//! Decision order:
//! (1) explicit remote intent (`name:model`) always forwards — the user
//!     named a remote;
//! (2) local lane when whisper-server + at least one ggml model are
//!     installed (lazy child, hot model swap);
//! (3) otherwise a teaching error (llama.cpp children do not transcribe).
//!
//! Multipart is parsed binary-safe: parts split ONLY on the exact random
//! boundary — audio bytes cannot forge one.
//!
//! The local lane REBUILDS the multipart from a verified field
//! whitelist (upstream has no OpenAI-compatible audio route — `/inference`
//! is the whole surface), so transcription/translation ride the same
//! lazy child. `/v1/audio/translations` forces `translate=true` on the
//! rebuilt request; upstream reports `"task": "translate"` back.
//! `timestamp_granularities` is consumed gateway-side: it forces the
//! `verbose_json` shape, turns on the engine's token timing, and the
//! gateway folds token timings into `segments[].words` (engine-derived
//! only — never interpolated).
//!
//! Async: upstream `/inference` is sync-only (no job surface to relay,
//! unlike the sd-server image lane), so `"async": "true"` as a form
//! field runs the same forward inside a spawned task and hands back a
//! gateway-owned job handle — `/v1/audio/jobs/{id}` polls it,
//! `/v1/audio/jobs/{id}/cancel` aborts the wait. Jobs die with the
//! gateway process, the same lifetime truth the image lane attaches to
//! its children.
//!
//! Streaming (F6): upstream has no partial-transcription surface (the
//! server is mutex-serialized batch), so `"stream": "true"` implements
//! progressive decode gateway-side — WAV PCM inputs split into
//! frame-aligned windows decoded sequentially on the same lazy child,
//! one SSE `chunk.completed` per window and a final
//! `transcript.completed`. Zero throughput cost: the child serializes
//! requests anyway, the windows are the same total decode.

use std::collections::HashMap;
use std::sync::Arc;

use axum::Extension;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, Method, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use blazar_runtime::whisper;
use futures::StreamExt as _;

use crate::proxy::openai_error;
use crate::remotes::split_remote;
use crate::state::AppState;

/// One parsed multipart part (cloned into async job tasks, which
/// outlive the request that received the bytes).
#[derive(Clone)]
pub(crate) struct Part {
    pub(crate) name: String,
    pub(crate) filename: Option<String>,
    pub(crate) content_type: Option<String>,
    pub(crate) data: Vec<u8>,
}

/// Boundary-aware multipart split (RFC 2046 essentials: `--boundary`
/// delimiters, CRLF-separated part headers, `--boundary--` terminator).
/// Returns None on structural garbage (caller 400s).
pub(crate) fn parse_multipart(body: &[u8], content_type: &str) -> Option<Vec<Part>> {
    let boundary = content_type
        .split(';')
        .map(str::trim)
        .find_map(|p| p.strip_prefix("boundary="))?
        .trim_matches('"');
    let delim = format!("--{boundary}");
    let mut parts = Vec::new();
    let mut pos = 0usize;
    while let Some(rel) = find_sub(&body[pos..], delim.as_bytes()) {
        let mut seg_start = pos + rel + delim.len();
        // Terminator?
        if body[seg_start..].starts_with(b"--") {
            break;
        }
        // Each delimiter is followed by CRLF before part headers.
        if body[seg_start..].starts_with(b"\r\n") {
            seg_start += 2;
        } else if body[seg_start..].starts_with(b"\n") {
            seg_start += 1;
        }
        let Some(hdr_end_rel) = find_sub(&body[seg_start..], b"\r\n\r\n")
            .or_else(|| find_sub(&body[seg_start..], b"\n\n"))
        else {
            break;
        };
        let sep_len = if body[seg_start + hdr_end_rel..].starts_with(b"\r\n\r\n") {
            4
        } else {
            2
        };
        let headers_raw = &body[seg_start..seg_start + hdr_end_rel];
        let data_start = seg_start + hdr_end_rel + sep_len;
        // Data runs to the NEXT delimiter that starts a line (preceded
        // by CRLF/LF) — binary audio containing the boundary bytes
        // mid-line is data, not a separator.
        let Some(next) = find_line_delim(body, data_start, delim.as_bytes()) else {
            break;
        };
        let mut data_end = next;
        if data_end >= 2 && &body[data_end - 2..data_end] == b"\r\n" {
            data_end -= 2;
        } else if data_end >= 1 && body[data_end - 1] == b'\n' {
            data_end -= 1;
        }
        parts.push(parse_part(headers_raw, &body[data_start..data_end])?);
        pos = next;
    }
    Some(parts)
}

/// Next delimiter occurrence that begins a line: preceded by CRLF or LF,
/// or at `from` itself (start of body). Mid-line matches are data.
fn find_line_delim(hay: &[u8], from: usize, delim: &[u8]) -> Option<usize> {
    let mut pos = from;
    while let Some(rel) = find_sub(&hay[pos..], delim) {
        let at = pos + rel;
        let line_start = at == from
            || (at >= 2 && &hay[at - 2..at] == b"\r\n")
            || (at >= 1 && hay[at - 1] == b'\n');
        if line_start {
            return Some(at);
        }
        pos = at + 1;
    }
    None
}

fn parse_part(headers_raw: &[u8], data: &[u8]) -> Option<Part> {
    let headers = String::from_utf8_lossy(headers_raw);
    let mut name = None;
    let mut filename = None;
    let mut content_type = None;
    for line in headers.lines() {
        let lower = line.to_ascii_lowercase();
        if lower.starts_with("content-disposition:") {
            for piece in line.split(';').map(str::trim) {
                if let Some(v) = piece.strip_prefix("name=") {
                    name = Some(v.trim_matches('"').to_string());
                } else if let Some(v) = piece.strip_prefix("filename=") {
                    filename = Some(v.trim_matches('"').to_string());
                }
            }
        } else if let Some(v) = lower.strip_prefix("content-type:") {
            content_type = Some(v.trim().to_string());
        }
    }
    Some(Part {
        name: name?,
        filename,
        content_type,
        data: data.to_vec(),
    })
}

fn find_sub(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || hay.len() < needle.len() {
        return None;
    }
    hay.windows(needle.len()).position(|w| w == needle)
}

fn field(parts: &[Part], key: &str) -> Option<String> {
    parts
        .iter()
        .find(|p| p.name == key && p.filename.is_none())
        .map(|p| String::from_utf8_lossy(&p.data).into_owned())
}

/// whisper-server `/inference` fields we forward, live-verified on
/// master b5130 (probe + binary strings). Everything else a client
/// sends is dropped rather than relayed into an upstream 400 — except
/// `timestamp_granularities`, which is CONSUMED gateway-side (see
/// `granularity_plan`): it forces the `verbose_json` shape and turns on
/// token-level timing, whose output the gateway folds into
/// `segments[].words`.
const FORWARDED_FIELDS: &[&str] = &[
    "response_format",
    "language",
    "temperature",
    "prompt",
    "beam_size",
    "no_timestamps",
    "temperature_inc",
    "best_of",
    "offset_t",
    "offset_n",
    "duration",
    "max_context",
    "max_len",
    "split_on_word",
    "word_thold",
    "entropy_thold",
    "logprob_thold",
    "no_fallback",
    "carry_initial_prompt",
    "detect_language",
    "audio_ctx",
    // Token-level timing and non-speech suppression — form-field names
    // verified in the b5130 binary strings.
    "token_timestamps",
    "suppress_non_speech",
    "suppress_nst",
    // Energy-VAD tuning (Silero; the child boots the VAD model when the
    // gateway is configured with one).
    "vad_simple",
    "vad_threshold",
    "vad_min_speech_duration_ms",
    "vad_min_silence_duration_ms",
    "vad_max_speech_duration_s",
    "vad_speech_pad_ms",
    "vad_samples_overlap",
    // Diarization and verbose-json shaping — live-probed on the b5130
    // child: `diarize=true` labels each segment with its channel's
    // speaker index, `no_language_probabilities=true` drops the
    // language_probabilities/detected_language keys from verbose_json.
    "diarize",
    "no_language_probabilities",
    // `translate` is handled by the caller: the transcriptions route
    // relays it as sent; the translations route owns it (always true).
];

/// Rebuilt `/inference` form fields (pure): whitelisted client fields
/// verbatim, `translate` per the route (`force_translate` overrides any
/// client value — a translations request always translates), and the
/// `response_format=json` default that keeps non-verbose clients on the
/// machine-readable shape (observed live: no default upstream).
fn forwarded_fields(parts: &[Part], force_translate: bool) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for key in FORWARDED_FIELDS {
        if let Some(v) = field(parts, key) {
            out.push(((*key).to_string(), v));
        }
    }
    let translate = if force_translate {
        "true".to_string()
    } else {
        field(parts, "translate").unwrap_or_else(|| "false".to_string())
    };
    out.push(("translate".to_string(), translate));
    if field(parts, "response_format").is_none() {
        out.push(("response_format".to_string(), "json".to_string()));
    }
    out
}

/// Rebuilt fields for the streaming lane: the whitelist minus the knobs
/// that would crop or shift a WINDOW instead of the file (a client's
/// `duration`/`offset_t`/`no_timestamps` mean the whole recording, and
/// timestamps are the payload of every streaming event), with
/// `response_format=verbose_json` forced — segments and per-chunk
/// durations are what the SSE events are built from.
fn forwarded_fields_stream(parts: &[Part], force_translate: bool) -> Vec<(String, String)> {
    const WINDOW_SHAPING: &[&str] = &["response_format", "no_timestamps", "duration", "offset_t"];
    forwarded_fields(parts, force_translate)
        .into_iter()
        .filter(|(k, _)| !WINDOW_SHAPING.contains(&k.as_str()))
        .chain([("response_format".to_string(), "verbose_json".to_string())])
        .collect()
}

/// What the client asked of `timestamp_granularities`. `OpenAI`'s contract
/// only defines `word` and `segment`, and both only ride on
/// `verbose_json` — the gateway enforces exactly that (and derives the
/// `words` arrays itself; see `enrich_verbose_json_with_words`).
#[derive(Debug, Clone, Copy, Default)]
struct Granularity {
    word: bool,
    segment: bool,
}

/// Field names a multipart client may carry the granularities in:
/// `OpenAI` SDKs repeat `timestamp_granularities[]` per value; curl users
/// send one `timestamp_granularities` part with comma-separated values.
/// Both parse; values may not mix casings of unknown words.
fn granularity_plan(parts: &[Part]) -> Result<Option<Granularity>, String> {
    let mut raw: Vec<String> = Vec::new();
    for name in ["timestamp_granularities[]", "timestamp_granularities"] {
        for p in parts
            .iter()
            .filter(|p| p.name == name && p.filename.is_none())
        {
            raw.extend(
                String::from_utf8_lossy(&p.data)
                    .split(',')
                    .map(str::trim)
                    .filter(|v| !v.is_empty())
                    .map(ToOwned::to_owned),
            );
        }
    }
    if raw.is_empty() {
        return Ok(None);
    }
    let mut gran = Granularity::default();
    for value in &raw {
        match value.to_ascii_lowercase().as_str() {
            "word" => gran.word = true,
            "segment" => gran.segment = true,
            other => {
                return Err(format!(
                    "timestamp_granularities accepts 'word' or 'segment' — got {other:?}. \
                     Word timings come back as segments[].words on verbose_json (derived \
                     from the engine's token timestamps, never interpolated); segment \
                     timings are always present on verbose_json"
                ));
            }
        }
    }
    // OpenAI's contract: granularities only exist on verbose_json. A
    // client naming another format says two contradictory things —
    // teach instead of silently picking one.
    if let Some(fmt) = field(parts, "response_format")
        && fmt.trim() != "verbose_json"
    {
        return Err(format!(
            "timestamp_granularities requires response_format=verbose_json — request \
             also named {fmt:?}: drop the explicit format (verbose_json is forced) or \
             set it to verbose_json"
        ));
    }
    Ok(Some(gran))
}

/// Apply a granularity plan to the rebuilt field set: `verbose_json`
/// forced (segment/word payloads live on that shape) and word requests
/// turn on the engine's token-level timing.
fn apply_granularity(fields: &mut Vec<(String, String)>, gran: Granularity) {
    set_field(fields, "response_format", "verbose_json");
    if gran.word {
        set_field(fields, "token_timestamps", "true");
    }
}

/// Replace an existing field's value or append it — the rebuilt form
/// must never carry duplicates (upstream reads the first occurrence).
fn set_field(fields: &mut Vec<(String, String)>, key: &str, value: &str) {
    if let Some(slot) = fields.iter_mut().find(|(n, _)| n == key) {
        slot.1 = value.to_string();
    } else {
        fields.push((key.to_string(), value.to_string()));
    }
}

/// `response_format=diarized_json` — a gateway-built format, not an
/// engine one: the child gets `verbose_json` + `diarize=true` and the
/// reply is mapped to the documented diarized shape. Stereo input is
/// what makes the labels meaningful (channel 0 = speaker 0, channel 1 =
/// speaker 1); mono collapses to a single speaker.
fn diarize_requested(parts: &[Part]) -> bool {
    field(parts, "response_format").is_some_and(|f| f.trim() == "diarized_json")
}

/// Engine diarization prefixes segment text with "(speaker N)" — the
/// mapped format carries the label as a field, so the prefix goes.
fn strip_speaker_prefix(text: &str) -> &str {
    let t = text.trim_start();
    let Some(rest) = t.strip_prefix("(speaker ") else {
        return t;
    };
    let Some((_, tail)) = rest.split_once(')') else {
        return t;
    };
    let body = tail.strip_prefix(' ').unwrap_or(tail);
    if body.is_empty() {
        // A bare "(speaker N)" prefix leaves nothing behind.
        return "";
    }
    body
}

/// Map the engine's diarized `verbose_json` to the `diarized_json` reply:
/// `{duration, text, segments[{id, start, end, text, speaker}]}`. The
/// speaker value rides verbatim (the engine's channel index); a segment
/// the engine left unlabeled omits the field instead of guessing.
/// `None` when the body carries no segments array — the caller fails
/// loud rather than relaying a wrong shape under a `diarized_json` name.
fn to_diarized_json(body: &serde_json::Value) -> Option<serde_json::Value> {
    let segments = body.get("segments")?.as_array()?;
    let mut mapped = Vec::with_capacity(segments.len());
    let mut texts: Vec<String> = Vec::with_capacity(segments.len());
    for (idx, seg) in segments.iter().enumerate() {
        let raw = seg
            .get("text")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        let clean = strip_speaker_prefix(raw);
        texts.push(clean.to_string());
        let mut out = serde_json::json!({
            "id": seg.get("id").cloned().unwrap_or(serde_json::json!(idx)),
            "start": seg.get("start").cloned().unwrap_or(serde_json::json!(0.0)),
            "end": seg.get("end").cloned().unwrap_or(serde_json::json!(0.0)),
            "text": clean,
        });
        if let Some(speaker) = seg.get("speaker") {
            out["speaker"] = speaker.clone();
        }
        mapped.push(out);
    }
    Some(serde_json::json!({
        "duration": body.get("duration").cloned().unwrap_or(serde_json::json!(0)),
        "text": texts.join(" ").trim().to_string(),
        "segments": mapped,
    }))
}

/// `OpenAI` audio params the local whisper lane cannot honor — each gets
/// a teaching 400 that names the nearest supported thing, never a
/// silent drop.
fn unsupported_field_error(parts: &[Part]) -> Option<String> {
    if let Some(v) = field(parts, "include") {
        return Some(format!(
            "include={v:?} is an OpenAI cloud extra (logprobs) the local whisper lane \
             cannot produce — drop it"
        ));
    }
    if let Some(v) = field(parts, "keywords") {
        return Some(format!(
            "keywords={v:?} biases gpt-transcribe on OpenAI's cloud; the local lane \
             takes `prompt` for vocabulary steering instead"
        ));
    }
    None
}

/// `(start, end)` timing off one `verbose_json` token entry. whisper.cpp
/// serves token timing as a `timestamps` pair; upstream units are
/// milliseconds while segment `start`/`end` are seconds (an upstream
/// quirk the caller normalizes against the segment bound).
fn token_times(token: &serde_json::Value) -> Option<(f64, f64)> {
    let ts = token.get("timestamps")?.as_array()?;
    if ts.len() < 2 {
        return None;
    }
    let (s, e) = (ts[0].as_f64()?, ts[1].as_f64()?);
    (e > s).then_some((s, e))
}

/// Group one segment's timed tokens into `OpenAI` `words` entries, using
/// the engine's own word-boundary convention (a token whose text begins
/// with a space or `▁` starts the next word — the `SentencePiece` marker).
/// Times come only from engine tokens: a word missing any token timing
/// is omitted rather than interpolated. Returns the words (empty when
/// nothing derivable).
fn words_from_tokens(tokens: &[serde_json::Value], seg_end: Option<f64>) -> Vec<serde_json::Value> {
    let mut words: Vec<serde_json::Value> = Vec::new();
    let mut current: Option<(String, f64, f64)> = None;
    let flush = |current: &mut Option<(String, f64, f64)>, words: &mut Vec<serde_json::Value>| {
        if let Some((text, start, end)) = current.take()
            && !text.is_empty()
        {
            words.push(serde_json::json!({"word": text, "start": start, "end": end}));
        }
    };
    for token in tokens {
        let Some(text) = token.get("text").and_then(serde_json::Value::as_str) else {
            continue;
        };
        let starts_word = text.starts_with(' ') || text.starts_with('▁');
        let piece = text.trim_start_matches([' ', '▁']);
        if starts_word {
            flush(&mut current, &mut words);
        }
        if piece.is_empty() {
            continue; // a bare separator: word already flushed
        }
        match token_times(token) {
            Some((start, end)) => match &mut current {
                Some((word, s, e)) if !starts_word => {
                    word.push_str(piece);
                    *e = end; // word extends through its last timed token
                }
                _ => current = Some((piece.to_string(), start, end)),
            },
            None => {
                if !starts_word && let Some((word, _, _)) = &mut current {
                    word.push_str(piece); // text rides along, timing stays absent
                } else {
                    flush(&mut current, &mut words); // untimed word start: cannot time it honestly
                }
            }
        }
    }
    flush(&mut current, &mut words);
    // Upstream quirk: token `timestamps` are milliseconds while segment
    // bounds are seconds. A word end beyond the segment's own end (plus
    // slack) but sane once divided by 1000 was milliseconds. Segments
    // past ~9 minutes are ambiguous under this rule — verified against
    // the running child in live validation.
    if let Some(seg_end) = seg_end {
        for w in &mut words {
            let end = w["end"].as_f64().unwrap_or(f64::INFINITY);
            let start = w["start"].as_f64().unwrap_or(f64::INFINITY);
            if end > seg_end + 1.5 && start * 0.001 <= seg_end + 1.5 && end * 0.001 <= seg_end + 1.5
            {
                w["start"] = serde_json::json!(start * 0.001);
                w["end"] = serde_json::json!(end * 0.001);
            }
        }
    }
    words
}

/// Fold token-level timing into `segments[].words` on a `verbose_json`
/// body (in place, only what the engine actually timed — never
/// interpolated). Engine-served `words` arrays pass through untouched.
/// Returns whether any words surfaced.
fn enrich_verbose_json_with_words(body: &mut serde_json::Value) -> bool {
    let Some(segments) = body.get_mut("segments").and_then(|s| s.as_array_mut()) else {
        return false;
    };
    let mut any = false;
    for seg in segments.iter_mut() {
        if seg
            .get("words")
            .and_then(|w| w.as_array())
            .is_some_and(|a| !a.is_empty())
        {
            any = true; // the engine already served word timings
            continue;
        }
        let Some(tokens) = seg.get("tokens").and_then(|t| t.as_array().cloned()) else {
            continue;
        };
        let seg_end = seg.get("end").and_then(serde_json::Value::as_f64);
        let words = words_from_tokens(&tokens, seg_end);
        if !words.is_empty() {
            seg["words"] = serde_json::json!(words);
            any = true;
        }
    }
    any
}

/// One wire-encoded SSE frame. `serde_json`'s Display is single-line
/// JSON, so the data field never breaks early.
fn sse_frame(event: &str, data: &serde_json::Value) -> Vec<u8> {
    format!("event: {event}\ndata: {data}\n\n").into_bytes()
}

/// Shift a decoded chunk's segment timestamps onto the source
/// recording's timeline (`start`/`end` are chunk-relative seconds in
/// upstream's `verbose_json`). Non-numeric or missing fields pass through
/// untouched — the event carries what the decoder reported.
#[allow(clippy::cast_precision_loss)] // ms → s float for JSON segment times
fn rebase_segments(segments: &mut serde_json::Value, offset_ms: u64) {
    let Some(list) = segments.as_array_mut() else {
        return;
    };
    let shift = offset_ms as f64 / 1000.0;
    for seg in list.iter_mut() {
        for key in ["start", "end"] {
            if let Some(v) = seg.get_mut(key).and_then(|v| v.as_f64()) {
                seg[key] = serde_json::json!(v + shift);
            }
        }
    }
}

/// One `/inference` POST against an already-ensured child: rebuilds
/// the multipart from the caller's field set and returns the raw
/// upstream triple. Shared by the sync forward, the async job task,
/// and the streaming lane (one call per progressive window).
async fn inference_post(
    state: &Arc<AppState>,
    port: u16,
    file_bytes: &[u8],
    filename: &str,
    mime: &str,
    fields: &[(String, String)],
) -> Result<(StatusCode, String, Bytes), (u16, String)> {
    let mut form = reqwest::multipart::Form::new().part(
        "file",
        reqwest::multipart::Part::bytes(file_bytes.to_vec())
            .file_name(filename.to_string())
            .mime_str(mime)
            .unwrap_or_else(|_| reqwest::multipart::Part::bytes(file_bytes.to_vec())),
    );
    for (key, v) in fields {
        form = form.text(key.clone(), v.clone());
    }
    let url = format!("http://127.0.0.1:{port}/inference");
    let resp = match state.http.post(&url).multipart(form).send().await {
        Ok(r) => r,
        Err(e) => return Err((502, format!("whisper inference: {e:#}"))),
    };
    let status = StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let ct = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/json")
        .to_string();
    let bytes = resp.bytes().await.unwrap_or_default();
    Ok((status, ct, bytes))
}

/// Resolve the configured `whisper_vad_model` to a spawn-ready path:
/// a bare name is looked up beside the whisper-server binary (the
/// engine dir ships `silero-vad ggml` files there), anything else is
/// taken as a user-owned path. A configured model that cannot be
/// resolved is a teaching error — VAD is an explicit user intent, and
/// silently booting without it would hide the miss.
fn resolve_vad_model(
    state: &AppState,
    bin: &std::path::Path,
) -> Result<Option<std::path::PathBuf>, String> {
    let Some(name) = state.config.whisper_vad_model.as_deref() else {
        return Ok(None);
    };
    let name = name.trim();
    if name.is_empty() {
        return Ok(None);
    }
    let candidate = if name.contains(std::path::is_separator) {
        std::path::PathBuf::from(name)
    } else {
        bin.parent()
            .map_or_else(|| std::path::PathBuf::from(name), |dir| dir.join(name))
    };
    if candidate.is_file() {
        Ok(Some(candidate))
    } else {
        Err(format!(
            "whisper_vad_model '{name}' not found (looked at {}) — place the \
             Silero VAD ggml in the whisper engine dir or give an absolute path; \
             unset the knob to boot without VAD",
            candidate.display()
        ))
    }
}

/// Local lane transport: ensure the lazy child (hot model swap on size
/// change), forward the multipart fields whisper-server understands,
/// and return the raw upstream triple. Shared by the sync path and the
/// async job task — the Err codes/messages are the sync route's exact
/// error surface, so both lanes fail identically.
#[allow(clippy::too_many_arguments)] // natural lane surface: state, parts, file, model, translate, stream, gran
async fn forward_local_raw(
    state: &Arc<AppState>,
    parts: &[Part],
    file: &Part,
    size: &str,
    bin: &std::path::Path,
    lib_dir: &std::path::Path,
    force_translate: bool,
    gran: Option<Granularity>,
    diarize: bool,
) -> Result<((StatusCode, String, Bytes), bool), (u16, String)> {
    let Some(model_path) = whisper::model_file(&state.dirs, size) else {
        return Err((500, format!("whisper model ggml-{size}.bin vanished")));
    };
    let vad_model = match resolve_vad_model(state, bin) {
        Ok(v) => v,
        Err(msg) => return Err((400, msg)),
    };
    let port = match state
        .whisper
        .ensure(
            size,
            &model_path,
            bin,
            lib_dir,
            std::time::Duration::from_mins(2),
            vad_model.as_deref(),
        )
        .await
    {
        Ok(p) => p,
        Err(e) => return Err((502, format!("whisper-server: {e:#}"))),
    };
    let mime = file
        .content_type
        .clone()
        .unwrap_or_else(|| "application/octet-stream".to_string());
    let mut fields = forwarded_fields(parts, force_translate);
    if let Some(gran) = gran {
        apply_granularity(&mut fields, gran);
    }
    if diarize {
        // diarized_json is gateway-built: the child decodes verbose_json
        // with diarization on, the reply is mapped on the way out.
        set_field(&mut fields, "response_format", "verbose_json");
        set_field(&mut fields, "diarize", "true");
    }
    let triple = inference_post(
        state,
        port,
        &file.data,
        file.filename.as_deref().unwrap_or("audio"),
        &mime,
        &fields,
    )
    .await?;
    let ((status, ct, bytes), words_derived) = match triple {
        (status, _ct, bytes) if diarize && status.is_success() => {
            // diarized_json: map the engine's diarized verbose_json to
            // the documented shape. A success body without a segments
            // array is not mappable — fail loud, never hand back
            // verbose_json under a diarized_json name.
            let parsed: serde_json::Value = match serde_json::from_slice(&bytes) {
                Ok(v) => v,
                Err(_) => {
                    return Err((
                        502,
                        "whisper-server returned a non-JSON body with diarization on — \
                         cannot build diarized_json"
                            .to_string(),
                    ));
                }
            };
            match to_diarized_json(&parsed) {
                Some(mapped) => match serde_json::to_vec(&mapped) {
                    Ok(out) => (
                        (status, "application/json".to_string(), Bytes::from(out)),
                        false,
                    ),
                    Err(_) => {
                        return Err((502, "diarized_json serialization failed".to_string()));
                    }
                },
                None => {
                    return Err((
                        502,
                        "whisper-server returned no segments with diarization on — cannot \
                         build diarized_json"
                            .to_string(),
                    ));
                }
            }
        }
        (status, ct, bytes) if gran.is_some_and(|g| g.word) && status.is_success() => {
            // Word granularity: fold the engine's token timings into
            // segments[].words before the body leaves the gateway. Parse
            // failures leave the body verbatim — the client still gets
            // the engine's own verbose_json.
            let mut v: serde_json::Value = match serde_json::from_slice(&bytes) {
                Ok(v) => v,
                Err(_) => return Ok(((status, ct, bytes), false)),
            };
            let derived = enrich_verbose_json_with_words(&mut v);
            match serde_json::to_vec(&v) {
                Ok(enriched) => ((status, ct, Bytes::from(enriched)), derived),
                Err(_) => ((status, ct, bytes), false),
            }
        }
        other => (other, false),
    };
    Ok(((status, ct, bytes), words_derived))
}

/// Sync local lane: same contract as before the async split — the raw
/// triple rendered as a verbatim passthrough, errors as `openai_error`
/// with the historical codes and messages.
#[allow(clippy::too_many_arguments)] // same surface as forward_local_raw
async fn forward_local(
    state: &Arc<AppState>,
    parts: &[Part],
    file: &Part,
    size: &str,
    bin: &std::path::Path,
    lib_dir: &std::path::Path,
    force_translate: bool,
    gran: Option<Granularity>,
    diarize: bool,
) -> Response {
    match forward_local_raw(
        state,
        parts,
        file,
        size,
        bin,
        lib_dir,
        force_translate,
        gran,
        diarize,
    )
    .await
    {
        Ok(((status, ct, bytes), words_derived)) => {
            let mut builder = Response::builder()
                .status(status)
                .header(axum::http::header::CONTENT_TYPE, ct);
            // Word granularity reports its own outcome — an unavailable
            // verdict is a capability statement, not a failure.
            if gran.is_some_and(|g| g.word) {
                builder = builder.header(
                    "x-blazar-word-timestamps",
                    if words_derived {
                        "derived"
                    } else {
                        "unavailable"
                    },
                );
            }
            builder
                .body(axum::body::Body::from(bytes))
                .unwrap_or_else(|_| openai_error(500, "response build").into_response())
        }
        Err((code, msg)) => openai_error(code, &msg),
    }
}

/// Registry ceiling: abandoned handles must not grow the map forever.
/// At the cap the oldest terminal job is evicted first (nobody polls
/// those anymore); if all are live, the oldest overall goes — its task
/// is aborted, the same as an explicit cancel.
const MAX_AUDIO_JOBS: usize = 256;

/// One job's lifecycle. `Queued` exists only between handle reservation
/// and the task's first instruction; `Completed` carries the upstream
/// triple verbatim so the poll response is the sync response, replayed.
#[derive(Clone)]
enum AudioJobState {
    Queued,
    Running,
    Completed {
        status: u16,
        content_type: String,
        body: Arc<Vec<u8>>,
    },
    Failed(String),
    Cancelled,
}

impl AudioJobState {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Running => "running",
            Self::Completed { .. } => "completed",
            Self::Failed(_) => "failed",
            Self::Cancelled => "cancelled",
        }
    }

    fn terminal(&self) -> bool {
        matches!(
            self,
            Self::Completed { .. } | Self::Failed(_) | Self::Cancelled
        )
    }
}

struct AudioJobEntry {
    state: AudioJobState,
    handle: Option<tokio::task::JoinHandle<()>>,
    seq: u64,
    created_at: u64,
}

/// Gateway-owned async audio jobs. Every method takes the inner lock
/// briefly and never across an await — job tasks re-enter the registry
/// to update their own state, so a held lock would deadlock the lane.
#[derive(Default)]
pub struct AudioJobs {
    inner: std::sync::Mutex<AudioJobStore>,
}

#[derive(Default)]
struct AudioJobStore {
    jobs: HashMap<String, AudioJobEntry>,
    seq: u64,
    boot_nanos: u64,
}

impl AudioJobs {
    /// Jobs not yet terminal (queued or running) — the `/metrics`
    /// activity gauge (audit MM9).
    #[must_use]
    pub fn active_count(&self) -> usize {
        let store = self.inner.lock().expect("audio jobs lock");
        store
            .jobs
            .values()
            .filter(|j| matches!(j.state, AudioJobState::Queued | AudioJobState::Running))
            .count()
    }

    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: std::sync::Mutex::new(AudioJobStore {
                boot_nanos: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0, |d| u64::try_from(d.as_nanos()).unwrap_or(0)),
                ..AudioJobStore::default()
            }),
        }
    }

    /// Reserve a job id and Queued slot. Called just before the task
    /// spawns; `set_handle` attaches the `JoinHandle` right after.
    fn reserve(&self) -> String {
        let mut store = self.inner.lock().expect("audio jobs lock");
        store.seq += 1;
        let id = format!("aj{}-{}", store.boot_nanos, store.seq);
        let seq = store.seq;
        let created_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        if store.jobs.len() >= MAX_AUDIO_JOBS {
            evict_one(&mut store);
        }
        store.jobs.insert(
            id.clone(),
            AudioJobEntry {
                state: AudioJobState::Queued,
                handle: None,
                seq,
                created_at,
            },
        );
        id
    }

    fn set_handle(&self, id: &str, handle: tokio::task::JoinHandle<()>) {
        let mut store = self.inner.lock().expect("audio jobs lock");
        if let Some(entry) = store.jobs.get_mut(id) {
            // A cancel that fired before the handle landed already
            // marked the job terminal — keep the handle anyway so a
            // later drop can still abort the straggler.
            entry.handle = Some(handle);
        }
    }

    fn mark_running(&self, id: &str) {
        let mut store = self.inner.lock().expect("audio jobs lock");
        if let Some(entry) = store.jobs.get_mut(id)
            && matches!(entry.state, AudioJobState::Queued)
        {
            entry.state = AudioJobState::Running;
        }
    }

    fn finish(&self, id: &str, status: u16, content_type: &str, body: Vec<u8>) {
        let mut store = self.inner.lock().expect("audio jobs lock");
        if let Some(entry) = store.jobs.get_mut(id)
            && matches!(entry.state, AudioJobState::Queued | AudioJobState::Running)
        {
            entry.state = AudioJobState::Completed {
                status,
                content_type: content_type.to_string(),
                body: Arc::new(body),
            };
        }
    }

    fn fail(&self, id: &str, message: String) {
        let mut store = self.inner.lock().expect("audio jobs lock");
        if let Some(entry) = store.jobs.get_mut(id)
            && matches!(entry.state, AudioJobState::Queued | AudioJobState::Running)
        {
            entry.state = AudioJobState::Failed(message);
        }
    }

    /// Cancel a queued/running job: abort its task, mark Cancelled.
    /// `None` = unknown id; `Some(false)` = already terminal (idempotent
    /// poll-friendly no-op).
    pub(crate) fn cancel(&self, id: &str) -> Option<bool> {
        let handle = {
            let mut store = self.inner.lock().expect("audio jobs lock");
            let entry = store.jobs.get_mut(id)?;
            if entry.state.terminal() {
                return Some(false);
            }
            entry.state = AudioJobState::Cancelled;
            entry.handle.take()
        };
        // Abort outside the lock: abort schedules onto the runtime and
        // can run arbitrary Drop code in the aborted future.
        if let Some(handle) = handle {
            handle.abort();
        }
        Some(true)
    }

    /// Poll payload for `/v1/audio/jobs/{id}`: status plus the upstream
    /// triple on completion (body parsed as JSON when possible, raw
    /// text otherwise — transcription bodies are JSON in practice).
    pub(crate) fn payload(&self, id: &str) -> Option<serde_json::Value> {
        let store = self.inner.lock().expect("audio jobs lock");
        let entry = store.jobs.get(id)?;
        let mut payload = serde_json::json!({
            "id": id,
            "object": "whisper.job",
            "status": entry.state.as_str(),
            "created_at": entry.created_at,
        });
        match &entry.state {
            AudioJobState::Completed {
                status,
                content_type,
                body,
            } => {
                payload["status_code"] = serde_json::json!(status);
                payload["content_type"] = serde_json::json!(content_type);
                payload["result"] = serde_json::from_slice::<serde_json::Value>(body)
                    .unwrap_or_else(|_| {
                        serde_json::Value::String(String::from_utf8_lossy(body).into_owned())
                    });
            }
            AudioJobState::Failed(message) => {
                payload["error"] = serde_json::json!(message);
            }
            _ => {}
        }
        Some(payload)
    }
}

/// Registry eviction: oldest terminal job first; if everything is live,
/// the oldest overall (aborted — an implicit cancel).
fn evict_one(store: &mut AudioJobStore) {
    let victim = store
        .jobs
        .iter()
        .min_by_key(|(_, e)| (u64::from(!e.state.terminal()), e.seq))
        .map(|(id, _)| id.clone());
    if let Some(id) = victim
        && let Some(entry) = store.jobs.remove(&id)
        && let Some(handle) = entry.handle
    {
        handle.abort();
    }
}

/// The gateway-only `async` knob: a plain form field (the audio routes
/// are multipart; images uses the same knob as a JSON field). Honored
/// on the local lane only — a remote forward streams the client's body
/// untouched, remotes keep their own sync contracts.
fn wants_async(parts: &[Part]) -> bool {
    field(parts, "async").is_some_and(|v| {
        let v = v.trim();
        v.eq_ignore_ascii_case("true") || v == "1"
    })
}

/// The `stream` knob (F6): same multipart-field convention as `async`.
/// Progressive SSE transcription on the local lane only — remotes keep
/// whatever contract they have upstream.
fn wants_stream(parts: &[Part]) -> bool {
    field(parts, "stream").is_some_and(|v| {
        let v = v.trim();
        v.eq_ignore_ascii_case("true") || v == "1"
    })
}

/// `stream=true` (F6): progressive transcription over SSE. WAV PCM
/// inputs split into `whisper_stream_chunk_ms` windows (frame-aligned,
/// lossless — see `whisper::split_wav`) decoded sequentially on the
/// same lazy child, so text surfaces every window instead of only when
/// the whole file finishes. Upstream is mutex-serialized batch (no
/// streaming surface exists to relay), so sequential windows cost zero
/// throughput versus one monolithic decode. Events: `chunk.completed`
/// per window (source-relative timestamps), one final
/// `transcript.completed`, `error` for mid-stream failures (once the
/// event stream is open, failures stop being HTTP status codes — every
/// major SSE API behaves the same), and `: keep-alive` comments during
/// long window decodes so proxies do not reap an idle connection.
/// Non-WAV inputs decode once and stream as a single final event: the
/// client contract stays uniform, and the format limitation is named
/// in the capabilities payload.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_lines)] // one streaming transcription relay: parse, rebased SSE, usage
#[allow(clippy::unused_async)] // axum handler shape; body streams via the response
#[allow(clippy::cast_possible_truncation)] // duration seconds → ms for segment math
#[allow(clippy::cast_sign_loss)] // negative durations are invalid input; clamp-by-cast is acceptable
async fn forward_local_stream(
    state: &Arc<AppState>,
    parts: &[Part],
    file: &Part,
    size: &str,
    bin: &std::path::Path,
    lib_dir: &std::path::Path,
    force_translate: bool,
) -> Response {
    let Some(model_path) = whisper::model_file(&state.dirs, size) else {
        return openai_error(500, &format!("whisper model ggml-{size}.bin vanished"));
    };
    let filename = file.filename.clone().unwrap_or_else(|| "audio".to_string());
    let fields = forwarded_fields_stream(parts, force_translate);
    // Emission plan before the stream opens: WAV splits become windows;
    // anything else is one window carrying the original bytes and mime.
    let windows: Vec<(Vec<u8>, u64, String)> =
        match whisper::split_wav(&file.data, state.config.whisper_stream_chunk_ms) {
            Some(split) => split
                .into_iter()
                .map(|c| (c.bytes, c.offset_ms, "audio/wav".to_string()))
                .collect(),
            None => vec![(
                file.data.clone(),
                0,
                file.content_type
                    .clone()
                    .unwrap_or_else(|| "application/octet-stream".to_string()),
            )],
        };
    let (tx, rx) = tokio::sync::mpsc::channel::<Vec<u8>>(4);
    let task_state = Arc::clone(state);
    let bin = bin.to_path_buf();
    let lib_dir = lib_dir.to_path_buf();
    let size = size.to_string();
    tokio::spawn(async move {
        // Cold child boot can take its whole ready timeout — that failure
        // and every later one surface as `error` events, never silence.
        let vad_model = match resolve_vad_model(&task_state, &bin) {
            Ok(v) => v,
            Err(msg) => {
                let frame = sse_frame("error", &serde_json::json!({"message": msg}));
                let _ = tx.send(frame).await;
                return;
            }
        };
        let port = match task_state
            .whisper
            .ensure(
                &size,
                &model_path,
                &bin,
                &lib_dir,
                std::time::Duration::from_mins(2),
                vad_model.as_deref(),
            )
            .await
        {
            Ok(p) => p,
            Err(e) => {
                let frame = sse_frame(
                    "error",
                    &serde_json::json!({"message": format!("whisper-server: {e:#}")}),
                );
                let _ = tx.send(frame).await;
                return;
            }
        };
        let mut all_segments: Vec<serde_json::Value> = Vec::new();
        let mut texts: Vec<String> = Vec::new();
        let mut duration_ms = 0u64;
        let mut language = serde_json::Value::Null;
        for (i, (bytes, offset_ms, mime)) in windows.iter().enumerate() {
            let attempt = inference_post(&task_state, port, bytes, &filename, mime, &fields);
            tokio::pin!(attempt);
            let mut keepalive = tokio::time::interval(std::time::Duration::from_secs(15));
            keepalive.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            let triple = loop {
                tokio::select! {
                    _ = keepalive.tick() => {
                        let _ = tx.send(b": keep-alive\n\n".to_vec()).await;
                    }
                    r = &mut attempt => break r,
                }
            };
            let (status, _ct, body) = match triple {
                Ok(t) => t,
                Err((code, msg)) => {
                    let frame = sse_frame(
                        "error",
                        &serde_json::json!({"message": msg, "status": code, "chunk": i}),
                    );
                    let _ = tx.send(frame).await;
                    return;
                }
            };
            if !status.is_success() {
                let frame = sse_frame(
                    "error",
                    &serde_json::json!({
                        "message": format!("whisper inference: HTTP {status}"),
                        "status": status.as_u16(),
                        "chunk": i,
                        "body": String::from_utf8_lossy(&body),
                    }),
                );
                let _ = tx.send(frame).await;
                return;
            }
            let mut v: serde_json::Value = match serde_json::from_slice(&body) {
                Ok(v) => v,
                Err(e) => {
                    let frame = sse_frame(
                        "error",
                        &serde_json::json!({
                            "message": format!("unparseable verbose_json from child: {e}"),
                            "chunk": i,
                        }),
                    );
                    let _ = tx.send(frame).await;
                    return;
                }
            };
            if let Some(lang) = v.get("language").filter(|l| !l.is_null()) {
                language = lang.clone();
            }
            let text = v
                .get("text")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("")
                .to_string();
            let mut segments = v
                .get_mut("segments")
                .cloned()
                .unwrap_or_else(|| serde_json::json!([]));
            rebase_segments(&mut segments, *offset_ms);
            if let Some(chunk_secs) = v.get("duration").and_then(serde_json::Value::as_f64) {
                duration_ms = offset_ms.saturating_add((chunk_secs * 1000.0) as u64);
            }
            let frame = sse_frame(
                "chunk.completed",
                &serde_json::json!({
                    "index": i,
                    "offset_ms": offset_ms,
                    "text": text,
                    "segments": segments,
                }),
            );
            all_segments.extend(segments.as_array().cloned().unwrap_or_default());
            texts.push(text);
            if tx.send(frame).await.is_err() {
                return; // client hung up: stop decoding, discard the rest
            }
        }
        let final_event = sse_frame(
            "transcript.completed",
            &serde_json::json!({
                "duration_ms": duration_ms,
                "language": language,
                "text": texts.concat(),
                "segments": all_segments,
                "chunks": windows.len(),
            }),
        );
        let _ = tx.send(final_event).await;
    });
    let body = axum::body::Body::from_stream(
        tokio_stream::wrappers::ReceiverStream::new(rx).map(Ok::<_, std::convert::Infallible>),
    );
    Response::builder()
        .status(200)
        .header(axum::http::header::CONTENT_TYPE, "text/event-stream")
        .header(axum::http::header::CACHE_CONTROL, "no-cache")
        // nginx and friends buffer proxied SSE unless told not to.
        .header("x-accel-buffering", "no")
        .body(body)
        .unwrap_or_else(|_| openai_error(500, "response build").into_response())
}

/// Submit an async job: reserve the handle, spawn the forward task,
/// answer with the gateway-owned job envelope (same shape vocabulary as
/// the images lane's `async` handles). No awaits — the spawned task owns
/// all the waiting.
#[allow(clippy::too_many_arguments)] // mirrors forward_local_raw plus owned parts
fn submit_async(
    state: &Arc<AppState>,
    parts: Vec<Part>,
    file: Part,
    size: String,
    bin: &std::path::Path,
    lib_dir: &std::path::Path,
    force_translate: bool,
    gran: Option<Granularity>,
    diarize: bool,
) -> Response {
    let id = state.audio_jobs.reserve();
    // Durable ledger write-through: the row outlives the gateway, the
    // input artifact makes a resubmit after an abandon possible without
    // re-uploading (bounded — oversized uploads skip the copy, the row
    // still lands).
    let mut request = serde_json::json!({
        "size": size,
        "filename": file.filename,
        "translate": force_translate,
    });
    if let Some(gran) = gran {
        request["timestamp_granularities"] = serde_json::json!({
            "word": gran.word,
            "segment": gran.segment,
        });
    }
    if diarize {
        request["response_format"] = serde_json::json!("diarized_json");
    }
    state.jobs.record_created(
        state,
        &id,
        "audio",
        field(&parts, "model").as_deref(),
        request,
    );
    state
        .jobs
        .record_input_artifact(&id, file.filename.as_deref(), &file.data);
    let task_id = id.clone();
    let task_state = Arc::clone(state);
    let bin = bin.to_path_buf();
    let lib_dir = lib_dir.to_path_buf();
    let task = tokio::spawn(async move {
        task_state.audio_jobs.mark_running(&task_id);
        task_state.jobs.record_running(&task_state, &task_id);
        match forward_local_raw(
            &task_state,
            &parts,
            &file,
            &size,
            &bin,
            &lib_dir,
            force_translate,
            gran,
            diarize,
        )
        .await
        {
            Ok(((status, ct, bytes), _words_derived)) => {
                // Word-derived bodies are already enriched inside
                // forward_local_raw — pollers see the same words the
                // sync lane returns.
                task_state
                    .audio_jobs
                    .finish(&task_id, status.as_u16(), &ct, bytes.to_vec());
                task_state
                    .jobs
                    .record_completed(&task_state, &task_id, &bytes, &ct);
            }
            Err((_code, msg)) => {
                task_state.audio_jobs.fail(&task_id, msg.clone());
                task_state.jobs.record_failed(&task_state, &task_id, &msg);
            }
        }
    });
    state.audio_jobs.set_handle(&id, task);
    let created_at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let body = serde_json::json!({
        "id": id,
        "object": "whisper.job",
        "status": "queued",
        "created_at": created_at,
        "poll_url": format!("/v1/audio/jobs/{id}"),
        "cancel_url": format!("/v1/audio/jobs/{id}/cancel"),
    });
    Response::builder()
        .status(200)
        .header(axum::http::header::CONTENT_TYPE, "application/json")
        .body(axum::body::Body::from(
            serde_json::to_vec(&body).unwrap_or_default(),
        ))
        .unwrap_or_else(|_| openai_error(500, "response build").into_response())
}

/// GET /v1/audio/jobs/{id} — poll a gateway-owned async audio job. The
/// live registry answers while this gateway lives; the durable ledger
/// answers after a restart (completed results and honest failures —
/// in-flight work cannot resume and is swept to `abandoned` at boot).
#[allow(clippy::unused_async)] // axum's Handler trait requires async fns
pub async fn audio_jobs_get(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(job_id): axum::extract::Path<String>,
) -> Response {
    if !job_id
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return openai_error(400, "invalid job id");
    }
    if let Some(payload) = state.audio_jobs.payload(&job_id) {
        return Response::builder()
            .status(200)
            .header(axum::http::header::CONTENT_TYPE, "application/json")
            .body(axum::body::Body::from(
                serde_json::to_vec(&payload).unwrap_or_default(),
            ))
            .unwrap_or_else(|_| openai_error(500, "response build").into_response());
    }
    // Live registry miss (evicted cap slot or a restarted gateway): the
    // ledger is the afterlife.
    if let Some(row) = state
        .with_store(|s| s.get_job(&job_id).ok().flatten())
        .flatten()
    {
        return axum::Json(crate::jobs::row_payload(&row)).into_response();
    }
    openai_error(
        404,
        "job not found — live audio handles are capped and gateway-owned; durable records \
         live at /v1/jobs/{id} (completed results survive restarts)",
    )
}

/// POST /v1/audio/jobs/{id}/cancel — abort a queued/running job. The
/// wait stops immediately; an in-flight upstream inference runs to
/// completion inside the child (whisper-server has no cancel surface —
/// the result is simply discarded). Terminal jobs answer with their
/// final state (idempotent).
#[allow(clippy::unused_async)] // axum's Handler trait requires async fns
pub async fn audio_jobs_cancel(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(job_id): axum::extract::Path<String>,
) -> Response {
    if !job_id
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return openai_error(400, "invalid job id");
    }
    // cancel() performs the abort side effect (a terminal/unknown job is
    // a no-op); either way the answer is the job's current state — 200
    // with the payload, 404 when no such job exists.
    if state.audio_jobs.cancel(&job_id) == Some(true) {
        state.jobs.record_cancelled(&state, &job_id);
    }
    if let Some(payload) = state.audio_jobs.payload(&job_id) {
        Response::builder()
            .status(200)
            .header(axum::http::header::CONTENT_TYPE, "application/json")
            .body(axum::body::Body::from(
                serde_json::to_vec(&payload).unwrap_or_default(),
            ))
            .unwrap_or_else(|_| openai_error(500, "response build").into_response())
    } else if let Some(row) = state
        .with_store(|s| s.get_job(&job_id).ok().flatten())
        .flatten()
    {
        // Registry miss but a ledger row exists (restarted gateway or an
        // evicted cap slot): close an in-flight row — its task died with
        // the old process — and answer from the row.
        if matches!(row.state.as_str(), "queued" | "running") {
            state.jobs.record_cancelled(&state, &job_id);
        }
        let fresh = state
            .with_store(|s| s.get_job(&job_id).ok().flatten())
            .flatten();
        match fresh {
            Some(fresh) => axum::Json(crate::jobs::row_payload(&fresh)).into_response(),
            None => openai_error(404, "job not found — durable records live at /v1/jobs/{id}"),
        }
    } else {
        openai_error(
            404,
            "job not found — live audio handles are capped and gateway-owned; durable records \
             live at /v1/jobs/{id}",
        )
    }
}

/// GET /v1/audio/capabilities — the audio lane's gateway-derived menu.
/// Unlike `/v1/images/capabilities` (a relay of the child's sampler
/// menu), whisper-server has no capabilities route, so this reads local
/// truth only: engine presence, pulled models, live child state, and
/// the async/idle knobs. Boots nothing.
pub async fn audio_capabilities(State(state): State<Arc<AppState>>) -> Response {
    let installed = whisper::server_bin(&state.dirs).is_some();
    let models = whisper::list_models(&state.dirs);
    let child = state
        .whisper
        .status()
        .await
        .map(|(_port, loaded)| serde_json::json!({ "alive": true, "loaded": loaded }));
    let body = serde_json::json!({
        "object": "whisper.capabilities",
        "engine": { "kind": "whisper", "installed": installed },
        "models": models,
        "child": child,
        "endpoints": [
            "/v1/audio/transcriptions",
            "/v1/audio/translations",
            "/v1/audio/jobs/{id}",
            "/v1/audio/jobs/{id}/cancel"
        ],
        "async": { "field": "async", "values": ["true", "1"], "local_lane_only": true },
        "stream": {
            "field": "stream",
            "values": ["true", "1"],
            "local_lane_only": true,
            "chunk_ms": state.config.whisper_stream_chunk_ms,
            "events": ["chunk.completed", "transcript.completed", "error"],
            "progressive": "wav pcm splits at frame boundaries; other formats decode as one \
             final event",
            "content_type": "text/event-stream",
        },
        "response_formats": [
            "json",
            "text",
            "srt",
            "verbose_json",
            "vtt",
            "diarized_json",
        ],
        "diarized_json": {
            "input": "stereo (channel 0 = speaker 0, channel 1 = speaker 1); mono collapses \
                      to a single speaker",
            "shape": "{duration, text, segments[{id, start, end, text, speaker}]} — speaker \
                      is the engine's channel index, segments the engine left unlabeled \
                      omit the field",
        },
        "timestamp_granularities": {
            "values": ["word", "segment"],
            "requires": "response_format=verbose_json — forced automatically when absent, \
                         400 when the request names another format",
            "words": "segments[].words derived from the engine's token timestamps, never \
                      interpolated — the sync lane reports derived|unavailable via the \
                      x-blazar-word-timestamps header",
        },
        "engine_fields": {
            "note": "whisper.cpp /inference knobs relayed verbatim from the request",
            "timing": ["token_timestamps", "split_on_word", "word_thold", "no_timestamps"],
            "vad": [
                "vad_simple", "vad_threshold", "vad_min_speech_duration_ms",
                "vad_min_silence_duration_ms", "vad_max_speech_duration_s",
                "vad_speech_pad_ms", "vad_samples_overlap",
            ],
            "decode": [
                "language", "temperature", "temperature_inc", "prompt", "beam_size", "best_of",
                "entropy_thold", "logprob_thold", "no_fallback", "carry_initial_prompt",
                "detect_language", "audio_ctx", "max_context", "max_len", "offset_t",
                "offset_n", "duration", "suppress_non_speech", "suppress_nst",
                "no_language_probabilities",
            ],
        },
        "idle_timeout_secs": state.config.whisper_idle_secs,
    });
    Response::builder()
        .status(200)
        .header(axum::http::header::CONTENT_TYPE, "application/json")
        .body(axum::body::Body::from(
            serde_json::to_vec(&body).unwrap_or_default(),
        ))
        .unwrap_or_else(|_| openai_error(500, "response build").into_response())
}

#[allow(clippy::too_many_lines)] // handler: parse, teach, gate, three delivery lanes
pub async fn audio_transcriptions(
    State(state): State<Arc<AppState>>,
    key_ext: Option<Extension<crate::keys::KeyCtx>>,
    uri: Uri,
    method: Method,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let content_type = headers
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    let Some(parts) = parse_multipart(&body, content_type) else {
        return openai_error(400, "malformed multipart body (no parseable boundary)");
    };
    let Some(file) = parts
        .iter()
        .find(|p| p.name == "file" && p.filename.is_some())
    else {
        return openai_error(400, "missing file part in multipart body");
    };
    let model = field(&parts, "model");

    // (1) Explicit remote intent wins: `name:model` never goes local.
    // F11: remote audio lanes pay key admission too (scope + rate +
    // request count) — every other remote lane has since the audit; the
    // token sniffer has nothing to read in whisper JSON (no usage
    // field), so request counts are the charge unit here.
    if let Some(model) = model.as_deref()
        && split_remote(model, &state.config).is_some()
    {
        if let Some(Extension(k)) = &key_ext
            && let Some(entry) = state.keys.entry(&k.name)
        {
            if let Err(rej) = state.keys.check(&entry, model) {
                return rej.to_response();
            }
            state.keys.charge_request(&k.name);
        }
        return crate::remotes::forward_with_health(
            &state,
            model,
            &method,
            uri.path_and_query().map_or(
                "/v1/audio/transcriptions",
                axum::http::uri::PathAndQuery::as_str,
            ),
            &headers,
            body,
        )
        .await;
    }

    // (2) Local lane: installed binary + pulled ggml model. Name the exact
    // missing half — "lane not installed" when the server binary is the
    // gap, "no model pulled" when the server is fine but transcription
    // has nothing to load (both halves observed live in validation).
    // The lane pays the same admission the remote branch just paid
    // (scope on the whisper size name + rpm/tpm/daily + request charge):
    // local compute is not a free lane on an authed gateway (audit MM1).
    if let Err(resp) =
        state.admit_or_respond(key_ext.as_ref(), model.as_deref().unwrap_or("whisper-1"))
    {
        return *resp;
    }
    let available = whisper::list_models(&state.dirs);
    let requested = model.as_deref().or(Some("whisper-1"));
    let server = whisper::server_bin(&state.dirs);
    let size = server
        .as_ref()
        .and_then(|_| whisper::resolve_model(requested, &available));
    // Request-shape errors precede lane availability: a contradictory
    // stream+async body is 400 even when no engine is installed.
    if wants_stream(&parts) && wants_async(&parts) {
        return openai_error(
            400,
            "stream and async are mutually exclusive: stream returns progressive SSE \
             events, async returns a pollable job handle — pick one",
        );
    }
    if let Some(msg) = unsupported_field_error(&parts) {
        return openai_error(400, &msg);
    }
    let gran = match granularity_plan(&parts) {
        Ok(g) => g,
        Err(msg) => return openai_error(400, &msg),
    };
    if gran.is_some() && wants_stream(&parts) {
        return openai_error(
            400,
            "timestamp_granularities applies to the buffered lanes — the SSE stream emits \
             its own segment events and has no words surface: drop stream or the \
             granularities field",
        );
    }
    let diarize = diarize_requested(&parts);
    if diarize && wants_stream(&parts) {
        return openai_error(
            400,
            "response_format=diarized_json applies to the buffered lanes — the SSE stream \
             emits its own segment events and carries no speaker labels: drop stream or \
             pick verbose_json",
        );
    }
    if let (Some((bin, lib_dir)), Some(size)) = (&server, size) {
        let stream = wants_stream(&parts);
        if stream {
            return forward_local_stream(&state, &parts, file, &size, bin, lib_dir, false).await;
        }
        if wants_async(&parts) {
            let file_part = file.clone();
            return submit_async(
                &state,
                parts,
                file_part,
                size.clone(),
                bin,
                lib_dir,
                false,
                gran,
                diarize,
            );
        }
        return forward_local(
            &state, &parts, file, &size, bin, lib_dir, false, gran, diarize,
        )
        .await;
    }

    // (3) Teaching error: no local lane and no remote intent.
    openai_error(
        501,
        &teaching_detail(
            server.is_some(),
            requested.unwrap_or("whisper-1"),
            &available,
        ),
    )
}

/// `/v1/audio/translations`: same decision order as transcriptions,
/// same lazy child — the only difference is `translate=true` forced on
/// the rebuilt `/inference` request (upstream has no separate route).
#[allow(clippy::too_many_lines)] // mirrors audio_transcriptions plus the translate flag
pub async fn audio_translations(
    State(state): State<Arc<AppState>>,
    key_ext: Option<Extension<crate::keys::KeyCtx>>,
    uri: Uri,
    method: Method,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let content_type = headers
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    let Some(parts) = parse_multipart(&body, content_type) else {
        return openai_error(400, "malformed multipart body (no parseable boundary)");
    };
    let Some(file) = parts
        .iter()
        .find(|p| p.name == "file" && p.filename.is_some())
    else {
        return openai_error(400, "missing file part in multipart body");
    };
    let model = field(&parts, "model");

    // (1) Explicit remote intent wins (F11 admission, same as
    // transcriptions — request counts are the charge unit here).
    if let Some(model) = model.as_deref()
        && split_remote(model, &state.config).is_some()
    {
        if let Some(Extension(k)) = &key_ext
            && let Some(entry) = state.keys.entry(&k.name)
        {
            if let Err(rej) = state.keys.check(&entry, model) {
                return rej.to_response();
            }
            state.keys.charge_request(&k.name);
        }
        return crate::remotes::forward_with_health(
            &state,
            model,
            &method,
            uri.path_and_query().map_or(
                "/v1/audio/translations",
                axum::http::uri::PathAndQuery::as_str,
            ),
            &headers,
            body,
        )
        .await;
    }

    // (2) Local lane, forced translation. Same admission as the remote
    // branch and every other gated lane (audit MM1).
    if let Err(resp) =
        state.admit_or_respond(key_ext.as_ref(), model.as_deref().unwrap_or("whisper-1"))
    {
        return *resp;
    }
    let available = whisper::list_models(&state.dirs);
    let requested = model.as_deref().or(Some("whisper-1"));
    let server = whisper::server_bin(&state.dirs);
    let size = server
        .as_ref()
        .and_then(|_| whisper::resolve_model(requested, &available));
    // Request-shape errors precede lane availability: a contradictory
    // stream+async body is 400 even when no engine is installed.
    if wants_stream(&parts) && wants_async(&parts) {
        return openai_error(
            400,
            "stream and async are mutually exclusive: stream returns progressive SSE \
             events, async returns a pollable job handle — pick one",
        );
    }
    if let Some(msg) = unsupported_field_error(&parts) {
        return openai_error(400, &msg);
    }
    let gran = match granularity_plan(&parts) {
        Ok(g) => g,
        Err(msg) => return openai_error(400, &msg),
    };
    if gran.is_some() && wants_stream(&parts) {
        return openai_error(
            400,
            "timestamp_granularities applies to the buffered lanes — the SSE stream emits \
             its own segment events and has no words surface: drop stream or the \
             granularities field",
        );
    }
    let diarize = diarize_requested(&parts);
    if diarize && wants_stream(&parts) {
        return openai_error(
            400,
            "response_format=diarized_json applies to the buffered lanes — the SSE stream \
             emits its own segment events and carries no speaker labels: drop stream or \
             pick verbose_json",
        );
    }
    if let (Some((bin, lib_dir)), Some(size)) = (&server, size) {
        let stream = wants_stream(&parts);
        if stream {
            return forward_local_stream(&state, &parts, file, &size, bin, lib_dir, true).await;
        }
        if wants_async(&parts) {
            let file_part = file.clone();
            return submit_async(
                &state,
                parts,
                file_part,
                size.clone(),
                bin,
                lib_dir,
                true,
                gran,
                diarize,
            );
        }
        return forward_local(
            &state, &parts, file, &size, bin, lib_dir, true, gran, diarize,
        )
        .await;
    }

    // (3) Teaching error: no local lane and no remote intent.
    openai_error(
        501,
        &teaching_detail(
            server.is_some(),
            requested.unwrap_or("whisper-1"),
            &available,
        ),
    )
}

/// 501 body for a local-lane miss, naming the EXACT missing half (pure,
/// unit-tested): the server binary vs the model payload.
#[must_use]
fn teaching_detail(has_server: bool, requested: &str, available: &[String]) -> String {
    let avail = if available.is_empty() {
        "<none pulled>".to_string()
    } else {
        available.join(", ")
    };
    if has_server {
        format!(
            "whisper server installed but no usable model for {requested:?} \
             — pull one: `blazar whisper --pull base`. \
             Available local models: {avail}."
        )
    } else {
        format!(
            "no transcription backend: whisper server not installed \
             (`blazar engine install --kind whisper`; legacy: \
             `blazar whisper --install`) and the request does not name a \
             remote (`whisper:<model>` with a [[remotes]] entry). \
             Available local models: {avail}."
        )
    }
}

#[cfg(test)]
#[allow(non_snake_case)] // suite convention: unit__scenario__expected (§6b)
mod tests {
    use super::*;

    fn body(boundary: &str, parts: &[(&str, &str, &[u8])]) -> (Vec<u8>, String) {
        // (name, filename, data) — data is embedded verbatim between the
        // delimiters, exactly like a real audio part would be.
        let mut b = Vec::new();
        for (name, filename, data) in parts {
            b.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
            b.extend_from_slice(
                format!(
                    "Content-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\n\
                     Content-Type: application/octet-stream\r\n\r\n"
                )
                .as_bytes(),
            );
            b.extend_from_slice(data);
            b.extend_from_slice(b"\r\n");
        }
        b.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
        let ct = format!("multipart/form-data; boundary={boundary}");
        (b, ct)
    }

    #[test]
    fn unit__parse_multipart__two_parts_with_binary_data() {
        let audio: Vec<u8> = (0..=255u8).cycle().take(1024).collect();
        let (b, ct) = body(
            "XbOuNdArY",
            &[
                ("model", "req.bin", b"whisper-1"),
                ("file", "audio.wav", &audio),
            ],
        );
        let parts = parse_multipart(&b, &ct).expect("parses");
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].name, "model");
        assert_eq!(parts[0].data, b"whisper-1");
        assert_eq!(parts[1].name, "file");
        assert_eq!(parts[1].filename.as_deref(), Some("audio.wav"));
        assert_eq!(
            parts[1].content_type.as_deref(),
            Some("application/octet-stream")
        );
        assert_eq!(parts[1].data, audio);
    }

    #[test]
    fn unit__parse_multipart__fake_boundary_inside_audio_ignored() {
        // Audio bytes that CONTAIN the delimiter string must not split the
        // part: only delimiters at true boundaries count. We craft data
        // with an embedded "--XbOuNdArY" and verify it stays in the data.
        let mut audio = b"RIFFxxxx".to_vec();
        audio.extend_from_slice(b"--XbOuNdArY\r\nContent-Disposition: forged");
        let (b, ct) = body("XbOuNdArY", &[("file", "a.wav", &audio)]);
        let parts = parse_multipart(&b, &ct).expect("parses");
        assert_eq!(parts.len(), 1, "forged boundary must not split");
        assert_eq!(parts[0].data, audio);
    }

    #[test]
    fn unit__parse_multipart__missing_boundary_param_none() {
        let (b, _) = body("XbOuNdArY", &[("file", "a.wav", b"abc")]);
        assert!(parse_multipart(&b, "multipart/form-data").is_none());
        assert!(parse_multipart(&b, "text/plain").is_none());
    }

    #[test]
    fn unit__parse_multipart__empty_file_part_is_data() {
        let (b, ct) = body("XbOuNdArY", &[("file", "a.wav", b"")]);
        let parts = parse_multipart(&b, &ct).expect("parses");
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0].data.len(), 0);
    }

    #[test]
    fn unit__teaching_detail__server_missing_names_install() {
        let d = teaching_detail(false, "whisper-1", &[]);
        assert!(d.contains("whisper server not installed"), "got: {d}");
        assert!(d.contains("`blazar whisper --install`"), "got: {d}");
        assert!(d.contains("<none pulled>"), "got: {d}");
    }

    #[test]
    fn unit__teaching_detail__model_missing_names_pull() {
        let d = teaching_detail(true, "whisper-1", &[]);
        assert!(
            d.contains("server installed but no usable model"),
            "got: {d}"
        );
        assert!(d.contains("`blazar whisper --pull base`"), "got: {d}");
        assert!(
            !d.contains("not installed"),
            "must not claim server missing: {d}"
        );
    }

    #[test]
    fn unit__teaching_detail__with_models_lists_them_and_request() {
        let avail = vec!["base".to_string(), "small".to_string()];
        let d = teaching_detail(true, "large-v3", &avail);
        assert!(d.contains("\"large-v3\""), "got: {d}");
        assert!(d.contains("base, small"), "got: {d}");
    }

    /// Form body where `fields` are plain value parts (no filename —
    /// what `field()` matches) plus one file part.
    fn mixed_body(
        boundary: &str,
        fields: &[(&str, &[u8])],
        file: (&str, &[u8]),
    ) -> (Vec<u8>, String) {
        let mut b = Vec::new();
        for (name, data) in fields {
            b.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
            b.extend_from_slice(
                format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes(),
            );
            b.extend_from_slice(data);
            b.extend_from_slice(b"\r\n");
        }
        b.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        b.extend_from_slice(
            format!(
                "Content-Disposition: form-data; name=\"file\"; \
                 filename=\"{}\"\r\n\r\n",
                file.0
            )
            .as_bytes(),
        );
        b.extend_from_slice(file.1);
        b.extend_from_slice(b"\r\n");
        b.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
        let ct = format!("multipart/form-data; boundary={boundary}");
        (b, ct)
    }

    #[test]
    fn unit__forwarded_fields__whitelist_forwards_verified_and_drops_unknowns() {
        // Verified fields ride the rebuilt form; OpenAI-only concepts and
        // the model knob (resolved earlier, upstream has no such field)
        // are dropped; response_format defaults to json; translate
        // defaults to false on the transcriptions route.
        let (b, ct) = mixed_body(
            "XbOuNdArY",
            &[
                ("model", b"whisper-1"),
                ("language", b"en"),
                ("beam_size", b"2"),
                ("timestamp_granularities[]", b"word"),
            ],
            ("audio.wav", b"RIFF"),
        );
        let parts = parse_multipart(&b, &ct).expect("parses");
        let fields = forwarded_fields(&parts, false);
        let get = |k: &str| fields.iter().find(|(n, _)| n == k).map(|(_, v)| v.as_str());
        assert_eq!(get("language"), Some("en"));
        assert_eq!(get("beam_size"), Some("2"));
        assert_eq!(get("response_format"), Some("json"), "default injected");
        assert_eq!(get("translate"), Some("false"), "default injected");
        assert!(
            fields
                .iter()
                .all(|(n, _)| n != "timestamp_granularities[]" && n != "model"),
            "unknowns dropped: {fields:?}"
        );
    }

    #[test]
    fn unit__forwarded_fields__force_translate_overrides_client_value() {
        // The translations route OWNS translate — even an explicit
        // client "false" must not disable it, and a client "true" must
        // not duplicate the part.
        let (b, ct) = mixed_body("XbOuNdArY", &[("translate", b"false")], ("a.wav", b"RIFF"));
        let parts = parse_multipart(&b, &ct).expect("parses");
        let fields = forwarded_fields(&parts, true);
        let translates: Vec<&str> = fields
            .iter()
            .filter(|(n, _)| n == "translate")
            .map(|(_, v)| v.as_str())
            .collect();
        assert_eq!(translates, vec!["true"], "forced + no duplicate");
    }

    #[test]
    fn unit__forwarded_fields__explicit_response_format_wins_over_default() {
        let (b, ct) = mixed_body(
            "XbOuNdArY",
            &[("response_format", b"verbose_json")],
            ("a.wav", b"RIFF"),
        );
        let parts = parse_multipart(&b, &ct).expect("parses");
        let fields = forwarded_fields(&parts, false);
        let n = fields
            .iter()
            .filter(|(name, _)| name == "response_format")
            .count();
        assert_eq!(n, 1, "exactly one response_format: {fields:?}");
        assert!(
            fields.contains(&("response_format".to_string(), "verbose_json".to_string())),
            "client value kept: {fields:?}"
        );
    }

    #[test]
    fn unit__wants_async__accepts_true_and_1_rejects_rest() {
        let mk = |v: &str| mixed_body("XbOuNdArY", &[("async", v.as_bytes())], ("a.wav", b"RIFF"));
        for yes in ["true", "TRUE", "1", " true "] {
            let (b, ct) = mk(yes);
            let parts = parse_multipart(&b, &ct).expect("parses");
            assert!(wants_async(&parts), "{yes:?} must request async");
        }
        for no in ["false", "0", "yes", ""] {
            let (b, ct) = mk(no);
            let parts = parse_multipart(&b, &ct).expect("parses");
            assert!(!wants_async(&parts), "{no:?} must stay sync");
        }
        // Absent field = sync, the default every existing client gets.
        let (b, ct) = mixed_body("XbOuNdArY", &[], ("a.wav", b"RIFF"));
        let parts = parse_multipart(&b, &ct).expect("parses");
        assert!(!wants_async(&parts));
    }

    /// Registry lifecycle: reserve → running → completed carries the
    /// upstream triple into the poll payload; fail and cancel are
    /// terminal and idempotent; unknown ids 404 (payload None).
    #[test]
    fn unit__audio_jobs__lifecycle_and_payload_shapes() {
        let jobs = AudioJobs::new();
        let id = jobs.reserve();
        assert_eq!(
            jobs.payload(&id).unwrap()["status"],
            "queued",
            "fresh reservation is queued"
        );
        jobs.mark_running(&id);
        assert_eq!(jobs.payload(&id).unwrap()["status"], "running");
        jobs.finish(&id, 200, "application/json", br#"{"text":"hi"}"#.to_vec());
        let done = jobs.payload(&id).unwrap();
        assert_eq!(done["status"], "completed");
        assert_eq!(done["status_code"], 200);
        assert_eq!(done["result"]["text"], "hi", "body parsed as JSON");
        // Terminal stays terminal: late finish/cancel never overwrite.
        jobs.fail(&id, "late failure".into());
        jobs.cancel(&id);
        assert_eq!(jobs.payload(&id).unwrap()["status"], "completed");

        // Non-JSON body replays as a raw string under "result".
        let id2 = jobs.reserve();
        jobs.mark_running(&id2);
        jobs.finish(&id2, 200, "text/plain", b"plain text".to_vec());
        assert_eq!(
            jobs.payload(&id2).unwrap()["result"],
            serde_json::json!("plain text")
        );

        // Failed jobs surface the sync lane's error message verbatim.
        let id3 = jobs.reserve();
        jobs.mark_running(&id3);
        jobs.fail(&id3, "whisper-server: boom".into());
        let failed = jobs.payload(&id3).unwrap();
        assert_eq!(failed["status"], "failed");
        assert_eq!(failed["error"], "whisper-server: boom");

        // Cancel of a live job flips it to cancelled, second cancel is
        // a no-op, and unknown ids have no payload.
        let id4 = jobs.reserve();
        jobs.mark_running(&id4);
        assert_eq!(jobs.cancel(&id4), Some(true));
        assert_eq!(jobs.cancel(&id4), Some(false), "terminal cancel idempotent");
        assert_eq!(jobs.payload(&id4).unwrap()["status"], "cancelled");
        assert!(jobs.payload("aj-nonexistent").is_none());
        assert_eq!(jobs.cancel("aj-nonexistent"), None);
    }

    /// At the cap the registry evicts the oldest TERMINAL job first and
    /// keeps live ones; ids stay unique across evictions.
    #[test]
    fn unit__audio_jobs__eviction_oldest_terminal_first() {
        let jobs = AudioJobs::new();
        // One live job reserved early, then a wave of terminal jobs past
        // the cap — the early live one must survive every eviction.
        let live = jobs.reserve();
        jobs.mark_running(&live);
        let mut first_terminal = String::new();
        for i in 0..=MAX_AUDIO_JOBS {
            let id = jobs.reserve();
            if i == 0 {
                first_terminal = id.clone();
            }
            jobs.finish(&id, 200, "application/json", Vec::new());
        }
        assert!(
            jobs.payload(&live).is_some(),
            "live job must not be evicted while terminal victims exist"
        );
        assert!(
            jobs.payload(&first_terminal).is_none(),
            "oldest terminal job is the eviction victim"
        );
    }

    /// Audit MM9: `active_count` is the /metrics footprint of the local
    /// audio lane — queued+running only; every terminal state (or
    /// eviction) stops counting.
    #[test]
    fn unit__audio_jobs__active_count_tracks_live_only() {
        let jobs = AudioJobs::new();
        assert_eq!(jobs.active_count(), 0, "empty registry");
        let a = jobs.reserve();
        assert_eq!(jobs.active_count(), 1, "queued counts as active");
        jobs.mark_running(&a);
        assert_eq!(jobs.active_count(), 1, "running still one active");
        let b = jobs.reserve();
        jobs.mark_running(&b);
        assert_eq!(jobs.active_count(), 2, "second live job counted");
        jobs.finish(&a, 200, "application/json", Vec::new());
        assert_eq!(jobs.active_count(), 1, "completed drops out");
        jobs.fail(&b, "boom".into());
        assert_eq!(jobs.active_count(), 0, "failed drops out");
        // Cancelled path too.
        let c = jobs.reserve();
        jobs.cancel(&c);
        assert_eq!(jobs.active_count(), 0, "cancelled drops out");
    }

    #[test]
    fn unit__wants_stream__accepts_true_and_1_rejects_rest() {
        let mk = |v: &str| mixed_body("XbOuNdArY", &[("stream", v.as_bytes())], ("a.wav", b"RIFF"));
        for yes in ["true", "TRUE", "1", " true "] {
            let (b, ct) = mk(yes);
            let parts = parse_multipart(&b, &ct).expect("parses");
            assert!(wants_stream(&parts), "{yes:?} must request streaming");
        }
        for no in ["false", "0", "yes", ""] {
            let (b, ct) = mk(no);
            let parts = parse_multipart(&b, &ct).expect("parses");
            assert!(!wants_stream(&parts), "{no:?} must stay non-streaming");
        }
        // Absent field = the batch default every existing client gets.
        let (b, ct) = mixed_body("XbOuNdArY", &[], ("a.wav", b"RIFF"));
        let parts = parse_multipart(&b, &ct).expect("parses");
        assert!(!wants_stream(&parts), "absent field must not stream");
    }

    #[test]
    fn unit__forwarded_fields_stream__forces_verbose_json_and_drops_decode_shifting_fields() {
        // The progressive lane decodes each window at its own offset, so
        // client response_format and window-shaping fields (duration,
        // offset_t, no_timestamps) must be dropped and verbose_json forced;
        // language and other verified knobs still ride.
        let (b, ct) = mixed_body(
            "XbOuNdArY",
            &[
                ("response_format", b"srt"),
                ("no_timestamps", b"true"),
                ("duration", b"5"),
                ("offset_t", b"1"),
                ("language", b"en"),
                ("beam_size", b"2"),
            ],
            ("a.wav", b"RIFF"),
        );
        let parts = parse_multipart(&b, &ct).expect("parses");
        let fields = forwarded_fields_stream(&parts, false);
        let get = |k: &str| fields.iter().find(|(n, _)| n == k).map(|(_, v)| v.as_str());
        assert_eq!(get("response_format"), Some("verbose_json"), "forced");
        assert_eq!(get("language"), Some("en"), "verified knob kept");
        assert_eq!(get("beam_size"), Some("2"), "verified knob kept");
        assert_eq!(get("translate"), Some("false"), "default injected");
        assert!(
            fields
                .iter()
                .all(|(n, _)| n != "no_timestamps" && n != "duration" && n != "offset_t"),
            "window-shaping fields dropped: {fields:?}"
        );
        let n = fields
            .iter()
            .filter(|(name, _)| name == "response_format")
            .count();
        assert_eq!(n, 1, "no duplicate response_format: {fields:?}");
    }

    #[test]
    fn unit__sse_frame__wire_format() {
        let mut v = serde_json::json!({"a": 1});
        let frame = sse_frame("chunk.completed", &v);
        assert_eq!(
            String::from_utf8_lossy(&frame),
            "event: chunk.completed\ndata: {\"a\":1}\n\n"
        );
        // Event names with dots and empty payloads both stay well-formed.
        v = serde_json::json!({});
        let frame = sse_frame("transcript.completed", &v);
        assert_eq!(
            String::from_utf8_lossy(&frame),
            "event: transcript.completed\ndata: {}\n\n"
        );
    }

    #[test]
    fn unit__rebase_segments__adds_offset_to_start_and_end() {
        // Production hands the extracted `segments` array, not the doc.
        let mut segments = serde_json::json!([
            {"id": 0, "start": 0.0, "end": 1.5, "text": "a"},
            {"id": 1, "start": 1.5, "end": 3.0, "text": "b"},
            {"id": 2, "start": "not-a-number", "end": null, "text": "c"}
        ]);
        rebase_segments(&mut segments, 6000);
        let segs = segments.as_array().expect("segments");
        assert_eq!(segs[0]["start"].as_f64(), Some(6.0));
        assert_eq!(segs[0]["end"].as_f64(), Some(7.5));
        assert_eq!(segs[1]["start"].as_f64(), Some(7.5));
        assert_eq!(segs[1]["end"].as_f64(), Some(9.0));
        // Non-numeric times pass through untouched rather than panicking.
        assert_eq!(segs[2]["start"].as_str(), Some("not-a-number"));
        assert!(segs[2]["end"].is_null());
    }

    #[test]
    fn unit__forwarded_fields__vad_and_token_fields_ride() {
        // The b5130-verified engine knobs must reach the rebuilt form
        // verbatim — VAD tuning and token timing are request-side
        // whisper.cpp fields, not gateway concepts.
        let (b, ct) = mixed_body(
            "XbOuNdArY",
            &[
                ("token_timestamps", b"true"),
                ("vad_threshold", b"0.4"),
                ("vad_min_silence_duration_ms", b"200"),
                ("suppress_non_speech", b"true"),
            ],
            ("a.wav", b"RIFF"),
        );
        let parts = parse_multipart(&b, &ct).expect("parses");
        let fields = forwarded_fields(&parts, false);
        let get = |k: &str| fields.iter().find(|(n, _)| n == k).map(|(_, v)| v.as_str());
        assert_eq!(get("token_timestamps"), Some("true"));
        assert_eq!(get("vad_threshold"), Some("0.4"));
        assert_eq!(get("vad_min_silence_duration_ms"), Some("200"));
        assert_eq!(get("suppress_non_speech"), Some("true"));
    }

    #[test]
    fn unit__granularity_plan__array_parts_commas_and_case() {
        // OpenAI SDK style (repeated [] parts), curl style (one
        // comma-separated part), and case-insensitive values all parse
        // to the same plan; absent field is None.
        let (b, ct) = mixed_body(
            "XbOuNdArY",
            &[
                ("timestamp_granularities[]", b"word"),
                ("timestamp_granularities[]", b"SEGMENT"),
            ],
            ("a.wav", b"RIFF"),
        );
        let parts = parse_multipart(&b, &ct).expect("parses");
        let gran = granularity_plan(&parts).expect("ok").expect("some");
        assert!(gran.word && gran.segment);

        let (b, ct) = mixed_body(
            "XbOuNdArY",
            &[("timestamp_granularities", b"word, segment")],
            ("a.wav", b"RIFF"),
        );
        let parts = parse_multipart(&b, &ct).expect("parses");
        let gran = granularity_plan(&parts).expect("ok").expect("some");
        assert!(gran.word && gran.segment);

        let (b, ct) = mixed_body("XbOuNdArY", &[], ("a.wav", b"RIFF"));
        let parts = parse_multipart(&b, &ct).expect("parses");
        assert!(granularity_plan(&parts).expect("ok").is_none());
    }

    #[test]
    fn unit__granularity_plan__unknown_value_and_format_conflict() {
        let (b, ct) = mixed_body(
            "XbOuNdArY",
            &[("timestamp_granularities[]", b"syllable")],
            ("a.wav", b"RIFF"),
        );
        let parts = parse_multipart(&b, &ct).expect("parses");
        let err = granularity_plan(&parts).expect_err("unknown value rejected");
        assert!(err.contains("'word' or 'segment'"), "got: {err}");
        assert!(err.contains("syllable"), "got: {err}");

        // Granularities only exist on verbose_json — an explicit other
        // format is a contradiction, taught, not silently resolved.
        let (b, ct) = mixed_body(
            "XbOuNdArY",
            &[
                ("timestamp_granularities[]", b"word"),
                ("response_format", b"srt"),
            ],
            ("a.wav", b"RIFF"),
        );
        let parts = parse_multipart(&b, &ct).expect("parses");
        let err = granularity_plan(&parts).expect_err("conflict rejected");
        assert!(err.contains("verbose_json"), "got: {err}");

        // Explicit verbose_json + word: fine.
        let (b, ct) = mixed_body(
            "XbOuNdArY",
            &[
                ("timestamp_granularities[]", b"word"),
                ("response_format", b"verbose_json"),
            ],
            ("a.wav", b"RIFF"),
        );
        let parts = parse_multipart(&b, &ct).expect("parses");
        let gran = granularity_plan(&parts).expect("ok").expect("some");
        assert!(gran.word && !gran.segment);
    }

    #[test]
    fn unit__apply_granularity__forces_verbose_json_and_token_timing() {
        fn get<'a>(k: &'a str, fields: &'a [(String, String)]) -> Option<&'a str> {
            fields.iter().find(|(n, _)| n == k).map(|(_, v)| v.as_str())
        }
        // From the forwarded_fields defaults (json injected): word
        // granularity flips the format to verbose_json and turns on
        // token timing; segment-only leaves token_timestamps alone; a
        // client token_timestamps value is overridden (word timing is
        // the point of the request).
        let (b, ct) = mixed_body("XbOuNdArY", &[], ("a.wav", b"RIFF"));
        let parts = parse_multipart(&b, &ct).expect("parses");
        let mut fields = forwarded_fields(&parts, false);
        apply_granularity(
            &mut fields,
            Granularity {
                word: true,
                segment: false,
            },
        );
        assert_eq!(get("response_format", &fields), Some("verbose_json"));
        assert_eq!(get("token_timestamps", &fields), Some("true"));
        // Exactly one of each key — duplicates would shift meaning
        // upstream (first occurrence wins).
        for key in ["response_format", "token_timestamps"] {
            let n = fields.iter().filter(|(n, _)| n == key).count();
            assert_eq!(n, 1, "{key} duplicated: {fields:?}");
        }

        let mut fields = forwarded_fields(&parts, false);
        apply_granularity(
            &mut fields,
            Granularity {
                word: false,
                segment: true,
            },
        );
        assert_eq!(get("response_format", &fields), Some("verbose_json"));
        assert_eq!(get("token_timestamps", &fields), None);
    }

    #[test]
    fn unit__unsupported_field_error__include_keywords_only() {
        let mk = |name: &str, value: &[u8]| {
            let (b, ct) = mixed_body("XbOuNdArY", &[(name, value)], ("a.wav", b"RIFF"));
            parse_multipart(&b, &ct).expect("parses")
        };
        let err = unsupported_field_error(&mk("include", b"logprobs")).expect("teaches");
        assert!(
            err.contains("include") && err.contains("logprobs"),
            "got: {err}"
        );

        let err = unsupported_field_error(&mk("keywords", b"whisper")).expect("teaches");
        assert!(
            err.contains("keywords") && err.contains("`prompt`"),
            "got: {err}"
        );

        // The supported vocabulary stays silent — diarized_json included
        // since W9 (gateway-built on the engine's live-probed diarize
        // field; mono input collapses to one speaker).
        for fmt in [
            "json",
            "text",
            "srt",
            "verbose_json",
            "vtt",
            "diarized_json",
        ] {
            let parts = mk("response_format", fmt.as_bytes());
            assert!(unsupported_field_error(&parts).is_none(), "{fmt} must pass");
        }
        let (b, ct) = mixed_body("XbOuNdArY", &[], ("a.wav", b"RIFF"));
        let parts = parse_multipart(&b, &ct).expect("parses");
        assert!(unsupported_field_error(&parts).is_none());
    }

    #[test]
    fn unit__to_diarized_json__strips_prefixes_labels_and_joins_text() {
        // Live-probed engine shape (b5130, stereo fixture): diarize
        // labels each segment with the channel's speaker index and
        // prefixes its text with "(speaker N)".
        let body = serde_json::json!({
            "task": "transcribe",
            "language": "en",
            "duration": 7.9,
            "text": "(speaker 0) Hello there. (speaker 1) Hi back.",
            "segments": [
                {"id": 0, "start": 0.0, "end": 3.4, "text": " (speaker 0) Hello there.",
                 "speaker": 0, "no_speech_prob": 0.1},
                {"id": 1, "start": 5.0, "end": 7.9, "text": "(speaker 1) Hi back.",
                 "speaker": 1, "no_speech_prob": 0.2},
            ],
        });
        let mapped = to_diarized_json(&body).expect("maps");
        assert_eq!(mapped["duration"], 7.9);
        assert_eq!(mapped["text"], "Hello there. Hi back.");
        let segs = mapped["segments"].as_array().expect("segments");
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0]["text"], "Hello there.");
        assert_eq!(segs[0]["speaker"], 0);
        assert_eq!(segs[1]["text"], "Hi back.");
        assert_eq!(segs[1]["speaker"], 1);
        assert!(
            segs[0].get("no_speech_prob").is_none(),
            "engine-only keys are dropped from the mapped shape"
        );

        // Unlabeled segment (mono or engine fallback): the field is
        // omitted, never guessed.
        let mono = serde_json::json!({
            "duration": 1.0,
            "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": "just words"}],
        });
        let mapped = to_diarized_json(&mono).expect("maps");
        assert!(mapped["segments"][0].get("speaker").is_none());
        assert_eq!(mapped["text"], "just words");

        // Bare "(speaker 0)" prefix with no text after it.
        assert_eq!(strip_speaker_prefix("(speaker 0)"), "");
        // No prefix at all.
        assert_eq!(strip_speaker_prefix("  plain"), "plain");

        // No segments array → None (caller fails loud, 502).
        assert!(to_diarized_json(&serde_json::json!({"text": "x"})).is_none());
    }

    #[test]
    fn unit__forwarded_fields__diarize_and_no_language_probabilities_ride() {
        let (b, ct) = mixed_body(
            "XbOuNdArY",
            &[
                ("diarize", b"true"),
                ("no_language_probabilities", b"true"),
                ("tinydiarize", b"true"), // unprobed on the child: stays out
                ("dtw", b"tiny"),         // no observable effect: stays out
            ],
            ("a.wav", b"RIFF"),
        );
        let parts = parse_multipart(&b, &ct).expect("parses");
        let fields = forwarded_fields(&parts, false);
        let get = |k: &str| fields.iter().find(|(n, _)| n == k).map(|(_, v)| v.as_str());
        assert_eq!(get("diarize"), Some("true"));
        assert_eq!(get("no_language_probabilities"), Some("true"));
        assert!(fields.iter().all(|(n, _)| n != "tinydiarize" && n != "dtw"));
    }

    #[test]
    fn unit__enrich_verbose_json_with_words__groups_tokens_by_leading_marker() {
        // whisper.cpp token timings arrive in MILLISECONDS while segment
        // bounds are seconds — the normalization against the segment
        // end converts (0,320)/(320,640) into 0.0/0.32/0.64. The
        // SentencePiece marker (▁) starts a new word; unmarked tokens
        // append to the current word ("▁cat" + "s" → "cats").
        let mut body = serde_json::json!({
            "text": "The cats",
            "segments": [{
                "id": 0, "start": 0.0, "end": 0.64, "text": " The cats",
                "tokens": [
                    {"text": "The", "timestamps": [0, 320]},
                    {"text": "▁cat", "timestamps": [320, 560]},
                    {"text": "s", "timestamps": [560, 640]}
                ]
            }]
        });
        assert!(enrich_verbose_json_with_words(&mut body));
        let words = body["segments"][0]["words"].as_array().expect("words");
        assert_eq!(words.len(), 2, "got: {words:?}");
        assert_eq!(words[0]["word"], "The");
        assert_eq!(words[0]["start"].as_f64(), Some(0.0));
        assert_eq!(words[0]["end"].as_f64(), Some(0.32));
        assert_eq!(words[1]["word"], "cats");
        assert_eq!(words[1]["start"].as_f64(), Some(0.32));
        assert_eq!(words[1]["end"].as_f64(), Some(0.64));
    }

    #[test]
    fn unit__enrich_verbose_json_with_words__seconds_units_and_missing_cases() {
        // Token timings already in seconds pass through untouched (no
        // spurious /1000): end 1.1 is within the segment bound.
        let mut body = serde_json::json!({
            "segments": [{
                "id": 0, "start": 0.0, "end": 1.2,
                "tokens": [
                    {"text": "▁hello", "timestamps": [0.0, 0.5]},
                    {"text": "▁world", "timestamps": [0.5, 1.1]}
                ]
            }]
        });
        assert!(enrich_verbose_json_with_words(&mut body));
        let words = body["segments"][0]["words"].as_array().expect("words");
        assert_eq!(words[0]["word"], "hello");
        assert_eq!(words[0]["end"].as_f64(), Some(0.5));
        assert_eq!(words[1]["word"], "world");
        assert_eq!(words[1]["start"].as_f64(), Some(0.5));

        // Engine-served words pass through untouched.
        let mut body = serde_json::json!({
            "segments": [{
                "id": 0, "start": 0.0, "end": 1.0,
                "words": [{"word": "kept", "start": 0.0, "end": 1.0}],
                "tokens": [{"text": "▁other", "timestamps": [0, 1000]}]
            }]
        });
        assert!(enrich_verbose_json_with_words(&mut body));
        let words = body["segments"][0]["words"].as_array().expect("words");
        assert_eq!(words.len(), 1);
        assert_eq!(words[0]["word"], "kept");

        // No timing anywhere: nothing derived (false), body unchanged —
        // never interpolated.
        let mut body = serde_json::json!({
            "segments": [{
                "id": 0, "start": 0.0, "end": 1.0,
                "tokens": [{"text": "▁untimed"}]
            }]
        });
        assert!(!enrich_verbose_json_with_words(&mut body));
        assert!(
            body["segments"][0].get("words").is_none(),
            "no fabricated words"
        );

        // No segments array at all: false, no panic.
        let mut body = serde_json::json!({"text": "bare"});
        assert!(!enrich_verbose_json_with_words(&mut body));
    }
}
