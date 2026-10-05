//! POST `/v1/audio/speech` + GET `/v1/audio/voices` — `OpenAI` TTS
//! shape on the local piper lane.
//!
//! `model` IS the piper voice id (honest mapping: `en_US-amy-medium`),
//! `response_format` speaks WAV (buffered, the default — a documented
//! deviation: `OpenAI` defaults to mp3), `pcm`/`pcm16` (raw s16le PCM
//! streamed sentence-by-sentence so time-to-first-audio scales with the
//! first sentence instead of the whole document), and the lossy family
//! mp3/opus/aac/flac (WAV transcoded through the system ffmpeg).
//! `speed` maps to piper's inverse length scale; `speaker`, `noise_scale`,
//! `noise_w`, `sentence_silence` ride as documented native extensions.
//! Remote intent (`name:model`) forwards like every other lane.

use axum::body::Bytes;
use axum::extract::{Extension, Query, State};
use axum::http::{HeaderMap, Method, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

use crate::proxy::openai_error;
use crate::remotes::split_remote;
use crate::state::AppState;

/// Bound mirrors the runtime cap; the route validates first so clients
/// get a 400 before any spawn.
const MAX_INPUT_CHARS: usize = 10_000;

/// Lossy egress containers, transcoded from piper's WAV through the
/// system ffmpeg. Muxer, codec, bitrate flags, and content type are
/// gateway-fixed per format: the client picks a container by name, never
/// raw ffmpeg argv.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LossyFormat {
    Mp3,
    Opus,
    Aac,
    Flac,
}

impl LossyFormat {
    /// (ffmpeg muxer, codec, extra output flags, HTTP content type).
    /// Bitrates sit at transparent-for-speech points for piper's
    /// 22.05 kHz mono; flac is lossless so it takes no rate flags.
    #[must_use]
    fn parts(
        self,
    ) -> (
        &'static str,
        &'static str,
        &'static [&'static str],
        &'static str,
    ) {
        match self {
            Self::Mp3 => ("mp3", "libmp3lame", &["-b:a", "128k"], "audio/mpeg"),
            Self::Opus => ("ogg", "libopus", &["-b:a", "64k"], "audio/ogg"),
            Self::Aac => ("adts", "aac", &["-b:a", "128k"], "audio/aac"),
            Self::Flac => ("flac", "flac", &[], "audio/flac"),
        }
    }
}

/// Egress container for one speech request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpeechFormat {
    Wav,
    Pcm,
    Lossy(LossyFormat),
}

/// Parse `response_format` per the `OpenAI` TTS surface plus the pcm16
/// alias. Errors carry the full supported list so a wrong pick teaches.
fn parse_speech_format(raw: Option<&str>) -> Result<SpeechFormat, String> {
    match raw {
        None | Some("wav") => Ok(SpeechFormat::Wav),
        Some("pcm" | "pcm16") => Ok(SpeechFormat::Pcm),
        Some("mp3") => Ok(SpeechFormat::Lossy(LossyFormat::Mp3)),
        Some("opus") => Ok(SpeechFormat::Lossy(LossyFormat::Opus)),
        Some("aac") => Ok(SpeechFormat::Lossy(LossyFormat::Aac)),
        Some("flac") => Ok(SpeechFormat::Lossy(LossyFormat::Flac)),
        Some(other) => Err(format!(
            "response_format {other:?} is not supported on the local lane — supported: \
             \"wav\" (default), \"pcm\"/\"pcm16\" (raw streamed s16le), and \"mp3\", \
             \"opus\", \"aac\", \"flac\" (transcoded via the system ffmpeg)"
        )),
    }
}

/// Parse the tuning knobs off the speech body: `OpenAI` `speed` plus the
/// native piper extensions (`speaker`, `noise_scale`, `noise_w`,
/// `sentence_silence`). Ranges mirror the runtime's checks so bad values
/// die as a 400 before any spawn; `instructions` teaches because piper
/// voices are fixed personalities, not steerable ones.
fn parse_native_options(req: &Value) -> Result<blazar_runtime::piper::SpeakOptions, String> {
    let mut opts = blazar_runtime::piper::SpeakOptions::default();
    if let Some(raw) = req.get("instructions")
        && !raw.is_null()
    {
        return Err(
            "instructions is not supported — piper voices are fixed personalities with no \
             style steering; pick a different voice (model) instead"
                .to_string(),
        );
    }
    match req.get("speed").and_then(Value::as_f64) {
        None => {}
        Some(s) if (0.25..=4.0).contains(&s) => opts.speed = Some(s),
        Some(s) => return Err(format!("speed {s} out of range (0.25..=4.0)")),
    }
    if let Some(raw) = req.get("speaker")
        && !raw.is_null()
    {
        match raw.as_u64() {
            Some(s) => opts.speaker = Some(s),
            None => return Err("speaker must be a non-negative integer".to_string()),
        }
    }
    for (field, slot) in [
        ("noise_scale", &mut opts.noise_scale),
        ("noise_w", &mut opts.noise_w),
    ] {
        match req.get(field).and_then(Value::as_f64) {
            None => {}
            Some(v) if (0.0..=2.0).contains(&v) => *slot = Some(v),
            Some(v) => return Err(format!("{field} {v} out of range (0.0..=2.0)")),
        }
    }
    match req.get("sentence_silence").and_then(Value::as_f64) {
        None => {}
        Some(v) if (0.0..=10.0).contains(&v) => opts.sentence_silence = Some(v),
        Some(v) => {
            return Err(format!(
                "sentence_silence {v} out of range (0.0..=10.0 seconds)"
            ));
        }
    }
    Ok(opts)
}

/// ffmpeg argv after the program name: WAV on stdin → the requested
/// container on stdout. Enum-derived only — no client string reaches
/// the process.
#[must_use]
fn ffmpeg_argv(format: LossyFormat) -> Vec<String> {
    let (muxer, codec, extra, _) = format.parts();
    let mut argv: Vec<String> = [
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "wav",
        "-i",
        "pipe:0",
    ]
    .iter()
    .map(|s| (*s).to_string())
    .collect();
    argv.extend(extra.iter().map(|s| (*s).to_string()));
    argv.extend(
        ["-c:a", codec, "-f", muxer, "pipe:1"]
            .iter()
            .map(|s| (*s).to_string()),
    );
    argv
}

/// Hard ceiling for one transcode. Synthesis itself is already bounded
/// by the piper timeout; this only guards a wedged ffmpeg.
const FFMPEG_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);

/// Transcode WAV bytes to a lossy container through the system ffmpeg.
/// A missing binary maps to the teaching 400 (deliberately NOT cached:
/// installing ffmpeg later just works on the next request), a non-zero
/// exit is 502 with the stderr tail, a hang is 504 with the child
/// killed by `kill_on_drop` under the timeout.
#[allow(clippy::result_large_err)] // axum Response is the lane's error currency
async fn transcode(wav: Vec<u8>, format: LossyFormat) -> Result<Vec<u8>, Response> {
    use tokio::io::AsyncWriteExt;
    let mut cmd = tokio::process::Command::new("ffmpeg");
    cmd.args(ffmpeg_argv(format))
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true);
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(openai_error(
                400,
                "ffmpeg is not on PATH — mp3/opus/aac/flac transcode through the system \
                 ffmpeg; \"wav\" (default) and \"pcm\" need none. Install ffmpeg or switch \
                 response_format",
            ));
        }
        Err(e) => return Err(openai_error(502, &format!("ffmpeg spawn failed: {e}"))),
    };
    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(&wav).await;
        let _ = stdin.shutdown().await;
    }
    match tokio::time::timeout(FFMPEG_TIMEOUT, child.wait_with_output()).await {
        Ok(Ok(out)) if out.status.success() => Ok(out.stdout),
        Ok(Ok(out)) => {
            let tail = String::from_utf8_lossy(&out.stderr);
            let tail = tail.lines().last().unwrap_or("").trim();
            Err(openai_error(
                502,
                &format!(
                    "ffmpeg exited {}{}",
                    out.status,
                    if tail.is_empty() {
                        String::new()
                    } else {
                        format!(": {tail}")
                    }
                ),
            ))
        }
        Ok(Err(e)) => Err(openai_error(502, &format!("ffmpeg wait failed: {e}"))),
        Err(_) => Err(openai_error(
            504,
            &format!(
                "ffmpeg transcode timed out after {} s",
                FFMPEG_TIMEOUT.as_secs()
            ),
        )),
    }
}

// One handler owns the buffered-WAV vs streamed-PCM split end to end;
// extracting sub-fns would separate the format decision from the
// admission/path checks it guards.
#[allow(clippy::too_many_lines)]
pub async fn audio_speech(
    State(state): State<Arc<AppState>>,
    key_ext: Option<Extension<crate::keys::KeyCtx>>,
    uri: Uri,
    method: Method,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let req: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return openai_error(400, &format!("body is not valid JSON: {e}")),
    };
    let Some(voice) = req.get("model").and_then(Value::as_str) else {
        return openai_error(
            400,
            "missing required field: model (a piper voice id, e.g. en_US-amy-medium)",
        );
    };
    let Some(text) = req.get("input").and_then(Value::as_str) else {
        return openai_error(400, "missing required field: input");
    };
    if text.trim().is_empty() {
        return openai_error(400, "input must be non-empty text");
    }
    if text.chars().count() > MAX_INPUT_CHARS {
        return openai_error(
            400,
            &format!(
                "input is {} chars (max {MAX_INPUT_CHARS}) — split long documents",
                text.chars().count()
            ),
        );
    }
    let format = match parse_speech_format(req.get("response_format").and_then(Value::as_str)) {
        Ok(f) => f,
        Err(msg) => return openai_error(400, &msg),
    };
    let opts = match parse_native_options(&req) {
        Ok(o) => o,
        Err(msg) => return openai_error(400, &msg),
    };

    // Remote intent wins: `name:model` never goes local.
    if split_remote(voice, &state.config).is_some() {
        if let Some(Extension(k)) = &key_ext
            && let Some(entry) = state.keys.entry(&k.name)
        {
            if let Err(rej) = state.keys.check(&entry, voice) {
                return rej.to_response();
            }
            state.keys.charge_request(&k.name);
        }
        return crate::remotes::forward_with_health(
            &state,
            voice,
            &method,
            uri.path_and_query()
                .map_or("/v1/audio/speech", axum::http::uri::PathAndQuery::as_str),
            &headers,
            body,
        )
        .await;
    }

    // Local lane — same admission the remote branch just paid (audit
    // MM1): scope on the voice id (this lane's "model") + rpm/tpm/daily
    // + request charge. Local synthesis is not a free lane.
    if let Err(resp) = state.admit_or_respond(key_ext.as_ref(), voice) {
        return *resp;
    }

    // Local lane. WAV: one buffered synthesis (existing contract). PCM:
    // synthesize the first sentence eagerly — a missing voice or lane
    // still maps to its teaching status before any bytes flow — then
    // stream the rest so time-to-first-audio tracks the first sentence,
    // not the whole document.
    if matches!(format, SpeechFormat::Pcm) {
        let chunks = split_streaming_chunks(text);
        let first = blazar_runtime::piper::synthesize(
            &state.dirs,
            voice,
            &chunks[0],
            opts,
            std::time::Duration::from_secs(120),
        )
        .await;
        let first_wav = match first {
            Ok(w) => w,
            Err(e) => return tts_error(&e),
        };
        let first_wav = limit_wav_peak(first_wav);
        let layout = match locate_pcm16(&first_wav) {
            Ok(l) => l,
            Err(reason) => {
                tracing::warn!("speech PCM stream: first chunk unusable: {reason}");
                return openai_error(500, "piper produced a malformed WAV container");
            }
        };
        let first_pcm = first_wav[layout.data.clone()].to_vec();

        let (tx, rx) = tokio::sync::mpsc::channel::<Result<axum::body::Bytes, std::io::Error>>(4);
        let _ = tx.send(Ok(Bytes::from(first_pcm))).await;
        let dirs = state.dirs.clone();
        let voice = voice.to_string();
        let rest = chunks[1..].to_vec();
        tokio::spawn(async move {
            for chunk in &rest {
                match blazar_runtime::piper::synthesize(
                    &dirs,
                    &voice,
                    chunk,
                    opts,
                    std::time::Duration::from_secs(120),
                )
                .await
                {
                    Ok(raw) => {
                        // Peak limiting is per chunk: streaming cannot
                        // look ahead, so each chunk is independently held
                        // under the same -1 dBFS ceiling.
                        let wav = limit_wav_peak(raw);
                        match locate_pcm16(&wav) {
                            Ok(l) => {
                                if tx
                                    .send(Ok(Bytes::from(wav[l.data.clone()].to_vec())))
                                    .await
                                    .is_err()
                                {
                                    return; // client went away
                                }
                            }
                            Err(reason) => {
                                tracing::warn!("speech PCM stream: chunk dropped ({reason})");
                                return;
                            }
                        }
                    }
                    Err(e) => {
                        // Headers are already committed to 200; the honest
                        // failure mode for a mid-stream synthesis error is
                        // a truncated stream plus a server-side log.
                        tracing::warn!("speech PCM stream ended early: {e:#}");
                        return;
                    }
                }
            }
        });

        let header_val = format!(
            "s16le;rate={};channels={};bits={}",
            layout.sample_rate, layout.channels, layout.bits
        );
        return Response::builder()
            .status(StatusCode::OK)
            .header(axum::http::header::CONTENT_TYPE, "audio/pcm")
            .header("x-blazar-pcm-format", header_val)
            .body(axum::body::Body::from_stream(
                tokio_stream::wrappers::ReceiverStream::new(rx),
            ))
            .unwrap_or_else(|_| openai_error(500, "response build").into_response());
    }

    // Local lane: the runtime error carries the exact teaching (install
    // vs pull vs voice list) — pass it through with the right status.
    let wav = match blazar_runtime::piper::synthesize(
        &state.dirs,
        voice,
        text,
        opts,
        std::time::Duration::from_secs(120),
    )
    .await
    {
        Ok(w) => limit_wav_peak(w),
        Err(e) => return tts_error(&e),
    };
    match format {
        SpeechFormat::Wav | SpeechFormat::Pcm => Response::builder()
            .status(StatusCode::OK)
            .header(axum::http::header::CONTENT_TYPE, "audio/wav")
            .body(axum::body::Body::from(wav))
            .unwrap_or_else(|_| openai_error(500, "response build").into_response()),
        // Lossy containers are inherently buffered: transcode the whole
        // WAV once, then serve bytes. PCM remains the streaming format.
        SpeechFormat::Lossy(lossy) => {
            let (_, _, _, content_type) = lossy.parts();
            match transcode(wav, lossy).await {
                Ok(bytes) => Response::builder()
                    .status(StatusCode::OK)
                    .header(axum::http::header::CONTENT_TYPE, content_type)
                    .body(axum::body::Body::from(bytes))
                    .unwrap_or_else(|_| openai_error(500, "response build").into_response()),
                Err(resp) => resp,
            }
        }
    }
}

/// One voice row in the voices listing: piper ids
/// (`<locale>-<name>-<quality>`) split into their parts for client-side
/// filtering and display; unparsable ids still list with null parts.
fn voice_row(id: &str, installed: bool, bytes: Option<u64>) -> Value {
    let parts = blazar_runtime::piper::parse_voice_id(id);
    let (language, name, quality) = match parts {
        Some((l, n, q)) => (Value::from(l), Value::from(n), Value::from(q)),
        None => (Value::Null, Value::Null, Value::Null),
    };
    let mut row = serde_json::json!({
        "object": "voice",
        "id": id,
        "language": language,
        "name": name,
        "quality": quality,
        "installed": installed,
    });
    if let Some(b) = bytes {
        row["bytes"] = Value::from(b);
    }
    row
}

/// Remote catalog rows kept per search response — enough to choose a
/// voice without dumping the whole multi-thousand-voice tree.
const VOICES_SEARCH_CAP: usize = 50;
/// Remote search query cap; the catalog match is a plain substring, so
/// long queries are pure payload.
const VOICES_QUERY_MAX_CHARS: usize = 100;
/// Wall clock for one catalog walk (paginated tree API over every
/// language directory).
const VOICES_SEARCH_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(20);

/// GET `/v1/audio/voices` — installed piper voices, plus an optional
/// `?search=` probe of the upstream `rhasspy/piper-voices` catalog
/// (results capped, marked with installed state). Boots nothing. A
/// failed search keeps the installed list and surfaces the error as
/// `search_error` instead of failing the endpoint.
#[allow(clippy::implicit_hasher)] // axum Query extractor fixes the map type
pub async fn voices(
    State(state): State<Arc<AppState>>,
    Query(params): Query<HashMap<String, String>>,
) -> Response {
    let installed = blazar_runtime::piper::list_voices(&state.dirs);
    let mut body = serde_json::json!({
        "object": "list",
        "data": installed
            .iter()
            .map(|id| voice_row(id, true, None))
            .collect::<Vec<_>>(),
    });
    if let Some(query) = params
        .get("search")
        .map(|s| s.trim())
        .filter(|q| !q.is_empty())
    {
        if query.chars().count() > VOICES_QUERY_MAX_CHARS {
            return openai_error(
                400,
                &format!("search query is capped at {VOICES_QUERY_MAX_CHARS} characters"),
            );
        }
        let search = async {
            let hf = blazar_runtime::hf::HfClient::new(std::env::var("HF_TOKEN").ok())?;
            blazar_runtime::piper::search_voices(&hf, query).await
        };
        match tokio::time::timeout(VOICES_SEARCH_TIMEOUT, search).await {
            Ok(Ok(remote)) => {
                let count = remote.len();
                let rows: Vec<Value> = remote
                    .into_iter()
                    .take(VOICES_SEARCH_CAP)
                    .map(|v| {
                        let installed_here = installed.contains(&v.id);
                        voice_row(&v.id, installed_here, Some(v.bytes))
                    })
                    .collect();
                body["search"] = serde_json::json!({
                    "query": query,
                    "count": count,
                    "truncated": count > rows.len(),
                    "results": rows,
                });
            }
            Ok(Err(e)) => {
                body["search_error"] = Value::String(format!("{e:#}"));
            }
            Err(_) => {
                body["search_error"] = Value::String(format!(
                    "voice search timed out after {} s",
                    VOICES_SEARCH_TIMEOUT.as_secs()
                ));
            }
        }
    }
    (
        StatusCode::OK,
        [(axum::http::header::CONTENT_TYPE, "application/json")],
        body.to_string(),
    )
        .into_response()
}

/// Map a piper synthesis error to its teaching response (404 for a
/// missing lane/voice, 400 otherwise). Shared by the buffered WAV path
/// and the PCM stream's eager first chunk.
fn tts_error(e: &anyhow::Error) -> Response {
    let msg = format!("{e:#}");
    let status = if msg.contains("not installed") || msg.contains("is not pulled") {
        StatusCode::NOT_FOUND
    } else {
        StatusCode::BAD_REQUEST
    };
    openai_error(status.as_u16(), &msg)
}

/// Piper normalises synthesis to full scale: every WAV peaks at exactly
/// 0 dBFS, one sample away from clipping under any downstream gain. The
/// egress scales such peaks down to this ceiling (−1 dBFS) instead of
/// touching synthesis — quiet audio stays byte-identical because the
/// scale only applies when the peak would exceed it.
const PEAK_CEILING: f32 = 0.891_250_9; // 10^(-1/20)

/// Scale a PCM16LE WAV's peak down to [`PEAK_CEILING`] when (and only
/// when) it exceeds it. Anything unexpected in the container is passed
/// through untouched with a warn log — honest, scope-bounded egress.
pub(crate) fn limit_wav_peak(mut wav: Vec<u8>) -> Vec<u8> {
    match limit_wav_peak_in_place(&mut wav) {
        Ok(()) => wav,
        Err(reason) => {
            tracing::warn!("speech WAV left unmodified by peak limiter: {reason}");
            wav
        }
    }
}

/// A located PCM16LE payload inside a RIFF/WAVE container.
struct WavLayout {
    channels: u16,
    sample_rate: u32,
    bits: u16,
    /// Byte range of the `data` chunk body inside the container.
    data: std::ops::Range<usize>,
}

/// Chunk-walk a RIFF/WAVE container to its PCM16LE data payload.
/// Chunks between fmt and data (LIST and friends) are stepped over by
/// declared size without being interpreted. Shared by the peak limiter
/// and the PCM stream extractor so both agree on one parser.
fn locate_pcm16(wav: &[u8]) -> Result<WavLayout, String> {
    let read_u16 = |b: &[u8], off: usize| u16::from_le_bytes([b[off], b[off + 1]]);
    if wav.len() < 12 || &wav[0..4] != b"RIFF" || &wav[8..12] != b"WAVE" {
        return Err("not a RIFF/WAVE container".to_string());
    }
    let mut fmt: Option<(u16, u16, u32, u16)> = None; // (format, channels, rate, bits)
    let mut data: Option<std::ops::Range<usize>> = None;
    let mut pos = 12;
    while pos + 8 <= wav.len() {
        let id = &wav[pos..pos + 4];
        let size =
            u32::from_le_bytes([wav[pos + 4], wav[pos + 5], wav[pos + 6], wav[pos + 7]]) as usize;
        let body = pos + 8;
        if body + size > wav.len() {
            return Err(format!(
                "chunk {} overruns the container",
                String::from_utf8_lossy(id)
            ));
        }
        match id {
            b"fmt " if size >= 16 => {
                fmt = Some((
                    read_u16(wav, body),
                    read_u16(wav, body + 2),
                    u32::from_le_bytes([
                        wav[body + 4],
                        wav[body + 5],
                        wav[body + 6],
                        wav[body + 7],
                    ]),
                    read_u16(wav, body + 14),
                ));
            }
            b"data" => data = Some(body..body + size),
            _ => {}
        }
        pos = body + size + (size & 1); // chunks are word-aligned
    }
    let (Some((audio_format, channels, sample_rate, bits)), Some(data)) = (fmt, data) else {
        return Err("missing fmt or data chunk".to_string());
    };
    if (audio_format, bits) != (1, 16) {
        return Err(format!("not PCM16LE (format {audio_format}, {bits} bits)"));
    }
    if data.len() % 2 != 0 {
        return Err("odd-length PCM16 data chunk".to_string());
    }
    Ok(WavLayout {
        channels,
        sample_rate,
        bits,
        data,
    })
}

/// Sentence chunking for the PCM stream. Sentences end at `.`, `!`, `?`,
/// `…` or a newline ONLY when followed by whitespace or end-of-text (so
/// decimals like `3.14` and abbreviations stay intact), then group into
/// chunks of at least [`STREAM_MIN_CHUNK_CHARS`] so piper is not spawned
/// per word, hard-splitting anything over [`STREAM_MAX_CHUNK_CHARS`] so
/// no single sentence can dominate time-to-first-audio.
const STREAM_MIN_CHUNK_CHARS: usize = 120;
const STREAM_MAX_CHUNK_CHARS: usize = 600;

fn split_streaming_chunks(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let terminator = |c: char| matches!(c, '.' | '!' | '?' | '…' | '\n');
    let mut sentences: Vec<String> = Vec::new();
    let mut cur = String::new();
    for (i, &c) in chars.iter().enumerate() {
        cur.push(c);
        let at_end = i + 1 == chars.len();
        if terminator(c) && (at_end || chars[i + 1].is_whitespace()) {
            sentences.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        sentences.push(cur);
    }

    let mut chunks: Vec<String> = Vec::new();
    let mut pending = String::new();
    for sentence in sentences {
        if sentence.chars().count() > STREAM_MAX_CHUNK_CHARS {
            if !pending.is_empty() {
                chunks.push(std::mem::take(&mut pending));
            }
            // Oversized sentence: fall back to clause breaks, then hard
            // slices, so chunk sizes stay bounded.
            let mut piece = String::new();
            for clause in sentence.split_inclusive([';', ':', ',']) {
                if piece.chars().count() + clause.chars().count() > STREAM_MAX_CHUNK_CHARS
                    && !piece.is_empty()
                {
                    chunks.push(std::mem::take(&mut piece));
                }
                piece.push_str(clause);
                while piece.chars().count() > STREAM_MAX_CHUNK_CHARS {
                    let mut take = STREAM_MAX_CHUNK_CHARS;
                    while !piece.is_char_boundary(take) {
                        take -= 1;
                    }
                    chunks.push(piece[..take].to_string());
                    piece = piece[take..].to_string();
                }
            }
            if !piece.is_empty() {
                chunks.push(piece);
            }
            continue;
        }
        pending.push_str(&sentence);
        if pending.chars().count() >= STREAM_MIN_CHUNK_CHARS {
            chunks.push(std::mem::take(&mut pending));
        }
    }
    if !pending.trim().is_empty() {
        chunks.push(pending);
    }
    // Input is validated non-empty; a whitespace-only edge must still
    // produce one synthesizable chunk, never an empty vec.
    if chunks.is_empty() {
        chunks.push(text.to_string());
    }
    chunks
}

fn limit_wav_peak_in_place(wav: &mut [u8]) -> Result<(), String> {
    let layout = locate_pcm16(wav)?;
    let data_off = layout.data.start;
    let data_len = layout.data.len();
    let samples: Vec<i16> = wav[data_off..data_off + data_len]
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| i16::from_le_bytes(*c))
        .collect();
    let peak = samples.iter().fold(0i16, |a, &s| a.max(s.abs()));
    let target = PEAK_CEILING * f32::from(i16::MAX);
    if f32::from(peak) <= target {
        return Ok(()); // quiet audio: byte-identical egress
    }
    let scale = target / f32::from(peak);
    for (slot, s) in wav[data_off..data_off + data_len]
        .as_chunks_mut::<2>()
        .0
        .iter_mut()
        .zip(samples)
    {
        // Float rounding back to i16 is intentional here: `as` saturates,
        // so the ceiling can never overshoot into clip.
        #[allow(clippy::cast_possible_truncation)]
        let scaled = (f32::from(s) * scale).round() as i16;
        slot.copy_from_slice(&scaled.to_le_bytes());
    }
    Ok(())
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    /// Minimal PCM16LE WAV builder; extra chunks are inserted verbatim
    /// between fmt and data like real piper output (LIST INFO tags).
    /// Test sizes are tiny by construction, so length casts cannot truncate.
    #[allow(clippy::cast_possible_truncation)]
    fn wav_bytes(samples: &[i16], mid_chunk: Option<(&[u8], &[u8])>) -> Vec<u8> {
        let mut w = Vec::new();
        let data: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        let fmt = {
            let mut f = Vec::new();
            f.extend_from_slice(&1u16.to_le_bytes()); // PCM
            f.extend_from_slice(&1u16.to_le_bytes()); // mono
            f.extend_from_slice(&22_050u32.to_le_bytes());
            f.extend_from_slice(&44_100u32.to_le_bytes()); // byte rate
            f.extend_from_slice(&2u16.to_le_bytes()); // block align
            f.extend_from_slice(&16u16.to_le_bytes()); // bits
            f
        };
        w.extend_from_slice(b"RIFF");
        let riff_len =
            4 + (8 + fmt.len()) + mid_chunk.map_or(0, |(_, d)| 8 + d.len()) + 8 + data.len();
        w.extend_from_slice(&(riff_len as u32).to_le_bytes());
        w.extend_from_slice(b"WAVE");
        w.extend_from_slice(b"fmt ");
        w.extend_from_slice(&(fmt.len() as u32).to_le_bytes());
        w.extend_from_slice(&fmt);
        if let Some((id, payload)) = mid_chunk {
            w.extend_from_slice(id);
            w.extend_from_slice(&(payload.len() as u32).to_le_bytes());
            w.extend_from_slice(payload);
            if payload.len() % 2 == 1 {
                w.push(0); // RIFF chunks are word-aligned
            }
        }
        w.extend_from_slice(b"data");
        w.extend_from_slice(&(data.len() as u32).to_le_bytes());
        w.extend_from_slice(&data);
        w
    }

    /// Chunk-walk to the PCM samples; mirrors the walker under test.
    fn pcm_samples(wav: &[u8]) -> Vec<i16> {
        let mut pos = 12;
        loop {
            let size = u32::from_le_bytes([wav[pos + 4], wav[pos + 5], wav[pos + 6], wav[pos + 7]])
                as usize;
            if &wav[pos..pos + 4] == b"data" {
                return wav[pos + 8..pos + 8 + size]
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| i16::from_le_bytes(*c))
                    .collect();
            }
            pos += 8 + size + (size & 1);
        }
    }

    fn peak_of(wav: &[u8]) -> i16 {
        pcm_samples(wav).iter().fold(0i16, |a, s| a.max(s.abs()))
    }

    #[test]
    fn unit__parse_speech_format__openai_surface_pcm16_alias_and_teaching() {
        assert_eq!(parse_speech_format(None), Ok(SpeechFormat::Wav));
        assert_eq!(parse_speech_format(Some("wav")), Ok(SpeechFormat::Wav));
        // OpenAI's pcm16 names the same raw s16le lane as pcm.
        assert_eq!(parse_speech_format(Some("pcm")), Ok(SpeechFormat::Pcm));
        assert_eq!(parse_speech_format(Some("pcm16")), Ok(SpeechFormat::Pcm));
        assert_eq!(
            parse_speech_format(Some("mp3")),
            Ok(SpeechFormat::Lossy(LossyFormat::Mp3))
        );
        assert_eq!(
            parse_speech_format(Some("flac")),
            Ok(SpeechFormat::Lossy(LossyFormat::Flac))
        );
        // Wrong picks teach the whole supported surface.
        let err = parse_speech_format(Some("wma")).unwrap_err();
        assert!(err.contains("wav"), "teaching lists wav: {err}");
        assert!(err.contains("pcm16"), "teaching lists the alias: {err}");
        assert!(
            err.contains("flac"),
            "teaching lists the lossy family: {err}"
        );
    }

    #[test]
    fn unit__ffmpeg_argv__gateway_fixed_enum_only() {
        // mp3: bitrate flag then codec/muxer, mp3 container on stdout.
        assert_eq!(
            ffmpeg_argv(LossyFormat::Mp3),
            vec![
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "wav",
                "-i",
                "pipe:0",
                "-b:a",
                "128k",
                "-c:a",
                "libmp3lame",
                "-f",
                "mp3",
                "pipe:1",
            ]
        );
        // flac is lossless: no rate flags at all.
        let flac = ffmpeg_argv(LossyFormat::Flac);
        assert!(!flac.contains(&"-b:a".to_string()));
        assert_eq!(flac[flac.len() - 3..], vec!["-f", "flac", "pipe:1"]);
    }

    #[test]
    fn unit__parse_native_options__ranges_and_teachings() {
        let base = serde_json::json!({
            "model": "en_US-amy-medium",
            "input": "hi",
        });
        let opts = parse_native_options(&base).expect("plain body parses");
        assert_eq!(opts, blazar_runtime::piper::SpeakOptions::default());

        let full = serde_json::json!({
            "speed": 2.0, "speaker": 2, "noise_scale": 0.7,
            "noise_w": 0.9, "sentence_silence": 0.5,
        });
        let opts = parse_native_options(&full).expect("full native body parses");
        assert_eq!(opts.speaker, Some(2));
        assert_eq!(opts.sentence_silence, Some(0.5));

        // OpenAI's instructions field teaches honestly: piper voices are
        // fixed, style is not steerable.
        let err =
            parse_native_options(&serde_json::json!({"instructions": "whisper"})).unwrap_err();
        assert!(err.contains("piper"), "teaching names the engine: {err}");

        assert!(
            parse_native_options(&serde_json::json!({"speaker": 1.5})).is_err(),
            "fractional speaker indices reject"
        );
        assert!(
            parse_native_options(&serde_json::json!({"noise_scale": 3.0})).is_err(),
            "noise_scale above the envelope rejects"
        );
        assert!(
            parse_native_options(&serde_json::json!({"sentence_silence": 30.0})).is_err(),
            "pathological sentence_silence rejects"
        );
        assert!(
            parse_native_options(&serde_json::json!({"speed": 5.0})).is_err(),
            "speed keeps the OpenAI range"
        );
    }

    #[test]
    fn unit__voice_row__splits_locale_name_quality_and_survives_odd_ids() {
        let row = voice_row("en_US-amy-medium", true, None);
        assert_eq!(row["id"], "en_US-amy-medium");
        assert_eq!(row["language"], "en_US");
        assert_eq!(row["name"], "amy");
        assert_eq!(row["quality"], "medium");
        assert_eq!(row["installed"], true);
        // Catalog rows carry their download size.
        let remote = voice_row("de_DE-karlsson-low", false, Some(22_000_000));
        assert_eq!(remote["bytes"], 22_000_000);
        // An unparsable id still lists (null parts), never drops.
        let odd = voice_row("custom", true, None);
        assert_eq!(odd["id"], "custom");
        assert_eq!(odd["language"], serde_json::Value::Null);
    }

    #[test]
    fn unit__limit_wav_peak__full_scale_wav_scaled_to_minus_1dbfs_rms_proportional() {
        // Half the samples ride the rail: piper's normalize-to-peak shape.
        let samples: Vec<i16> = (0..2000)
            .map(|i| {
                if i % 2 == 0 {
                    i16::MAX
                } else {
                    (i % 100) as i16 * 60
                }
            })
            .collect();
        let wav = wav_bytes(&samples, Some((b"LIST", b"INFOblazar-test")));
        let scaled = limit_wav_peak(wav.clone());
        let peak = f32::from(peak_of(&scaled));
        // -1 dBFS as a LITERAL, deliberately not PEAK_CEILING: pinning
        // the ceiling against its own constant is circular and let a
        // mutation of the constant to 1.0 pass green (audit MM11).
        // 10^(-1/20) = 0.8912509...
        let target = 0.891_250_9_f32 * f32::from(i16::MAX);
        assert!(
            (target - peak).abs() <= 1.0,
            "peak {peak} vs target {target}"
        );
        // Proportional scale: sample 60*99 must map by the same factor.
        // Test math is bounded well inside i16; the round-back cast truncates by design.
        #[allow(clippy::cast_possible_truncation)]
        let expect = ((60.0 * 99.0) * (target / f32::from(i16::MAX))).round() as i16;
        let got = pcm_samples(&scaled)[samples.len() - 1];
        assert!(
            (expect - got).abs() <= 2,
            "rms shape changed: {got} vs {expect}"
        );
        assert!(scaled.len() == wav.len(), "container must not resize");
    }

    #[test]
    fn unit__limit_wav_peak__quiet_wav_byte_identical() {
        let samples: Vec<i16> = (0..512).map(|i| ((i % 97) as i16 - 48) * 100).collect();
        let wav = wav_bytes(&samples, None);
        let out = limit_wav_peak(wav.clone());
        assert_eq!(out, wav, "quiet audio must egress byte-identical");
    }

    #[test]
    fn unit__limit_wav_peak__garbage_passthrough() {
        let garbage = vec![0xFF; 64];
        assert_eq!(limit_wav_peak(garbage.clone()), garbage);
    }

    #[test]
    fn unit__limit_wav_peak__non_pcm16_passthrough() {
        let mut wav = wav_bytes(&[1000, -1000], None);
        // Flip the format code to float (3): out of limiter scope.
        wav[20] = 3;
        wav[21] = 0;
        assert_eq!(limit_wav_peak(wav.clone()), wav);
    }

    #[test]
    fn unit__split_streaming_chunks__groups_sentences_and_keeps_decimals() {
        // Short text: all sentences group into ONE chunk (no piper-spam).
        let one = "One. Two words. Three short sentences here.";
        assert_eq!(split_streaming_chunks(one).len(), 1);

        // Decimal points and mid-word dots never terminate a sentence:
        // `3.14` stays glued to its sentence.
        let dec = "Pi is 3.14 rounded. It is not 2.71 at all.";
        let chunks = split_streaming_chunks(dec);
        assert_eq!(chunks.len(), 1, "short decimal text stays one chunk");
        assert!(chunks[0].contains("3.14"));

        // Two long sentences (each >= MIN on its own): one chunk each.
        let s1 = format!("Alpha {}. ", "sentence body words ".repeat(10));
        let s2 = format!("Beta {}. ", "other body words here ".repeat(10));
        assert!(s1.chars().count() >= STREAM_MIN_CHUNK_CHARS);
        let chunks = split_streaming_chunks(&format!("{s1} {s2}"));
        assert_eq!(chunks.len(), 2, "each >=MIN sentence is its own chunk");
        assert!(chunks[0].starts_with("Alpha"));
        assert!(chunks[1].trim_start().starts_with("Beta"));
    }

    #[test]
    fn unit__split_streaming_chunks__oversized_sentence_hard_split_bounded() {
        // No terminator at all: hard slices, each within the cap.
        let blob: String = "w".repeat(1_500);
        let chunks = split_streaming_chunks(&blob);
        assert!(chunks.len() >= 3, "1500 chars must not be one chunk");
        assert!(
            chunks
                .iter()
                .all(|c| c.chars().count() <= STREAM_MAX_CHUNK_CHARS),
            "every chunk must be bounded"
        );
        assert_eq!(chunks.concat(), blob, "hard split must be lossless");
        assert!(chunks.iter().all(|c| !c.is_empty()));
    }

    #[test]
    fn unit__locate_pcm16__reports_layout_and_steps_mid_chunks() {
        let samples = [100i16, -200, 300];
        let wav = wav_bytes(&samples, Some((b"LIST", b"INFOmid-chunk")));
        let layout = locate_pcm16(&wav).expect("valid piper-shaped WAV");
        assert_eq!(layout.channels, 1);
        assert_eq!(layout.sample_rate, 22_050);
        assert_eq!(layout.bits, 16);
        let expect: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        assert_eq!(&wav[layout.data.clone()], &expect[..]);
        // Garbage is rejected, not guessed at.
        assert!(locate_pcm16(&[0xFF; 64]).is_err());
    }
}
