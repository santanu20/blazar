//! POST `/v1/audio/speech` — `OpenAI` TTS shape on the local piper lane.
//!
//! `model` IS the piper voice id (honest mapping: `en_US-amy-medium`),
//! `response_format` speaks WAV only in this phase (the lane is a
//! one-shot binary writing a WAV; container conversion is a codec
//! concern, not a synthesis one), `speed` maps to `piper`'s inverse
//! length scale. Remote intent (`name:model`) forwards like every
//! other lane.

use axum::body::Bytes;
use axum::extract::{Extension, State};
use axum::http::{HeaderMap, Method, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use serde_json::Value;
use std::sync::Arc;

use crate::proxy::openai_error;
use crate::remotes::split_remote;
use crate::state::AppState;

/// Bound mirrors the runtime cap; the route validates first so clients
/// get a 400 before any spawn.
const MAX_INPUT_CHARS: usize = 10_000;

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
    match req.get("response_format").and_then(Value::as_str) {
        None | Some("wav") => {}
        Some(other) => {
            return openai_error(
                400,
                &format!(
                    "response_format {other:?} is not supported on the local lane — \
                     piper speaks WAV (omit response_format); convert downstream if you need {other}"
                ),
            )
        }
    }
    let speed = match req.get("speed").and_then(Value::as_f64) {
        None => None,
        Some(s) if (0.25..=4.0).contains(&s) => Some(s),
        Some(s) => return openai_error(400, &format!("speed {s} out of range (0.25..=4.0)")),
    };

    // Remote intent wins: `name:model` never goes local.
    if split_remote(voice, &state.config).is_some() {
        if let Some(Extension(k)) = &key_ext {
            if let Some(entry) = state.keys.entry(&k.name) {
                if let Err(rej) = state.keys.check(&entry, voice) {
                    return rej.to_response();
                }
                state.keys.charge_request(&k.name);
            }
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

    // Local lane: the runtime error carries the exact teaching (install
    // vs pull vs voice list) — pass it through with the right status.
    match blazar_runtime::piper::synthesize(
        &state.dirs,
        voice,
        text,
        speed,
        std::time::Duration::from_secs(120),
    )
    .await
    {
        Ok(wav) => {
            let wav = limit_wav_peak(wav);
            Response::builder()
                .status(StatusCode::OK)
                .header(axum::http::header::CONTENT_TYPE, "audio/wav")
                .body(axum::body::Body::from(wav))
                .unwrap_or_else(|_| openai_error(500, "response build").into_response())
        }
        Err(e) => {
            let msg = format!("{e:#}");
            let status = if msg.contains("not installed") || msg.contains("is not pulled") {
                StatusCode::NOT_FOUND
            } else {
                StatusCode::BAD_REQUEST
            };
            openai_error(status.as_u16(), &msg)
        }
    }
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

fn limit_wav_peak_in_place(wav: &mut [u8]) -> Result<(), String> {
    let read_u16 = |b: &[u8], off: usize| u16::from_le_bytes([b[off], b[off + 1]]);
    if wav.len() < 12 || &wav[0..4] != b"RIFF" || &wav[8..12] != b"WAVE" {
        return Err("not a RIFF/WAVE container".to_string());
    }
    // Chunk walk by declared size: LIST and friends between fmt and data
    // are preserved byte-for-byte without being interpreted.
    let mut fmt: Option<(u16, u16)> = None; // (audio_format, bits_per_sample)
    let mut data: Option<(usize, usize)> = None; // (offset, len)
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
                fmt = Some((read_u16(wav, body), read_u16(wav, body + 14)));
            }
            b"data" => data = Some((body, size)),
            _ => {}
        }
        pos = body + size + (size & 1); // chunks are word-aligned
    }
    let (Some(fmt), Some((data_off, data_len))) = (fmt, data) else {
        return Err("missing fmt or data chunk".to_string());
    };
    if fmt != (1, 16) {
        return Err(format!("not PCM16LE (format {}, {} bits)", fmt.0, fmt.1));
    }
    if data_len % 2 != 0 {
        return Err("odd-length PCM16 data chunk".to_string());
    }
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
        let target = PEAK_CEILING * f32::from(i16::MAX);
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
}
