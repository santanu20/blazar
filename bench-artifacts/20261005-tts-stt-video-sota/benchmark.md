# 2026-10-05 — TTS/STT + image/video/realtime SOTA live receipts

Session that brought the audio/image/video/realtime API surfaces to their
current shape (Waves W1–W8 of plan dp-20261005-tts-stt-image-video-sota).
This file records the LIVE end-to-end receipts (no mocks; every row below
was a real request against a rigged daemon + real engine children).

## Environment

| Item | Value |
|---|---|
| blazar | 0.20.0 debug build (working tree, post W1–W8 + video-delete fix) |
| rig | isolated XDG daemon `127.0.0.1:11501`, DB seeded from a store snapshot, engines/models/voices/whisper symlinked read-only |
| GPU | RTX 4070 Laptop 8 GiB (another actor's mistralrs rig held 6.7 GiB during planning; lanes below ran after it exited) |
| engines | piper 2023.11.14-2, whisper.cpp b5130 (ggml-base), sd-server master-929-3f8527a |
| models | en_US-amy-medium voice; qwen-image-2.1-uncensored:q6_k + qwen_image_2.1_vae + umt5-xxl encoder; wan_2.1_comfyui_repackaged (t2v 1.3B bf16 + wan vae); chat qwen3-0.6b:q4_0 |
| ffmpeg | 6.1.1 (system, optional transcode dep) |

## Receipts

| # | Lane | Receipt | Evidence |
|---|---|---|---|
| 1 | TTS voices | GET /v1/audio/voices lists installed voice with parsed locale/name/quality | curl 200, `voices` body in daemon log |
| 2 | TTS wav | default format, piper native RIFF PCM16 mono 22050 Hz | `tts-wav.wav` (117 KB) |
| 3 | TTS mp3 | ffmpeg transcode lane, ID3v2.4 + MPEG L3 128 kbps 22.05 kHz | `tts-mp3.mp3` (54 KB) |
| 4 | TTS flac | lossless container | `tts-flac.flac` (51 KB) |
| 5 | TTS native knobs | speaker/noise_scale/noise_w/sentence_silence ride argv, valid WAV out | `tts-native-knobs.wav` |
| 6 | STT word granularity | verbose_json forced, engine-native words[] passed through (token_timestamps), header `x-blazar-word-timestamps: derived` | `stt-word-granularity.json` + `stt-word-headers.txt` |
| 7 | STT srt relay | verbatim relay, content-type application/x-subrip | `stt-srt-relay.srt` |
| 8 | STT teachings | diarized_json / include=logprobs / keywords all 400 with menu + prompt steering | session log (daemon.log rig) |
| 9 | images generations | qwen-image 512x512 PNG, 73 s cold (boot+load+gen) | `img-generations.png` |
| 10 | images quality+style | quality=low style=vivid translated via live capabilities defaults, 200 in 44 s warm | `img-quality-style.json` |
| 11 | images variations | multipart image → native img2img init_image, EMPTY prompt accepted by engine, 200 in 46 s warm | `img-variations.png` |
| 12 | video submission | seconds:"4" → 61 aligned frames (4n+1) admission math; VRAM teaching 400 fired honestly at 512x512 and 384x384; `vram_overcommit:true` accepted | `video-async-post.json` (async job plane) |
| 13 | video render | wan2.1 t2v 61 frames vp8 384x384 @16 fps, container 3.75 s; Sora object: status completed, progress 100, seconds 4.0 (request-echo arm), size echo | `video-content.webm` (1.4 MB) + ffprobe |
| 14 | Sora verbs | GET /v1/videos list (object:list, first_id/last_id, historical rows seconds=0.313 = 5/16 exact), GET /{id} shapes, GET /{id}/content bytes+mime | session log |
| 15 | video DELETE contract | completed → DELETE → cancelled; content → 404 "cancelled or deleted"; idempotent second DELETE (fix landed same session: store `delete_job_result` + `JobRuntime::record_deleted` + handler arms) | session log |
| 16 | realtime v2 events | full session walk: session.created → session.update→updated → conversation.item.created → buffer.append/commit → input_audio_transcription.completed → response.audio_transcript.delta/done → response.audio.delta×~100 → response.audio.done → buffer.clear→cleared → response.cancel-outside-turn teaching → unknown-event teaching naming the vocabulary | rt.py transcript (session log) |

## Known gaps (honest, engine-bound)

- whisper `diarize`/`tinydiarize`/`dtw`/`no_language_probabilities` form
  fields are NOT whitelisted: binary-verified as CLI flags but behavioral
  field acceptance unproven (needs stereo probe audio + a dtw model file).
  Gateway drops them per the documented unknown-field contract until
  proven; diarized_json stays a teaching 400.
- Video generation held the full GPU for ~11 min per 61-frame render on
  this 8 GiB laptop part under overcommit; the admission gate's default
  refusal at 512x512 was correct and honest.
- No e2e receipt for opus/aac lossy TTS (mp3+flac cover the transcode
  lane; argv builder is unit-pinned for all four).
