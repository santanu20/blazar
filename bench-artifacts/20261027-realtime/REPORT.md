# /v1/realtime voice session — first live receipt

**Date:** 2026-10-27 · **Rig:** isolated daemon `127.0.0.1:11437` (XDG sandbox,
symlinked real assets, sqlite `.backup` store) · **Binary:** debug build of
working tree (includes the `AppendErr` decode/cap split) · **Host:** production
box, ollama 11434 and the user daemon 11435 untouched.

## Loop under test

```
piper TTS → PCM → /v1/realtime WS (append+commit)
  → whisper STT → qwen3-0.6b chat → piper TTS → base64 audio deltas
```

## Results

| Stage | Evidence |
|---|---|
| Session | `session.created` (id `rt…`, model/voice/stt echo) → `session.updated` (instructions applied) |
| STT | transcript `" The quick brown fox jumps over the lazy dog.\n"` — similarity 0.9778 vs synthesized phrase |
| Chat | one-sentence reply honoring the `instructions` system prompt |
| TTS | 44× `response.audio.delta` (161,440 bytes) → `response.audio.done`; reassembled reply = valid WAV 22.05 kHz mono 16-bit 3.66 s |

| Timing | ms |
|---|---|
| piper synthesis (input phrase) | 677 |
| WS connect + session setup | 31 |
| full turn (STT+chat+TTS) | 7,416 |

## Event cascade (exact order)

`session.created → session.updated → input_audio_transcription.completed →
response.audio_transcript.delta → response.audio_transcript.done →
response.audio.delta ×44 → response.audio.done`

Full trace: `events.json` · machine receipt: `receipt.json` · audio:
`tts_input.wav`, `tts_reply.wav`.

## Contract pins (gateway test suite, 13/13)

- missing `?model=` → HTTP 400 before upgrade (auth still enforced first)
- lifecycle: session.update echo, item.created ids (`item_1`, `realtime.item`),
  buffer clear ack, unknown-event teaching, idle-cancel teaching
- empty-buffer commit / `response.create` → `invalid_request`
- junk audio turn (no whisper engine) → `input_audio_transcription.failed` +
  teaching error, **socket survives**
- bad base64 append → `invalid_request` (non-fatal; fixed this wave — was
  session-fatal `buffer_overflow` close), >20 MiB → `buffer_overflow` close

## Change shipped alongside

`realtime.rs`: `append_audio` now splits `AppendErr::Decode` (taught, session
continues — a one-frame client bug must not cost the conversation context)
from `AppendErr::Cap` (fatal, the 20 MiB memory bound holds). Both call sites
(main loop + mid-turn dispatch) route through `append_failed`; unit pin
`unit__append_audio__decode_taught_cap_fatal`.

## Reproduce

```bash
# rig: XDG_CONFIG_HOME/XDG_DATA_HOME sandbox, symlink models/engines/whisper/voices,
# sqlite .backup of blazar.db, config `port = 11437`; ./target/debug/blazar serve
python3 - <<'PY'  # drive: TTS phrase → WS append/commit → assert cascade
# (client script preserved in session records; core = websockets + b64 PCM)
PY
cargo nextest run -p blazar-gateway -E 'test(realtime)'
```
