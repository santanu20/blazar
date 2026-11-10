"""Resident Kokoro TTS server for the blazar kokoro lane.

Stdlib http.server only - numpy and kokoro_onnx ride the lane venv the
shim execs this script with. Endpoints:

  GET  /health  -> {"status": "ok"} once the model is resident (the
                   load happens at boot, so health == loaded)
  GET  /voices  -> {"voices": [...]} sorted voice-pack names
  POST /tts     -> {"input", "voice", "speed", "lang"} -> audio/wav
                   (16-bit PCM mono, 24 kHz - kokoro's native rate)

Voice errors from kokoro (unknown name, empty input) map to 400 with
the message intact so the gateway can teach; everything else is a 500.
The server binds 127.0.0.1 only - it is a lane-local child, not a
public endpoint.
"""

import io
import json
import os
import sys
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from kokoro_onnx import Kokoro

LANE = Path(__file__).resolve().parent
DEFAULT_VOICE = "af_heart"

kokoro = Kokoro(str(LANE / "kokoro.onnx"), str(LANE / "voices.bin"))


def voice_names():
    """Voice packs as a sorted list, whatever container np.load gave us
    (NpzFile exposes .files; a pickled dict exposes .keys)."""
    try:
        return sorted(str(v) for v in kokoro.voices.files)
    except AttributeError:
        pass
    try:
        return sorted(str(v) for v in kokoro.voices.keys())
    except AttributeError:
        return []


class Handler(BaseHTTPRequestHandler):
    def reply_json(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self.reply_json(200, {"status": "ok"})
        elif self.path == "/voices":
            self.reply_json(200, {"voices": voice_names()})
        else:
            self.reply_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/tts":
            self.reply_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            self.reply_json(400, {"error": "request body must be JSON"})
            return

        text = str(body.get("input", "")).strip()
        voice = str(body.get("voice") or DEFAULT_VOICE)
        if voice == "kokoro":
            voice = DEFAULT_VOICE
        speed = float(body.get("speed", 1.0))
        lang = str(body.get("lang", "en-us"))
        if not text:
            self.reply_json(400, {"error": "input must be non-empty text"})
            return

        try:
            samples, sample_rate = kokoro.create(
                text, voice=voice, speed=speed, lang=lang
            )
        except ValueError as err:
            # kokoro's own contract violations (unknown voice, bad speed)
            self.reply_json(400, {"error": str(err)})
            return

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes((samples * 32767).astype("<i2").tobytes())
        wav_bytes = buf.getvalue()

        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(wav_bytes)))
        self.end_headers()
        self.wfile.write(wav_bytes)

    def log_message(self, format, *args):  # noqa: A002 - stdlib signature
        # Lane child: stay quiet unless BLAZAR_DEBUG is set. os.environ,
        # not sys.environ - some hardened interpreters scrub the latter
        # while os.environ keeps working.
        if os.environ.get("BLAZAR_DEBUG"):
            sys.stderr.write("kokoro-server: " + (format % args) + "\n")


def main():
    argv = sys.argv[1:]
    port = 8201
    if "--port" in argv:
        port = int(argv[argv.index("--port") + 1])
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
