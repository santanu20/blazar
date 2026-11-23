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
Every failure is answered as JSON - a lane child must never drop the
connection, because the gateway surfaces the body as the teaching text.
The server binds 127.0.0.1 only - it is a lane-local child, not a
public endpoint.
"""

import io
import json
import math
import os
import sys
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from kokoro_onnx import Kokoro

LANE = Path(__file__).resolve().parent
DEFAULT_VOICE = "af_heart"
# A request is a small JSON object; anything past this is a client bug.
MAX_BODY_BYTES = 1 << 20

kokoro = Kokoro(str(LANE / "kokoro.onnx"), str(LANE / "voices.bin"))
# ThreadingHTTPServer answers requests concurrently while the Kokoro session
# is one shared object whose create() is not documented as thread-safe, so
# synthesis is serialized rather than racing on the model.
synthesis_lock = threading.Lock()


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
        self.send_header("Connection", "close")
        self.close_connection = True
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self.reply_json(200, {"status": "ok"})
        elif self.path == "/voices":
            self.reply_json(200, {"voices": voice_names()})
        else:
            self.reply_json(404, {"error": "not found"})

    def read_json_body(self):
        """The request body as a JSON object, or None after replying 4xx."""
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            self.reply_json(400, {"error": "Content-Length must be a number"})
            return None
        if length < 0:
            self.reply_json(400, {"error": "Content-Length must not be negative"})
            return None
        if length > MAX_BODY_BYTES:
            # Drain a bounded prefix so the client reads the 413 instead of
            # hitting a reset while it is still writing the body.
            remaining = min(length, MAX_BODY_BYTES)
            while remaining > 0:
                chunk = self.rfile.read(min(remaining, 65536))
                if not chunk:
                    break
                remaining -= len(chunk)
            self.reply_json(
                413, {"error": f"request body exceeds {MAX_BODY_BYTES} bytes"}
            )
            return None
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            self.reply_json(400, {"error": "request body must be JSON"})
            return None
        if not isinstance(body, dict):
            self.reply_json(400, {"error": "request body must be a JSON object"})
            return None
        return body

    def do_POST(self):
        if self.path != "/tts":
            self.reply_json(404, {"error": "not found"})
            return
        body = self.read_json_body()
        if body is None:
            return

        text = str(body.get("input", "")).strip()
        voice = str(body.get("voice") or DEFAULT_VOICE)
        if voice == "kokoro":
            voice = DEFAULT_VOICE
        lang = str(body.get("lang", "en-us"))
        if not text:
            self.reply_json(400, {"error": "input must be non-empty text"})
            return

        try:
            speed = float(body.get("speed", 1.0))
            # NaN/inf or out-of-range speeds produce garbage audio; reject
            # them with the lane's 400 shape instead of a dropped connection.
            if not math.isfinite(speed) or not (0.5 <= speed <= 2.0):
                self.reply_json(
                    400,
                    {
                        "error": f"speed must be a finite number in [0.5, 2.0], got {body.get('speed')!r}"
                    },
                )
                return
        except (TypeError, ValueError):
            self.reply_json(
                400, {"error": f"speed must be a number, got {body.get('speed')!r}"}
            )
            return

        try:
            with synthesis_lock:
                samples, sample_rate = kokoro.create(
                    text, voice=voice, speed=speed, lang=lang
                )
        except ValueError as err:
            # kokoro's own contract violations (unknown voice, bad speed)
            self.reply_json(400, {"error": str(err)})
            return
        except Exception as err:
            # The lane contract promises a 500 over a dropped connection
            # for anything the model raises that is not a contract error.
            self.reply_json(500, {"error": f"synthesis failed: {err}"})
            return

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            # Samples outside [-1, 1] would wrap when cast to int16; clipping
            # keeps the waveform bounded instead of turning into noise.
            wav.writeframes(
                (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()
            )
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
