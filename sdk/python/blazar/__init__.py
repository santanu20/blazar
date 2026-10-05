"""Zero-dependency Python client for Blazar.

Speaks the gateway's native OpenAI-compatible surface (`/v1/*`) plus the
operational endpoints that set Blazar apart (`/api/ps`, `/api/explain`).
Standard library only — `pip install` nothing.

    from blazar import Client

    blazar = Client()                      # http://127.0.0.1:11434
    reply = blazar.chat("qwen3-0.6b:q4_0",
                        [{"role": "user", "content": "hi"}])
    print(reply["choices"][0]["message"]["content"])

Streaming iterates server-sent events as they land:

    for chunk in blazar.chat("qwen3-0.6b:q4_0", msgs, stream=True):
        delta = chunk["choices"][0]["delta"].get("content", "")
        print(delta, end="", flush=True)

Batch (Anthropic dialect), completion metadata, and realtime voice:

    batch = blazar.anthropic_batch_create([
        {"custom_id": "req-1",
         "params": {"model": "qwen3-0.6b:q4_0", "max_tokens": 256,
                    "messages": [{"role": "user", "content": "hi"}]}},
    ])
    # ... poll blazar.anthropic_batch(batch["id"]) until "ended" ...
    rows = blazar.anthropic_batch_results(batch["id"])

    reply = blazar.chat(model, msgs, metadata={"trace": "abc"})
    card = blazar.metadata_update(reply["id"], {"trace": "abc-2"})

    with blazar.realtime(model) as rt:
        out = rt.voice_turn(pcm_bytes)   # STT -> chat -> TTS, one call
"""

from __future__ import annotations

import base64
import json
import os
import socket
import typing as t
import urllib.error
import urllib.request

__all__ = ["BlazarError", "Client", "RealtimeSession"]
__version__ = "0.2.0"


class BlazarError(RuntimeError):
    """Gateway rejected or failed a request.

    Carries the HTTP status and the teaching message Blazar returns in
    `error.message` — Blazar errors explain the fix, so surface them
    verbatim instead of guessing.
    """

    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"[{status}] {message}")
        self.status = status
        self.message = message


class Client:
    """Synchronous Blazar client.

    :param base_url: gateway origin, default ``http://127.0.0.1:11434``
    :param api_key:  bearer key when the gateway runs with named keys
    :param timeout:  per-request timeout in seconds (streaming readers
                     inherit the same deadline)
    """

    def __init__(
        self,
        base_url: t.Optional[str] = None,
        api_key: t.Optional[str] = None,
        timeout: float = 300.0,
    ) -> None:
        import os

        self.base_url = (
            base_url or os.environ.get("BLAZAR_URL") or "http://127.0.0.1:11434"
        ).rstrip("/")
        self.api_key = api_key or os.environ.get("BLAZAR_API_KEY")
        self.timeout = timeout

    # -- chat / completions -------------------------------------------------

    def chat(
        self,
        model: str,
        messages: t.List[t.Dict[str, t.Any]],
        *,
        stream: bool = False,
        tools: t.Optional[t.List[t.Dict[str, t.Any]]] = None,
        best_of: t.Optional[int] = None,
        mcp: t.Optional[str] = None,
        **kwargs: t.Any,
    ) -> t.Any:
        """POST /v1/chat/completions.

        ``best_of`` fans the request out server-side and returns the
        winner. ``mcp`` selects a configured MCP tool catalog (``"all"``
        or a server name); the gateway mediates the tool loop — it
        requires ``stream=False``.
        """
        body: t.Dict[str, t.Any] = {"model": model, "messages": messages, **kwargs}
        if stream:
            body["stream"] = True
        if tools is not None:
            body["tools"] = tools
        if best_of is not None:
            body["best_of"] = best_of
        if mcp is not None:
            body["mcp"] = mcp
        if stream:
            return self._stream("/v1/chat/completions", body)
        return self._request("/v1/chat/completions", body)

    def ollama_chat(
        self,
        model: str,
        messages: t.List[t.Dict[str, t.Any]],
        **kwargs: t.Any,
    ) -> t.Dict[str, t.Any]:
        """POST /api/chat — the ollama dialect, for migrant scripts."""
        return self._request(
            "/api/chat", {"model": model, "messages": messages, **kwargs}
        )

    def models(self) -> t.List[t.Dict[str, t.Any]]:
        """GET /v1/models."""
        return self._request("/v1/models", None)["data"]

    def metadata_update(
        self, completion_id: str, metadata: t.Dict[str, str]
    ) -> t.Dict[str, t.Any]:
        """POST /v1/chat/completions/{id} — edit a completion's metadata.

        Works on completions created non-streaming with a ``metadata``
        field; the gateway keeps a bounded card store for them.
        """
        from urllib.parse import quote

        return self._request(
            f"/v1/chat/completions/{quote(completion_id, safe='')}",
            {"metadata": metadata},
        )

    # -- anthropic batches ---------------------------------------------------

    def anthropic_batches(self) -> t.List[t.Dict[str, t.Any]]:
        """GET /v1/messages/batches — most recent first."""
        return self._request("/v1/messages/batches", None)["data"]

    def anthropic_batch(self, batch_id: str) -> t.Dict[str, t.Any]:
        """GET /v1/messages/batches/{id} — status and request counts."""
        return self._request(f"/v1/messages/batches/{batch_id}", None)

    def anthropic_batch_create(
        self, requests: t.List[t.Dict[str, t.Any]]
    ) -> t.Dict[str, t.Any]:
        """POST /v1/messages/batches — queue an Anthropic-dialect batch.

        Each item is ``{"custom_id": ..., "params": {/v1/messages body}}``.
        Poll ``anthropic_batch`` until ``processing_status == "ended"``,
        then read ``anthropic_batch_results``.
        """
        return self._request("/v1/messages/batches", {"requests": requests})

    def anthropic_batch_cancel(self, batch_id: str) -> t.Dict[str, t.Any]:
        """POST /v1/messages/batches/{id}/cancel."""
        return self._request(
            f"/v1/messages/batches/{batch_id}/cancel", {}, method="POST"
        )

    def anthropic_batch_delete(self, batch_id: str) -> t.Dict[str, t.Any]:
        """DELETE /v1/messages/batches/{id} — archive; stays readable."""
        return self._request(f"/v1/messages/batches/{batch_id}", None, method="DELETE")

    def anthropic_batch_results(self, batch_id: str) -> t.List[t.Dict[str, t.Any]]:
        """GET /v1/messages/batches/{id}/results — JSONL rows as dicts.

        Only after the batch ``ended``; otherwise the gateway 400s with a
        teaching message (raised as ``BlazarError``).
        """
        text = self._request_text(f"/v1/messages/batches/{batch_id}/results")
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    # -- realtime voice ------------------------------------------------------

    def realtime(
        self,
        model: str,
        *,
        voice: t.Optional[str] = None,
        stt_model: t.Optional[str] = None,
    ) -> "RealtimeSession":
        """Open a /v1/realtime voice session (STT -> chat -> TTS).

        Returns a connected ``RealtimeSession`` — use it as a context
        manager, then feed PCM16 mono 24 kHz audio via ``voice_turn``.
        """
        return RealtimeSession(self, model, voice, stt_model)

    # -- operations ----------------------------------------------------------

    def ps(self) -> t.Dict[str, t.Any]:
        """GET /api/ps — residents, engine posture, sleep state, remotes."""
        return self._request("/api/ps", None)

    def explain(self, model: str) -> t.Dict[str, t.Any]:
        """GET /api/explain/{model} — the routing decision card."""
        from urllib.parse import quote

        return self._request(f"/api/explain/{quote(model, safe='')}", None)

    def failover(self) -> t.Dict[str, t.Any]:
        """GET /api/failover — chain registry state (sticky, benched)."""
        return self._request("/api/failover", None)

    def mcp_servers(self) -> t.Any:
        """GET /api/mcp — configured MCP tool-catalog servers."""
        return self._request("/api/mcp", None)

    # -- transport -----------------------------------------------------------

    def _request(
        self,
        path: str,
        body: t.Optional[t.Dict[str, t.Any]],
        *,
        method: t.Optional[str] = None,
    ) -> t.Any:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            self.base_url + path,
            data=data,
            method=method or ("POST" if data else "GET"),
        )
        req.add_header("Content-Type", "application/json")
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = resp.read()
        except urllib.error.HTTPError as e:
            raise self._blazar_error(e) from None
        return json.loads(payload)

    def _request_text(self, path: str) -> str:
        req = urllib.request.Request(self.base_url + path, method="GET")
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            raise self._blazar_error(e) from None

    def _stream(
        self, path: str, body: t.Dict[str, t.Any]
    ) -> t.Iterator[t.Dict[str, t.Any]]:
        data = json.dumps(body).encode()
        req = urllib.request.Request(self.base_url + path, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Accept", "text/event-stream")
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            raise self._blazar_error(e) from None
        with resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if line.startswith("data:"):
                    payload = line[len("data:") :].strip()
                    if payload and payload != "[DONE]":
                        yield json.loads(payload)

    @staticmethod
    def _blazar_error(e: urllib.error.HTTPError) -> BlazarError:
        try:
            payload = json.loads(e.read().decode("utf-8", "replace"))
            message = (
                payload.get("error", {}).get("message")
                or payload.get("error")
                or str(e)
            )
        except Exception:
            message = str(e)
        return BlazarError(e.code, str(message))


class RealtimeSession:
    """One /v1/realtime voice session over a raw WebSocket (RFC 6455).

    No third-party libraries: the handshake and frames are implemented on
    ``socket`` so the SDK stays zero-dependency. Use as a context manager::

        with blazar.realtime("qwen3-0.6b:q4_0") as rt:
            out = rt.voice_turn(pcm)        # dict: transcript, reply, audio
            for event in rt.events():       # or drive the protocol by hand
                ...

    Audio in and out is PCM16LE mono 24 kHz; append events take raw bytes
    and base64-encode them here.
    """

    def __init__(
        self,
        client: Client,
        model: str,
        voice: t.Optional[str] = None,
        stt_model: t.Optional[str] = None,
    ) -> None:
        from urllib.parse import urlencode, urlparse

        parsed = urlparse(client.base_url)
        if parsed.hostname is None:
            raise BlazarError(0, f"cannot derive host from {client.base_url!r}")
        query = {"model": model}
        if voice:
            query["voice"] = voice
        if stt_model:
            query["stt_model"] = stt_model
        path = "/v1/realtime?" + urlencode(query)

        self._deadline_secs = client.timeout
        raw = socket.create_connection(
            (parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)),
            timeout=client.timeout,
        )
        sock: socket.socket = raw
        if parsed.scheme == "https":
            import ssl

            sock = ssl.create_default_context().wrap_socket(
                raw, server_hostname=parsed.hostname
            )
        self._sock = sock
        try:
            self._handshake(client, parsed, path)
        except Exception:
            self.close()
            raise
        self._recv_buf = b""

    # -- websocket plumbing --------------------------------------------------

    def _handshake(self, client: Client, parsed: t.Any, path: str) -> None:
        import hashlib
        import secrets

        key = base64.b64encode(secrets.token_bytes(16)).decode()
        headers = [
            f"GET {path} HTTP/1.1",
            f"Host: {parsed.netloc}",
            "Upgrade: websocket",
            "Connection: Upgrade",
            f"Sec-WebSocket-Key: {key}",
            "Sec-WebSocket-Version: 13",
        ]
        if client.api_key:
            headers.append(f"Authorization: Bearer {client.api_key}")
        sock = self._sock
        assert sock is not None
        sock.sendall(("\r\n".join(headers) + "\r\n\r\n").encode())

        response = b""
        while b"\r\n\r\n" not in response:
            chunk = sock.recv(4096)
            if not chunk:
                raise BlazarError(0, "realtime handshake: connection closed")
            response += chunk
        status_line = response.split(b"\r\n", 1)[0].decode("latin-1")
        if " 101 " not in status_line:
            body = response.split(b"\r\n\r\n", 1)[1]
            raise BlazarError(
                0, f"realtime upgrade refused: {status_line} {body[:300]!r}"
            )
        expected = base64.b64encode(
            hashlib.sha1(
                (key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()
            ).digest()
        ).decode()
        if expected.encode() not in response:
            raise BlazarError(0, "realtime handshake: Sec-WebSocket-Accept mismatch")

    def _send_frame(self, payload: bytes, opcode: int = 1) -> None:
        sock = self._sock
        if sock is None:
            raise BlazarError(0, "realtime session is closed")
        mask = os.urandom(4)
        header = bytearray([0x80 | opcode])  # FIN + opcode
        length = len(payload)
        if length < 126:
            header.append(0x80 | length)
        elif length < 1 << 16:
            header.append(0x80 | 126)
            header += length.to_bytes(2, "big")
        else:
            header.append(0x80 | 127)
            header += length.to_bytes(8, "big")
        header += mask
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        sock.sendall(bytes(header) + masked)

    def _recv_exact(self, count: int) -> bytes:
        while len(self._recv_buf) < count:
            sock = self._sock
            if sock is None:
                raise BlazarError(0, "realtime session is closed")
            chunk = sock.recv(65536)
            if not chunk:
                raise BlazarError(0, "realtime socket closed mid-frame")
            self._recv_buf += chunk
        out, self._recv_buf = self._recv_buf[:count], self._recv_buf[count:]
        return out

    def _recv_frame(self) -> t.Tuple[int, bytes]:
        first, second = self._recv_exact(2)
        opcode = first & 0x0F
        length = second & 0x7F
        if length == 126:
            length = int.from_bytes(self._recv_exact(2), "big")
        elif length == 127:
            length = int.from_bytes(self._recv_exact(8), "big")
        if second & 0x80:  # server frames arrive unmasked per RFC 6455
            mask = self._recv_exact(4)
            data = bytes(
                b ^ mask[i % 4] for i, b in enumerate(self._recv_exact(length))
            )
        else:
            data = self._recv_exact(length)
        return opcode, data

    # -- protocol ------------------------------------------------------------

    def send(self, event: t.Dict[str, t.Any]) -> None:
        """Send one realtime event (e.g. input_audio_buffer.append)."""
        self._send_frame(json.dumps(event).encode())

    def events(self) -> t.Iterator[t.Dict[str, t.Any]]:
        """Yield server events until the socket closes or errors."""
        while True:
            opcode, data = self._recv_frame()
            if opcode == 0x8:  # close
                return
            if opcode == 0x9:  # ping -> pong
                self._send_frame(data, opcode=0xA)
                continue
            if opcode in (0x1, 0x2, 0x0):  # text / binary / continuation
                if data:
                    yield json.loads(data.decode("utf-8", "replace"))

    def append_audio(self, pcm: bytes) -> None:
        """Queue PCM16 mono 24 kHz audio for the next utterance."""
        self.send(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm).decode(),
            }
        )

    def commit(self) -> None:
        """End the utterance: STT -> chat -> TTS runs server-side."""
        self.send({"type": "input_audio_buffer.commit"})

    def voice_turn(self, pcm: bytes) -> t.Dict[str, t.Any]:
        """One utterance, start to finish.

        Returns ``{"session", "transcript", "reply", "audio"}`` where
        ``audio`` is the spoken reply as PCM16 mono 24 kHz bytes. Server
        error events raise as ``BlazarError`` carrying the teaching text.
        """
        self.append_audio(pcm)
        self.commit()
        out: t.Dict[str, t.Any] = {
            "session": None,
            "transcript": "",
            "reply": "",
            "audio": b"",
        }
        import time

        deadline = time.monotonic() + self._deadline_secs
        for event in self.events():
            etype = event.get("type", "")
            if etype == "session.created":
                out["session"] = event.get("session")
            elif etype == "input_audio_transcription.completed":
                out["transcript"] = event.get("transcript", "")
            elif etype == "response.audio_transcript.delta":
                out["reply"] += event.get("delta", "")
            elif etype == "response.audio.delta":
                out["audio"] += base64.b64decode(event.get("delta", ""))
            elif etype == "response.audio.done":
                break  # terminal: TTS chunks stream after transcript.done
            elif etype == "error":
                err = event.get("error", {})
                raise BlazarError(0, str(err.get("message") or err))
            if time.monotonic() > deadline:
                raise BlazarError(0, "realtime voice_turn: session deadline exceeded")
        return out

    def close(self) -> None:
        """Close the socket (best effort; idempotent)."""
        sock = getattr(self, "_sock", None)
        if sock is not None:
            try:
                self._send_frame(b"", opcode=0x8)
            except Exception:
                pass
            try:
                sock.close()
            except Exception:
                pass
            self._sock = None

    def __enter__(self) -> "RealtimeSession":
        return self

    def __exit__(self, *exc: t.Any) -> None:
        self.close()
