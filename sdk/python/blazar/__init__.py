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
"""

from __future__ import annotations

import json
import typing as t
import urllib.error
import urllib.request

__all__ = ["BlazarError", "Client"]
__version__ = "0.1.0"


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

    def _request(self, path: str, body: t.Optional[t.Dict[str, t.Any]]) -> t.Any:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            self.base_url + path, data=data, method="POST" if data else "GET"
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
