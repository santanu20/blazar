# blazar-python

Zero-dependency Python client for [Blazar](https://github.com/santanu20/blazar),
the local model server. Standard library only.

```bash
pip install .        # from a checkout of this repository
```

Or drop `blazar/__init__.py` on `PYTHONPATH` — there is nothing to build.

## Usage

```python
from blazar import Client

blazar = Client()  # BLAZAR_URL env or http://127.0.0.1:11434; Client("http://host:port") wins

reply = blazar.chat("qwen3-0.6b:q4_0", [{"role": "user", "content": "hi"}])
print(reply["choices"][0]["message"]["content"])

# streaming
for chunk in blazar.chat("qwen3-0.6b:q4_0",
                         [{"role": "user", "content": "count to five"}],
                         stream=True):
    print(chunk["choices"][0]["delta"].get("content", ""), end="", flush=True)

# tools (gateway-side MCP loop, buffered)
tools = [{"type": "function", "function": {"name": "get_weather",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]
out = blazar.chat("qwen3-0.6b:q4_0", [{"role": "user", "content": "weather in Tokyo?"}],
                  tools=tools, mcp="all")

# operations
blazar.ps()               # residents, posture, sleep state, remotes
blazar.explain("qwen3-0.6b:q4_0")   # routing decision card
blazar.failover()         # chain state
blazar.models()
```

Errors raise `blazar.BlazarError` carrying the HTTP status and the
gateway's teaching message (`error.message`) — surface it verbatim;
Blazar errors explain the fix.

## API

| Method | Endpoint | Notes |
|---|---|---|
| `chat(model, messages, *, stream, tools, best_of, mcp, **kw)` | `POST /v1/chat/completions` | `mcp` requires `stream=False` |
| `ollama_chat(model, messages, **kw)` | `POST /api/chat` | ollama dialect |
| `models()` | `GET /v1/models` | |
| `ps()` | `GET /api/ps` | |
| `explain(model)` | `GET /api/explain/{model}` | |
| `failover()` | `GET /api/failover` | |
| `mcp_servers()` | `GET /api/mcp` | |

Any other chat-completions field (temperature, max_tokens, response_format,
…) rides through `**kw` verbatim.
