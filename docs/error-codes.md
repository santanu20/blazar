# Error codes

Every error the gateway itself renders on an OpenAI-shaped surface
carries a stable machine-readable `blazar_code` string alongside the
human message — unlike messages and HTTP statuses, a code never
renames, renumbers, or disappears. Codes are **additive-only**: once
shipped, `code` ↔ HTTP status ↔ meaning are frozen contract.

The live catalog is self-describing:

```
curl http://127.0.0.1:11435/api/errors
```

Envelope shape (typed example):

```json
{
  "error": {
    "message": "no such model: foo",
    "type": "blazar_error",
    "code": 404,
    "blazar_code": "MODEL_NOT_FOUND"
  }
}
```

`code` remains the numeric HTTP status (unchanged); `blazar_code` is
the stable classification. Typed choke points (supervision lifecycle,
admission refusals, auth/key middleware, host guard) emit precise
codes; every other gateway-rendered error derives a generic code
(`BAD_REQUEST`, `INTERNAL`, …) from its status, so the field is
present on every path. Errors forwarded verbatim from an engine child
are untouched (they speak the child's own dialect). The Ollama-dialect
lane (`/api/chat`, `/api/generate`) keeps its own minimal error shape.

Source of truth: `crates/blazar-gateway/src/error_codes.rs` — a unit
test pins this table against the enum, so the two cannot drift.

## Catalog (v1)

| Code | HTTP | Meaning | Remediation |
|---|---|---|---|
| `MODEL_NOT_FOUND` | 404 | no model with this name is installed | pull the model first: `blazar pull <model>` |
| `MODEL_UNSUPPORTED_ENGINE` | 400 | format/quantization not supported by the engine lane it reached | pick a sibling quant/format, or install/select another engine (`blazar engine --help`) |
| `MODEL_LOAD_TIMEOUT` | 503 | model did not become healthy within model_load_timeout | check `blazar ps`; raise model_load_timeout on slow boxes |
| `ENGINE_CIRCUIT_OPEN` | 503 | engine keeps crashing, breaker open | inspect crash causes, then `blazar ps --reset` |
| `ALL_SLOTS_BUSY` | 503 | all serving slots busy, bounded admission wait expired | retry when capacity frees, or lower concurrent load |
| `MODEL_TOO_LARGE` | 503 | cannot fit this machine even empty (admission floor) | smaller quantization or more VRAM/RAM |
| `INSUFFICIENT_MEMORY` | 507 | box lacks memory now; freeing co-residents would change it | evict co-resident models (`blazar ps`) or smaller quantization |
| `ENGINE_CRASHED` | 502 | engine child crashed while loading or serving | check daemon logs for child stderr; `blazar ps` shows restarts |
| `INTERNAL` | 500 | unexpected internal error | check daemon logs; report with the x-blazar-trace-id |
| `PROMPT_TOO_LONG` | 400 | prompt exceeds effective ctx, would be silently truncated | shorten prompt, raise ctx, or set prompt_preflight = false |
| `VISION_PROJECTOR_MISSING` | 400 | vision request against a model with no projector sidecar | `blazar mmproj <model> <mmproj.gguf path>` while stopped |
| `CAPABILITY_VERIFIED_FAILED` | 400 | verified certificate shows FAIL for a needed capability | `blazar scorecard <model>` for passing siblings, or `blazar model-doctor <model>` |
| `DIFFUSION_TEXT_REFUSAL` | 400 | diffusion component-set model on a text/embedding surface | use the images/videos surface or a text model |
| `UNAUTHORIZED` | 401 | missing or invalid API key | send Authorization: Bearer or x-api-key |
| `KEY_SCOPE_FORBIDDEN` | 403 | API key not scoped for this model | use a key whose scopes cover the model |
| `HOST_FORBIDDEN` | 403 | unrecognized Host header on a loopback bind (DNS-rebinding guard) | bind to the LAN address or set host to match |
| `RATE_LIMITED` | 429 | API key exceeded one of its budgets | retry after retry_after_secs or raise the budget |
| `BAD_REQUEST` | 400 | malformed or invalid request (generic) | fix the request body per the API docs |
| `NOT_FOUND` | 404 | resource not found (generic) | check the resource id and route |
| `PAYLOAD_TOO_LARGE` | 413 | request body too large | send a smaller body |
| `UNPROCESSABLE` | 422 | request semantically invalid for this surface | fix the request semantics |
| `BAD_GATEWAY` | 502 | invalid response from the engine child (generic) | retry; check daemon logs if persistent |
| `UNAVAILABLE` | 503 | service temporarily unavailable (generic) | retry after capacity frees; `blazar ps` |
| `GATEWAY_TIMEOUT` | 504 | gateway timeout | retry or raise the client timeout |
| `INSUFFICIENT_STORAGE` | 507 | insufficient storage or memory (generic) | free memory/storage or smaller footprint |
