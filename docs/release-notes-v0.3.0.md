# pallama 0.3.0 — the semantic reliability layer

The gap between "the API returned 200" and "the semantic operation succeeded" is
now pallama's to close. Everything below is warn-only by default and never
touches a byte of your traffic.

## Sentinel: response-semantics observation

Every chat-family request — `/v1/chat/completions`, `/v1/completions`,
`/v1/responses` (stream + non-stream), `/api/chat`, `/api/generate` — is
observed on a bounded side-channel (no hot-path parsing, no body mutation,
parity-tested byte-for-byte):

- `ctx_truncated` / `ctx_near_limit` — finish_reason `length` and 90% prompts
  with usage counts, the serving ctx, and the fix named
- `tool_args_invalid_json` / `tool_name_unknown` — streamed tool-call fragments
  merged and validated; names checked against the request's tool list
- `schema_violation` — `json_object` must parse; `json_schema` validates
  (local-only jsonschema build — no network resolution, ever)
- `empty_response` / `reasoning_no_answer` — the silent-failure classes
- `stalled_stream` — stream alive, no chunks for `sentinel_stall_secs`
- `template_no_tools` — pre-inference: tools requested but the model's GGUF
  chat template cannot render them (`x-pallama-warnings` header before the
  first token — the root cause of plain-text-instead-of-tool-calls)

## `pallama watch`

`pallama watch` (or `GET /api/watch`, SSE) tails sentinel records LIVE: one
compact line per request as it completes, detections + retry hints
underneath — truncation, malformed tool calls, schema violations and stalls
appear on your terminal the moment they happen, not in a post-mortem. The
live stream is a bounded broadcast (slow consumers get a resync note);
history stays `why`'s job.

## `pallama why [trace]`

Every request's trace id (response header `x-pallama-trace-id`) is now
answerable: what the model returned, what was wrong with it, which knob fixes
it, and the exact retry policy per detection. `GET /api/why` for frameworks.
Records persist across daemon restarts (bounded JSONL under `run/`, rotate at
1 MiB). `pallama doctor` summarizes the last 24h offline.

## Enforce (opt-in)

`sentinel_enforce = true` (config) or `X-Pallama-Enforce: 1|0` (per request,
either direction): invalid tool args / unknown tools / schema violations
become named 422s on **non-streaming** chat requests (both APIs). Streaming
stays warn-only — bytes are already on the wire. Oversized bodies pass
through loudly.

## Also in this release

- `X-Pallama-Num-Ctx` request header: per-request context on the OpenAI path
  (the protocol has no such field) — same restart-once semantics as
  `options.num_ctx`. Closes gap-analysis row 15.
- `GgufMeta.chat_template` extracted (capability heuristics).
- `record.ms` is true request latency (decoded from the trace id).

## Numbers

220 tests (12 sentinel integration suites incl. parity-under-observation),
`clippy -D warnings` clean, zero new runtime deps beyond the gated
jsonschema (MIT, `default-features = false`).
