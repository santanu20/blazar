# Gemini generateContent dialect lane (E4) - live receipt

Date: 2026-11-26
Question: can a Gemini-SDK client (contents/parts, `functionCall` round
trips, `data:` SSE frames) drive the local engine through Blazar with the
full spine (admission, budgets, preflight, failover guards) behind a flag?

## Environment

- Blazar release binary: 64193352 B (sandbox daemon, same build as the chain receipts)
- Model: qwen2.5-0.5b-instruct (GGUF, Q4_K_M) on engine b12728-cuda
- Sandbox: hardlinked engines + live DB copy; mem floor override 2048 MiB (mechanism run)
- Route: POST /v1beta/models/{model}:generateContent and :streamGenerateContent
  (mounted as a single catch-all segment - the router's path matcher allows one
  parameter per segment, so `{model}:{verb}` cannot be expressed directly;
  the handler splits on the last colon, which also keeps tag-style model names
  parseable)

## Non-stream generateContent

POST with systemInstruction "Be terse." and temperature 0, maxOutputTokens 64:
HTTP 200, role model, finishReason STOP,
answer "One to five is one, two, three, four, five.", usageMetadata prompt 26 /
candidates 15 / total 41.
The terse system instruction visibly shapes the answer.

## streamGenerateContent

HTTP 200: 35 `data: {...}` frames and
0 `event:` lines (Gemini streams carry bare data frames,
not named events), finishReason and usageMetadata present in the tail. First
frame: data: {"candidates":[{"content":{"parts":[{"text":"Sure"}],"role":"model"},"index":0}],"modelVersion

## Function-call round trip

Arm 1 (functionDeclarations get_weather + "What is the weather in Paris?
Use the tool."): HTTP 200, parts = ['functionCall'], the
model answers a single functionCall {"args": {"city": "Paris"}, "name": "get_weather"}.
Arm 2 (functionResponse {temp: 12} appended as the next user turn):
HTTP 200 - the translated tool-result turn is served and the model answers
from it: {"candidates":[{"content":{"parts":[{"text":"The current temperature in Paris is 12\u00b0C."}],"role":"model"}

## Refusals and flag

- candidateCount 2: HTTP 400 {"error":{"code":400,"message":"candidateCount > 1 is not served on the Gemini translate lane - it would silently drop candidates; call n times, or use the OpenAI surface with n","
- :countTokens (unknown verb): HTTP 404 {"error":{"code":404,"message":"unknown verb \"countTokens\" - this lane serves :generateContent and :streamGenerateContent","status":"NOT_FOUND"}}
- gemini_translate = false: HTTP 404 - the route is unmounted
  (the 404 is Blazar's plain unknown-route envelope, not the Gemini one).

## Reading

The dialect is a first-class lane, not a thin shim: requests ride the same
admission, budget-reservation, preflight, workload classification and
failover-local-only guards as the Anthropic lane. Tool calls use the
name-as-id convention (the id of an OpenAI tool call is the Gemini function
name), which round-trips: arm 2's functionResponse turn is accepted and
answered. candidateCount > 1 is refused with teaching instead of silently
dropping candidates; `generationConfig.topK` is dropped with a logged warning
(no OpenAI chat equivalent - never silent, never hostile).

## Limits

- Shapes verified against the fetched Gemini API reference; no google-genai
  SDK client ran on this host (the gateway contract tests + these live arms
  are the evidence).
- GET /v1beta/models (SDK model discovery) is out of scope for this lane.
- Function-call strength depends on the served model (the 0.5b answered a
  clean single call here; weaker checkpoints may answer in text - both are
  honest engine behavior, the lane translates either).

## Artifacts

- e4_gemini_live.json (this receipt's data)
- e4_via_daemon_on.log / e4_via_daemon_off.log (sandbox daemon logs)
- e4_clippy2.log, e4_final_nextest.log (3287/3287), e4_gates.log,
  e4_final_rel.log (release build)
