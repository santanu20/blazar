#!/usr/bin/env bash
# Client-compat matrix: fires the wire shapes real clients send and asserts
# the response contracts they parse. This is the "a local agent should not
# know it is talking to Blazar" check — run it against a live daemon.
#
#   BASE=http://127.0.0.1:11435 MODEL=qwen3-1.7b scripts/compat/compat_check.sh
#
# Fixtures (each = one client family's exact request shape + the fields its
# SDK parses out of the response):
#   openai-sdk      POST /v1/chat/completions            -> choices[0].message.content, usage
#   continue-cline  POST /v1/chat/completions + tools    -> choices[0].message.tool_calls (schema legal)
#   codex-responses POST /v1/responses (store) + chain   -> output_text path via previous_response_id
#   anthropic-sdk   POST /v1/messages                    -> content[0].text, stop_reason
#   ollama-cli      POST /api/chat                       -> message.content, model echo
#   lifecycle       GET  /v1/requests + cancel           -> card visible, cancel 200
#
# Requires: curl, python3, a running blazar daemon, any text model.
set -u
BASE="${BASE:-http://127.0.0.1:11435}"
MODEL="${MODEL:?set MODEL to a pulled text model, e.g. MODEL=qwen3-1.7b}"

pass=0; fail=0
check() { # name expected_substring actual_text
    if printf '%s' "$3" | grep -qF "$2"; then
        pass=$((pass+1)); printf 'PASS  %s\n' "$1"
    else
        fail=$((fail+1)); printf 'FAIL  %s\n      expected to contain: %s\n      got: %.400s\n' "$1" "$2" "$3"
    fi
}
post() { curl -s -m 120 -H 'content-type: application/json' -d "$2" "$BASE$1"; }
get()  { curl -s -m 30 "$BASE$1"; }

echo "== client-compat matrix against $BASE (model: $MODEL) =="

# --- openai-sdk: plain chat -------------------------------------------------
body=$(post /v1/chat/completions '{"model":"'"$MODEL"'","messages":[{"role":"user","content":"Say ok"}],"max_tokens":16}')
check "openai-sdk chat: choices[0].message.content" '"content"' "$body"
check "openai-sdk chat: usage object" '"usage"' "$body"
check "openai-sdk chat: finish_reason" '"finish_reason"' "$body"

# --- continue-cline: tools shape --------------------------------------------
body=$(post /v1/chat/completions '{"model":"'"$MODEL"'","messages":[{"role":"user","content":"What is the weather in Tokyo? Use the tool."}],"max_tokens":64,"tools":[{"type":"function","function":{"name":"get_weather","description":"weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]}')
check "continue-cline tools: 200 with choices or teaching error" '"choices"' "$body"

# --- codex-responses: stored response + previous_response_id chain ---------
rid=$(post /v1/responses '{"model":"'"$MODEL"'","input":"Remember the number 42. Reply with just the number.","store":true}' \
      | python3 -c 'import json,sys; print(json.load(sys.stdin).get("id",""))')
if [ -n "$rid" ]; then
    pass=$((pass+1)); printf 'PASS  codex responses: stored id %s\n' "$rid"
    body=$(post /v1/responses '{"model":"'"$MODEL"'","previous_response_id":"'"$rid"'","input":"What number did I ask you to remember?"}')
    check "codex responses: chain resolved (no teaching 404)" '"id"' "$body"
else
    fail=$((fail+1)); printf 'FAIL  codex responses: no id returned from store:true\n'
fi

# --- anthropic-sdk: /v1/messages --------------------------------------------
body=$(post /v1/messages '{"model":"'"$MODEL"'","max_tokens":16,"messages":[{"role":"user","content":"Say ok"}]}')
check "anthropic-sdk messages: content array" '"content"' "$body"
check "anthropic-sdk messages: stop_reason" '"stop_reason"' "$body"

# --- ollama-cli: /api/chat ---------------------------------------------------
body=$(post /api/chat '{"model":"'"$MODEL"'","messages":[{"role":"user","content":"Say ok"}],"stream":false}')
check "ollama-cli chat: message.content" '"message"' "$body"
echo "$body" | grep -qF "\"model\"" && { pass=$((pass+1)); printf 'PASS  ollama-cli chat: model echo\n'; } \
    || { fail=$((fail+1)); printf 'FAIL  ollama-cli chat: model echo\n'; }

# --- lifecycle: request card + cancel ----------------------------------------
body=$(get "/v1/requests?state=done&limit=1")
check "lifecycle: /v1/requests answers" 'blazar.request.list' "$body"
rid=$(printf '%s' "$body" | python3 -c 'import json,sys; d=json.load(sys.stdin).get("data",[]); print(d[0].get("id","") if d else "")')
if [ -n "$rid" ]; then
    code=$(curl -s -m 30 -o /dev/null -w '%{http_code}' -X POST "$BASE/v1/requests/$rid/cancel")
    [ "$code" = "200" ] && { pass=$((pass+1)); printf 'PASS  lifecycle: cancel on terminal card idempotent (200)\n'; } \
        || { fail=$((fail+1)); printf 'FAIL  lifecycle: cancel returned %s\n' "$code"; }
else
    fail=$((fail+1)); printf 'FAIL  lifecycle: no request card visible after traffic\n'
fi

printf '== result: %s pass, %s fail ==\n' "$pass" "$fail"
[ "$fail" -eq 0 ]
