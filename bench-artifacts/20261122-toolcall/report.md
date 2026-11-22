# Tool-calling quality battery

2026-11-22T11:09:14+00:00 | arm gateway | blazar 0.34.3 | commit 3c26234 | engine b12606-cuda

## qwen2.5-0.5b-instruct - 13/13 passed (1 expected-failure)

- PASS | T1 single-call name+args - status=200 calls=['get_weather'] args=[{'city': 'Tokyo'}]
- PASS | T2 pick correct tool of three - status=200 picked=['calculator'] args=[{'expression': '47 * 93'}]
- PASS | T3 tool_choice=required - status=200 calls=1 attempts=1
- PASS | T4 tool_choice dict-form pin - status=200 picked=['get_weather']
- PASS | T5 parallel-call emission (measured) - calls_in_one_turn=2
- PASS | T6 tool result -> grounded answer - status=200 answer='The current temperature in Tokyo is 21 degrees Celsius, and it is light rain.[]'
- PASS | T7 strict json_schema output - status=200 budget=256 finish_reason=stop content='{\n  "city": "Oslo",\n  "temp_c": 15\n}'
- PASS | T8 streaming tool deltas - streams=[['{"city": "Berlin"}']]
- PASS | T9 nested-schema args - status=200 args={'origin': {'city': 'Tokyo', 'country': 'Japan'}, 'passengers': 3, 'class': 'economy'}
- XFAIL | T10 malformed-args history tolerance - status=500 calls=0 answer='' error={"code": 500, "message": "Failed to parse tool call arguments as JSON: [json.exception.parse_error.101] parse error at line 1, column 14: sy
  reason: upstream llama-server rejects spec-legal malformed tool-args history (json parse_error.101 -> 500); gateway repair lane pending approval
- PASS | T11 tool-error recovery - status=200 retry_args=[] answer='It looks like the user is asking about the weather in a specific city, but they '
- PASS | T12 streaming call assembly - sse_done=True calls=[X7C9lL2cnj6gW0YrJg03P07gRi0l02gN:get_weather({"city": "Osaka"})]
- PASS | L1 tools-vs-no-tools latency (p50 of 3) - with=32ms without=29ms delta=+3ms (prompt-token cost of carrying 3 tool schemas; gateway transport is the same path)

