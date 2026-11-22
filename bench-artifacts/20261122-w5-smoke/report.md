# Tool-calling quality battery

2026-11-22T09:56:50+00:00 | arm gateway | blazar 0.34.3 | commit e93e852 | engine b12606-cuda

## qwen2.5-0.5b-instruct - 12/13 passed

- PASS | T1 single-call name+args - status=200 calls=['get_weather'] args=[{'city': 'Tokyo'}]
- PASS | T2 pick correct tool of three - status=200 picked=['calculator'] args=[{'expression': '47 * 93'}]
- PASS | T3 tool_choice=required - status=200 calls=1 attempts=1
- PASS | T4 tool_choice dict-form pin - status=200 picked=['get_weather']
- PASS | T5 parallel-call emission (measured) - calls_in_one_turn=2
- PASS | T6 tool result -> grounded answer - status=200 answer='The current weather in Tokyo is light rain with a temperature of 21 degrees Celsius.[]'
- PASS | T7 strict json_schema output - status=200 budget=256 finish_reason=stop content='{\n  "city": "Oslo",\n  "temp_c": 12\n}'
- PASS | T8 streaming tool deltas - streams=[['{"city": "Berlin"}']]
- PASS | T9 nested-schema args - status=200 args={'origin': {'city': 'Tokyo', 'country': 'Japan'}, 'passengers': 3, 'class': 'economy'}
- FAIL | T10 malformed-args history tolerance - status=500 calls=0 answer=''
- PASS | T11 tool-error recovery - status=200 retry_args=[] answer="I'm sorry, I don't have more information about the city you asked for. Could you"
- PASS | T12 streaming call assembly - sse_done=True calls=[VCINuaCuCemXD7S2WnZZ9WXRFNQQensb:get_weather({"city": " Osaka"})]
- PASS | L1 tools-vs-no-tools latency (p50 of 3) - with=31ms without=30ms delta=+1ms (prompt-token cost of carrying 3 tool schemas; gateway transport is the same path)

