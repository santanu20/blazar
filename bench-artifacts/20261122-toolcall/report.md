# Tool-calling quality battery

2026-11-22T16:46:31+00:00 | arm gateway | blazar 0.34.3 | commit e267ad7 | engine b12606-cuda

## qwen2.5-0.5b-instruct - 13/13 passed

- PASS | T1 single-call name+args - status=200 calls=['get_weather'] args=[{'city': 'Tokyo'}]
- PASS | T2 pick correct tool of three - status=200 picked=['calculator'] args=[{'expression': '47 * 93'}]
- PASS | T3 tool_choice=required - status=200 calls=1 attempts=1
- PASS | T4 tool_choice dict-form pin - status=200 picked=['get_weather']
- PASS | T5 parallel-call emission (measured) - calls_in_one_turn=2
- PASS | T6 tool result -> grounded answer - status=200 answer='The current temperature in Tokyo is 21 degrees Celsius with light rain.[]' attempts=1
- PASS | T7 strict json_schema output - status=200 budget=256 finish_reason=stop content='{\n  "city": "Oslo",\n  "temp_c": 13\n}'
- PASS | T8 streaming tool deltas - streams=[['{"city": "Berlin"}']]
- PASS | T9 nested-schema args - status=200 args={'origin': {'city': 'Tokyo', 'country': 'Japan'}, 'passengers': 3, 'class': 'economy'}
- PASS | T10 malformed-args history tolerance - status=200 calls=1 answer='' repair='repaired=1 dropped=0'
- PASS | T11 tool-error recovery - status=200 retry_args=[] answer='The weather in Springfield is currently 63 degrees Fahrenheit and 83.3% humidity'
- PASS | T12 streaming call assembly - sse_done=True calls=[Nd6tNh43gtv5NO8LQwJt6gV6sSQdbutl:get_weather({"city": "Osaka"})]
- PASS | L1 tools-vs-no-tools latency (p50 of 3) - with=30ms without=31ms delta=-1ms (prompt-token cost of carrying 3 tool schemas; gateway transport is the same path)

