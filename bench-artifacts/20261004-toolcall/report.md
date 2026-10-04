# Tool-calling quality battery

2026-10-04T17:48:27+00:00 · blazar 0.20.0 · commit e5a53d5 · engine None

## qwen3.5-9b — 9/9 passed

- PASS · T1 single-call name+args — status=200 calls=['get_weather'] args=[{'city': 'Tokyo'}]
- PASS · T2 pick correct tool of three — status=200 picked=['calculator'] args=[{'expression': '47 * 93'}]
- PASS · T3 tool_choice=required — status=200 calls=1 attempts=1
- PASS · T4 tool_choice dict-form pin — status=200 picked=['get_weather']
- PASS · T5 parallel-call emission (measured) — calls_in_one_turn=2
- PASS · T6 tool result -> grounded answer — status=200 answer='The current weather in Tokyo is 21°C with light rain.[]'
- PASS · T7 strict json_schema output — status=200 content='{\n  "city": "Oslo",\n  "temp_c": 8\n}'
- PASS · T8 streaming tool deltas — streams=[['{"city":"Berlin"}']]
- PASS · L1 tools-vs-no-tools latency (p50 of 3) — with=223ms without=100ms delta=+123ms (prompt-token cost of carrying 3 tool schemas; gateway transport is the same path)

