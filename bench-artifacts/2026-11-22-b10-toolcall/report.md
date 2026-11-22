# Tool-calling quality battery

2026-11-22T10:01:23+00:00 | arm gateway | blazar 0.34.3 | commit e93e852 | engine b12606-cuda

## qwen2.5-0.5b-instruct - 12/13 passed

- PASS | T1 single-call name+args - status=200 calls=['get_weather'] args=[{'city': 'Tokyo'}]
- PASS | T2 pick correct tool of three - status=200 picked=['calculator'] args=[{'expression': '47 * 93'}]
- PASS | T3 tool_choice=required - status=200 calls=1 attempts=1
- PASS | T4 tool_choice dict-form pin - status=200 picked=['get_weather']
- PASS | T5 parallel-call emission (measured) - calls_in_one_turn=2
- PASS | T6 tool result -> grounded answer - status=200 answer='The current temperature in Tokyo is 21°C with light rain.[]'
- PASS | T7 strict json_schema output - status=200 budget=256 finish_reason=stop content='{\n  "city": "Oslo",\n  "temp_c": 14\n}'
- PASS | T8 streaming tool deltas - streams=[['{"city": "Berlin"}']]
- PASS | T9 nested-schema args - status=200 args={'origin': {'city': 'Tokyo', 'country': 'Japan'}, 'passengers': 3, 'class': 'economy'}
- FAIL | T10 malformed-args history tolerance - status=500 calls=0 answer='' error={"code": 500, "message": "Failed to parse tool call arguments as JSON: [json.exception.parse_error.101] parse error at line 1, column 14: sy
- PASS | T11 tool-error recovery - status=200 retry_args=[] answer="It seems there's an inconsistency in the input. Could you please specify a count"
- PASS | T12 streaming call assembly - sse_done=True calls=[BdFRxg8uDSKyMrBMBB14avgjzeqN6lJv:get_weather({"city": "Osaka"})]
- PASS | L1 tools-vs-no-tools latency (p50 of 3) - with=31ms without=30ms delta=+2ms (prompt-token cost of carrying 3 tool schemas; gateway transport is the same path)

## qwen2.5-0.5b-instruct-4bit - 6/13 passed

- FAIL | T1 single-call name+args - status=200 calls=[] args=[]
- FAIL | T2 pick correct tool of three - status=200 picked=[] args=[]
- FAIL | T3 tool_choice=required - status=400 calls=0 attempts=2
- FAIL | T4 tool_choice dict-form pin - status=400 picked=[]
- PASS | T5 parallel-call emission (measured) - calls_in_one_turn=0
- PASS | T6 tool result -> grounded answer - status=200 answer='The current temperature in Tokyo is 21 degrees Celsius with a light rain condition.[]'
- PASS | T7 strict json_schema output - status=200 budget=256 finish_reason=stop content='{\n  "city": "Oslo",\n  "temp_c": 12\n}'
- FAIL | T8 streaming tool deltas - streams=[[]]
- FAIL | T9 nested-schema args - status=200 calls=[]
- PASS | T10 malformed-args history tolerance - status=200 calls=0 answer='Sure, please try again.'
- PASS | T11 tool-error recovery - status=200 retry_args=[] answer='Could you please provide me with the name of the city?'
- FAIL | T12 streaming call assembly - sse_done=True calls=[]
- PASS | L1 tools-vs-no-tools latency (p50 of 3) - with=46ms without=42ms delta=+4ms (prompt-token cost of carrying 3 tool schemas; gateway transport is the same path)

