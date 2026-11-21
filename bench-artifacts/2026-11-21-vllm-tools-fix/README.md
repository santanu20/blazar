# vLLM tools lane fix - live certification

Date: 2026-11-21. Binary: tree release build (0.33.0 dev tree, vllm-0.34.0 engine).

## Defect

Campaign `2026-11-21-vllm-sglang-lanes` tools cells on the vllm lane failed:
the gateway-spawned vllm child rejected every `tools` + `tool_choice="auto"`
request (400) because `compile_vllm` never emitted the tool-call pair. vLLM
only serves tool requests when the server itself boots with
`--enable-auto-tool-choice --tool-call-parser <parser>` (argparse contract:
the boolean alone is a TypeError; the parser must be one of the 62 names
registered in this build).

## Fix

- `[vllm] tool_call_parser` knob (`VllmTuning`, pin verbatim).
- Family auto-detect when unset: `vllm_tool_parser_for(architecture)` prefix
  table (Qwen3Coder->qwen3_coder, Qwen->hermes, Llama4->llama4_json,
  Llama3->llama3_json, Mistral, DeepseekV3, Jamba, InternLM, Granite;
  unmapped families spawn WITHOUT the pair - tools keep the honest 400
  instead of a wrong parser corrupting calls).
- The pair is atomic; both flags are compiler-owned (extra_args RESERVED).
- Registry: vllm first-class lane 8 -> 10 flags; engine-coverage.md
  regenerated; `engine_coverage.py --check` green.

## Live proof (XDG-isolated daemon, port 8199, model_overrides pin = vllm)

1. Cold auto tool_choice: HTTP 200 in 32.0s (boot included) - the 400 is
   gone. The 0.5B model answered with markdown-JSON content (model-level
   format compliance, lane accepted the request).
2. Forced tool_choice=get_weather: HTTP 200 in 0.66s with a structured
   `tool_calls` array - `get_weather({"city": "Tokyo"})` - proving the
   hermes parser converts model output to the OpenAI shape.
3. Plain chat regression: HTTP 200 in 32ms.
4. Child argv (live capture): `--enable-auto-tool-choice
   --tool-call-parser hermes` - byte-exact compiler emission.
5. Teardown: SIGTERM -> port free, 0 vllm/EngineCore orphans, VRAM 16 MiB.

Artifacts: tools-auto.json, tools-forced.json, plain.json, child-argv.txt.
Caveat: tree carries a concurrent session's in-flight wave (user-authorized
'ignore collision'); the binary is the tree build.
