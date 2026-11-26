# vLLM suffix decoding through Blazar (E2 receipt)

Question: does Blazar expose and correctly drive vLLM's draft-free
speculative decoding (SuffixDecoding) as a first-class lane knob, and does
request-level `n > 1` pass through the OpenAI-compatible surface?

Date: 2026-11-26. Lane: vllm-0.34.0 (pip, upstream release), spawned through
the vllm-server shim. Blazar: worktree sota-roadmap-20261123, release binary
built 2026-11-26 10:13:48.

## Environment

- Host: Intel Core i7-14650HX, 16 cores, 13674 MiB RAM, Linux 7.0.0-43-generic.
- GPU: NVIDIA GeForce RTX 4070 Laptop GPU, 8188 MiB (7807 MiB reported free).
- Engine: vLLM 0.34.0 (venv under the engine dir), FLASH_ATTN attention
  backend, triton JIT kernels, CUDA graphs (PIECEWISE 57 + FULL 51 shapes on
  the probe arm). The engine ran on the host toolchain; the lane needs no
  bundled CUDA.
- Model: qwen2.5-coder-0.5b-instruct, BF16 safetensors dir, 988 MiB, 1 shard,
  16k compiled context (max_model_len 16384), no quantization config.
- Prompt: "Count from 1 to 120 in words.", 200 completion tokens,
  temperature 0 (three steady runs per arm plus one streamed run).
- Sandbox harness for the through-Blazar arms; the direct probe ran the shim
  with the same flags the ladder would emit.

## Direct probe (engine alone, through the shim)

Argv: `~/.local/share/blazar/engines/vllm-0.34.0/vllm-server ~/.local/share/blazar/models/qwen2.5-coder-0.5b-instruct.d --host 127.0.0.1 --port 41240 --served-model-name qwen2.5-coder-0.5b-instruct --spec-method suffix --spec-tokens 4 --gpu-memory-utilization 0.5 --max-model-len 8192`.

Startup: ready in 48.0 s. One 32-token completion:
HTTP 200, answer starts
"Here are the words for each number:

1. One
2. Two
3. Three...". Engine log records the speculator
engaged ("max_num_scheduled_tokens is set to 2048 based on the speculative
decoding settings") and CUDA graph capture finishing in 4 s at 0.38 GiB.
In-flight VRAM: 1076 -> 4037 MiB
(weights 1.22 GiB + graphs); MemAvailable
4520 -> 3456 MiB.

## Serving arms through Blazar

| arm | first request s | steady tok/s (3 runs) | median tok/s | TTFT ms | server RSS MiB | MemAvailable MiB |
|---|---|---|---|---|---|---|
| baseline (no spec knobs) | 44.5 | 186.7 / 190.6 / 190.5 | 190.5 | 36.2 | 1371 | 5354 -> 5116 |
| suffix (spec_method=suffix, K=4) | 60.4 | 767.9 / 884.4 / 933.5 | 884.4 | 26.8 | 757 | 5155 -> 5739 |

Baseline child argv tail: `--host 127.0.0.1 --port 44855 --served-model-name qwen2.5-coder-0.5b-instruct --max-model-len 16384 --gpu-memory-utilization 0.7 --enable-prompt-tokens-details --enable-auto-tool-choice --tool-call-parser hermes`.
Suffix child argv tail: `--served-model-name qwen2.5-coder-0.5b-instruct --max-model-len 16384 --gpu-memory-utilization 0.7 --enable-prompt-tokens-details --enable-auto-tool-choice --tool-call-parser hermes --spec-method suffix --spec-tokens 4`.

The suffix argv carries `--spec-method suffix --spec-tokens 4` exactly where
the compile ladder emits them (after the tool-call pair); the baseline argv
carries no spec flags. `/api/ps` reports the same engine row and ctx (16384)
on both arms; the suffix arm additionally records the planned-fact line
"planned spec decode: suffix with 4 speculative tokens per step (unverified)" beside the standing liveness line
("effective config unverified: vllm exposes no capacity introspection surface (liveness confirmed via /v1/models)").

## Reading the speedup honestly

Median steady decode: 190.5 -> 884.4 tok/s (4.6x), TTFT
36.2 -> 26.8 ms. This prompt is a counting
exercise whose three runs share long suffixes, so the cross-request corpus
hits SuffixDecoding's best case: later runs reuse accepted token paths from
earlier ones. Expect smaller gains on single-shot, cold-corpus workloads; the
corpus warms as traffic repeats (agent loops, retries, structured output are
the natural beneficiaries). First-request latency grows
44.5 -> 60.4 s (speculator setup plus a
lazier warmup path).

## Quality

- Temperature-0 outputs are byte-identical between the baseline and suffix
  arms (recorded: yes): speculative acceptance is verified against
  the greedy target, so engaging the speculator does not change the sampled
  tokens.
- Both arms answer the counting prompt with the same words list.

## n > 1 through the OpenAI surface

`POST /v1/chat/completions` with `n=4, temperature=0.7, seed=42`,
`max_tokens=60`: HTTP 200, 4 choices,
4 distinct texts, all finish_reason stop, usage
prompt 42 / completion
53 tokens. The gateway's byte-faithful
chat lane passes `n` to the engine verbatim.

Greedy `n` (temperature 0, `n=3` on `/v1/completions`): the engine itself
refuses with HTTP 400, `{"error": {"message": "n must be 1 when using greedy sampling, got 3.", "type": "BadRequestError", "param": null, "code": 400}}`. Blazar relays
that refusal; sampling with `n > 1` needs a non-greedy sampler.

## Limits

- Greedy sampling caps `n` at 1 (engine-side constraint, relayed honestly).
- The 4.6x figure is workload-dependent (see the reading
  section); a cold-corpus benchmark would show the floor, not the ceiling.
- Capacity introspection stays liveness-only (no /server_info-style surface
  on vLLM); the planned line is labeled unverified by design.
- One host, one GPU, one model: the arms compare lanes, not hardware.
- `spec_num_tokens` is bound below the compiled context at config time; the
  engine re-validates against max_model_len at spawn.

## Artifacts

- e2_vllm_live.json (raw arms JSON), e2_vllm_direct.log (probe log),
  e2_via_daemon_baseline.log, e2_via_daemon_suffix.log (daemon logs),
  e2_vllm_live.py (harness), this receipt pair.
