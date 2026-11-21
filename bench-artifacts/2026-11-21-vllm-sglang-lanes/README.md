# vLLM + SGLang text-lane completion campaign

First `bench_matrix` campaign covering the two venv-served engine lanes that the
2026-11-20 wave-final render lacked: `vllm-0.34.0` (gateway-only venv lane) and
`sglang-0.5.23`, on `qwen2.5-coder-0.5b-instruct` (HF safetensors, 953 MiB).
Closes the benchmark-lane gap tracked in docs/issues/003-engine-benchmark-lane-completion.md
for these two lanes on this hardware.

- date: 2026-11-21
- hardware: RTX 4070 Laptop 8 GB, driver 595.99.02
- sandbox daemon binary: tree `blazar 0.34.0` release build
- invocation: see `campaign_cmd.txt`; artifacts dir `20261121-111035`
- 48 cells (deduped), 35 ok / 3 error / rest reference+special lanes; full
  provenance per cell in `cells.jsonl` (child argv, blazar version, power state)

## Headline results

- sglang lane: 202 t/s single-stream decode, 14.8k t/s prefill, TTFT ~18 ms,
  VRAM ~2.0 GiB (default profile)
- vllm lane: ~190 t/s decode, ~14.0k t/s prefill, TTFT ~19 ms, VRAM 6.2 GiB
  (0.85 GPU-mem util default)
- ollama reference: 338 t/s decode (small-model GGUF lane, different runtime
  class - not comparable to the safetensors venv lanes)
- quality lane ran (code/instruct/multilingual/niah/reason/safety/schema suites)
- QC drift gate flagged 4 sglang config A/Bs whose greedy output differs from
  default: `deterministic_on`, `page_64`, `torch_compile_on`, `kv_dtype_e4m3`.
  Expected physics (kernel/paging/fp8-KV changes numerics); flagged, not
  failures. See benchmark.md "qc drift gate" section.

## Error cells (documented)

- `vllm tools` - REAL capability gap: gateway-spawned vllm rejects `"auto"`
  tool choice (`--enable-auto-tool-choice` + `--tool-call-parser` not emitted
  by the compile path). Follow-up filed in MEMORY; tools on the vllm lane
  should either gain first-class compile support or an explicit
  not-certified disposition.
- `sglang quality`, `vllm quality` - `mem_guard: GPU too busy for quality
  lane` environment guards during a contaminated window (see caveats), not
  product behavior; the quality lane scored where it ran.

## Caveats (read before citing numbers)

- TREE-STATE: the working tree carries a concurrent in-flight session's
  uncommitted wave (supervisor/gateway/bench_matrix changes) plus this
  session's lanes; the sandbox binary is that combined tree. User-authorized
  ("ignore collision, do all").
- The harness was terminated twice mid-campaign by an external SIGKILL (empty
  stderr, no kernel OOM record, processes setsid-detached; suspected external
  sweep - concurrent-session activity). cells.jsonl dedup made the campaign
  resumable; one duplicate key from a brief double-launch was removed before
  rendering (33 -> 32 unique, final pass added 16 more).
- venv lanes are gateway-routed only (`server=None`): no direct-leg cells, so
  gateway-overhead charts are skipped by design for this campaign; the
  saturation A/B lives in 2026-11-21-gateway-saturation instead.
- Single box, single model class: numbers characterize this hardware/model
  pair, not a general claim.
