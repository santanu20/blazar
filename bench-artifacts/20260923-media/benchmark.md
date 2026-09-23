# Blazar benchmark matrix

- **date**: 2026-09-23 13:13:45
- **model**: `sd_xl_base_1.0.safetensors` (6616 MiB)
- **gpu**: NVIDIA GeForce RTX 4070 Laptop GPU / driver 580.178.04
- **engines**: b5130, master-890-74988b2, piper
- **harness**: bench_matrix v2 — `--engines master-890-74988b2 b5130 --providers blazar --skip-ppl --skip-greedy --skip-features --skip-conc --skip-idle --skip-ctxcurve --skip-variants --media-runs 3 --md BENCHMARK.md`
- **blazar**: `blazar 0.10.0` (sandbox daemon binary)
- all blazar-owned rows measured by `blazar 0.10.0`

## Notable findings

- 1 cell(s) aborted on ENVIRONMENT guards (GPU/RAM co-residency), not product behavior — see Failed cells.

## Feature matrix

| feature | ollama(documented) |
|---|---|
| `anthropic-api` | - |
| `ctx-override` | Y |
| `embeddings` | Y |
| `grammar-gbnf` | - |
| `json-schema` | Y |
| `kv-quant` | - |
| `lora-adapter` | Y |
| `metrics-endpoint` | - |
| `paged-attn` | - |
| `parallel-np` | Y |
| `quant-on-load` | - |
| `rerank` | - |
| `slots-sessions` | - |
| `spec-decode` | - |
| `tokenize-endpoint` | Y |
| `vision-mmproj` | Y |

## Failed cells


**environment** (box/co-residency guards — NOT blazar defects):

- `master-890-74988b2` / blazar / {'config': 'single-stream'}: blazar cell crashed: MemAvailable 9054 MiB < needed ~9294 MiB (model 6616 MiB + headroom). A co-resident blazar/ollama engine is likely holding memory: stop it for the validation window.

## Reading this report

- `direct` = raw child spawn on a probed free port (no gateway).
- `blazar` = full gateway path inside a sandboxed daemon (profile compiler, routing, auth); `child_argv` in cells.jsonl holds the resolved engine argv.
- `ollama` = HTTP-only reference against the host service, one cell.
- decode counts ALL emitted tokens (content + reasoning/thinking); `usage`/engine counters are authoritative when present (`tokens src`).
- prefill t/s (cold) = prompt tokens / first-token time on an uncached token-targeted prompt; (cached) = same prompt re-sent (child prompt-cache path). ollama prefill uses engine-side prompt_eval counters, which EXCLUDE template tokens — ollama prefill reads high relative to the 512-token lanes.
- blazar speed rows show the resolved slot/context shape (`child: np=… ctx=…`) parsed from the recorded child argv — auto-slots may differ from the direct rows' explicit np.
- decode-lane TTFT rides the child's prompt cache after run 1 (warm path); the prefill-lane cold/cached pair is the honest cache story at real prompt sizes.
- GPU/power peaks sampled at ~1.2 s cadence (max across NVIDIA GPUs); very short bursts may undersample.
- Cells append to `cells.jsonl` and resume across reruns (keyed on engine/provider/params/model/harness-version).
