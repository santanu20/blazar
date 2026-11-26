# M1 - Blazar vs ollama 0.40.17 head-to-head receipt
Matched campaign per roadmap M1: same model (Qwen2.5-0.5B-Instruct Q4_K_M), same prompts, temperature 0, seed 7, warm cache state, plus the non-stream per-request gateway overhead lane in microseconds.
## Environment
| field | value |
|---|---|
| ollama_version | 0.40.17 |
| ollama_quant | Q4_K_M |
| blazar_bin_size | 63927088 |
| worktree_head | 517578ea6832211ee648fc763ef050486d76aa02 |
| port | 34855 |
| direct_port | 8298 |
| gpu | {'util_pct': 90, 'vram_total_mib': 8188, 'vram_used_mib': 6088} |

Matched pair: ollama `qwen2.5:0.5b` (397,821,319 bytes) vs blazar `qwen2.5-0.5b-instruct`.

## Warm ladder (stream)
| product | C | ok/err | TTFT p50 ms | TTFT p95 ms | decode tok/s p50 | aggregate tok/s |
|---|---|---|---|---|---|---|
| ollama | 1 | 32/0 | 19.5 | 76.96 | 107.1 | 103.9 |
| ollama | 2 | 32/0 | 1382.07 | 1517.42 | 91.3 | 89.5 |
| ollama | 4 | 32/0 | 2463.2 | 4719.98 | 82.1 | 99.8 |
| ollama | 8 | 32/0 | 3525.84 | 4503.37 | 181.5 | 177.6 |
| blazar | 1 | 32/0 | 14.47 | 20.36 | 95.5 | 101.8 |
| blazar | 2 | 32/0 | 29.42 | 37.54 | 87.0 | 179.2 |
| blazar | 4 | 32/0 | 1200.46 | 1843.37 | 75.7 | 162.3 |
| blazar | 8 | 32/0 | 34.27 | 23036.1 | 71.6 | 124.1 |

## Warm ladder (non-stream)
| product | C | ok/err | wall p50 ms | wall p99 ms | aggregate tok/s |
|---|---|---|---|---|---|
| ollama | 1 | 32/0 | 698.83 | 809.74 | 182.3 |
| ollama | 2 | 32/0 | 1164.54 | 1876.6 | 151.2 |
| ollama | 4 | 32/0 | 2720.91 | 3922.41 | 122.2 |
| ollama | 8 | 32/0 | 6512.91 | 8446.78 | 103.7 |
| blazar | 1 | 32/0 | 1682.75 | 2058.33 | 79.6 |
| blazar | 2 | 32/0 | 1904.72 | 2213.31 | 137.5 |
| blazar | 4 | 32/0 | 2021.27 | 2336.71 | 235.5 |
| blazar | 8 | 32/0 | 2137.33 | 2460.83 | 452.4 |

## Cold start (one-shot)
- ollama: ok=True wall_ms=4019.207
- blazar: ok=True wall_ms=3076.457

## Per-request gateway overhead (the published figure)
Blazar gateway vs matched direct child (gateway engine argv minus the auth-file flag, state dirs redirected, same binary/model/ctx): C=1, non-stream, max_tokens=16, N=200 paired requests.
```json
{
 "delta_us_p50": 765.0,
 "delta_us_p95": 16547.0,
 "delta_us_p99": 36820.0,
 "direct_wall_p50_ms": 89.854,
 "direct_wall_p95_ms": 97.987,
 "direct_wall_p99_ms": 123.895,
 "gateway_over_direct_ratio_p50": 1.02,
 "gateway_wall_p50_ms": 91.609,
 "gateway_wall_p95_ms": 95.268,
 "gateway_wall_p99_ms": 99.834,
 "pairs": 200
}
```

## Quality (temp-0, 20-prompt battery)
```json
{
 "prompts": 20,
 "exact_match": 9,
 "exact_match_pct": 45.0,
 "token_agreement_pct": 57.2
}
```

### Spot samples (first 2 prompts)
**Prompt 0:** What is the capital of France? Answer with the city name only.
- ollama: `Paris`
- blazar: `Paris`
**Prompt 1:** Count from one to ten, one number per line.
- ollama: `Sure, here are the numbers from one to ten listed one per line:  1 2 3 4 5 6 7 8 9 10`
- blazar: `1 2 3 4 5 6 7 8 9 10`

## Caveats
- Engine builds differ by product reality: ollama 0.40.17 bundles its own llama.cpp fork; blazar serves via its llama.cpp b12728-cuda lane. The overhead lane isolates the gateway hop by comparing blazar-gateway vs blazar-direct on the SAME engine build.
- ollama was treated read-only (probes + generation + request-scoped keep_alive=0 for the cold unload; no restart, no store change).
- Co-tenancy: another blazar session (GPU port 36829) was live on this host during parts of the run; per-level GPU snapshots are in warm.jsonl and any contended levels are identifiable there - no request was discarded.
- Charts: m1-charts/m1-stream-ttft_p50_ms.svg, m1-charts/m1-stream-aggregate_tps.svg, m1-charts/m1-overhead-us.svg.

## Reproduction
The campaign harness is scripts/bench_ollama_head_to_head.py (repo): it drives the repository validation harness (scripts/validate.py Sandbox/Daemon) and the matched-argv machinery of scripts/bench_saturation.py against the release binary; charts render via scripts/bench_charts.py.
```bash
python3 scripts/bench_ollama_head_to_head.py --lane pilot   # env evidence + probes
python3 scripts/bench_ollama_head_to_head.py --lane warm
python3 scripts/bench_ollama_head_to_head.py --lane cold
python3 scripts/bench_ollama_head_to_head.py --lane overhead
python3 scripts/bench_ollama_head_to_head.py --lane quality && \
  python3 scripts/bench_ollama_head_to_head.py --lane receipt
```
Raw JSONL (every request incl. failures): `<sandbox>`.
