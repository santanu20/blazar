# Best-of-N fan-out — live receipt (2026-09-27)

Harness: scratch XDG daemon (`/tmp/opencode/blazar-live-val2`, port 11499, config
`slots = 8`), qwen2.5-0.5b-instruct:q4_k_m on CUDA (b11202), non-stream
`/api/chat` with an ollama-dialect JSON-schema `format`, temp 0.9, num_predict 64,
5 seeds per cell, driver `bench_bestof.py` (py-spy attached, workers healthy on
socket reads).

Child argv: `-np 8 --spec-type ngram-simple` (n-gram auto-fallback compounding
with the fan-out).

## Contract proof

| Cell | eval_count (5 runs) | header | wall (warm) |
|---|---|---|---|
| baseline | 19, 18, 19, 18, 20 | — | 0.07–0.08 s |
| best_of=2 | 38, 36, 38, 36, 39 | `asked=2 used=2 usage=prompt:122 completion:38` | 0.08–0.10 s |
| best_of=4 | 76, 72, 76, 72, 78 | `asked=4 used=4 usage=prompt:244 completion:76` | 0.10–0.13 s |

- eval_count and prompt usage scale exactly ×N: every candidate generated and
  was billed once (summation contract).
- `blazar_bestof_fanouts_total 10`, `blazar_bestof_degraded_total 0` (fresh
  daemon; all 10 knobbed requests engaged).
- First bench pass (slots unset → child `-np 1`) degraded all 10 fan-outs with
  `degraded_total 10` and no headers — the free-slot guard demonstrably refuses
  to spend capacity a slots=1 child does not have.
- Judging: schema-constrained candidates all valid → tie → candidate 0
  (deterministic ladder). Winner rode the unchanged ollama response path
  (schema-valid content, `done_reason: stop`).
- Overhead: N=4 adds ~30–50 ms wall on this 0.5 B model (candidates decode in
  parallel on separate slots; baseline warm ~0.07 s).

## Files

- `cells.jsonl` — per-request rows (wall, eval_count, header, content head)
- `summary.json` — mean wall per cell + metrics snapshot
