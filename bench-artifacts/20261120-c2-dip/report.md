# Gateway-overhead C2-dip rerun - 2026-11-20

## Question

The 2026-10-31 after-fix gateway-overhead ladder
(`docs/research/2026-10-31-recursive-self-improvement-research.md`, fix
`299bf5a`) measured gateway/direct tok/s ratios of C1 1.135, **C2 0.792**,
C4 1.017, C8 0.986. The C2 dip (below parity while C1/C4 sit above it) was
unexplained; the standing suspect was `-np 2` slot/concurrency interplay at
the 2-slot boundary. This rerun asks one question: does the C2 dip reproduce
under a cleaner methodology?

## Methodology (improvement over S7)

- Sequential legs, evict between: the S7 caveat was a same-box dual-child
  setup where both engine children held the GPU simultaneously - each leg's
  numbers carried cross-leg contention. Here the gateway leg ran alone, the
  model was evicted through the gateway (`/api/evict`), and the direct child
  then ran alone. Ambient contention from other sessions' processes was
  sampled (nvidia-smi before/after each leg; the box was quiet - no other
  compute apps during either leg).
- Same engine binary (fork llama.cpp a7f77a78-cuda via `llama-server`), same
  model file (Qwen3-0.6B-Q4_0.gguf), same adopted profile shape:
  `--flash-attn auto --ctx-size 32768 --threads 16 --gpu-layers 999 -np 2
  --embeddings --pooling last --spec-type ngram-simple --cache-ram 4102
  --kv-unified --offline --load-mode mlock --lookup-cache-dynamic <same
  lcache file>`. Full gateway-composed argv captured in `child-argv.txt`;
  the direct leg drops only gateway plumbing (`--api-key-file`,
  `--slot-save-path`, `--props`, `--no-webui`) and pins `--port`.
- Same ladder as S7: 2 warm-ups, then per C in {1,2,4,8}: 3 reps x C
  concurrent non-stream `/v1/chat/completions`, temp 0, max_tokens 64,
  identical fixed body; min-wall of the 3 reps -> tok/s = C*64/min_wall.
  3 reps is coarse; the question is a 20% dip, not a 2% effect.
- Gateway leg: worktree `target/release/blazar serve` (blazar 0.33.0
  gateway, XDG-sandboxed store copy, engine dir symlinked to the real one).

## Results (min-wall tok/s, ratio = gateway/direct)

| C | gateway | direct | ratio | S7 after-fix ratio |
|---|--------:|-------:|------:|-------------------:|
| 1 | 344.3 | 350.5 | 0.982 | 1.135 |
| 2 | 551.3 | 585.4 | **0.942** | **0.792** |
| 4 | 570.0 | 582.6 | 0.978 | 1.017 |
| 8 | 568.5 | 633.3 | 0.898 | 0.986 |

Raw: `gateway.log`, `direct.log`, `ladder.json`, ambient samples in this
directory.

## Finding

**The C2 dip does not reproduce.** The sequential-leg ladder is flat
(0.90-0.98 across C): no C2-specific anomaly, no C1 overshoot. The C2 ratio
(0.942) now sits *above* C8 (0.898). Two readings:

1. The original C2 0.792 - and the C1 1.135 overshoot - are consistent with
   same-box dual-child contention noise in the S7 methodology (whichever
   child faced a busier GPU during its rep window lost; min-of-3 over
   0.2-0.9 s walls amplifies this), not with a gateway defect at the
   2-slot boundary. This rerun cannot prove a negative about the original
   run's box state; it removes the dip under the cleaner methodology.
2. What remains is a small, monotonic ratio decline with C (0.98 -> 0.90),
   i.e. the ordinary cost of queueing/dispatch across 8 concurrent relayed
   streams at saturation (~570 tok/s gateway ceiling vs ~633 direct on this
   engine/argv). That is a different, expected shape - not the C2 anomaly.

## Conclusion

Gateway-overhead workstream stays closed as scoped by the research note: no
C2-specific defect to chase. If anyone wants to shave the residual ~10% at
C8 saturation, that is regular saturation-path profiling (queue/dispatch,
not slot-boundary), best done on a quiet box with more reps.

## Reproduce

```
# gateway leg: XDG-sandboxed store copy (fork engine + qwen3-0.6b), then
BLAZAR_PORT=11531 target/release/blazar serve   # warm, then:
python3 ladder.py gateway                        # this dir's harness shape
# evict, then direct leg mirroring child-argv.txt minus gateway plumbing
python3 ladder.py direct
```

- Hardware: RTX 4070 Laptop 8 GiB; i7-14650HX; Linux Mint; driver 595.99.02.
- Blazar: worktree release build at follow-up-1 (fa6acc3 lineage), engine
  fork llama.cpp a7f77a78-cuda (llama-server reports 0.34.4).
- Power: AC. Box ambient quiet during both legs (see ambient-*.txt).
