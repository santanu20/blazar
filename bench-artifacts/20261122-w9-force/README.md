# Force-length saturation rerun (w9)

Rerun of the w6 campaign with the committed `--force-length` harness fix
(EOS suppressed via logit_bias on ids 151645,151643, verified from the
GGUF KV header: `tokenizer.ggml.eos_token_id = 151645`). Same model,
rungs, rounds, daemon build, and acceptance gates as w6.

## What the fix changed

- `gen_tokens` is uniformly 4096 (16 streams x 256 forced) on every leg:
  the bimodal [13|256] EOS noise that failed w6's matched band is gone.
- Direct t/s is stable across rounds (846-1169 at C=16) - the fixed-length
  wall works as designed.

## Verdicts

| gate | verdict | evidence |
|---|---|---|
| single_request_overhead | pass | -1.0% (unchanged from w6) |
| p95_interactive_gain | pass | orchestration p95 TTFT still multiple-x under load |
| matched_overhead_band | fail | worst CI95-lower 0.7915 at rung 8 (target >= 0.90) |
| orchestration_aggregate_gain | fail | healthy rounds 1529-1862 t/s vs direct ~1000 (1.5-1.8x) but rounds 5-8 errored (see below) |
| reliability | fail | 48 request errors, all in orchestration C=16 rounds 5-8 |

## Root causes (from daemon.log + receipt)

1. **matched band residual is a real mid-rung fixed cost, amplified by
   environment noise.** C=2 gateway is faster (1.043); C=4/C=8 show a
   consistent -7 to -8%; C=1 per-round t/s varies 282-1346 (sd ~300) on
   BOTH legs identically - host/thermal noise the CI absorbs. On a 0.5b
   at ~1000 t/s each request is ~4 ms of child time, so the ~80 t/s gap
   is a sub-millisecond per-request gateway cost. Follow-up: mid-rung
   admission/translation micro-optimization, not a harness issue.

2. **Child crash under peak saturation, then a bounded dark window.**
   At 11:54:34 (orchestration C=16 round 5, ~8 s into the round) all 16
   streams cut simultaneously ("upstream body failed: error decoding
   response body") - the llama-server child died mid-generation. Every
   subsequent request in rounds 5-8 hit the crash-window retry ladder
   (reap + respawn + retry-once) and failed within 6-13 ms per request,
   returning 502 until the lane recovered after the leg ended
   (post-campaign chat 200 on a respawned child, VRAM normal). The
   fast-fail shape is consistent with the ensure_key circuit breaker
   bounding crash-looping while 16 concurrent reapers race - the
   per-request retry ladder cannot rescue a whole saturation leg.
   Follow-up (separate change): saturation-shaped crash recovery
   (cooldown-aware re-admission or leg-level lane warm-up).

The child crash trigger itself (llama-b12606 under 16x forced-length
decode with 2-token logit_bias) has no captured child stderr in this
sandbox configuration - flagging, not claiming.

## Verdict on --force-length itself

The harness fix is correct and necessary: it converts the matched band
from "fails on measurement artifact" (w6) to "fails on real, localized,
actionable cost" (this run). Keep it on for future qualification runs.
