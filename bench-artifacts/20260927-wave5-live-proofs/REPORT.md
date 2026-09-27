# Wave-5 live proofs — A2 cost telemetry, A3 cascade, D1 ubatch governor (2026-09-27)

Branch `feature/frontier-roi` @ 61b76fa · binary v0.12.0 (`/usr/local/bin/blazar`)
Scratch env: `/tmp/opencode/blazar-live-val2` (XDG-isolated, port 11499, slots=8, engine b11202-cuda copy)
Model: `qwen2.5-0.5b-instruct` (0.5B) + `qwen3.5-9b` (cascade escalation target)

## A2 — per-model decode-rate hints + route cost (commit 4e9a669)

Two sequential chats → `blazar_model_decode_tokens_per_second{model=…}` present and
EWMA-blended across completions (229.1 → 252.8, then 245.9 → 261.1 across reruns;
eval_count 96 each). `route_cost` pure-fn taxonomy unit-pinned (queue wait, starvation,
cache-miss prefill, decode, neutral-rate floor).

## A3 — agentic cascade router (commit ffff340)

| Case | Header (`x-blazar-cascade`) | Verdict |
|---|---|---|
| first-pass | `tried=qwen2.5-0.5b-instruct,qwen3.5-9b served=qwen2.5-0.5b-instruct reason=first-pass` | 200, small model served, chain billed correctly |
| escalation | `tried=no-such-model-x,qwen3.5-9b served=qwen3.5-9b reason=escalated` | 200, 404-candidate judged fail → 9b served with a stop-finish answer |

Counters: `blazar_cascade_runs_total` present, `blazar_cascade_escalations_total` 1.
(An earlier run showed `reason=none-passed` when the driver's 48-token budget forced
`done_reason=length` on the 9b — the judge refusing a truncated answer is by design;
budget raised to 160 tokens fixed the driver, not the product.)

## D1 — adaptive ubatch governor (commit 61b76fa)

Config `ubatch_auto = true`, slots=8, 16 concurrent workers × num_predict 1024
(the B2-proven saturation shape):

- 09:11:31 WARN `ubatch governor: saturation streak — reshaping with a larger prefill micro-batch ceiling` → tier 1024
- 09:11:41 INFO `adaptive slots: holding admissions to drain live streams for the reshape`
- 09:13:31 WARN again → tier **2048** (rung-by-rung escalation, exactly as designed)
- Gauge `blazar_ubatch_governor{model=…,tier="2048"} 1`

The respawned child's `--ubatch-size` argv flip requires the drain to complete
(never kills live streams by design); flag emission is unit-pinned
(`unit__ubatch_override__emitted_only_when_set` + injection pins).

Driver post-mortem (3 silent-failure modes, product untouched): (1) 6 workers < 8
slots never saturated; (2) `chat()` has no `timeout` kwarg — worker `TypeError`
swallowed by `except: pass` spun 16 threads with zero requests (two runs lost);
(3) `D1_queue_nonzero_seen` checked gauge-name presence, not value. wave5d driver is
clean-room phase-2-only with raw urllib and a value-capturing queue probe.

## A/B dense lane — no regression from the full wave-5 binary

v0.12.0 baseline (isolated worktree build) vs current binary, spec lane off
(`spec_flags: []` both sides), 6 repetitive + 6 concurrent cells:

| Phase | tok/s | wall |
|---|---|---|
| repetitive | 350.1 → 350.8 (**+0.2%**) | −0.5% |
| concurrent | 235.8 → 230.7 (**−2.2%**) | +8.7% |

Concurrent deltas match the pre-wave-5 sweep (−3.1%, p>0.35) and sit inside the
measured baseline self-variance (±4.6%) — same machine-noise band, not a wave-5
introduction. Receipts: `ab-baseline-r2.json`, `ab-current-r2.json`.
