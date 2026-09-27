# A/B regression bench — v0.12.0 baseline vs frontier-roi current — 2026-09-27

Question: did the frontier-roi implementations (ngram fallback, per-model
hints, spec governor, best-of-N, differential KV, preload/RAM-warm, bank
multi-slot, symlink guard, deadlock + spec-threading fixes) degrade serving
performance?

## Design

- Baseline binary: v0.12.0 tag (c0440ab) built in an isolated
  `git worktree` at `isolated git worktree (baseline v0.12.0 build)` (shared tree untouched).
- Current binary: `target/release/blazar` @ 6657aa0 (all waves + Fix-1/Fix-2).
- Same scratch XDG root (port 11499), same minimal config (host/port/slots=8 —
  keys valid on both binaries), same model `qwen2.5-0.5b-instruct`, temp 0,
  seed 42, num_predict 96, 1 excluded warm-up, 3 s settle grace.
- Lanes:
  - **dense** — current forced dense via `BLAZAR_SPEC_AUTO_NGRAM=false` vs
    baseline default (v0.12.0 has no fallback → dense for a no-pair model).
    Isolates unintended gateway/supervisor request-path cost.
  - **default** — both defaults; current engages the ngram fallback (intended
    feature effect).
- Load: 6 sequential repetitive (JSON-continuation) + 6 concurrent distinct
  questions over 4 workers. n = 18 baseline (3 pooled runs incl. a
  self-variance duplicate) vs 12 current for dense lanes.

## Results

Engagement verified from live child argv: baseline runs = no spec flags both
lanes; current-dense = none; current-default = `--spec-type ngram-simple` +
persisted `.lcache`.

### Dense lane (unintended-regression check) — medians

| Phase | Metric | baseline | current | delta | Mann-Whitney p |
|---|---|---|---|---|---|
| repetitive | tok/s | 356.5 | 354.3 | −0.6% | 0.703 |
| repetitive | wall s | 0.28 | 0.28 | +0.7% | 0.538 |
| concurrent | tok/s | 232.4 | 225.2 | −3.1% | 0.352 |
| concurrent | wall s | 0.38 | 0.41 | +8.9% | 0.446 |

**No statistically significant difference on any metric** (all p > 0.35;
concurrent IQRs interleave: baseline [0.31, 0.43] vs current [0.27, 0.47] —
scheduling noise on a shared `-np 8` child, not a shift).

### Default lane (intended feature effect)

| Phase | tok/s median | delta |
|---|---|---|
| repetitive | 357.6 → 722.1 | **+102%** (ngram lookup speculation) |
| concurrent | 236.7 → 233.1 | −1.5% (noise-level; consistent with the −3% non-repetitive figure in 20260927-spec-ngram-default) |

## Verdict

**No performance regression from the implementation set.** The only
default-path behavior change is the ngram fallback itself: ~+102% on
repetitive traffic, ~−1..3% on non-repetitive (accepted under the 5%
threshold when the default was decided; disable via `spec_auto_ngram=false`
or `spec = "off"`).

## Notes

- Baseline self-variance measured by running the identical baseline binary
  twice (dense + default lanes are the same code path on v0.12.0):
  repetitive ±0.8%, concurrent ±4.6% on tok/s — the concurrent lane is
  inherently noisy, hence the pooled Mann-Whitney verdict over medians.
- Driver: `scratch XDG root/bench_ab.py` (exact-argv daemon
  stop; per-cell JSON in this directory).
