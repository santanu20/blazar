# Saturation acceptance verdicts

Receipt: `bench-artifacts/2026-11-22-a4-resident/sat_receipt.json`

## Targets

| metric | target |
|---|---|
| single_request_overhead_pct | 2.0 |
| matched_overhead_band_pct | 10.0 |
| orchestration_aggregate_gain_pct | 10.0 |
| p95_interactive_gain_pct | 15.0 |
| reliability_max_failed_requests | 0 |

## Verdicts

| metric | verdict | detail |
|---|---|---|
| single_request_overhead | pass | CI95 upper overhead <= 2% - measured -45.5% |
| matched_overhead_band | fail | CI95 lower ratio >= 0.90 at every evidenced rung - measured |
| orchestration_aggregate_gain | pass | CI95 lower gain >= +10% - measured 41.6% |
| p95_interactive_gain | pass | CI95 lower p95 improvement >= +15% - measured 220.0% |
| reliability | pass | failed requests <= 0 - measured |
| resource_efficiency | report-only | orchestration tok/s-per-GiB >= matched legs - measured |

**Overall: fail** (not-judged: none)
