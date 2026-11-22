# Saturation acceptance verdicts

Receipt: `bench-artifacts/2026-11-22-a4-pinned2/sat_receipt.json`

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
| single_request_overhead | pass | CI95 upper overhead <= 2% - measured -23.4% |
| matched_overhead_band | fail | CI95 lower ratio >= 0.90 at every evidenced rung - measured |
| orchestration_aggregate_gain | insufficient-evidence | no orchestration leg in receipt |
| p95_interactive_gain | insufficient-evidence | no orchestration leg in receipt |
| reliability | pass | failed requests <= 0 - measured |
| resource_efficiency | report-only | orchestration tok/s-per-GiB >= matched legs - measured |

**Overall: fail** (not-judged: orchestration_aggregate_gain, p95_interactive_gain)
