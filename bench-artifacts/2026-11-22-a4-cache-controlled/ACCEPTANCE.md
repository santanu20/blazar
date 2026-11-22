# Saturation acceptance verdicts

Receipt: `bench-artifacts/2026-11-22-a4-cache-controlled/sat_receipt.json`

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
| single_request_overhead | insufficient-evidence | cold-start mode (lane_mode=evict-per-round): steady-state metrics require resident lanes |
| matched_overhead_band | insufficient-evidence | cold-start mode (lane_mode=evict-per-round): steady-state metrics require resident lanes |
| orchestration_aggregate_gain | insufficient-evidence | cold-start mode (lane_mode=evict-per-round): steady-state metrics require resident lanes |
| p95_interactive_gain | insufficient-evidence | cold-start mode (lane_mode=evict-per-round): steady-state metrics require resident lanes |
| reliability | pass | failed requests <= 0 - measured |
| resource_efficiency | report-only | orchestration tok/s-per-GiB >= matched legs - measured |

**Overall: incomplete** (not-judged: single_request_overhead, matched_overhead_band, orchestration_aggregate_gain, p95_interactive_gain)
