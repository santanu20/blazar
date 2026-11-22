# Saturation acceptance verdicts

Receipt: `bench-artifacts/20261122-w6-orch/sat_receipt.json`

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
| single_request_overhead | pass | CI95 upper overhead <= 2% - measured -1.0% |
| matched_overhead_band | fail | CI95 lower ratio >= 0.90 at every evidenced rung - measured |
| orchestration_aggregate_gain | pass | CI95 lower gain >= +10% - measured 30.3% |
| p95_interactive_gain | pass | CI95 lower p95 improvement >= +15% - measured 735.8% |
| reliability | pass | failed requests <= 0 - measured |
| resource_efficiency | insufficient-evidence | no --vram-poll readings in receipt |

**Overall: fail** (not-judged: resource_efficiency)
