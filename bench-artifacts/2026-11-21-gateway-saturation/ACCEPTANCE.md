# Saturation acceptance verdicts

Receipt: `bench-artifacts/2026-11-21-gateway-saturation/sat_receipt.json`

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
| single_request_overhead | insufficient-evidence | too few paired rounds (rounds=2 < 4) |
| matched_overhead_band | insufficient-evidence | no rung with enough rounds (rounds=0 < 4) |
| orchestration_aggregate_gain | insufficient-evidence | no orchestration leg in receipt |
| p95_interactive_gain | insufficient-evidence | no orchestration leg in receipt |
| reliability | pass | failed requests <= 0 - measured |
| resource_efficiency | insufficient-evidence | no --vram-poll readings in receipt |

**Overall: incomplete** (not-judged: single_request_overhead, matched_overhead_band, orchestration_aggregate_gain, p95_interactive_gain, resource_efficiency)
