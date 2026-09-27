# Spec governor — saturation parking, live end-to-end proof

- **Date:** 2026-09-27 · **Blazar:** feature/frontier-roi (B2 governor + spec-mode threading fix) · **Engine:** b11202-cuda (llama.cpp, ubuntu-cuda-12.8-x64)
- **Model:** qwen2.5-0.5b-instruct Q4_K_M (no catalog draft pair → auto engages n-gram lookup)
- **Setup:** isolated XDG daemon (port 11499), `BLAZAR_SPEC_AUTO_MANAGE=true`, 16 concurrent workers × `num_predict` 1024, 170 s load + 45 s settle. Child argv verified via `/proc/<pid>/cmdline` (XDG-environ pid scan); metrics scraped every 10 s; SSE `/api/events` captured.

## Verdict

| Check | Result |
|---|---|
| Initial child argv | `--spec-type ngram-simple` + `--lookup-cache-dynamic …​.lcache` (speculation on) |
| `spec_governor_off` event | fired at t=122 s, `reason: "saturation"` (2-window streak) |
| Gauge at settle | `blazar_spec_governor{model="qwen2.5-0.5b-instruct",state="parked_off"} 1` |
| Reshaped child argv (mid-drain) | **dense — no spec flags** |
| Reshaped child argv (final) | **dense — no spec flags** |
| n-gram children expose `llamacpp:spec_decode_*` | yes (acceptance lane viable: 1257/22788 ≈ 5.5% accepted under mixed load, below the 0.15 floor) |

The trigger is the supervisor's per-model admission pressure (slot over-subscription), not the box-wide queue depth — scraped `blazar_queue_depth` reads 0 between 10 s samples while per-request bursts saturate; the governor still fires, by design.

## Notes

- First run of this bench (pre spec-mode fix) exposed that the parked override never reached `profile::compile` — the respawn kept `--spec-type ngram-simple`. That run also surfaced the v0.12.0 `prefix_affinity` DashMap self-deadlock at 512 conversations (fixed in 48fcd71). This receipt is from the fixed binary (45d89c1 + 7fd6875).
- Reshape path exercised in full: drain-hold → evict → respawn → session-bank restore; live streams were never killed.
- Raw scrapes + events: `governor-saturation.json`.
