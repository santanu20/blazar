# Spec-governor default-flip - live validation campaign

**Date:** 2026-10-27 | **Binary:** 0.28.0+governor-flip (release build, uncommitted tree at the session HEAD) | **Engine:** llamacpp b11785-cuda | **Model:** qwen2.5-0.5b-instruct Q4_K_M (469 MiB) | **Rig:** port 11437, XDG-sandboxed temp rig, zero mocks.

## The change

`spec_auto_manage: false -> true` (crates/blazar-core/src/config.rs).

Default deployments run `spec = "auto"` with `spec_auto_ngram = true` - a
size-blind n-gram fallback. Until now its safety net (the spec governor:
park speculation after sustained collapse, restore after a quiet streak)
shipped disarmed. The flip makes the guarded feature and the guard ship in
the same default posture.

## Why no code change to slots

The bench campaign (`20261027-e2e-perf-rebaseline`) showed the daemon pinned
at `-np 2` across the whole C=1..8 ladder and read as a C>=4 throughput gap
(0.56x-0.61x). Codegraph tracing showed that is **correct burst behavior**:
`adaptive_slots` (default-on) needs ~60 s of sustained queue pressure before
adopting; bench cells are ~30 s bursts, so it never engaged. The gap was a
measurement-window artifact, not a bug. Scenario (a) below proves the
machinery engages and closes the gap under sustained load.

## Live scenarios (all on the release binary, pure daemon-driven)

### (a) Sustained C8 x 270 s under pure defaults

Daemon log cascade (`daemon-governor-excerpt.log`):

| t | event |
|---|---|
| 06:29:42 | adaptive slots adopted 2->8 (drain + reshape, KV bank carries over) |
| 06:30:22 | **spec governor PARKED** - reason=saturation, respawn dense |
| 06:30:42 | adaptive slots adopted 8->16 (`SLOTS_ADOPT_CAP`) |

Load: **1645/1646 ok** (one 502 inside a reshape drain window = 0.06%),
TTFT p50 43.8 ms, aggregate **779.8 t/s** vs **452 t/s** stuck at np=2 in
the bench campaign -> **1.72x under sustained load**. TTFT max 65.9 s = the
one stream that waited through two respawns' drain windows.

### (b) Park -> restore full cycle

`blazar_spec_governor{model,state}` gauge observed live in `/metrics`:
`watching` at t+40 s -> `parked_off` at t+100 s. Restore fired at 06:40:22 -
exactly the 10 x 60 s quiet-window streak - and n-gram re-engaged at
06:40:43 with the adopted slot count retained (slots stay; only speculation
toggles). Trickle C1 x 720 s across the restore window: 9616/9616 ok,
TTFT p50 6.3 ms. Note: quiet windows only tick while the child is awake -
a sleeping child pauses the streak clock.

### (c) Overlay pin exemption (re-validated 07:14)

`[model_overrides."qwen2.5-0.5b-instruct"] spec = "ngram"` pins speculation:
the governor must stay inert even under the same pressure that parks an
unpinned child. Result: **980/980 ok**, TTFT p50 44.0 ms, **693.6 t/s**,
**zero governor log lines** in the run window while adaptive slots climbed
2->8->16 under identical C8 pressure. The pin wins live.

*Provenance note:* the original scenario-(c) numeric artifact was overwritten
by a stray post-teardown driver invocation; this table cites the clean re-run
on the same rig and pin config, captured before any further tooling touched
the rig.

## Gates

`cargo nextest --workspace --release` 2230/2230 | `cargo fmt --check` clean |
`cargo clippy` clean | `scripts/check_config_doc_drift.py` OK | new pin
`unit__spec_governor__default_on_healthy_never_parks` (governor suite 7/7).

## Files

`receipt.json` (structured) | `daemon-governor-excerpt.log` (25-line cascade)
| `gov-load-{a2,restore,c}.log` (per-scenario summaries) |
`gov-monitor-{a2,restore}.log` (gauge poll timelines) | `gov-load.py` (driver)
| `config-defaults.toml` / `config-pin-exempt.toml`.
