Title: Daemon log sinks go silently dark under load (heartbeat + poison-recovery hardening)

Priority: P1 | Wave: hotfix | Backlog: - | Effort: S | Depends: -

## Problem

On 2026-11-22, during the a4-resident saturation campaign (receipt:
`bench-artifacts/2026-11-22-a4-resident/`), the daemon's three
independent observability sinks stopped emitting for roughly ten
minutes while the daemon kept serving traffic correctly:

- `run/daemon.log` (tracing file arm, `TeeWrite` in
  `crates/blazar-core/src/telemetry.rs`): zero bytes between
  08:17:50Z and 08:27:43Z, then the SIGTERM drain line wrote fine.
- `run/sentinel.jsonl` (own fd + `parking_lot::Mutex`, unbuffered
  `writeln!`): zero records for the same window.
- stdout/stderr (`daemon.out`): silent in the window.

During the window the receipt proves ~248 gateway chat requests
completed with zero errors, and the adaptive-slots supervisor adopted
np 2->8 (the orchestration leg sustained ~1300 tok/s at C=16, which
requires the reshaped child) - but the adoption INFO event is missing
from the log. The failure hid its own evidence: three sessions
independently misdiagnosed the silence as "idle daemon", "dead
daemon", and "rotated log" before byte-level forensics bounded the
window.

## Forensics (what was ruled out)

- Mutex poisoning alone: a poisoned tracing file mutex skips only the
  file arm; stderr and sentinel have separate locks and also went
  dark. The SIGTERM line wrote through the same tracing subscriber at
  window end.
- Blocked writers: a blocked/queued emitter flushes late lines after
  recovery; the window has zero late lines and requests completed
  freely throughout.
- Event-source idle: the receipt proves heavy traffic and adaptation
  during the window.
- Rotation, ENOSPC, tmpfs quirks, double-init: all excluded by
  artifact inspection (`daemon.log` was 958 KiB, disk had 34 GiB free,
  `/tmp` is ext4, `init_tracing` runs once).
- Controlled reproduction (2026-11-22, `bench-artifacts/2026-11-22-logfreeze-repro/`):
  faithful workload shape (same harness, same model, same unpinned
  config, concurrent memory pressure), watchdog on log growth +
  connection count, strace on write syscalls. Result: negative - 46
  samples at >=3 established connections (max 16), longest
  log-static-under-load stretch 8s, daemon.log +192 KiB, sentinel
  +154 KiB, adoption event logged. No freeze.

Root cause remains open. The surviving family is a runtime-level
event-emission stall with intact task execution; no mechanism has
been found in static analysis, and the one controlled repro attempt
did not trigger it.

## Hardening (this issue's deliverable)

The incident's real damage was unobservability: nobody could prove
WHEN emission stopped or that the daemon was healthy. Two changes
make any recurrence self-evidencing:

1. Reaper heartbeat (`Supervisor::spawn_reaper`,
   `crates/blazar-runtime/src/supervisor.rs`): one INFO line every
   60s (`target: "blazar::heartbeat"`, with resident-model and
   in-flight counts). A silent window now appears as a timestamp gap
   between heartbeat lines - provable, bounded, and distinguishable
   from an idle daemon (which still heartbeats).
2. Poison-robust log arm (`TeeWrite` in
   `crates/blazar-core/src/telemetry.rs`): a poisoned file mutex no
   longer permanently silences the file arm; the guard is recovered
   via `PoisonError::into_inner` (writes continue) and a one-time
   stderr marker names the moment. Previously one panic mid-write
   would have darkened the durable log forever with no signal.

## Acceptance criteria

- [x] Freeze window bounded from artifacts (not memory) and recorded
- [x] Controlled repro attempted with instrumentation; negative result
      preserved (`bench-artifacts/2026-11-22-logfreeze-repro/`)
- [x] Heartbeat line observable in a live daemon log at 60s cadence
- [x] Poisoned-mutex recovery unit-tested (`tee_write` poisoned lock test)
- [ ] Root cause identified (open; needs a recurrence with heartbeat +
      strace attached, or a new lead)

## Files

crates/blazar-core/src/telemetry.rs, crates/blazar-runtime/src/supervisor.rs
