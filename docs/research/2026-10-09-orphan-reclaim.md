# Orphan engine-child reclaim — root cause and fix (2026-10-09)

## Symptom

A daemon restart left a live engine child on the GPU (mlx child holding
516 MiB after `systemctl stop`/`start`, observed 2026-10-09 ~20:5x IST).
The child kept its port and VRAM until manually killed.

## Root cause (three independent layers, all live-proven)

1. **The existing sweeps only ever sent SIGTERM and walked away.**
   `serve()` preflight has two: `sweep_orphaned_gpu_engines` (nvidia-smi
   tenants) and `sup.sweep_orphans()` (`run/*.pid` files). Neither waits
   nor escalates — a child that hangs in graceful shutdown survives both.
2. **The GPU sweep was blind to venv engines.** Its pid-recycling guard
   compared `readlink /proc/<pid>/exe` against the engines dir, but the
   kernel resolves the venv symlink: `exe` reads
   `/usr/bin/python3.12` while `cmdline` argv[0] keeps the invoked
   `<engines>/sglang-0.5.21/venv/bin/python`. Both main GPU lanes
   (sglang, mlx) never matched. nvidia-smi compounds this by showing the
   sglang tenant as `sglang::scheduler` (setproctitle), not a path.
3. **In-cgroup children die with the unit (KillMode=control-group), but
   session daemons' children are outside it.** A `blazar serve` run
   outside systemd dies (crash, kill, closed terminal) with no cgroup
   sweep; PDEATHSIG SIGTERM is the only signal and a slow-draining child
   outlives any observer. The observed orphan carried real (non-sandbox)
   paths — spawned by a session daemon, not the unit.

## Fix (design: env marker + escalated boot sweep)

- **Marker:** `serve()` sets `BLAZAR_DAEMON_DATA_DIR=<data dir>` before
  spawning anything. Every child and grandchild captures it at exec
  (kernel semantics: `/proc/<pid>/environ` is exec-time env), so identity
  survives setproctitle renames and subprocess trees.
- **Boot reclaim** (`blazar-runtime/src/daemon.rs`
  `reclaim_marker_orphans`): after `DaemonLock` acquire (the lock proves
  no live sibling owns these processes), scan `/proc`, TERM every
  marked pid (identity re-verified first), poll liveness for a 10 s
  grace, SIGKILL survivors (identity re-verified again). One INFO line
  per reclaim; straggler summary line. Linux-only, in parity with
  PDEATHSIG; documented no-op elsewhere.
- **Escalation shared with the legacy pidfile sweep:** `sweep_orphans()`
  now collects what it TERMed and runs the same TERM → grace → KILL
  ladder (liveness-only identity: pidfile strays carry no env marker).
- **Guard repair** (`sweep_orphaned_gpu_engines`): pid check now reads
  `cmdline` argv[0] (unresolved path) instead of `exe` — venv engines
  become visible to the legacy net, which is kept for one cycle as a
  belt-and-braces transition.

## Live validation (production binary, 2026-10-09 22:24–22:28 IST)

| Probe | Result |
|---|---|
| Marker propagation | child + scheduler grandchild environ carry `BLAZAR_DAEMON_DATA_DIR` (grandchild pid 185245 reclaimed via inherited marker) |
| Escalation ladder | TERM-immune marked decoy: TERM at 22:27:07.711 → SIGKILL at 22:27:17.727 (10.016 s grace, exact) → `preflight: reclaimed … 1 straggler(s) needed SIGKILL` |
| Realistic path | session daemon SIGKILLed with warm sglang child (6.4 GiB held): unit start → healthz in 3 s, boot sweep reclaimed child 185189 + grandchild 185245 within 2 s, GPU → 15 MiB |
| No false positives | boot with zero orphans: silent, healthz 1 s |
| No-regression | full `cargo nextest` 2083/2083, fmt + clippy clean (only pre-existing proc-macro-error2) |

Honest notes: PDEATHSIG alone does eventually drain sglang (~45 s
observed) — the sweep makes reclamation immediate and guaranteed rather
than eventual; non-Linux hosts keep PDEATHSIG parity only (documented
no-op); the legacy nvidia-smi sweep is retained one cycle with the
repaired argv[0] guard.

## Reproduce

```bash
# session-daemon orphan (the production scenario)
systemctl stop blazar
setsid nohup /usr/local/bin/blazar serve >/tmp/sd.log 2>&1 &
curl -sX POST localhost:11435/api/warm -d '{"model":"qwen3-1.7b","wait":true}'
kill -9 $(pgrep -xf "/usr/local/bin/blazar serve")
systemctl start blazar          # journal: "reclaiming orphan engine child …"
journalctl -u blazar --since "1 minute ago" | grep -aE "orphan|reclaim"
```

Artifacts: `bench-artifacts/20261009-orphan-reclaim/` (journal receipt,
session-daemon consoles, GPU-tenant probe).
