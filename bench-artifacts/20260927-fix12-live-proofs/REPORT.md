# Fix-1 + Fix-2 live proofs — 2026-09-27

Branch `feature/frontier-roi` @ 86dbfa4 · binary v0.12.0 (release, `/usr/local/bin/blazar`)
Scratch env: `/tmp/opencode/blazar-live-val2` (XDG-isolated, port 11499, engine b11202-cuda copy, no symlinks)
Model: `qwen2.5-0.5b-instruct` (flat name, 469 MiB)

## Fix-1 — best-of-N degrade is client-visible (commit b04927a)

Config `slots = 1` (forces headroom 0), `/api/chat` non-stream with `best_of = 2`.

| Check | Result |
|---|---|
| HTTP status | 200 (normal body, no fan-out) |
| Response header | `x-blazar-best-of: asked=2 used=1 reason=headroom` |
| `blazar_bestof_degraded_total` | 1 |

Evidence: `fix12-live.json` fields `F1_*`.

## Fix-2 — multi-slot session bank (commit 86dbfa4)

### Slot occupancy recon (direct child API, `-np 8` child)

6 PARALLEL distinct 24-token conversations occupy slots 0, 1, 2, 5, 6, 7
(60 prompt tokens each). Sequential chats always land slot 0 (LRU) — parallel
conversations are required to occupy multiple slots.

### Path 1: gateway eviction (`keep_alive: 0` unload ping)

Bank swept exactly the 6 occupied slots — daemon debug lines
`banked _auto-2048[-sN] checkpoint: 60 tokens` per slot; empty slots' 0-token
save files dropped (llama-server writes a 36 B file even when `n_saved = 0`;
the sweep deletes them).

### Path 2: daemon SIGTERM (exact-argv pid match, harness-fixed)

Directory listing after shutdown (see `sigterm-bank-listing.txt`):

```
_auto-2048            _auto-2048-s5          + .identity.json per file
_auto-2048-s1         _auto-2048-s6
_auto-2048-s2         _auto-2048-s7
```

All 6 occupied slots banked; empty slots (3, 4) absent; identity manifests
written for every surviving file.

### Restore

Restart → spawn → `restored banked session _auto-16384 (372 tokens into slot
KV)` (earlier run, same binary; wording unchanged for log-greppers).
Restore iterates slot-0 file first, then `-s<id>` files sorted, posting
`/slots/{id}?action=restore` per file.

## Methodology notes

- Early SIGTERM runs banked nothing — harness pid matcher (cmdline substring
  `blazar`+`serve`) also matched `llama-server` children and killed the child
  mid-bank. Fixed with exact-argv matching (`argv[0]` endswith `/blazar`,
  `argv[1] == serve`). Product code was correct throughout.
- The pre-fix prod observation (`sessions/<model>/` empty under auto slots)
  was a real slot-0-only limitation; the multi-slot sweep removes it.
- ctx in the bank filename varies with auto-sizing per config
  (`_auto-16384` vs `_auto-2048`) — never hardcode.
