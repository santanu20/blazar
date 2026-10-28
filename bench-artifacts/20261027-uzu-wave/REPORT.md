# uzu wave — 2026-10-27

**Question:** why is [trymirai/uzu](https://github.com/trymirai/uzu) fast, and what can blazar borrow?

## Why uzu is fast (analysis verdict)

uzu is an embeddable Rust inference *library* (Swift/Python/TS bindings), Metal+CPU only,
Apple-first. Its speed comes from: (1) in-process use — no daemon/HTTP hop; (2) Apple-only
specialization (Metal kernels + unified-memory zero-copy, exactly 2 backends); (3) curated
per-model quants (mirai-m:4/mirai-l:8); (4) shared chat instances. Measured: 0.8B→501 tok/s,
230M→1270 tok/s on M5 Max 128GB — good, not magic. It is not a blazar competitor (library vs
serving gateway; no HTTP server mode; no Linux) and cannot be an engine lane.

## What blazar borrowed

| Idea | Outcome |
|---|---|
| Surface measured tok/s per model in catalog UX | **F1 SHIPPED** — `blazar list` SPEED column + `speed_tps` JSON field, fed by existing bench receipts (best `tg` row; honest `-` when never benched). Live: 4 benched models show 390/415/42/47 t/s. |
| Unified local+cloud model catalog | **F2 PROVEN SHIPPED — zero code change.** OpenAI-compatible cloud providers already work as `[[remotes]]` entries: catalog merge (`owned_by: blazar-remote`, `engine: <name>`), prefix `remote:model` + bare-name fallback routing, bearer auth, capacity signals degrade gracefully for non-blazar peers. Live receipt: 4/4 probes green against an OpenAI-dialect loopback stub. |

## What blazar deliberately did not borrow

- **In-process embedding** — contradicts the serving-gateway architecture (many clients, one
  loaded model, lifetime management; uzu's use case is one app embedding one model).
- **Apple-only specialization** — blazar is cross-platform by mandate; the mlx lane already
  covers Apple silicon natively.

## Reproduce

- F1: `blazar list` (SPEED column) and `blazar list --json` (`speed_tps`) — see
  `f1-list-output.txt` / `f1-list-output.jsonl`.
- F2: boot a daemon with `f2-rig-config.toml`-style `[[remotes]]` pointing at any
  OpenAI-compatible base URL; call `/v1/models`, then chat with `remote:model` and with a
  bare remote-only name. Stub peer: `stub_openai_remote.py` (loopback).

## Files

- `receipt.json` — structured receipt (probes, verdicts, pin names)
- `f1-list-output.txt` / `.jsonl` — live F1 output on the real store
- `f2-rig-config.toml` — the rig config used for the F2 probes
- `stub_openai_remote.py` — OpenAI-dialect stub peer (same class as repo test fixtures)
