# ADR: OpenAI-Host Lane (LiteRT-LM / Foundry Local) — Phase A interop

**Date:** 2026-11-07 · **Status:** Phase A ACCEPTED (interop proven); Phase B (full `EngineKind::OpenAiHost`) planned · **Decision session:** dp-20261107-wave3-fork-watch-tune-gate-openaihost (branch B, score 8.80)

## Context

The Atomic-Chat parity program (items 1-7, `2026-11-06-adoption-plan-items-3-7.md`) left one adoption item open: **5.2 openai-host lane** — on-device OpenAI-compatible inference hosts that are not llama.cpp forks and must not be force-fitted into the engine-build machinery. Verified vendor landscape (2026-11-07):

| Host | Serving shape | Platforms | Notes |
|---|---|---|---|
| LiteRT-LM (google-ai-edge, v0.19.x) | `litert-lm serve` — OpenAI server, default :9379, `/v1/models`, `/v1/chat/completions` (SSE), `/v1/embeddings` | Linux/macOS/Windows/Android/RPi | **Registry-wide serve** (no model pin in argv; models = aliases in `~/.litert-lm`); MTP spec decode; multimodal; pip-installable |
| Foundry Local (microsoft) | OpenAI-compat incl. Responses format | Windows + macOS (Apple silicon) + Linux | Curated catalog; CLI `foundry`; service details unverified (no Windows/mac box here) |

## Decision

**Phase split** (branch B of the planning session):

- **Phase A (this session): interop via the existing `[[remotes]]` lane, zero product scaffolding.** Prove the gateway can drive an openai-host today, capture the receipts, fix whatever real bugs surface. No `EngineKind` addition, no supervisor changes.
- **Phase B (own session + own deep_planning): `EngineKind::OpenAiHost`** with a descriptor registry (litert-lm first; foundry Windows-gated), supervised lifecycle (install/health/respawn), routing arm, and V1-V5 lifecycle tests. ~~Must first answer **shared-child vs per-model spawn**~~ **RESOLVED 2026-11-07** (session dp-20261107-openaihost-phase-b-spawn-model, score 8.05): **shared supervised child, router pattern** — see "Phase B design decision" below.

Rejected: full kind now (~15-file blast radius stacked on two uncommitted waves; the supervisor question unanswered = fragile); remotes-only forever (abandons supervised lifecycle, install lane, and routing integration).

## Phase-A evidence (all live, 2026-11-07, this machine)

Setup: `uv venv ~/.local/share/blazar/litert-lm/venv` + `litert-lm` 0.19.1; imported `litert-community/SmolLM2-360M-Instruct` (356.4 MiB, alias `smollm2-360m`); `litert-lm serve --host 127.0.0.1 --port 9379`; one `[[remotes]]` entry `name="litert"`.

| Receipt | Result |
|---|---|
| Explicit prefix route `litert:smollm2-360m` (buffered) | 200; model rewritten to `smollm2-360m` in flight; usage relayed |
| Same, `stream: true` | SSE chunks relayed byte-for-byte |
| Bare-name fallback `smollm2-360m` (local miss) | peer found via forced presence probe; forwarded; answered |
| `/v1/models` peer merge | `smollm2-360m`, `owned_by: blazar-remote`, `engine: litert` (after first fallback warmed presence — snapshot semantics, documented) |
| Direct-to-peer control (fixed content-length) | 200 — isolates the framing bug below |

Config restored to pre-test state after validation (no dead endpoints in user config). The venv (`~/.local/share/blazar/litert-lm/`) and the imported model (`~/.litert-lm/models/smollm2-360m/`) stay as the Phase-B foundation.

## Bug found and fixed: chunked request framing

**Root cause:** `remotes::forward_openai` wrapped the fully-buffered body in `reqwest::Body::wrap_stream(...)`. A streaming body has no known length, so hyper framed every forwarded request with `transfer-encoding: chunked`. Chunked request bodies are legal HTTP but **Python `http.server`-based OpenAI servers cannot decode them** — `litert-lm serve` answered `400 Invalid JSON` before its handler ran. Symptom isolated by a direct A/B (chunked header → 400; fixed length → 200, same body).

**Fix:** `reqwest::Body::from(body)` (sets `content-length`) in `remotes.rs`; the only request-side `wrap_stream` in the workspace (sibling lanes all pass fixed-length bodies). Pinned by `e2e__remote_forward__request_body_fixed_length_not_chunked` (raw one-shot peer asserts: no `transfer-encoding` in the forwarded head, honest `content-length`, model prefix stripped, body valid JSON) — the stub-llama-server cannot discriminate this because hyper reads chunked fine.

**Blast radius:** every `[[remotes]]`/federation forward to a chunked-intolerant peer (Python http.server family). Chunked-tolerant peers (hyper/uvicorn-based vLLM, llama.cpp, Ollama) were unaffected — which is why this survived until a Python-based peer was actually driven.

## Consequences

- `[[remotes]]` is a viable stopgap for openai-hosts today: prefix routing, streaming, bare-name fallback, peer listing all work; no supervised lifecycle (no auto-install, no health-gated respawn, no bench lane) — that gap is exactly Phase B.
- The `Remote` schema (`name`/`url`/`key`/`allow_insecure_http`) stays unchanged.
- Phase B must solve: descriptor registry, pip/uv install lane (mirror `sglang_install.rs`), health via `GET /v1/models`, shared-child supervision, Windows/mac verification for foundry.

## Phase B design decision — shared supervised child (router pattern)

**Session:** dp-20261107-openaihost-phase-b-spawn-model (branch `shared-child-router-pattern`, 8.05/10).

**Forcing fact (verified 2026-11-07, installed binary):** `litert-lm serve --help` offers only `--config/--host/--port/--cors-origin/--verbose` — there is **no per-model argv**, and the registry (`~/.litert-lm`) is process-global. Per-model spawn is vendor-blocked; working around it via per-child `HOME` or unverified `--config` registry keys would be fragile hidden state.

**Decision:** one supervised `litert-lm serve` child per installed OpenAiHost engine row, keyed ROUTER_KEY-style (single child per lane, never per model) — the exact pattern the supervisor already runs for llama-server `--models-preset` (spawn_router_instance / ensure_router_detached / ps_router).

Sketch for the build session:

1. `EngineKind::OpenAiHost` (+ `is_lazy_lane`, `serving_lane_typed`, `route_format` arms — ~50 caller sites, exhaustive sibling-caller scan).
2. Engine impl: `build_argv` = `serve --host --port` from the supervisor endpoint (model arg ignored, like router); `health_check` = `GET /v1/models` 200 (the `/health` JSON contract is llamacpp-specific); `enumerate_devices` = empty — litert self-manages GPU memory, admission treats the lane as opaque rather than double-budgeting.
3. `ps` rows from the child's own `/v1/models` (ps_router precedent); explain/recommend/offers integration like every lane.
4. Pull lane: `.litertlm` bundle into the blazar models dir (ModelRow, new FormatShape) + `litert-lm import`; `rm` removes the store row AND the registry alias (`litert-lm delete`) — no registry orphans.
5. Install lane: uv venv machinery mirroring `sglang_install.rs`; capability-lanes entry; V1-V5 lifecycle tests + stub-serve e2e.
6. Build-session step 0 (verification, not design): does `litert-lm import` hardlink or copy? If copy, note the disk cost or adopt via the existing cross-store hardlink machinery.
7. Foundry Local stays `[[remotes]]`-shaped (it is a self-managing service); revisit only with a Windows/mac validation box.

## Phase B — shipped (2026-11-07, same day)

All seven sketch points landed; `EngineKind::OpenAiHost` is a first-class lane.

**Surfaces:** core (`engine_kind.rs` variant + `ShardFormat::LitertBundle` + `LaneError::LitertUnserved` + routing arms + `FromStr`/`ToSql`/`FromSql`; `store.rs` `is_litert_bundle`; `config.rs` warm-peg exemption; `profile.rs` `compile_openaihost` fork with reserved-flag gate) · runtime (`OpenAiHostEngine` impl; supervisor `OPENAIHOST_KEY` lane with collapse redirect in `spawn_instance_forced`/captive; `manifest.rs` `probe_openaihost` (venv metadata + `serve --help` flag mining); `litert_install.rs` uv-venv install lane; `hf.rs` `.litertlm` pull fork with root-bundle selection + `litert-lm import`; `models.rs` rm/copy contracts) · gateway (`/api/ps` lane expansion via child `/v1/models`; explain format arm) · CLI (10 dispatch sites: install/update/offers/doctor/serve/knob hints; `engine install --kind openaihost`).

**Design deltas vs the sketch:** import HARDLINKS (step 0 answer — same-fs zero-copy, no adoption machinery needed); registry-wide serving honored exactly (collapse in supervisor, not per-model spawn); no capability-lanes entry — that registry pins `repo@ref_sha` for source-build lanes, while litert is a pip lane (`engine install --kind openaihost`, PyPI latest via doctor); devices stay opaque (no double VRAM budget).

**Tests (2502 passed, 31 binaries, +5):** `integration__openaihost_lane__shared_child_collapse_and_teardown` (collapse onto `_openaihost`, two models share one child/port, no per-model rows, authless loopback, shutdown kills the child and closes the port — V1/V5 class), routing truth tests (single-lane domain, unserved teaching, reverse-invisibility, pin honored), `compile_openaihost` reserved-flag gate, `version_tuple` pre-release tolerance, shim-script contract (extracted `shim_script` fn, no tautology), `draft_aware_name` `.litertlm` stem naming.

**Live receipts (target/debug binary, real litert-lm 0.19.1):** `engine install --kind openaihost` (uv venv + 6 flags probed + registered; note: newest install claims the global active slot — `engine use b12129-cuda` restores the serving default) · `pull litert-community/SmolLM2-360M-Instruct` → 356 MiB bundle + store row + registry alias, all three verified · buffered chat 200 with real generation · SSE streaming clean · `ps` shows the per-model row (synthetic lane key hidden) · `rm` removes row + bundle + alias together (third-artifact contract). Hint corrected to the hyphenated repo name (the underscore guess in the first hint was wrong; real repo verified via HF API).

**Open items:** Windows `.cmd` shim variant **shipped 2026-11-07** (`SHIM_NAME`/`shim_script_windows` in litert_install.rs — batch forwarder, `venv/Scripts` layout helpers, probe + finder contracts unit-pinned on both platforms; **cross-compile-proven**: `cargo check`/`clippy -D warnings` clean for `x86_64-pc-windows-gnu --all-targets` — every `#[cfg(windows)]` arm compiles for a real Windows target. msvc cross-build from Linux is blocked by `ring`'s build script (MSVC toolchain required — CI builds it on Windows runners); runtime `cmd.exe` behavior now **wine-proven on the dev box**: `scripts/validate.py --phase=os` cross-links the real `blazar.exe` (mingw windows-gnu) and drives it through the shipped `.cmd` shim inside wine's cmd.exe — argv forwarding, exit-code relay (success 0 / clap error 2), and a store-aware subcommand all pass; first Windows CI run narrows the remainder to real-hardware deltas. **Windows-python gold tier (2026-11-07, wine):** CPython 3.12.10 (python.org installer `/quiet`) → `-m venv` with real `Scripts/` layout → pip-installed litert-lm 0.19.1 (wheels fetched host-side `--platform win_amd64`, installed `--no-index` around flaky in-wine pypi egress) → the shipped `.cmd` shim forwards to the real `Scripts\litert-lm.exe` and renders the full CLI surface (`serve`/`import`/`delete`) — the entire install-lane contract is behaviorally proven under Windows userland. **Foreign-arch tier (qemu):** `aarch64-unknown-linux-gnu` cross-build (`RUSTFLAGS +fp16`, required by gemm-common 0.19 fp16 asm — aarch64 release CI must set it) runs `--version` + `engine list` under `qemu-aarch64-static -L /usr/aarch64-linux-gnu`. All wired as `scripts/validate.py --phase=os` (pass=9 boundary=2 fail=0; msvc + macos stay honest boundaries). Foundry Local still `[[remotes]]`-shaped pending a Windows/mac box. `engine update --kind openaihost --check` live-verified 2026-11-07: `litert-lm 0.19.1 is current (PyPI latest: 0.19.1)`.
