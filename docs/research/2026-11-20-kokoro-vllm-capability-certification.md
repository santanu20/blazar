# Kokoro + vLLM capability certification — 2026-11-20

Closes the two lanes the capability census flagged as lacking live evidence
(`docs/issues/002-vllm-kokoro-capability-certification.md`): Kokoro had no
registry entry at all and vLLM had never been live-served through a Blazar
gateway on this box. Receipts and raw artifacts live in
`bench-artifacts/20261120-kokoro-vllm-cert/`; this note records the method and
the two root-cause findings that came out of the work.

## What was certified

| Lane | Capability | Evidence now |
|---|---|---|
| kokoro | speech_synthesis | LiveCertified — `POST /v1/audio/speech` (af_heart) -> 200 in 3 s incl. lazy boot, RIFF 24 kHz mono 16-bit 5.09 s wav; `/v1/audio/voices` lists the kokoro section |
| vllm | serving lane E2E | explain `source=pin` -> child spawn with exactly the registry emission surface -> greedy echo 200 (43 s cold incl. boot, 113 ms warm) -> SIGTERM reap in 2 s, VRAM back to idle |

vLLM per-family rows stay at their pre-existing honest statuses (Probed /
FirstClass); one lane-level E2E pass certifies the spawn path and the four
feature-meaningful emitted flags (`--served-model-name`, `--max-model-len`,
`--gpu-memory-utilization`, `--enable-prompt-tokens-details`), which now have
entries in `scripts/certification_receipts.json`. `--kv-transfer-config` and
`--max-num-seqs` are opt-in/pin-only and remain uncertified until a workload
exercises them.

## Root cause 1: vLLM manifest under-probe (store said 1 flag)

The store's vllm-0.34.0 manifest held `flags=['--help']` — a stale probe from
before the `--help=all` fix. vLLM 0.34 prints a config-group *index* for bare
`--help`; the full surface needs `--help=all` (428 flags after Blazar's
`parse_help`). The probe fix already existed in the working tree
(`crates/blazar-runtime/src/engine/manifest.rs`, `probe_vllm`); re-running
`blazar engine reprobe vllm-0.34.0` refreshed the store manifest. Lesson:
manifest flag-count regressions on an engine row are a probe-code smell, not
an engine change — reprobe before trusting them.

## Root cause 2: isolated-daemon certification pattern

`blazar serve` reads only `XDG_DATA_HOME`/`XDG_CONFIG_HOME` (no CLI overrides),
so per-campaign daemons are cheap: snapshot the store with python's sqlite
`backup()` API into a scratch data home, write a two-line scratch config with
the `model_overrides.<name>.engine` pin, and point both XDG vars at it. The
scratch daemon shares nothing with the resident daemon except the engines and
model files (absolute paths in the store), so certifications never disturb the
running service. This is the pattern to reuse for every future lane cert.

## Operational notes re-confirmed

- vLLM cold boot on the 0.5 B BF16 lane: ~40 s of the 43 s cold request;
  everything after is ~113 ms gateway+engine round trip.
- Clean SIGTERM teardown reaps the whole process tree (serve -> EngineCore ->
  resource_tracker) and returns VRAM to idle; no new `/dev/shm` mmaps. The
  stale `torch_*` shm files already present carry old pids (known crash-debris
  issue, documented in the 2026-11-19 KV data-plane note — not from this run).
- Kokoro is certified against the *installed* daemon (lazy child per request);
  no config surface is involved.
