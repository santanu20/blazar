# Enterprise cold-start & idle practices — wave 2 receipt (2026-10-09)

Scope: approved follow-ups from the cold-start wave (sglang /proc key leak,
idle CPU burn) + survey of how production serving stacks handle cold start,
idle cost, and warmup. All numbers live-measured on this machine; raw probe
script + JSON sit in `bench-artifacts/20261009-enterprise-perf-wave2/`.

## What changed in blazar

### 1. Child-auth argv lane refused (leak removal, not a regression)

`mint_child_auth` (crates/blazar-runtime/src/supervisor.rs) previously fell
back to `--api-key <secret>` for engines without `--api-key-file` (sglang
0.5.21 today). `/proc/<pid>/cmdline` is world-readable on stock Linux, so
that key guarded nothing while publishing a `plm_` secret to every local
process. The lane is now refused: argv-only engines mint **no** key and log
one INFO line naming the real boundary (127.0.0.1 bind + random high port +
residency-bounded lifetime — identical posture to the mlx/freetoken lanes
that already run auth-less). The llamacpp keyfile lane (0600 file, path-only
argv) is untouched; `validate.py --phase=knobs_behavior` re-proves
direct-to-child 401 + proxied 200.

Residual boundary, stated plainly: any local process that discovers the
child's random port can talk to it unauthenticated. This was already true
before (the key was readable from /proc); now nothing pretends otherwise.

Upstream gap to file (sglang): `--api-key-file` or env-var key support.
Blazar picks it up with zero code change the moment an engine manifest
grows the flag (the mint is manifest-driven).

### 2. `--sleep-on-idle` default-on for sglang (idle CPU)

Live-measured on a resident idle `qwen3-1.7b`: the sglang scheduler
subprocess busy-polled at ~101% of one core continuously. sglang 0.5.21's
`--sleep-on-idle` swaps the poll loop for ZMQ blocking sleep. Emission is
default-on in `compile_sglang` lifecycle hygiene, manifest-gated **silently**
(default posture must not manufacture per-spawn warn noise on older
engines), and stands down for `sleep_on_idle = false` (explicit opt-out
knob) or a user `extra_args` pin.

| measurement | before | after |
|---|---|---|
| idle scheduler CPU (% of one core, 15s stat delta) | ~101% | **1%** |
| TTFT warm vs after 70s idle (temp-0, 64 tok stream) | 49ms | 47–63ms band, same 5-probe run |
| temp-0 output sha across sleep boundary | — | identical (5/5) |

Wake cost is noise-scale: the after-idle probe sits inside the warm probes'
own variance band, and deterministic output is unchanged.

Note: this flag is CPU-idle hygiene only — it is NOT vLLM-style weight
offload (see mapping below). Blazar's idle story on the sglang lane remains
full residency + `idle_sleep_secs` eviction; the REPL pre-spawn from the
previous wave hides the re-spawn cost from the user.

## How production stacks handle this (mapping)

| system | mechanism | blazar equivalent |
|---|---|---|
| vLLM sleep mode | `/sleep`+`/wake_up` HTTP; level 1 offloads weights to CPU RAM + drops KV, level 2 drops both; fast wake for same model | gap on sglang lane (upstream has no equivalent); llama lane `--sleep-idle-seconds` is the same spirit; blazar compensates with REPL pre-spawn + admission coalescing |
| NVIDIA Triton ModelWarmup | config-declared synthetic requests run at model load; readiness gated on warmup completion | shipped: health-gated readiness + warm-peg burst at spawn |
| Ollama keep_alive | default 5m idle unload | `idle_sleep_secs = 300` default (configurable) — parity |
| KServe / Ray Serve | scale-to-zero + min-replicas warm pools + keep-alive windows | single-node posture: resident model + idle eviction + pre-spawn; warm-pool N/A by design |
| TGI | router warmup + startup probe | warm-peg + healthz gating — parity |

Primary sources: docs.vllm.ai sleep_mode feature page; NVIDIA Triton server
docs (model_configuration.md, ModelWarmup); KServe scaling docs; Ollama
keep_alive FAQ. Extracts cached at `/tmp/opencode/` during research (not
shipped).

## Verification ledger

- `cargo nextest run`: 2077/2077 (4 new sleep-on-idle tests, 1 rewritten
  mint test; all pre-existing sglang/lifecycle tests untouched-green)
- clippy: clean; fmt: clean
- deployed via systemd 19:43 IST; daemon census post-deploy: zero
  argv-fallback WARNs, warm-peg clean (0.3s, 4/4 burst), INFO teaching
  line present
- child cmdline: zero `plm_` occurrences; `--sleep-on-idle` present
- idle CPU: 1% of one core (was ~101%)
- validate.py: `--phase=knobs_argv` (7 checks, 0 fail) and
  `--phase=knobs_behavior` (`child_auth direct 401 / proxied 200` ok)

## Upstream filings queued (notes for maintainers)

1. sglang: request `--api-key-file` (or env `SGLANG_API_KEY`) so the
   loopback child can carry real auth without argv exposure.
2. sglang (optional, larger): vLLM-style weight-offload sleep levels for
   multi-model laptops — would let blazar keep several models "warm" in
   system RAM without VRAM pressure.
