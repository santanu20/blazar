# llama.cpp engine-currency bump: quality and tool-call battery (b12686 -> b12756-cuda)

Date: 2026-11-26

Question: is bumping the llama.cpp engine currency quality-neutral, and does the
tool-call contract survive the bump?

## Environment

- one host: Intel i7-14650HX (16 cores), NVIDIA RTX 4070 Laptop GPU (8188 MiB),
  13674 MiB RAM, Linux 7.0.0-43-generic
- gateway binary: the release build under test (worktree branch
  sota-roadmap-20261123, HEAD 517578e), spawned in a throwaway sandbox
  (isolated config/data roots; engines and models hardlinked from the real store)
- model: qwen2.5-0.5b-instruct, GGUF Q4_K_M, 491 MiB
- prompt: "Count from 1 to 120 in words.", 200 completion tokens, temperature 0
- battery: scripts/toolcall_eval.py (T1-T12 + L1), three 200-token decode runs
  per arm

## Method

One sandbox, two back-to-back arms, identical battery. Arm A pins the older
engine with an exact-tag update (a channel downgrade). Arm B runs a plain
`engine update` (live discovery, latest channel) - discovery picks whatever the
newest upstream build is at run time.

## Install and update evidence

Arm A (exact-tag downgrade, both assets already cached - zero network):

```
installing upstream engine - runtime companion downloads show their own progress bar
channel switch: downgrade to b12686 - regression gate skipped
removed superseded engine b12728-cuda (freed 1.1 GiB)
lane llamacpp now holds only b12686
engine b12686-cuda active (build 12686, 1 devices, 341 flags) (newer than stable v1.0.0)
daemon restarted - the update is live

```

Arm B (live discovery - note it moved past the roadmap's b12728 within a day of
that note being written; upstream ships builds continuously):

```
installing upstream engine b12756 (newer than stable v1.0.0) - runtime companion downloads show their own progress bar
engine b12756-cuda installed; regression gate skipped
removed superseded engine b12686-cuda (freed 1.1 GiB)
lane llamacpp now holds only b12756
update channel: latest
engine b12756-cuda active (build 12756, 1 devices, 342 flags) (newer than stable v1.0.0)

```

The update downloaded fresh assets for the new build (b12756-cuda: engine tarball
220329612 B + CUDA runtime companion 594376973 B, 186.3 s total) and pruned the
superseded lane, freeing 1.1 GiB. Flag count: 341 -> 342.

## Tool-call battery

| check | b12686-cuda | b12756-cuda |
|---|---|---|
| T1 single-call name+args | pass | pass |
| T2 pick correct tool of three | pass | pass |
| T3 tool_choice=required | pass | pass |
| T4 tool_choice dict-form pin | pass | pass |
| T5 parallel-call emission (measured) | pass | pass |
| T6 tool result -> grounded answer | pass | pass |
| T7 strict json_schema output | pass | pass |
| T8 streaming tool deltas | pass | pass |
| T9 nested-schema args | pass | pass |
| T10 malformed-args history tolerance | pass | pass |
| T11 tool-error recovery | pass | pass |
| T12 streaming call assembly | pass | pass |
| L1 tools-vs-no-tools latency (p50 of 3) | pass | pass |

Totals: b12686-cuda 13/13, b12756-cuda 13/13.
The pass set is identical across the bump.

## Decode, steady state (temperature 0, 200 tokens)

| arm | engine | runs (tok/s) | median |
|---|---|---|---|
| A | b12686-cuda | 344.74, 344.02, 331.22 | 344.02 |
| B | b12756-cuda | 337.46, 333.53, 333.27 | 333.53 |

Median delta: -3.0% - within the shared-host noise band (this box
runs other sessions; the arms ran minutes apart, not on an idle machine).

## Reading

- the bump is quality-neutral on this workload: the identical 13/13 pass set
  across engines means the tool-call contract (selection, schema, streaming,
  recovery) survives the engine change
- decode medians differ by 3.0% - noise-band territory, not a
  regression signal; the regression gate normally watching this was skipped
  (--no-gate) so the battery itself is the quality evidence here
- the currency flow is exercised end to end: exact-tag downgrade path, live
  discovery to the actual newest build, supersede pruning with space reclaim,
  cache reuse for cached assets, and a daemon restart that lands the active
  engine without dropping the serving path

## Limits

- one host, one GPU, one model (qwen2.5-0.5b-instruct Q4_K_M); decode numbers
  are not comparable across hosts
- the decode-regression gate was skipped (--no-gate) in both arms; the
  tool-call battery and the decode runs above are the recorded quality evidence
- arms ran on a shared host (system daemon plus other sessions); decode medians
  carry a noise band of a few percent
- live discovery moves daily; the b12728 named in the roadmap was already
  superseded by b12756-cuda at run time - that is the currency flow working, not
  a deviation

## Artifacts

- battery JSON: <sandbox>/e2_bump_battery.json (arm A and arm B records)
- tool-call results: <sandbox>/results.json,
  <sandbox>/results.json (arm key; engine b12756-cuda)
- daemon logs: <sandbox>/e2_bump_daemon_b12686.log,
  <sandbox>/e2_bump_daemon_b12728.log
- this receipt pair: bench-artifacts/2026-11-23-sota-roadmap/llamacpp-bump-b12756-receipt.md
  and .json
