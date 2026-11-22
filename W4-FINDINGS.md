# W4 Campaign Findings — 2026-11-22

Scope: A4 cache-controlled overhead, B7 lane-quality matrix, B10 tool-call
A/B. All numbers come from committed harness runs under
`bench-artifacts/2026-11-22-*`; receipts are the source of truth, this file
is the navigation layer.

## A4 — matched overhead, orchestration gain, and the measurement traps

Seven campaigns; each killed a specific confound. Verdicts per
`scripts/bench_saturation.py` acceptance engine (paired per-round bootstrap,
CI95, B=2000, seed 1337).

| Campaign | Mode | Headline verdict |
|---|---|---|
| `a4-cache-controlled` | evict-per-round, tg128 | overhead +? (cold), band FAIL — noise |
| `a4-tg512` | evict-per-round, tg512 | **single_request_overhead PASS +1.3%** (honest cold number) |
| `a4-resident` | resident, natural EOS | orch +41.6% / p95 +220%, band FAIL (shape + ngram confounds) |
| `a4-pinned2` | pinned, natural EOS | band FAIL 0.8965 — early-stop mixture imbalance |
| `a4-pinned` | pinned spec=off, natural EOS | band FAIL — early-stop fiction (see below) |
| `a4-pinned3` | pinned slots=2 spec=off, **force-length** | **band PASS 0.9143, overhead PASS −0.8%** — the clean matched proof |
| `a4-orch` | unpinned, force-length, 3rd leg | **p95 PASS** (TTFT 60–115 ms vs direct 1533–2538 ms), orch-vs-direct +19% |

### The two measurement bugs this wave found and fixed

1. **Early-stop fiction.** Greedy temp-0 completions on short prompts
   early-stop; `sys_tps = requested_tg / wall` then reports 8300 t/s for a
   ~340 t/s engine. Every "fast round" in the first campaigns was this
   artifact; the "slow rounds" were the only full-length ones.
   `llama-bench` (matched build) ground truth: pp128 12772 t/s, tg64
   391.6 t/s. Fixes: actual-token accounting
   (`usage.completion_tokens`, content-delta fallback) and
   `--force-length --eos-token-ids` (logit_bias EOS suppression) so every
   stream decodes exactly `tg` tokens.
2. **Adaptive-shape contamination.** Mid-campaign slot adoption (2→8→16→8)
   silently reshaped the gateway child between rungs; the C=16 "wins" in
   `a4-resident` measured a np=8 child against a np=2 direct child. The
   harness now records `gateway_np_by_rung` from /proc after every gateway
   rung and surgically excludes reshaped rungs from the matched-band and
   single-request verdicts (orchestration verdicts are exempt — adaptive
   shape is their subject).

### What the clean numbers say

- **Matched gateway hop cost: 1–3 % throughput** (pinned slots=2, spec off,
  forced length, 12 rounds; medians C=1 346/338 → C=16 606/590 t/s).
- **Orchestration p95 TTFT: 15–25× better under load** (gateway admission
  flattens TTFT to 60–330 ms while the direct child wave-queues to 1.9–3.1 s
  at C≥8).
- **Aggregate ceiling law**: throughput caps at `np × per-slot speed`
  (~290–350 t/s here). Orchestration wins at high C come from slot-count
  adaptation, not from the hop.
- **Known confound kept visible**: the gateway child carries
  `--lookup-cache-dynamic`/`--slot-save-path` banks (warm-state product
  feature), so unpinned C=1 "gateway faster" numbers are warm-state
  advantages, not overhead wins. Orchestration-vs-matched-gateway aggregate
  verdicts flip sign with ngram-acceptance lottery between campaigns
  (+41.6 % vs −24 %); orchestration-vs-**direct** is the stable claim.

### Product findings routed for follow-up

- **C=4 admission pacing — CLOSED 2026-11-22, not a defect.** The
  original signal (resident campaign: gateway C=4 TTFT p95 1.6 s vs
  direct 0.5 s, queue_wait_sum 2.68 s) does not survive clean
  measurement: in the pinned force-length run
  (`bench-artifacts/2026-11-22-a4-pinned3/`) gateway C=4 TTFT is
  p50 900 ms vs direct 849 ms, p95 1795 vs 1768 — the same 1–6 % hop
  band as throughput. The resident-run elevation was the early-stop
  mixture confound (streams that stop at 13 tokens free slots early on
  one leg only), and the observed queue_wait equals the physics of
  np=2 slots at C=4 (streams 3–4 wait out one full 512-token decode
  of the first pair), not a dispatch-release granularity problem. No
  code change without a defect; reopening trigger = TTFT p95 gateway
  − direct > 10 % in a pinned force-length campaign.
- **Daemon log-freeze — hardened, root cause still open.** 3 forensic
  repros of `run/daemon.log` + `sentinel.jsonl` writers freezing while
  the daemon kept serving; controlled instrumented repro (strace +
  growth watchdog) was negative. Shipped in 6383a67: reaper heartbeat
  (60 s INFO breadcrumb with resident/inflight — a silent window is
  now a provable timestamp gap, live-validated at exactly 60.002 s
  cadence) and `TeeWrite` poison recovery (a poisoned file mutex no
  longer permanently darkens the durable log arm). Full forensics:
  `docs/issues/012-silent-log-freeze.md`.

## B7 — lane × suite quality matrix (same weights family, three formats)

`scripts/bench_matrix.py --quality-only`, suites seed 1337, greedy parity.

| Lane (model) | code | instruct | multilingual | niah | reason | safety | schema | embed | conc-probe |
|---|---|---|---|---|---|---|---|---|---|
| llamacpp GGUF q4_k_m (0.5b) | 2/5 | 4/10 | 3/8 | 6/6 | 0/12 | 12/12 | 6/6* | 6/6 | — |
| mlx 4bit MLX-layout (0.5b) | 3/5 | 2/10 | 4/8 | 6/6 | 0/12 | 8/12 | 6/6 | n/a | 6/18, 0 err |
| vllm AWQ (8b) | deferred | | | | | | | | |

\* direct-leg 5/6 vs gateway 6/6 on llamacpp; blazar-only embed suite 6/6.

- Same 0.5b weights family, different format+engine → **measurably
  different task success** (safety 12/12 vs 8/12, instruct 4/10 vs 2/10,
  code 2/5 vs 3/5). This is the raw signal a quality-aware router consumes.
- **vllm 8B lane deferred**: the sandbox host-RAM guard refused the run
  (MemAvailable 5.7 GiB < model 5.8 GiB + headroom ≈ 8.3 GiB) with two
  agent sessions + desktop resident. Receipt kept in
  `bench-artifacts/2026-11-22-b7-vllm/` (error cell). Retry in a low-load
  window; the sglang-8B tool-call battery (fe30846) already covers that
  lane's tool dimension.
- Harness gap found while mis-pairing a lane: an MLX-layout dir row swept
  with `--engines vllm` collapses `text_engines` silently (one-kind domain
  rule). The harness now logs the collapse reason loudly. The
  `qwen2.5-0.5b-instruct-4bit` row is `mlx-community/...` — MLX-layout, mlx
  lane only; the only vllm/sglang-eligible dir on this box is
  `qwen3-8b-awq` (AWQ carries `quant_method`).
- Venv-lane observability gap: `KIND_CMDLINE_MARKERS` doesn't recognize the
  mlx python child → per-child attribution shows zeros and `gpu_peak` is
  system-total for venv lanes. Follow-up candidate.

## B10 — tool-call A/B per lane (gateway 8199, sandbox daemon)

| Lane (model) | Score | Hard failures |
|---|---|---|
| llamacpp GGUF (0.5b) | **12/13** | T10 malformed-args history → **HTTP 500** |
| mlx 4bit (0.5b) | **6/13** | T3/T4 `tool_choice=required`/dict-pin → **HTTP 400**; T1/T2/T8/T9/T12 no calls / empty streams |
| sglang AWQ (8b, fe30846) | 8/9 | — (concurrent session's committed battery) |

Tool-call reliability is lane-dependent, and two lanes have transport-class
edges worth product attention: malformed tool history crashes the llamacpp
lane's request (500, service survives) while mlx tolerates it (200);
`tool_choice` pinning is rejected with 400 on the mlx lane. Gateway-level
tolerance mapping is a candidate improvement — receipts in
`bench-artifacts/2026-11-22-b10-toolcall/`.

## Harness additions this wave (uncommitted → this commit)

- `bench_saturation.py`: actual-token accounting, `--force-length` /
  `--eos-token-ids`, `lane_mode` + cold-start verdict gating,
  `gateway_np_by_rung` shape-drift receipts + surgical rung exclusion,
  warm-before-capture + child-argv polling, direct-child stderr capture,
  JSON round-trip level-key coercion, tg default 256. Selftest 36/36.
- `bench_matrix.py`: `--quality-only` lane mode (quality legs only, all
  speed lanes skipped), loud text-lane-collapse logging.
- `toolcall_eval.py`: `BLAZAR_TOOLCALL_OUT` output-dir override (default
  unchanged).

## Reproduce

Every artifacts dir carries `campaign_cmd.txt` / `sat_receipt.json` with the
exact argv, daemon pid, child argv, and per-round legs; acceptance verdicts
re-judge offline via `bench_saturation.py --from-receipt <dir>`.
