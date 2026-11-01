# Recursive Self-Improvement Loops — Deep Analysis + SOTA Research + ROI/Feasibility Roadmap

Date: 2026-10-31 · Kind: research/advisory (no code changed) · Sources: live code audit (codegraph + rg), MEMORY.md verified state, bench-artifacts/20261031-benchmark-audit, web SOTA research (cited inline)

---

## 1. Deep analysis — where Blazar actually stands

### 1.1 What the runtime already has (verified in code, not assumed)

Blazar is not missing self-improvement *organs* — it has most of them. What it lacks is the **nervous system that connects them into an autonomous cycle**. Verified surfaces:

| Organ | Where (file:line) | Trigger today | State |
|---|---|---|---|
| Measured config search | `tune_search` crates/blazar-runtime/src/bench.rs:739 | Manual CLI (`tune --search`) | Live llama-bench grid (KV×fa×threads×batch), hardware-shaped legs |
| Speculation race | `sglang_spec_mode_race` bench.rs:1182 | Manual CLI (`tune --spec-race`) | dense vs ngram vs dflash vs draft; writes `spec_receipts` |
| Ngram param search | `tune --ngram` (main.rs:266) | Manual CLI | size_m/min_hits → global config |
| Spec governor (B2) | supervisor.rs:431, lib.rs:1336 | **Automatic, online** | 60s collapse windows park speculation; quiet streak recovers; Prometheus gauge |
| Ubatch governor (D1) | lib.rs:1339 | **Automatic, online** | per-model prefill micro-batch shaping |
| Adaptive slots (LC4) | `adaptive_slots_tick` supervisor.rs:7678, `adaptive_slots_reap`:8006 | **Automatic, online** | 10s tick: saturation → reshape; warm-transitional-child protection; pinned-pressure teach events |
| Adaptive decode gate | config.rs:1173 (default true) | Automatic | per-session decode gating |
| Sentinel telemetry | sentinel.rs (`judge`, `watch`, `why`) | Automatic, per-request | detections, verdict stats, broadcast channel |
| Degrades registry | blazar_runtime::degrades (61d90c6) | Automatic | endpoint-keyed, respawn-clean, stamps requests + `/api/why` |
| Usage rollups | usage.rs:199 `/api/usage` | Automatic | day×model token/latency/TTFT rollups, 60s flush |
| Failure-KB / certs | model_caps (latest-wins), admission teach | Automatic on errors | teaches 502/504/lane-retry sites; doctor rows |
| Gap detector | `blazar doctor` (main.rs:318) | Manual | missing/stale bench, tune, capability certificate |
| Decode-regression gate | tune-baseline bench compare (main.rs:770/868) | On tune | blocks decode regressions; skippable flag |
| Receipt store | store.rs:103-197 | Passive | `bench_history`, `bench_results`, `spec_receipts`, `usage_daily`, `model_caps`, `completion_cards` |
| Federation routing | federation.rs:453 `route`, remotes.rs `score_peer`/`rank_peers_by_capacity` + **C4 prefix→remote stickiness** (remotes.rs:47) | Automatic | health/capacity/TTFT-EWMA scoring, prefix affinity for remotes |
| Semantic cache | semcache.rs (`SemanticCache`, cosine lookup) | Automatic | embedding-based response cache |

### 1.2 Verified open defects in the loop (evidence: bench-artifacts/20261031-benchmark-audit)

These are the **sensor errors** — every downstream automated decision reads this data:

| # | Defect | Consequence |
|---|---|---|
| F1 | Bench receipts orphan on engine update (all rows pre-b12015) | tune gates consult stale data silently |
| F2 | Bench harness pins threads=8; adopted profile runs 16 | cross-receipt comparisons are apples-to-oranges |
| F3 | spec-race ngram arm runs UNTUNED (ngram_*=0) → ratio 1.0003 receipt persisted | biases future spec=auto resolution against ngram |
| F4 | Auto-draft pairing is lane-blind (sglang base + GGUF draft; self-pairing) | wasted arms, warn noise, misleading dense-fallback receipts |
| F6 | KV quant measured FASTER than f16 (+11-15% pp) yet only engaged under VRAM pressure | free throughput left on the table by default |

Plus (MEMORY.md, 2026-10-31 validate run): CONFIG COVERAGE phase gate misses 110 knob rows; P1 posture: store-error≡absent at 44 sites.

### 1.3 The structural insight

Map Blazar onto the canonical self-improving-system cycle:

```
SENSE ──▶ DECIDE ──▶ ACT ──▶ VERIFY ──▶ RECORD ──▶ (loop)
```

- **SENSE**: strong (sentinel, usage rollups, degrades, governors' telemetry windows)
- **DECIDE**: weak-by-design — exists only as tiny per-concern governors (B2/D1/LC4) and manual doctor; no unified "what should improve next" brain; no receipt-freshness reasoning
- **ACT**: strong but manual (tune/bench/spec-race are user-invoked CLI; nothing runs them autonomously)
- **VERIFY**: strong (decode-regression gate, cert two-shape tools, golden suites, validate.py)
- **RECORD**: strong (receipt tables, latest-wins, completion cards)

The highest-ROI move is NOT a new algorithm — it is **closing the loop: making DECIDE+ACT automatic and receipt-driven**, with the existing gates as the safety interlock. That converts Blazar from "a server with excellent manual tuning tools" into "a server that tunes itself while you sleep" — which none of ollama/vllm/llama.cpp do out of the box.

---

## 2. Web SOTA research (what the frontier does)

### 2.1 Simulation/surrogate-based autotuning — Vidur (Microsoft, MLSys'24; arXiv 2405.05465)

High-fidelity LLM-inference simulator: profiles operators once, then **predicts** latency/throughput for arbitrary configs (<5% error validated), and searches the config space **in simulation** instead of live measurement (Vidur-global-search / Vidur-tradeoff-search). TensorRT-LLM's **AutoTuner** applies the same pattern in production: runtime profiling drives kernel selection (docs.nvidia.com/deepwiki TRT-LLM AutoTuner architecture).

**Blazar angle:** `bench_history` already accumulates (model, engine_tag, ctx, kv, threads, batch, pp, tg, hw_fingerprint) rows. A learned surrogate — even a conservative per-(model-family × engine × hw_fingerprint) regression/k-NN over past receipts — can (a) pre-narrow the live tune grid to uncertain cells (measure-only-where-uncertain = active learning), (b) detect drift (prediction vs. one live probe → receipt-staleness score), (c) power what-if capacity advice (`plan --install` already ranks fit; surrogate makes it measured, not estimated). This cuts tune cost from "N full llama-bench legs" toward "1-2 confirmatory probes" after the first run per model-family.

### 2.2 Adaptive speculation — AdaEDL (NeurIPS; arXiv 2410.18351), SVIP, DeepSeek DSpark

AdaEDL: **training-free, parameter-free** early-stop of drafting via an entropy-based lower bound on token-acceptance probability — draft length adapts per request instead of a static hyperparameter. SVIP (aclanthology): draft-model self-verification length policy for long context. DeepSeek's DSpark (public inference_deployment_qa.md, already evidenced in our vllm#54082 comment): confidence head + entropy gating, verified length 5.00→3.2-4.4 at ~unchanged acceptance, +4-7% @ ~60 seqs/rank, +11% @ 96 seqs; **SGLang already ships DSpark + `--enable-deterministic-inference`**.

**Blazar angle:** three rungs on our lanes — (1) sglang lane: adopt the upstream entropy/spec flags through the knob registry when present (cheap, evidence-gated); (2) llamacpp lane: draft-max is static upstream, but our B2 governor already has per-model acceptance telemetry — a *client-side* adaptive draft-length policy is engine-flag-gated, not fabricated (llama-server exposes draft params at launch only → rung 3 = upstream conversation, track llama.cpp issues); (3) generalize B2 from "park/unpark speculation" to "shape draft length from the same acceptance-rate window" — the governor already computes the needed statistic.

### 2.3 Learning schedulers/routers — TRAIL (Harvard), rl-inference-scheduler, PEACE

- **TRAIL** (minlanyu.seas.harvard.edu): embedding-based scheduling — learns request representations to improve response time.
- **rl-inference-scheduler** (github.com/manishklach): DQN/PPO agents vs FIFO/SJF/MemAware heuristics on a simulated server with TTFT SLOs + KV memory pressure.
- **PEACE** (computer.org): preemptive cluster scheduling, cuts p99 queueing delay on Azure traces.

**Blazar angle:** single-box 8GiB market ≠ datacenter RL. The idiomatic first rung is **deterministic measured-argmax routing**: bucket workloads by observable features (context length, concurrency class, tools on/off, stream, modality) from sentinel ctx; per bucket, route engine-lane by measured receipts (bench_results/spec_receipts), latest-wins. That's TRAIL's idea with receipts instead of embeddings — zero training, fully explainable via `/api/why` (explain.rs already narrates routing). RL (contextual bandit) is the *second* rung, only if the deterministic router's ceiling shows in receipts.

### 2.4 KV-cache-centric architecture — Mooncake, vLLM router, SGLang Unified Radix Cache

- **Mooncake** (kvcache-ai.github.io; arXiv): KVCache-centric disaggregated serving; Oct 2026: TensorRT-LLM merged a native Mooncake Store connector into KVCMv2 — content-addressed KV pooling + prefix reuse **across serving instances**.
- **vLLM router** (Feb 2026): prefix-cache-aware routing for P/D disaggregation — route to the pod that can reuse the prefix while balancing load.
- **SGLang Unified Radix Cache** (lmsys.org, Aug 2026): one radix tree for hybrid-attention models.

**Blazar angle:** C4 prefix→remote stickiness already gives us prefix-affinity for federation. Rung 2 = **prefix-affinity for local multi-replica/multi-lane children** (hash of normalized prompt prefix → child sticky-table, same bounded-eviction pattern as remotes.rs:47). Full Mooncake-style KV pooling across processes is datacenter-scale — **not YAGNI** for local-first Blazar; revisit only if multi-node federation becomes a shipped use case.

### 2.5 Architecture validation — NVIDIA Dynamo

Dynamo (open source, **built in Rust**, developer.nvidia.com): low-latency modular inference framework fronting SGLang/TensorRT-LLM/vLLM with KV-aware routing, disaggregated phases, request cancellation. Validates Blazar's Rust-gateway-over-multi-engine architecture at datacenter scale; its KV-aware router heuristics are worth reading for our federation scorer.

---

## 3. The proposal — Blazar's Recursive Self-Improvement Loop (SDDVR)

One cycle, all organs already exist; the work is the connective tissue + hygiene:

```
        ┌────────────────────────────────────────────────────────────┐
        │                     SENSE (exists)                        │
        │  sentinel detections · /api/usage rollups · degrades      │
        │  registry · governor windows (B2/D1) · slot pressure      │
        └───────────────┬────────────────────────────────────────────┘
                        ▼
        ┌────────────────────────────────────────────────────────────┐
        │  DECIDE (NEW — the Autotune Steward, daemon-side)         │
        │  receipt freshness vs engine_tag/hw_fingerprint (fixes F1)│
        │  surrogate prediction from bench_history (Vidur-lite)     │
        │  doctor-gap logic promoted from CLI into the daemon       │
        │  → emits prioritized improvement queue (bench/tune/race)  │
        └───────────────┬────────────────────────────────────────────┘
                        ▼  (idle-window gate: --sleep-on-idle signal +
                        │   VRAM headroom check; NEVER fights prod child)
        ┌────────────────────────────────────────────────────────────┐
        │  ACT (exists, currently manual)                           │
        │  bench → tune --search → tune --ngram → spec-race         │
        │  ordered correctly (fixes F3 ordering, F4 lane pairing)    │
        └───────────────┬────────────────────────────────────────────┘
                        ▼
        ┌────────────────────────────────────────────────────────────┐
        │  VERIFY (exists — the interlock)                          │
        │  decode-regression gate · quality certs · golden suites   │
        │  hysteresis: governor-style streaks before any adoption;  │
        │  adoption = receipt + BlazarEvent + doctor row (no silent │
        │  config mutation — H2/H4 posture preserved)               │
        └───────────────┬────────────────────────────────────────────┘
                        ▼
        ┌────────────────────────────────────────────────────────────┐
        │  RECORD (exists)  bench_results · spec_receipts ·         │
        │  model_caps latest-wins · /api/why narrates every change  │
        └───────────────┬────────────────────────────────────────────┘
                        └────────────── back to SENSE (recursion) ────┘
```

Design invariants (from the codebase's own doctrine):
- **Fail-loud, teach-first**: every auto-adoption writes a receipt and an event; `/api/why` and `blazar why` narrate what the steward changed and why (explain.rs pattern).
- **Deterministic, observable** (H11): measured-argmax decisions, never vibes; the surrogate only *proposes*, live probes confirm.
- **Anti-flap**: governor streak/hysteresis pattern reused for adoptions (never adopt off one window).
- **VRAM-peaceful**: idle-window gating + the existing orphan-sweep/pinned-pressure machinery; a measurement child never evicts a warm serving child (the LC4 warm-transitional-child lesson generalized).

---

## 4. ROI × Feasibility roadmap

| Pri | Item | ROI | Effort | Feasibility | Notes |
|---|---|---|---|---|---|
| **P0.1** | Receipt-hygiene wave: S2 lane-aware draft pairing + S3 ngram-before-race ordering + S4 thread alignment + S5 engine-update receipt invalidation (doctor row + auto-restamp) | **High** — fixes every downstream decision | S (days) | Trivial | Pure correctness; no new machinery; closes F1-F4 |
| **P0.2** | KV-quant default via measured receipts (S1): q8_0 adoption for VRAM-comfortable llamacpp models through tune fa-leg | High (+11-15% pp measured) | S | Trivial | Gate = tune receipt per model; quality verified ppl-neutral (literature + our kv-quality r1-r5 campaigns) |
| **P1.1** | **Autotune Steward** daemon loop (SDDVR connective tissue): doctor-gap logic in-daemon + idle-window queue executing bench→tune→race in the correct order, VRAM-gated, event-visible | **Very high** — this IS the recursive loop; differentiator vs ollama/vllm | M (1-2 waves) | High — all organs exist; risk is scheduling safety, solved by idle gate + interlock | Config: `autotune_steward = "off\|advise\|auto"` (advise = queue+doctor rows only, default; auto = act in idle windows) |
| **P1.2** | Vidur-lite surrogate over bench_history (per model-family × engine × hw_fingerprint): pre-narrow tune grid + drift detector (predict-vs-probe residual ⇒ staleness score feeding P1.1's queue) | High (tune cost ↓ 3-10×; F1 class eliminated structurally) | M | High — data already accumulates; start linear/k-NN, no ML dep | Keeps "measure to confirm" — surrogate never adopts alone |
| **P2.1** | Adaptive speculation: (a) sglang lane entropy/spec flags via knob registry when upstream exposes (DSpark already in sglang); (b) B2 governor upgraded to shape draft length from its own acceptance-rate window | Med-High (AdaEDL: -20-35% latency on high-entropy workloads; DSpark +4-11% tput) | M | (a) trivial flag adoption; (b) engine-gated | llamacpp upstream draft-len is launch-static — do NOT fabricate client-side speculation; track upstream |
| **P2.2** | Workload-aware measured-argmax lane routing: sentinel ctx buckets (ctx-len, concurrency, tools, stream) × bench receipts → deterministic router + `/api/why` narration; replaces/augments quality auto-route | Med-High for multi-lane boxes | M | High — zero training, receipts only | TRAIL-with-receipts; contextual bandit later only if ceiling shows |
| **P2.3** | Local prefix-affinity child selection (C4 pattern generalized to local replicas/multi-lane) | Med (TTFT on multi-turn; cache_bust interplay must be respected) | S-M | High — remotes.rs:47 pattern to copy | Skip full Mooncake KV pooling — not YAGNI local-first |
| **P3.1** | Contextual-bandit router (ε-greedy/UCB over P2.2 buckets with sentinel-scored reward) | Unknown until P2.2 ceiling measured | L | Medium | Only with receipts proving deterministic ceiling; must stay explainable |
| **P3.2** | Adaptive semcache threshold (accept-threshold tuned from sentinel divergence feedback) | Med | S-M | High | Watch false-positive risk; conservative bounds |
| **P3.3** | Config-coverage gate completion (110 orphan knob rows) + P1 store-error≡absent (44 sites fail-closed) | Hygiene — protects the loop's own config plane | M | High | Pre-existing open items; steward depends on knob truth |

**Explicitly rejected for now:** RL cluster scheduling (PEACE/DQN class — wrong scale for single-box), cross-process KV pooling (Mooncake/KVCMv2 — datacenter pattern, revisit under federation multi-node), LLM-driven self-tuning agents (non-deterministic, violates H11 posture; the measured-argmax path is stronger and auditable).

---

## 5. Why this wins (competitive framing)

- **ollama**: no measured tuning, no receipts, no governors, static defaults. Blazar + P0/P1 = "it got faster overnight and can prove why."
- **vLLM/SGLang**: excellent engines, but tuning is docs + flags the user must read. Dynamo adds datacenter-scale routing, not local self-tuning. **Self-improvement on a single box with receipts is an open lane.**
- Everything proposed reuses verified organs; the differentiator is the *loop*, and the loop is connective tissue, not research risk.

## 7. Reassessment addendum (2026-11-01) — external review verified against live code

An independent feasibility review of this doc was fact-checked against `main @ c7282bb` (v0.29.0). Results:

### 7.1 Claims VERIFIED (accepted — new P0 items)

| Claim | Evidence (verified this session) |
|---|---|
| **UsageMeter::flush() loses drained data on failure** — REAL bug, survived the silent-fallback audit | usage.rs:88 — `map.drain()` runs BEFORE `Store::open()`; open-fail returns `Ok(0)` with buckets already gone; per-row `add_usage_delta(...)?` aborts mid-batch losing the rest. Worse: the warn says "will retry next tick" — false, nothing remains to retry. Fix = drain-after-open + transactional batch + requeue-on-failure |
| **spec_receipts lacks provenance** | store.rs:120-130 — columns are model/engine_tag/mode/tok_per_s/dense_tps/ratio/measured_at only. No hw_fingerprint, ngram params, tuned-vs-default flag, draft identity, or rep count. Blocks surrogate + safe auto-mode |
| **Tuned-profile persistence writes `hw_fingerprint: None`** | bench.rs:1000. One-site fix — store API (store.rs:1602-1614) and model_caps already write real fingerprints |
| **S3 only partially closed** | main.rs:14837-14843 prints untuned-ngram warning, but the receipt persists unmarked; `best_measured_spec_mode` (supervisor.rs:236-245) gates only on `ratio < 1.05` — a noisy untuned arm above 1.05 is selectable. Fix = tuned-provenance column + exclusion, or require tuned config for eligible persistence |
| **No KV quality gate in tune adoption** | bench.rs `tune_search` rg: zero quality/ppl/eligibility refs — argmax tg adopts q5/q4 blind. BUT: `registry/kv-quant-quality.json` ppl registry + `kv_advisory` (explain.rs:472) already exist — the eligibility policy has a live data source; this is wiring, not a new campaign |
| **Ubatch governor default off** | config.rs:3001 `ubatch_auto: false` |

### 7.2 Claim CORRECTED (stale evidence)

**"Gateway overhead 0.547 @C8 = top optimization target"** — that was the PRE-FIX number. Root cause (single-flight coalescing serializing parallel duplicates behind the leader's full generation) was found and fixed same night in `299bf5a` (default off). After-fix receipt (`gateway-overhead-ladder-after-fix.json`): **C1 1.135, C4 1.017, C8 0.986, p99 parity @C8 (0.869 vs 0.864)** — gateway is at aggregate parity with the direct engine. Residual: **C2 = 0.792** (below parity while C1/C4 are above — unexplained boundary anomaly, worth one bounded pass; suspects: -np 2 slot/concurrency interplay at the 2-slot boundary). Gateway-overhead as a P0 workstream is therefore mostly closed; remaining scope is the C2 dip, not the ladder.

### 7.3 Amendments ACCEPTED into the architecture

1. **SCHEDULE promoted to a first-class SDDVR step** (between DECIDE and ACT): exclusive maintenance lease, VRAM/RAM headroom gate, no-in-flight/no-queued-for-target check, conflict arbitration with engine updates, durable bounded job queue, timeout/cancel/recovery. In-repo proof this is needed: the LC4 warm-child churn regression (supervisor.rs:13542 — a coordination loop evicted the warm child the drain had installed) is exactly the failure class an unsynchronized steward would reproduce at scale.
2. **Advise-mode staging**: Steward ships advise-only first (deterministic prioritized queue + explanations via CLI/API/doctor, zero mutations); auto mode only after advise trust + maintenance controller exit criteria. Matches house doctrine (spec_auto_manage flipped default-on only post-proof; singleflight flipped default-off post-measurement).
3. **Autonomous adoption pipeline fixed order**: baseline → tune-search → ngram refine → spec-race → candidate remeasure → quality+regression validation → adopt → post-adoption monitor + rollback; every transition = receipt + BlazarEvent (lifecycle events don't exist yet — new event enum arms).
4. **Vidur-lite 3-10× treated as hypothesis** — holdout evaluation vs full-grid baseline before any claim; surrogate proposes, never adopts.
5. **P2 routing starts 3 dimensions only** (ctx-len bucket, concurrency bucket, engine lane) — persisted telemetry lacks tools-indicator and per-bucket usage rows today; extend after the deterministic baseline.

### 7.4 Consolidated wave plan (supersedes §4 ordering)

| Wave | Deliverable | Exit criterion |
|---|---|---|
| 0 | Usage flush transactional + requeue (fix the reviewer's find) | Failed flush loses zero measurements; warn tells the truth |
| 0 | Receipt provenance: spec_receipts axes (hw_fingerprint, spec params, tuned flag, draft id, reps) + bench.rs:1000 hw fix | No unqualified or misleading receipts; schema migration v18+ |
| 0 | KV quality eligibility wired into tune adoption (consult kv-quant-quality registry per grade: q8 needs per-model evidence, q5 stronger, q4 explicit; context-window-matched) | Throughput alone can never adopt a degrading grade |
| 0 | Untuned-ngram receipt exclusion (S3 close-out) + C2-dip bounded investigation | best_measured_spec_mode reads only provenance-eligible receipts |
| 1 | Steward advise mode | Correct prioritized recommendations, zero mutations |
| 2 | Maintenance controller (SCHEDULE) + Steward auto mode | Benchmarks never compete with production; verified adopt+rollback |
| 3 | Vidur-lite surrogate (holdout-gated) | Beats/closes-on full-grid at lower cost, measured |
| 4 | Workload-aware deterministic routing (3 dims) | Measured latency/throughput win |
| 5 | Bandit / adaptive semcache | Justified by observed deterministic ceiling |

Post-v0.29.0 commits already landed (context for readers): `e493946` (S2-S5 + S1/S6/S7 receipts), `299bf5a` (singleflight fix), `ba120ff` (tune adoption writes engine-tagged bench receipt — closes the F1 doctor false-positive).

### 7.5 Third review adjudication (2026-11-01) — "measured optimizer, not autonomous self-improving AI"

Verdict: **APPROVE the reframing; merge two new structural elements; one defer-condition amended.** The review introduces no new code claims requiring verification — its four P0 findings are exactly the ones verified against live source in §7 (usage.rs:88 drain-before-open; spec_receipts schema store.rs:120-130 lacking provenance; no KV-quality consultation in tune_search adoption; bench.rs:1000 `hw_fingerprint: None`). Its Wave 0 ≡ §7.4 Wave 0; its maintenance-lease/VRAM-not-traffic and advise-first requirements ≡ accepted amendments 1-2; its rejections (RL, Mooncake, LLM agent, custom speculation policies beyond engine-controllable params) match ours.

**Accepted — new structural elements:**

1. **Append-only `optimization_experiments` ledger** (lifecycle: attempted → measured → quality-passed → adopted → rolled-back; every candidate an immutable row carrying model/engine/hw/profile/config hashes + benchmark/quality/memory metrics JSON). The adopted profile becomes a pointer to the best *verified* experiment. This supersedes "extend spec_receipts with more axes" as the general substrate — spec_receipts/bench_results stay as latest-wins read models over the ledger. Rides the same schema-v18+ migration as Wave 0 provenance. Closes the review's core risk: a fast-but-unverified experiment can never masquerade as a proven configuration.
2. **Multi-objective candidate verification** (Interactive: p95 TTFT + inter-token latency; Throughput: aggregate tok/s at target concurrency; Memory-efficient: peak VRAM/RAM at target ctx/C) with hard eligibility constraints first, Pareto selection among eligible — replacing single tg-mean argmax in tune adoption. Enabling diff: bench capture must add ITL p50/p95 and per-config peak VRAM/RAM (llama-bench does not emit these; extend the drive-style concurrent harness already in bench.rs). Adopted into Wave 2 as the candidate-verification stage, alongside the 3-condition minimum matrix (single short-prompt / single long-prompt / concurrent representative) and workload-representative re-measurement before adoption — the latter guards exactly the C2-dip class (search-metric ≠ serving-metric) left open in §7.
3. **CLI surface `blazar optimize <model> --plan | --run`** — same semantics as Steward advise/auto with better product naming; adopt the rename. Durable state machine (DETECTED→QUEUED→LEASE→BASELINE→SEARCH→VERIFY→ADOPT/REJECT→MONITOR→ROLLBACK/COMPLETED) with persisted transitions and `autotune = off | advise | scheduled` replaces the free-form "auto" flag.

**Amended:** Vidur-lite demoted to P4-never → keep **data-gated, not deferred-indefinitely**. The experiments ledger IS the surrogate's training set; every Wave 2 run enriches it for free. Promotion trigger: coverage criterion (≥N models × M configs × same hw fingerprint) + holdout win vs full-grid at lower cost. Prefix-affinity rejection reasoning accepted (prompt-hash ≠ KV-reuse proof; Dynamo couples cache-reuse with projected load) — already P2-conditional on real multi-replica, no change.

Merged wave plan: Wave 0 unchanged **+ experiments ledger in the same migration**; Wave 1 = `optimize --plan`; Wave 2 = `optimize --run` + multi-objective verification + maintenance lease; Waves 3-5 unchanged (Vidur data-gated). Objective renamed: *reliable, measurable, reversible optimization that earns autonomy through evidence* — operationally identical to the advise-first→earned-autonomy path already accepted in §7.3.

### 7.6 Architecture freeze (2026-11-01, final review) — IMPLEMENT NOW

Fourth review approves the §7.5 merged architecture with four final adjustments. **Architecture frozen after this section; no further proposal expansion before Wave 0 ships.**

**Adjustment 1 — ledger normalized into append-only tables** (never mutate a lifecycle row):
| Table | Responsibility |
|---|---|
| `optimization_experiments` | Immutable candidate identity: model artifact hash, engine tag, hw fingerprint, config hash/json |
| `optimization_measurements` | Append-only benchmark + quality measurement rows (TTFT/ITL percentiles, tps, VRAM/RAM, quality result) |
| `optimization_events` | Append-only lifecycle transitions + reasons (attempted/measured/quality-passed/adopted/rolled-back) |
| `active_profiles` | Pointer to the currently approved experiment |

`bench_results` / `spec_receipts` remain runtime read models, linked (experiment_id) to the experiment that produced them. Reproducibility + rollback history without disrupting runtime lookup paths.

**Adjustment 2 — explicit objective, not bare Pareto**: three named objectives (Interactive: min p95 TTFT+ITL · Throughput: max aggregate tok/s at defined C · Memory-efficient: min peak VRAM/RAM given min-throughput+quality floors). Hard quality/memory/reliability constraints first → eligible Pareto set → chosen objective + deterministic tie-breaks. No universal profile.

**Adjustment 3 — benchmark the actual serving path**: llama-bench win ≠ gateway-path win. Validate the drive-style harness captures required E2E metrics; baseline vs candidate under identical workload, warm-up, repeated **alternating** measurements; adopt only when improvement > measurement noise + minimum threshold. Missing telemetry (e.g. VRAM) makes that decision **ineligible** — never a silent estimate substitution.

**Adjustment 4 — staged autonomy**: `optimize --plan` (read-only) → `optimize --run` (benchmark/search/verify, NO adoption) → `optimize --apply` (adopt verified candidate) → `autotune = scheduled` (only after repeated successful manual cycles; lease must prove draining, resource conflicts, timeouts, restarts, rollback). 8 GB GPU assumption: benchmarking may require **stopping the model child** — never assume VRAM for both.

**Authorized scope**: Wave 0 (evidence integrity: telemetry loss, receipt provenance, KV quality gates, speculation eligibility) → Wave 1 (`--plan`) → Wave 2 (`--run`: experiments, multi-objective verification, maintenance control, rollback) → Wave 3 (scheduled, after repeated manual success). NOT authorized now: Vidur-lite, bandits, RL routing, local prefix affinity, custom adaptive speculation — evidence-gated future options. Performance-improvement percentages come only from Wave 2 measurements; no claims before.

**Wave 0 status: LANDED 2026-11-01** (schema v19; usage flush transactional + requeue; spec_receipts provenance + untuned-ngram exclusion; KV-quality gate via `blazar-core/src/kv_quality.rs`, registry moved to `crates/blazar-core/registry/`; append-only ledger `optimization_experiments/_measurements/_events` + `active_profiles` pointer wired through `persist_profile`; real hw fingerprints on all new receipts). Live-verified E2E on the production store: tune-search gate fired (unevidenced grades excluded at ctx, f16 won), full ledger chain written, spec-race rows carry provenance. 2096 tests green. Next: Wave 1 `optimize --plan`.

**Wave 1 status: LANDED 2026-11-01** — `blazar optimize <model> --plan` (advise-only). Deterministic adviser in `blazar-core/src/optimize.rs` reads only stored evidence: engine presence, bench freshness (lane + hardware + age), tuned-profile provenance (cites the ledger experiment # and profile hash; serving ctx from profile `-c` else config default), KV-grade quality eligibility at the serving ctx (per-grade, registry-backed), speculation qualification (ratio ≥ 1.05 AND tuned; untuned ngram receipts flagged EXCLUDED), capability certificate presence, workload-telemetry sufficiency (30-day request floor). Each recommendation carries the exact command, the why, the datum, and a benchmark-session budget; unreadable evidence degrades to a warn finding naming the error. CLI renders the card (`--json` emits `blazar.optimize.plan`); bare `optimize` without `--plan` teaches advise-only. Live-verified read-only on the production store (content-hash unchanged) and through the validation harness (wave13 lanes). 2339 tests green. Next: Wave 2 `optimize --run`.

## 6. Sources


- Vidur — arXiv:2405.05465 (MLSys'24); github.com/microsoft/vidur; pyshine overview (extracted)
- AdaEDL — arXiv:2410.18351 (NeurIPS'24); SVIP — aclanthology (self-verification length policy)
- DSpark — DeepSeek public inference_deployment_qa.md (via our vllm#54082 evidence comment, 2026-10-20); sglang ships DSpark + `--enable-deterministic-inference` (verified in-thread)
- TRAIL — Harvard (minlanyu.seas.harvard.edu PDF); rl-inference-scheduler — github.com/manishklach; PEACE — IEEE computer.org (Azure traces)
- Mooncake — kvcache-ai.github.io (KVCMv2 TRT-LLM connector, 2026-10); arXiv Mooncake paper; vLLM router prefix-cache-aware routing (barrahome.org 2026-02, ai.engineer talk)
- SGLang Unified Radix Cache — lmsys.org 2026-08; RadixCache deepwiki
- NVIDIA Dynamo — developer.nvidia.com/dynamo; TensorRT-LLM AutoTuner — docs.nvidia.com / deepwiki
- Local evidence: bench-artifacts/20261031-benchmark-audit/report.md (F1-F6, S1-S7); crates/* live source audit (this session, codegraph)
