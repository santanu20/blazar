# Load-aware slot re-provisioning — design spike

Date: 2026-10-29
Scope: LC4 adaptive-slot capacity shaping under bursty concurrency.
Status: D1-D5 implemented 2026-10-30 + measured (first gate FAIL on
this hardware class — scenario mismatch, see §4 Results; machinery
live-proven; D5 + pre-warm shipped in Wave C answer the findings).

## 1. Problem statement

Blazar's throughput at elevated concurrency is queueing-bound, not
compute-bound. The 2026-10-29 gateway audit measured the resident
default lane at **C4 = 0.56× and C8 = 0.61×** of C1 aggregate decode
throughput (bench receipts in `bench-artifacts/`): the child's slot
count caps parallel decode, excess requests park on admission, and the
per-stream rate stays flat while the queue wait grows.

The adaptive-slots machinery exists and is healthy
(`adaptive_slots_tick`, 10s reaper cadence). Three structural gaps
keep it from closing the burst window:

| # | Gap | Code anchor (verified) |
|---|-----|------------------------|
| G1 | **Manual `slots` pins are excluded outright.** A config or overlay that sets `slots = 2` (a common production shape, and the shape the C4/C8 numbers ran under) never adapts — the exclusion treats every pin as a policy statement. | `adaptive_slots_tick`: `overlay.slots.is_some() → skip` |
| G2 | **60s to first adoption.** `SLOTS_STREAK_TICKS = 6` at a 10s tick means a burst shorter than ~60s never earns a reshape; by the time adoption fires, the burst is draining. | `SLOTS_STREAK_TICKS` (const = 6) |
| G3 | **Reshape drains by parking admissions.** `drain_reshape_queue` holds new admissions until in-flight reaches 0, then respawns (a full model-load of wall time). Under a sustained stream, parked waiters absorb the tail latency the reshape was meant to remove. | `drain_reshape_queue` + `reshape_draining` |

What already works (keep): pressure gauges with window-peak latching
(one adoption reaches the demanded shape instead of +1 stairs),
demand-recency keeping streaks alive across request-granularity dips,
decay back to the natural shape after a quiet window, quiesce-only
respawn (never kills a live stream), fit-safe total-ctx re-spend
(`-np` divides the same ctx budget), `SLOTS_ADOPT_CAP` ceiling.

## 2. Candidate designs

### D1 — Burst fast-path (addresses G2)

**Status: IMPLEMENTED 2026-10-30** (`adaptive_slots_tick`,
`SLOTS_FAST_STREAK_TICKS = 2`, `SLOTS_FAST_COOLDOWN = 300s`,
`fast_adopted_at` per-model map; cooldown also cleared on adoption
decay and on spawn-failure rollback; adoption step stays
demand-sized off the window peak, capped by `SLOTS_ADOPT_CAP` —
`peak ≥ 2 × effective` shortens only the streak, never the sizing.
Tests: `unit__adaptive_slots__burst_fast_path_adopts_at_two_ticks`,
`..._fast_path_cooldown_demotes_to_standard_window`,
`..._sub_double_peak_keeps_standard_window`.)

First-step acceleration: when the latched window peak attests
`peak ≥ 2 × effective_slots`, adopt after **2 saturated ticks (20s)**
instead of 6, stepping straight to `min(peak, cap)`. Rationale: a
2× overcommitment peak is not noise (each park climbs through every
level, so the peak is demand-attested); waiting 60s for what 20s of
data already proved is the latency cost. Anti-flap stays intact —
decay hysteresis is untouched, and a mis-adopted shape still decays
after the quiet window.

Risk: reshape churn on sawtooth loads (burst-quiet-burst every ~30s).
Mitigation: keep a per-model adoption cooldown (e.g. no second
fast-path within 5 min; the standard 6-tick path stays available).

### D2 — Pinned-slot teach + opt-in override (addresses G1)

**Status: IMPLEMENTED 2026-10-30** (`pinned_pressure_teach`,
supervisor.rs; `adaptive_override` on `ModelOverride` — config.rs;
`SLOTS_PINNED_TEACH_DEDUP = 600s` warn dedup; `pinned_streak` /
`pinned_warned_at` maps; `SlotsPinnedPressure` event surfaces as
`slots_pinned_pressure` on the gateway event stream. Remedy grammar:
`blazar config set model_overrides."<model>".adaptive_override on`.
Tests: `unit__pinned_lane__sustained_pressure_teaches_once`,
`unit__pinned_lane__adaptive_override_lifts_exclusion`.)

Do NOT silently adapt pinned lanes (the pin is policy). Instead:
1. When a pinned lane shows sustained pressure (same gauge), emit a
   `SlotsPinnedPressure` event + log line with the remedy:
   `blazar config set <model>.adaptive_override on` (new overlay knob).
2. `adaptive_override = true` lifts ONLY the manual-pin exclusion for
   that model; replicas/deterministic exclusions stay.

This turns a silent queueing floor into a one-command fix, keeps
explicit pins authoritative by default, and needs no new machinery —
the tick already computes everything before the exclusion check.

### D3 — Drain starvation bound (addresses G3)

**Status: IMPLEMENTED 2026-10-30** (budget path —
`SLOTS_DRAIN_BOUND = 90s`; `reshape_draining` values became
`ReshapeDrain::{Parking(Instant), Cutover}`: a busy drain older than
the bound flips to `Cutover` + warn + `SlotsDrainCutover{waited_secs}`
event and stops parking new admissions; a new larger adoption removes
the marker (re-arms the budget); the quiesce path still re-parks fresh.
The second-replica pre-warm alternative stays DEFERRED (VRAM-gated —
see §5). Test:
`unit__adaptive_slots__drain_bound_trips_cutover_opens_gate`.)

The drain path parks admissions indefinitely under sustained streams.
Bound it: if `reshape_draining` age exceeds a budget (e.g. 90s), stop
holding NEW admissions and instead **complete-and-cutover** — let new
requests into the old shape until the next natural quiesce. The
reshape is a latency optimization; a 90s parking floor can cost more
tail latency than the reshape recovers.

Alternative for stream-heavy lanes (chat with long generations):
**second-replica pre-warm** — spawn a second instance at the target
shape, switch routing, then evict the old one at quiesce. Costs a
model-load of VRAM during cutover (8 GB cards may not host two
residents of a 4 GB model at useful ctx) — viable only where VRAM
headroom exists; gate on the existing fit walk.

**Status: IMPLEMENTED 2026-10-30 (Wave C)** — `drain_reshape_queue`'s
not-busy arm now spawns the replacement under `model#warm` via
`spawn_instance_captive` (captive = the old resident is never picked
as the eviction victim): old child keeps serving while the warm child
loads, so the reshape's admission gap closes to zero. On Ready the
warm instance is re-keyed onto `model` (pidfile/apikey renames +
`name` rebuild through `Arc::try_unwrap`) and `SlotsPrewarmCutover` +
`SlotsReshaped` fire; a fit-refused pre-warm (two residents will not
fit — the common case on 8 GB cards for 4 GB models) logs and falls
back to the verbatim evict-then-ensure path. Trade-off: the warm
child starts without the old child's KV bank (a cache re-prefill,
~100 ms class, versus the old path's full admission drain; cutover
only happens at in-flight == 0, so no live stream is touched).
Tests: `unit__prewarm__cutover_rekeys_and_retires_old`,
`unit__prewarm__refused_falls_back_and_holds`,
`integration__prewarm_reshape__cutover_same_key_new_argv` (real stub
child: post-cutover resident re-keyed, argv `-np 4`, both events).

### D5 — Decode-scaling awareness (from the 2026-10-30 measurements)

**Status: IMPLEMENTED 2026-10-30 (Wave C)** — `note_model_throughput`
now routes each per-request decode sample into two per-model EWMA
classes by live concurrency (solo at in-flight ≤ 1, loaded at depth
`n+1`, concurrency blended alongside rate);
`decode_scaling_ratio` returns `loaded_rate × loaded_conc /
solo_rate` when both classes carry fresh samples (10 min window,
else the gate stands down — no evidence, no verdict).
`adaptive_decode_gate` (default on) suppresses slot adoption when
the ratio < `DECODE_SCALE_ADOPT_MIN = 1.25`: sustained queueing
teaches `SlotsDecodeBound` (600 s dedup, remedy in the log) instead
of paying a 25-30 s reshape that measured-evidence says buys queue
relief only. Tests: `unit__decode_scaling__class_routing_and_freshness`,
`unit__adaptive_slots__decode_bound_suppresses_adoption` (suppression +
kill-switch + healthy-ratio adopts). Live expectation on
granite-tiny@4070 (finding 1 in §4): the C8 cell should now REFRAIN —
`SlotsDecodeBound` fires where the 2026-10-30 ladder adopted and lost
25 % wall to the transition.

**Boundary findings from the Wave C live runs (forensic cells F1/F2,
receipts in `bench-artifacts/20261030-dgate-slot-adaptation/`):**

- granite-tiny@4070 sits AT the threshold: conc-2 ratio measures
  ≈ 1.22-1.27 vs `DECODE_SCALE_ADOPT_MIN = 1.25`, and conc-2 scaling
  is a weak predictor of conc-8 benefit (conc-8 aggregate ≈ 1.19). A
  model this close to the line is gated or not by measurement noise —
  the forensic F1 cell (conc-8-seeded hints) still adopted.
- Tail-of-burst pollution: the last request of an N-deep burst notes
  at live-concurrency 1, landing in the solo class and dragging the
  solo rate down — inflating the ratio. Unit tests seed clean classes
  and never see this. Residual-counter note-time routing (excluding
  the noter's own contribution) is future work.
- Honest posture: D5 is a conservative gate, not a guarantee. For a
  model known GPU-bound the durable protection is a manual `slots`
  pin (the tick then teaches instead of adapting) or
  `adaptive_decode_gate = false` for deliberate reshape-freezing.

### D4 — Pressure telemetry (cross-cutting)

**Status: IMPLEMENTED 2026-10-30** (`slots_gauge_snapshot` on the
supervisor — per-model rows dedup replica keys; `/metrics` families
`blazar_slot_pressure_per_model`, `blazar_slots_adopted_per_model`
(adopted rows only), `blazar_slots_effective_per_model`. Test:
`unit__slots_gauge_snapshot__dedupes_replica_keys`.)

Publish `slot_pressure` and `adopted_slots` as gauges on `/metrics`
(event stream already carries `SlotsAutoAdopted`). The C4/C8 audit
needed offline log parsing; gauges make the queueing floor visible in
`blazar top` and scorecards.

## 3. VRAM budget constraint (why not always `-np` high)

`-np N` divides the same total ctx budget across N slots — per-slot
context halves. For agentic/serving workloads that need ≥ 16k ctx per
stream, raising np past `total_ctx / 16k` degrades capability, not
just ITL (np8 ITL 60ms vs np4 37ms, flagprobe 2026-09-29 receipt).
The spike target is **burst-fit shapes** (np sized to the observed
window peak, capped by the profile walk), not max-np defaults.

## 4. Measurement plan (ties to the 2026-10-29 engine bench)

Re-run the concurrency ladder (C1/C2/C4/C8) on
`ibm-granite_granite-4.0-tiny-preview` (Q4_K_M, MoE) with:

- A: b11946-cuda, natural shape (baseline for the MoE-fusion delta)
- B: b11982-cuda, natural shape (engine-isolates the CUDA MoE fusions
  #31288/#31188 — same harness, same model, same box)
- C: b11982-cuda, D1 fast-path simulated by a manual `slots` raise at
  t+20s (stand-in for the adoption change; isolates the reshape win
  from the engine delta)

Decision gate: D1/D2 proceed only if C shows aggregate C4 throughput
≥ +25% over B AND p99 queue wait drops ≥ 30%. Engine-delta receipts
land in `bench_results` (PK: model + engine_tag) — both legs persist
independently.

### Results (measured 2026-10-30 — receipts in
`bench-artifacts/20261030-dgate-slot-adaptation/`)

Harness: sandboxed gateway (validate.py Sandbox + bench_matrix
`conc_suite`), granite-4.0-tiny-preview Q4_K_M on b11998-cuda, RTX 4070
Laptop. Legs: B = `slots=2` pin, C = pin + `adaptive_override on`.
Two ladder shapes: short-burst (8 rounds/cell) and sustained
(24 rounds/cell ≈ 100-190s), plus a continuous-arrival cell (no round
barrier) for the D2 teach.

**Gate verdict: FAIL as measured — root cause is a scenario mismatch,
not an implementation defect.** Every mechanism fired correctly and is
live-proven:

| Evidence | Result |
|---|---|
| D1 fast-path (live) | C8-C: `adaptive slots adopted from=2 to=8 fast_path=true` at ~20s of 6-deep queue pressure (sustained ladder) |
| D2 teach (live) | Continuous cell: WARN `sustained queueing behind a manual slots pin … lift it per model with: blazar config set model_overrides."<model>".adaptive_override on` at ~60s (pinned_slots=2, pressure=2), deduped, pin stayed authoritative (zero adoptions, 180/180 streams ok) |
| D3 bound (unit) | Drain bound never tripped live — drains completed in 10s < 90s budget. Correct: the bound is a starvation safety valve, not a regular path |
| D4 gauges (live) | All cells: `blazar_slot_pressure` / `slots_adopted` / `slots_effective` tracked every transition (e.g. 6/2/– → 0/8/8 through the adopt) |

Numbers (sys t/s, ttfb p99):

| Cell | B pin2 | C pin2+override | Δ agg | Δ queue |
|---|---|---|---|---|
| short C4 | 141.0 / 1886ms | 140.5 / 1767ms | −0.4% | −6.3% |
| short C8 | 141.9 / 4246ms | 100.9 / 18786ms | −28.7% | +343% |
| sustained C4 | 138.6 / 1898ms | 140.6 / 1802ms | +1.4% | −5.1% |
| sustained C8 | 140.2 / 4154ms | 104.7 / 17717ms | −25.3% | +327% |

Three findings explain every cell:

1. **Decode is GPU-bound at 2 slots on this model class.** Leg B is
   flat ~140 t/s aggregate from C2 upward (C1 ≈ 95) — the tiny MoE
   saturates the GPU at batch 2, so extra slots cannot raise aggregate
   throughput, only relieve queue wait. The gate's +25% aggregate
   premise presumes slot-starved decode; it does not hold here. Where
   it does hold (dense models, partial-offload lanes, batch-scaling
   decode, admission-bound long-ctx), the machinery applies as built.
2. **The reshape transition costs 25-30s once per episode** (10s drain
   hold + 4 GB model reload) and post-adopt per-stream gain was only
   +3% (sum 5989 vs 5799 t/s) — on minutes-scale loads the transition
   dominates, which is the entire C8-C regression. The structural fix
   is the deferred D3 second-replica pre-warm (zero-drain cutover),
   not more adoption heuristics.
3. **Wave-structured (barrier-round) load resets saturation streaks.**
   Rounds of ~4-7s busy + gap rarely give 6 consecutive saturated 10s
   ticks, so neither the standard window nor the D2 teach fires at C4
   (peak queue 2 < 2×effective 2 → fast path ineligible by design);
   C8's peak 6 ≥ 2×2 is why only it fast-adopts. The 2× fast-path
   threshold is protective here: forcing C4 adoption would add
   transition cost with no aggregate upside on GPU-bound decode.

Consequences recorded for the next iteration:

- **D5 (implemented 2026-10-30, Wave C): decode-scaling awareness.**
  Adopt only when recent decode throughput scales with in-flight depth
  — the GPU-bound case now teaches instead of reshaping. See the D5
  section above.
- **D3 second-replica pre-warm** moved from "deferred" to IMPLEMENTED
  (Wave C, see the D3 section above): it removes the transition cost
  that finding 2 quantifies, wherever VRAM can host the transient
  double residency; the fit-refused fallback keeps the old path.
  Live-proven 2026-10-30 (F2 run 6): pin 2 seeds → adopt 2→6 → warm
  child spawns `np=6` (delegation precedence) → single zero-gap
  cutover → `blazar_slots 6 == slots_configured 6`, 536/536 requests
  ok, no churn. Three defects found and fixed on the way: the tick
  now skips `#warm` transitional children, a Ready warm child is
  consumed instead of re-spawned, and the re-key retries a bounded
  `Arc::try_unwrap` (the detached settle-task no longer pins the
  Instance — `settled_mib` is now `Arc<AtomicU64>`). A fourth,
  deeper fix: LC4 spawn precedence — `slots` + `adaptive_override`
  now lets the adopted shape reach the child (the pin seeds, the
  delegation authorizes); excluded models hand shape authority back
  (`adopted_slots` cleared on exclusion).
- **Gate metric revision:** future slot work should gate on queue-wait
  (ttfb p99) under *continuous* arrival load, not aggregate throughput
  under barrier-synchronized rounds.

## 5. Non-goals

- No default-on behavior change for pinned lanes (D2 is opt-in).
- No cross-model slot pools (per-key isolation is a correctness
  invariant of the replica-key design).
- No speculative pre-warm on idle (cost of a wrong guess is a
  model-load of VRAM + cold start).
