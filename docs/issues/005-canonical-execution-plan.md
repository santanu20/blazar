Title: Introduce the canonical ExecutionPlan

Priority: P0 | Wave: 2 | Backlog: B02 | Effort: L | Depends: 001, 003

## Problem

Execution decisions are spread across several mechanisms that all work but
never meet:

- Cmd::Plan / Cmd::Fit / Cmd::Explain in crates/blazar-cli/src/main.rs
- per-objective engine orders (Throughput / Latency / Quality) in
  crates/blazar-core/src/optimize.rs
- CLI preset registry (preset_find) in crates/blazar-cli/src/main.rs
- capacity fit (async fn fit) in crates/blazar-gateway/src/capacity.rs,
  mounted at /api/fit in crates/blazar-gateway/src/lib.rs
- supervisor admission (fit guard, last-resort automatic tensor-parallelism)
  in crates/blazar-runtime/src/supervisor.rs

There is no typed, explainable artifact that captures the decision itself.

## Approach

1. ExecutionPlan as a data model, not an argv string: model,
   selected_engine, context_size, slots, kv_policy, speculation_policy,
   parallelism plan, confidence, reasons, warnings.
2. Two-stage engine selection: hard-constraint filter first (model
   architecture, file format, quantization, modality, GPU/memory fit, engine
   availability, explicit user restriction), then evidence-scored ranking
   over the survivors.
3. Scoring rule: missing benchmark evidence must depress a candidate's
   score, never default it high. Only applicable, measured evidence counts.
4. Absorb the per-objective orders in crates/blazar-core/src/optimize.rs
   into the planner so exactly one decision path remains.
5. blazar explain renders the plan (reasons, warnings, confidence).

## Acceptance criteria

- [ ] ExecutionPlan type landed in the appropriate crate layer
- [ ] Planner emits typed plans with reasons/warnings for arbitrary models
- [ ] Single decision path: engine orders subsumed, no parallel recommender
- [ ] Missing-evidence scoring rule documented and unit-tested
- [ ] blazar explain renders the plan

## Out of scope

Switching blazar run to the planner (issue 006).
