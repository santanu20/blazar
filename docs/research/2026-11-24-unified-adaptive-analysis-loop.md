# Unified adaptive analysis loop — architecture and implementation plan

**Status:** architecture proposal attached to PR #17. The phases below distinguish implemented PR behavior from future work; this document is not evidence that the complete loop is already shipped.

## Objective

Evolve Blazar from a collection of inference-time controls into one **evidence-guided adaptive inference system**. Each request should receive an explainable serving plan, a bounded compute plan, an appropriate inference strategy, independent verification where available, and an auditable result receipt. Evidence from completed requests and benchmark campaigns may improve future policies only through controlled evaluation and adoption.

The loop must be useful across supported models and engine families without pretending that every backend exposes the same capabilities.

## Existing components to compose — not duplicate

| Existing component | Existing responsibility | Role in the unified loop |
|---|---|---|
| `blazar_core::execution_plan` | Explainable engine, context, slots, KV and speculation decision | Serving-plan component |
| `gateway::workload` | Dialect-neutral workload and resource classification | Workload identity and routing/admission context |
| `blazar_core::optimize` | Evidence findings, benchmark provenance and operator recommendations | Offline optimization/evidence consumer |
| Existing benchmark campaigns and `scripts/bench_matrix.py` | Performance, resource and quality measurements with receipts | Experimental evaluation substrate |
| `gateway::adaptive` | Bounded candidate, revision and PUCT execution | Request-time controller |
| `gateway::reasoning_search` | Search tree, evaluator interface, uncertainty and verifier state | One strategy executor, not the system-level planner |

Do not move gateway dependencies into `blazar-core`. A future composition layer belongs in the gateway (for example, `gateway::analysis`) and may consume the core execution plan. Keep `execution_plan` and inference-compute planning as separate typed decisions.

## Target request path

1. **Normalize:** protocol adapters normalize OpenAI, Anthropic and Ollama request shapes into the canonical internal request. Unsupported surfaces retain their existing behavior and return explicit capability errors.
2. **Describe:** derive the existing `WorkloadDescriptor`; capture task constraints, modality, tool use, streaming mode, latency target and user pins.
3. **Resolve serving:** obtain the existing `ExecutionPlan` from model format, routing policy, installed capabilities, hardware and evidence. Never override an explicit engine/model/config pin.
4. **Plan inference:** select a strategy and allocate a single request-scoped compute budget. Initial difficulty estimates must be labeled as heuristics until trained and calibrated.
5. **Execute:** run direct inference, bounded revision, Best-of-N, self-consistency, beam/PUCT search, planning/execution separation, tool/evidence workflows or another registered strategy.
6. **Evaluate:** invoke applicable deterministic verifiers, sandboxed execution, evidence checks and independently operated calibrated evaluators. LLM critique is advisory; self-consistency is not proof.
7. **Allocate or stop:** continue only when expected marginal quality gain justifies marginal cost and all hard limits still permit another step.
8. **Select:** choose or synthesize the strongest permitted candidate. Preserve verifier status; never upgrade unknown to verified because a judge or majority agrees.
9. **Receipt:** persist a sanitized, provenance-carrying execution receipt, then return the normal protocol response plus a stable, low-cardinality result receipt.
10. **Learn safely:** use offline benchmark analysis and shadow/canary evaluation to propose policy changes. Request-time adaptation must not silently mutate global serving defaults.

## Proposed composition contract

The following is conceptual; it is not yet a committed Rust API.

```rust
struct AnalysisPlan {
    workload: WorkloadDescriptor,
    execution: ExecutionPlan,
    objective: ObjectiveProfile,
    compute: ComputeBudget,
    strategy: InferenceStrategy,
    verifier_policy: VerifierPolicy,
    policy_version: String,
    evidence_snapshot: Option<EvidenceSnapshotRef>,
    pins: Vec<PolicyPin>,
}

struct ComputeBudget {
    max_model_calls: u32,
    max_prompt_tokens: u64,
    max_completion_tokens: u64,
    max_reward_calls: u32,
    max_tool_calls: u32,
    max_wall_time_ms: u64,
    max_cost_micros: Option<u64>,
}

struct AnalysisDecision {
    outcome: OutcomeStatus,
    selected_candidate: Option<String>,
    stop_reason: StopReason,
    verification: VerificationSummary,
    consumed: ComputeUsage,
    receipt_id: ReceiptId,
}
```

The composition layer owns the objective and policy decision, not engine spawning. The execution planner remains authoritative for serving feasibility; the queue/admission layer remains authoritative for resource availability; API keys remain authoritative for caller permissions and hard limits.

## Compute and quality rules

- Charge every generation, candidate, revision, tool and evaluator call to the same request budget. Keep local estimated state-token accounting separate from provider-reported usage; neither may silently substitute for the other.
- Track wall time and actual token usage continuously. A configured evaluator timeout is a per-call ceiling, not an aggregate budget.
- Treat PRM scores as usable reward signals only when the evaluator is independently operated, its calibration identity matches the configured pin, and its domain/OOD acceptance policy permits the task. A matching string ID is provenance, not proof of calibration quality.
- With no calibrated reward, use hard verifiers where applicable and conservative candidate selection. Generator-supplied priors are proposal heuristics only.
- Hard verifier failure dominates preference, consensus and reward-model scores. Unknown verification remains unknown.
- Preserve an incumbent when a revision fails, times out, is malformed or does not improve the defined objective.
- Keep strategy-specific capabilities explicit. For example, caller-owned tools remain handoffs unless the gateway is the authorized executor; streaming and media paths must not be claimed as supported until tested.

A useful strategy-level continuation rule is:

`continue` only when expected marginal quality improvement exceeds its marginal cost threshold, subject to hard latency, token, cost, safety and caller-permission budgets. The threshold must be objective-specific and measured, not a universal magic number.

## Evaluator and verifier portfolio

| Level | Evaluator | What it can establish |
|---|---|---|
| 1 | Deterministic response contracts | Schema, exact arithmetic, explicit format and constraints |
| 2 | Sandboxed execution | Compilation, unit tests, reproducible code/task execution |
| 3 | Evidence/provenance checks | Whether factual claims are supported by retrieved evidence |
| 4 | Calibrated process reward | Domain-bounded assessment of intermediate progress |
| 5 | LLM critique, pairwise judge or self-consistency | Advisory ranking; never independent proof |

Each evaluator needs an identity/version, declared task domain, timeout and output schema. Calibration metadata should include evaluation set/version, sample count, calibration metrics, OOD policy and timestamp. If these are absent, mark the evaluator uncalibrated.

External reward services receive task/partial-state content. Configuration must be explicit opt-in and documented as a data egress boundary; apply data minimization, HTTPS or loopback-only transport, secret isolation and a retention/privacy contract.

## Learning and adoption are a separate control loop

Do not make each online request a training event that can mutate global policy. Separate:

- **Online adaptation:** allocate the current request's bounded compute from request features, objective, current SLO/queue state and a pinned policy version.
- **Offline analysis:** aggregate receipts, benchmark outcomes, costs and resource measurements; train/calibrate difficulty and evaluator models; compare policies on held-out data.
- **Adoption:** shadow evaluation, canary, explicit promotion, versioned rollback. Keep the previous policy if confidence, quality, cost or reliability gates fail.

Initially use transparent, deterministic policies. Introduce contextual bandits or learned routing only after the evidence ledger has representative task-level outcomes and a reliable offline replay path. Avoid uncontrolled online exploration on customer traffic.

## Benchmark acceptance

Reuse the existing benchmark campaign framework, hardware/resource sampling and receipt/archive workflow. Extend adaptive evaluation; do not create a second unrelated benchmark ledger.

Required strategy arms:

- Direct/greedy baseline.
- Auto/sequential revision.
- Best-of-N and self-consistency.
- Beam search.
- PUCT without a calibrated PRM.
- PUCT with calibrated PRM.
- Planning/execution split.
- Candidate synthesis where the task admits a meaningful checker.

Required task families: held-out math/reasoning, code with executable checkers, structured-output contracts, evidence-grounded QA, tool-use completion, long-context retrieval and open-ended tasks with human or validated rubric evaluation.

Match model revision, prompt, sampling, task split and total compute budget. Charge selection, critique, evaluator and tool calls. Report task-level confidence intervals and the quality/cost/latency Pareto frontier, plus per-domain failure and calibration analysis. Repeated samples of a handful of templates are a smoke test, not a reasoning leaderboard. Never claim SOTA from a smoke suite.

## Phased delivery

### Phase 0 — correctness and truthful receipts (part of PR #17)

- Neutralize uncalibrated evaluator scores before pruning, value backup and answer ranking.
- Keep generator priors separate and explicitly heuristic.
- Use conservative state identity; case-folding and whitespace collapse are unsafe for code and indentation-sensitive tasks.
- Name exact-state duplicate suppression accurately; it is not yet a shared-node transposition DAG.
- Test hard-verifier precedence, uncalibrated fallback, duplicate-state safety, budgets, failed evaluator calls and final verification status.

### Phase 1 — shared analysis composition

- Add a gateway-level typed composition plan that consumes existing `WorkloadDescriptor` and `ExecutionPlan`.
- Separate serving feasibility, inference strategy, objective, verifier policy, compute budget and pinned policy version.
- Keep protocol adapters thin and add an explicit support matrix for local/remote engine, dialect, streaming, MCP and special lanes.
- Preserve admission, key-token reservation, engine affinity, caller-owned tool handoff and existing direct-path behavior.

### Phase 2 — evaluation and evidence

- Extend the existing campaign harness to adaptive strategies and matched compute budgets.
- Persist request/strategy/evaluator provenance and actual usage with the same evidence conventions already used for engine benchmarks.
- Add deterministic code execution and evidence-grounded verifier adapters behind explicit sandbox and privacy boundaries.

### Phase 3 — calibrated evaluators and better policies

- Build domain-specific PRM datasets from search-generated partial states and independent outcomes.
- Measure calibration and OOD performance; pin evaluator versions; fail closed to uncalibrated behavior when evidence is stale or outside scope.
- Train difficulty and marginal-value estimators only after labelled data exists.

### Phase 4 — safe policy optimization

- Replay held-out workloads offline; compare rule policies and learned alternatives.
- Shadow, canary, promote and rollback policy versions using explicit quality, latency, cost and reliability gates.
- Extend `optimize` to additional engine families only where their capabilities and evidence support equivalent decisions.

### Phase 5 — model-internal recurrence research

Gateway-level repeated endpoint calls are not hidden-state recurrence. True latent recurrence needs a model-specific engine capability for forward-pass/state access, recurrent layer-window execution, cache-correct state handling, halting/damping and architecture-aware MoE support. Treat this as a separate backend project with model-specific benchmarks.

## Explicit non-goals for PR #17

PR #17 should deliver a correct, bounded request-time adaptive foundation plus this integration plan. It should not be described as delivering a trained PRM, a complete evidence-grounded verifier, graph MCTS, learned online policy optimization, universal streaming/media support or model-internal latent recurrence. Those require separate implementation and acceptance gates.
