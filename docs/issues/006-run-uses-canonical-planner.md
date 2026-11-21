Title: Route blazar run through the canonical planner

Priority: P0 | Wave: 2 | Backlog: B02 | Effort: L | Depends: 005, 007, 008

## Problem

The run path is the most battle-tested lane in the repo (scripts/validate.py
end-to-end suite). Moving it onto ExecutionPlan must preserve every current
default routing decision while gaining fallback and verification behavior
that today does not exist.

## Approach

1. Golden tests first: capture current engine selection + argv for a model
   matrix before any switchover.
2. Fallback policy: classify startup failure (recoverable or not); retry only
   recoverable failures; evaluate the next compatible engine; recompile its
   engine-specific profile; start and verify; record the fallback reason in
   the plan.
3. Never silently change a user-pinned engine; never fall back to a
   different model or quantization without explicit policy allowing it.
4. Runtime verification per issue 008 gates "started" against "started as
   planned".

## Acceptance criteria

- [ ] Golden parity tests green before and after switchover
- [ ] Fallback chain implemented with recorded reasons
- [ ] Pinned engine and model/quant never silently overridden (tests)
- [ ] scripts/validate.py end-to-end suite green
- [ ] Fresh-install path: pull a model, run it, no manual engine/slot/kv/spec
      choices required
