Title: Effective-runtime configuration verification

Priority: P0 | Wave: 2 | Backlog: B02 | Effort: M | Depends: 005

## Problem

A requested argv is not proof the engine accepted or applied it. Engines
differ in what they expose after spawn (llama.cpp serves a props surface,
sglang serves server info, others expose little). Today Blazar verifies
readiness after spawn; it does not verify the effective configuration
against the plan.

## Approach

1. Per-engine introspection adapter where the engine exposes effective
   config; an explicit not-observable disposition where it does not.
2. After spawn: compare observed effective config against the plan; compare
   observed memory against the capacity estimate.
3. Mismatches and degraded capabilities feed back into the admission/teach
   surfaces as actionable diagnostics, never silently.

## Acceptance criteria

- [ ] Introspection adapter per engine that supports one
- [ ] Not-observable engines carry an explicit disposition
- [ ] Observed-vs-estimate memory delta recorded in receipts
- [ ] Mismatch produces an actionable diagnostic
- [ ] At least one live verification per engine lane
