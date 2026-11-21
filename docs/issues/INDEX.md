# Engineering issue drafts: capability truth, planner, and evidence waves

Status: FILED AND CLOSED with evidence (2026-11-21) - tracker issues #5-#15
on github.com/santanu20/blazar. Every factual claim was live-verified
against the tree on 2026-11-20 (31-point harness covering paths, symbols,
behavior claims, and absence claims); each close comment cites its
implementation and receipt artifacts. The vllm tools gap called out under
issue #7 (benchmark lane completion) was fixed afterwards - see
bench-artifacts/2026-11-21-vllm-tools-fix/.

Wave map (dependency order):

| File | Title | Pri | Wave | Backlog | Depends |
|---|---|---|---|---|---|
| 001-engine-capability-inventory.md | Complete engine capability inventory for all engine kinds | P0 | 1 | B01 | - |
| 002-vllm-kokoro-capability-certification.md | Add vLLM and Kokoro capability certification | P0 | 1 | B01 | 001 |
| 003-engine-benchmark-lane-completion.md | Benchmark lane completion for all supported engines | P0 | 1 | B04 | 002, 004 |
| 004-request-stage-timing.md | Request-stage timing in the gateway | P0 | 1 | B05.1 | - |
| 005-canonical-execution-plan.md | Introduce the canonical ExecutionPlan | P0 | 2 | B02 | 001, 003 |
| 006-run-uses-canonical-planner.md | Route blazar run through the canonical planner | P0 | 2 | B02 | 005, 007, 008 |
| 007-automatic-feature-policy.md | Centralized automatic feature policy | P0 | 2 | B03 | - |
| 008-effective-runtime-verification.md | Effective-runtime configuration verification | P0 | 2 | B02 | 005 |
| 009-gateway-saturation-overhead.md | Profile and reduce gateway saturation overhead | P1 | 3 | B05 | 004 |
| 010-workload-descriptors.md | Normalized workload descriptors across API dialects | P1 | 4 | B06 | - |
| 011-kv-interop-certification-matrix.md | Cross-engine KV interoperability certification matrix | P1 | 5 | B07 | - |

Deferred on purpose (do not file): new inference engine work, a distributed
KV service, RL-based scheduling, wholesale promotion of upstream engine flags
to first-class config, workspace reorganization crates, automatic speculative
or parallelism activation without evidence, federation expansion before
single-node placement is measured.

Filing, after review:

    gh issue create --title "<title>" --body-file docs/issues/<file>.md

Conventions: references cite symbols and repo-relative paths, never line
numbers (the tree moves constantly).
