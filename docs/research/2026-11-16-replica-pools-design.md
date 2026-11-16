# Replica Pools — Adaptive Multi-Member Model Routing (Design)

Date: 2026-11-16
Status: design, pre-implementation (P0 of the phased plan below)
Scope decision: recorded in `dp-20261116-replica-pools-scope` — design doc first,
implementation waits for a clean config tree and for stream-true telemetry to cook.

## Goal

Give a model a logical service identity that can be served by multiple replicas
(local instance + remote peers), and route each request to the member with the
best expected outcome — without giving up the two defaults that make routing
today trustworthy: local ownership wins, and explicit model pins are
authoritative.

Motivation: today's choice is binary. A model either exists locally (request
served locally, federation never consulted) or it does not (request goes to the
best-scoring remote). Failover chains add *ordered* fallback, not adaptive
selection. Nothing load-balances across a local instance and its remote
replicas, and nothing weighs queue pressure against warm state when picking a
member.

## Non-goals

- Not a Kubernetes-style scheduler or a distributed KV transfer layer — members
  are independent serving stacks.
- Not replacing `[[failover]]` chains; pools and chains compose (a pool member
  can itself resolve through a chain).
- No automatic model replication or placement yet — membership is declared, not
  inferred (placement suggestions are a P2 stretch).
- No quality-blind racing of members (contrast `best_of`, which already exists
  and keeps its own lane).

## What exists today (anchors)

| Capability | Anchor | Notes |
|---|---|---|
| Remote peer scoring | `crates/blazar-gateway/src/remotes.rs` `PeerScore` | Priority = field order: `tier` (warm=0 < unknown=1 < cold=2; unknown outranks cold because unknown may be warm, cold certainly needs a load) → `est_wait_ms` → `free_vram`. |
| Queue-wave wait estimate | `remotes.rs` `score_peer` | `ceil((Q_peer + Q_gw) / slots) * TTFT_EWMA`, TTFT defaulting to 750 ms until a success teaches the EWMA. Pinned by `unit__score_peer__tier_dominates_wait_numbers` and the queue-wait tests. |
| Peer health signals | `remotes.rs` `RemoteHealth` | `in_flight`, `ttft_ewma_ms` (taught by first success), cached capacity snapshots (`insert_capacity`, bounded by `REMOTE_CAPACITY_CAP`). |
| Routing introspection | `crates/blazar-gateway/src/federation.rs` `route()` | Admin-tier gated; snapshots cached peer signals once before scoring. This is the observability endpoint, not the request path. |
| Request-path ownership rule | federation request path | Local row exists → serve locally; federation only consulted for non-local models. The pool seam (below) is exactly this branch. |
| Member warm-up pattern | `federation.rs` `warm()` | `warm_scope_gate` + `ensure_with_admission(Priority::Low, WorkClass::Interactive)` + resident `ps()` json. Reusable per member. |
| Array-of-tables config precedent | `[[failover]]`, `[[keys]]`, `[[mcp]]` in `blazar-core/src/config.rs` | Pool config follows the same serde shape; `Config::default()` precedent `failover: Vec::new()`. |
| Stream-true service times | `crates/blazar-gateway/src/otlp.rs` `spanned` | Since 85c226c spans end at stream completion — the future source of real per-member service-time distributions. |
| Config hot-reload | `state.cfg()` (`Arc<Config>` shared with the supervisor) | Pool membership changes ride the existing reload path. |

## Signal availability (the honest table)

| Signal | Status | Source |
|---|---|---|
| Warm/cold/unknown tier | exists today | `PeerScore.tier` |
| Queue depth (peer-reported in-flight + local leases) | exists today | `score_peer` inputs |
| Slots / concurrency capacity | exists today | cached capacity snapshots |
| TTFT EWMA (remote) | exists today | `RemoteHealth.ttft_ewma_ms` |
| Free VRAM | exists today | `PeerScore.free_vram` tie-break |
| Prompt-token count (prefill cost) | needs plumbing | body sniff already extracts `model` on metered lanes (`request_log`); token estimate needs a cheap tokenizer heuristic (chars/token ratio) first |
| Expected output tokens (decode cost) | needs plumbing | config-declared default per pool until measured |
| Cache/prefix affinity | needs telemetry | per-member prompt-cache classification exists for warm/cold TTFT split (`state.rs` metrics block); member-keyed affinity needs a new counter |
| True stream service time | cooking | wave-3 spans; needs a consumer to fold into per-member EWMA |
| Telemetry freshness | needs plumbing | timestamp on capacity snapshots; degrade to `unknown` tier when stale |

## Proposed config schema

```toml
[[model_pools]]
name = "coding"                 # logical service identity clients address
strategy = "adaptive"           # see Strategies; default "ordered" (failover-like)

[[model_pools.members]]
peer = "local"                  # "local" or a configured remote label
model = "qwen3-coder"           # the concrete model row on that member

[[model_pools.members]]
peer = "workstation"
model = "qwen3-coder"
```

Semantics:

- Addressing `coding` (pool name) routes adaptively; addressing `qwen3-coder`
  directly keeps today's behavior unchanged (local ownership wins, pins
  authoritative).
- `strategy = "ordered"` degenerates to a failover chain over members — the
  migration-safe default.
- A pool name colliding with a real model row is a config-parse refusal (same
  class of teaching error as existing validation families).

## Resolution seam

In the request path where model resolution happens today (the local-row-exists
branch of federation routing): a resolved pool name expands to its member list;
each member maps to either the local serving lane or a remote forward exactly as
a non-local model does today. No new transport, no new forwarding code — the
pool is a *member enumeration + scoring* layer above both existing lanes.

Scoring reuses `PeerScore` semantics verbatim for remote members; the local
member gets an equivalent score synthesized from the same ingredients the
supervisor already exposes (resident/warm state, queue depth from
`state.queue.depth()`, capacity from the local ps() lane). Local and remote
members then compete in one ordered list.

## Strategies

| Strategy | Rule | Signals required |
|---|---|---|
| `ordered` | first healthy member (failover semantics) | existing |
| `least_wait` | min `est_wait_ms` | existing |
| `warm_first` | tier, then wait | existing |
| `adaptive` | least expected completion = wait + expected service time; warm-preference tie-break; hysteresis window to prevent thrash | existing + declared defaults; improves as P2 telemetry lands |

`adaptive` must degrade gracefully: missing signals fall back to declared
defaults and the `ordered` result — never to an error.

## Phasing

- **P0 (this doc)** — design recorded; no code.
- **P1 — pool-aware fallback routing** — schema + resolution seam + `ordered`
  and `least_wait` strategies using ONLY existing signals; pool route()
  introspection reuses the admin gate; `blazar route <pool>` explain output.
  Everything in P1 is testable against stub peers with synthetic
  `RemoteHealth` inputs, same pattern as the `score_peer` unit tests.
- **P2 — telemetry-fed adaptive** — per-member service-time EWMA consumed from
  wave-3 spans; cache-affinity counter; prompt-size estimate; staleness decay;
  hysteresis + anti-thrash; bench lane comparing pool vs single-member under
  the existing benchmark harness (concurrency ladder, TTFT/tail columns).

## Test plan (P1 preview)

- Unit: pool parse (collision refusal, member validation, default strategy);
  member scoring parity (local-vs-remote synthesis equal-shape `PeerScore`);
  `ordered`/`least_wait` selection matrices; hysteresis window logic (P2).
- Integration: request to pool name routes to highest-scoring stub member;
  member failure falls to next (compose with existing failover tests).
- Live: two-sandbox lane — local member + stub remote; concurrency burst shows
  queue-aware switching in `blazar why` output (the `why` explanation names the
  chosen member and the score ingredients).

## Risks / open questions

- **Thrash**: adaptive switching under bursty load oscillates. Mitigation:
  hysteresis window + sticky tier preference; measure in the P2 bench lane.
- **Stale signals**: cached capacity can lie (eviction between snapshot and
  use). Mitigation: freshness timestamp → `unknown` tier decay; already
  partially modeled by the unknown-beats-cold tier order.
- **Config surface growth**: one more array-of-tables; acceptable next to
  `[[failover]]`, and pools default to empty = zero request-path cost.
- **Open**: should a pool member's *health* failure temporarily demote it
  below `ordered` position rather than skip (fail-fast vs fail-slow)? P1 ships
  skip (failover parity); revisit with bench data.
- **Open**: prompt-cache affinity across heterogeneous engines — semantics of a
  "shared prefix" across engines differ; P2 starts local-member-only affinity.
