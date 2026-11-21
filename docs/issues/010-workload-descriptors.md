Title: Normalized workload descriptors across API dialects

Priority: P1 | Wave: 4 | Backlog: B06 | Effort: M | Depends: -

## Problem

The admission queue classifies requests into WorkClass {Interactive, Tool,
RawLong} (crates/blazar-gateway/src/queue.rs) with WFQ by API-key weight and
EDF via the x-blazar-deadline-ms header, honored across the OpenAI, Ollama,
and Anthropic surfaces. Classification is coarse: media, embedding, and
long-prefill work all collapse into the same three classes. There is no
workflow header and none is needed: continuity already rides the pool
affinity machinery in crates/blazar-gateway/src/pools.rs.

## Approach

1. WorkloadDescriptor: kind (interactive chat, agent continuation, tool
   execution, long prefill, batch generation, embedding, reranking, realtime
   audio, image generation, video generation), priority, optional deadline,
   resource class, optional workflow identity.
2. Derive descriptors from endpoint, request structure, modality, streaming
   behavior, deadline, and tool declarations.
3. Keep the WFQ/EDF mechanics; do not fan out into per-kind queues until a
   measured behavior difference justifies one.
4. Workflow continuity builds on pool affinity: stable identity across
   calls, turn/stage awareness, KV-affinity information. No raw prompts in
   telemetry.

## Acceptance criteria

- [ ] Descriptor derived consistently across the OpenAI, Ollama, and Anthropic lanes (tests)
- [ ] Existing priority/fairness guarantees pinned by tests
- [ ] Media jobs cannot starve interactive chat under admission control (test)
- [ ] Agent continuations retain pool affinity across turns (test)
