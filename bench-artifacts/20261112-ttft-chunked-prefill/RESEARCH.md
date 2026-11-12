# TTFT research notes — modern no-quality-loss prefill acceleration

Date: 2026-11-12. Identifiers resolved live against the arXiv API (export.arxiv.org) — not cited from memory.

## Bench-verified on this box (receipt in README.md)

- **Chunked prefill** (`sglang.chunked_prefill_size = 2048`): -24.7% uncached TTFT on a ~10K-token prompt (35.1s -> 26.4s), zero quality loss — scheduling knob, sampling untouched. Warm path rides the radix prefix cache (~0.04s) regardless of chunk size.
- **Prefix/radix cache**: already default-on and typed; warm TTFT is cache-hit dominated.

## Cited, research-grade (not upstream production flags today)

| Paper | arXiv (resolved) | Idea | Quality posture |
|---|---|---|---|
| CacheBlend: Fast LLM Serving for RAG with Cached Knowledge Fusion | 2405.16444v3 | Reuse per-document KV from cache for multi-doc RAG prompts; re-attends only cross-document fusion positions (~15% tokens) instead of full re-prefill | Training-free, loss-bounded by design |
| Speculative Prefill: Turbocharging TTFT with Lightweight and Training-Free Token Importance Estimation | 2502.02789v2 | Draft-model token-importance guess -> prefill skips low-importance token positions; big TTFT cuts at minor loss tradeoffs (tunable) | Training-free; quality knob explicit |
| Hydragen: High-Throughput LLM Inference with Shared Prefixes | 2402.05099v2 | Batched shared-prefix decode attention (throughput, not TTFT, but same prefix-economics family) | Lossless for shared-prefix workloads |

Newer siblings surfaced by the same queries (area is active, watch for upstream adoption):
- Decoupled Attention Fusion (2607.21599) — RAG KV reuse.
- CacheClip (2510.10129) — RAG KV reuse.
- Cross-Family Speculative Prefill (2603.02631) — long-context compression via small draft models.

## Excluded after failed resolution

- "FaCache" — no arXiv resolution at query time; not cited (naming may collide with non-indexed work).

## Flag-exposed but environment-gated on this box

- **Two-batch overlap** (`sglang.enable_two_batch_overlap`, typed): multi-GPU MoE/DP-attention throughput path; single-GPU tp=1 cannot host attn_dp >= 2 (upstream asserts `tp_size % attn_dp_size == 0` — now also a Blazar config rule). Not a general TTFT knob.

## Production answer today (vLLM/SGLang)

Chunked prefill + prefix cache (+ PD prefill scaling, which Blazar now supervises via the sgl-router lane). The research-grade items above are the gradual-adooption pipeline to watch.
