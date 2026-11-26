# Blazar: SOTA Improvement Feasibility and ROI Research

Date: 2026-11-23
Status: research, read-only. Nothing here is authorisation to change code.
Each item needs its own approval and its own implementation run (pin test
first, live validation, receipt).
Baseline: Blazar 0.34.3 at commit `00c15bd`. Upstream state verified 2026-11-23.

## 1. Bottom line

Blazar's measured strength is orchestration on one box: tail latency under
load, idle wake, concurrency scale-out and cross-engine KV reuse all have
receipts (sections 3.1 and 3.2). The ollama comparison rows were measured on ollama
0.33.3 (current: v0.40.17). They show wins on tail latency under load, idle
wake and concurrency, a tie on the seven-suite quality score, and losses on
ITL p99 and cold time-to-first-token.

Blazar is not yet SOTA in the strict sense because six things are missing or
unproven:

1. **Transport.** The TLS listener speaks HTTP/1.1 only and never sets
   `TCP_NODELAY`. The plain listener does both.
2. **Admission and queueing.** Prompt size is estimated as bytes/4, so the fit
   gate is wrong on CJK, image-heavy and tool-heavy prompts. The queue has no
   preemption.
3. **Prefix routing.** Blazar routes on a hash affinity. sgl-router, NVIDIA
   Dynamo and AWS SageMaker now route on a prefix index fed by engine KV events.
4. **Hardware.** The GPU probe is NVIDIA-first and leaves an empty device list
   on other vendors. There is no Apple Silicon receipt, while ollama has run
   MLX by default there since v0.40.0.
5. **Proof.** Greedy output through the gateway matches the engine 5/20 times
   and the stated cause is unproven. 37 store reads treat "store failed" as
   "not found". The project's own documents carry 15 contradictions, including
   three different overhead headlines.
6. **Low-VRAM MoE.** SGLang 0.5.23 ships Paged Experts (opt-in) for MoE models
   larger than VRAM. Blazar does not expose it. Out-of-memory is one of
   ollama's biggest keyword clusters (1,275 issues) and #11772 asks for this
   exact feature (74 reactions).

Raw kernel speed belongs to the engines, so SOTA here means: the best measured
single-box outcome across engines, hardware and operating systems, with
receipts no local rival publishes.

**Do first** (all on the existing test bed, see section 8): prove the greedy
divergence (F1), re-run the ollama head-to-head on a current ollama and add a
per-request overhead lane (M1), make store failures fail loudly (F3), fix the
TLS listener (T1), count tokens exactly (S1).

## 2. Method, evidence tiers and scoring

### 2.1 What was examined

- **Internal, documents:** all 46 files in `docs/research/`, `BENCHMARK.md`,
  `docs/benchmarks.md`, `docs/issues/INDEX.md`, `bench-artifacts/INDEX.md` and
  the `CHANGELOG.md` entries down to 0.30.1.
- **Internal, code:** symbol-level audit of `crates/blazar-gateway`,
  `crates/blazar-runtime` and `crates/blazar-core`. Absence claims are ripgrep
  zero-hit checks re-run at `00c15bd`. No build was run.
- **External, first-hand:** GitHub API for releases, issues, pull requests and
  READMEs of ollama, llama.cpp, vLLM, SGLang, Dynamo, llm-d and others.
  arXiv API to resolve paper identifiers before citing them.
- **External, secondary:** web search for third-party benchmarks and posts.

### 2.2 Evidence tags

| Tag | Meaning |
|---|---|
| S1 | Blazar documents and benchmark notes in this repository |
| S2 | Blazar code, read directly at `00c15bd` |
| S3 | ollama GitHub (releases, issues, docs) |
| S4 | llama.cpp GitHub |
| S5 | vLLM v0.34.0 release notes |
| S6 | SGLang v0.5.23 release notes, sgl-router README, pull requests |
| S7 | NVIDIA Dynamo README, llm-d, gateway-api-inference-extension |
| S8 | Repository metadata (stars, licence, release dates) |
| S9 | Secondary web source (blog or comparison article) |
| S10 | Paper whose arXiv identifier resolved on 2026-11-23 |

Performance numbers from S5, S6, S7 and S9 are the publisher's claims. None
were reproduced here.

### 2.3 Scoring rubric

Feasibility and ROI are ordinal scores from 0 to 10. They are judgements
against the rubric below, not measured gains. No percentage benefit is
claimed anywhere in this document.

- **Feasibility** = architecture fit + upstream capability to ride + can be
  validated on hardware the project already has.
- **ROI** = user-visible gain x breadth of users who feel it, divided by effort.
- **Effort:** S = under a week, M = two to four weeks, L = one to three months
  or needs hardware the project does not have.
- **Confidence:** High = evidence in this repository or on its test bed.
  Medium = upstream evidence only. Low = cause unknown.

### 2.4 Limits of this research

- One test bed: 8 GiB VRAM, 16 GiB RAM, Linux. MoE, 8B-model and HiCache
  lanes could not be exercised beyond what earlier campaigns recorded.
- No Apple, AMD or Intel hardware. Section 7 phase 2 is unvalidated.
- Six arXiv identifiers could not be confirmed (no record retrieved or an
  unrelated paper, section 12) and are not cited.
- ollama issue counts are keyword substring hits over 11,664 issues. They
  show where the volume is, not how many users are affected.

## 3. Where Blazar stands today

### 3.1 Measured position (S1 unless noted)

Campaign build: llama.cpp `b12606`-cuda, qwen2.5-0.5b-instruct q4_k_m, Blazar
0.34.3, from `BENCHMARK.md`.

| Measure | Through Blazar | Direct engine | ollama | Read |
|---|---|---|---|---|
| Decode, tokens/s | 330.8 | 338.0 | 341.0 (row marked not comparable) | 1.8 to 2.1% below direct |
| Prefill, tokens/s, cold | 18,888.7 | 20,010.9 | n/a | 5.6% below direct |
| Prefill, tokens/s, cached | 73,216.2 | 91,594.3 | n/a | 20% below direct |
| TTFT p50, ms | 8.0 | 6.1 | n/a | +1.9 ms |
| ITL p99, ms | 4.4 | 3.9 | n/a | +0.5 ms |
| Sleep to wake, 0.5B | 27 ms | n/a | 1,753 ms | 65x faster |
| TTFT p99 at C=4, 9B | 554 ms | n/a | 15.9 s | 29x lower |
| Greedy match vs direct, 20 prompts | 5/20 | 20/20 | 15/20 run to run | open, see F1 |

The cached-prefill gap is the one row where Blazar trails by a wide margin.
The direct row uses one slot of 16,384 context and the gateway row two slots
of 32,768, so they are not like-for-like. It is worth one matched rerun.

Other recorded results (S1): reshape from 2 to 16 slots took 868 to 1,831
tokens/s with TTFT p50 404 to 46 ms. Gateway over direct throughput across the
C=1 to C=16 ladder ranged 0.94 to 1.20 in the latest campaign. An earlier
campaign measured C=2 at 0.792 and it is still unexplained (F2). Cross-engine
KV reuse is live: llama.cpp to vLLM reported 320 of 343 tokens cached.

The ollama rows above were taken on ollama 0.33.3. ollama is now 0.40.17.
Every ollama comparison in the repository is stale (M1).

### 3.2 Capability position (S1, S2)

| Area | Blazar today | Rival position | Verdict | Evidence |
|---|---|---|---|---|
| API | 139 routes: OpenAI, Ollama, Anthropic, realtime websocket | ollama: OpenAI, Anthropic subset, System One decision models | Leads. No Gemini dialect. `n>1` is not handled | S2, S3 |
| Scheduling | Priority, work class, deadline, fair share per key, reserved capacity, bounded wait | ollama: not assessed here | Differentiator | S1, S2 |
| Speculative decoding | `spec=auto` with saturation governor and acceptance receipts | ollama: no user control (issue 5800 open since 2024) | Leads | S1, S3 |
| KV reuse | Slot banks, cross-engine translation, session pins | ollama: not assessed here | Differentiator | S1, S2 |
| Idle management | Ready, Sleeping, Evicted ladder with reaper | ollama: `keep_alive` only | Leads | S1, S2 |
| Media | TTS, ASR, CLIP and SigLIP embeddings, 39 diffusion families | ollama: speech models since 0.40.9, no image generation | Leads | S1, S3 |
| Engine install | NVIDIA, AMD, Intel and Apple assets from upstream builds | ollama: MLX default on Apple | Parity on paper, no AMD, Intel or Apple receipt | S2, S3 |
| Security | Scoped keys, budgets, MCP allowlists, signed capability registry | ollama: not assessed here | Differentiator | S1, S2 |
| Gateway transport | HTTP/1.1 on TLS, no compression, system allocator | Bifrost and LiteLLM market per-request overhead in microseconds | Gap | S2, S9 |
| Agent tooling | `blazar launch` points any CLI at the daemon, optional warm-up and key | ollama: 27 documented integrations, agent-specific launch behaviour | Behind on convenience | S2, S3 |
| Tracing | Own trace id, hand-rolled OTLP export | W3C traceparent is the norm | Gap in interop | S2 |

### 3.3 Engine currency

| Component | Blazar runs | Upstream latest | Delta | Evidence |
|---|---|---|---|---|
| llama.cpp | `b12606` (campaign build) | `b12651`, 2026-11-23 | 45 builds. 3 of the 45 commits are server changes | S1, S4 |
| vLLM | 0.34.0 | 0.34.0, 2026-11-16 | Current | S1, S5 |
| SGLang | 0.5.23 | 0.5.23, 2026-11-02 | Current | S1, S6 |
| mlx-lm | 0.32.2 | 0.32.2, 2026-10-16 | Current | S1, S8 |
| mistral.rs | 0.9.5 in campaigns, 0.9.3 in the pin note | 0.9.5, 2026-10-19 | Pin note and campaigns disagree | S1, S8 |
| ollama (comparison only) | 0.33.3 in comparison rows | 0.40.17, 2026-11-10 | Comparison rows stale | S1, S3 |

### 3.4 Internal weak spots confirmed in code at `00c15bd` (S2)

| Weak spot | Where | Proof |
|---|---|---|
| TLS listener has no ALPN, so HTTP/1.1 only; no `TCP_NODELAY` | `crates/blazar-gateway/src/lib.rs`, `axum_server::from_tcp_rustls` | `alpn` has 0 hits in `crates/`. `set_nodelay` appears once, in the plain-listener branch. The listener comment puts the Nagle penalty at 40 to 60 ms on first token |
| Admission estimates tokens as bytes/4 plus 1,500 per image | `crates/blazar-gateway/src/preflight.rs` | `IMAGE_TOKEN_EST = 1500`, `(bytes / 4).max(1)`, tools counted as bytes/4 |
| No preemption | workspace | `preempt` has 0 hits |
| No W3C trace context | workspace | `traceparent` has 0 hits |
| No response compression | workspace | `CompressionLayer` has 0 hits |
| System allocator | workspace | no `mimalloc`, `jemalloc` or `global_allocator` |
| No `/v1/tokenize` route, no Gemini dialect | workspace | 0 hits each |
| Non-NVIDIA GPUs can be invisible | `crates/blazar-runtime/src/probe.rs` | Comment: "non-NVIDIA boxes without a census keep the empty list" |
| Store failures read as absent | `crates/blazar-core/src/store.rs` and callers | 37 `with_store` chains have an `.ok().flatten()` within four lines (a text match, an upper bound until each is read). 68 such sites workspace-wide. `docs/research/2026-10-31-silent-fallback-audit.md` counted 44 or more |
| No prefix index in the gateway | gateway and runtime | The four `radix` mentions are engine flag names and comments |
| No hugepage or NUMA handling | workspace | 0 hits |

## 4. External SOTA ladder (2026-11-23)

| Dimension | Current reference | Blazar | Verdict | Evidence |
|---|---|---|---|---|
| Engine kernels | llama.cpp `b12651`; vLLM 0.34.0 (Model Runner V2 only, claims up to 1.95x decode); SGLang 0.5.23 megakernel and monokernel plugins (claims up to 44% and 75% faster single-user decode on GB300 and MI350X) | Rides upstream builds | Parity by design. Not a lane to enter | S4, S5, S6 |
| Gateway overhead | Bifrost README claims 11 microseconds per request at 5,000 RPS | 1.8% below direct decode, +1.9 ms TTFT. No per-request figure | Not comparable today. Add the metric | S1, S2, S9 |
| Transport | HTTP/2 on TLS everywhere | HTTP/1.1 on TLS, Nagle on | Gap, cheap to close | S2 |
| Scheduling | vLLM 0.34.0 opt-in cache-aware admission (#54625) and NUMA binding (#61258). Llumnix (2406.03243) and a 2026 preemption study (2611.20106) | Priority, deadline and fairness done. No preemption. Token counts approximate | Leads ollama. Behind research frontier | S2, S5, S10 |
| Prefix and KV-aware routing | sgl-router (load, global prefix match, session affinity). Dynamo KV-aware routing (claims 2x TTFT). SageMaker `KV_AWARE` strategy. llm-d router. Preble paper (2407.00023) | Hash affinity plus `KvRouteScore` | Behind | S1, S2, S6, S7, S9, S10 |
| Low-VRAM MoE | SGLang Paged Experts (claims up to 30x faster than CPU offload). ktransformers (claims 3 to 28x). HF expert-streaming post | RAM ledger and engine offload knobs. Forced MoE reshape cost 25 to 29% before the decode-scaling gate | Behind. Requested by ollama users (#11772, 74 reactions) | S1, S6, S9 |
| Speculative decoding | vLLM: suffix decoding on MRV2 without Arctic Inference (#60339), plugin API (#63205). SGLang: EAGLE3, DFLASH. 2026 papers on content-chosen drafters (2611.05611) and narrower drafts (2611.23835) are engine-side | `spec=auto`, governor, race receipts. llama.cpp EAGLE3 and MTP | Level with vLLM in mechanism, ahead in policy | S1, S5, S6, S10 |
| Cold start and model load | Dynamo ModelExpress (claims 7x faster startup). vLLM checkpoint and restore endpoints (#62570). ServerlessLLM paper (2401.14351) | 27 ms sleep to wake. 0.82 s orchestration on a 4.23 s cold 9B load. No size check at spawn | Leads ollama. Behind datacenter class | S1, S2, S5, S7, S10 |
| API breadth | ollama: Anthropic subset, System One. vLLM Rust frontend: Responses API and `n>1` (#65625) | 139 routes. No Gemini, no `n>1` | Leads, two gaps | S2, S3, S5 |
| Agent and IDE ecosystem | ollama: 27 integration guides, `ollama launch claude` with deferred MCP tool loading, `--restore`. LM Studio: MCP client | MCP loop with per-key scopes. `blazar launch` is generic | Behind on convenience | S2, S3, S9 |
| Observability | W3C traceparent and OpenTelemetry | Own trace id, OTLP over HTTP JSON | Gap in interop | S2 |
| Hardware breadth | ollama issue volume: Intel GPU 2,803 hits, AMD 2,063, ROCm 1,022. OpenVINO and NPU requests open | Asset matrix covers AMD, Intel, Apple. Probe is NVIDIA-first. LiteRT-LM host lane exists | ollama's weakest area is Blazar's opening | S2, S3 |
| Packaging trust | ollama open: licence notices (#3185), digest mismatch (#941), installer checksums (#19787) | Release sha256 verified by installer, whole-file sha256 at pull, range resume | Leads, small gaps | S2, S3 |
| Datacenter orchestration | Dynamo, llm-d 1.0.0, KServe, AIBrix | Out of scope by project decision | Not a target | S1, S7 |

One third-party signal on visibility: a 2026-11-20 comparison of 36 engines
and 7 routers did not show Blazar in the section retrieved (S9, low
confidence). That is a distribution problem, not a feature problem.

## 5. Feasibility x ROI matrix

Scores are rubric judgements (section 2.3). Effort and confidence are defined
there. Evidence tags are in section 2.2.

| ID | Improvement | Feasibility | ROI | Effort | Confidence | Evidence |
|---|---|---|---|---|---|---|
| F1 | Prove or fix greedy transparency (5/20) | 8 | 8 | S-M | High | S1 |
| F2 | Root-cause the C=2 throughput dip (0.792) | 7 | 6 | S | High | S1 |
| F3 | Store failures fail loudly (37 sites), pin per site | 9 | 6 | S | High | S1, S2 |
| F4 | Close the `mlx_signal` routing item with an end-to-end repro | 9 | 3 | S | High | S1, S2 |
| F5 | Reconcile 15 document contradictions, one overhead headline | 9 | 5 | S | High | S1 |
| F6 | Root-cause the log-sink freeze | 5 | 5 | M | Low | S1 |
| M1 | Head-to-head vs ollama 0.40.17, plus per-request overhead lane | 9 | 8 | S | High | S1, S3 |
| T1 | TLS: ALPN `h2` + `http/1.1`, nodelay acceptor | 9 | 6 | S | High | S2 |
| T2 | Accept W3C traceparent, emit it to OTLP | 9 | 5 | S | High | S2 |
| T3 | Allocator A/B, adopt only on a receipt | 8 | 4 | S | Medium | S2 |
| S1 | Exact token admission, `/v1/tokenize`, `/v1/detokenize` | 7 | 7 | M | High | S2 |
| S2 | Per-backend admission floor table from the hardware census | 8 | 6 | S-M | Medium | S1 |
| S3 | Queue preemption for tool and high classes, opt-in | 5 | 6 | M | Medium | S2, S10 |
| S4 | Global prefix index fed by engine KV events | 5 | 6 | M-L | Medium | S2, S6, S10 |
| H1 | Vendor-neutral GPU census | 8 | 7 | M | High | S2 |
| H2 | Apple Silicon receipts and parity vs ollama MLX default | 4 | 8 | M, needs hardware | Low | S1, S3 |
| H3 | AMD ROCm and Intel SYCL lane receipts | 3 | 7 | L | Low | S2, S3 |
| H4 | NPU decision: LiteRT-LM receipt, OpenVINO research | 3 | 5 | L | Low | S3, S8 |
| E1 | Expose SGLang Paged Experts with a hardware-adaptive policy | 5 | 7 | M | Medium | S6 |
| E2 | Engine currency: llama.cpp `b12651`, vLLM `n>1` and suffix decoding | 8 | 5 | S-M | High | S4, S5 |
| E3 | `blazar launch` agent presets and dry-run self-test | 8 | 6 | M | High | S2, S3 |
| E4 | Gemini `generateContent` dialect behind a flag | 7 | 5 | M | Medium | S2 |
| E5 | Release hygiene: licence notices, installer checksums, multi-file GGUF import | 9 | 6 | S-M | High | S3 |

## 6. What each item is and why it ranks there

**F1 Greedy transparency.** Direct engine matches itself 20/20. Through the
gateway at two slots of 32,768 context it is 5/20. `BENCHMARK.md` attributes
this to multi-slot batching numerics. An October note measured 14/20 and a
carried-over note claims byte identity at one slot. Three statements, no
experiment. Decisive test on the existing box: per-token hash of 20 greedy
prompts through the gateway at one slot versus the direct engine at one slot.
If that is 20/20 the cause is the engine's batched sampler and the fix is a
permanent single-slot parity pin plus a documented boundary. If it is not, the
divergence is in the gateway and it bisects over slot restore, KV bank and the
speculation governor.

**F2 C=2 dip.** `docs/research/2026-10-31-recursive-self-improvement-research.md`
records gateway over direct throughput of 0.792 at C=2 against 1.135, 1.017 and 0.986 at C=1, 4 and
8, and calls it unexplained. Stage timing already records queue wait, admission
and dispatch. One instrumented run at C=2 with slot count pinned should show
whether it is the two-slot boundary, a reshape mid-run, or the admission gate.

**F3 Store failures.** Up to 37 call chains turn a failed store read into "no
row". A corrupted or locked store then looks like "model not found" or "key
absent". That is the failure mode the project's own rules forbid.
`docs/research/2026-10-31-silent-fallback-audit.md` listed it as P1 on
2026-10-31. Replace each with
explicit error propagation and one pin test per site. Highest certainty on
this list.

**F4 `mlx_signal`.** `docs/research/2026-10-29-deep-dive-implementation-roadmap.md`
lists a routing bug for MLX quantised directories. The code now has an
on-disk layout fallback and a unit pin
(`unit__mlx_signal__token_and_layout_probe_fallback` in `store.rs`), added
two days later. The unit pin covers the token path. One end-to-end pull and
route repro closes the item.

**F5 Document reconciliation.** Fifteen contradictions were found across the
research notes (section 12). The user-visible one is the overhead headline:
about 2% in `docs/benchmarks.md`, 1 to 3% in `BENCHMARK.md`, 5 to 6% in two
October notes. One number, one place, linked from the others.

**F6 Log-sink freeze.** All three log sinks went dark for about ten minutes
under load. A controlled reproduction was negative. Keep the issue open and
attach heartbeat and syscall trace at the next recurrence. Low confidence by
nature.

**M1 Current head-to-head.** Every ollama comparison used 0.33.3. ollama
shipped 18 releases between 0.40.0 and 0.40.17, including MLX-by-default on
Apple and concurrent serving on Apple silicon. Re-run the matched campaign
(same model, quant, sampling, cache state) against 0.40.17 and add a
non-stream per-request overhead lane in microseconds, which is the figure the
gateway market publishes. ollama is a benchmark counterpart only, never a
dependency.

**T1 TLS listener.** Two small changes in `lib.rs`: set ALPN to `h2` and
`http/1.1` in the rustls config, and use axum-server's nodelay acceptor. Pin
with an end-to-end SSE test over TLS asserting the negotiated protocol.

**T2 Trace context.** Read inbound `traceparent`, continue the span in the
existing OTLP exporter, echo it beside `x-blazar-trace-id`. No new
dependency is needed.

**T3 Allocator.** Switching to mimalloc is a two-line change. Adopt it only
if a saturation receipt shows a gain outside run-to-run noise.

**S1 Exact tokens.** The fit gate refuses prompts it mis-measures. Either ask
the child `/tokenize` endpoint (already used by count_tokens) or use the HF
`tokenizers` crate with a small cache. Ship `/v1/tokenize` and
`/v1/detokenize` on the same path. Also makes TTFT and tokens-per-second
per-token instead of per-chunk.

**S2 Admission floor table.** Vulkan-class admission constants are applied to
every backend. Derive per-backend floors from the hardware census the
project already collects.

**S3 Preemption.** A tool call queued behind a long prefill waits up to its
deadline. Preempting the lowest-priority in-flight request for tool and high
classes, behind a flag with a thrash guard. Llumnix (2406.03243) is the
reference for rescheduling requests between instances, and 2611.20106
(2026-11-15) studies preemption for long outputs (S10). Engine support for
mid-sequence pause varies, hence feasibility 5.

**S4 Prefix index.** Today's affinity is a hash of system prompt and
conversation. A global prefix index fed by engine KV events is what sgl-router
and Dynamo do. Preble (2407.00023) is the scheduling reference, and 2611.12304
(2026-11-09) reports agent-adaptive serving at scale (S10). Valuable for agent
workloads with 90% shared prefixes. Needs
engine event surfaces Blazar does not consume yet.

**H1 GPU census.** Add `rocm-smi`, oneAPI level-zero or sysfs, `sysctl` for
Apple and Vulkan device enumeration as fallbacks before the empty list.
Fixture-tested with captured outputs, so it needs no hardware to build.

**H2 to H4 Hardware receipts.** AMD and Intel users file the most ollama
issues. Blazar already installs the right upstream builds but has no receipt
on that hardware and no Metal receipt at all. Each lane stays experimental
until a receipt from that hardware class exists.

**E1 Paged Experts.** SGLang's stated motivation is "laptops and gaming PCs
pair 4 to 16 GB VRAM with 32 to 64 GB RAM". That is Blazar's user. The exact
CLI flag was not located in `server_args.py` at tag v0.5.23, so the first step
is finding it. Experimental until a big-RAM receipt.

**E2 Engine currency.** `b12606` to `b12651` is low impact for serving.
vLLM `n>1` and suffix decoding are two certified-flag candidates. Run quality
and tool-call batteries before and after any bump.

**E3 `blazar launch` presets.** The command exists and is generic: environment
variables, optional warm-up, optional key. ollama's launch knows per-agent
behaviour (deferred MCP tool loading for Claude Code, session restore, thinking
defaults). A preset table plus a dry-run self-test closes most of the gap.

**E4 Gemini dialect.** A translation lane like the existing Anthropic one,
behind a flag, with contract tests. Serves the large body of Gemini-SDK code.

**E5 Release hygiene.** Licence notices in artifacts (ollama issue 3185 has
285 reactions and is open since 2024), installer checksum verification, and a
check that multi-file GGUF import works (ollama issue 5245).

## 7. Phased roadmap and exit gates

| Phase | Items | Exit gate |
|---|---|---|
| 0. Foundations and on-box truth | F1, F2, F3, F4, F5, F6, M1 | Each F item has a pin test or a recorded negative result. One overhead headline. A receipt file for the ollama 0.40.17 head-to-head |
| 1. Polish and scheduling | T1, T2, T3, S1, S2, S3, S4 | TLS SSE end-to-end test green. Traceparent visible in an exported span. Admission exactness test on CJK, image and tool-heavy prompts. Preemption off by default with thrash tests |
| 2. Hardware breadth | H1, H2, H3, H4 | Census fixture tests per vendor. One live receipt per hardware class before any lane leaves experimental |
| 3. Features | E1, E2, E3, E4, E5 | Each feature has an end-to-end test and a receipt. Release checklist carries the E5 items |

Every phase also requires the full test suite green and a live validation run.
Order inside a phase is free. Phase 1 does not wait for phase 0, but F1 and
M1 should start first because every later claim depends on them.

## 8. Do first

| Rank | Item | Why now |
|---|---|---|
| 1 | F1 greedy transparency | A 5/20 match undermines every quality claim until it is explained. The test costs hours |
| 2 | M1 current head-to-head | All ollama comparisons were run on 0.33.3; current is 0.40.17. Also gives the microsecond figure the market compares on |
| 3 | F3 store failures | Silent failure class, highest certainty, small |
| 4 | T1 TLS listener | Tiny change, removes a 40 to 60 ms first-token penalty for every TLS client |
| 5 | S1 exact tokens | Fixes wrong refusals and enables honest per-token metrics |

## 9. ollama open complaints and Blazar coverage

ollama keyword hits are substring matches over 11,664 issues, 2,667 open
(S3). "Blazar state" is from S1 and S2. "Not verified" means this research
did not check.

| Complaint | ollama state | Blazar state | Verdict |
|---|---|---|---|
| Out of memory, 1,275 hits. VRAM not released, 193 hits | Open | Idle ladder evicts to CPU or disk. RAM ledger refuses what does not fit | Solved by architecture |
| MoE weights should offload to CPU, issue 11772 | Open | Engine offload knobs and RAM ledger. Paged Experts not exposed (E1) | Partial |
| Context window handling, issue 1005. `num_ctx`, 799 hits | Open | Fit check before dispatch, capacity and fit endpoints | Solved |
| Slow first token, 897 hits. TTFT p99 at C=4 | Open | 554 ms versus 15.9 s on 9B, recorded | Leads |
| Cold start, 71 hits | Open | 27 ms sleep to wake versus 1,753 ms | Leads |
| Intel Arc, issue 1590. OpenVINO, issue 2169. Intel NPU, issue 3004. AMD NPU, issue 5186. Snapdragon, issue 5360 | Open | Intel and AMD assets install from upstream builds. NPU via LiteRT-LM host lane. No receipts | Opportunity (H1 to H4) |
| Reranking, issue 3368 (290 reactions, open since 2024) | Open | Rerank route and rerank lanes | Solved |
| Multimodal and image embeddings, issues 5304 and 4296 | Open | CLIP and SigLIP lanes, multimodal embeddings | Solved |
| Speculative decoding control, issue 5800 | Open since 2024-07 | `spec=auto`, `tune --spec`, receipts | Leads |
| MCP support, issue 7865 | Open, partly addressed by `ollama launch claude` | MCP loop in the gateway with per-key scopes | Leads |
| `/metrics` endpoint, issue 3144 | Open | Prometheus `/metrics`, `/api/quantiles` | Solved |
| Digest mismatch, issue 941. Download speed, issue 2006. Pull stalls | Open | sha256 at pull, range resume, parallel lane, throttle | Solved |
| Licence notices missing, issue 3185 (285 reactions) | Open | Listed as an open release item | Gap (E5) |
| Installer should verify checksums, issue 19787 | Open | `install.sh` verifies the release sha256 | Solved |
| Multi-file GGUF import, issue 5245 (165 reactions) | Open | Not verified | Gap (E5) |
| Home-directory clutter, issue 228. Autostart toggle, issue 162 | Open | Not verified | Open question |
| GPU energy per response, issue 19789 | Open | Power recorded in campaigns, not exposed per request | Low-priority gap |
| Image generation, issue 786 | Open | 39 diffusion families | Leads |
| Tool parser bugs, issues 19854 and 19852. Stream ends silently, issue 19792 | Open, new | Pass-through lanes are byte-faithful. Translation lanes have a tool-call battery | Leads, keep the battery |
| Structured output with OpenAI SDK, issue 11691 | Closed upstream | Structured-output header. Streaming fence normalisation on hold | Partial |

## 10. What stays parked

Unchanged from earlier project decisions, and this research found no reason
to reopen them:

- A Blazar-owned inference engine or kernel lane. Engines own kernels.
- Kubernetes-class control planes (Dynamo, llm-d, KServe, AIBrix).
- Cross-node prefill and decode disaggregation.
- RL or bandit routing, a distributed KV service, semantic cache on by default.
- HTTP/3. No local rival ships it and no user request was found.

## 11. Risks

| Risk | Mitigation |
|---|---|
| F1 shows the divergence is the engine's batched sampler | Keep a single-slot parity pin, document the engine boundary, do not patch around it upstream |
| Engine bumps change greedy or tool-call behaviour | Run quality and tool-call batteries before and after. Keep a rollback pin |
| Preemption thrashes under load | Opt-in flag, minimum in-flight age, per-key budget, governor switch-off |
| Hardware lanes claimed without receipts | Stay experimental until a receipt from that hardware class exists |
| Prefix index disagrees with engine-internal caches | Index is advisory. Route score stays deterministic and logged |
| Paged Experts flag or semantics change upstream | Pin to SGLang 0.5.23 behaviour with a capability probe |
| Recommendation drift | This document is research. Each item re-plans at its own start |

## 12. Not verified, open questions

- Six arXiv identifiers are not cited. Four returned no record before the
  API rate-limited: 2405.19715, 2406.06282, 2312.12456, 2611.10219 (the MoE
  expert-offloading paper). Two resolved to unrelated papers: 2402.16810,
  2507.07291.
- SGLang Paged Experts: exact CLI flag not located.
- ollama "native go-engine" (issue 15051) is a feature request that closed on
  2026-10-20. Whether an engine shipped is unresolved. Do not claim either
  way.
- Whether ollama 0.40.17 is faster or slower than 0.33.3 on this test bed.
- Multi-file GGUF import, data-directory placement and autostart control in
  Blazar.
- Blazar behaviour on Apple, AMD and Intel hardware.
- Global (non-key) rate limiting in the gateway was not assessed.
- The 15 document contradictions, in brief: q8_0 KV speed (neutral, minus 33
  tokens/s, plus 11 to 15%, in three notes). Greedy identity versus 5/20.
  mistral.rs pin 0.9.3 versus campaigns on 0.9.5. Prefill and decode
  disaggregation "deferred" versus shipped. Kokoro, SGLang `/generate` and
  llama.cpp `/tts` each "rejected" in one note and shipped later. Overhead
  headline 2% versus 1 to 3% versus 5 to 6%. Speculation "plus 92%" versus a
  measured minus 25.3% on dense. Adaptive slots "lead" versus minus 25 to 29%
  on a MoE model. Two notes both claim the "tenth engine lane". "Overhead
  mostly closed" versus "largest gap". Closure of issues 004 to 007 has no
  changelog entry.

## 13. How to re-check the external facts

```
gh api repos/ollama/ollama/releases/latest --jq .tag_name
gh api repos/ollama/ollama/issues/3368 --jq '.reactions["+1"],.state'
gh api search/issues -f q='repo:ollama/ollama is:issue OOM' --jq .total_count
gh api repos/vllm-project/vllm/releases/latest --jq .tag_name
gh api repos/sgl-project/sglang/releases/latest --jq .tag_name
gh api repos/sgl-project/sglang/pulls/42557 --jq .title
curl -sL 'https://arxiv.org/api/query?id_list=2406.03243,2611.20106&max_results=20' | rg -o '<title>[^<]*'
```

The plain-HTTP arXiv endpoint now redirects with an empty body, the default
page size is 10 entries, and requests a few seconds apart get rate-limited.

## 14. Sources

**In this repository (S1, S2)**

- `BENCHMARK.md`, `docs/benchmarks.md`, `bench-artifacts/INDEX.md`
- `docs/issues/INDEX.md`, `CHANGELOG.md`
- `docs/research/2026-10-02-competitor-pain-points-audit.md`
- `docs/research/2026-10-05-sota-audit.md`
- `docs/research/2026-10-28-magnitude-comparison.md`
- `docs/research/2026-10-29-deep-dive-implementation-roadmap.md`
- `docs/research/2026-10-31-engine-deep-dive-execution.md`
- `docs/research/2026-10-31-recursive-self-improvement-research.md`
- `docs/research/2026-10-31-silent-fallback-audit.md`
- `docs/research/2026-11-19-all-engine-kv-reuse-research.md`
- `crates/blazar-gateway/src/lib.rs`, `preflight.rs`, `proxy.rs`, `queue.rs`
- `crates/blazar-runtime/src/probe.rs`, `supervisor.rs`
- `crates/blazar-core/src/store.rs`, `config.rs`
- `crates/blazar-cli/src/main.rs` (`launch_cmd`)

**ollama (S3)**

- Releases v0.40.0 (2026-09-25) and v0.40.17 (2026-11-10):
  https://github.com/ollama/ollama/releases
- Issues 162, 228, 941, 1590, 2006, 2169, 3004, 3113, 3144, 3185, 3368, 5186,
  5245, 5304, 5360, 5800, 7865, 8618, 1005, 10792, 11691, 11772, 15051,
  19787, 19792, 19852, 19854: https://github.com/ollama/ollama/issues
- Docs: `docs/integrations`, `docs/capabilities`, `docs/api`

**Engines (S4, S5, S6, S8)**

- llama.cpp releases: https://github.com/ggml-org/llama.cpp/releases
- vLLM v0.34.0: https://github.com/vllm-project/vllm/releases/tag/v0.34.0
- SGLang v0.5.23: https://github.com/sgl-project/sglang/releases/tag/v0.5.23
- SGLang Paged Experts: https://github.com/sgl-project/sglang/pull/42557 and
  https://github.com/sgl-project/sglang/pull/45787
- sgl-router README in the SGLang repository

**Runtimes, routers, gateways (S7, S8)**

- https://github.com/ai-dynamo/dynamo
- https://github.com/llm-d/llm-d
- https://github.com/maximhq/bifrost
- https://github.com/kvcache-ai/ktransformers
- https://github.com/mudler/LocalAI
- https://github.com/microsoft/BitNet
- https://github.com/google-ai-edge/LiteRT-LM
- https://lmstudio.ai/docs/app

**Secondary (S9), by title, host and date**

- "Bifrost vs LiteLLM Rust: Do the 7x Benchmark Claims Hold Up?", dev.to,
  2026-11-10
- "LiteLLM vs Bifrost: Independent LLM Gateway Benchmark",
  computingforgeeks, 2026-11-04
- "Benchmarking the LiteLLM Rust AI Gateway", LiteLLM blog, 2026-07-22
- Ollama, LM Studio and llama.cpp speed comparison, kingy.ai, 2026-11-16
- "Local Inference Engines & Runtimes Compared 2026", promptquorum,
  2026-11-20
- SageMaker KV cache-aware routing announcement, 2026-11-16
- llm-d prefix-cache-aware router measurements, levelupit.xyz, 2026-11-20
- "180B Parameters, 32GB of RAM" MoE inference post, Hugging Face blog,
  2026-11-18
- "Why Is Ollama So Slow?", 2026-11-13. "Ollama Memory Limit", 2026-11-06

**Papers with resolved arXiv identifiers (S10)**

- 2309.06180 PagedAttention
- 2403.02310 Sarathi-Serve
- 2406.03243 Llumnix: dynamic scheduling for LLM serving
- 2407.00023 Preble: distributed prompt scheduling
- 2401.14351 ServerlessLLM
- 2404.16283 Andes: quality of experience in text streaming
- 2407.00023 Preble: distributed prompt scheduling for LLM serving
- 2611.12304 Evolving Infra: agent-adaptive LLM serving at scale (2026-11-09)
- 2611.20106 Making Preemption Productive in Long-Output LLM Serving
  (2026-11-15)
- 2401.09670 DistServe and 2407.00079 Mooncake (prefill and decode
  disaggregation, parked)
- 2401.10774 Medusa, 2310.07177 Online Speculative Decoding,
  2408.08696 Token Recycling, 2504.19874 TurboQuant
- 2503.01840 EAGLE-3, 2402.12374 Sequoia, 2611.05611 Hybrid Spark
  (content-chosen drafter), 2611.23835 NarrowSpec (narrower drafts)
- 2303.06865 FlexGen (offloading, background for the MoE ledger)
