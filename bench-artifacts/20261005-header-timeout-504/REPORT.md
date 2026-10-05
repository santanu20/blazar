# Buffered long-generation 504 — root cause + industry norms + proposed fix (2026-10-05)

Status: root-caused and measured from source; live rig repro parked (box engine
store was mid-swap by a concurrent session — engines dir empty at probe time).
Repro prompt preserved in `task0.json` (reason-suite ruminator task 0, expected
answer 265, think-mode, never converges under 8192 tokens).

## Mechanics (proxy.rs)

`child_send` (proxy.rs:657-688) treats the window from request-written to the
child's **first response byte** as the header phase and bounds it with
`tokio::time::timeout(child_header_timeout_secs)` — default **120 s**, 0
disables. On expiry it logs a warning, then **synchronously evicts the whole
child** (`state.sup.evict(&engine.key)` — TERM→KILL, slot map drop) and
returns `504 ChildSendError::HeaderTimeout`. The SdCpp lane already carries a
separate 900 s knob for exactly this class (diffusion headers arrive only at
completion); text lanes kept the 120 s wedged-child default.

Design intent (code comment): defense against a live-wedged child — observed
in the wild as an `/api/chat` parked 300 s+ pre-first-byte while the child
kept serving later requests.

## The false-positive class

A **buffered (non-stream) long-reasoning request** is byte-identical to a
wedged child from the header phase's point of view: an LLM server sends
headers only when the body is ready, so a legitimate >120 s generation trips
the same bound. Consequences, all measured or source-proven:

1. The request 504s despite a fully healthy child mid-generation.
2. The synchronous eviction **kills every sibling in-flight stream** on a
   multi-slot instance — one client's buffered reasoning request destroys
   other clients' active generations.
3. Cold respawn cost lands on the next request.

Live evidence from the reason-suite campaign: think-mode generations of
1492 tokens completed in ~38 s; ruminators ran 95-151 s+ and one 8192-token
generation drew a gateway 504 mid-stream with the client timeout at 600 s —
the 504 was gateway-side (b3/m0097). Streaming never trips the bound because
headers flush at first token.

## Industry norms (verified, not assumed)

- **vLLM: no server-side generation timeout.** The `--timeout` family covers
  shutdown, health probes, and distributed init — none cap a request's
  generation; requests run to completion or client disconnect. The 504s users
  report with vLLM are their own reverse proxies (nginx `proxy_read_timeout`
  60 s default), which vLLM documents keep-alive comment workarounds for
  (`--stream-interval...` family; docs.vllm.ai configuration).
- **ollama: no generation timeout either.** GitHub issue ollama/ollama#5081
  ("Timeout for long generation") resolves as a **model-load** timeout
  (`OLLAMA_LOAD_TIMEOUT`, 5 m default) — generation itself is unbounded;
  `keep_alive` governs unloading, not in-flight requests.

Neither competitor evicts the engine because one request is slow. Blazar's
wedged-child defense is a real capability they lack — but as shipped it
conflates "slow generation" with "wedged child" on buffered requests.

## Proposed fix (behavior change — needs approval, non-breaking to config)

On `HeaderTimeout`, before evicting: **probe the child's `/slots`** (precedent:
the sleep-state `/props` probe in `ps`, and the existing `/slots` proxy route).

- Any slot actively generating → child is NOT wedged → skip the eviction,
  return a **teaching 504** naming the two remedies (`"stream": true` or raise
  `child_header_timeout_secs`), keep the child serving.
- All slots idle/dead or probe fails → wedged as originally designed → evict
  synchronously (unchanged).

This preserves the wedged-child defense (a truly wedged child has no active
slots) and matches the ollama/vLLM norm (slow buffered generation never kills
an engine). Sibling streams survive. Cost: one loopback HTTP probe on an
already-degraded path (the 120 s bound already fired).

Config surface unchanged; docs note for `child_header_timeout_secs` in
7.SETUP.md ("header phase = request→first byte; buffered long-reasoning may
need a higher value or streaming").

## Live validation (2026-10-05, after fix)

Rig: real daemon (fresh debug build), real llama.cpp child (b11417-cuda),
real Qwen3.5-9B-Q4_K_M, box store snapshot via sqlite backup API.

| Scenario | Result |
|---|---|
| Buffered "write numbers 1-30000" (`num_predict` 16384, temp 0) + concurrent stream witness | HTTP 504 at exactly **120.0 s** — single ceiling, teaching body exactly once, no retry. Child KEPT: `/api/ps` shows the model resident/ready, warm follow-up answered in **0.207 s**, sibling stream uninterrupted through the whole window |
| Same scenario, pre-fix binary | HTTP 504 at **240 s** (two 120 s ceilings — an unguarded lane-handler retry re-waited on the kept child), teaching text duplicated after "retry on respawned child:" |
| Think-mode ruminator (reason-suite task 0) | Converged at 112 s < ceiling — no trip (thinking length is nondeterministic; the honest case) |
| Wedge class (e2e stub pin, unchanged path) | Evict + one in-band retry + "wedged child evicted" body — legacy behavior intact |

Daemon log line for the kept-child case, verbatim:
`child silent for 120s but its slots are generating — buffered long
generation; child kept`.

Post-live gap found and fixed: three lane handlers (ollama non-stream chat,
ollama stream, openai responses buffered) retried the header-timeout before
stringification reached the guarded match — now all early-return the teaching
504 without respawn, and the e2e pin asserts no-retry + single teaching text.

Scenario notes for reproducers: "write banana forever" at temp 0 terminates
in ~7 s (not a ruminator); a numbers-list prompt fills the token budget
reliably. e2e pins use the stub child with a 1 s ceiling — the retry gap was
invisible there (retry cost only 2 s and both halves carried the teaching
text), which is why the tightened assertions matter.
