# Workload descriptors - live validation receipt

- Date: 2026-11-21
- Binary: tree debug build (post double-count fix), `blazar 0.33.0` dev tree
- Daemon: XDG-isolated sandbox (scratch data/config dirs), port 8199,
  sqlite-consistent store snapshot + engines symlink (relative `server_path`
  rows re-root under the sandbox data dir without it)
- Model: qwen2.5-0.5b-instruct:q4_k_m (llama.cpp lane b12578-cuda)

## What this validates

One `blazar_workload_requests_total` counter bump per request, with the kind
derived from the request surface:

| Request                          | kind counted        | round 1 | round 2 |
|----------------------------------|---------------------|---------|---------|
| POST /v1/chat/completions        | interactive_chat    | 2       | 1       |
| POST /v1/chat/completions +tools | tool_execution      | 2       | 1       |
| POST /v1/embeddings              | embedding           | 0 (chat)| 1       |
| POST /v1/audio/speech (kokoro)   | speech_synthesis    | 0       | 1       |

## Round 1 findings (the bug this receipt pins)

1. Handlers with two admission points (`openai_proxy` ensure + SLO gate,
   `scoped_proxy`, `responses_api`) classified at both, double-counting.
2. The default embeddings path rides `openai_proxy`, so embeddings were
   counted as `interactive_chat`; the surface must derive from the URI.
3. `kokoro_speech` admits lazily (no engine spawn up front) and never
   passed any classifier, so speech was never counted.

## Fix

- `openai_proxy`: one `workload_surface`/`workload_class` derivation before
  the first admission point; surface from the URI (`/embeddings` ->
  Embeddings, rerank/score/decisions -> Rerank, else Chat); both admission
  sites reuse it.
- `scoped_proxy` / `responses_api`: gate site reuses the `class` binding
  derived before `ensure_with_admission`.
- `kokoro_speech`: `workload::note(SpeechSynthesis)` at entry.

## Artifacts

- `chat2.json`, `tools2.json`, `embed2.json` (896-dim), `tts2.wav`
  (RIFF/WAVE mono 24 kHz) - round-2 responses.
- Final counters (round 2, `GET /metrics`):

```
blazar_workload_requests_total{kind="interactive_chat"} 1
blazar_workload_requests_total{kind="tool_execution"} 1
blazar_workload_requests_total{kind="long_prefill"} 0
blazar_workload_requests_total{kind="embedding"} 1
blazar_workload_requests_total{kind="reranking"} 0
blazar_workload_requests_total{kind="image_generation"} 0
blazar_workload_requests_total{kind="video_generation"} 0
blazar_workload_requests_total{kind="speech_synthesis"} 1
blazar_workload_requests_total{kind="transcription"} 0
```

## Verification

- `cargo nextest run -p blazar-gateway`: 938 passed (full crate suite).
- `cargo clippy -p blazar-gateway --all-targets`: no findings.
- Teardown: SIGTERM -> clean exit, port freed, zero orphan children
  (llama-server / vllm), VRAM back to 16 MiB baseline.

## Reproduction

```bash
# sandbox daemon (see docs/research/2026-11-20-kokoro-vllm-capability-certification.md
# for the XDG isolation recipe), then:
curl -s $B/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"qwen2.5-0.5b-instruct:q4_k_m","messages":[{"role":"user","content":"say hi"}],"max_tokens":8}'
curl -s $B/v1/embeddings -H 'content-type: application/json' \
  -d '{"model":"qwen2.5-0.5b-instruct:q4_k_m","input":"hello"}'
curl -s $B/v1/audio/speech -H 'content-type: application/json' \
  -d '{"model":"kokoro","input":"...","voice":"af_heart"}' -o tts2.wav
curl -s $B/metrics | grep '^blazar_workload'
```
