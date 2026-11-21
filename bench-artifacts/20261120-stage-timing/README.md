# Request stage timing - live receipt (2026-11-20)

Issue: docs/issues/004-request-stage-timing.md (B05.1). Verifies the
per-request stage instrumentation end-to-end on a real spawn: every boundary
stamp fires, both readout surfaces aggregate it, teardown is clean.

## Methodology

XDG-isolated debug daemon (debug build of the working tree, port 8199):

- store = sqlite-consistent snapshot of the live store (python `backup()`)
- config = `{host = "127.0.0.1", port = 8199}`, no other knobs
- engines dir symlinked from the real data home (snapshot rows carry
  relative `server_path`s that re-root under the overridden data dir -
  without the symlink every spawn 500s with the teaching error)
- model = qwen2.5-0.5b-instruct:q4_k_m on the llama.cpp lane
- traffic = 1 cold + 4 warm chat completions (3 non-stream, 1 stream,
  30 SSE frames), greedy, max_tokens 30

## Observed stages (GET /metrics, 5 counted requests)

| Stage | p50 | p99 | Notes |
|---|---|---|---|
| auth | 87.5 us | 250 us | keyless path; keyed adds one hash lookup |
| body | 10 ms | 250 ms | includes first-hit lane resolution; p99 tail = cold request |
| admission | 0.375 ms | 2.5 s | p50 = uncontended ensure; the 2.5 s outlier is the cold spawn wait |
| dispatch | 100 ms | 250 ms | child header latency, warm |
| model_ttft | 100 ms | 250 ms | first streamed byte after slot grant |
| stream | 3.75 ms | 100 ms | decode span of short greedy outputs |

GET /api/quantiles `stages` block mirrors the same six stages as
`{count, p50_ms, p95_ms, p99_ms}` objects (null-on-empty verified on a
fresh boot before traffic).

Semantics check: gateway overhead (auth+body+admission+dispatch) p50 sums
to ~110 ms warm, of which ~100 ms is dispatch (child HTTP round-trip) -
the pure-gateway slice (auth+body+admission-before-dispatch) is sub-ms to
~10 ms, consistent with the design target of single-digit-ms overhead.

The stream request's first/last-token stamps land through the usage tap
(`note_chunk` on the connection task, past the request task-local scope) -
count 5 includes it; stages with unfired boundaries stayed uncounted
(early 500s during the broken-symlink phase observed `auth` only, as
designed - never fabricated).

## Teardown

SIGTERM: daemon exited in <10 s, `evict: terminating child and releasing
the slot` + KV banking logged, 0 orphan llama-server processes, GPU back
to 16 MiB baseline.

## Artifacts

- `resp1.json` - cold chat response (200, 1.84 s incl. lane bring-up)
- `stream.txt` - SSE stream (30 frames)
- `daemon.log` - full daemon log incl. profile lines and clean stop

## Reproduction

```
mkdir -p /tmp/stage-cert/{data,config}/blazar
python3 - <<'PY'
import sqlite3
src = sqlite3.connect('<real>/blazar.db'); dst = sqlite3.connect('/tmp/stage-cert/data/blazar/blazar.db')
src.backup(dst); dst.close(); src.close()
PY
printf 'host = "127.0.0.1"\nport = 8199\n' > /tmp/stage-cert/config/blazar/config.toml
ln -s <real>/share/blazar/engines /tmp/stage-cert/data/blazar/engines
XDG_DATA_HOME=/tmp/stage-cert/data XDG_CONFIG_HOME=/tmp/stage-cert/config \
  ./target/debug/blazar serve &
curl -s http://127.0.0.1:8199/metrics | grep '^blazar_stage'
curl -s http://127.0.0.1:8199/api/quantiles | jq .stages
```
