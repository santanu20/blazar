# Kokoro + vLLM live capability certification — 2026-11-20

Certifies the two lanes that lacked live receipts (issue 002,
`docs/issues/002-vllm-kokoro-capability-certification.md`): the Kokoro TTS lane
end-to-end and the vLLM serving lane end-to-end through a Blazar gateway.

Environment: RTX 4070 8 GB, driver 580.x, Linux. Blazar tree build at v0.34.x
working tree (uncommitted concurrent-session engine changes included — the
`vllm serve` spawn used the tree's `compile_vllm` argv emitter; installed
daemon for Kokoro was v0.34.4).

## Kokoro speech synthesis (main daemon, port 11435)

- `POST /v1/audio/speech` `{"model":"kokoro","voice":"af_heart","input":"Blazar
  kokoro certification: the quick brown fox jumps over the lazy dog."}`
- Result: HTTP 200 in 3 s (includes lazy child boot), `kokoro.wav` = 244,204
  bytes, RIFF/WAVE, mono 24 kHz 16-bit PCM, 5.09 s duration.
- `GET /v1/audio/voices` returns the `kokoro` section with `af_heart` present
  (`voices.json`).
- Artifact: `kokoro.wav` (local; binary not tracked by git), `voices.json`.

## vLLM serving lane (isolated debug daemon, port 8199)

Method: XDG-isolated daemon (`XDG_DATA_HOME`/`XDG_CONFIG_HOME` pointed at a
scratch home with a sqlite-consistent snapshot of the store and a minimal
`config.toml` pinning `[model_overrides."qwen2.5-coder-0.5b-instruct"] engine =
"vllm"`), built from the working tree. Pin confirmed via
`GET /api/explain/qwen2.5-coder-0.5b-instruct:bf16` → `engine.source: "pin"`,
`kind: "vllm"`, `tag: "vllm-0.34.0"`.

- Cold `POST /v1/chat/completions` (greedy, echo prompt): HTTP 200 in 43 s
  including vLLM child boot; exact deterministic echo
  `BLAZAR VLLM CERT OK`; usage 41 prompt + 9 completion tokens with
  `prompt_tokens_details.cached_tokens` populated (`--enable-prompt-tokens-details`
  working).
- Warm repeat: HTTP 200 in 113 ms, same deterministic echo.
- Child argv captured (`ps`): `vllm serve <model.d> --host 127.0.0.1 --port
  <ephemeral> --served-model-name qwen2.5-coder-0.5b-instruct --max-model-len
  16384 --gpu-memory-utilization 0.85 --enable-prompt-tokens-details` — exactly
  the registry lane's emission surface (6 flags live-observed;
  `--kv-transfer-config`/`--max-num-seqs` are opt-in and were not exercised).
- Teardown: SIGTERM → daemon exited in 2 s, vLLM child + EngineCore reaped, GPU
  memory back to 16 MiB idle, no new `/dev/shm` mmaps from this run.

Artifacts: `req.json`, `resp.json` (cold), `resp2.json` (warm),
`daemon-tail.log`.

## Reproduce

Kokoro (any running daemon):

    curl -o kokoro.wav http://127.0.0.1:11435/v1/audio/speech \
      -H 'content-type: application/json' \
      -d '{"model":"kokoro","voice":"af_heart","input":"..."}'

vLLM (isolated home): snapshot store with python `sqlite3` backup API into
`<scratch>/data/blazar/blazar.db`, write `<scratch>/config/blazar/config.toml`
with the pin above and a free port, then
`XDG_DATA_HOME=<scratch>/data XDG_CONFIG_HOME=<scratch>/config \
./target/debug/blazar serve`.
