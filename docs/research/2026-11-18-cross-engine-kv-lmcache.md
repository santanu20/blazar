# Persistent Shared KV via a Standalone LMCache Daemon - Blazar

Date: 2026-11-18 (live findings 2026-11-19)
Wave: shared KV store (LMCache)
Status: IMPLEMENTED - `[kv.lmcache]` daemon + sglang lane opt-in, verified
live end-to-end. Cross-engine reuse (sglang <-> vllm) is REFUSED at
validation: live verification proved the daemon has no cross-layout
converter (foreign-layout blocks restore as garbage). Details in
"Live findings" below.

## The problem

Blazar runs several GPU engine lanes (llama.cpp, sglang, vllm, ...) each with
their own in-process KV cache. A prompt prefix computed by one engine
instance is locked inside that instance: a restart of the same engine
(llama.cpp slot banks, vllm's own OffloadingConnector fs tier) can reuse it,
but another instance of the same lane never could. Cross-INSTANCE prefix
reuse - instance A computes, instance B (after evict/respawn) restores -
needs a KV store outside both processes. The stretch goal was cross-ENGINE
reuse (sglang computes, vllm restores); live verification showed upstream
cannot serve it correctly yet, so Blazar gates it off and ships the
cross-instance lane.

## Approach survey (angle ladder, rung 4 = platform)

- **Gateway-mediated transfer (approach B)** was rejected: the gateway would
  have to decode/re-encode engine-specific KV layouts it does not own,
  duplicating what a KV-store daemon already does, poorly.
- **LMCache** is the native sharing mechanism both target engines already
  integrate: vLLM ships a connector class for it, SGLang has a first-class
  flag pair. LMCache's own `lmcache server` process owns the store; engines
  are clients. That makes the daemon the platform rung - Blazar supervises
  the process, emits the config, and never touches the KV bytes.

## Verified engine surfaces (ground truth: installed packages on the dev box)

### vLLM 0.34.0 (venv `~/.local/share/blazar/engines/vllm-0.34.0`)

- Connector class `LMCacheMPConnectorUpstream` (`vllm/distributed/kv_transfer/kv_connector/v1/lmcache_mp_connector.py:462-500`);
  registry string `"LMCacheMPConnector"` confirmed in the factory
  (`kv_transfer/factory.py`).
- `--kv-transfer-config` extra_config keys (verbatim from the connector):
  - `lmcache.mp.host` - default `"tcp://localhost"` (scheme-qualified)
  - `lmcache.mp.port` - default `5555`
  - `lmcache.mp.mq_timeout` (300), `lmcache.mp.heartbeat_interval` (10) -
    left at defaults; Blazar pins only host/port.
- `kv_cache_access = LEGACY_IPC` (cudaIpc): same-host, same-user, shared
  `/dev/shm` - satisfied by Blazar's single-host supervision model.
- Module-top `import lmcache` -> the vllm venv must carry the `lmcache` pip
  package (Blazar detects + teaches, never installs).

### SGLang 0.5.23 (venv `~/.local/share/blazar/engines/sglang-0.5.23`)

- Flags: `--enable-lmcache` + `--lmcache-config-file <yaml>`
  (`python/sglang/srt/arg_groups/fields/memory.py:256-262`).
- Upstream constraints encoded as Blazar compile-time gates:
  XOR `--enable-hierarchical-cache` (HiCache), no speculative decoding,
  no data/attention parallelism (`dp_size = 1`, `attn_dp_size = 1`),
  radix cache required (on by default - Blazar emits no XOR flag).
- Consumer: `UnifiedLMCacheMPConnector`
  (`srt/mem_cache/storage/lmcache/lmcache_unified_radix_cache.py`) ->
  `lmcache` pip required in the sglang venv as well.

### LMCache 0.5.8rc1 (installed into both venvs)

- Started at 0.5.7.post1; upgraded to **0.5.8rc1** after live proof showed
  0.5.7.post1's vLLM integration crashes on vllm 0.34.0's scheduler API
  (`AttributeError: 'CachedRequestData' object has no attribute
  'resumed_req_ids'` at `lmcache_mp_connector.py:1874` - vllm 0.34.0's
  `CachedRequestData` carries only
  req_ids/new_token_ids/new_block_ids/num_computed_tokens/num_output_tokens;
  0.5.8rc1 guards with `getattr(cached_reqs, "resumed_req_ids", ...)` at
  connector lines 213 and 2039).
- Client YAML contract (`v1/client/client_config.py:27-50` in 0.5.8rc1;
  `v1/multiprocess/client_config.py` in 0.5.7): exactly
  `{mp_host, mp_port}` (+ optional `extra_config: {}`). Host is BARE - the
  client prepends `tcp://` itself when the scheme is absent. Env overrides
  `LMCACHE_MP_HOST/LMCACHE_MP_PORT` exist; Blazar writes the file instead so
  the config is visible on disk.
- `lmcache server` CLI (pinned from live `--help` on BOTH versions): MP bind
  `--host` (default localhost) `--port` (default 5555); HTTP frontend
  `--http-host` (default 127.0.0.1) `--http-port` (default 8080);
  `--l1-size-gb`; `--l1-init-size-gb` (lazy allocator init, DEFAULT 20 GiB -
  Blazar pins it equal to `l1_size_gb` so a 4 GiB pool never boots with a
  20 GiB init); `--eviction-policy {LRU,ARC,IsolatedLRU,noop}`; worker /
  blend knobs exist (`--max-workers`, `--engine-type blend`,
  `--enable-blend-reorder`) but are not needed for the shipped same-lane
  path.
- Health: `GET /healthcheck` on the HTTP frontend (inspects internal state).
  Banner disable env: `LMCACHE_DISABLE_BANNER=1`.

### Dependency note (live-verified)

`lmcache==0.5.7.post1` pins `numpy<=2.2.6` (py<3.14): installing it into the
engine venvs downgraded numpy 2.3.5 -> 2.2.6. Both engines were re-imported
clean on 2.2.6 (sglang 0.5.23, vllm 0.34.0, torch 2.14.x intact) before this
was accepted as the install strategy. protobuf in the vllm venv settled at
6.33.6 under the same resolve.

## Blazar design

- **Config surface** (`[kv.lmcache]`): `enabled` (default false), `host`
  (127.0.0.1), `port` (5556 - one off upstream's 5555 default so a hand-run
  `lmcache server` never collides), `http_port` (8765, loopback healthcheck
  frontend), `l1_size_gb` (4, validated 1..=64), `eviction_policy` (LRU,
  allowlist = the daemon CLI's argparse choices).
- **Lane opt-in**: `sglang.lmcache.enabled` (global + per-model override,
  whole-struct replace like every sglang knob), gated on
  `[kv.lmcache] enabled` - rejected at parse time with the fix in the
  message, mirrored in the profile compiler for programmatic callers.
  The vllm leg `[vllm.kv_offload] connector = "lmcache"` is REFUSED at
  validation (see Live findings): the enum variant stays as a documented
  guardrail with a teaching error, and the built-in `offload` connector
  remains that lane's persistent tier.
- **Profile compiler**: sglang emits the flag pair with
  `--lmcache-config-file <data_dir>/kv/lmcache/sglang.yaml`. Compile-time
  gates: spec XOR, dp=1, daemon-on. The vllm connector arm is a
  defense-in-depth refusal (config validation already blocks it).
- **Runtime supervision** (`blazar-runtime/src/lmcache.rs`): lazy child -
  spawned only when a lane's argv dials it (`argv_dials_lmcache` predicate on
  both the profile argv at spawn and each live instance's stored argv at
  reap, the same predicate both sides); fingerprint (host/port/http_port/l1/
  eviction) drift respawns; `/healthcheck` polling with a 60 s ready
  deadline and fail-fast on child exit (daemon.log named in the error);
  `kill_on_drop` + unix parent-death tie; idle reap on the supervisor reaper
  loop when zero consumers remain (budget = `idle_timeout_secs`); graceful
  shutdown on supervisor shutdown. Binary discovery walks the lane venvs
  (sorted, `venv/bin/lmcache` unix / `venv/Scripts/lmcache.exe` windows) -
  any lane's copy serves every engine over the MP transport. Missing binary
  = teaching error with the `uv pip install` line; the product never
  installs lmcache itself.
- **Registry honesty** (`blazar-runtime/src/kv`): the sglang LMCache lane
  reports persistent/prefix-blocks/cross-process = true (the daemon is the
  store) and gateway-commanded export/import = false with a teaching that
  transfers are daemon-mediated - a capability statement, never a simulated
  transfer.

## Live findings (2026-11-19, this box: 8 GiB VRAM, qwen2.5-1.5b-instruct BF16)

The daemon supervision, argv emission, client YAML, lazy spawn, healthcheck,
evict-survival and idle-reap paths were all exercised live through the real
gateway (`blazar serve`, model pinned per phase via
`model_overrides.<name>.engine` + `POST /api/config/reload`). The matrix:

| Path | Result | Evidence |
|---|---|---|
| sglang -> daemon store | works | daemon.log `Stored 256 of 256 requested tokens in 0.023 s` |
| sglang -> daemon -> NEW sglang instance | **works, correct** | evict + respawn + same prompt: HTTP 200, sane reply, `usage.prompt_tokens_details.cached_tokens = 256` of 343, daemon.log `Retrieved 256 tokens in 0.030 s` |
| sglang -> daemon -> vllm (cross-engine) | **mechanically retrieves, semantically CORRUPT** | engine logs `External prefix cache hit rate: 74.6%` (256/343) + daemon `Retrieved 256 tokens`, but the reply is token soup; a cache-MISS control prompt on the same server answers coherently in 0.56 s |
| vllm -> daemon store | never fired under light load | connector stores are eviction-triggered offload semantics (`lmcache.mp.save_decode_cache`/`save_unfull_chunk` default off); zero vllm store events across all runs |

Why cross-engine reads garbage - layout mismatch, verified from the daemon's
own registration logs plus installed-source reads:

- sglang registers `EngineKVFormat.NL_X_NB_BS_NH_HS`: 56 layer tensors
  (K and V counted separately for this 28-layer model), `tokens_per_block=1`
  (token-major).
- vllm registers `EngineKVFormat.NL_X_NB_BS_NH_CS`: 28 fused layer groups
  (K|V in one `cs=256` axis), page 64 (block-major).
- The retrieve path (`lmcache/v1/multiprocess/modules/
  lmcache_driven_transfer.py:984`) resolves stored objects by token-key and
  copies them into the registered destination tensors; the tree has NO
  format-conversion function (`rg "def convert|reformat"` = zero hits in
  `v1/`). Only "different engine groups within ONE registration" are
  handled (`kv_layer_groups.py`). The `blend` module (re-RoPE + per-token
  scatter, could in principle place per-token chunks into foreign pages)
  only builds under `--engine-type blend` and is NOT cross-layout
  conversion either. Key matching is layout-blind, so the corruption is
  silent - hence Blazar refuses the lane instead of shipping garbage.

Side findings recorded for the next engine-currency pass:

- flashinfer-python 0.7.3 on PyPI is a PURE wheel (py3-none-any): vllm 0.34.0
  JIT-compiles ops at boot and dies on an old nvcc (`Unknown option
  '--compress-mode=size'`, CUDA 12.0 system compiler). Fix verified live:
  install the prebuilt kernel wheels from
  `https://flashinfer.ai/whl/cu130` (`flashinfer-cubin`,
  `flashinfer-jit-cache`, `flashinfer-jit-cache-sm89`, matching the torch
  CUDA); `download-kernels` bulk mode times out mid-download.
- lmcache 0.5.7.post1 vs vllm 0.34.0: `CachedRequestData` API drift
  (see version note above) - 0.5.8rc1 is the minimum working pairing.
- An unexplained one-off: a blazar-spawned vllm child once wedged after a
  successful cross-engine retrieval (no headers within the gateway's 120 s
  child timeout); the same argv run manually completed (garbage, per the
  corruption finding). Not pursued - the lane is refused.

## Verification

- Unit: config truth table (defaults, daemon validate errors, sglang lane
  gate + the model_overrides smuggle path, vllm connector refusal with the
  daemon both on and off, hicache XOR), vllm compile refusal for the
  lmcache connector (defense in depth), sglang flag pair + yaml path,
  spec/dp/daemon gates, argv predicate both kinds, discovery across fake
  venv layouts, client YAML content, registry caps honesty.
- Live: the matrix above; teardown left zero orphan `lmcache` processes
  (supervisor shutdown kills the daemon; lazy respawn verified by killing
  the daemon mid-session and watching the supervisor bring it back with
  identical argv).

## Sources

- vLLM 0.34.0 installed source: `lmcache_mp_connector.py`, `factory.py`
  (venv path above).
- SGLang 0.5.23 installed source: `arg_groups/fields/memory.py`,
  `mem_cache/storage/lmcache/lmcache_unified_radix_cache.py`.
- LMCache 0.5.8rc1 installed source (both venvs): `v1/client/
  client_config.py`, `cli/commands/server.py`, `v1/multiprocess/http_apis/
  info_api.py`, `v1/common/kv_format/{detection.py,specs/}`,
  `v1/multiprocess/modules/lmcache_driven_transfer.py`,
  `v1/multiprocess/modules/blend/` (PyPI wheel; docs.lmcache.ai
  configuration pages 404 at research time - the installed source is the
  authoritative contract).
- Upstream docs: docs.vllm.ai (KV offloading + external connector examples),
  docs.sglang.ai (HiCache/LMCache integration pages),
  docs.flashinfer.ai/installation.html (prebuilt kernel wheels index).
