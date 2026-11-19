# llama.cpp -> vLLM Bank Import: the KV Data Plane

Date: 2026-11-19
Wave: Branch B of the KV campaign (PR4 shipped the policy layer; this is the
first wired converter). Research parent: `2026-11-19-kv-translation-framework.md`.

## Problem

PR4 made cross-lane KV reuse *decidable* (typed converters, hard blocks, honest
`no bridge is wired` verdicts) but moved no bytes. The banked question was
whether a real converter could be built without reverse-engineering an
unstable format or shipping a maintenance trap. Answer: yes, because both ends
are fully deterministic and one of them (vLLM's FS offload tier) is
self-documenting - a spawned engine writes ground-truth files we can byte-diff
against.

Why this pair first: llama.cpp slot banks and vLLM's persistent offload tier
are the two formats Blazar already owns end-to-end (PR1 session banks, PR3b
`[vllm.kv_offload]`), and the LMCache wave proved the daemon path cannot do
cross-engine transfer (layout-blind retrieval = garbage). The import tool is
the bridge the daemon will never ship.

## Source format: llama.cpp slot-save banks (engine b12360, grammar source-verified + oracle-validated)

Container (llama-server `POST /slots/:id` save, same file the supervisor banks
as `_auto-*`):

```
u32 magic 0x67677371   ; 'gsgg' on disk (bytes 71 73 67 67)
u32 version 4
u32 n_token_count
i32 tokens[n]          ; may contain -1 gap markers; import requires none in the
                       ; cached prefix
KV sections...         ; loop while next u32 != SCKP magic and not EOF
[ SCKP appendix ]      ; server slot-checkpoint blobs - skipped, never read
```

SCKP appendix (server-context.cpp:2799 @b12360): magic 0x504b4353, version 1,
count, then per checkpoint three length-prefixed buffers (target/default/spec
decoding state). These are rewound intermediate checkpoints - irrelevant to KV.

KV section (llama-kv-cache.cpp state_write:2233):

```
u32 n_stream            ; import enforces 1
per stream:
  u32 cell_count        ; 0 -> skip stream
  per cell: i32 pos, u32 n_seq_id, n_seq_id x i32 seq_id
            [12-byte llama_kv_cell_ext {i32 x, y, tok} per cell if the model
             class writes it - DSA-family; parser tries no-ext first, retries
             with ext, refuses if both walks fail]
u32 v_trans             ; 0 or 1
u32 n_layer, u32 n_rot_k, u32 n_rot_v   ; rot values read+ignored (post-RoPE)
KEYS per layer:
  i32 ggml_type         ; must be 1 (F16)
  u64 row_bytes         ; must equal kv_heads * head_dim * 2 - cross-checked
  cell_count x row_bytes raw rows, token-major
VALUES per layer:
  v_trans == 0: same shape as keys
  v_trans == 1: i32 type, u32 v_size_el, u32 n_embd_v_gqa, then per column j
                per contiguous cell-range: range_len x v_size_el bytes
                (column-major; regrouped to rows at parse time)
```

Notes pinned empirically:
- The engine at rest writes `v_trans = 0` (row-major); the v_trans=1 branch is
  parsed for completeness and regrouped honestly.
- Hybrid/ISWA models serialize 2+ sections sequentially (gemma3-270m = 3 SWA
  layers + 15 base layers). This release refuses multi-section banks with a
  teaching rather than guessing the layer-to-engine mapping; single-section
  dense models import.
- Oracle: a real bank on this box (gemma3-270m, 293,588 B, engine b12360)
  walks EOF-exact under this grammar: section 1 = 10 cells x 3 layers, section
  2 = 10 cells x 15 layers, every tensor 512-B F16 rows + 12-B header, tokens
  `[-1, 1, 10, 2, 105, ...]`, SCKP fills the tail. The committed unit tests
  encode this grammar with synthetic fixtures.

## Sink format: vLLM 0.34.0 FS offload tier (replicated byte-exactly)

- Rank dir: `<root>/<safe_model_name>_<digest12>_r0` (`safe` = path slashes to
  `_`; rank 0 because Blazar spawns tp=1/pp=1).
- `config.json` inside the rank dir = the FileMapper fields dict:
  `{model_name, tokens_per_hash, blocks_per_file, tp_size, pp_size,
  canonical_format}` - model_name is the exact model *path* string Blazar
  passes on the vllm command line (identity-relevant, not cosmetic).
  `digest12 = sha256(canonical_json(fields))[:12]`, canonical =
  `json.dumps(..., sort_keys=True, separators=(',',':'))`.
- Block file: `<hash[..3]>/<hash[3..5]>/<hash_hex>_g<group_idx>.bin` where
  `hash_hex` is the chain hash of the block's token ids and `group_idx` the
  prefix-cache group (0 for dense models). Written `.part` + rename.
- Page bytes: canonical `v1-nhd` layout (LBNHC live on this box, flash-attn):
  per block, layer-major, per layer per KV head per token: K head-slice then
  V head-slice (fused `2*head_size` inner dim), 64 tokens per block.

## The addressing hash: pickle-protocol-5 SHA-256 (decoded + pinned)

vLLM hashes blocks as `sha256(pickle.dumps((parent_hash, tuple(tokens), None),
protocol=5))`, seeded from `sha256(pickle.dumps('vllm-none-hash'))`. Pickle-5
frames are not documented anywhere we could find; we decoded them from golden
vectors generated through vLLM's own `vllm.v1.core.kv_cache_utils` on this box:

```
80 05                      PROTO 5
95 <u64 LE frame len>      FRAME - length INCLUDES the STOP byte
43 <u8 len> <bytes> 94     SHORT_BINBYTES (parent hash) + MEMOIZE
28 ... 74 94               MARK + items + TUPLE + MEMOIZE   (tuples >= 4 items)
85 / 86 / 87 94            TUPLE1 / TUPLE2 / TUPLE3 + MEMOIZE (shorter tuples)
4e                         NONE
2e                         STOP
ints: 4b <u8> (0..=255) | 4d <u16 LE> (256..=65535) | 4a <i32 LE> (rest)
```

Golden vectors committed as unit tests include: the none-hash seed, the
BININT1/BININT2/BININT opcode boundaries (255, 256, 300, 65535, 65536,
151643), a 2-block chain, and a full digest12 - all cross-generated through
the venv's vLLM, not hand-computed.

## Gates (order matters; every refusal teaches)

1. `[kv.translation] enabled` - the PR4 knob owns converter admission.
2. Provenance: GGUF `general.file_type` must be a lossless container class
   (F16/BF16); `general.base_model_id` must equal the HF repo id after
   normalization (lowercase, alphanumerics only) - spelling variants pass,
   org-less basenames refuse (ambiguous), missing id refuses with the
   `gguf-set-metadata` fix in the message.
3. RoPE fingerprint equality (theta + scaling, read from both GGUF kv pairs
   and HF `config.json`).
4. Tensor geometry equality (layers x loops, kv_heads, derived head_dim).
5. PR4 translation verdict on the layout pair must be `Transformable`
   (`defuse+repage` for this lane pair).
6. Bank grammar (above): magic, version, single stream, contiguous positions
   from 0, no gap markers in the cached prefix, F16 rows matching geometry.

## What this wave deliberately does NOT do

- No automatic import on bank save (gateway coupling before the data plane is
  proven = the LMCache trap; revisit once live evidence accumulates).
- No multi-section (hybrid/SWA) banks, no quantized KV dtypes, no dtype casts,
  no partial-block tails (vLLM stores whole blocks; tails are dropped and
  reported).
- Kernel-level numeric drift (llama.cpp F16 kernels vs vLLM kernels on
  identical weights) is accepted approximation - the cache restores attention
  *state*, not weights; the ship gate is a sane reply through the real
  gateway, not bitwise equality with a natively-computed prefix.

## Verification

- Unit: 14 golden/grammar/gate tests in `runtime/kv/repack.rs` + CLI row test
  (all cross-checked against vLLM-generated vectors and the grammar above).
- Live (ship gate): F16 GGUF twin of `qwen2.5-1.5b-instruct` built from the HF
  dir, served on the llama.cpp lane, banked, imported, then the same prefix
  through the vllm lane gateway must report `cached_tokens > 0` with a sane
  reply. Findings recorded below after the run.

## Live findings (2026-11-19, all through the real gateway unless noted)

Proof pair: `qwen2.5-1.5b-f16` (F16 GGUF converted from the on-box HF dir,
stamped `general.base_model_id = Qwen/Qwen2.5-1.5B-Instruct`) -> `qwen2.5-1.5b-instruct`
(HF dir), 28 layers, 2 KV heads, head_dim 128, 343-token Kestrel prompt.

- **Token envelope correction (root cause found during the live walk):** the
  slot-save token array is not a plain id list - it is llama.cpp's
  `server_tokens::serialize()` envelope: `[-1 marker][i32 version=1][u32
  n_real][real ids][u32 media_count]...`. Verified against three real banks
  (404, 295 and 14 header tokens all reconcile). The parser handles both the
  envelope and the legacy plain list; multimodal media chunks refuse.
- **Tokenizer equality holds exactly:** the bank's real ids match a
  transformers 5.x `apply_chat_template` tokenization of the same chat 343/343
  - the hash chain built from bank ids addresses exactly what vLLM computes
  from its own tokenization.
- **Import:** `blazar kv import` wrote 6 blocks x 64 tokens (tail 20 tokens
  dropped and reported). Block file = 1.8 MB = 28L x 64T x 2H x 128D x 2B x
  (K|V) exactly. Digest12 and the first block's chain hash independently
  recomputed through the venv's vLLM matched the emitted rank dir and file
  name; the engine booted against the imported config.json without creating a
  competing rank dir.
- **Engine-level restore:** direct vLLM (manual, OffloadingConnector with the
  same TieringOffloadingSpec the profile compiler emits): `cached_tokens 320`
  of 343 stable across retries with sane, rule-correct replies - imported
  bytes are semantically valid, not the garbage class the LMCache
  layout-blind transfer produced. First request after import can report 0
  (async external lookup registers one pass late) - retry before judging.
- **Gateway-level restore (ship gate):** through blazar's gateway
  (`/v1/chat/completions`, blazar-spawned vllm child): `prompt_tokens 343,
  cached_tokens 320`, sane reply. Full chain live: llama.cpp compute ->
  slot bank -> kv import -> vllm FS tier -> cached serve.
- **Operational findings:** vLLM's OffloadingConnector CPU tier allocates
  `cpu_bytes_gib` (default 2 GiB) in `/dev/shm` at boot - small `/dev/shm`
  fails engine init; crashed vLLM cores LEAK their `vllm_offload_*.mmap`
  shm files (rm before respawn). `gguf_set_metadata.py` can only modify
  existing keys - adding `base_model_id` to a fresh convert requires the
  reader->writer copy flow. Teardown verified: SIGTERM to the debug daemon
  reaped every child, zero orphan processes, and deleting the rank dir is a
  complete rollback.


## Live findings - reverse direction (2026-11-19, wave B)

- The reverse codec (`vllm -> llama.cpp`) reads the engine-written tier for
  the Kestrel prefix (5 full blocks, 320 tokens; the 23-token tail is not a
  full block and was never offloaded), repacks a slot-save bank, and the
  restore gate accepts it end-to-end through the gateway:
  `restored banked session _auto-16384 into slot 0 (320 tokens into slot KV)`.
- **n_rot oracle**: the engine writes `n_rot_k = n_rot_v = 0` in slot saves
  (post-RoPE rows carry no rotary dimension). A bank with `n_rot = head_dim`
  is rejected at restore with a misleading `No available space` space error;
  the same bytes with `0/0` restore cleanly (`n_restored: 320`). The codec
  now writes zeros, pinned by a live A/B on a manual llama-server.
- **Identity mirror rules** (root-caused from two skipped restores): the
  manifest must carry the spawn's *resolved* values - pinned slots divide the
  requested ctx across slots (pinned 2 x 16384 -> bank `_auto-8192`, manifest
  ctx 8192), and auto slots (config 0) resolve against live VRAM at spawn
  time, which no offline import can predict - the import refuses with a
  pin-your-slots teaching instead of writing a bank the gate silently skips.
- **Restore semantics on installed engines**: both llama.cpp builds here
  (b12521-cuda and the sensenova fork) accept slot restores and inject KV,
  then re-prefill the prompt anyway - verified with engine-native save files
  (exact-prompt replay after a native restore still evaluates all tokens).
  This is the restored-then-empty class the restore truth-check comment
  already guards for. The import therefore ships data-plane-correct
  (engine-accepted, full prefix injected); the prompt-eval warm-start payoff
  is pending upstream behavior, symmetric to how the forward direction
  needed vLLM's own loader to realize `cached_tokens`.

## Sources

- llama.cpp b12360: `src/llama-kv-cache.cpp` (state_write:2233),
  `src/llama-context.cpp`, `src/server-context.cpp` (SCKP:2799),
  `src/llama-kv-cells.h`; empirical oracle bank on this box (engine b12360).
- vLLM 0.34.0 (venv sources): `vllm/v1/kv_offload/file_mapper.py`,
  `vllm/v1/kv_offload/tiering/fs/manager.py`, `vllm/v1/kv_offload/base.py`,
  `vllm/v1/core/kv_cache_utils.py` (hash_block_tokens), `vllm/utils/hashing.py`,
  `vllm/v1/kv_offload/canonical_mapping.py`; golden vectors generated through
  the engine's own import path.
