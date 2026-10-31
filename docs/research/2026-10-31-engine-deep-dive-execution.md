# Engine deep-dive execution wave — 2026-10-31

Executes the 2026-10-30 deep-dive suggestion list (user-approved: "do all except file split and
manually validate"). All items resolved or honestly dispositioned; receipts below.

## 1. Engine currency

| Lane | Result |
|---|---|
| llamacpp b11998 → **b12015** | ACTIVE (334-flag manifest probed, daemon restarted by the CLI, lane receipt: `/v1/chat/completions` temp-0 exact reply through the spawned b12015 child; zero knob-coverage loss — spawn manifest gate passed every emitted flag) |
| sglang **0.5.23** | e2e lane receipt LIVE (gateway → child, exact reply) + **C-ladder receipt** `bench-artifacts/20261030-sglang-0523-c-ladder/` — 0.5.21→0.5.23 parity within ±2.3% ITL p99 at C1/2/4/8 on qwen3-1.7b, zero failures both campaigns; detokenizer wave shows no delta at 66 tok/s (not detokenizer-bound — gains live at higher token rates) |
| mistralrs / mlx | NO-OP — v0.9.5 / 0.32.2 still latest (verified releases + PyPI); rerun lanes on next drops |

## 2. vid_gen reference media on `/v1/videos` (new capability)

Relay was already verbatim (`translate_to_native`); the real gaps were the gates. Landed:

- `parse_ref_media` — structural validation mirroring sd-server's parser exactly
  (`ref_videos` array-of-non-empty-frame-arrays, `ref_audios`/`ref_video_audios` string arrays,
  pairing ≤ videos, null-clears, image-route taught 400 naming `/v1/videos`).
- Ref frames priced into the VRAM scratch admission at the calibrated per-frame rate
  (`vram_overcommit` lever unchanged).
- `ref_media_capability_gate` — pre-empts only when the child's `generation_features` advertises
  an audio ref capability ABSENT (missing key on old engines = pass-through, child is authority).
- `child_capabilities` — caps cache now holds the whole document (defaults + features, one
  coherent snapshot, same TTL).
- Docs: `4.API_SPEC.md` /v1/videos row; 3 new unit pins (593→596 lib tests).

**Live receipts (production daemon, release binary, zero mocks):** three taught 400s exact
(shape / pairing counts / image-route); **positive render SEALED** — 2 base64 ref frames → 74s
→ 5-frame VP8 WebM 320x320, container QC'd (matroska/webm, dims/codec verified). Capability-gate
advertised-false branch: unit-proven; live proof needs an audio-VAE model (wan22-s2v pull —
documented gap). Observed layering on junk audio payload with an advertised capability:
gateway passes, child's verbatim error rides back (by design).

## 3. Pull-era quartet

- **mistralrs `--tok-model-id`** — FIXED: text GGUFs now auto-derive from `row.repo` behind a
  plausibility gate (registry/local-path junk refused). Precedence chain pinned in tests:
  file metadata > explicit user pin > repo guess > teaching refusal
  (`unit__mistralrs_argv__derived_tok_model_id_wins_over_user_extra` unchanged-green;
  `multimodal_gguf_rides_derived_tok_model_id` extended to the new contract; 2 new pins).
- **piper quant fuzzy / `rm` name-file / whisper-piper ENGINE dagger** — archaeology verdict:
  NO root cause exists in current code (voice resolution is exact-id; `rm` resolves colon→flat
  before removal and guards shared assets; whisper/piper voices are not models-table rows and
  render no ENGINE cell). The MEMORY flags predate refactors that removed the offending shapes.
  Need original repro commands to act further — NOT guessing (H6).

## 4. Orphan adoption on daemon restart

**Already shipped** (v0.25.0: `reclaim_marker_orphans`, TERM→10s→KILL, exec-env marker,
identity re-read before signal). Verified LIVE (V4): marked sacrificial child spawned →
deploy restart → journal `reclaiming orphan engine child pid 812064` → child reaped, daemon
healthy. Kill+respawn (not adopt) confirmed as the right complexity call. Roadmap row CLOSED.

## 5. CDN "throttle" — root cause + fix (user escalation: "fix cdn throttle professionally")

Two layered causes, both resolved:

1. **PMTU quirk (deterministic):** DF-1472 dies mid-path, ≤1428 passes; route cache learns
   per-dest `mtu 1480` but MSS locks at SYN per-flow. Fixed with a mangle-rule TCPMSS clamp
   (MSS 1360) on outbound SYNs — full speed verified (16.5 MB/s Cloudflare through the clamp).
2. **CDN abuse flagging (the throttle itself):** the day's pile-up of aborted transfers
   (3 killed installs + a curl resume loop + parallel probes from one IP) escalated
   GitHub's edge from range-501s to full stalls. Fix = stop hammering, cool down (~30 min),
   then ONE clean retry via the sanctioned path — `blazar engine update --no-gate` completed
   b12015 at full speed.

**Follow-up filed (not edited — gh.rs carries a concurrent uncommitted wave):** the engine
download retry loop is unbounded by design ("no total cap" for slow links); against an
abuse-throttled edge that means an install can hang ~forever with the read-timeout merely
cycling retries. Proposed: min-progress guard (abort + teaching error when aggregate asset
progress over N minutes < X bytes). Also: never hand-seed the download cache with
`curl -C -` loops — a server 200-reply to a Range request silently restarts the file from
offset 0 (observed 34 MB → 21 MB regression mid-seed).

## 6. Bench receipts

- sglang C-ladder: campaign + version-delta REPORT.md + INDEX row (12 cells, first attempt's
  VRAM-floor failures kept as floor-guard evidence, not data).
- sdcpp master-959 A/B: **CDN-dependent** (2 GB old-tag download during an abuse-flag window =
  not sane); master-1028 lane itself is current and live-proven. Re-run the A/B next session
  from a clean network state.

## Gates

fmt clean · clippy clean (workspace, all-targets) · nextest 2296/2296 · validate --smoke
116/0 (goldens regen'd on the verified release binary — the two regen'd rows were the
concurrent wave's `blazar_slots_source` metric and `adaptive_decode_gate` knob) · production
deployed (release build, daemon active, healthz ok).
