# Comprehensive Validation — All Engines, All Configs, All Commands, All Flags (2026-10-03)

Scope mandate: manually validate every engine lane, every config knob, every command,
every flag in on/off states — live, no mocks, no shortcuts — to surface hidden bugs,
integration gaps, and silently broken behavior. This report is the complete record:
phases P0–P7, every defect found, root causes, fixes, pins, and receipts.

## 1. Method and environment

| Item | Value |
|---|---|
| Binary under test | blazar v0.19.0 (rebuilt from main @ d410aca + P7 fixes, `target/release/blazar`) |
| Hardware | RTX 4070 Laptop 8 GiB VRAM, 13 GB RAM, 24 cores |
| OS | Linux; prod daemon :11435 throughout |
| Harness | `scripts/validate.py` (221-check FAST suite, sandboxed XDG, own port) + bespoke probes |
| Engine lanes installed | llamacpp b11370-cuda, sglang 0.5.21, sdcpp master-929, whisper b5130, mistral.rs v0.9.4, piper 2023.11.14, mlx 0.32.0 |
| Receipts | Session scratch ledger: per-phase subdirectories `p0`–`p6`, `golds_*.log`, `validate_run*.log`, `FINDINGS.md` |

Validation axes executed:

- **P0 surface census**: 48 CLI commands / 167 flags / 101 command leaf paths / 345 config
  fields (196-knob Config struct; 179-knob harness `_K` registry) / 90–91 gateway routes
  (100% documented in API_SPEC + README) / 65 `BLAZAR_*` env vars / 7 engine lanes.
- **P1 baseline**: cargo-nextest 1750/1750, clippy clean, fmt clean.
- **P2 read-only CLI**: 28 probes, all PASS.
- **P3 knob matrix**: 158-knob set/get/invalid/unset round-trips + 45-boolean on/off
  matrix + 33-miss census + 13 remaining knobs (69/69) incl. live `/proc/<pid>/cmdline`
  argv A/B for llamacpp and mistral.rs flags.
- **P4 engine lanes (live E2E, spawn→probe→teardown)**: 7/7 green after fixes
  (llamacpp via harness; sglang, whisper, piper, mlx, sdcpp, mistralrs via `p4/lanes.py`
  with VRAM-before/after and child-argv assertions).
- **P5 command probes**: all 48 commands exercised incl. the 8 the harness registry had
  missed (connect, explain, model-doctor, prune, replicate, route, storage, warm).
- **P6 env overrides**: 65 env vars probed (2 harness probe bugs found and fixed, 0 product bugs).
- **P7 fix wave**: all real defects fixed with unit pins + live re-validation.

## 2. Defect ledger (found → disposition)

| ID | Defect | Disposition |
|---|---|---|
| BUG-1 | `/v1/messages` non-stream "empty content" | **Dissolved** — thinking-only response with `stop_reason=max_tokens` is faithful Anthropic behavior; harness probe misread. Probe fixed to assert well-typed blocks. |
| BUG-2 | `config unset <bare section leaf>` claimed "not pinned" while the pin stayed | **FIXED** — shared key classifier; unset now resolves section leaves (single-section scan, multi-hit ambiguity teaching). |
| BUG-3 | `config get <bare>` on 38 Option-None leaves → "unknown config key" | **FIXED** — get falls back to schema knowledge → `<not set>`. |
| BUG-3b | Section parents: get "unknown key", set → raw TOML parse error | **FIXED** — parent get prints pins/teaching with full leaf list; parent set teaches `parent.leaf value` form. |
| BUG-4 | `semantic_cache.ttl_secs` get 600 vs `warm_peg.default` `<not set>` asymmetry | **Dissolved** — file-template contents difference; consistent file-view design. |
| BUG-5 | Doctor/list blind to missing blobs | **Dissolved** — doctor warns ("unreadable or incomplete: … re-pull or rm") and list shows `missing!`; earlier probe was rc-only and missed warn rows. |
| BUG-6 | mistralrs default-shaped chats 400 "model not found" through gateway | **FIXED** — proxy mutation clobber chain replaced by single-parse pipeline (`apply_child_request_mutations`); response/SSE model re-stamp added so child ids never leak. |
| BUG-7 | mlx lane healthz "-1" | **Dissolved** — `/healthz` is `text/plain ok` by design; probe JSON-parsed it. |
| BUG-8 | 10 stale goldens | **FIXED** — every diff reviewed against CHANGELOG + live binary as intended evolution (slots columns, +8 commands, +5 knobs, doctor engine rows, adaptive restyle), regenerated, 19/19 PASS. |
| BUG-9 | mlx lane spurious `ctx_truncated`/`ctx_near_limit` (ctx=0) | **FIXED** — sentinel treats ctx 0/absent as unknown; never flags on unknown windows. |
| BUG-10 | model-doctor table `TESTED unknown` | **FIXED** — numeric epoch rendered as `YYYY-MM-DD HH:MM:SS UTC` (dependency-free civil-days conversion). |
| BUG-11 | `mmproj_path = ""` in a hand-edited/legacy store read as projector-attached | **FIXED** — `model_from_row` normalizes empty-string dialect to `None` (same precedent as `components`); a text model can no longer be misclassified as multimodal (cache_reuse skip, mmproj lazy/offload). Found via harness rebase incident H-2. |
| H-1 | Full-mode run deleted a real-store blob (`Qwen3-0.6B-Q4_0.gguf`) | **FIXED (harness)** — pull-battery cleanup `rm` followed the copied DB's absolute real path after the pull deduped onto the existing row. Sandbox now rebases every model/loras path onto sandbox hardlinks (destructive ops can only unlink sandbox links); both battery cleanups made dedup-aware; product exonerated (alias-safety held, adopted-path delete is by design). Real store self-healed via `blazar pull`. Proven live: sandbox `rm qwen3-0.6b` leaves the real blob intact. |
| H-2 | Post-rebase run failed `cache_reuse` — rebase wrote `''` for NULL `mmproj_path` | **FIXED (harness + BUG-11)** — rebase preserves NULL; product hardened to tolerate the empty-string dialect regardless. |
| H-1r | Run #6 same sentinel FAILs — the H-1 `preexisting` guard was computed but never consumed | **FIXED (harness)** — guard wired to the cleanup `rm`. |
| H-3 | Run #7 (first-ever deep command lanes): `run.miss-pulls`/`pull` probes assumed an empty store; `GH_TOKEN` lost at relaunch | **FIXED (harness)** — probes made dedup-aware (`row_existed` branches); token sourced via `gh auth token`. The silent dedup is the product's data-hazard guard (refuses to re-pull over an existing row) — correct behavior. |
| H-4 | Run #8: `_promote_fixture` sqlite3.Row iteration crash; `list` table-truncation blindness; `--tag` flag that does not exist | **FIXED (harness)** — `row.keys()` iteration; `list --json` JSONL parsing; positional TAG conflict probe (`engine update --all b000000` exercises the real rc1 teaching). |
| H-5f | Auto-boot burned the full 15s systemd poll whenever a user unit exists but serves another port | **FIXED** — after a 5s grace, one `is-active` query decides: only an `activating` unit keeps the budget; everything else yields to the detached fallback (exit-watched retry converges if the unit binds late). Live-measured 20s → 8.2s on this box. |
| H-5 | Run #9: `whisper.transcribe`/`run.miss-pulls` rc=1 — CLI auto-boot bind race during phase-daemon churn | **FIXED (harness + product)** — harness: `ensure_phase_daemon()` health-poll before auto-boot lanes. Product root cause (spawn-once, death-blind boot loop): `ensure_daemon` now keeps the child handle and retries a spawn ONLY after an observed child exit — bounded (3 attempts, 1s backoff, healthy-recheck between attempts), never stacking daemons. Unit-pinned policy + live-proven all three scenarios (free-port clean boot unchanged; port race recovers to rc=0 via retry; permanent holder fails fast in 14s with honest teaching vs 30s blind stall). |

Hidden breakage caught beyond the ledger: a deleted model blob with a live DB row
(self-healed via `blazar pull`; surfaced the doctor-warn behavior above), and stale
interrupted `.part` pulls in old sandboxes.

## 3. Fix detail and pins

### 3.1 Gateway request pipeline (BUG-6) — `crates/blazar-gateway/src/proxy.rs`
Root cause: three body mutators (`inject_include_usage` → `rewrite_child_model` →
`normalize_think_for_engine`) each re-serialized from the original pre-parsed body,
dropping predecessors' edits; the mistralrs think bridge fires on every
think-control-free chat, so the child-model stamp was always lost → child 400.
Fix: single parse, ordered mutations on one `serde_json::Value`, single serialization;
byte-identical passthrough when nothing changed. Response side: buffered and SSE paths
re-stamp `model` to the caller's exact spelling (new `SseRestamper`, frame-preserving).
Pins: 6 ported wrapper tests + `mistralrs_default_chat_keeps_model_stamp_and_pins_think`,
`mistralrs_stream_chat_keeps_usage_and_model_stamp`, `sglang_stream_chat…`,
`mlx_stamp_applies_across_routes`, 4 SSE restamper tests. Gateway suite 546/546.
Live: mistralrs/mlx/sglang lanes green with `resp_model` = store name; 10/10 SSE frames clean.

### 3.2 Config key resolution (BUG-2/3/3b) — `crates/blazar-cli/src/main.rs`
New shared classifier (root knob → section parent → single-section leaf → ambiguity →
unknown) used by get/set/unset bare arms; root-scoped writes can no longer clobber
same-named pins inside `[engine_env]`. 7 unit pins + live 22/22 matrix. Full suite
1765/1765 (baseline 1750 + 15 new).

### 3.3 Sentinel context window (BUG-9) — `crates/blazar-gateway/src/sentinel.rs`
`ctx = 0` (mlx "managed by runtime") mapped to unknown; truncation/near-limit guards
skip unknown windows. Pin: `zero_ctx_window_is_unknown_not_a_ceiling`. Live mlx lane:
zero spurious warnings (was 2+ per chat).

### 3.4 Doctor certificate rendering (BUG-10) — `crates/blazar-cli/src/main.rs`
`epoch_to_utc` (Hinnant civil-from-days, no datetime dependency); TESTED row renders
human-readable UTC. Pins: calendar boundaries + render. Live receipt: 2026-10-03 10:58:43 UTC.

## 4. Engine lane E2E results (P4)

| Lane | Model | Result | Notes |
|---|---|---|---|
| llamacpp | qwen3-0.6b Q4_0 | PASS | harness lane; all API phases |
| sglang | qwen3-1.7b (safetensors) | PASS | chat 200, clean teardown, resp_model stamped |
| mistralrs | qwen2.5-0.5b-instruct GGUF | PASS after BUG-6 fix | was 400 on every default-shaped chat |
| sdcpp | qwen-image-2.1 Q6_K | PASS | 512×512 PNG magic; vulkan VRAM accounted |
| whisper | ggml-base | PASS | transcription 200 (tone input → `[Music]`) |
| piper | en_US-amy-medium | PASS | RIFF WAV 125 KB |
| mlx | qwen2.5-0.5b-instruct-4bit | PASS | serves on Linux CPU backend; routing + teardown clean; BUG-9 fixed |

Cross-lane probe: piper synth → whisper transcribe round-trip available in `p4/` receipts.

## 5. Harness hardening (validate.py)

The full-mode (non-FAST) runs surfaced six harness incidents (H-1, H-2, H-1r, H-3,
H-4, H-5 in the ledger above) — every one root-caused to the harness, the product
exonerated each time, and each fix proven by a subsequent run. The recurring classes:

- **Store-sharing safety (H-1/H-1r)**: the sandbox copies the real DB but originally
  kept absolute blob paths — a destructive probe could unlink real files. The sandbox
  now rebases every model/loras path onto in-sandbox hardlinks before any probe runs,
  and battery cleanups are dedup-aware (never `rm` a row the harness did not create).
- **Dialect preservation (H-2)**: NULL DB columns must stay NULL through rebase.
- **Probe assumptions (H-3/H-4)**: dedup-aware branches for stores that already hold
  the model; assertions read `list --json`, never adaptive-width table stdout; probe
  flags verified against `--help` before use.
- **Daemon churn (H-5)**: `ensure_phase_daemon()` health-polls before lanes that
  auto-boot a daemon, avoiding the CLI detached-spawn bind race.

Earlier fixes carried forward: command registry +8 commands/+12 leaf paths,
knob registry +4, whisper env probe 19→2000 (floor 1000), lora.add real-adapter
staging + refusal row, `engine update --all` evidence + boundary + conflict rows,
BUG-1 probe asserts block types, goldens regenerated after review.

## 6. Final suite status

**validate.py run #10 (FULL mode, post-fix binary + hardened harness): 226 checks, 0 FAILED.**
Command coverage 112/112 leaf paths; config coverage 227/227 knobs — 250/250 knob
flows verified with evidence (argv, file, or boundary classification); gates 10/10;
goldens 19/19; parity 22/22; user config.toml verified untouched. Deep batteries all
executed live: `engine.build` compiled llama.cpp b11375 from source, `whisper.transcribe`
rc=0, `run.miss-pulls` dedup arm, quantize happy/refusal/imatrix, tts pull+synthesize,
`engine update --all` live walk, predictive_preload boundary-classified (VRAM admission
floors exceed the 8 GiB card — correct classification, not a skip).

| Suite | Result |
|---|---|
| cargo-nextest full workspace | **1767/1767 PASS** (baseline 1750 at session start + 17 new pins) |
| cargo clippy --workspace --all-targets | clean (only pre-existing proc-macro-error2 future-incompat notice) |
| cargo fmt --check | clean |
| validate.py FAST (run #3, post-fix) | **221/221 PASS** (run #2 same tree pre-fix: 16 fails) |
| validate.py FULL mode | see receipt run below — deep batteries: predictive_preload, engine update --all live walk, engine.build, whisper.pull+transcribe, tts pull+synthesize, quantize happy/refusal/imatrix, run.miss-pulls |
| Engine lanes E2E | 7/7 PASS (llamacpp, sglang, mistralrs, sdcpp, whisper, piper, mlx) |

## 7. Reproduce

```bash
cargo nextest run -E 'all()'          # 1767 tests
python3 scripts/validate.py --smoke   # FAST subset
python3 scripts/validate.py           # full suite (all deep batteries)
GH_TOKEN=$(gh auth token) python3 scripts/validate.py   # with engine.build lane
```

Receipts: session scratch ledger (`FINDINGS.md` is the session ledger; per-phase
subdirectories `p0`–`p6`, `golds_*.log`, `validate_full_final*.log` run series).
