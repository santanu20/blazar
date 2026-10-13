# E2E Perf Audit Round 2 — 2026-10-13

**Subject:** the uncommitted 2026-10-12/13 wave (model scan/import `scan.rs`, GH atom fallback `gh.rs`, ollama-store reuse `registry.rs`, schema v16 `model_modalities`, gateway streaming/non-stream fast paths in `proxy.rs`/`openai.rs`/`translate.rs`/`classify.rs`, honest-stop + doctor-skew in `main.rs`, `validate.py` battery extensions) — validated end-to-end with zero quality loss and zero behavior break, per the standing directive.

**Verdict: GREEN.** Every lane sits inside the direct-control noise band; two lanes improved evidence (byte-parity now PROVEN, not inferred). The wave also entered the repo with its lint gate closed (fmt + clippy were failing at session start — audited and fixed behavior-zero before any measurement).

Chain: measurement → attribution → verdict, with reproduce commands per section.

---

## 1. Environment receipt

| Item | Value |
|---|---|
| Repo | blazar @ ebab414 + uncommitted wave (16 files; post-lint-fix tree verified by `git diff --numstat` sweep) |
| Binary under test | `target/release/blazar`, sha256 `ccb3a2b37a43d1eddeb67d09033459c21def006944ff7e3d0e7a81c30ba75ada` |
| Engine | llamacpp **b11580-cuda** (`~/.local/share/blazar/engines/b11580-cuda/llama-b11580/llama-server`) |
| Model | qwen2.5-0.5b-instruct Q4_K_M (store row, gguf/llamacpp lane) |
| Hardware | RTX 4070 Laptop 8188 MiB, x86_64, MemAvailable ≥ 7.2G at each stage |
| Test gauges | workspace nextest **2183/2183**, `cargo fmt --check` clean, `cargo clippy --workspace --all-targets` 0 errors |
| Prior audit | docs/research/2026-10-12-e2e-perf-audit.md (post-fix baseline: 96.0 µs/token daemon CPU, SseThinkFilter 0.37%) |

## 2. Lint-gate closure (audit finding #1)

**Finding:** the wave had never passed the fmt/clippy gate — `cargo fmt --check` FAIL, clippy 101 hits incl. 7 errors (runtime crate) + 2 surfaced later (gateway crate, masked until runtime compiled).

**Fixes (all behavior-zero, convention `#[allow(clippy:…)] // <why>` used 129× elsewhere):**
- `gh.rs`: doc backticks for env names; `collapsible_if` collapsed with let-chain.
- `registry.rs`: `too_many_lines`/`too_many_arguments` allows with why-comments (cohesive pull/repoll decision fns).
- `scan.rs`: `#[must_use]` on `scan_reusable_models`.
- `openai.rs`: extracted `model_modality_cache()` helper from `models()` (125→<100 lines, store-call order preserved); `and_then(Result::ok)`; doc backticks.

**Pin:** full nextest re-run after the fixes — 2183/2183 PASS (76.7 s). Gate green: fmt, clippy, tests.

## 3. Streaming profile (greedy load)

Method: sandboxed daemon (validate.py Sandbox/Daemon rig), 14 requests × 1024 tokens greedy, `perf record -F 499 --call-graph dwarf` on the daemon; ad-hoc driver (same shape as the 2026-10-12 audit rig), receipts summarized in the tables below.

| Metric | r2 | 2026-10-12 post-fix | Verdict |
|---|---|---|---|
| Daemon CPU / token | **96.8 µs** | 96.0 µs | flat (±5% band) |
| Aggregate decode | 487.7 t/s | ~330 (box lighter now) | control shifts, ratio holds |
| Context switches / token | 1.61 | ~1.6 | flat |
| Wave symbols (profile share) | SseThinkFilter::feed 0.29%, LineBuffer::feed 0.20%, sentinel cluster ~3%, finalize/suppress **no rows ≥0.05%** | SseThinkFilter 0.37% | sub-noise |

Attribution (both perf events, halves summed): kernel sched/mm/syscall ~35%, tokio+hyper+http ~30%, loopback netfilter ~11%, serde/json ~6%, allocator ~5%, observation ~3% — the same structural split as 2026-10-12. Residual remains structural (kernel + wakeup), not wave-related.

## 4. A/B campaign (direct vs gateway, b11580-cuda)

Command (receipt `bench-artifacts/20261013-e2e-perf-audit-r2/campaign_cmd.txt`): bench_matrix direct+blazar, conc sweep 1/2/4/8, perf-only lanes. 24 cells landed (`cells.jsonl`, `summary.txt`, `report.md`, charts).

| Metric | r2 | 2026-10-12 ref | Verdict |
|---|---|---|---|
| Direct np1 decode (control) | 349.5 t/s | 347.4 | control band ✓ |
| Blazar default decode | 319.0 t/s | 324.0 | −1.5% (noise) |
| Gateway overhead (vs direct np4) | **−6.8%** | −7.1% | flat |
| TTFT p50 | 9 ms (direct 6 ms) | same shape | known ~3 ms gateway add |
| C4 ITL p99 | **6.63 vs 8.77 ms** | — | gateway wins |
| C8 ITL p99 | **6.17 vs 12.05 ms** | 5.78 vs 10.08 | gateway wins ~2× (held) |
| Greedy transparency | **14/20 exact** (direct self 20/20) | 14/20, ratio 0.941 | identical band; divergence = multi-slot batching numerics, not translation drift |

Campaign EXIT=1 is ollama-reference-lanes only (no comparable ollama model installed — environmental; direct/blazar cells unaffected).

## 5. Non-stream byte-parity (suppress-only plain-chat lane) — PASS

The wave's zero-parse fast path claim: a suppress-only, marker-free non-stream body must ship engine bytes untouched (single parse, conditional re-serialize skipped when nothing changed).

Probe (ad-hoc, method exact): sandboxed daemon; same temp-0 payload posted gateway ×2 and direct-child ×2 (child bearer minted per-child via `--api-key-file`, read from child argv); bodies normalized by byte-splice masking ONLY volatile numerics (`created`, `id`, `*_ms`, `*_per_second` — proven per-request-varying even direct-vs-direct); token counts and key order left byte-exact.

**Result: all four normalized bodies identical** (`sha_norm` equal across gw1/gw2/d1/d2). Any gateway re-serialization would reorder keys (llama-server nlohmann sorted-key output vs serde insertion order) and diverge — it does not. Zero-parse lane proven live, byte-level.

## 6. Gap re-checks (carried from round 1)

- Owner-scoped store queries (schema v15): EQP re-verified — responses hit `idx_responses_conversation`, jobs scan is LIMIT-terminated with temp-B top-N on a quota-bounded table; sub-ms. No gap.
- `path_gated_for` sglang const-scan (~7 entries) and singleflight config-first gate: no gap.
- Sentinel per-frame parse: off-relay by design (structural floor, 2026-10-12 verdict stands).

## 7. validate.py full battery — PASS

Full unfiltered run of `python3 scripts/validate.py` (preconditions: stub rebuilt first, main daemon stopped, `BLAZAR_BIN=./target/release/blazar`, `BLAZAR_VALIDATE_MODEL=qwen2.5-0.5b-instruct`). Two runs total:

- **Run 1 (EXIT=1, 399 checks / 7 FAIL)** — all three clusters harness-side, not product bugs:
  1. `cli ps` lane rc=-13 once (SIGPIPE on a socket-reset write) — subprocess capture verified sound; **did not recur in run 2** (4/4 PASS rc=0 model_visible=True) → classified transient environmental blip.
  2. Goldens ×4 `completions.{bash,fish,powershell,zsh}` sha drift — the wave's new `import --scan --json` clap surface changed completions output. Blessed via `--phase=golds --update-goldens` (same designed mechanism as config.fresh-keys 164→165 precedent); golds now 19/19.
  3. Coverage ×2 `flag.import: unmapped live flag --json/--scan` — fixed by adding an `import.scan` + `import.scan.json` battery (rc + SOURCE/MODEL header; JSONL rows with keys source/id/bytes/suggestion) and mapping both flags to it in `_FLAG_EVIDENCE['import']`. Live-verified first (4 reusable models on box → rows=4, lines=9).
- **Run 2 (EXIT=0): 396 checks, 0 FAILED** — every phase fail=0 (inventory 41, api 47, wave 32, golds 19, wave13 19, parity 22, sentinel 16, behavior 11, auth 6, …). Battery side-effects self-restored: engine active left on `b11580-cuda` (anchor rolls to a fresh source build mid-battery, then engine.update/rollback scenes restore), user `config.toml` untouched (sha 3333401d…).
- Harness delta for this audit (validate.py only, `/execute` validation-infrastructure authority): the `import.scan` battery + flag-evidence rows above; goldens + goldens.json blessed. No product code touched post-campaign.

## 8. Verdict + follow-ups

- **No conditional fix triggered** — every measured lane within the direct-control noise band; max-3-fix-cycles budget unused.
- Follow-up (environmental, non-blocking): pull a qwen2.5-0.5b-class ollama tag to re-arm ollama reference lanes in future campaigns.
- Follow-up (ops): systemd unit `blazar` can show `failed` while a daemon process still serves 11435 (pid outlives unit tracking) — `doctor`-skew catches the binary side; consider a unit-level watchdog note in docs.

## Reproduce

```bash
# lint gate + tests
cargo fmt --check && cargo clippy --workspace --all-targets && cargo nextest run --workspace
# campaign
uv run python scripts/bench_matrix.py --model qwen2.5-0.5b-instruct --engines b11580-cuda \
  --providers direct blazar --conc-sweep 1,2,4,8 --skip-ppl --skip-tools --skip-quality \
  --skip-media --skip-features --skip-ctxcurve --skip-reshape --skip-idle --skip-variants \
  --blazar-bin ./target/release/blazar \
  --artifacts-dir bench-artifacts/20261013-e2e-perf-audit-r2 \
  --md bench-artifacts/20261013-e2e-perf-audit-r2/report.md
# full battery (preconditions: daemon stopped, stub rebuilt, BLAZAR_BIN + BLAZAR_VALIDATE_MODEL set)
python3 scripts/validate.py
```

Profile (§3) and byte-parity (§5) rigs were ad-hoc sandbox drivers around validate.py's Sandbox/Daemon classes; their methods are specified exactly in those sections and rebuild in minutes.
