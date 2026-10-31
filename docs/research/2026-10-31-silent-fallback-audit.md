# Silent-Fallback Audit — full-codebase line-level sweep (2026-10-31)

Scope: every shipped-source line of blazar (4 crates, `validate.py`, `scripts/`, `sdk/`) audited for
§5 H2-class defects (silent error masking / silent fallback), H3 (catch-all suppression), and H1
(not fail-fast). Read-only audit — no source was modified. Full machine-readable evidence:
`bench-artifacts/20261031-silent-fallback-audit/{findings.jsonl, classified.jsonl, inventory.json}`
(gitignored receipts, local by design).

## 1. Methodology

1. **Mechanical full-coverage scan** — a line scanner examined **all 124 files / 204,563 lines**
   (per-file line sums reconcile with `wc -l`, sum-check OK). 16 pattern families in Rust
   (`.ok()`, `unwrap_or*`, `let _ =`, `if let Ok`, swallow match arms, `filter_map`,
   `catch_unwind`, `is_ok` guards, `map_err(|_|…)`) plus Python (`except Exception`,
   `contextlib.suppress`) and shell (`|| true`, `2>/dev/null`).
2. **Observation heuristic** — a hit is *OBSERVED* (bounded, observable fallback = legitimate per
   H2) when a log macro / teaching / event emit sits within ±6 lines or inside the fallback
   closure (brace-depth scan). Multi-line closures were extension-scanned to their close.
3. **Domain gating** — Option-domain receivers (`.get()`, JSON accessors, precedence chains)
   separated from Result-domain receivers via backward statement reconstruction.
4. **Manual review** — 74 high-risk sites window-reviewed by hand (evidence quoted below);
   remaining shapes classified by rule with their line text retained in `classified.jsonl`.

Honest scope statement: "line-by-line" here = every line machine-examined + every
non-obviously-benign candidate manually windowed. Semantic fallbacks expressed as plain conditionals
(e.g. `if !path.exists() { … }`) are outside pattern reach; the repo-hinted classes (notify results,
fire-and-forget spawns, config-default-on-error) were additionally swept by name.

## 2. Numbers

| Class | Count | Meaning |
|---|---|---|
| TEST | 794 | inside `#[cfg(test)]` — scaffolding, excluded |
| OBSERVED | 170 | bounded + observable (warn/teach/event in scope) — **legit** |
| BENIGN-OPTION | 159 | Option-domain absent-key defaults — designed defaults |
| SILENT-CAND + AMBIG | 1,780 | deep-classified into 17 danger families below |

Raw total 2,903 hits. Per-crate split (all classes): cli 639, core 230, gateway 1,099,
runtime 702, scripts/sdk 133.

## 3. Proven violations (fix recommended)

Each verified against the real source window; severity by blast radius.

| # | Site | Evidence | Why it violates H2 | Severity |
|---|---|---|---|---|
| V1 | `blazar-gateway/src/openai.rs:838` | `let bytes = child.bytes().await.unwrap_or_default();` inside the MCP-lane response rebuild (status + headers forwarded verbatim, line 835–846) | Engine returned success; if the byte body read fails mid-transfer the client receives **the child's success status with an empty body** — a complete-looking wrong answer, no log. Must map read failure → 502 (the `Err(resp)` sibling path already errors properly). | **HIGH** (user-facing answer path) |
| V2 | `blazar-gateway/src/whisper.rs:601` | `let bytes = resp.bytes().await.unwrap_or_default(); Ok((status, ct, bytes))` | Same shape on the transcription passthrough: read failure ⇒ empty body + engine's success status ⇒ client gets an empty/invalid payload as success. | **HIGH** |
| V3 | `blazar-cli/src/main.rs:2932` | `let cfg = blazar_core::Config::load(d).unwrap_or_default();` (doctor lane-state fn) | **Doctor** is the diagnostic tool. A corrupt/unreadable config silently becomes *default* config — lane advice computed against knobs the user is not actually running, no warn. Absent-file → default is fine (`Config::load` creates it); **parse-error must surface as a doctor FAIL row** instead. | MED |
| V4 | `blazar-cli/src/main.rs:3469` | same expression, model-lane doctor path | Same class as V3. | MED |
| V5 | `blazar-gateway/src/batch.rs:209–216` | `let Ok(entries) = std::fs::read_dir(…) else { return 0; }; entries.flatten().filter_map(|e| read_json(&e.path())).filter(…).count()` | `active_batch_count` feeds **owner batch quotas** (BLZ security wave). `read_dir` error → count 0 → quota never trips (**fail-open**); unparseable batch rows silently undercount. Quota controls should fail closed or at least count-unknown. | MED (security-adjacent) |
| V6 | `blazar-runtime/src/engine/gh.rs:694` | `let body = resp.text().await.unwrap_or_default(); Some(parse_asset_hrefs(&body, …))` | Body-read failure yields `Some(empty-asset-list)` — a **fetch failure indistinguishable from "release has no assets"** downstream (engine update decisions). All sibling failures in the same match return `None`; the read failure should too. | LOW-MED |

## 4. Flagged judgment calls (tighten when touched)

| # | Site | Issue |
|---|---|---|
| F1 | `blazar-gateway/src/images.rs:2832–2845` | sdcpp old-engine guard (`build_number < 929` refuse) computes `too_old` via store `.ok()` chains ending `.unwrap_or(false)` — store/manifest error **fails the guard open** (old broken engine accepted silently). Version-parse itself fails closed (`manifest.rs:571/728` → 0 < 929), but the store/manifest *read* does not. |
| F2 | `blazar-runtime/src/daemon.rs:47–50` | pidfile present but unparseable → `existing = 0` → treated as **stale lock → takeover**. A held-but-corrupt pidfile double-boots (bounded by the port bind guard, but takeover-on-parse-failure should refuse with a teaching error instead). |
| F3 | `blazar-gateway/src/failover.rs:118,142,171,237` | `self.state.lock().ok()?` — poisoned mutex silently degrades `plan()`/registry reads to "no chain ⇒ serve base". Repo precedent `config.rs:3646` uses `PoisonError::into_inner`; failover state should match (poison = a panic happened; masking it hides that). |
| F4 | `blazar-cli/src/main.rs:4375,4389,4440,4455` | doctor sections use `store.list_engines()/list_models().unwrap_or_default()` — store **read errors render as empty sections** in the diagnostic tool instead of a FAIL check row (same class as the `ps`-masking bug fixed in the S3 wave; doctor should be the place store errors are *loudest*). |
| F5 | `blazar-gateway/src/ollama.rs:1620–1626` vs `1662–1671` | Same metadata read, two policies: 1666 documents fail-open (`None` ⇒ no refusal) for unreadable GGUF; 1624 caches `supports=false` on **read error** — inconsistent posture inside one feature (thinking-support display). |
| F6 | `scripts/validate.py:2749` | GPU-count probe returns 0 on any exception — a GPU box with a hiccuping `nvidia-smi` is classified CPU-only for memory gating (harness-only blast radius). |

## 5. Systemic postures (design observations, not per-line bugs)

- **P1 — `with_store(|s| … .ok().flatten())`: 44+ single-line sites (+ multi-line variants)** —
  store **errors** are conflated with **absence** repo-wide. Direction is usually fail-closed
  (404 / refuse), so it is not fail-open, but a corrupted store surfaces to users as
  "model not found" everywhere at once. Recommendation: a `with_store_absent` helper that
  distinguishes Err (warn-once, then None) from honest-absent; migrate call sites gradually.
- **P2 — error-detail enrichment idiom** (31 sites): `resp.text().await.unwrap_or_default()` **inside
  `!is_success` branches** — the error always propagates; the default only affects message richness.
  Legitimate; do not "fix" these — they are the codebase's consistent fail-loud idiom.
- **P3 — `let _ = tx.send(...)` / cleanup kills** (~108 sites): channel-closed = receiver-gone is the
  intended Rust idiom; `start_kill()` drops appear only in cleanup paths. Benign.
- **P4 — JSON accessor defaults** (`as_bool/as_u64/as_str().unwrap_or(…)`): Option-domain absent-key
  defaults = API spec defaults (e.g. `stream` absent ⇒ false). Benign.
- **P5 — precedence chains** (`overlay.unwrap_or(global)` in `config.rs`/`profile.rs`): designed,
  documented precedence (pin > adaptive > auto). Benign.
- **P6 — probes** (health/modality/capacity/tokenize, ~140 sites): bounded-timeout probes whose
  failure direction is either no-cache (re-probe later) or estimate-fallback (`bytes/4`), several
  with explicit design comments (openai.rs:76–81, preflight.rs:134). Benign.
- **P7 — shell `|| true` / `2>/dev/null` (26 sites)**: all in install/uninstall/cleanup paths —
  idempotent best-effort teardown. Benign.
- **P8 — validate.py harness**: 57 `boundary()` calls = documented deliberate-degrade design;
  sampled `except Exception` sites are bounded with loud re-raises (3144 raises with log tail).
  bench_matrix `suppress(Exception)` around telemetry polls (3498) slightly weakens receipts —
  acceptable for tooling, noted for polish.

## 6. Explicitly verified legitimate (sample of documented contracts)

`session_identity.rs:92–98` (None ⇒ unverifiable ⇒ caller warns), `store.rs:521–527` (honest
no-hardware-axis), `sentinel.rs:854` (unreadable GGUF ⇒ doctor flags), `ollama.rs:1662–1664`
(documented fail-open distinction), `build.rs:835–839` (documented best-effort floor stamp with
named degraded mode), `audit.rs:76–78` (documented best-effort model sniff), `hf.rs:3071–3087`
(sha256 verify fails **closed** — unverifiable candidate is rejected).

## 7. Recommended fixes (NOT applied — this audit is read-only)

1. V1/V2: replace `bytes().await.unwrap_or_default()` with match → `Err` ⇒ 502 teaching error
   ("engine response truncated: {e}"). Two-line fixes; add gateway unit pins each.
2. V3/V4: doctor paths — on `Config::load` Err emit a FAIL check row (config-parse fail-site class
   already exists in the doctor taxonomy) instead of `unwrap_or_default`.
3. V5: `active_batch_count` — read_dir error ⇒ treat as unknown ⇒ refuse new batches with teaching
   (fail-closed) or surface via warn + skip quota admission for that owner this call.
4. V6: return `None` (align with the sibling `_ => None` arm).
5. F1–F5: guard fail-closed / pidfile refuse-on-corrupt / `into_inner` for failover locks / doctor
   FAIL rows / unify 1624 with the 1666 policy.
6. P1: introduce the warn-once store-error helper; adopt opportunistically.

## 8. Coverage & limitations

- 124 files / 204,563 lines scanned; line sums reconcile. `target/`, goldens, and gitignored
  artifacts excluded by construction (fd over shipped source lists only).
- 794 TEST-class hits excluded from verdicts (test scaffolding legitimately bounds failures).
- `catch_unwind`: **zero** instances repo-wide. `map_err(|_|…)`: 18 sites, all replacing error
  *detail* while still propagating (not silent).
- Pattern-blind spot: conditional-replacement fallbacks; mitigated by the name-sweeps above.
- The two HIGH violations are request-path; both have crisp fixes and pin-testable shapes.

Artifacts: `bench-artifacts/20261031-silent-fallback-audit/` — `findings.jsonl` (2,903 rows,
per-line evidence), `classified.jsonl` (1,780 danger-family verdict inputs),
`inventory.json` (per-file coverage proof).

## 9. Deep verification pass (second pass, same day)

Every PROVEN/FLAGGED verdict re-verified against full caller context; two additional
pattern-blind classes swept. Results:

- **V1 confirmed at full depth** — `openai.rs:810-851`: the MCP-lane `return match` hands the
  rebuilt response (child status + headers + `bytes`) straight to the client; nothing downstream
  validates. Byte-read failure ⇒ success-status empty body. Fix unchanged.
- **V2 confirmed with nuance** — caller `whisper.rs:695-711`: the **diarize** lane re-parses the
  body and fails loud (`from_slice` match with explicit "fail loud, never hand back verbose_json
  under a diarized_json name" contract); the **non-diarize** lanes pass the triple through
  untouched ⇒ empty-bytes-with-200 reaches the client there. Violation stands; primary fix site
  remains `whisper.rs:601` (map read failure → 502 at the source).
- **V3/V4 confirmed precisely** — `config.rs:3528-3540`: absent file → default created
  (documented); corrupt file → `from_toml` returns `Err("parse: …")` → `load` propagates Err →
  the doctor-side `unwrap_or_default()` is what masks it. (from_toml's own api_keys migration is
  deliberately loud — "nothing silently lost" — which makes the doctor masking clearly out of
  posture.) Fix: doctor emits a config-parse FAIL check row instead.
- **V5 confirmed exactly** — `batch.rs:519-523` and `1366-1370`: admission refuses when
  `active_batch_count(…) >= max_active_batches_per_key`; `read_dir` Err ⇒ 0 ⇒ never refuses
  (fail-open) on BOTH dialects (OpenAI + internal JSONL). Unit cover exists only for the happy
  path (`unit__active_batch_count__owner_scoped_non_terminal`).
- **V6 confirmed** — consumers in `engine/mod.rs` (11 sites) and `upgrade.rs` iterate
  `release.assets` find-by-name; `Some(empty)` yields "asset not found" errors that list zero
  available names — network failure misdiagnosed as empty release. Stand LOW-MED.
- **Detached-task panic class — honest downgrade**: 97 `tokio::spawn` sites, zero `set_hook`,
  zero `catch_unwind`. Task panics are NOT silent: the default panic hook prints to stderr
  (captured by the daemon log / journal). Remaining gap is structural-observability polish
  (a panic hook forwarding to tracing), NOT an H2 violation. Recorded as polish, not a finding.
- **Env-override fallbacks** (`env::var(k).ok()…`, 43 sites in main.rs): absent-override →
  default is correct optional-override semantics (absent ≠ error); empty-string-is-unset filter
  (`config.rs:5118/5285`) is deliberate. Benign idiom.

**Pass verdict: 6/6 proven violations and 6/6 flagged sites hold under deep verification;
no verdict changed; one suspected class (task panics) honestly reclassified as polish.**

## §10 — Fix wave addendum (2026-10-31, same day; user approved "all")

All six proven violations plus the P1 systemic posture were fixed, tested, and live-validated.
No flagged site (F1–F6) was touched; those remain posture decisions for the maintainer.

| # | Fix (file:line post-fix) | Shape |
|---|--------------------------|-------|
| E1 (V1) | `openai.rs:838` MCP lane | `child.bytes()` match — read Err → 502 `mcp response body read failed (child status N)` |
| E2 (V2) | `whisper.rs:601` `inference_post` | body read `map_err` → `(502, "whisper inference body read: …")` — fixes all 3 passthrough lanes at source |
| E3 (V3) | `main.rs` `lane_state_for` | Store::open / list_engines / Config::load masks → `LaneState::Missing(Vec::new(), teach)` with unreadable-path reasons; corrupt-config advice now says defaults would be wrong. Adjacent same-class arms in the same fn completed together (disclosed) |
| E4 (V4) | `main.rs` `doctor_routing` | config-first ordering; every store/list/config error → `Check::fail("routing", …)` row instead of silently-empty sections |
| E5 (V5) | `batch.rs` `active_batch_count` | `Result<u64, String>` fail-closed counting: unreadable dir → Err → admission 500; unparseable/torn meta counts as active; `read_dir` NotFound → Ok(0) (first batch never refused). Both admission callers (OpenAI + internal JSONL dialects) map Err → 500. `sweep_expired` gained mtime-based GC of torn metas past the retention window so fail-closed counting cannot wedge the quota |
| E6 (V6) | `gh.rs` `list_releases_via_web` | per-tag `resp.text()` Err → `None` (assets unknown ≠ zero assets); all-None after join → `bail!("asset tables … unreadable across all N feed tags")` |
| E7 (P1) | `state.rs` `with_store` | warn-once `AtomicBool` — first mid-run `Store::open` failure logs `"store open failed … persisted state reads as absent until it recovers"`; success re-arms the flag |

**Gates:** `cargo fmt` clean; `clippy --workspace --all-targets -D warnings` zero workspace
errors; full workspace nextest **2308/2308** (baseline 2185 + 3 new pins:
`unit__active_batch_count__missing_dir_is_zero`, `__unreadable_dir_is_error`,
`__corrupt_meta_counts_as_active`; existing happy-path pin updated to `Result` shape).

**Live validation (XDG-sandboxed serve, real engine row, zero mocks):**
- Corrupt `config.toml` → `blazar doctor` renders `config fail` **and** new
  `routing fail config unreadable — lane advice unavailable: config: parse: TOML…` (E4).
- Torn batch meta parked in the index → `POST /v1/batches` → **400 "key … already running the
  configured maximum of 1 concurrent batches"** (fail-closed counting, E5).
- Batches dir replaced by a file → **500 "batch index unreadable …: Not a directory (os error 20)"** (E5).
- Clean index → upload + batch create → 200 in_progress (no over-refusal regression).
- Dead-proxy `blazar engine update` → loud GitHub transport error (E6 direction); the
  web-lane all-None bail sits behind the 403/429 rate-limit gate — compile + logic verified.
- Garbage store at boot → serve exits fail-fast (pre-existing boot guard); E7's warn lane is
  defense-in-depth for mid-run store breakage (compile-proven; cold start already loud).
- V1/V2 lanes: existing e2e suites (`e2e__whisper_*`, `e2e__mcp__*`) green in the 2308 run;
  the fault shapes are now match arms that cannot return an empty success body.

**Not fixed (open posture decisions):** F1 sdcpp old-engine guard fail-open, F2 pidfile
unparseable → stale-lock takeover, F3 poisoned-lock serve-base, F4 doctor section-level store
masks (fn-level arms fixed in E4; section-level `unwrap_or_default()` rows remain), F5
ollama.rs metadata double-policy, F6 validate.py GPU probe. Residual P1: 44+
`with_store(...ok().flatten())` sites still read store-error ≡ absent (fail-closed direction)
beyond the new warn-once; err() helper types 500s as `invalid_request_error` (pre-existing shape).

## §11 — Flagged-site disposition (2026-10-31, follow-up wave; commits f119861 + c246095)

| # | Disposition | Change |
|---|-------------|--------|
| F1 | Kept open, documented | sdcpp upscale gate comment now states the audited fail-open posture and why (self-describing child response; store errors already warn/fail elsewhere) |
| F2 | Fixed | Unparseable/unreadable pidfile → serve refuses with a remedy naming the exact file (was: guess stale, take over). Live-proven: garbage pidfile → exit 1 + remedy text |
| F3 | Fixed | 4 × `state.lock().ok()?` → `unwrap_or_else(PoisonError::into_inner)` (matches the file's own pin/unpin posture). Poison-recovery pin added |
| F4 | Fixed | `doctor_sglang_fit` / `doctor_kv_budget` emit fail rows on store-open / list_engines / list_models errors (were: silently empty sections). Live-proven with a corrupt store |
| F5 | Fixed | `hf_chat_template` → `Option<String>`; read failures return uncached (unknown), readable-but-templateless still caches false. Cache-unpoisoned pin added |
| F6 | Fixed | `validate.py` GPU probe failure → `None` + WARN (unknown), no longer misread as 0-free/absent; both call sites distinguish unknown from a genuinely full GPU. Live-proven with a failing fake `nvidia-smi` |

Wire-shape follow-up: batch-lane `err()` now types ≥500 bodies as
`server_error` (commit c246095); 4xx stays `invalid_request_error`.

Gates: fmt ✓, clippy -D 0 (workspace), nextest **2312/2312** (4 new pins:
daemon garbage-pidfile, failover poison recovery, think-cache unpoisoned,
err typing).
