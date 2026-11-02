# Magnitude harvest - finalized high-ROI roadmap

**Date:** 2026-10-28
**Basis:** comparison doc `2026-10-28-magnitude-comparison.md` + codebase seam verification (CodeGraph + source reads, this session)
**Scope decision:** everything at the orchestration layer that Blazar's existing machinery can absorb. Nothing at the kernel layer (own compiler/kernels = Magnitude's thesis, ~565k LoC, not our leverage).

---

## 0. Verified seam inventory (what the waves plug into)

Every claim below was verified in source this session - these are the load-bearing anchors:

| Seam | Location | Status |
|---|---|---|
| `Tuner` (bench/load/replica/adopt) | `crates/blazar-runtime/src/bench.rs:68` | exists; `bench_default` is deliberately "no grid" (line 641) - **the gap F1 fills** |
| Bench receipt persistence w/ evidence axes | `put_bench_result_with_axes` + `EvidenceAxes { quant, hw_fingerprint, workload, profile_hash }` (store, called from `bench` cmd main.rs:13463-13479) | exists - receipts are already keyed by hardware fingerprint + quant |
| `autopilot` wiring | `autopilot_recs` main.rs:9079 - already recommends `bench` / `tune` / `model-doctor` from stored timestamps | exists - F1/F2 results surface here for free |
| `ModelOverride` overlay (39 fields) | config.rs:1307 - `cache_type_k/v`, `spec`, `slots`, `pin`, `chat_template_kwargs`, ... | exists - tuned winners land as overlay rows, reversible |
| Spec pair catalog + compile gates | catalog.rs `spec_pairs`; profile.rs pair-gate tests (`unit__spec_dspark__*`, `unit__spec_auto__no_pair_for_model__*`) | exists - prefix-table only, no acceptance data (**F2's gap**) |
| Spec governor + adaptive slots | supervisor.rs + `spec_auto_manage` (default-on since 2026-10-27) | exists - runtime safety net for F1/F2 pins |
| Capacity census | gateway capacity.rs `/api/capacity` (device census + per-resident VRAM) | exists - F3/F5 input |
| Intent postures | `intent_plan` main.rs:9920 (6 postures) | exists - F3's preference axis |
| SPEED column / N-col list renderer | `render_list_table` (uzu wave F1) | exists - F3/F4 display |
| Eviction victim policy | supervisor.rs `victim_key` (+ pinned-protection unit tests) | exists - untouched by these waves |
| Route census rule | gateway lib.rs + census + inventory JSON + compat test (same-change rule) | constraint for F4's new route |

**Design constraints honored:** every wave is receipt-gated (no unverified defaults flip), overlay-reversible (pins are `config unset`-able), governor-protected (spec changes stay park-able), and store-schema changes bump `SCHEMA_VERSION` with an upgrade-pin test (durable lesson from the security wave).

---

## 1. F1 - Bounded knob-search auto-tuner (`blazar tune --search`) - **top ROI**

**Gap.** `blazar bench` runs one deterministic profile; `tune` does three fixed A/Bs (load-warmup, replicas, ctx). Nobody searches the knob space. Magnitude's entire pitch is per-device search - at the kernel layer. The same doctrine applied to our engine-reachable knobs is the cheapest decode-speed counterattack available.

**Knob ladder (llama.cpp lane first; per-lane tables later):**

| Knob | Candidates | Cost model |
|---|---|---|
| `cache_type_k` / `cache_type_v` | f16, q8_0, q5_0, q4_0 (v follows k by default) | KV-bandwidth-bound decode wins; quality gate required |
| spec mode | off, ngram(auto), catalog pair/MTP | verify-width arithmetic vs draft cost |
| `spec_draft_n_ubatch` | engine default, 128, 256 | draft-path batch shape |
| threads | auto, physical, physical/2 | CPU-lane / hybrid offload relevance |
| warmup | on/off (reuses `load_search` evidence) | cold-start vs idle CPU |

**Algorithm (Magnitude tuning.md doctrine, adapted):**
1. **Budget:** wall-clock allowance (default 120s, `--budget`), with a protected confirmation reserve (~20%) - no candidate promotion after the clock; overrun = published default, never extended (their "failed budget result, not permission to extend").
2. **Warm-start:** seed ladder order from the closest prior receipt with matching `EvidenceAxes` (hw_fingerprint + quant + lane); same-identity reuse never re-times (their "a later load consumes that result without new timing").
3. **Screen-then-confirm (elimination):** every candidate measured once at the serving-shape probe (decode tg at target ctx - llama-bench already emits pp/tg rows); top-2 survivors re-measured x3, **median** wins; noise gate = winner must beat incumbent beyond spread (their "gain beyond measurement noise").
4. **Quality gate = hard constraint, never a term:** greedy-parity check vs baseline config (reuse the bench harness's determinism receipts discipline - 20/20 ratio-1.0 pattern); cache-type candidates additionally run a short coherence probe. Faster-but-degenerate = rejected before timing counts (their "numerical eligibility is a hard constraint").
5. **Retained evidence:** every measured candidate persists as a bench-receipt row (config, ts, verdict) - not just the winner (their retention doctrine) - so `--budget` extensions and α-tuning re-rank without re-running.
6. **Publish:** winner lands as a `model_overrides` row (fields exist) + `tuned_at` stamp; autopilot's existing "bench newer than tuned" rec drives refresh cycles.

**Seams:** `Tuner::run`/`adopt` (bench.rs), store receipts, `ModelOverride`, `autopilot_recs`, knob_registry row, validate.py knob row + golden (CLI output changes), docs 7.SETUP/9.USAGE.

**Effort:** M (one focused session + validation session). **Expected win:** 5-30% decode on KV-bound quants + spec-positive workloads; converts every existing bench receipt into a warm-start asset. **Risk:** low - search is offline (spawned llama-bench/server children, ephemeral ports, kill discipline already in `Tuner`), pins reversible, governor still guards runtime.

---

## 2. F2 - Drafter catalog + acceptance receipts (data-driven "best drafter per model")

**Gap.** Magnitude ships a matched drafter for *every* catalog model (MTP/DFlash/DSpark/LiLiCorr). Blazar has: prefix-table `spec_pairs` (qwen3-*->qwen3-0.6b, qwen3.5-9b MTP), ngram fallback (governor-protected), and an open size-aware-policy question with n=2 data points (MEMORY).

**Algorithm:**
1. **Acceptance telemetry persistence:** the governor already measures acceptance at runtime - persist per (model, drafter) -> acceptance α, net t/s, parked-count as EvidenceAxes receipts (new small table, same schema-bump discipline).
2. **Drafter selection = expected-tokens model:** for drafter with measured acceptance α and width d, expected accepted tokens/round = Σᵢ₌₁..d αⁱ; choose argmax of measured net t/s across the 2-3 viable candidates (catalog pair, MTP head, ngram). Width follows measured acceptance and measured costs - never a fixed constant (their "draft as wide as pays").
3. **`blazar tune --spec`:** F1's search restricted to the spec axis (off / ngram / pair / MTP x ubatch), pinning the winner; every run banks an acceptance receipt even when the winner is "off" (honest negatives).
4. **Size-aware policy:** becomes an emergent table (α + net t/s receipts per size class) instead of a guessed gate - closes the open follow-up with data.

**Seams:** catalog.rs pair table (+ per-pair metadata), supervisor governor hooks, bench receipts, profile.rs compile gates, autopilot rec (`spec unpinned & receipts exist -> tune --spec`).

**Effort:** M. **Win:** closes the most visible decode gap vs Magnitude without owning kernels; every model gets a receipt-backed drafter decision.

---

## 3. F3 - First-load assessment + ranked install (`blazar plan --install`)

**Gap.** Magnitude's best consumer UX: hardware assessment -> ranked recommendations (fastest->smartest) with intelligence index, memory estimate, speed estimate. Blazar has every ingredient (census, fit ladder, receipts, SPEED column) but no pull-recommendation flow - `plan` only previews runs of *installed* models.

**Algorithm:**
1. **Hard filter:** census (VRAM, unified memory) x fit ladder (`estimate_kv_vram_charge` + quant size) at target ctx -> compatible set.
2. **Score:** `score = w_speed | norm(speed) + w_iq | norm(intelligence_index)`; preference axis reuses the intent-posture vocabulary (`fastest|faster|balanced|smarter|smartest` maps onto w_speed/w_iq); speed = receipt tg when it exists, else family size-class prior (labeled estimate, never guess-presented-as-measurement).
3. **Output:** ranked pull menu (name, quant, mem estimate, fit verdict, SPEED/measured column, one-line why) -> `blazar pull` direct.
4. **Intelligence index:** curated static per-family table shipped in the catalog (data maintenance, not code risk); sourced values documented.

**Seams:** capacity census, fit ladder, catalog, pull, `render_list_table`, `plan` cmd family, docs.

**Effort:** M (mostly UX + data). **Win:** the on-ramp Magnitude wins consumers with, delivered headless.

---

## 4. F4 - Usage analytics (`/api/usage` + `/ui` panel + `blazar usage`)

**Gap.** Magnitude's Usage tab: per-model input/cached/output tokens, gen speed, TTFT, today/all-time. Blazar emits request-log lines and /metrics gauges but keeps no rollups.

**Algorithm:** daily rollup table `(model, day) -> (in_tokens, cached_in, out_tokens, gen_secs, ttft_count, ttft_sum, reqs, errs)` fed from the request-completion path (same source as request-log); `/api/usage?range=today|all`; `/ui` panel + `blazar usage` table. Bounded work: one store table (SCHEMA_VERSION bump + upgrade-pin test), one route (census-rule: route + census row + inventory JSON + compat test, same change), one panel.

**Effort:** S-M. **Win:** dashboard parity; cheap, visible, agent-friendly (TTFT by model is the metric agents feel).

---

## 5. F5 - Legacy-GPU Vulkan engine lane

**Gap.** Magnitude explicitly excludes pre-sm_80 NVIDIA from CUDA (Turing->Vulkan, Pascal unsupported) and floors macOS at 15. Blazar's coverage follows upstream engines - a Vulkan-built llama.cpp channel covers old NVIDIA + AMD + Intel that Magnitude's floors reject, turning their exclusions into our moat.

**Algorithm:** engine registry row (Vulkan build tag, channel policy), census-driven routing (CUDA lane preferred when sm>=80; Vulkan lane otherwise), doctor lane + validate row, headless-CI smoke where possible (Vulkan on CI runners is the hard part - may stay receipt-manual on real hardware).

**Effort:** M (packaging + validation; zero architecture change - engine-channel machinery is proven). **Win:** "runs where they can't" story; old-GPU user base is unserved by Magnitude.

---

## 6. F6 - `connect --install-skill` (S)

Connect lanes gain an optional skill/prompt-file install per harness (Magnitude auto-installs a skill into Pi). Per-harness template in the existing connect lane table. Stickiness per agent, one afternoon.

## 7. F7 - Native packages (M)

deb/rpm/pacman/dmg/exe in release.yml (packaging/ dir exists; binaries + crates.io today). Distribution parity with their signed installers; signing keys are the only external dependency.

## 8. F8 - Head-to-head bench vs `magnitude serve` (S-M, evidence)

Settles the open perf question with receipts: shared-model intersection (qwen/gemma/llama families), RTX 4070 rig, `bench_matrix.py` extended with a magnitude lane (OpenAI dialect -> trivial to add as a "remote-style" reference lane). Reproducibility receipt per repo mandate. Do after F1/F2 so our lane runs with tuned pins.

---

## 9. Priority order & sequencing

| Order | Wave | Effort | Decode win | UX win | Coverage win | Depends on |
|---|---|---|---|---|---|---|
| 1 | F1 knob-search tuner | M | ●● | ● | - | receipts (exist) |
| 2 | F2 drafter receipts | M | ●● | ● | - | F1's search harness, governor (exists) |
| 3 | F3 ranked install | M | - | ●●● | - | census+fit (exist) |
| 4 | F4 usage analytics | S-M | - | ●● | - | request path (exists) |
| 5 | F5 Vulkan lane | M | - | - | ●●● | engine channels (exist) |
| 6 | F6 skill install | S | - | ● | ● | connect lanes (exist) |
| 7 | F7 packages | M | - | - | ●● | release.yml |
| 8 | F8 head-to-head | S-M | evidence | - | - | F1+F2 |

F1->F2 is the decode counterattack (directly answers Magnitude's kernel lead with orchestration-layer search). F3->F4 is UX parity. F5 is the coverage moat. All waves preserve existing behavior by construction (additive pins, additive tables, additive lanes; nothing flips a default without a receipt).

## 10. Explicitly rejected (with reasons)

- **Own-kernel engine / numerical compiler** - Magnitude's thesis, ~565k LoC, different company-scale bet; Blazar's leverage is the layer above (3-28x receipts already live there).
- **Electron desktop app** - heavy dependency posture; F3+F4 deliver the same consumer value headless through `/ui`.
- **NVRTC-style on-device kernel compilation** - kernel layer; engines own it upstream.
- **Per-family hand-authored drafter kernels (LiLiCorr-class)** - F2's data-driven selection over engine drafters is the 80/20.

## 11. Evidence index

Seam verification: CodeGraph explorations + source reads this session (bench.rs Tuner/bench_default/load_search/replica_search/adopt; main.rs bench cmd:13419-13484, autopilot_recs:9079, intent_plan:9920; config.rs ModelOverride:1307 + cache-type precedence tests; catalog spec-pair gates in profile.rs tests; supervisor victim_key tests; capacity.rs census; responses.rs / sentinel.rs / http_pool.rs read for pattern discipline). Magnitude-side inputs: `2026-10-28-magnitude-comparison.md` and its evidence index (tuning.md, drafting-and-verification.md, scheduling.md, compatibility.md, models.mdx).
