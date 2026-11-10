# Adoption Plan — Items 3–7 from the Atomic-Chat Comparison

**Date:** 2026-11-06 · **Basis:** tree `e48142e` (v0.29.0+46). Every seam cited was verified in source this session. Ordered by ROI; each item lists: today's state (verified) → plan → effort → risks → verification.

**Status (2026-11-07):** executed on v0.30.1 tree, all live-validated — Item 3 ✓ (wave 2 curated registry shipped; wave 1 partially via `gguf.rs` `mtp_layers` → `show` draft field; wave 3 = the printed hint lines) · Item 4 ✓ (`engine_recommend` in optimize.rs, `engine recommend` CLI + `/api/engine/recommend` gateway; arch axis intentionally omitted — no architectures in Manifest) · Item 5.1 ✓ (prism lane in registry/capability-lanes.json, ref_sha 6447906) · Item 5.2 DEFERRED (openai-host lane, separate session) · Item 6 ✓ (`--recipe ud-nextn|ud-combined` on `blazar quantize`, FAIL FAST when engine lacks `--tensor-type-file`) · Item 7 ✓ (packaging/ preexisted; added `[package.metadata.binstall]` + README AUR link). Suite: 2494 nextest pass, clippy `-D warnings` clean, fmt clean.

---

## Item 3 — Draft-ecosystem pairing (UDT combined GGUFs + curated collections)

**Today (verified):** `drafts_cmd` (crates/blazar-cli/src/main.rs:13985) is a flat HF search — 4 queries per family (`{family} eagle3 gguf`, `{family} mtp gguf`, `{family} 0.5b/0.6b gguf`), dedup + downloads ranking. Doc comment is honest: "suggestions are candidates, not pairings". Separately: `gguf.rs` already detects MTP tensors (real-MTP tests behind `BLAZAR_TEST_MTP_GGUF`), `profile.rs:997` gates embedded MTP on `engine_has_mtp && mtp_arch_proven`, and the spec knob family (`spec`, `spec-draft-model`, `spec-draft-n-max`) exists in the registry.

**Plan (3 waves):**
1. **Combined-file recognition** — `gguf.rs` metadata reader already walks tensors: flag rows whose GGUF carries BOTH full-model tensors AND `*.nextn.*`/`mtp.*` tensors as `draft: combined` (surfaced in `blazar show`). Profile side: for combined rows, spec-draft-model resolves to the SAME path (`-m == -md` pattern — no second mmap, mirroring the UDT single-file convention). One detection fn + one profile arm.
2. **Curated draft registry** — small data file beside `catalog.json` (same pattern): family/arch → known draft sources (unsloth `*-MTP-GGUF`, AtomicChat UDT collections), each entry carrying pairing mode (`separate-head` | `combined`), preferred quant, and naming convention. `drafts_cmd` consults it BEFORE raw HF search; entries are evidence-backed (linked receipt or published runbook), not vibes.
3. **Store+engine-aware output** — `drafts` filters by the installed llama row's mined `spec_types` (manifest.rs already mines them; `engine delta` diffs them — same source) and prints an executable plan: exact `blazar pull` lines + the `[[model_overrides]]` stanza + the `tune --spec` verification command. Keep the house rule: tune proves, drafts only nominates.

**Effort:** 1 = small; 2 = small (data + lookup); 3 = small-medium (CLI formatting + manifest join).
**Risks:** registry staleness (mitigate: registry is overrides-only, raw search stays the fallback); wrong pairing claims (mitigate: every curated entry names its receipt).
**Verification:** unit pins on combined-tensor detection (synthetic GGUF fixture); live: pull one UDT-class file → `show` reports combined → chat runs with spec engaged → `tune --spec` receipt banked.

---

## Item 4 — Hardware-tier build ranking ("best engine build for this GPU")

**Today (verified):** all inputs exist but nothing joins them: `hardware.rs` (`GpuInfo`, `Hardware`, VRAM totals), `probe.rs` (nvidia-smi census, `pci_gpu_vendors`, mem), `manifest.rs` (per-build-row mined flags/spec_types/architectures), `engine delta` (set-diff), `optimize.rs` (Severity/`OptimizeRec` — datum-backed recommendations with benchmark budget), `bench.rs` (pp/tg history per engine tag feeding tune gates), `capability_registry.rs` (lanes with `architectures` + `backends`).

**Plan:**
1. **Pure ranker** — new `engine_recommend(inputs) -> Vec<Rec>` beside the existing optimize planner (same crate, same style): inputs = hardware census + installed engine rows + offers lanes + bench history; output ranks candidate builds per model (or per store overall). Rank axes, all datum-backed: (a) backend match to present hardware (cuda compute-cap vs manifest backend tag; vulkan fallback; cpu floor), (b) measured bench receipts for that tag if any (pp/tg history — else explicit `measured: null`, house style), (c) capability coverage (arch/spec-type set from manifest), (d) regression-gate status of the row. No network, no estimates — a store+manifest join.
2. **Surfaces** — `blazar engine recommend [model]` (human table + `--json`) and gateway `/api/engine/recommend`; wire one doctor Advisor check ("active build is not the recommended one — because X").
3. **Tier ladder later (optional)** — GPU-name → performance-tier table for machines with zero bench history; strictly a display heuristic, labeled as such.

**Effort:** medium (one pure fn + 2 thin surfaces + pins). No new deps.
**Risks:** ranking implies authority it lacks without receipts — mitigated by always printing the datum (or `measured: null`) and never auto-switching engines; recommend only.
**Verification:** unit pins with seeded store/manifests (cuda-vs-vulkan-vs-cpu matrices); live: on this RTX 4070 box recommend the active cuda row with its bench receipt; on cpu-only sandbox recommend the cpu lane.

---

## Item 5 — NPU/mobile accelerator lanes (GenieX, LiteRT-LM, Apple FM, Foundry Local, PrismML)

**Today (verified):** `EngineKind` = {LlamaCpp, MistralRs, Sglang, Mlx, Sdcpp, Whisper, Piper}; sglang proves the "supervise an external OpenAI-compatible server child" pattern (`sglang_install.rs` + spawn/probe/forward); `engine build --fork owner/repo@<sha>` + `engine offers`/`install --lane` (capability_registry with `architectures`+`backends`, validators, auto-retire) already absorb llama.cpp forks as lanes; `[[remotes]]` already routes to any OpenAI-compatible endpoint incl. loopback.

**Plan — sequenced by ROI, one new abstraction total:**
1. **PrismML lane (first, cheapest):** it is a llama.cpp fork → lands EXACTLY in the existing fork-lane machinery. Curate a registry entry (Bonsai 1/2-bit tensor types, arch set), `engine install --lane`, validate GGUF loads + chat, done. No code beyond registry data unless validators need a Bonsai tensor-type check.
2. **Generic `openai-host` lane (the root-cause abstraction):** one new EngineKind for "external binary we spawn that serves OpenAI-compat on a port we pick" — the sglang pattern generalized: install step (vendor-specific), spawn+readiness probe, forward via existing proxy machinery, lifecycle under supervisor (crash-restart, mem-guard, rollback). This ONE lane then carries Foundry Local, LiteRT-LM, and any future vendor runtime without bespoke integrations. H23-shaped: five integrations become one abstraction + five install descriptors.
3. **Foundry Local on the openai-host lane** (Windows-mainstream value; Microsoft ships the binary — no container involved, satisfies the no-infra rule).
4. **LiteRT-LM on the openai-host lane** (PyPI-wheel distribution per their ADR — treat the wheel like any engine binary asset; Jetson/ChromeOS-class targets).
5. **GenieX + Apple Foundation Models — hold for demand:** GenieX is Snapdragon-niche and bundle-provisioning is vendor-gated; Apple FM needs a Swift/Rust shim server exposing OpenAI-compat (macOS-only sidecar → also rides openai-host via loopback once the shim exists). Both are "wait for a user with the hardware" items; the abstraction from (2) makes each a descriptor, not a project.

**Effort:** 1 = registry entry + validation day; 2 = the real work (medium-large: install/spawn/probe/forward wiring + lifecycle tests); 3–4 = descriptor + install script each once (2) exists; 5 = deferred.
**Risks:** openai-host lanes dilute the "we own the argv" supervision quality (children with opaque internals) — mitigate: capability probing via `/v1/models` + readiness, honest degradation, no pretending to knobs we can't set; vendor installers changing shape (pin versions per lane descriptor, same as engine tags).
**Verification:** PrismML: Bonsai GGUF chat 200 receipt + lane auto-retire test when upstream gains the tensor types; openai-host: stub-server e2e (spawn/probe/forward/crash-restart/stop-drain — V1–V5 lifecycle tests per §9), then live Foundry run on a Windows box.

---

## Item 6 — `blazar quantize` UDT-style conventions (Q8_0 nextn pins, Q6_K attn)

**Today (verified):** `quantize.rs` wraps `llama-quantize` + `llama-imatrix` (calibration file support), CLI already has `--imatrix`, `--allow-requantize`, `--verify` with `--max-degradation` — a quality budget already exists.

**Plan:**
1. **Mask generation Blazar-side (no tool changes):** `gguf.rs` tensor-name scan → generate a mask file (tensor-glob → quant) implementing the conventions: pin `blk.*.nextn.*` + `mtp.*` to Q8_0 (draft acceptance), lift `attn_q`/`attn_k` to Q6_K (KV-quant pairing). Ship as recipes: `blazar quantize <model> Q4_K_M --recipe udt-nextn` (and `--recipe turbo3-pair` once a turbo lane exists). The mask WRITER is pure Blazar code — pinned by unit tests on synthetic tensor lists.
2. **Mask consumption gated on tool support:** llama-quantize mask/leave-quant flag surface varies by build (their fork carries mask files; upstream moves) — probe the installed tool's supported flags (same mining approach as server flags) and pass through when present; otherwise teach honestly ("installed llama-quantize has no mask flag; the fork lane does").
3. **Receipts:** every masked quant run goes through the existing `--verify` degradation budget; bank ppl deltas per recipe so recipes carry evidence like everything else.

**Effort:** 1 = small-medium; 2 = small (probe + passthrough + teaching); 3 = existing machinery.
**Risks:** convention drift (UDT masks are their fork's convention, not a standard) — mitigate: recipes are named data files, tunable, and cite the source runbook; quality claims only from our own verify receipts.
**Verification:** unit pins on mask generation; live: quantize a small model with `--recipe udt-nextn`, verify tensor quants in the output GGUF metadata, ppl within budget vs plain Q4_K_M.

---

## Item 7 — Distribution & discovery (the cheapest, highest-visibility item)

**Today (verified):** GitHub release binaries + crates.io publish + CI; README install = release binaries / crates.io / source. No package-manager surfaces, no HF presence, no community hub.

**Plan — all release-artifact-backed, zero self-hosted infra (respects the no-infra rule):**
1. **cargo-binstall compatibility** — ship `[package.metadata.binstall]` in the workspace manifests pointing at release assets: `cargo binstall blazar` becomes the one-line install for Rust-adjacent users. Smallest possible change, immediate payoff.
2. **Package-manager manifests (all consume GitHub release tarballs — nothing hosted by us):**
   - Homebrew tap (`santanu20/homebrew-blazar`, one formula; `brew install santanu20/blazar/blazar`)
   - AUR PKGBUILD (`blazar-bin` pulling the release tarball; community can then `yay -S blazar-bin`)
   - Winget manifest + Scoop app manifest (Windows: `winget install blazar`, `scoop install blazar`)
3. **HF org presence** — an org with curated collections (the pairing recipes from Item 3, KV-quality receipts from the turbo campaign, example model sets). Their HF org is a major discovery channel; costs content work only.
4. **Community hub without infra** — GitHub Discussions (Q&A + showcase categories) over Discord-for-now; benchmark receipts as the launch artifact (the 1.4%-overhead + Ollama-tail tables are genuinely differentiated — lead with them on HN/r/LocalLLaMA when 0.30 ships).
5. **Versioned install docs** — README install section gains the one-liners from (1)/(2) as they land; `tests/install_e2e.sh` already exists to keep the release path honest — extend it to smoke-test one manifest-based install path per release.

**Effort:** 1 = hours; 2 = ~a day total across the four manifests; 3–4 = content time, not code.
**Risks:** manifests rot when asset naming changes (mitigate: install_e2e smoke tests per release); tap/manifest trust (sign releases — cosign or GitHub attestation — before pushing package managers hard).
**Verification:** each manifest install executed in a clean environment (VM/container is fine for TESTING installs — the no-infra rule governs what we ship, not how we verify); receipts appended to the release checklist.

---

## Sequencing recommendation

| Order | Item | Why here |
|---|---|---|
| 1 | 7 (binstall + manifests) | Cheapest, highest visibility; ride the 0.30 release |
| 2 | 5.1 PrismML lane | Registry entry on existing machinery; days, not weeks |
| 3 | 3 (drafts pairing) | Small; pairs naturally with the spec receipts work |
| 4 | 4 (engine recommend) | Pure join over existing facts; unlocks advisor parity |
| 5 | 6 (quantize recipes) | Small; bigger payoff after a turbo fork lane exists (items 1–2 of the main report) |
| 6 | 5.2 openai-host abstraction | The one large investment; unlocks Foundry/LiteRT/GenieX/Apple-FM as descriptors |

Cross-item note: Items 1–2 of the main report (TurboQuant absorption, spec+mmproj receipts) amplify 6 and 3 — sequence them together in one "spec-decode season".
