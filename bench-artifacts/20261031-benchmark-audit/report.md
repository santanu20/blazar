# Benchmark audit — 2026-10-31 (quality-neutral improvement pass)

## Environment
- Blazar: target/release @ 9c265ce-lineage + session commits (131ce7e, ade46d2)
- Engine: llama.cpp b12015-cuda (official ggml-org binary; upgrade b12020 pending)
- GPU: RTX 4070 Laptop 8 GB VRAM; RAM 13.6 GB; 16 cores; driver 595.99.02
- Daemon: live (pid 1395277); models: qwen3-0.6b (Q4_0), qwen2.5-0.5b-instruct, qwen3-1.7b (sglang lane)

## Runs performed (receipts persisted in-store + this dir)
1. `blazar bench qwen3-0.6b` → pp512 19,277 / pp2048 20,239 / tg128 378.08 tok/s (f16/f16, threads 8)
2. `blazar bench qwen2.5-0.5b-instruct` → pp512 22,207 / pp2048 25,496 / tg128 396.38 tok/s
3. `blazar tune qwen3-0.6b --search` → 60 configs measured; adopted: fa on, threads 16, -np 2, ngram-simple spec, mlock, dynamic lookup cache, ctx 16384
4. `blazar tune qwen3-0.6b --spec-race` → dense 41.8 / ngram 41.84 (ratio 1.0003) / dflash 42.7 (+2.2%, under 5% threshold → correctly not adopted); 3 receipts saved
5. Prior same-day KV-FA matrix (bench-artifacts/20261031-kv-fa-quant-audit): q8_0 +11% pp, q8_0/q4_0 +15% pp vs f16, zero FA-conversion warnings

## Findings

| # | Finding | Evidence |
|---|---------|----------|
| F1 | Bench receipts were stale across engine updates — all prior rows on b11370/b1198x tags, none on active b12015; tune gates consult these | store bench_results before today |
| F2 | Bench harness pins threads=8 while the tune grid measured 16 as the winner — receipts mix thread counts, so cross-receipt comparisons (tune gates) compare different configs | bench_results payload n_threads=8 vs adopted argv --threads 16 |
| F3 | spec-race ngram arm runs BEFORE any ngram parameter tuning (ngram_* config all 0; winner argv carries no ngram size flags) — the arm measured ≈ dense (ratio 1.0003) and that receipt now feeds spec=auto resolution | spec_receipts row (ngram, 41.836, 1.0003) + config.toml zeros |
| F4 | Auto-draft pairing picks lane-incompatible drafts: qwen3-1.7b (sglang) paired with GGUF draft → warn + dense fallback; qwen3-0.6b draft resolves to the model itself → arm skipped | live `blazar ps` warn line + race stdout "draft arm skipped" |
| F5 | dflash +2.2% on 0.6b — small model is GPU-compute-bound at decode; speculation headroom grows with model size | spec-race table |
| F6 | KV quant lanes are measured FASTER than f16 on this engine (+11-15% pp) yet the capacity ladder only engages them under VRAM pressure; default spawns run f16 KV | kv-fa matrix receipt |

## Suggestions (quality-neutral, evidence-backed)

1. **S1 — adopt q8_0 KV via measured tune for VRAM-comfortable llamacpp models.** q8_0 KV perplexity delta is negligible (literature-consistent) and this engine measures +11% pp / +4-6% tg vs f16. Mechanism exists (`tune --search` fa-leg); run it per model after engine updates. Quality impact: ~none; verify per-model with the tune receipt.
2. **S2 — fix draft-pairing compatibility before racing:** filter catalog drafts by lane (GGUF→llamacpp, safetensors EAGLE→sglang) and exclude self-pairing. Removes warn noise, wasted arms, and misleading "dense fallback" receipts. Pure scheduling correctness — zero quality impact.
3. **S3 — reorder ngram tuning: run `tune --ngram` (size_m/min_hits search) BEFORE `--spec-race`**, or give the race's ngram arm sane defaults — today's race measures an untuned ngram and persists a near-1.0 receipt that biases future spec=auto resolution. The CLI already prints "next: --ngram refines" only when ngram wins — invert the ordering.
4. **S4 — align bench harness threads with the adopted profile** (or probe threads like the tune grid): receipts currently mix 8-thread benches with 16-thread adopted profiles; gates comparing across them compare apples to oranges.
5. **S5 — auto-refresh (or doctor-flag) bench receipts on engine update:** every engine tag bump silently orphans the receipt set (F1); a doctor row "bench receipts predate active engine" + one-command refresh closes the gap.
6. **S6 — bigger-model speculation headroom:** dflash/draft racing pays on ≥4B-class models on this box; queue `--spec-race` for qwen3.5-9b and qwen3vl-8b when VRAM allows (speculation is lossless — verified acceptance).
7. **S7 — gateway-overhead ladder (follow-up):** direct llama-server vs Blazar gateway TTFT/throughput at C=1..8 not re-measured this session; prior methodology exists in bench-artifacts; run as its own controlled pass.

## Reproducibility
- All raw outputs: this directory (`bench-*.txt`, `tune-*.txt`); receipts in-store (bench_results, bench_history, spec_receipts on b12015-cuda).
- Commands: `blazar bench <model>`; `blazar tune <model> --search|--spec-race`.
- Config snapshot: ~/.config/blazar/config.toml (ngram_* = 0 at race time).

## S1-S7 implementation wave (2026-10-31, same day)

| # | Type | Outcome |
|---|------|---------|
| S1 | RUN | `tune --search qwen2.5-0.5b-instruct`: 60 configs, adopted (fa on, threads 16, -b 2048, -np 2, ngram-simple, kv-unified, mlock, lcache, ctx 16384; f16 KV measured best at this size — q8_0 NOT adopted, measured choice). Receipt: tune-search-qwen2.5-0.5b-instruct.txt |
| S2 | CODE | Lane-aware draft pairing: `pick_draft`/`DraftPick`/`DraftLane`/`lane_for` (supervisor.rs) — self-pair + GGUF/safetensors lane mismatch degrades dense with ONE teaching warn at arming, receipt probes no longer count unusable drafts as available, race arm skips with printed reason. Live-proven: self-pair warn fired in sandbox serve log. Pin: unit__pick_draft__lane_aware_and_self_paired |
| S3 | CODE | ngram race arm carries tuned shape: `spec_mode_race(..., ngram_knobs)` emits size-m/size-n/min-hits when tuned; untuned prints notice. Live-proven in the S6 race output |
| S4 | CODE | `blazar bench` measures with the serving profile's pinned --threads (bench_default_threads; unpinned = llama-bench default, same as serve) |
| S5 | CODE | doctor "bench receipts" section: per-model warn when latest receipt's engine_tag != active engine (latest_bench_result) |
| S6 | RUN | `tune --spec-race qwen3.5-9b`: dense 38.3 / ngram 36.3 (untuned, flagged by S3 notice) / dflash 38.4 — no adoption (<5%); draft arm capacity-skipped (8 GiB card). Receipt: tune-spec-race-qwen3.5-9b.txt |
| S7 | RUN | Gateway-overhead ladder (qwen3-0.6b, adopted argv, 8 req × 64 tok, 3 reps, min-wall): gateway/direct tok-tps = 0.934 @C1, 0.770 @C2, 0.764 @C4, **0.547 @C8**; p50 0.186 vs 0.171 @C1, 0.928 vs 0.608 @C8. Caveat: same-box dual-child, sandbox gateway, gateway lane adds sentinel+translate+routing. **Flagged as the top optimization target.** Receipt: gateway-overhead-ladder.json |

Quality-neutrality validation: greedy (temp 0, seed 7) A/B dense vs adopted
ngram-simple lane, 3 prompts — **byte-identical outputs** (acceptance-verified
speculation, empirically pinned). No inference-path change in S2-S5
(scheduling/receipt hygiene only).

Gates: fmt, clippy -D 0, nextest 2319/2319 (+1 pin).
