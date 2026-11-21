# 2026-11-21 media lanes (issue 003: kokoro bench lane)

Campaign: `python3 scripts/bench_matrix.py --model qwen2.5-coder-0.5b-instruct --media-only --media-runs 3 --fresh`

## Why this campaign exists

The Nov-20 render had no kokoro bench lane. `run_media_tts_kokoro_cell` (added to
`scripts/bench_matrix.py` for this wave) closes the gap: engine-bundled weights, no
`stage_voices` dependency, buffered-wav requests (the lane rejects streaming by design),
`af_heart` voice, cold/warm/RTF recorded like the piper cell.

## Result

| Lane | Engine | Cells | Verdict |
|---|---|---|---|
| media-tts | kokoro-1.0 | 1 ok / 0 err | benchmarked - lane gap closed |
| media-tts + media-tts-conc | piper 2023.11.14-2 | 2 ok / 0 err | benchmarked (reference) |
| media-whisper | b5907 | 1 ok / 0 err | benchmarked |
| media-image | sdcpp master-1043 | 1 ok / 3 err | sage_attn_on ok; 3 cells aborted on the harness RAM guard (below) |
| media-video | sdcpp master-1043 | 1 ok / 0 err | benchmarked (320x320, frames 5/13/33, steps 8) |

QC drift gate: clean. Full per-cell metrics in `benchmark.md` / `summary.txt` / `cells.jsonl`.

## Failed cells (documented, environment - not product)

Three image cells (baseline, `fa_off`, `vae_tiling_on`) aborted on the harness memory
guard: `MemAvailable 6.2-7.5 GiB < needed ~7.9 GiB` (qwen-image weights 5.5 GiB +
headroom). This 13.7 GiB-RAM dev box runs the desktop session plus agent tooling; the
co-residency math fails at boot, before any Blazar code path runs. The `sage_attn_on`
variant and the video cell both fit and completed after cache settle, proving the lane
end-to-end. Rerun on a larger-RAM host for full image A/B coverage.

## Caveats

- Tree state: run against a tree with a concurrent in-flight wave (user-authorized
  "ignore collision, do all"); release binary built from that tree (`blazar 0.34.0`).
- Single box (RTX 4070 8 GB, 13.7 GB RAM) - absolute numbers are machine-specific.
- Media lanes are gateway-routed by design; there is no direct-leg comparison for TTS
  media cells (piper/kokoro are lazy in-process lanes, not standalone HTTP servers).
