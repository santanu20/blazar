# KV continuity, differential quant, preload, RAM-warm — live receipts

- Date: 2026-09-27 · Branch: `feature/frontier-roi` @ f525512 · Daemon: v0.12.0 (release build)
- Rig: scratch XDG root (`scratch XDG root`, port 11499), engine b11202-cuda (copied, no symlinks), model `qwen2.5-0.5b-instruct` (q4_k_m, 468 MiB), 16c/13.7G RAM/7.8G VRAM.
- Driver: `bench_wave4.py` (phases B/C/C2) + manual micro-test (phase A; driver phases A kept racing spawn-settle — see NOTES).

| # | Proof | Verdict | Evidence |
|---|-------|---------|----------|
| A1 | Shutdown banking writes slot-0 KV | PASS | SIGTERM → `sessions/qwen2.5-0.5b-instruct-f829161bb633b3da/_auto-16384` (601.4K) + `_auto-16384.identity.json` created |
| A2 | Restart restores banked prefix | PASS | daemon log: `restored banked session _auto-16384 (50 tokens into slot KV)`; response `prompt_eval_cached_count: 24`; model recalled codeword `BASILISK-7` across restart |
| B | Differential K/V quant argv | PASS | child argv: `--cache-type-k q8_0 --cache-type-v q4_0` (alongside `-np 8 --spec-type ngram-simple --kv-unified --cache-ram 4102 --load-mode mlock --slot-save-path ...`) |
| C | Startup preload before first request | PASS | config `preload = ["qwen2.5-0.5b-instruct"]` → daemon log `startup preload list engaged count=1`; llama-server child running with zero client requests (90s poll) |
| C2 | Idle-to-RAM page-cache warm | PASS | after idle evict: `idle-to-RAM warm: weights re-paged into the OS cache model=qwen2.5-0.5b-instruct bytes=491400032 ms=58` (468 MiB in 58 ms) |

## Config under test (phase A micro-test)
```toml
host = "127.0.0.1"; port = 11499; slots = 1
idle_sleep_secs = 3600; idle_timeout_secs = 7200
```
Phase B/C/C2 config: `slots = 8`, `cache_type_k = "q8_0"`, `cache_type_v = "q4_0"`, `preload = [...]`, `idle_sleep_secs = 20`, `idle_timeout_secs = 30`.

## NOTES (validation-methodology, not product bugs)
- Bank requires the instance Ready post spawn-settle (~1.5s after first serve). SIGTERM inside that window skips banking by design. Drivers must give settle grace.
- Children in `--sleep-idle-seconds` sleep wake-reload on the bank POST and return ok-but-empty saves; keep sleep-idle long when validating banking.
- Config guard: `idle_timeout_secs must be >= idle_sleep_secs` (validated at startup — fast-fail confirmed live).
- Flat model names in `preload` (no `:quant` tag); unknown names warn + daemon stays up (warn-not-fail contract observed live in driver v1).
- Raw artifacts: `scratch XDG root/wave4-live.json`, `microA.log`, `microA2.log`, `daemon-wave4.log`.
