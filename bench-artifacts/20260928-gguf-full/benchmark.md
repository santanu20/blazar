# Blazar benchmark matrix

- **date**: 2026-09-29 08:06:23
- **model**: `Qwen3.5-9B-Q4_K_M.gguf` (5366 MiB)
- **gpu**: NVIDIA GeForce RTX 4070 Laptop GPU / driver 595.91.07
- **engines**: b11202-cuda, b5130, inventory, master-920-2f88688, ollama-host, piper, sglang-0.5.19, v0.9.4
- **harness**: bench_matrix v2 — `--model qwen3.5-9b --artifacts-dir bench-artifacts/20260928-gguf-full --md BENCHMARK.md`
- **blazar**: `blazar 0.13.0` (sandbox daemon binary)
- all blazar-owned rows measured by `blazar 0.13.0`

## Notable findings

- gateway vs direct decode (`b11202-cuda`): 41.1 vs 41.2 t/s = -0.2% (parity).
- gateway vs direct decode (`b11202-cuda`): 41.6 vs 41.2 t/s = +0.8% (parity).
- gateway vs direct decode (`b11202-cuda`): 41.0 vs 41.2 t/s = -0.5% (parity).
- gateway vs direct decode (`b11202-cuda`): 41.0 vs 41.2 t/s = -0.5% (parity).
- gateway vs direct decode (`b11202-cuda`): 40.7 vs 41.2 t/s = -1.2% (parity).
- gateway vs direct decode (`b11202-cuda`): 41.0 vs 41.2 t/s = -0.6% (parity).
- gateway vs direct decode (`b11202-cuda`): 41.0 vs 41.2 t/s = -0.6% (parity).
- gateway vs direct decode (`b11202-cuda`): 41.6 vs 41.2 t/s = +0.9% (parity).
- gateway vs direct decode (`b11202-cuda`): 18.5 vs 41.2 t/s = -55.0% (regression).
- gateway vs direct decode (`b11202-cuda`): 40.6 vs 41.2 t/s = -1.4% (parity).
- gateway vs direct decode (`b11202-cuda`): 41.0 vs 41.2 t/s = -0.5% (parity).
- gateway vs direct decode (`b11202-cuda`): 41.6 vs 41.2 t/s = +0.8% (parity).
- gateway vs direct decode (`b11202-cuda`): 41.1 vs 41.2 t/s = -0.4% (parity).
- gateway vs direct decode (`b11202-cuda`): 41.6 vs 41.2 t/s = +0.8% (parity).
- gateway vs direct decode (`b11202-cuda`): 41.0 vs 41.2 t/s = -0.6% (parity).
- gateway vs direct decode (`b11202-cuda`): 41.0 vs 41.2 t/s = -0.6% (parity).
- gateway vs direct decode (`b11202-cuda`): 41.0 vs 41.2 t/s = -0.6% (parity).
- gateway vs direct decode (`v0.9.4`): 41.1 vs 21.3 t/s = +93.3% (gain).
- gateway vs direct decode (`v0.9.4`): 41.6 vs 21.3 t/s = +95.5% (gain).
- gateway vs direct decode (`v0.9.4`): 41.5 vs 21.3 t/s = +95.1% (gain).
- gateway vs direct decode (`v0.9.4`): 41.1 vs 21.3 t/s = +93.1% (gain).
- gateway vs direct decode (`v0.9.4`): 41.1 vs 21.3 t/s = +93.1% (gain).
- gateway vs direct decode (`v0.9.4`): 41.4 vs 21.3 t/s = +94.9% (gain).
- gateway vs direct decode (`v0.9.4`): 41.0 vs 21.3 t/s = +92.9% (gain).
- gateway vs direct decode (`v0.9.4`): 41.1 vs 21.3 t/s = +93.1% (gain).
- gateway vs direct decode (`v0.9.4`): 41.0 vs 21.3 t/s = +93.0% (gain).
- gateway vs direct decode (`v0.9.4`): 41.6 vs 21.3 t/s = +95.5% (gain).
- gateway vs direct decode (`v0.9.4`): 41.0 vs 21.3 t/s = +92.7% (gain).
- gateway vs direct decode (`v0.9.4`): 41.0 vs 21.3 t/s = +92.8% (gain).
- concurrency system throughput (`b11202-cuda`, 4 streams): 67.5 vs direct 110.6 t/s = 0.61x.
- kv=q8_0 (`b11202-cuda`): decode -0.5% vs baseline
- pa=on (`b11202-cuda`): decode +0.6% vs baseline
- spec=ngram-simple (`b11202-cuda`): decode -0.5% vs baseline
- mmproj=True (`b11202-cuda`): decode +0.4% vs baseline
- pa=off (`v0.9.4`): decode -14.3% vs baseline
- gateway greedy transparency (`b11202-cuda`): 16/20 exact vs same-engine direct — NOT TRANSPARENT.

## Speed (serving, streaming)

| engine | kind | provider | params | ttft p50 (ms) | ttft p99 (ms) | itl p50 (ms) | itl p99 (ms) | decode t/s | prefill t/s (cold) | prefill t/s (cached) | tokens src |
|---||---||---||---||---||---||---||---||---||---||---||
| b11202-cuda | llamacpp | direct | ctx=4096 np=1 | 127 | 132 | 24 | 32 | 41.2 | 1265.4 | 7373.7 | chunks |
| b11202-cuda | llamacpp | direct | ctx=4096 np=4 | 123 | 125 | 24 | 25 | 41.6 | 1309.9 | 7467.2 | chunks |
| b11202-cuda | llamacpp | direct | ctx=16384 np=1 | 123 | 125 | 24 | 25 | 41.6 | 1308.9 | 7439.5 | chunks |
| b11202-cuda | llamacpp | direct | ctx=16384 np=4 | 122 | 123 | 24 | 25 | 41.4 | 1340.8 | 7303.2 | chunks |
| b11202-cuda | llamacpp | direct | ctx=4096 kv=q8_0 np=1 | 119 | 121 | 24 | 26 | 41.0 | 1389.3 | 7737.8 | chunks |
| b11202-cuda | llamacpp | direct | ctx=4096 np=1 pa=on | 124 | 124 | 24 | 25 | 41.5 | 1316.2 | 7490.3 | chunks |
| b11202-cuda | llamacpp | direct | ctx=4096 np=1 spec=ngram-simple | 122 | 125 | 24 | 25 | 41.0 | 1334.6 | 7380.1 | chunks |
| b11202-cuda | llamacpp | direct | ctx=4096 mmproj=True np=1 | 123 | 124 | 24 | 26 | 41.4 | 1309.8 | 7456.5 | chunks |
| v0.9.4 | mistralrs | direct | ctx=4096 np=1 | 127 | 141 | 47 | 54 | 21.3 | 273.8 | 323.5 | chunks |
| v0.9.4 | mistralrs | direct | ctx=4096 np=4 | 168 | 191 | 47 | 56 | 21.1 | 255.5 | 297.9 | chunks |
| v0.9.4 | mistralrs | direct | ctx=16384 np=1 | 151 | 185 | 46 | 55 | 21.5 | 281.3 | 345.2 | chunks |
| v0.9.4 | mistralrs | direct | ctx=16384 np=4 | 145 | 219 | 47 | 56 | 21.1 | 273.3 | 312.5 | chunks |
| v0.9.4 | mistralrs | direct | ctx=4096 np=1 pa=off | 154 | 214 | 55 | 64 | 18.2 | 204.2 | 236.1 | chunks |
| b11202-cuda | llamacpp | blazar | config=default child: np=2 ctx=8192 | 124 | 174 | 24 | 26 | 41.1 | 1302.3 | 7212.0 | chunks |
| b11202-cuda | llamacpp | blazar | config=single-stream child: np=1 ctx=16384 | 125 | 127 | 24 | 25 | 41.6 | 1313.0 | 7239.3 | chunks |
| b11202-cuda | llamacpp | blazar | config=cont_batching_off child: np=2 ctx=8192 | 125 | 177 | 24 | 26 | 41.0 | 1307.7 | 7191.5 | chunks |
| b11202-cuda | llamacpp | blazar | config=fa_on child: np=2 ctx=8192 | 124 | 131 | 24 | 25 | 41.0 | 1318.9 | 7293.1 | chunks |
| b11202-cuda | llamacpp | blazar | config=fa_off child: np=2 ctx=8192 | 160 | 351 | 24 | 26 | 40.7 | 1158.8 | 7434.7 | chunks |
| b11202-cuda | llamacpp | blazar | config=kv_unified_on child: np=2 ctx=8192 | 122 | 127 | 24 | 25 | 41.0 | 1371.7 | 7327.4 | chunks |
| b11202-cuda | llamacpp | blazar | config=kv_unified_off child: np=2 ctx=8192 | 123 | 126 | 24 | 26 | 41.0 | 1353.3 | 7293.5 | chunks |
| b11202-cuda | llamacpp | blazar | config=swa_full_on child: np=2 ctx=8192 | 123 | 125 | 24 | 25 | 41.6 | 1316.6 | 7460.8 | chunks |
| b11202-cuda | llamacpp | blazar | config=no_kv_offload_on child: np=2 ctx=8192 | 150 | 205 | 54 | 60 | 18.5 | 931.3 | 5074.5 | chunks |
| b11202-cuda | llamacpp | blazar | config=cache_q8_q8_0 child: np=2 ctx=8192 | 124 | 125 | 25 | 26 | 40.6 | 1312.3 | 7580.5 | chunks |
| b11202-cuda | llamacpp | blazar | config=ctx_checkpoints_4 child: np=2 ctx=8192 | 121 | 125 | 24 | 26 | 41.0 | 1317.7 | 7538.2 | chunks |
| b11202-cuda | llamacpp | blazar | config=spec_off child: np=2 ctx=8192 | 119 | 132 | 24 | 25 | 41.6 | 1308.3 | 7349.5 | chunks |
| b11202-cuda | llamacpp | blazar | config=deterministic_on child: np=1 ctx=16384 | 123 | 124 | 24 | 25 | 41.1 | 1312.0 | 7424.7 | chunks |
| b11202-cuda | llamacpp | blazar | config=mmproj_offload_off child: np=2 ctx=8192 | 124 | 124 | 24 | 25 | 41.6 | 1313.7 | 7229.0 | chunks |
| b11202-cuda | llamacpp | blazar | config=threads_batch_8 child: np=2 ctx=8192 | 126 | 174 | 24 | 25 | 41.0 | 1310.9 | 7246.5 | chunks |
| b11202-cuda | llamacpp | blazar | config=cache_reuse_256 child: np=2 ctx=8192 | 125 | 175 | 24 | 26 | 41.0 | 1287.3 | 7243.0 | chunks |
| b11202-cuda | llamacpp | blazar | config=kv_unified_per_slot_4096 child: np=2 ctx=8192 | 123 | 124 | 24 | 26 | 41.0 | 1330.7 | 7549.6 | chunks |
| v0.9.4 | mistralrs | blazar | config=default child: np=2 ctx=8192 | 122 | 124 | 24 | 26 | 41.1 | 1379.9 | 7160.5 | chunks |
| v0.9.4 | mistralrs | blazar | config=paged_attn_off child: np=2 ctx=8192 | 119 | 125 | 24 | 25 | 41.6 | 1370.2 | 7447.0 | chunks |
| v0.9.4 | mistralrs | blazar | config=single-stream child: np=1 ctx=16384 | 122 | 123 | 24 | 25 | 41.5 | 1403.8 | 7654.8 | chunks |
| v0.9.4 | mistralrs | blazar | config=kv_unified_on child: np=2 ctx=8192 | 124 | 125 | 24 | 26 | 41.1 | 1367.1 | 7414.7 | chunks |
| v0.9.4 | mistralrs | blazar | config=kv_unified_off child: np=2 ctx=8192 | 120 | 125 | 24 | 26 | 41.1 | 1312.5 | 7397.5 | chunks |
| v0.9.4 | mistralrs | blazar | config=spec_off child: np=2 ctx=8192 | 123 | 124 | 24 | 25 | 41.4 | 1325.0 | 7317.3 | chunks |
| v0.9.4 | mistralrs | blazar | config=deterministic_on child: np=1 ctx=16384 | 118 | 122 | 24 | 25 | 41.0 | 1313.0 | 7244.9 | chunks |
| v0.9.4 | mistralrs | blazar | config=pa_mem_0.85 child: np=2 ctx=8192 | 122 | 124 | 24 | 25 | 41.1 | 1326.4 | 7667.6 | chunks |
| v0.9.4 | mistralrs | blazar | config=pa_mem_0.55 child: np=2 ctx=8192 | 121 | 122 | 24 | 25 | 41.0 | 1316.9 | 7371.5 | chunks |
| v0.9.4 | mistralrs | blazar | config=mr_batch_64 child: np=2 ctx=8192 | 123 | 126 | 24 | 25 | 41.6 | 1366.6 | 7352.9 | chunks |
| v0.9.4 | mistralrs | blazar | config=mr_prefix_cache_256 child: np=2 ctx=8192 | 121 | 174 | 24 | 26 | 41.0 | 1314.6 | 7358.3 | chunks |
| v0.9.4 | mistralrs | blazar | config=mr_enc_cache_512mb child: np=2 ctx=8192 | 121 | 123 | 24 | 26 | 41.0 | 1369.8 | 7396.1 | chunks |
| ollama-host | ollama | ollama | reference=True ⚠ serves 'qwen3.5:9b' — t/s NOT comparable | 126 | 132 | 25 | 75 | 40.6 | 1409.3 | 7372.4 | engine_counters |
| b11202-cuda | llamacpp | ctxcurve-blazar | ctx=2048 child: np=4 ctx=8192 | 125 | 182 | 24 | 26 | 41.5 | 1324.2 | 7176.7 | chunks |
| b11202-cuda | llamacpp | ctxcurve-blazar | ctx=8192 child: np=1 ctx=8192 | 122 | 329 | 24 | 25 | 41.4 | 1376.5 | 7358.1 | chunks |
| b11202-cuda | llamacpp | ctxcurve-blazar | ctx=16384 child: np=1 ctx=16384 | 122 | 317 | 24 | 25 | 41.0 | 1330.1 | 7410.0 | chunks |
| v0.9.4 | mistralrs | ctxcurve-blazar | ctx=2048 child: np=4 ctx=8192 | 123 | 176 | 24 | 25 | 41.0 | 1369.2 | 7304.5 | chunks |
| v0.9.4 | mistralrs | ctxcurve-blazar | ctx=8192 child: np=1 ctx=8192 | 122 | 323 | 24 | 26 | 41.0 | 1334.5 | 7442.1 | chunks |
| v0.9.4 | mistralrs | ctxcurve-blazar | ctx=16384 child: np=1 ctx=16384 | 120 | 308 | 24 | 25 | 41.7 | 1322.2 | 7395.3 | chunks |
| ollama-host | ollama | ctxcurve-ollama | ctx=2048 | 135 | - | - | - | 40.5 | - | - | - |
| ollama-host | ollama | ctxcurve-ollama | ctx=8192 | 128 | - | - | - | 40.5 | - | - | - |
| ollama-host | ollama | ctxcurve-ollama | ctx=16384 | 130 | - | - | - | 40.6 | - | - | - |

## Resources & cold start

| engine | provider | params | load s | daemon boot s | cold 1st req s | GPU peak (MiB) | GPU power (W) | RSS peak (MiB) | teardown |
|---||---||---||---||---||---||---||---||---||
| b11202-cuda | direct | ctx=4096 np=1 | 9.55 | - | - | 5270 | 55.5 | 4408 | ok |
| b11202-cuda | direct | ctx=4096 np=4 | 4.53 | - | - | 5410 | 56.0 | 5672 | ok |
| b11202-cuda | direct | ctx=16384 np=1 | 2.51 | - | - | 5666 | 55.9 | 5715 | ok |
| b11202-cuda | direct | ctx=16384 np=4 | 2.51 | - | - | 5804 | 56.7 | 5718 | ok |
| b11202-cuda | direct | ctx=4096 kv=q8_0 np=1 | 2.51 | - | - | 5210 | 56.4 | 5719 | ok |
| b11202-cuda | direct | ctx=4096 np=1 pa=on | 2.53 | - | - | 5270 | 56.8 | 5716 | ok |
| b11202-cuda | direct | ctx=4096 np=1 spec=ngram-simple | 2.51 | - | - | 5270 | 55.8 | 5719 | ok |
| b11202-cuda | direct | ctx=4096 mmproj=True np=1 | 3.01 | - | - | 6394 | 55.9 | 5721 | ok |
| v0.9.4 | direct | ctx=4096 np=1 | 11.04 | - | - | 7042 | 45.2 | 6692 | ok |
| v0.9.4 | direct | ctx=4096 np=4 | 7.51 | - | - | 7042 | 44.1 | 7842 | ok |
| v0.9.4 | direct | ctx=16384 np=1 | 8.51 | - | - | 7042 | 44.9 | 7952 | ok |
| v0.9.4 | direct | ctx=16384 np=4 | 6.01 | - | - | 7042 | 44.0 | 7962 | ok |
| v0.9.4 | direct | ctx=4096 np=1 pa=off | 11.03 | - | - | 6882 | 33.7 | 4867 | ok |
| b11202-cuda | blazar | config=default | - | 0.53 | 5.08 | 5458 | 55.6 | 2202 | ok |
| b11202-cuda | blazar | config=single-stream | - | 0.52 | 4.95 | 5666 | 55.2 | 2104 | ok |
| b11202-cuda | blazar | config=cont_batching_off | - | 0.52 | 4.79 | 5458 | 55.9 | 2103 | ok |
| b11202-cuda | blazar | config=fa_on | - | 0.52 | 4.79 | 5458 | 55.1 | 2101 | ok |
| b11202-cuda | blazar | config=fa_off | - | 0.52 | 4.88 | 5696 | 56.4 | 2345 | ok |
| b11202-cuda | blazar | config=kv_unified_on | - | 0.52 | 4.87 | 5458 | 55.6 | 2152 | ok |
| b11202-cuda | blazar | config=kv_unified_off | - | 0.52 | 4.93 | 5458 | 56.9 | 2196 | ok |
| b11202-cuda | blazar | config=swa_full_on | - | 0.52 | 4.91 | 5458 | 56.1 | 2102 | ok |
| b11202-cuda | blazar | config=no_kv_offload_on | - | 0.52 | 5.54 | 5106 | 52.1 | 2465 | ok |
| b11202-cuda | blazar | config=cache_q8_q8_0 | - | 0.52 | 4.95 | 5338 | 55.5 | 2084 | ok |
| b11202-cuda | blazar | config=ctx_checkpoints_4 | - | 0.52 | 4.49 | 5477 | 55.5 | 2102 | ok |
| b11202-cuda | blazar | config=spec_off | - | 0.52 | 4.64 | 5452 | 55.2 | 2102 | ok |
| b11202-cuda | blazar | config=deterministic_on | - | 0.52 | 4.8 | 5666 | 57.1 | 2105 | ok |
| b11202-cuda | blazar | config=mmproj_offload_off | - | 0.51 | 4.86 | 5458 | 56.6 | 2102 | ok |
| b11202-cuda | blazar | config=threads_batch_8 | - | 0.52 | 4.83 | 5458 | 55.7 | 2203 | ok |
| b11202-cuda | blazar | config=cache_reuse_256 | - | 0.52 | 4.83 | 5458 | 55.7 | 2202 | ok |
| b11202-cuda | blazar | config=kv_unified_per_slot_4096 | - | 0.52 | 4.82 | 5458 | 55.3 | 2102 | ok |
| v0.9.4 | blazar | config=default | - | 0.52 | 4.63 | 5458 | 56.8 | 2101 | ok |
| v0.9.4 | blazar | config=paged_attn_off | - | 0.52 | 4.62 | 5458 | 55.7 | 2102 | ok |
| v0.9.4 | blazar | config=single-stream | - | 0.52 | 4.6 | 5666 | 55.7 | 2105 | ok |
| v0.9.4 | blazar | config=kv_unified_on | - | 0.52 | 4.58 | 5464 | 55.5 | 2129 | ok |
| v0.9.4 | blazar | config=kv_unified_off | - | 0.52 | 4.62 | 5465 | 56.8 | 2196 | ok |
| v0.9.4 | blazar | config=spec_off | - | 0.52 | 4.68 | 5458 | 56.5 | 2102 | ok |
| v0.9.4 | blazar | config=deterministic_on | - | 0.52 | 4.68 | 5666 | 56.4 | 2104 | ok |
| v0.9.4 | blazar | config=pa_mem_0.85 | - | 0.52 | 4.64 | 5458 | 56.6 | 2101 | ok |
| v0.9.4 | blazar | config=pa_mem_0.55 | - | 0.53 | 4.64 | 5458 | 55.7 | 2101 | ok |
| v0.9.4 | blazar | config=mr_batch_64 | - | 0.52 | 4.58 | 5464 | 55.9 | 2102 | ok |
| v0.9.4 | blazar | config=mr_prefix_cache_256 | - | 0.52 | 4.59 | 5464 | 55.9 | 2202 | ok |
| v0.9.4 | blazar | config=mr_enc_cache_512mb | - | 0.52 | 4.58 | 5464 | 57.1 | 2101 | ok |
| ollama-host | ollama | reference=True | - | - | - | 6570 | 55.4 | - | ok |
| b11202-cuda | ctxcurve-blazar | ctx=2048 | - | - | - | 5646 | - | - | ok |
| b11202-cuda | ctxcurve-blazar | ctx=8192 | - | - | - | 5402 | - | - | ok |
| b11202-cuda | ctxcurve-blazar | ctx=16384 | - | - | - | 5660 | - | - | ok |
| v0.9.4 | ctxcurve-blazar | ctx=2048 | - | - | - | 5777 | - | - | ok |
| v0.9.4 | ctxcurve-blazar | ctx=8192 | - | - | - | 5415 | - | - | ok |
| v0.9.4 | ctxcurve-blazar | ctx=16384 | - | - | - | 5673 | - | - | ok |
| ollama-host | ctxcurve-ollama | ctx=2048 | 6.14 | - | - | 6368 | - | - | ok |
| ollama-host | ctxcurve-ollama | ctx=8192 | 5.55 | - | - | 6566 | - | - | ok |
| ollama-host | ctxcurve-ollama | ctx=16384 | 5.31 | - | - | 6830 | - | - | ok |

## Concurrency (parallel streams)

| engine | provider | streams | sys t/s | sum stream t/s | ttft max (ms) | ttft spread (ms) | itl p99 (ms) | ok/errors | wall (s) |
|---||---||---||---||---||---||---||---||---||
| b11202-cuda | blazar | conc=8 config=conc_default | 58.6 | 571.0 | 16272 | 16034.9 | 79.24 | 16/0 | 34.96 |
| b11202-cuda | blazar | conc=8 config=adaptive_slots_off | 58.9 | 575.9 | 16588 | 16351.5 | 61.9 | 16/0 | 34.8 |
| b11202-cuda | blazar | conc=8 config=poll_50 | 58.7 | 575.2 | 16675 | 16441.5 | 30.33 | 16/0 | 34.91 |
| b11202-cuda | conc-direct | conc=4 | 110.6 | 116.7 | 276 | 4.5 | 41.39 | 4/0 | 4.63 |
| b11202-cuda | conc-blazar | conc=4 rounds=3 | 67.5 | 433.1 | 4205 | 3996.0 | 81.32 | 12/0 | 22.75 |
| v0.9.4 | conc-blazar | conc=4 rounds=3 | 67.4 | 434.3 | 4216 | 4004.4 | 32.78 | 12/0 | 22.79 |
| ollama-host | conc-ollama | conc=4 rounds=3 | 33.4 | 486.9 | 16572 | 16441.2 | 74.69 | 12/0 | 45.98 |
- sys t/s = total tokens / wall (true system throughput); sum stream t/s = sum of per-stream rates. sum >> sys means streams were serialized (queued on a single slot) rather than served concurrently.


## Quality — perplexity (identical pinned args)

| engine | perplexity | ± err | wall (s) | note |
|---|---|---|---|---|
| b11202-cuda | 17.3512 | 0.91919 | 10.0 | lower = better text fit |
| v0.9.4 | 17.3512 | 0.91919 | 10.4 | lower = better text fit |
- corpus: deterministic offline repo text (code-heavy) — PARITY-ONLY; absolute PPL is not comparable to published wiki-text perplexities.

## Quality — greedy parity vs `b11202-cuda`-direct (backend numerics)

| engine | exact matches | ratio mean | ratio min | first divergence (median chars) |
|---|---|---|---|---|
| b11202-cuda *(self — trivially 1.0) | 20/20 | 1.0 | 1.0 | 928.5 |
| v0.9.4 | 0/20 | 0.5143 | 0.0 | 0.0 |

## Quality — gateway transparency (blazar path vs direct, same engine)

| engine | exact matches | ratio mean | ratio min | first divergence (median chars) |
|---|---|---|---|---|
| b11202-cuda | 16/20 | 0.9404 | 0.0169 | 822.5 |
- expectation: 20/20 exact, ratio 1.0. A miss has TWO possible causes: gateway translation defect (sampler remap / template drift), or multi-slot batching numerics (child -np > 1 changes float reduction order; near-tie logits flip). Pin `slots = 1` and re-run: still <20/20 = translation defect, 20/20 = slot-count numerics (upstream physics).
## Feature matrix

| feature | b11202-cuda | v0.9.4 | ollama(documented) |
|---|---|---|---|
| `anthropic-api` | - | Y | - |
| `ctx-override` | Y | Y | Y |
| `embeddings` | Y | - | Y |
| `grammar-gbnf` | Y | - | - |
| `json-schema` | Y | Y | Y |
| `kv-quant` | Y | Y | - |
| `lora-adapter` | Y | Y | Y |
| `metrics-endpoint` | Y | Y | - |
| `paged-attn` | Y | Y | - |
| `parallel-np` | Y | Y | Y |
| `quant-on-load` | Y | Y | - |
| `rerank` | Y | - | - |
| `slots-sessions` | Y | - | - |
| `spec-decode` | Y | Y | - |
| `tokenize-endpoint` | Y | - | Y |
| `vision-mmproj` | Y | Y | Y |

## Reading this report

- `direct` = raw child spawn on a probed free port (no gateway).
- `blazar` = full gateway path inside a sandboxed daemon (profile compiler, routing, auth); `child_argv` in cells.jsonl holds the resolved engine argv.
- `ollama` = HTTP-only reference against the host service, one cell.
- decode counts ALL emitted tokens (content + reasoning/thinking); `usage`/engine counters are authoritative when present (`tokens src`).
- prefill t/s (cold) = prompt tokens / first-token time on an uncached token-targeted prompt; (cached) = same prompt re-sent (child prompt-cache path). ollama prefill uses engine-side prompt_eval counters, which EXCLUDE template tokens — ollama prefill reads high relative to the 512-token lanes.
- blazar speed rows show the resolved slot/context shape (`child: np=… ctx=…`) parsed from the recorded child argv — auto-slots may differ from the direct rows' explicit np.
- decode-lane TTFT rides the child's prompt cache after run 1 (warm path); the prefill-lane cold/cached pair is the honest cache story at real prompt sizes.
- GPU/power peaks sampled at ~1.2 s cadence (max across NVIDIA GPUs); very short bursts may undersample.
- Cells append to `cells.jsonl` and resume across reruns (keyed on engine/provider/params/model/harness-version).
