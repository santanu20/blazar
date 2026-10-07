---
layout: doc
title: Benchmarks
description: Measured engine and gateway performance — direct-engine parity, concurrency scaling, tail latency, and lifecycle — with receipts.
doc_kind: Documentation
---
## How we benchmark

Every number on this page comes from a scripted, reproducible campaign against
real engine binaries on real hardware — direct engine versus the same engine
behind the Blazar gateway, identical model, quantization, context, sampling,
and cache state. The full receipt tables, failed cells, and caveats live in
[BENCHMARK.md](https://github.com/santanu20/blazar/blob/main/BENCHMARK.md);
this page shows the campaign charts and headline results.

**Test bed (campaign 2026-09-29):** Intel Core i7-14650HX (24 threads) ·
NVIDIA GeForce RTX 4070 Laptop, 8 GiB, driver 580.173.02 · 16 GiB RAM ·
Linux Mint 22.3, kernel 7.0.0-31-generic · model Qwen3.5-9B-Q4_K_M.gguf.
Gateway measured at blazar 0.13.0; ratios travel better than absolutes.

## Headline results

<div class="stat-grid" markdown="1">

- **~2%** gateway overhead — 40.6 t/s through the gateway vs 41.5 t/s direct
- **104.8 t/s** aggregate at C=4 (40.0 → 65.9 → 104.8 across the sweep)
- **554 ms** TTFT p99 at C=4 under sustained bursts
- **117 ms** tool-call TTFT p50 vs 284 ms (ollama) — 2.4×
- **3.63 s** idle-wake to first token
- **49.1 → 63.0 t/s** adaptive reshape under sustained load, zero failed requests

</div>

## Flagship GGUF campaign — Qwen3.5-9B-Q4_K_M

### Single-stream decode throughput

<img src="/blazar/assets/bench/flagship/speed-single-stream.svg" alt="Single-stream decode throughput: gateway vs direct engine" class="bench-chart">

### Gateway overhead

<img src="/blazar/assets/bench/flagship/gateway-overhead.svg" alt="Gateway overhead versus direct engine" class="bench-chart">

### Concurrency scaling

<img src="/blazar/assets/bench/flagship/concurrency-throughput.svg" alt="Concurrency scaling: aggregate throughput vs parallel streams" class="bench-chart">

### Concurrency tail latency

<img src="/blazar/assets/bench/flagship/concurrency-ttft.svg" alt="TTFT tail latency vs concurrency" class="bench-chart">

### Resource cost vs concurrency

<img src="/blazar/assets/bench/flagship/concurrency-vram.svg" alt="VRAM and memory cost vs concurrency" class="bench-chart">

### Long-context degradation

<img src="/blazar/assets/bench/flagship/ctx-curve.svg" alt="Throughput degradation across context lengths" class="bench-chart">

### VRAM vs context

<img src="/blazar/assets/bench/flagship/vram-vs-context.svg" alt="VRAM footprint across context lengths" class="bench-chart">

### Cold start and idle wake

<img src="/blazar/assets/bench/flagship/lifecycle-cold-idle.svg" alt="Cold start and idle wake latency" class="bench-chart">

## SGLang safetensors campaign

The same harness against the SGLang lane on quantized safetensors — a
different engine, the same gateway behavior.

<img src="/blazar/assets/bench/sglang/speed-single-stream.svg" alt="SGLang campaign: single-stream throughput" class="bench-chart">

<img src="/blazar/assets/bench/sglang/concurrency-throughput.svg" alt="SGLang campaign: concurrency scaling" class="bench-chart">

<img src="/blazar/assets/bench/sglang/concurrency-ttft.svg" alt="SGLang campaign: TTFT tail latency" class="bench-chart">

<img src="/blazar/assets/bench/sglang/concurrency-vram.svg" alt="SGLang campaign: resource cost vs concurrency" class="bench-chart">

<img src="/blazar/assets/bench/sglang/ctx-curve.svg" alt="SGLang campaign: long-context degradation" class="bench-chart">

<img src="/blazar/assets/bench/sglang/vram-vs-context.svg" alt="SGLang campaign: VRAM vs context" class="bench-chart">

<img src="/blazar/assets/bench/sglang/lifecycle-cold-idle.svg" alt="SGLang campaign: cold start and idle wake" class="bench-chart">

## Reproduce

The harness and per-campaign reports ship in the repo
under `bench-artifacts/` (summary layer: reports + `INDEX.md`); raw
cells (JSONL), charts, and media payloads live on the
[bench-archive release](https://github.com/santanu20/blazar/releases/tag/bench-archive-2026-10).
The full methodology, checker-verified quality
suites, and every caveat are documented in
[BENCHMARK.md](https://github.com/santanu20/blazar/blob/main/BENCHMARK.md#reproduce).
