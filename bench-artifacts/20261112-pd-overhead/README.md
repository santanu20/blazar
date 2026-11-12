# PD disaggregation overhead - direct lane vs pair+router (single-GPU cost receipt)

- Date: 2026-11-12 | Blazar: 2385c4a+ (7-commit session tree) | binary v0.31.0
- Engine: sglang-0.5.23 lane, mooncake-transfer-engine==0.3.11 (TCP), sglang-router==0.3.2
- Model: qwen2.5-coder-0.5b-instruct (bf16, 942 MiB) | GPU: RTX 4070 Laptop 8 GiB (single)
- Workload: native `/generate?model=` non-stream, ~2.4K-token prompt, max_new_tokens 8, temp 0
- Method: 1 cold (incl. spawn+load) + 3 warm per arm; daemon restart between arms;
  knobs `sglang.disagg_pair=true` + `sglang.disagg_transfer_backend=mooncake_tcp` for arm B.

## Results

| arm | cold (wall) | warm wall (3x) | e2e_latency cold | e2e warm | cached_tokens warm |
|---|---|---|---|---|---|
| A direct lane | 27.07 s | 0.06-0.07 s | 0.144 s | ~0.055 s | 2407 |
| B pair + sgl-router | 67.00 s | 0.07-0.08 s | 0.207 s | ~0.070 s | 2407 |

## Reading

- Cold: pair spawns 3 processes (decode + prefill + router) -> ~2.5x cold-start (2x model
  load + 2x torch ctx + router worker discovery; first request may also hit the router's
  one-shot `server_selection_failed` 503 during async discovery).
- Steady state: +27-44% per-request e2e overhead (router hop + mooncake KV transfer +
  retraction-none decode path). Radix-cached warm requests skip the transfer; overhead
  shrinks to the router hop.
- **Single-GPU PD is a topology/wiring tier, not a perf win** - prefill and decode share
  one GPU, so there is no prefill-scaling benefit to offset the hop. PD pays off
  multi-GPU/multi-node where prefill workers scale independently.
- No quality loss: temperature-0 determinism, identical cached_tokens counts.

## Reproduce

```
blazar stop qwen2.5-coder-0.5b-instruct            # cold arm A
# POST /generate?model=qwen2.5-coder-0.5b-instruct (native body, ~2.4K-token prompt)
blazar config set sglang.disagg_transfer_backend mooncake_tcp
blazar config set sglang.disagg_pair true
sudo systemctl restart blazar.service              # spawn paths need restart
# POST same body (retry once if router 503 server_selection_failed)
blazar config unset sglang.disagg_pair && blazar config unset sglang.disagg_transfer_backend
sudo systemctl restart blazar.service
```
