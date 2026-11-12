# Uncached-TTFT: chunked prefill 2048 vs 8192

{
  "title": "Uncached-TTFT: chunked_prefill_size 2048 vs default 8192",
  "date": "2026-11-12",
  "model": "qwen2.5-coder-0.5b-instruct (bf16 safetensors)",
  "engine": "sglang-0.5.23 lane, daemon-managed child, RTX 4070 Laptop 8GiB",
  "method": "native /generate?model= via gateway, non-stream, ~10K-token prompt (16470 chars), max_new_tokens 4, temperature 0, 1 cold + 3 warm per arm, daemon restart between arms",
  "arms": {
    "default_chunk_8192": {
      "cold_s": 35.08,
      "warm_s": [
        0.04,
        0.04,
        0.04
      ]
    },
    "chunked_prefill_2048": {
      "cold_s": 26.41,
      "warm_s": [
        0.04,
        0.5,
        0.04
      ]
    }
  },
  "result": "uncached/cold prefill TTFT -24.7% (35.08s -> 26.41s) with chunked_prefill_size 2048; warm runs ~0.04s both arms (radix prefix-cache hit dominates once cached)",
  "quality": "zero quality loss - scheduling knob, sampling untouched (temp 0, same output ids class)",
  "caveats": "single box, single model, 10K-prompt regime; matches prior receipt profile.rs:6948 (qwen3-1.7b, 0.5.21, 2048); 0.50 outlier in 2048 arm = one scheduler retrace",
  "reproduce": "blazar config set sglang.chunked_prefill_size 2048; restart daemon; POST /generate?model=qwen2.5-coder-0.5b-instruct with a >8192-token prompt"
}
