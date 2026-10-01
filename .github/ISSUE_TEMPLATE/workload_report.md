---
name: Workload report
about: How Blazar behaves under your real workload (good or bad)
labels: workload
body:
  - type: textarea
    id: workload
    attributes:
      label: The workload
      description: What you run through Blazar — client(s), concurrency, prompt sizes, streaming or not, models.
      placeholder: |
        Codex CLI agent, 1-2 concurrent sessions, long contexts
        (~30k), mostly streaming chat with tool calls.
    validations:
      required: true
  - type: textarea
    id: hardware
    attributes:
      label: Hardware + engine
      description: GPU + VRAM (or CPU-only), engine lane in use (`blazar ls` ENGINE column), quant.
    validations:
      required: true
  - type: textarea
    id: measurements
    attributes:
      label: What you measured
      description: TTFT / tokens-per-second / queue waits / VRAM pressure / failure rates — numbers if you have them (`blazar bench`, `/metrics`, `/api/ps`). Anecdotes are fine too, marked as such.
  - type: dropdown
    id: verdict
    attributes:
      label: Overall verdict
      options:
        - Works well
        - Mixed
        - Painful
    validations:
      required: true
  - type: textarea
    id: pain
    attributes:
      label: The single worst friction point
      description: If we could fix one thing about how Blazar serves THIS workload, what is it?
---
