# SGLang current-state, decision APIs, and the unified engine feature matrix

Date: 2026-10-12 · Scope: research report (no code change) ·
Sources: docs.sglang.io (fetched 2026-10-12, `.md` endpoints + `llms.txt`),
PyPI package metadata, llama.cpp `tools/server/README.md` @ master,
mistral.rs release notes, stable-diffusion.cpp repo meta, blazar codebase
(`crates/blazar-gateway`, `crates/blazar-core/src/profile.rs`, validate.py
battery). Every upstream claim below was fetched today; unverifiable claims
are marked explicitly.

## 1. What "SGLang V1" means now

SGLang's V1 runtime (the zero-overhead scheduler / overlap-scheduler /
cancel-at-will rearchitecture announced on the SGLang blog in the 0.4.x era)
is no longer a separate thing to adopt: it has been the only engine
architecture for the entire 0.5.x line. There is no "V1 page" in current
docs — the feature pages (session-aware radix cache, hierarchical cache,
DP attention, speculative decoding, PD disaggregation) ARE the V1 feature
set absorbed into the product.

Blazar's position: **current**. The sglang lane's flag surface is verified
against 0.5.19 and **0.5.21** (`sglang_flags_0521`, probe receipt in
`engines.manifest`, 2026-10-09), and 0.5.21 is the latest PyPI release
today. Repo moved orgs: `sglang-ai/sglang` → `sgl-project/sglang`
(PyPI homepage; docs live at docs.sglang.io / docs.sglang.ai — both serve
the same Mintlify site).

Note: the original V1 announcement blog (blog.sglang.ai) returned an empty
body to a plain fetch today (JS-rendered); the performance claims from that
post are deliberately NOT repeated here — only the doc-verifiable feature
list above.

## 2. "Decision API" — resolved

The term maps to a real, currently-landing model class with TWO dialects:

### 2.1 SGLang `/v1/decisions` — typed decisions on ANY chat model

- Request: `{ "input": str, "questions": [ { "id", "type":
  "choice"|"score"|"yes_no", "question", "options"?: [{name, description}],
  "levels"?: [str] } ] }`.
- Answer per question id: `{ "type", "probabilities"` (one value per
  option/level, or `yes`/`no`, summing to 1)`, "choice"` (argmax option) or
  `"score"` (probability-weighted mean level index)`, "label_mass"` (the
  full-vocabulary probability mass on the answer labels — a built-in
  confidence/quality signal).
- Mechanics: no text is generated, nothing is parsed — the model's
  next-token scores at the answer position are read directly. **No special
  checkpoint needed: any generation model with a Jinja chat template
  works.** Questions are answered independently, scored together in one
  batch.
- Availability: **nightly-only today** ("Until a release contains it,
  install a nightly build" — docs). NOT in 0.5.21.
- Sibling endpoints in the same family: `/v1/score` (decoder-only scoring),
  `/classify` (reward model), `/v1/rerank` (cross-encoder), plus MoE expert
  distribution capture and `/tokenize` // `/detokenize`.

### 2.2 llama.cpp decision models — native checkpoints via `/v1/systemone`

- Model metadata now carries an `architecture` object with
  `input_modalities` / `output_modalities`; `decisions` in
  `output_modalities` marks a **native decision model** (a trained
  checkpoint — e.g. Laya, which blazar already routes: systemone laya =
  ggml-org/Laya-GGUF). A chat model that merely classifies via prompts
  does NOT get `decisions`.
- Native decision models are served through **`/v1/systemone`** (the
  "TypeSafe-compatible System One API").
- Discovery works WITHOUT loading the model: GET `/v1/models` returns the
  same `architecture` object for unloaded/sleeping models.
- Momentum: llama.cpp b11572 (2026-10-12, the day of this report) added
  `pplx-decider` + "vllm d2/d3" model support — the decider family is
  actively landing upstream right now.

### 2.3 Blazar's position and the unified opportunity — SHIPPED 2026-10-12

Blazar's `/v1/classify` is now a dual-dialect lane: llamacpp → child
`/v1/systemone` (native decision checkpoints, unchanged), sglang → child
`/v1/decisions` (typed questions on ANY chat model). The installed
0.5.21 release child serves `/v1/decisions` live — the docs'
"nightly-only" note is stale; verified end-to-end (card: top/labels
sorted by probability, `confidence` = `label_mass`, model echoed as the
requested name). MistralRs/MLX lanes keep the teaching refusal.

1. **DONE** — sglang-lane decisions via `/v1/decisions` (see above).
2. **DONE in part** — `label_mass` surfaces as the classify card's
   `confidence` on the decisions path (the systemone path keeps
   llama-server's own `confidence` field, unchanged).
3. **Decision-model discovery parity** (deferred, see §5).

## 3. Stale-gate finding — FIXED 2026-10-12

`LLAMACPP_ONLY_PATHS` blanket-blocked 19 child surfaces for non-llamacpp
lanes. Live probe of a booted sglang 0.5.21 child showed it serves
`/tokenize`, `/detokenize`, `/v1/systemone` (its own dialect) and
`/v1/rerank` natively — so the gate is now per-lane
(`path_gated_for`): sglang passes those four byte-faithfully (the
child's own validation errors surface verbatim, e.g. its `/tokenize`
wants `{prompt}`, not llama-server's `{content}` dialect); every other
listed surface still teaches on sglang; mistralrs/mlx/sdcpp keep the
full gate. The teaching message text is unchanged. 404-verified absent
on the sglang child: /apply-template /infill /completion /embeddings
/rerank /reranking /props /slots /v1/stream (+ /v1/embeddings needs
`--is-embedding`, which blazar emits only for embedder rows — the bare
`/embeddings` native path stays gated).

## 4. Unified engine feature matrix (2026-10-12)

Engines: llama.cpp (b11572 today; blazar box b11539-cuda active, b11574
available) · sglang 0.5.21 (= blazar's verified target) · mlx-lm 0.32.0 ·
mistral.rs v0.9.4 (2026-09-24; v0.9.3 brought first-class FP8 + concurrent
DFlash decode + scheduler/concurrency fixes) · sdcpp (7.6k stars, pushed
today; SD/Flux/Wan/Qwen-Image/Z-Image — blazar's diffusion lane).

| Surface | llama.cpp | sglang 0.5.21 | mlx-lm 0.32 | mistral.rs | blazar unifies via |
|---|---|---|---|---|---|
| Chat/completions + streaming | ✓ | ✓ | ✓ | ✓ (tool emission = "lottery", battery-documented) | OpenAI + ollama + Anthropic `/v1/messages` translate lanes |
| Tools (forced choice) | ✓ | ✓ parser map (`--tool-call-parser`, blazar maps qwen3/qwen25/…) | ✗ voluntary calls only (gateway gate rejects forced) | partial | translate + per-lane gates |
| Structured output | ✓ gbnf/JSON schema | ✓ (incl. reasoning-model variant, `--grammar-backend`) | ✗ | ✓ | gateway schema lanes |
| Speculative decoding | ✓ EAGON/n-gram/self-draft (`--spec-*`, blazar tune `--spec-race`) | ✓ EAGLE-3 + adaptive (`--speculative-*` incl. accept thresholds — flag-verified) | ✗ | ✓ DFlash concurrent (v0.9.3) | tune/bench lanes |
| Radix / prefix cache | partial (prompt cache) | ✓ session-aware radix + eviction policies + HiCache hierarchical | ✗ | ✗ | blazar session bank + cache-reuse tuning |
| KV quantization | ✓ | ✓ (`--kv-cache-dtype`) | ✗ | ✓ | profile ladder |
| Embeddings | ✓ pooling | ✓ (`/encode`, `--is-embedding`; blazar emits the flag) | ✗ | ✓ | `/v1/embeddings` + late-chunking |
| Rerank | ✓ `/v1/rerank` | ✓ `/v1/rerank` native | ✗ | ✗ | ollama-lane rerank; **OpenAI-lane gate stale (§3)** |
| Decisions | ✓ native models `/v1/systemone` + `architecture` discovery | ✓ `/v1/decisions` ANY chat model (nightly-only) | ✗ | ✗ | `/v1/classify` (llamacpp-only today) — bridge plan §2.3 |
| Anthropic `/v1/messages` | — | ✓ upstream compat (Claude Code, prefix-cache tuning) | — | — | blazar has its own translate layer (engine-agnostic) |
| Ollama-compat API | — | ✓ upstream | — | — | blazar IS the ollama API (port 11434 parity mode) |
| LoRA | ✓ | ✓ serving (multi, `base:adapter` stamp — blazar rerank lane uses it) | ✓ | ✓ | blazar lora lanes + sync |
| Multimodal (vision) | ✓ mmproj | ✓ (incl. `--mm-attention-backend`, blazar-flag-verified) | ✓ | ✓ | vision lanes |
| PD/EP/DP parallelism | ✗ | ✓ PD + EPD disagg, DP/DPA router, prefill/decode CP, expert parallel | ✗ | ✗ | out of scope for single-box blazar today |
| MoE expert analytics | ✗ | ✓ expert-distribution capture endpoints | ✗ | ✗ | unexploited — candidate for scorecard/why |
| Quant formats | GGUF (all quants) | safetensors AWQ/GPTQ/FP8/bitsandbytes + torchao (config-truth verified live) | MLX 4/8bit dirs | GGUF+safetensors | format-based auto-routing |
| CPU/offload | ✓ | `--cpu-offload-gb` | Apple-only | ✓ | fit/planner |
| Sleep/wake | ✓ (sleep-on-idle class flags, 0.5.21-flag-verified) | `--sleep-on-idle` (0.5.21) | ✓ | ✗ | blazar rotation/keep-alive economy |

(✓/✗ cells for mlx/mistral reflect blazar's live battery + lane gates
where external docs weren't fetched today; sglang/llama.cpp cells are
doc-fetched. mistral.rs embedding ✓ and mlx LoRA ✓ ride blazar lane
verification, marked as such.)

## 5. Ranked recommendations — status

1. **DONE (2026-10-12)** — stale sglang gate split per-lane; four
   verified surfaces pass; teaching text unchanged (§3).
2. **DONE (2026-10-12)** — decision-API bridge: `/v1/classify` serves
   the sglang lane via `/v1/decisions`, `label_mass` → `confidence`
   (§2.3). Battery: new per-lane check green; llamacpp card contract
   byte-identical.
3. **BLOCKED BY HARDWARE (2026-10-13, evidence complete)** —
   expert-distribution endpoints → `blazar scorecard`/`why`. An MoE
   child was pulled to live-validate (`ibm-granite/granite-4.0-h-micro`,
   the smallest sglang-servable MoE/hybrid checkpoint at 6.4 GiB), but
   sglang 0.5.21 cannot initialize its hybrid mamba state pool on an
   8 GiB card under ANY knob combination tried (ctx 8192,
   `--max-running-requests 2`, `--cpu-offload-gb 3` via the
   `sglang.cpu_offload_gb` ladder pin, `--mamba-full-memory-ratio 0.5`,
   `--mamba-max-states-per-path 2`): every attempt dies with
   `Hybrid (mamba/linear-attention) state cache is too small …
   max_num_reqs=0 — use GPUs with more memory` (sglang's own error).
   Every other safetensors MoE is larger (granite-4.0-h-tiny is
   12.9 GiB; LFM/Qwen MoEs ≥ 8 GiB of weights alone). Until a
   bigger-GPU box or a sub-4 GiB MoE checkpoint exists, the dump
   format cannot be probed honestly and no parser ships against a
   guessed shape. Revisit on 24 GiB+ hardware. (Side finding: stale
   engine children from crashed sessions can squat VRAM and
   contaminate spawn diagnosis — check `nvidia-smi` for foreign
   pids before reading ENGINE_CRASHED as a model-fit verdict.)
4. **DONE (2026-10-13, live-proven)** — `architecture.output_modalities`
   passthrough: llama-server ≥ b11572 exposes the object; engine
   updated to b11574-cuda via the new token-free atom fallback, plain
   Laya (modern-bert) advertises `output_modalities: ["decisions"]` on
   its child, and the gateway now captures it on first listing (store
   schema v16 `model_modalities` cache) and merges it into
   `capabilities` — `GET /v1/models` shows `laya:bf16 ["decisions"]`,
   persisted across daemon restarts. The pplx-decider-class
   `mys/laya-typed-decisions` GGUF (arch `ggmlc`) is pulled and
   routed but NOT loadable by any released llama.cpp yet — arch
   lands post-b11574; the row activates itself when upstream ships
   it.
5. **DONE (2026-10-13)** — bonus from this workstream: engine updates
   no longer need GitHub API budget. `blazar engine update` falls
   back to the anonymous `releases.atom` feed +
   `releases/expanded_assets` HTML table when api.github.com 403s the
   IP (digest-less assets install as "unverified", the same mark pip
   lanes carry). Live-proven: `engine update all` with zero tokens
   walked 6/6 lanes, installed b11574-cuda + sdcpp master-959 through
   the fallback while the API was still rate-limited.
6. **Watch list**: sglang nightly → release cadence for decisions;
   llama.cpp decider family growth (pplx-decider, vllm d3, b11572+;
   `ggmlc` arch for typed-decisions GGUFs); mistral.rs v0.9.x
   scheduler line; mlx-lm release notes (0.32.0 today).

## Reproduce the fetches

```sh
curl -sL https://docs.sglang.io/llms.txt                     # page index
curl -sL https://docs.sglang.io/docs/basic_usage/native_api.md
curl -sL https://docs.sglang.io/docs/supported-models/decision_models.md
curl -s  https://pypi.org/pypi/sglang/json | jq -r .info.version
curl -sL https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/server/README.md
```

Blazar-side anchors: `LLAMACPP_ONLY_PATHS` (openai.rs:1040),
`classify` (classify.rs:108 → child `/v1/systemone`), `rerank`
(ollama.rs:3002 → child `/v1/rerank`, sglang stamp aware),
`sglang_flags_0521` (profile.rs:15199), `sglang_tool_call_parser`
(engine_impl.rs:1114).
