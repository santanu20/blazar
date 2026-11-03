# G4/G8 end-state architecture research

Date: 2026-11-03
Status: RESEARCH COMPLETE - awaiting implementation authorization
Scope: the two remaining "rest" items after the universal-architecture program
(G1/G2/G3/G3b/G5/G6/G7 shipped; session r12). Boundary constraint from the
user: blazar must not re-implement engine capabilities in the gateway.

## Item 1 - G4: ollama-exported arches (glmocr, ggmlc)

### Finding: this is not an engine gap at all

The upstream feature request ggml-org/llama.cpp#19335 ("Add Support for GLM-OCR
Model") was closed as completed via PR #19677, and ggml-org ships an official
GGUF mirror: ggml-org/GLM-OCR-GGUF (47k downloads).

Live header probe (range request, this session, 2026-11-03):

- GLM-OCR-Q8_0.gguf      -> general.architecture = glm4   (main model)
- mmproj-GLM-OCR-Q8_0.gguf -> general.architecture = clip (vision tower)

The main weights are a plain glm4 checkpoint - an arch our installed engine
(llama.cpp b12020) has loaded since long before this build; strings probe
confirms the glm4/chatglm4 loader plus the clip tower family. The "glmocr"
arch token exists only in ollama's export pipeline, which renames and
re-shapes tensors into their own serving format. The same is true of "ggmlc"
(classifier exports): the official equivalent, ggml-org/Laya-GGUF
(modern-bert), is already in our catalog and serving live.

Conclusion: no converter, no engine fork, no weight surgery is needed. What
is missing is PULL-SIDE STEERING - blazar today accepts an ollama-exported
pull, records a row the engine cannot boot (502 at serve time), and only
teaches via the list dagger after the fact.

### Recommended architecture: official-repo steering (3 small pieces)

1. Catalog entry: `glm-ocr` -> ggml-org/GLM-OCR-GGUF (class: vision; the
   mmproj attaches automatically through the existing vision lane - the same
   machinery qwen3vl rows already ride; Q8_0 950MB + mmproj 484MB).
2. Pull-time teaching: when a pulled GGUF's arch token is a known ollama
   remap (start set: glmocr, ggmlc), stop with a redirect teaching naming
   the official repo (`this is an ollama-exported layout the engine cannot
   load; the official engine-native export is ggml-org/GLM-OCR-GGUF - blazar
   pull glm-ocr`). Placement: the hf.rs pull path where arch is already
   extracted, next to the unknown-diffusion-family teaching precedent.
3. Close the loop on blazar#2 (the ollama-import tracking issue): the
   steering + catalog entry is the complete fix available today.

Rejected alternatives:
- Convert-on-import (shelling to convert_hf_to_gguf.py): solves a problem
  that no longer exists - official GGUFs are published; python+repo
  dependency for zero benefit. Rejected.
- Rust-side tensor remap: weight surgery in blazar; violates the boundary
  mandate; upstream publishes the artifacts. Rejected.

Live validation plan (when authorized): sandbox pull glm-ocr -> row (arch
glm4) + mmproj auto-attach -> OCR one image through /v1/chat/completions ->
receipt; ggmlc teaching probe on the existing laya-typed-decisions row's
repo pattern.

## Item 2 - G8: kokoro (TTS)

### Finding: kokoro has no engine today; piper1-gpl does not bring it

Evidence (all probed this session):

- llama.cpp b12020 libllama/libmtmd strings: no kokoro arch, no kokoro
  mtmd tower (gen-audio towers: qwen3tts, pockettts, lfm3a, soprano,
  muse_glimmer/spark).
- llama.cpp v0.8.0 release (older than our b12020; nightly b11960 cited
  inside it) changelog: new TTS types SOPRANO and LFM3A, Muse Spark Open
  audio - still no kokoro. Kokoro is absent from latest upstream.
- piper1-gpl (OHF-Voice, GPL-3.0, active, v1.8.0): voices are VITS
  onnxruntime exports from the same rhasspy/piper-voices tree (docs/
  VOICES.md) - the exact voice format our existing piper lane already
  serves. It adds python-wheel distribution (piper_tts cp39-abi3 wheels)
  on top of what we have and brings ZERO new voice architectures.
  Rejected as an engine lane on both counts (no kokoro, python runtime
  dependency in a native-binary engine fleet).

### Options ranked

A. Engine-upgrade path (recommended, zero work): when llama.cpp gains a
   kokoro tower, the G1 engine speech lane serves it with no gateway
   change (the lane already forwards /tts for audio-output rows). Track
   upstream; the engine-update flow picks it up.
   Meanwhile the G1 lane already covers high-quality speech: Qwen3-TTS
   (live receipt), plus pockettts/LFM3-Audio/soprano GGUFs upstream.

B. In-process kokoro-onnx lane (optional future wave): kokoro is a small
   ONNX model; an ort-crate lane would mirror the candle CLIP in-process
   pattern. Costs: onnxruntime dependency footprint, espeak-ng/misaki
   G2P porting to rust, a permanent inference-maintenance tail for one
   voice model. Only justified if kokoro becomes a hard user requirement
   before an engine ships it. (kokoro-onnx project state NOT verified
   this session - verify before ever starting this.)

C. piper1-gpl engine adapter: rejected on evidence above.

### Recommendation

Ship item 1 (steering + catalog + teaching) - small, fully testable, ends
the last false-negative of the architecture matrix. Leave G8 on the
engine-upgrade path (A) and document kokoro alternatives users can pull
today.

## Evidence index

- b12020 strings: glm4/chatglm4 + clip towers present; glmocr/kokoro absent.
- GLM-OCR-GGUF headers: glm4 + clip (range-request probe, 2026-11-03).
- ggml-org#19335 closed via #19677; ggml-org/GLM-OCR-GGUF live (46,963
  downloads at probe time).
- llama.cpp v0.8.0 changelog: SOPRANO + LFM3A TTS types, no kokoro.
- piper1-gpl VOICES.md: VITS/onnx voices only; releases are python wheels.
