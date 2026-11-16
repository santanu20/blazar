# ComfyUI engine lane — need & feasibility research

Date: 2026-11-14 · Status: research complete, decision pending · Scope: 10th engine lane (ComfyUI) alongside sdcpp for diffusion/video.

## Verdict up front

- **Need: real, but it is a *differentiation* need, not a coverage need.** Plain txt2img/img2img/video is already served lighter by the sdcpp lane. What only ComfyUI brings: node-graph **workflows** (multi-model chained pipelines in one request), the **custom-node ecosystem** (official Registry), and **day-0 support for new diffusion families** (Comfy-Org repackages same-day; sd.cpp lags architectures).
- **Feasibility: HIGH.** Every integration primitive already exists in blazar: venv-pinned lane installs, git/tag pinning, model staging via hardlinks + component taxonomy, a shallow-translator images gateway, async job routes, capabilities relay, WS/SSE streaming, idle reapers, VRAM admission. Effort ≈ one feature wave (Phase A/B) + one follow-up wave (Phase C).
- **License: GPL-3.0 — compatible with our model.** Process-spawned engine child, zero code linking, same isolation as every other engine (blazar gateway stays MIT/Apache-2.0). Verified from upstream LICENSE file.

## Upstream facts (all verified 2026-11-14)

### Project state

| Fact | Value | Source |
|---|---|---|
| Repo | comfyanonymous/ComfyUI (canonical org Comfy-Org), 146,779 stars, 17,510 forks, pushed 2026-11-14 | GitHub API |
| License | GPL-3.0 | LICENSE file |
| Versioning | Proper tags: v0.45.3 latest; weekly-ish minor cadence (v0.35→v0.45.0 over Sep 9–Nov 10 2026, 12 releases) | GitHub releases API |
| Release assets | **Windows portable 7z only** (nvidia, nvidia_arm64, nvidia_cu126, intel, amd) — no Linux/macOS binaries; those install = git clone + venv + pip | GitHub releases API |
| Docs | docs.comfy.org, fully machine-readable (`/llms.txt` index, `.md` fetch), incl. Server Routes + Startup Flags references | fetched |

### Runtime & platforms

- aiohttp server (`python main.py`), default 127.0.0.1:8188; headless by nature (frontend is a separate artifact fetched only for the UI route; API needs no browser).
- Python 3.13 in the Windows portable; PyTorch >= 2.7 supported, CUDA 13.0 stable channel (`--extra-index-url https://download.pytorch.org/whl/cu130`).
- Backends: NVIDIA CUDA (Linux+Win), AMD ROCm 7.2 Linux + ROCm-on-Windows (RDNA 2/3/3.5/4), Intel Arc (torch.xpu), Apple MPS, Ascend NPU / Cambricon MLU / Iluvatar (exotic), CPU. **No Vulkan** — sdcpp keeps that niche.
- Windows/Linux/macOS officially supported.

### Server surface (what a lane would drive)

HTTP+WS API (docs: `comms_routes`, `api-examples`):

- `POST /prompt` — queue a **workflow graph JSON** (`{node_id: {class_type, inputs}}`) with `client_id`/optional `prompt_id`; async queue built-in. `GET /prompt` = queue status.
- `GET /history/{prompt_id}` — results (outputs per node, filenames); `GET /view?filename&subfolder&type` — fetch bytes.
- `GET /object_info[/{node}]` — **full self-describing node catalog** (every node's params, types, defaults) — the natural source for a capabilities endpoint, exactly like sdcpp's `/sdcpp/v1/capabilities` relay.
- `GET /models/{folder}`, `/embeddings`, `/system_stats` (python + devices + VRAM — admission-grade telemetry), `/features`, `/workflow_templates`.
- `POST /upload/image`, `/upload/mask` (img2img/mask input), `/queue` (manage), `/interrupt`, `/free` (unload models).
- `GET /ws` — execution events (`status`, `execution_start`, `execution_cached`, `executing`, `progress`, `executed`) + `SaveImageWebsocket` binary frames.

Launch flags relevant to a lane (from `comfy/cli_args.py` source):

- Net: `--listen`, `--port`, `--enable-cors-header`, `--tls-keyfile/certfile`, `--max-upload-size`.
- Directories: `--base-directory`, `--models-directory`, `--input/output/temp-directory`, **`--extra-model-paths-config` (one or more YAMLs)** — the model-staging seam; ComfyUI's folder taxonomy (checkpoints, vae, text_encoders, controlnet, loras...) matches the component sets blazar already pulls from Comfy-Org repos.
- VRAM: `--reserve-vram`, `--vram-headroom` (DynamicVRAM), `--cuda-device`, `--cuda-malloc/--disable-cuda-malloc`, `--cpu`.
- Behavior: `--disable-auto-launch`, `--front-end-version`, `--database-url`, `--feature-flag KEY[=VALUE]`.

### Ecosystem

- **ComfyUI Registry** (registry.comfy.org) = official custom-node package registry (thousands of node packs; Manager installs from it). Custom nodes are arbitrary Python — security-sensitive, must be opt-in.
- **SwarmUI** (mcmonkeyprojects, MIT, 4,784 stars, active) is the precedent: it wraps ComfyUI as a backend behind its own friendlier API — proves the wrap-pattern works, and it doesn't cover our lane (local API gateway, headless, model-store-native).
- ComfyUI ships an official **MCP server** (docs `agent-tools/mcp`) — composes with our MCP-client surface later.
- Comfy-Org repos are already the de-facto diffusion model distribution format: blazar's sdcpp lane *already pulls* `Comfy-Org/Wan_2.1_ComfyUI_repackaged` split_files and hardlinks byte-exact files from users' ComfyUI-populated hf-hub caches (CHANGELOG 0.19.0 era). Model store overlap ≈ free.

## Need analysis — what it adds vs sdcpp

| Capability | sdcpp lane today | ComfyUI lane |
|---|---|---|
| Plain txt2img/img2img (SD1.5/SDXL/Flux/Qwen-Image/Z-Image/Chroma) | yes, light binary, Vulkan+CUDA | yes, but heavier (torch venv) — no advantage |
| Video (Wan T2V/S2V) | yes (`vid_gen`) | yes + richer video pipelines (AnimateDiff-class chains, I2V graphs) |
| Multi-model chained pipelines (gen→upscale→restore, ControlNet stacks, IPAdapter, regional conditioning) | **no** (one component set per child) | **yes — the core differentiator** (one request = whole DAG) |
| Custom-node ecosystem | no | yes (Registry; opt-in, default-deny for security) |
| New model family support latency | sd.cpp arch lag (weeks-months) | day-0 (Comfy-Org same-day repackages; core nodes ship with examples) |
| Vulkan / low-end iGPU | yes | no |
| Interactive node UI | no | yes (not our product face; gateway stays the API) |

Unique gateway surface this unlocks: **workflow-as-endpoint** — user pins a workflow template (exported API-format JSON), blazar exposes it as a named endpoint with declared inputs (seedable from `/object_info` schemas). Neither ollama nor any local competitor serves that.

Honest counterweights:

1. Install weight: torch venv ~3-6 GB vs sd-server ~200 MB binary. Two diffusion lanes coexisting is fine (lazy lanes), but disk cost must be taught at install.
2. Global single execution queue per instance — concurrency story is "one workflow at a time per child" (matches our lazy-lane + idle-reaper model; no slot parallelism like llama.cpp).
3. Custom nodes = arbitrary code execution. Default lane must ship **core nodes only**; node packs only via explicit `engine install <tag> --kind comfyui --with <registry-spec>` (mirrors the sglang `--with` + DSpine plugin precedent), never auto-installed.
4. GPL-3.0: process isolation, no linking — same posture as shipping any GPL engine child; gateway license unaffected.

## Feasibility — integration sketch

Existing blazar primitives cover every requirement:

1. **Install lane** (`comfyui_install.rs`, pattern: kokoro_install.rs + sglang lane):
   - Linux/macOS: `git clone` at pinned release tag + `uv venv` + GPU-vendor-correct torch wheel index + `pip install -r requirements.txt` (pinned manifest, sha-verified assets where possible — matches the no-docker, cross-OS rule).
   - Windows+NVIDIA (and intel/amd variants): **use the official portable 7z asset** — self-contained embedded Python, no venv needed; closest match to our engine-binaries model. Asset picker mirrors sdcpp's GPU-vendor picker.
2. **Model staging**: store rows → `--models-directory` + generated `extra_model_paths.yaml` mapping our component taxonomy to ComfyUI folders; hardlink from store blobs (existing hardlink-staging + hf-cache reuse machinery). Comfy-Org component sets pulled by the sdcpp lane are directly reusable.
3. **Supervision**: child = `python main.py --listen 127.0.0.1 --port <ephemeral> --disable-auto-launch ...`; health via `/system_stats`; idle reaper (whisper-mirror 900s); VRAM admission from `/system_stats` devices (honest — no argv-predictable VRAM like GGUF).
4. **Gateway**:
   - Phase A (native passthrough): mount ComfyUI API routes under `/comfy/...` (or per-model prefix) — `/prompt`, `/history`, `/view`, `/object_info`, `/queue`, `/interrupt`, `/upload/*`; WS `/ws` relayed (we already relay WS for realtime voice).
   - Phase B (OpenAI translation): `/v1/images/generations|edits` shallow-translator to curated per-family workflow templates (KSampler graphs for SD/SDXL, Flux sampler nodes, Wan video) — reuses the images.rs translator pattern (`n`→batch Count node, `size`→EmptyLatent, quality/style → template params, everything else verbatim into template variables); async rides `/prompt` + `/history` polling mapped to existing job routes; streaming rides `/ws` progress → SSE frames.
   - Phase C (differentiator): `POST /v1/workflows` (register API-format JSON + declared inputs) + `POST /v1/workflows/{id}/run` + capabilities from `/object_info`.
5. **Config exposure**: `[comfyui]`/flat `comfyui_*` knobs for the launch surface (`extra_model_paths`, `reserve_vram`, `vram_headroom`, `cuda_device`, `cpu`, `front_end_version`, `max_upload_size`, `extra_args` manifest-gated passthrough refusing reserved flags) — same three-tier contract as every lane (first-class / passthrough / request-time), with `engine_coverage.py` dispositioning the full flag universe from `cli_args.py`.
6. **Certification**: live spawn receipts (workflow submit → image bytes round-trip), the engine-coverage doc row, bench harness lane (direct-vs-gateway cells like every other lane; quality = deterministic seed parity).

Risks / open questions:

- Torch wheel matrix per OS x GPU (mac MPS wheel sizes; ROCm-Windows wheel availability; Intel xpu channel) — mitigated by the Windows portable 7z lane on Win, and by failing with teaching (our install lanes already do GPU-vendor detection for sdcpp).
- Node-catalog drift across weekly upstream releases — mitigated by pinned-tag installs (our unpinned-latest-with-fallback pattern applies to the lane tag policy) and `/object_info`-derived (never hardcoded) capability surfaces.
- Frontend download at first boot — API-only operation needs no frontend; if the `/` UI route is hit, ComfyUI fetches it on demand (document; offline boxes just don't get the UI route).

## Recommendation

Proceed as a **lazy opt-in lane** (Phase A+B in one wave, Phase C as a follow-up): `blazar engine install --kind comfyui`. It differentiates (workflows + ecosystem + day-0 models) without displacing sdcpp (light, Vulkan, single-family serving). Default-deny custom nodes; portable-7z lane on Windows; pinned-tag venv lane elsewhere.

## Sources

- GitHub API: repos/comfyanonymous/ComfyUI (+releases, tags, license) — 2026-11-14
- docs.comfy.org: system_requirements, manual_install, api-examples, comms_routes (Server Routes), agent-tools/mcp, llms.txt — fetched 2026-11-14
- Upstream source: comfy/cli_args.py (master), LICENSE (GPL-3.0)
- SwarmUI repo stats (GitHub API, 2026-11-14)
- Local: CHANGELOG 0.19.0-era diffusion lane notes; docs/research/2026-10-31-wan22-s2v-blueprint.md (Comfy-Org repackaged component sourcing)
