# registry.ollama.ai pull protocol — implementation spec

Status: spec (not yet implemented) · Verified live against
`registry.ollama.ai` on 2026-09-10 (ollama v0.33.x era registry) ·
Owner: pull/gateway lane

Goal: `pallama pull qwen3:0.6b` resolves against the ollama registry with
the same download discipline as the HuggingFace lane (`crates/pallama-runtime/src/hf.rs`):
allowlisted redirects, sha256-verified blobs, resumable part-files, honest
errors. Closes the ecosystem-gravity gap (audit G4): users arrive with
ollama shortnames muscle memory.

Non-goals: push, private-namespace auth (OAuth flow), `tags/list` discovery
(the registry returns 404 for it — discovery stays on HF search), importing
ollama's blob/manifest disk layout.

---

## 1. Verified wire protocol (live probes, 2026-09-10)

### 1.1 Name resolution

`types.DefaultName` semantics (ollama/ollama `types/model/name.go`):
`qwen3` → `registry.ollama.ai/library/qwen3:latest`; explicit
`registry.ollama.ai/user/foo:tag` respected verbatim. A `@sha256:...`
digest suffix pins immutable content.

| Step | Request | Result (verified) |
|------|---------|-------------------|
| Manifest | `GET /v2/{ns}/{model}/manifests/{tag}` | 200; body is Docker distribution manifest v2 (`schemaVersion: 2`, `mediaType: application/vnd.docker.distribution.manifest.v2+json`) |
| Blob | `GET /v2/{ns}/{model}/blobs/sha256:{digest}` | **307** → Cloudflare R2 presigned URL (`*.r2.cloudflarestorage.com`, `X-Amz-Expires=86400`); Range requests honored on the redirected target |
| Missing repo | any of the above | 404, `text/plain` body "404 page not found" |
| Tags list | `GET /v2/{ns}/{model}/tags/list` | 404 — not offered |

Quirks a client must tolerate:

- The manifest response `content-type` is `text/plain; charset=utf-8` even
  though the body is the Docker-v2 manifest JSON. Do NOT switch on the
  header; send `Accept: application/vnd.docker.distribution.manifest.v2+json`
  and parse the body.
- No `docker-content-digest` response header observed; verify the manifest
  by hashing the body when a `@sha256:` pin is present.

### 1.2 Manifest shape (captured example: `library/qwen3:0.6b`)

```json
{"schemaVersion":2,
 "mediaType":"application/vnd.docker.distribution.manifest.v2+json",
 "config":{"mediaType":"application/vnd.docker.container.image.v1+json",
           "digest":"sha256:b0830f…","size":490},
 "layers":[
  {"mediaType":"application/vnd.ollama.image.model",    "digest":"sha256:7f4030…","size":522640096},
  {"mediaType":"application/vnd.ollama.image.template", "digest":"sha256:ae370d…","size":1723},
  {"mediaType":"application/vnd.ollama.image.license",  "digest":"sha256:d18a5c…","size":11338},
  {"mediaType":"application/vnd.ollama.image.params",   "digest":"sha256:cff3f3…","size":120}]}
```

Layer semantics (ollama source `server/images.go:63-82`, DeepWiki cross-check):

| Media type | Content | Pallama mapping |
|---|---|---|
| `…image.model` | **raw GGUF** — verified: first bytes of the blob are `GGUF` magic v3 | weights file, existing `models/` layout |
| `…image.projector` | mmproj GGUF | `mmproj_path` pairing (vision lane) |
| `…image.adapter` | LoRA adapter | `loras` list |
| `…image.template` | Go-template prompt template | stored sidecar; pallama passes `--jinja` and reads the GGUF's own chat template — template layer is informational unless GGUF lacks one |
| `…image.params` | JSON sampler/stop defaults, e.g. `{"temperature":0.6,"top_k":20,"top_p":0.95,"stop":[…]}` | map to `sampler_defaults` (existing knob) + `stop` |
| `…image.license` | license text | sidecar file `<model>-<tag>.LICENSE` next to weights (display in `pallama show`) |
| config layer | `model_format`/`model_family`/`model_type`("751.63M" human string)/`file_type`("Q4_K_M") + `rootfs.diff_ids` | quant + family metadata for `pallama list` display; `diff_ids` cross-check layer digests |

GGUF health lint (`gguf_health_warning`) runs on the downloaded model
layer exactly as for HF pulls; quant string for `fit_rows` comes from
config `file_type` (map ollama names → pallama suffixes; identical
vocabulary: `Q4_K_M`, `Q8_0`, …).

## 2. Client flow

```
parse target (registry.ollama.ai default, library ns, :latest tag)
  ├─ host != registry.ollama.ai → existing HF path (unchanged)
  ├─ GET manifest (Accept docker-v2)          404 → "no such model" error
  ├─ validate: schemaVersion==2, ≥1 model layer, digest syntax
  ├─ for each wanted layer (model, projector, adapter):
  │    GET blob → 307 → allowlisted signed URL → resume-aware download
  │    (.part file, Range append — reuse hf.rs machinery)
  │    stream-hash sha256 → mismatch = delete partial + named error
  ├─ params/template/license/config → small-blob fetch, sidecars
  ├─ GGUF lint + Store row (bytes, quant, family) + mmproj pairing
  └─ progress events (existing pull event channel, per-layer percent)
```

Deduplication decision: tags are mutable manifests; two tags may share a
model-layer digest. Keep pallama's flat file layout and name files
`{ns}-{model}-{tag}.gguf`; do NOT build an ollama-style content-addressed
blob store (YAGNI — disk dedupe via hardlinks can come later behind the
same Store row, invisible to the profile layer).

## 3. Security invariants (mirror hf.rs, same tests)

1. **Token discipline:** an optional registry token attaches ONLY to
   first-party `registry.ollama.ai` requests; NEVER to the R2 redirect
   target (the presigned URL is already authorized — forwarding a token
   to a third-party host is exfiltration). `is_allowed_download_host`
   gains `registry.ollama.ai` (manifest) and `*.r2.cloudflarestorage.com`
   (blob redirects); any other redirect host = block + named error,
   identical to the HF `cdn-lfs` handling.
2. **Hash-verify everything:** every blob's sha256 recomputed during
   download; manifest pinned by `@sha256:` when present (hash the body).
   Mismatch → delete partial → error naming digest + layer media type.
3. **Range-resume only on same-digest .part files** (existing `.part`
   scheme; verify size-then-hash-on-completion).
4. **No exec/eval of template layer** — stored verbatim as data.

## 4. Integration surface

| Surface | Change |
|---|---|
| `parse_pull_target` (hf.rs) | recognize `registry.ollama.ai/…` host + bare-namespace shortnames when HF resolution is not requested (`ollama:` explicit prefix optional; bare shortname stays HF-first unless it contains `:` — `qwen3:0.6b` has no HF analogue shape, route to ollama registry) |
| pull pipeline | new `mod registry` in pallama-runtime beside `hf.rs`, sharing the download core (extract `download_core.rs` from hf.rs — resume/hash/allowlist already 90% generic) |
| store | ModelRow gains nothing new; quant/family from config layer |
| sampler_defaults | `…image.params` seeds per-model sampler defaults + stop sequences |
| events | existing pull-progress channel; one event per layer |
| CLI | `pallama pull qwen3:0.6b` — zero new flags; `--registry hf|ollama` explicit override optional |
| docs/README | pull examples both registries; shortname table |

Config: `PALLAMA_REGISTRY_TOKEN` env (optional, default none — the public
`library/*` needs no auth; private namespaces are the user's signal to
authenticate).

## 5. Test plan

- Wire-mock unit: manifest 200/404/redirect-loop; text/plain content-type
  quirk tolerated; layer set without model layer → named error.
- Security: token never appears on redirect-host request (assert on mock
  capture, mirrors `integration__token_only_on_first_party_host`);
  non-allowlisted redirect host blocked (`integration__redirect_to_non_allowlisted_host`
  pattern); sha mismatch deletes partial.
- Resume: interrupted blob resumes via Range on the SAME digest.
- Golden: captured `qwen3:0.6b` manifest JSON as fixture; layer mapping
  table asserted (model→weights, projector→mmproj, params→sampler).
- Live smoke (gated, non-CI): pull `qwen3:0.6b` (522 MiB) end-to-end,
  spawn, one generation.

## 6. Open questions

- Private namespaces auth: ollama routes browser OAuth through
  ollama.com; API-key flow undocumented — defer until requested.
- Manifest tag mutability: re-pull semantics = replace row + weights if
  digest changed (same as HF ref move); consider `@sha256:` pin display
  in `pallama list`.
- Whether ollama moves to OCI 1.1 artifact media types: the Docker-v2
  envelope with `…ollama.image.*` layers is current as of 2026-09-10;
  pin parsing to the envelope, treat unknown layer media types as
  skip-with-warning (forward-compatible).
