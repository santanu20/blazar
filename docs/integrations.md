---
layout: doc
title: Integrations
description: Point any OpenAI, Ollama, or Anthropic client at Blazar — and let blazar connect wire Codex, Claude Code, Continue, Cline, and Open WebUI for you.
doc_kind: Documentation
---
## Three API dialects, one endpoint

The gateway listens on `http://127.0.0.1:11435` and speaks the OpenAI,
Ollama, and Anthropic wire protocols natively. Most clients need nothing but
a base-URL change:

| Client family | Point it at | Surface |
|---|---|---|
| **OpenAI SDK** | `http://127.0.0.1:11435/v1` | `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/embeddings`, `/v1/rerank`, `/v1/batches`, `/v1/files`, `/v1/audio/*`, `/v1/images/*`, `/v1/videos/*` |
| **Ollama clients** | `OLLAMA_HOST=http://127.0.0.1:11435` | `/api/chat`, `/api/generate`, `/api/tags`, `/api/ps`, `/api/pull`, `/api/embeddings`, `/api/embed`, `/api/rerank` |
| **Anthropic SDK** | `http://127.0.0.1:11435` | `/v1/messages` |

Blazar runs alongside an existing Ollama install, and can later serve on
`11434` as a port-level replacement.

The full endpoint inventory — jobs, requests, conversations, files, batches,
audio, images, video, capacity, and diagnostics — is specified in the
[API specification](/4.API_SPEC.html).

## `blazar connect` — wired for you

Agent CLIs and UIs have their own config files. `blazar connect` writes the
right entries for each supported client:

```sh
blazar connect              # print the client-integration plan
blazar connect --write      # apply with backup; rolls back if the test request fails
```

| Client | Integration point |
|---|---|
| **Codex CLI** | `~/.codex/config.toml` |
| **Claude Code** | `~/.claude/settings.json` |
| **Continue** | `~/.continue/config.yaml` |
| **Cline** | VS Code UI |
| **Open WebUI** | env at startup |

`--write` never leaves a client broken: the existing config is backed up,
the new entry is applied, a test request is fired, and a failure rolls the
change back.

## Ollama drop-in

Existing Ollama tooling works after re-pointing:

```sh
OLLAMA_HOST=http://127.0.0.1:11435 ollama list
```

or set `port = 11434` in the Blazar config for a drop-in once the Ollama
daemon is removed — same clients, same habits, multi-engine serving
underneath.
