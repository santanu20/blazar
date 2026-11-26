# Blazar launch agent presets, dry-run and restore (E3 receipt)

Date: 2026-11-26. Question: does `blazar launch` close the gap to
ollama's per-agent behaviour - preset env knowledge for the common
agent CLIs, a rendered dry-run with secrets redacted, a replayable
last launch, and a self-test that fails loud when the table rots?

## Environment

- binary: ~/Projects/blazar-sota/target/release/blazar (63,850,200 bytes, release)
- host: i7-14650HX (16 cores), RTX 4070 8188 MiB, Linux
- sandbox: isolated config/data dirs, hardlinked engines, model
  qwen2.5-0.5b-instruct (GGUF Q4_K_M, 491 MiB) in the copied store
- preset knowledge source: ollama's cmd/launch integrations (fetched
  2026-11-26); the env rows below are rendered live from the table

## Preset table

| preset | aliases | protocol | model arg |
|---|---|---|---|
| claude | claude-code, claudecode | anthropic | --model |
| codex | | openai | --model |
| opencode | | openai | (none - reads its config) |
| cline | claude-dev | openai | (none) |
| aider | aider-chat | openai | --model |
| dsh | | ollama | (none) |

Any other word launches generically with exactly the four historic
base-URL rows - byte-compatible with the pre-E3 behavior.

## Self-test and table

`launch --self-test` exit 0:

```
launch presets: 6 healthy
```

`launch --list` exit 0 prints every row with aliases
(first rows shown):

```
launch presets (agent CLIs with per-CLI env knowledge):
  claude     [claude-code, claudecode]  Claude Code (Anthropic protocol; deferred MCP, provider pinned here)
  codex       OpenAI Codex CLI (OpenAI protocol)
  opencode    opencode agent CLI (OpenAI protocol; reads its own config)
```

## Dry-run

Claude arm (`launch --dry-run --warm qwen2.5-0.5b-instruct --key <name> claude
--verbose`) exit 0: argv `claude --model qwen2.5-0.5b-instruct
--verbose`, env block of 22 rows - the four base
rows, 15 Claude preset rows (provider pinned to
this launch, deferred MCP via ENABLE_CLAUDEAI_MCP_SERVERS=false,
cloud-provider and telemetry switches off, model routing rows):

```
OPENAI_BASE_URL=http://127.0.0.1:47583
OPENAI_API_BASE=http://127.0.0.1:47583/v1
ANTHROPIC_BASE_URL=http://127.0.0.1:47583
OLLAMA_HOST=127.0.0.1:47583
CLAUDE_CODE_PROVIDER_MANAGED_BY_HOST=1
ENABLE_CLAUDEAI_MCP_SERVERS=false
CLAUDE_CODE_USE_BEDROCK=
CLAUDE_CODE_USE_VERTEX=
CLAUDE_CODE_USE_FOUNDRY=
CLAUDE_CODE_USE_ANTHROPIC_AWS=
CLAUDE_CODE_USE_ANTHROPIC_GOOGLE_CLOUD=
CLAUDE_CODE_USE_MANTLE=
CLAUDE_CODE_USE_GATEWAY=
CLAUDE_CODE_ATTRIBUTION_HEADER=0
CLAUDE_CODE_TOTAL_TOKENS_REMINDER=<redacted>
DISABLE_ERROR_REPORTING=1
DISABLE_FEEDBACK_COMMAND=1
CLAUDE_CODE_DISABLE_FEEDBACK_SURVEY=1
ANTHROPIC_MODEL=qwen2.5-0.5b-instruct
ANTHROPIC_SMALL_FAST_MODEL=qwen2.5-0.5b-instruct
OPENAI_API_KEY=<redacted>
ANTHROPIC_API_KEY=<redacted>
```

Secrets render as `<redacted>` (3 rows here: the two
key rows plus one pre-fix false positive, explained below). The
capture - CLAUDE_CODE_TOTAL_TOKENS_REMINDER=<redacted> - shows the pre-fix renderer over-redacting
a non-secret switch (its value is the literal `off`); this live run is
what caught it. The renderer now redacts by name suffix (_KEY/_TOKEN)
and secret-name substrings only, pinned by
`unit__render_env__redacts_key_rows_only`. The state file records
the key NAME, never the secret.

Generic arm (`launch --dry-run some-cli --flag`, daemon down) exit
0: the four historic rows and the no-preset note;
a dry-run without --warm never needs the daemon.

Codex arm (`launch --dry-run --warm qwen2.5-0.5b-instruct codex`) exit
0: `codex --model qwen2.5-0.5b-instruct` argv, base rows only -
per-CLI knowledge stays per-CLI (no Claude hygiene rows leak).

## Real launch, state, restore

`launch --warm qwen2.5-0.5b-instruct /bin/echo hello-launch` exit
0: warm-up loads the model, the child execs with
the preset env, stdout carries the child's own words:

```
pre-warming qwen2.5-0.5b-instruct ...
launching /bin/echo against http://127.0.0.1:47583
hello-launch
```

State file written after the exec (key name only, no secret):

```
{
 "program": "/bin/echo",
 "model": "qwen2.5-0.5b-instruct",
 "key_name": null,
 "argv_tail": [
  "hello-launch"
 ]
}
```

`launch --restore` exit 0 replays the selection from
the state file alone - warm-up, exec, child output:

```
pre-warming qwen2.5-0.5b-instruct ...
launching /bin/echo against http://127.0.0.1:47583
hello-launch
```

Restore guards (exit 1 before any daemon contact):

```
blazar: --restore cannot be combined with a CLI word or --warm (the restored launch carries its own selection)
blazar: --restore cannot be combined with a CLI word or --warm (the restored launch carries its own selection)
blazar: no launch to restore: ~/.config/blazar/launch-state.json: No such file or directory (os error 2)
```

The refusals mirror ollama's rule - a restored launch carries its own
selection, so combining --restore with a CLI word or --warm refuses;
a missing state file is a loud error naming the path.

## Product bug found and fixed by this validation

The pre-E3 warm-up POST ignored `--key`, so a keyed daemon answered
the warm-up with 401. Root cause: the launch key was resolved lazily
after the warm request had already gone out without auth. Fix: key
resolution is eager (before warm, and even in dry-run so an unknown
key name fails loud without side effects) and the warm POST carries
the bearer token when a key is resolved. The invariant is documented
at the warm site in the CLI.

## Limits

- The real exec is proven with /bin/echo as the child: no claude or
  codex CLI is installed on this host, and the CLIs that are present
  (opencode) are interactive TUIs unsuited to a scripted arm. The
  argv/env/exit-code contract is what a preset can prove headlessly.
- Preset env knowledge is verified against ollama's launch source,
  not against each CLI's own documentation; the table is data-driven
  and `--self-test` plus the unit battery keep it internally sound.
- One host, one model, one keyless exec arm (keyed rendering is
  covered by the claude dry-run arm and the redaction pin).
- The gemini CLI deliberately has no preset: its model routing needs
  per-key setup that belongs to the roadmap's next item, not a
  guessed env row.

## Artifacts

- this receipt pair (md + json)
- <sandbox>/e3_launch_live.json (all 10 arms, raw)
- <sandbox>/e3_launch_live.py (arm driver)
- unit battery: launch_presets.rs (9 pins) + 2 CLI pins
  (clap surface, restore refusals)
- RED receipt <sandbox>/e3_launch_red.txt (8 todo!() failures before
  the module existed)

