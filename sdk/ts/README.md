# blazar-sdk (TypeScript)

Zero-dependency TypeScript client for the
[blazar](https://github.com/santanu20/blazar) inference control plane.
Node's built-in `fetch` does the transport — nothing to install beyond
this package. Node >= 18.

## Install

```sh
npm install blazar-sdk
```

## Use

```ts
import { Client } from "blazar-sdk";

const blazar = new Client(); // BLAZAR_URL env or http://127.0.0.1:11435

// one-shot
const reply = await blazar.chat("qwen3-0-6b", [
  { role: "user", content: "Reply with exactly: OK" },
]);
console.log(reply.choices[0]?.message?.content);

// streaming
for await (const delta of blazar.chatStream("qwen3-0-6b", [
  { role: "user", content: "Count from one to five." },
])) {
  process.stdout.write(String(delta.choices?.[0]?.delta?.content ?? ""));
}

// operations surfaces
const models = await blazar.models();
const resident = await blazar.ps();
const whyThisEngine = await blazar.explain("qwen3-0-6b");
```

## Errors

Non-2xx responses throw `BlazarError` with `status` (HTTP code) and
`blazarCode` (the stable `blazar_code` string from the gateway error
catalog — see `GET /api/errors`), when present.

## Environment

| Variable | Default | Purpose |
|---|---|---|
| `BLAZAR_URL` | `http://127.0.0.1:11435` | gateway origin |
| `BLAZAR_API_KEY` | none | bearer key for gated daemons |

## Roadmap

v1 covers chat (one-shot + streaming), models, ps, explain, failover,
and MCP server listing. Realtime voice, anthropic batch jobs, and
speech/transcribe live in the python SDK today and will land here as
the surfaces stabilize.
