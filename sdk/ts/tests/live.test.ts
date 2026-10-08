import test from "node:test";
import assert from "node:assert/strict";
import { Client, BlazarError } from "../src/index.ts";

// Live tests: run against a scratch daemon started from the
// feat/error-code-catalog worktree build (the blazar_code assertions
// need the catalog, which main does not carry yet). Start the daemon,
// then: BLAZAR_TEST_URL=http://127.0.0.1:11613 npm run test:live
const base = process.env.BLAZAR_TEST_URL;
const model = process.env.BLAZAR_TEST_MODEL ?? "qwen3-0-6b";
const skip = !base;

test("models endpoint lists the pulled model", { skip }, async () => {
  const c = new Client(base);
  const r = await c.models();
  assert.ok(Array.isArray(r.data), "models.data is an array");
});

test("chat one-shot returns text and usage", { skip }, async () => {
  const c = new Client(base);
  const r = await c.chat(model, [{ role: "user", content: "Reply with exactly: OK" }], {
    temperature: 0,
    max_tokens: 16,
  });
  const text = r.choices[0]?.message?.content ?? "";
  assert.match(text.trim(), /^OK\b/);
  assert.ok((r.usage?.total_tokens ?? 0) > 0);
});

test("chatStream reassembles the same answer", { skip }, async () => {
  const c = new Client(base);
  let text = "";
  for await (const delta of c.chatStream(model, [
    { role: "user", content: "Reply with exactly: OK" },
  ], { temperature: 0, max_tokens: 16 })) {
    const choices = delta.choices as Array<{ delta?: { content?: string } }> | undefined;
    text += choices?.[0]?.delta?.content ?? "";
  }
  assert.match(text.trim(), /^OK\b/);
});

test("ps lists the spawned child", { skip }, async () => {
  const c = new Client(base);
  const rows = await c.ps();
  const family = model.split("-").slice(0, 2).join("-");
  assert.ok(rows.some((r) => String(r.model).includes(family)));
});

test("unknown model carries blazar_code MODEL_NOT_FOUND", { skip }, async () => {
  const c = new Client(base);
  await assert.rejects(
    () => c.chat("no-such-model-xyz", [{ role: "user", content: "hi" }]),
    (e: unknown) => {
      assert.ok(e instanceof BlazarError);
      assert.equal(e.status, 404);
      assert.equal(e.blazarCode, "MODEL_NOT_FOUND");
      return true;
    },
  );
});

test("explain returns a decision card", { skip }, async () => {
  const c = new Client(base);
  const card = await c.explain(model);
  const m = card.model as { name?: string } | undefined;
  assert.equal(typeof m?.name, "string");
});
