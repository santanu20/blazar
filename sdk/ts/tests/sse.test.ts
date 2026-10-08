import test from "node:test";
import assert from "node:assert/strict";
import { SseParser } from "../src/index.ts";

function drain(p: SseParser): string[] {
  const out: string[] = [];
  for (;;) {
    const ev = p.nextEvent();
    if (ev === null) return out;
    out.push(ev.data);
  }
}

test("lf-separated events parse in order", () => {
  const p = new SseParser();
  p.feed('data: {"a":1}\n\ndata: {"a":2}\n\n');
  assert.deepEqual(drain(p), ['{"a":1}', '{"a":2}']);
});

test("crlf line endings accepted", () => {
  const p = new SseParser();
  p.feed('data: {"a":1}\r\n\r\ndata: {"a":2}\r\n\r\n');
  assert.deepEqual(drain(p), ['{"a":1}', '{"a":2}']);
});

test("lone-cr line endings accepted", () => {
  const p = new SseParser();
  p.feed('data: {"a":1}\r\rdata: {"a":2}\r');
  assert.deepEqual(drain(p), ['{"a":1}']);
});

test("comments and keep-alives skipped", () => {
  const p = new SseParser();
  p.feed(": ping\n\ndata: {\"a\":1}\n\n: ping\n\n");
  assert.deepEqual(drain(p), ['{"a":1}']);
});

test("chunks splitting a line anywhere still parse once", () => {
  const p = new SseParser();
  p.feed('data: {"a"');
  assert.deepEqual(drain(p), []);
  p.feed(':1}\n');
  assert.deepEqual(drain(p), ['{"a":1}']);
});

test("chunk boundary between cr and lf holds the line until lf arrives", () => {
  const p = new SseParser();
  p.feed('data: {"a":1}\r');
  assert.deepEqual(drain(p), []); // CR is the last byte: cannot know yet
  p.feed("\ndata: {\"a\":2}\n\n");
  assert.deepEqual(drain(p), ['{"a":1}', '{"a":2}']);
});

test("leading space after colon is stripped once", () => {
  const p = new SseParser();
  p.feed('data:  spaced\n\n');
  assert.deepEqual(drain(p), [" spaced"]);
});
