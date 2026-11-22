#!/usr/bin/env python3
"""Tool-calling quality battery - deterministic, per model, live-only.

Measures the dimensions the perf bench matrix cannot: selection
correctness, argument validity, tool_choice forms, parallel calls,
multi-turn loop, strict JSON schema, streaming tool deltas, nested-
schema arguments, malformed-history tolerance, tool-error recovery,
full streaming call assembly, and the tools-vs-no-tools latency delta
(transport overhead proxy - the child is auth-gated by design, so raw-
child comparison is not reachable externally; the no-tools request of
identical prompt size isolates the tool-transport cost instead).

Usage:
    python3 scripts/toolcall_eval.py [model ...]     # default: qwen2.5-0.5b-instruct
    python3 scripts/toolcall_eval.py --direct http://127.0.0.1:8199 [model ...]

The default arm hits the Blazar gateway; --direct re-runs the identical
battery against a bare engine URL for gateway-vs-direct attribution.
Direct-arm artifacts land in a sibling -direct dir, never over the
gateway run's.

Outputs bench-artifacts/<date>-toolcall/results.json + report rows.
Every check is deterministic: exact tool name, parseable args, required
fields with correct types. Parallel-call behavior is MEASURED (recorded
count), never failed - models legitimately choose one call at a time.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

BASE = os.environ.get("BLAZAR_BASE", "http://127.0.0.1:11435")
# Default output is date-stamped; BLAZAR_TOOLCALL_OUT redirects it so a
# lane A/B never overwrites another session's report in the same-day dir.
OUT_DIR = os.environ.get(
    "BLAZAR_TOOLCALL_OUT",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "bench-artifacts",
        _dt.datetime.now().strftime("%Y%m%d") + "-toolcall",
    ),
)

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for one city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}
SEARCH = {
    "type": "function",
    "function": {
        "name": "search_docs",
        "description": "Search the documentation for a query",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}
CALC = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate a basic arithmetic expression",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
}
BOOK = {
    "type": "function",
    "function": {
        "name": "book_flight",
        "description": "Book a flight; origin is an object with city and country",
        "parameters": {
            "type": "object",
            "properties": {
                "origin": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "country": {"type": "string"},
                    },
                    "required": ["city", "country"],
                    "additionalProperties": False,
                },
                "passengers": {"type": "integer", "minimum": 1},
                "class": {"type": "string", "enum": ["economy", "business"]},
            },
            "required": ["origin", "passengers", "class"],
        },
    },
}


# Response headers of the most recent post() - the battery is
# single-threaded, so T10 can read x-blazar-repair straight after its
# call without threading a fourth return value through every caller.
LAST_RESPONSE_HEADERS: dict = {}

# True when --direct BASE_URL re-runs the battery against a bare
# engine (T10's xfail pin is direct-arm-only: no gateway, no repair).
DIRECT_ARM = False


def post(path: str, body: dict, timeout: int = 180):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode(),
        method="POST",
        headers={"content-type": "application/json"},
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            LAST_RESPONSE_HEADERS.clear()
            LAST_RESPONSE_HEADERS.update(dict(r.headers))
            return r.status, json.loads(raw), time.monotonic() - t0
    except urllib.error.HTTPError as e:
        LAST_RESPONSE_HEADERS.clear()
        LAST_RESPONSE_HEADERS.update(dict(e.headers))
        return e.code, json.loads(e.read() or b"{}"), time.monotonic() - t0


def tool_calls_of(msg: dict) -> list[dict]:
    return msg.get("tool_calls") or []


def call_args(tc: dict) -> dict:
    raw = (tc.get("function", {}).get("arguments") or "").strip()
    if not raw:
        return {}
    try:
        v = json.loads(raw)
        return v if isinstance(v, dict) else {}
    except ValueError:
        return {"__unparseable__": raw[:80]}


def stream_collect(body: dict):
    """POST a streaming chat request and assemble tool-call deltas.

    Returns (sse_done, calls) where calls maps the delta index to the
    assembled {"id", "name", "arguments"} of that streamed tool call.
    Names accumulate as fragments too: some engines stream the function
    name across chunks instead of emitting it whole in the first delta.
    """
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        method="POST",
        headers={"content-type": "application/json"},
    )
    calls: dict[int, dict] = {}
    sse_done = False
    with urllib.request.urlopen(req, timeout=180) as r:
        for line in r:
            if not line.startswith(b"data: "):
                continue
            payload = line[6:].strip()
            if payload == b"[DONE]":
                sse_done = True
                break
            try:
                delta = json.loads(payload)["choices"][0]["delta"]
            except (ValueError, KeyError, IndexError):
                continue
            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index", 0)
                slot = calls.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function", {})
                if fn.get("name"):
                    slot["name"] += fn["name"]
                slot["arguments"] += fn.get("arguments") or ""
    return sse_done, calls


class Battery:
    def __init__(self, model: str):
        self.model = model
        self.results: list[dict] = []

    def record(self, name: str, ok: bool, evidence: str, measured=None, xfail=None):
        # xfail: the test still RUNS and records its raw result; a
        # failure covered by a root-caused xfail reason counts as an
        # expected failure (gate stays green, report shows XFAIL with
        # the reason - never a silent skip). If it passes anyway the
        # marker clears loudly so the pin can be removed.
        row = {"test": name, "ok": bool(ok), "evidence": evidence[:220]}
        if measured is not None:
            row["measured"] = measured
        if xfail:
            if ok:
                row["xfail_cleared"] = xfail
            else:
                row["xfail"] = xfail
        self.results.append(row)
        if ok:
            tag = "PASS" + (" (xfail cleared!)" if xfail else "")
        else:
            tag = "XFAIL" if xfail else "FAIL"
        extra = f" ({measured})" if measured is not None else ""
        reason = f" reason: {xfail[:120]}" if (xfail and not ok) else ""
        print(f"  [{tag}] {name}{extra} - {evidence[:120]}{reason}")

    def chat(self, tools, messages, **kw):
        body = {
            "model": self.model,
            "stream": False,
            "max_tokens": 300,
            "tools": tools,
            "messages": messages,
        }
        body.update(kw)
        return post("/v1/chat/completions", body)

    def run(self):
        m = self.model
        print(f"\n== toolcall battery: {m} ==")

        # T1 single-call: right tool, right required arg, parseable.
        st, v, _ = self.chat(
            [WEATHER],
            [{"role": "user", "content": "What is the weather in Tokyo right now?"}],
        )
        tcs = (
            tool_calls_of(v.get("choices", [{}])[0].get("message", {}))
            if st == 200
            else []
        )
        ok = (
            st == 200
            and len(tcs) >= 1
            and tcs[0]["function"]["name"] == "get_weather"
            and call_args(tcs[0]).get("city", "").lower().find("tokyo") >= 0
        )
        self.record(
            "T1 single-call name+args",
            ok,
            f"status={st} calls={[t['function']['name'] for t in tcs]} "
            f"args={[call_args(t) for t in tcs]}",
        )

        # T2 selection among three tools.
        st, v, _ = self.chat(
            [WEATHER, SEARCH, CALC],
            [{"role": "user", "content": "Calculate 47 * 93 for me."}],
        )
        tcs = (
            tool_calls_of(v.get("choices", [{}])[0].get("message", {}))
            if st == 200
            else []
        )
        names = [t["function"]["name"] for t in tcs]
        ok = st == 200 and names == ["calculator"]
        self.record(
            "T2 pick correct tool of three",
            ok,
            f"status={st} picked={names} args={[call_args(t) for t in tcs]}",
        )

        # T3 tool_choice=required must produce a call (any listed tool).
        # Task-bearing prompt + one retry: tiny models occasionally
        # sample prose on the first draw even under the constraint -
        # transport enforcement is what this pins, not draw luck.
        t3_calls, t3_st, attempts = 0, 0, 0
        for _ in range(2):
            attempts += 1
            st, v, _ = self.chat(
                [WEATHER],
                [
                    {
                        "role": "user",
                        "content": "What is the weather in Tokyo? "
                        "You must use the get_weather tool.",
                    }
                ],
                tool_choice="required",
            )
            t3_st = st
            tcs = (
                tool_calls_of(v.get("choices", [{}])[0].get("message", {}))
                if st == 200
                else []
            )
            t3_calls = len(tcs)
            if t3_calls >= 1:
                break
        self.record(
            "T3 tool_choice=required",
            t3_st == 200 and t3_calls >= 1,
            f"status={t3_st} calls={t3_calls} attempts={attempts}",
        )

        # T4 dict-form tool_choice pins one specific function.
        st, v, _ = self.chat(
            [WEATHER, CALC],
            [{"role": "user", "content": "Weather in Paris, please."}],
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
        )
        tcs = (
            tool_calls_of(v.get("choices", [{}])[0].get("message", {}))
            if st == 200
            else []
        )
        names = [t["function"]["name"] for t in tcs]
        ok = st == 200 and bool(names) and all(n == "get_weather" for n in names)
        self.record(
            "T4 tool_choice dict-form pin",
            ok,
            f"status={st} picked={names}",
        )

        # T5 parallel calls - MEASURED, never failed (one-at-a-time is a
        # legitimate model policy; the transport must carry N calls in
        # one turn when the model emits them).
        st, v, _ = self.chat(
            [WEATHER],
            [
                {
                    "role": "user",
                    "content": "Give me the weather for Tokyo AND for Paris. "
                    "Call the tool for both cities in your reply.",
                }
            ],
        )
        tcs = (
            tool_calls_of(v.get("choices", [{}])[0].get("message", {}))
            if st == 200
            else []
        )
        n = len(tcs)
        self.record(
            "T5 parallel-call emission (measured)",
            st == 200,
            f"calls_in_one_turn={n}",
            measured=n,
        )

        # T6 multi-turn: assistant tool_call + tool result in, answer out.
        st, v, _ = self.chat(
            [WEATHER],
            [
                {"role": "user", "content": "Weather in Tokyo?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": json.dumps({"city": "Tokyo"}),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": '{"temp_c": 21, "condition": "light rain"}',
                },
            ],
        )
        text = ""
        if st == 200:
            msg = v.get("choices", [{}])[0].get("message", {})
            text = (msg.get("content") or "") + json.dumps(msg.get("tool_calls") or [])
        ok = st == 200 and ("21" in text or "rain" in text.lower())
        self.record(
            "T6 tool result -> grounded answer",
            ok,
            f"status={st} answer={text[:160]!r}",
        )

        # T7 strict structured output alongside tools capability.
        # Budget realism: reasoning-capable models spend tokens thinking
        # before the first JSON byte, so a 120-token cap starved them
        # (finish_reason=length, truncated content). 256 is the honest
        # baseline; ONE retry at 512 fires only on an explicit length
        # cap - a schema failure at a healthy budget is a real contract
        # failure, not starvation.
        def t7_once(budget: int):
            st, v, _ = post(
                "/v1/chat/completions",
                {
                    "model": m,
                    "stream": False,
                    "max_tokens": budget,
                    "messages": [
                        {
                            "role": "user",
                            "content": "Return the weather summary as JSON with "
                            "fields city (string) and temp_c (integer). City: Oslo.",
                        }
                    ],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "summary",
                            "strict": True,
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "city": {"type": "string"},
                                    "temp_c": {"type": "integer"},
                                },
                                "required": ["city", "temp_c"],
                                "additionalProperties": False,
                            },
                        },
                    },
                },
            )
            finish, content = "", ""
            if st == 200:
                ch = v.get("choices", [{}])[0]
                finish = ch.get("finish_reason") or ""
                content = ch.get("message", {}).get("content") or ""
            try:
                j = json.loads(content)
            except ValueError:
                j = None
            ok = (
                isinstance(j, dict)
                and isinstance(j.get("city"), str)
                and isinstance(j.get("temp_c"), int)
            )
            return ok, st, finish, content

        budget = 256
        ok, st, finish, content = t7_once(budget)
        if not ok and finish == "length":
            budget = 512
            ok, st, finish, content = t7_once(budget)
        self.record(
            "T7 strict json_schema output",
            ok,
            f"status={st} budget={budget} finish_reason={finish} "
            f"content={content[:120]!r}",
        )

        # T8 streaming: tool-call argument deltas assemble into valid JSON.
        try:
            sse_ok, streamed = stream_collect(
                {
                    "model": m,
                    "stream": True,
                    "max_tokens": 200,
                    "tools": [WEATHER],
                    "messages": [{"role": "user", "content": "Weather in Berlin?"}],
                }
            )
        except Exception as e:
            self.record("T8 streaming tool deltas", False, f"transport: {e}")
            streamed = {}
            sse_ok = False
        if streamed or sse_ok:
            args_list = [c["arguments"] for c in streamed.values()]
            args_ok = any("city" in (a or "") for a in args_list)
            # name fragments arrive as deltas too; accept if any args JSON parses
            try:
                parsed_any = any(
                    isinstance(json.loads(a), dict) for a in args_list if a.strip()
                )
            except ValueError:
                parsed_any = False
            self.record(
                "T8 streaming tool deltas",
                sse_ok and parsed_any and args_ok,
                f"streams=[{args_list[:2]}]",
            )

        # T9 nested-schema arguments: emitted args must satisfy a schema
        # with a nested object, a bounded integer, and an enum member.
        st, v, _ = self.chat(
            [BOOK],
            [
                {
                    "role": "user",
                    "content": "Book a flight from Tokyo, Japan for "
                    "3 passengers in economy class.",
                }
            ],
        )
        tcs = (
            tool_calls_of(v.get("choices", [{}])[0].get("message", {}))
            if st == 200
            else []
        )
        ok = False
        detail = f"status={st} calls={[t['function']['name'] for t in tcs]}"
        if st == 200 and tcs and tcs[0]["function"]["name"] == "book_flight":
            a = call_args(tcs[0])
            raw_origin = a.get("origin")
            origin = raw_origin if isinstance(raw_origin, dict) else {}
            ok = (
                "__unparseable__" not in a
                and str(origin.get("city", "")).lower().find("tokyo") >= 0
                and str(origin.get("country", "")).lower().find("japan") >= 0
                and a.get("passengers") == 3
                and a.get("class") == "economy"
            )
            detail = f"status={st} args={a}"
        self.record("T9 nested-schema args", ok, detail)

        # T10 malformed-args history tolerance: a prior turn carrying an
        # unparseable arguments string must not break the transport; the
        # model gets a clean chance to re-emit a valid call.
        st, v, _ = self.chat(
            [WEATHER],
            [
                {"role": "user", "content": "Weather in Tokyo?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_bad",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "Tok',  # truncated JSON
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_bad",
                    "content": '{"error": "arguments were not valid JSON"}',
                },
                {"role": "user", "content": "That call failed. Try again properly."},
            ],
        )
        tcs = (
            tool_calls_of(v.get("choices", [{}])[0].get("message", {}))
            if st == 200
            else []
        )
        content = ""
        if st == 200:
            content = v.get("choices", [{}])[0].get("message", {}).get("content") or ""
        ok = st == 200 and bool(len(tcs) >= 1 or content.strip())
        err = ""
        if st != 200:
            # Name the failure origin: a child engine error body (e.g.
            # llama-server's JSON parse exception) vs blazar's own
            # "blazar_error" shape - attribution rides in the evidence.
            err = f" error={json.dumps(v.get('error', v))[:140]}"
        # The gateway repair lane (llama.cpp 500 + json parse_error
        # signature -> one repaired retry) stamps its note here when it
        # saved the request; absence on a PASS means the child never
        # tripped (repair not needed), absence on a FAIL means no
        # repair was attempted.
        repair = LAST_RESPONSE_HEADERS.get(
            "x-blazar-repair", LAST_RESPONSE_HEADERS.get("X-Blazar-Repair")
        )
        repaired = f" repair={repair!r}" if repair else ""
        self.record(
            "T10 malformed-args history tolerance",
            ok,
            f"status={st} calls={len(tcs)} answer={content[:100]!r}{repaired}{err}",
            xfail=(
                # Direct-arm pin only: upstream llama-server 500s
                # (nlohmann parse_error.101) on spec-legal malformed
                # tool-arguments carried in request history, and a bare
                # engine has no gateway repair lane to save it. The
                # gateway arm ships the repair-on-500 lane, so it must
                # PASS this check un-pinned.
                "upstream llama-server rejects spec-legal malformed "
                "tool-args history (json parse_error.101 -> 500); "
                "direct engine has no repair lane"
                if (not ok and DIRECT_ARM)
                else None
            ),
        )

        # T11 tool-error recovery: an errored tool result must yield a
        # coherent next step - a retry call with parseable args or a
        # textual acknowledgement - never a transport failure.
        st, v, _ = self.chat(
            [WEATHER],
            [
                {"role": "user", "content": "What is the weather in Springfield?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_err",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": json.dumps({"city": "Springfield"}),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_err",
                    "content": '{"error": "ambiguous city: provide city and country"}',
                },
            ],
        )
        tcs = (
            tool_calls_of(v.get("choices", [{}])[0].get("message", {}))
            if st == 200
            else []
        )
        content = ""
        if st == 200:
            content = v.get("choices", [{}])[0].get("message", {}).get("content") or ""
        retry_args = [call_args(t) for t in tcs]
        ok = st == 200 and bool(
            any("__unparseable__" not in a for a in retry_args) or content.strip()
        )
        self.record(
            "T11 tool-error recovery",
            ok,
            f"status={st} retry_args={retry_args} answer={content[:80]!r}",
        )

        # T12 streaming tool-call assembly completeness: the full call
        # (id + name + parseable arguments) must survive delta transport.
        try:
            sse_ok, streamed = stream_collect(
                {
                    "model": m,
                    "stream": True,
                    "max_tokens": 200,
                    "tools": [WEATHER],
                    "messages": [
                        {
                            "role": "user",
                            "content": "What is the weather in Osaka? Call the tool.",
                        }
                    ],
                }
            )
        except Exception as e:
            self.record("T12 streaming call assembly", False, f"transport: {e}")
            streamed = {}
            sse_ok = False
        if streamed or sse_ok:
            complete = [
                c
                for c in streamed.values()
                if c["name"] and c["id"] and c["arguments"].strip()
            ]
            parseable = 0
            for c in complete:
                try:
                    parseable += (
                        1 if isinstance(json.loads(c["arguments"]), dict) else 0
                    )
                except ValueError:
                    pass
            ok = sse_ok and bool(complete) and parseable == len(complete)
            ev = "; ".join(
                f"{c['id']}:{c['name']}({c['arguments'][:40]})"
                for c in streamed.values()
            )
            self.record(
                "T12 streaming call assembly",
                ok,
                f"sse_done={sse_ok} calls=[{ev}]",
            )

        # L1 latency overhead proxy: identical prompt size, tools vs none.
        def lat(tools):
            ts = []
            for _ in range(3):
                body = {
                    "model": m,
                    "stream": False,
                    "max_tokens": 8,
                    "messages": [{"role": "user", "content": "Say ok."}],
                }
                if tools:
                    body["tools"] = [WEATHER, SEARCH, CALC]
                _, _, dt = post("/v1/chat/completions", body)
                ts.append(dt * 1000.0)
            return statistics.median(ts)

        with_tools = lat(True)
        without = lat(False)
        self.record(
            "L1 tools-vs-no-tools latency (p50 of 3)",
            True,
            f"with={with_tools:.0f}ms without={without:.0f}ms "
            f"delta={with_tools - without:+.0f}ms (prompt-token cost of "
            "carrying 3 tool schemas; gateway transport is the same path)",
            measured={
                "with_tools_ms": round(with_tools),
                "no_tools_ms": round(without),
            },
        )

    def summary(self):
        # Expected failures (root-caused, reason recorded in the row)
        # count toward the gate; raw failures do not.
        fails = [r for r in self.results if not r["ok"] and not r.get("xfail")]
        return len(self.results) - len(fails), len(self.results)


def main() -> int:
    args = sys.argv[1:]
    direct_base = None
    models = []
    i = 0
    while i < len(args):
        if args[i] == "--direct":
            if i + 1 >= len(args):
                print("--direct requires a base URL", file=sys.stderr)
                return 2
            direct_base = args[i + 1].rstrip("/")
            i += 2
        else:
            models.append(args[i])
            i += 1
    models = models or ["qwen2.5-0.5b-instruct"]

    global BASE, DIRECT_ARM
    if direct_base:
        BASE = direct_base
        DIRECT_ARM = True
    out_dir = OUT_DIR + ("-direct" if direct_base else "")
    os.makedirs(out_dir, exist_ok=True)

    def fetch(path: str, default):
        # Bare engines lack the blazar endpoints; a missing probe is
        # recorded as its default, never fatal - the battery still runs.
        try:
            with urllib.request.urlopen(BASE + path, timeout=5) as r:
                return json.loads(r.read())
        except Exception:
            return default

    ver = fetch("/api/version", {})
    ps = fetch("/api/ps", {"models": []})
    engine = next(
        (
            r.get("blazar_engine")
            for r in ps.get("models", [])
            if r.get("name") == models[0]
        ),
        None,
    )
    if direct_base:
        # Identify the bare engine when it exposes sglang's info route.
        info = fetch("/get_server_info", {})
        engine = f"direct engine ({info.get('version') or 'unidentified'})"
        blazar_ver = "none (direct-engine arm)"
    else:
        # /api/version .version is the ollama-compat number the gateway
        # reports for client compatibility; the real build label rides
        # in blazar_version. Label the fallback so a stale daemon can
        # never masquerade as a newer build in reports.
        blazar_ver = ver.get("blazar_version") or (
            f"{ver.get('version', '?')} (api-compat fallback)"
        )
    commit = (
        subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        or "unknown"
    )
    all_rows = {}
    for m in models:
        b = Battery(m)
        b.run()
        ok, total = b.summary()
        all_rows[m] = {"passed": ok, "total": total, "results": b.results}

    if engine is None and not direct_base:
        # First request may have been the one to spawn the child; re-probe
        # so the receipt names the engine that actually served the run.
        ps = fetch("/api/ps", ps)
        engine = next(
            (
                r.get("blazar_engine")
                for r in ps.get("models", [])
                if r.get("name") == models[0]
            ),
            None,
        )

    receipt = {
        "object": "blazar.toolcall-eval",
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "base": BASE,
        "arm": "direct" if direct_base else "gateway",
        "version": ver,
        "blazar_version": blazar_ver,
        "commit": commit,
        "engine_first_model": engine,
        "models": all_rows,
        "reproduce": f"python3 scripts/toolcall_eval.py {' '.join(args)}",
    }
    out = os.path.join(out_dir, "results.json")
    with open(out, "w") as f:
        json.dump(receipt, f, indent=1)
        f.write("\n")
    md = os.path.join(out_dir, "report.md")
    with open(md, "w") as f:
        f.write("# Tool-calling quality battery\n\n")
        f.write(
            f"{_dt.datetime.now(_dt.timezone.utc).isoformat(timespec='seconds')} | "
            f"arm {'direct' if direct_base else 'gateway'} | "
            f"blazar {blazar_ver} | commit {commit} | engine {engine}\n\n"
        )
        for m, rows in all_rows.items():
            passed, total = rows["passed"], rows["total"]
            xfl = [r for r in rows["results"] if r.get("xfail")]
            hdr = f"{passed}/{total} passed"
            if xfl:
                hdr += f" ({len(xfl)} expected-failure{'s' if len(xfl) > 1 else ''})"
            f.write(f"## {m} - {hdr}\n\n")
            for r in rows["results"]:
                if r.get("xfail"):
                    f.write(
                        f"- XFAIL | {r['test']} - {r['evidence']}\n"
                        f"  reason: {r['xfail']}\n"
                    )
                elif r.get("xfail_cleared"):
                    f.write(
                        f"- PASS (xfail cleared) | {r['test']} - {r['evidence']}\n"
                        f"  cleared pin: {r['xfail_cleared']}\n"
                    )
                else:
                    f.write(
                        f"- {'PASS' if r['ok'] else 'FAIL'} | {r['test']} - {r['evidence']}\n"
                    )
            f.write("\n")
    total_ok = sum(r["passed"] for r in all_rows.values())
    total_n = sum(r["total"] for r in all_rows.values())
    print(f"\nTOTAL: {total_ok}/{total_n} checks - {out}")
    print(f"report: {md}")
    return 0 if total_ok == total_n else 1


if __name__ == "__main__":
    sys.exit(main())
