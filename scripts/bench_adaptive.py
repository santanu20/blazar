#!/usr/bin/env python3
"""Compare Blazar chat with adaptive inference off, auto and deep.

A smoke benchmark, not a broad reasoning leaderboard. Keep model, host, cache
state, concurrency and request set identical when comparing modes.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

CASES: tuple[dict[str, Any], ...] = (
    {
        "id": "arithmetic",
        "messages": [{"role": "user", "content":
            "Calculate 17 * (23 - 8) + 144 / 12. Return only the number."}],
        "expected": "267",
    },
    {
        "id": "table_lookup",
        "messages": [{"role": "user", "content":
            "Oak=12, Pine=19, Birch=16. Return the highest label and value."}],
        "expected": ("pine", "19"),
    },
    {
        "id": "json_schema",
        "messages": [{"role": "user", "content":
            "Extract Ada Lovelace, born in 1815, as JSON."}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "person_record",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "birth_year": {"type": "integer"},
                    },
                    "required": ["name", "birth_year"],
                    "additionalProperties": False,
                },
            },
        },
        "expected": {"name": "Ada Lovelace", "birth_year": 1815},
    },
)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def check(case: dict[str, Any], content: str) -> bool:
    expected = case["expected"]
    if isinstance(expected, str):
        return expected in content
    if isinstance(expected, tuple):
        lowered = content.lower()
        return all(item in lowered for item in expected)
    candidate = content.strip()
    fence = chr(96) * 3
    if candidate.startswith(fence):
        candidate = candidate.strip(chr(96))
        if candidate.lower().startswith("json"):
            candidate = candidate[4:].strip()
    try:
        return json.loads(candidate) == expected
    except json.JSONDecodeError:
        return False


def request_once(endpoint: str, model: str, case: dict[str, Any],
                 mode: str, api_key: str | None, timeout: float) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": case["messages"],
        "stream": False,
        "temperature": 0.2,
    }
    if "response_format" in case:
        body["response_format"] = case["response_format"]
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    if mode != "off":
        headers["X-Blazar-Adaptive"] = mode

    req = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    started = time.perf_counter()
    status = 0
    receipt = None
    response: dict[str, Any] = {}
    error = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as result:
            status = result.status
            receipt = result.headers.get("x-blazar-adaptive-result")
            response = json.loads(result.read())
    except urllib.error.HTTPError as exc:
        status = exc.code
        receipt = exc.headers.get("x-blazar-adaptive-result")
        try:
            response = json.loads(exc.read())
        except (json.JSONDecodeError, OSError):
            response = {}
        error = "HTTP " + str(status)
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        error = str(exc)

    latency = (time.perf_counter() - started) * 1000
    content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
    usage = response.get("usage", {})
    return {
        "case": case["id"],
        "status": status,
        "success": status == 200 and check(case, str(content)),
        "latency_ms": latency,
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
        "adaptive_receipt": receipt,
        "error": error,
    }


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(row["latency_ms"]) for row in rows]
    return {
        "requests": len(rows),
        "task_pass_rate": (
            sum(bool(row["success"]) for row in rows) / len(rows) if rows else 0.0
        ),
        "p50_latency_ms": statistics.median(latencies) if latencies else None,
        "p95_latency_ms": percentile(latencies, 0.95),
        "prompt_tokens": sum(row["prompt_tokens"] for row in rows),
        "completion_tokens": sum(row["completion_tokens"] for row in rows),
        "total_tokens": sum(row["total_tokens"] for row in rows),
        "errors": [row["error"] for row in rows if row["error"]],
        "receipts": sorted(
            {row["adaptive_receipt"] for row in rows if row["adaptive_receipt"]}
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:11435")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--modes", default="off,auto,deep")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.runs < 1 or args.warmup < 0 or args.timeout <= 0:
        parser.error("--runs must be positive; --warmup non-negative; --timeout positive")
    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    if not modes or any(mode not in {"off", "auto", "deep"} for mode in modes):
        parser.error("--modes accepts off, auto, and deep")

    endpoint = args.base_url.rstrip("/") + "/v1/chat/completions"
    report: dict[str, Any] = {
        "endpoint": endpoint,
        "model": args.model,
        "runs_per_case": args.runs,
        "warmup_runs_per_case_mode": args.warmup,
        "cases": [case["id"] for case in CASES],
        "results": {},
        "note": (
            "This smoke suite is not a held-out reasoning benchmark. Keep model, "
            "host, concurrency, cache state and request set fixed for comparisons."
        ),
    }
    for mode in modes:
        rows = []
        for case in CASES:
            for _ in range(args.warmup):
                request_once(endpoint, args.model, case, mode, args.api_key, args.timeout)
            for run_index in range(args.runs):
                result = request_once(
                    endpoint, args.model, case, mode, args.api_key, args.timeout
                )
                rows.append({"run": run_index + 1, **result})
        report["results"][mode] = {"summary": summarise(rows), "samples": rows}

    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
