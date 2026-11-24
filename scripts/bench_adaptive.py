#!/usr/bin/env python3
"""Compare Blazar chat with adaptive inference off, auto and deep.

A smoke benchmark, not a broad reasoning leaderboard. Keep model, host, cache
state, concurrency and request set identical when comparing modes.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

CASES: tuple[dict[str, Any], ...] = (
    {
        "id": "arithmetic_nested",
        "messages": [{"role": "user", "content":
            "Calculate 17 * (23 - 8) + 144 / 12. Return only the number."}],
        "expected": 267,
    },
    {
        "id": "arithmetic_precedence",
        "messages": [{"role": "user", "content":
            "Compute 3 * (42 - 17) + 6^2. Return only the number."}],
        "expected": 111,
    },
    {
        "id": "arithmetic_fraction",
        "messages": [{"role": "user", "content":
            "Evaluate (144 / 12) * (19 - 7) / 5 * 5. Return only the number."}],
        "expected": 144,
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
    if case["id"].startswith("arithmetic_"):
        # Exact numeric match; never accept an explanation that merely
        # contains the expected number among unrelated/wrong calculations.
        match = re.fullmatch(r"\s*([-+]?\d+(?:\.\d+)?)\s*[.!]?\s*", content)
        return match is not None and math.isclose(
            float(match.group(1)), float(expected), rel_tol=1e-9, abs_tol=1e-9
        )
    if isinstance(expected, tuple):
        # Require the correct label/value pair, not merely both tokens
        # appearing somewhere in an unrelated answer.
        lowered = content.lower()
        return re.search(
            r"\bpine\b(?:\s+|\s*[:=|,-]\s*)19\b|"
            r"\b19\b(?:\s+|\s*[:=|,-]\s*)\bpine\b",
            lowered,
        ) is not None
    candidate = content.strip()
    lines = candidate.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        if lines and lines[0].strip().lower() == "json":
            lines = lines[1:]
    try:
        return json.loads("\n".join(lines).strip()) == expected
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
    candidate_receipt = None
    response: dict[str, Any] = {}
    error = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as result:
            status = result.status
            receipt = result.headers.get("x-blazar-adaptive-result")
            candidate_receipt = result.headers.get("x-blazar-adaptive-candidates")
            response = json.loads(result.read())
    except urllib.error.HTTPError as exc:
        status = exc.code
        receipt = exc.headers.get("x-blazar-adaptive-result")
        candidate_receipt = exc.headers.get("x-blazar-adaptive-candidates")
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
        "candidate_receipt": candidate_receipt,
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


def per_case_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["case"]), []).append(row)
    return {
        case: {
            "requests": len(items),
            "pass_rate": (
                sum(bool(item["success"]) for item in items) / len(items)
                if items else 0.0
            ),
            "p50_latency_ms": statistics.median(
                [float(item["latency_ms"]) for item in items]
            ) if items else None,
        }
        for case, items in sorted(grouped.items())
    }


def bootstrap_mean_ci(
    values: list[float], *, seed: int, rounds: int = 2000
) -> tuple[float, float] | None:
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    estimates = [
        statistics.fmean(rng.choices(values, k=len(values)))
        for _ in range(rounds)
    ]
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def paired_comparison(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    left = {(row["case"], row["run"]): row for row in baseline}
    right = {(row["case"], row["run"]): row for row in candidate}
    keys = sorted(left.keys() & right.keys())
    accuracy_delta = [
        float(bool(right[key]["success"])) - float(bool(left[key]["success"]))
        for key in keys
    ]
    latency_delta = [
        float(right[key]["latency_ms"]) - float(left[key]["latency_ms"])
        for key in keys
    ]
    token_delta = [
        float(right[key]["total_tokens"]) - float(left[key]["total_tokens"])
        for key in keys
    ]
    accuracy_ci = bootstrap_mean_ci(accuracy_delta, seed=seed)
    latency_ci = bootstrap_mean_ci(latency_delta, seed=seed + 1)
    return {
        "paired_requests": len(keys),
        "accuracy_lift": statistics.fmean(accuracy_delta) if accuracy_delta else None,
        "accuracy_lift_95pct_bootstrap_ci": list(accuracy_ci) if accuracy_ci else None,
        "latency_delta_ms_mean": statistics.fmean(latency_delta) if latency_delta else None,
        "latency_delta_ms_95pct_bootstrap_ci": list(latency_ci) if latency_ci else None,
        "total_token_delta": sum(token_delta) if token_delta else 0,
        "wins": sum(delta > 0 for delta in accuracy_delta),
        "ties": sum(delta == 0 for delta in accuracy_delta),
        "losses": sum(delta < 0 for delta in accuracy_delta),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:11435")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--modes", default="off,auto,deep")
    parser.add_argument("--seed", type=int, default=20261124)
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
    rng = random.Random(args.seed)
    samples: dict[str, list[dict[str, Any]]] = {mode: [] for mode in modes}
    execution_order: list[dict[str, Any]] = []
    # Interleave and randomize mode order within each case/run so thermal,
    # cache-warmth and time-of-run effects do not systematically favor one arm.
    for case in CASES:
        for _ in range(args.warmup):
            warmup_modes = list(modes)
            rng.shuffle(warmup_modes)
            for mode in warmup_modes:
                request_once(endpoint, args.model, case, mode, args.api_key, args.timeout)
        for run_index in range(args.runs):
            run_modes = list(modes)
            rng.shuffle(run_modes)
            execution_order.append({
                "case": case["id"],
                "run": run_index + 1,
                "mode_order": run_modes,
            })
            for mode in run_modes:
                result = request_once(
                    endpoint, args.model, case, mode, args.api_key, args.timeout
                )
                samples[mode].append({"run": run_index + 1, **result})

    report["randomization_seed"] = args.seed
    report["execution_order"] = execution_order
    for mode in modes:
        rows = samples[mode]
        report["results"][mode] = {
            "summary": summarise(rows),
            "per_case": per_case_summary(rows),
            "samples": rows,
        }

    if "off" in samples:
        report["paired_vs_off"] = {
            mode: paired_comparison(samples["off"], samples[mode], seed=args.seed + 10)
            for mode in modes if mode != "off"
        }

    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
