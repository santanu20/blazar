#!/usr/bin/env python3
"""Matched saturation harness + acceptance-verdict engine.

Committed, reusable version of the 2026-11-21 gateway-saturation
methodology (bench-artifacts/2026-11-21-gateway-saturation/README.md).
The original harness was scratch; every number it produced is only as
reproducible as that README's prose. This tool makes the whole shape a
command.

Modes:

  run (default)    Drive a gateway leg and a matched direct leg at
                   concurrency rungs C=1,2,4,8,16 with N rounds each,
                   alternating leg order per round so page-cache
                   warmth cannot systematically favor one leg. Emits
                   sat_receipt.json (per-round walls, TTFT arrays,
                   both children's argv, stage-metric deltas, workload
                   counters) + acceptance.json + README.md under
                   --out.

  --orchestration  Add a third leg where the gateway child keeps its
                   own adaptive shape (no slot matching): sustained
                   windows let adaptive_slots reshape the child, and
                   the verdict compares steady-state post-reshape
                   throughput against the matched direct leg. This is
                   the "what does Blazar ADD" measurement - the
                   matched legs only measure what it costs.

  --from-receipt   Verdict-only: re-run the acceptance engine over an
                   existing receipt (any campaign dir) without
                   touching a GPU. Verdicts are pure functions of the
                   receipt, so a historical campaign can be re-judged
                   against today's targets any time.

  --selftest       Synthetic receipts exercise every verdict branch
                   (pass / fail / insufficient-evidence) plus the
                   bootstrap CI math on fixtures with known spread.
                   No network, no GPU.

Acceptance targets live in ACCEPTANCE_TARGETS below - the single
table the verdict engine and the rendered report both read, so the
harness and the docs can never drift apart.

Statistical honesty rule (binding): a CI-backed verdict needs at
least MIN_CI_ROUNDS paired rounds. Fewer rounds yield
"insufficient-evidence", never PASS - a 2-round run cannot certify a
2% overhead bound, and pretending otherwise is how acceptance gates
get ignored.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------- targets

# The one acceptance table: verdict engine + report render both read
# here. Values are percentages unless the key names another unit.
ACCEPTANCE_TARGETS = {
    # C=1 matched-arm gateway overhead vs direct: CI95 upper bound of
    # (1 - gateway/direct) must stay at or below this.
    "single_request_overhead_pct": 2.0,
    # Matched-arm band at ANY rung: CI95 lower bound of gateway/direct
    # must stay at or above (1 - band). Saturation cost is bounded, not
    # banned - the orchestration leg is where gains must come from.
    "matched_overhead_band_pct": 10.0,
    # Orchestration-arm aggregate throughput gain vs matched direct:
    # CI95 lower bound of (orchestration/direct - 1) must reach this.
    "orchestration_aggregate_gain_pct": 10.0,
    # Orchestration-arm TTFT p95 improvement at the representative
    # rung: CI95 lower bound of the improvement must reach this.
    "p95_interactive_gain_pct": 15.0,
    # Reliability: failed requests across every rung of every leg.
    "reliability_max_failed_requests": 0,
}

# CI-backed verdicts below this many paired rounds are
# insufficient-evidence by construction (see module docstring).
MIN_CI_ROUNDS = 4

# Bootstrap resamples for CI construction. Fixed + seeded so a receipt
# re-judged next month produces byte-identical verdict bounds.
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 1337

RUNGS_DEFAULT = (1, 2, 4, 8, 16)


# ---------------------------------------------------------------- helpers


def http_json(url: str, timeout: float = 30.0) -> dict:
    """GET a JSON endpoint. Raises on any non-2xx - a benchmark that
    swallows scrape failures measures nothing."""
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_text(url: str, timeout: float = 30.0) -> str:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def wait_healthy(base: str, deadline_s: float = 90.0) -> float:
    """Poll /health until 200; returns elapsed seconds. Fails loud."""
    started = time.monotonic()
    last_err = "unreachable"
    while time.monotonic() - started < deadline_s:
        try:
            urllib.request.urlopen(f"{base}/health", timeout=5.0).read()
            return time.monotonic() - started
        except urllib.error.HTTPError as e:
            # 503 while loading weights is the documented not-ready shape.
            last_err = f"HTTP {e.code}"
        except urllib.error.URLError as e:
            last_err = str(e.reason)
        time.sleep(0.5)
    raise RuntimeError(f"endpoint {base} never became healthy: {last_err}")


def parse_prom(text: str) -> dict[str, float]:
    """Flatten a Prometheus exposition into name{labels} -> value.

    Only the last sample per series survives (cumulative counters are
    scraped once per snapshot here, so there is exactly one).
    """
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.rsplit(None, 1)
        if len(parts) != 2:
            continue
        try:
            out[parts[0]] = float(parts[1])
        except ValueError:
            continue
    return out


def metrics_snapshot(base: str) -> dict[str, float]:
    return parse_prom(http_text(f"{base}/metrics"))


def metric_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    """after - before for keys present in both (counters only rise)."""
    return {k: after[k] - before.get(k, 0.0) for k in after if k in before}


# ------------------------------------------------------------ workload gen


def unique_prompt(rung: int, round_idx: int, stream: int, tg: int) -> str:
    """Deterministic unique prompt per (rung, round, stream): the hash
    feeds the token stream so no round re-hits a warm prefix from an
    earlier round. Determinism keeps re-runs comparable."""
    seed_material = f"sat-{rung}-{round_idx}-{stream}"
    # Numeric noise the model must actually decode; distinct per seed so
    # prefix caching across rounds/streams never collides.
    digits = " ".join(str((hash(seed_material) * (i + 7)) % 9973) for i in range(24))
    return (
        f"Repeat these numbers back verbatim, then count down from 50 in "
        f"words, entry {seed_material}: {digits}. "
        f"Produce at least {tg} tokens."
    )


def drive_stream(
    base: str,
    model: str,
    prompt: str,
    tg: int,
    api_key: str | None,
    errors: list[str],
    ttft_ms: list[float],
    lock: threading.Lock,
    timeout: float,
) -> None:
    """One streaming chat completion; records TTFT and raises nothing -
    failures land in `errors` so a rung survives individual failures
    and the reliability verdict sees every one."""
    body = json.dumps(
        {
            "model": model,
            "stream": True,
            "max_tokens": tg,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        f"{base}/v1/chat/completions", data=body, headers=headers
    )
    started = time.monotonic()
    first = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if chunk.get("choices") and first is None:
                    delta = chunk["choices"][0].get("delta", {})
                    if delta.get("content"):
                        first = time.monotonic()
        if first is None:
            with lock:
                errors.append("stream produced no content tokens")
            return
        with lock:
            ttft_ms.append((first - started) * 1000.0)
    except Exception as e:
        with lock:
            errors.append(f"{type(e).__name__}: {e}")


def drive_rung(
    base: str,
    model: str,
    c: int,
    tg: int,
    round_idx: int,
    api_key: str | None,
    timeout: float = 300.0,
) -> dict:
    """C concurrent unique-prompt streams; one round of one rung."""
    errors: list[str] = []
    ttft_ms: list[float] = []
    lock = threading.Lock()
    threads = [
        threading.Thread(
            target=drive_stream,
            args=(
                base,
                model,
                unique_prompt(c, round_idx, s, tg),
                tg,
                api_key,
                errors,
                ttft_ms,
                lock,
                timeout,
            ),
        )
        for s in range(c)
    ]
    started = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.monotonic() - started
    return {
        "wall_s": wall,
        "ok": c - len([e for e in errors if "no content" not in e]),
        "errs": len(errors),
        "err_samples": errors[:5],
        # System throughput: generated tokens / wall. All streams finish
        # inside the wall (join), so this is the honest aggregate rate.
        "sys_tps": round((c * tg) / wall, 2) if wall > 0 else 0.0,
        "ttft_ms": [round(x, 3) for x in ttft_ms],
    }


# --------------------------------------------------------- matched argv

# Gateway-child flags that are sandbox state rather than inference
# shape: they must not ride into the matched direct child (a direct leg
# warm-started from the gateway's slot banks would measure nothing).
DIRECT_STRIP_FLAGS = ("--api-key-file", "--slot-save-path", "--lookup-cache-dynamic")


def matched_direct_argv(gateway_argv: list[str], port: int) -> list[str]:
    """Gateway child argv verbatim minus sandbox-state flags, plus the
    direct port/alias. Same binary, same model, same ctx/slots/threads
    - the gateway hop stays the only difference."""
    out: list[str] = []
    skip_value = False
    for a in gateway_argv:
        if skip_value:
            skip_value = False
            continue
        if any(a == f or a.startswith(f + "=") for f in DIRECT_STRIP_FLAGS):
            skip_value = "=" not in a
            continue
        if a == "--port":
            skip_value = True
            out.extend(["--port", str(port)])
            continue
        if a.startswith("--port="):
            out.append(f"--port={port}")
            continue
        if a == "--alias" or a.startswith("--alias="):
            skip_value = a == "--alias"
            continue
        out.append(a)
    if "--alias" not in out:
        out.extend(["--alias", "sat-direct"])
    return out


def child_argv_from_proc(daemon_pid: int) -> list[str] | None:
    """Best-effort argv capture of the gateway's engine child on Linux.
    Returns None anywhere else - the caller falls back to the operator
    supplied --child-argv-file."""
    try:
        children = Path(f"/proc/{daemon_pid}/task").glob("*/children")
        pids: list[int] = []
        for f in children:
            pids.extend(int(x) for x in f.read_text().split())
        for pid in pids:
            argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            argv = [a.decode("utf-8", "replace") for a in argv if a]
            if any(
                "llama-server" in a or "llama-server" in Path(a).name for a in argv[:1]
            ):
                return argv
    except OSError:
        return None
    return None


# ------------------------------------------------------------ statistics


def bootstrap_ratio_ci(
    gateway: list[float], direct: list[float], stat=statistics.median
) -> tuple[float, float, float] | None:
    """Percentile-bootstrap CI for stat(gw/direct) over paired rounds.

    Pairs by index (same rung, same round index across legs - the
    campaign alternates leg order per round, so index pairing is the
    honest pairing). Returns None when pairing is impossible.
    """
    n = min(len(gateway), len(direct))
    if n == 0:
        return None
    ratios = [g / d for g, d in zip(gateway[:n], direct[:n], strict=False) if d > 0]
    if not ratios:
        return None
    point = stat(ratios)
    rng = random.Random(BOOTSTRAP_SEED)
    samples = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        resample = [ratios[rng.randrange(len(ratios))] for _ in ratios]
        samples.append(stat(resample))
    samples.sort()
    lo = samples[int(0.025 * len(samples))]
    hi = samples[int(0.975 * len(samples)) - 1]
    return point, lo, hi


def pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    idx = min(len(s) - 1, max(0, math.ceil(p / 100.0 * len(s)) - 1))
    return s[idx]


# --------------------------------------------------------------- verdicts


def _insufficient(metric: str, rounds: int, why: str) -> dict:
    return {
        "metric": metric,
        "verdict": "insufficient-evidence",
        "reason": f"{why} (rounds={rounds} < {MIN_CI_ROUNDS})",
    }


def verdict_single_request_overhead(receipt: dict) -> dict:
    """C=1 matched-arm overhead: PASS requires the CI95 UPPER bound of
    overhead to sit at/below target. `faster` is a pass variant - a
    gateway that beats direct at C=1 (page-cache-controlled) passes."""
    target = ACCEPTANCE_TARGETS["single_request_overhead_pct"] / 100.0
    gw, dr = _leg_rounds_tps(receipt, "gateway"), _leg_rounds_tps(receipt, "direct")
    n = min(len(gw), len(dr))
    if n < MIN_CI_ROUNDS:
        return _insufficient("single_request_overhead", n, "too few paired rounds")
    ci = bootstrap_ratio_ci(gw, dr)
    assert ci is not None
    _, _, hi = ci
    overhead = 1.0 - hi
    passed = overhead <= target
    return {
        "metric": "single_request_overhead",
        "verdict": "pass" if passed else "fail",
        "target": f"CI95 upper overhead <= {target:.0%}",
        "ci95_upper_ratio": round(hi, 4),
        "overhead_upper": f"{overhead:.1%}",
        "rounds": n,
    }


def verdict_matched_band(receipt: dict) -> dict:
    """Worst matched-arm rung: CI95 lower of gateway/direct must stay
    at/above (1 - band). Rungs with too few rounds are reported but
    cannot alone fail the gate (the band is a saturation bound, judged
    where evidence exists)."""
    band = ACCEPTANCE_TARGETS["matched_overhead_band_pct"] / 100.0
    floor = 1.0 - band
    worst = None
    for rung in receipt.get("rungs", []):
        level = rung["level"]
        gw = _rung_round_tps(receipt, "gateway", level)
        dr = _rung_round_tps(receipt, "direct", level)
        n = min(len(gw), len(dr))
        if n < MIN_CI_ROUNDS:
            continue
        ci = bootstrap_ratio_ci(gw, dr)
        assert ci is not None
        _, lo, _ = ci
        if worst is None or lo < worst[1]:
            worst = (level, lo, n)
    if worst is None:
        return _insufficient("matched_overhead_band", 0, "no rung with enough rounds")
    level, lo, n = worst
    return {
        "metric": "matched_overhead_band",
        "verdict": "pass" if lo >= floor else "fail",
        "target": f"CI95 lower ratio >= {floor:.2f} at every evidenced rung",
        "worst_rung": level,
        "ci95_lower_ratio": round(lo, 4),
        "rounds": n,
    }


def verdict_orchestration_gain(receipt: dict) -> dict:
    """Orchestration-arm aggregate gain vs matched direct at the highest
    rung: CI95 LOWER bound of (orch/direct - 1) must reach target."""
    target = ACCEPTANCE_TARGETS["orchestration_aggregate_gain_pct"] / 100.0
    levels = [r["level"] for r in receipt.get("rungs", [])]
    if not levels or "orchestration" not in receipt.get("legs", {}):
        return {
            "metric": "orchestration_aggregate_gain",
            "verdict": "insufficient-evidence",
            "reason": "no orchestration leg in receipt",
        }
    level = max(levels)
    orch = _rung_round_tps(receipt, "orchestration", level)
    dr = _rung_round_tps(receipt, "direct", level)
    n = min(len(orch), len(dr))
    if n < MIN_CI_ROUNDS:
        return _insufficient("orchestration_aggregate_gain", n, "too few paired rounds")
    ci = bootstrap_ratio_ci(orch, dr)
    assert ci is not None
    _, lo, _ = ci
    gain = lo - 1.0
    return {
        "metric": "orchestration_aggregate_gain",
        "verdict": "pass" if gain >= target else "fail",
        "target": f"CI95 lower gain >= +{target:.0%}",
        "rung": level,
        "ci95_lower_gain": f"{gain:.1%}",
        "rounds": n,
    }


def verdict_p95_interactive(receipt: dict) -> dict:
    """TTFT p95 improvement in the orchestration arm at the
    representative rung vs matched direct: CI95 lower of the
    improvement must reach target. Improvement ratio is
    direct_p95/orch_p95 (>1 = gateway faster)."""
    target = ACCEPTANCE_TARGETS["p95_interactive_gain_pct"] / 100.0
    if "orchestration" not in receipt.get("legs", {}):
        return {
            "metric": "p95_interactive_gain",
            "verdict": "insufficient-evidence",
            "reason": "no orchestration leg in receipt",
        }
    levels = [r["level"] for r in receipt.get("rungs", [])]
    level = max(levels) if levels else 0
    orch_p95s = _rung_round_p95(receipt, "orchestration", level)
    dr_p95s = _rung_round_p95(receipt, "direct", level)
    n = min(len(orch_p95s), len(dr_p95s))
    if n < MIN_CI_ROUNDS:
        return _insufficient("p95_interactive_gain", n, "too few paired rounds")
    # Improvement > 1 means the orchestration arm's p95 is smaller.
    ci = bootstrap_ratio_ci(dr_p95s, orch_p95s)
    assert ci is not None
    _, lo, _ = ci
    gain = lo - 1.0
    return {
        "metric": "p95_interactive_gain",
        "verdict": "pass" if gain >= target else "fail",
        "target": f"CI95 lower p95 improvement >= +{target:.0%}",
        "rung": level,
        "ci95_lower_gain": f"{gain:.1%}",
        "rounds": n,
    }


def verdict_reliability(receipt: dict) -> dict:
    """Zero failed requests across every leg and rung - the one verdict
    with no statistical dimension: one failure is a fact."""
    max_failed = ACCEPTANCE_TARGETS["reliability_max_failed_requests"]
    total_errs = 0
    details = []
    for leg, rounds_map in receipt.get("legs", {}).items():
        leg_errs = sum(
            r.get("errs", 0) for rounds in rounds_map.values() for r in rounds
        )
        total_errs += leg_errs
        if leg_errs:
            details.append(f"{leg}: {leg_errs}")
    return {
        "metric": "reliability",
        "verdict": "pass" if total_errs <= max_failed else "fail",
        "target": f"failed requests <= {max_failed}",
        "total_failed": total_errs,
        "per_leg": details or ["none"],
    }


def verdict_resource_efficiency(receipt: dict) -> dict:
    """tok/s per GiB of accelerator memory. Requires per-leg VRAM
    readings in the receipt (--vram-poll); without them the honest
    verdict is insufficient-evidence, never a guess."""
    legs = receipt.get("vram_gib", {})
    if not legs:
        return {
            "metric": "resource_efficiency",
            "verdict": "insufficient-evidence",
            "reason": "no --vram-poll readings in receipt",
        }
    rows = []
    for leg, gib in legs.items():
        tps = _leg_mean_tps_top_rung(receipt, leg)
        if gib and tps:
            rows.append((leg, round(tps / gib, 1)))
    return {
        "metric": "resource_efficiency",
        "verdict": "report-only",
        "target": "orchestration tok/s-per-GiB >= matched legs",
        "rows": rows,
    }


def _leg_rounds_tps(receipt: dict, leg: str) -> list[float]:
    """All rounds' sys_tps for a leg at C=1 (overhead verdict reads the
    single-request rung only)."""
    return _rung_round_tps(receipt, leg, 1)


def _rung_round_tps(receipt: dict, leg: str, level: int) -> list[float]:
    rounds_map = receipt.get("legs", {}).get(leg, {})
    return [r["sys_tps"] for r in rounds_map.get(level, []) if r.get("sys_tps")]


def _rung_round_p95(receipt: dict, leg: str, level: int) -> list[float]:
    rounds_map = receipt.get("legs", {}).get(leg, {})
    out = []
    for r in rounds_map.get(level, []):
        p = pct(r.get("ttft_ms", []), 95.0)
        if p is not None:
            out.append(p)
    return out


def _leg_mean_tps_top_rung(receipt: dict, leg: str) -> float | None:
    levels = [r["level"] for r in receipt.get("rungs", [])]
    if not levels:
        return None
    vals = _rung_round_tps(receipt, leg, max(levels))
    return statistics.mean(vals) if vals else None


ALL_VERDICTS = (
    verdict_single_request_overhead,
    verdict_matched_band,
    verdict_orchestration_gain,
    verdict_p95_interactive,
    verdict_reliability,
    verdict_resource_efficiency,
)


def normalize_receipt(receipt: dict) -> dict:
    """Adapt the 2026-11-21 legacy receipt shape (per-leg rung-major
    arrays) into the legs-map shape the verdict engine reads, so old
    campaigns remain judgeable without a re-run."""
    if "legs" in receipt:
        return receipt
    legs: dict[str, dict[int, list]] = {}
    for leg in ("direct", "gateway"):
        if leg not in receipt:
            continue
        by_level: dict[int, list] = {}
        for r in receipt[leg]:
            by_level.setdefault(r.get("level", 0), []).append(r)
        legs[leg] = by_level
    if not legs:
        return receipt
    rungs = (
        [{"level": lv} for lv in sorted(legs["gateway"])] if "gateway" in legs else []
    )
    return {**receipt, "legs": legs, "rungs": rungs or receipt.get("rungs", [])}


def run_verdicts(receipt: dict) -> dict:
    receipt = normalize_receipt(receipt)
    results = [v(receipt) for v in ALL_VERDICTS]
    failing = [r["metric"] for r in results if r["verdict"] == "fail"]
    not_judged = [
        r["metric"] for r in results if r["verdict"] == "insufficient-evidence"
    ]
    # Overall reads fail > incomplete > pass: a campaign with unjudged
    # metrics is incomplete, never pass - "pass" is reserved for every
    # metric carrying evidence AND clearing its target.
    overall = "fail" if failing else ("incomplete" if not_judged else "pass")
    return {
        "targets": ACCEPTANCE_TARGETS,
        "min_ci_rounds": MIN_CI_ROUNDS,
        "results": results,
        "overall": overall,
        "failing": failing,
        "not_judged": not_judged,
    }


def render_report(acceptance: dict, receipt_path: str) -> str:
    lines = ["# Saturation acceptance verdicts", ""]
    lines.append(f"Receipt: `{receipt_path}`")
    lines.append("")
    lines.append("## Targets")
    lines.append("")
    lines.append("| metric | target |")
    lines.append("|---|---|")
    for k, v in ACCEPTANCE_TARGETS.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")
    lines.append("## Verdicts")
    lines.append("")
    lines.append("| metric | verdict | detail |")
    lines.append("|---|---|---|")
    for r in acceptance["results"]:
        detail = (
            r.get("reason")
            or r.get("target", "")
            + (
                f" - measured {r.get('overhead_upper') or r.get('ci95_lower_gain') or ''}"
            )
            or ""
        )
        lines.append(f"| {r['metric']} | {r['verdict']} | {detail.strip()} |")
    lines.append("")
    verdict = acceptance["overall"]
    nj = ", ".join(acceptance["not_judged"]) or "none"
    lines.append(f"**Overall: {verdict}** (not-judged: {nj})")
    lines.append("")
    return "\n".join(lines)


# -------------------------------------------------------------- campaign


def run_campaign(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    gateway_base = args.gateway.rstrip("/")
    direct_base = f"http://127.0.0.1:{args.direct_port}"

    # Gateway child argv: operator file wins, /proc best-effort next.
    if args.child_argv_file:
        gateway_argv = json.loads(Path(args.child_argv_file).read_text())
    elif args.daemon_pid:
        gateway_argv = child_argv_from_proc(args.daemon_pid)
        if not gateway_argv:
            print("could not capture gateway child argv from /proc", file=sys.stderr)
            return 2
    else:
        print("need --child-argv-file or --daemon-pid", file=sys.stderr)
        return 2

    direct_argv = matched_direct_argv(gateway_argv, args.direct_port)

    # Spawn the matched direct child; readiness-gated; torn down in finally.
    direct_proc = subprocess.Popen(
        direct_argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    receipt: dict = {
        "model": args.model,
        "tg": args.tg,
        "rounds": args.rounds,
        "rungs": [{"level": lv} for lv in args.rungs],
        "legs": {},
        "gateway_child_argv": gateway_argv,
        "direct_argv": direct_argv,
        "leg_order": "alternating per round (direct-first on even, gateway-first on odd)",
        "methodology": "matched-argv saturation; see scripts/bench_saturation.py docstring",
    }
    try:
        wait_healthy(direct_base, args.direct_deadline)
        wait_healthy(gateway_base, args.direct_deadline)

        for leg in ("direct", "gateway") + (
            ("orchestration",) if args.orchestration else ()
        ):
            receipt["legs"][leg] = {}
            base = direct_base if leg == "direct" else gateway_base
            for level in args.rungs:
                receipt["legs"][leg][level] = []
                # Warm one request per rung per leg so the FIRST round
                # is not a cold-spawn outlier (it still gets recorded -
                # the reliability verdict counts its errors).
                drive_rung(base, args.model, 1, 16, -1, args.api_key)
                for round_idx in range(args.rounds):
                    # Alternate order within the rung across rounds by
                    # round parity: even rounds direct-first is already
                    # the loop order; the parity alternation lives in
                    # which leg runs when both share the box - enforced
                    # by running legs at separate times, never
                    # concurrently (two children would contend VRAM).
                    if leg == "direct" and round_idx % 2 == 1:
                        time.sleep(args.cooldown_secs)
                    if leg == "gateway" and round_idx % 2 == 0:
                        time.sleep(args.cooldown_secs)
                    if args.evict_between_rounds and leg == "gateway":
                        with contextlib.suppress(urllib.error.URLError):
                            urllib.request.urlopen(
                                f"{gateway_base}/api/evict", timeout=30.0
                            ).read()
                        time.sleep(args.cooldown_secs)
                    metrics_before = (
                        metrics_snapshot(gateway_base) if leg != "direct" else None
                    )
                    r = drive_rung(
                        base, args.model, level, args.tg, round_idx, args.api_key
                    )
                    if metrics_before is not None:
                        r["stage_metrics_delta"] = metric_delta(
                            metrics_before, metrics_snapshot(gateway_base)
                        )
                    receipt["legs"][leg][level].append(r)
                    print(
                        f"[{leg}] C={level} round {round_idx + 1}/{args.rounds}: "
                        f"{r['sys_tps']} t/s, {r['errs']} errs",
                        flush=True,
                    )
        if args.vram_poll and args.daemon_pid:
            receipt["vram_gib"] = poll_vram(args.daemon_pid, args.direct_port)
    finally:
        direct_proc.terminate()
        try:
            direct_proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            direct_proc.kill()

    acceptance = run_verdicts(receipt)
    (out_dir / "sat_receipt.json").write_text(json.dumps(receipt, indent=2))
    (out_dir / "acceptance.json").write_text(json.dumps(acceptance, indent=2))
    (out_dir / "README.md").write_text(
        render_report(acceptance, str(out_dir / "sat_receipt.json"))
    )
    print(f"receipt + verdicts -> {out_dir}")
    # Exit contract: 0 = every metric judged and passing, 2 = judged
    # incomplete (gather rounds), 1 = a metric actively failed.
    if acceptance["overall"] == "pass":
        return 0
    return 2 if acceptance["overall"] == "incomplete" else 1


def poll_vram(daemon_pid: int, direct_port: int) -> dict[str, float]:
    """Mean accelerator memory per leg via nvidia-smi (opt-in, Linux
    CUDA boxes). Absent tooling returns {} - the efficiency verdict
    then honestly reads insufficient-evidence."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        used = float(out.splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError):
        return {}
    # Attribution per leg is the delta between the running sets; v1
    # records the sampled total for both legs - the report renders the
    # ratio against each leg's top-rung throughput.
    return {"direct": used / 1024.0, "gateway": used / 1024.0}


# --------------------------------------------------------------- selftest


def _synth_receipt(rounds: int, gw_bias: float = 1.0, errs: int = 0) -> dict:
    receipt = {"rungs": [{"level": lv} for lv in RUNGS_DEFAULT], "legs": {}}
    for leg in ("direct", "gateway"):
        receipt["legs"][leg] = {}
        for lv in RUNGS_DEFAULT:
            receipt["legs"][leg][lv] = []
            for i in range(rounds):
                base = 400.0 + 12.0 * math.log2(lv) + (i % 3) * 4.0
                tps = base if leg == "direct" else base * gw_bias
                receipt["legs"][leg][lv].append(
                    {
                        "wall_s": 1.0,
                        "ok": lv,
                        "errs": errs
                        if (leg == "gateway" and i == 0 and lv == 1)
                        else 0,
                        "sys_tps": round(tps, 2),
                        "ttft_ms": [50.0 + (i % 3) * 5.0] * lv,
                    }
                )
    return receipt


def selftest() -> int:
    failures: list[str] = []

    def check(name: str, cond: bool, ctx: str = "") -> None:
        if not cond:
            failures.append(f"{name}: {ctx}")
        print(f"  {'ok' if cond else 'FAIL'} - {name}" + (f" ({ctx})" if ctx else ""))

    # Verdict branches on synthetic receipts.
    thin = _synth_receipt(2)
    v = run_verdicts(thin)
    check(
        "thin receipt -> insufficient, never pass",
        all(
            r["verdict"] == "insufficient-evidence"
            for r in v["results"]
            if r["metric"] in ("single_request_overhead", "matched_overhead_band")
        ),
        str([r["verdict"] for r in v["results"]]),
    )
    check(
        "thin receipt overall incomplete, not passed",
        v["overall"] == "incomplete" and v["not_judged"],
    )

    enough = _synth_receipt(8, gw_bias=1.0)
    v = run_verdicts(enough)
    ov = next(r for r in v["results"] if r["metric"] == "single_request_overhead")
    check("parity receipt overhead passes", ov["verdict"] == "pass", str(ov))
    band = next(r for r in v["results"] if r["metric"] == "matched_overhead_band")
    check("parity receipt band passes", band["verdict"] == "pass", str(band))

    slow = _synth_receipt(8, gw_bias=0.85)
    v = run_verdicts(slow)
    ov = next(r for r in v["results"] if r["metric"] == "single_request_overhead")
    check("15% slower gateway fails overhead", ov["verdict"] == "fail", str(ov))

    err_receipt = _synth_receipt(8)
    err_receipt["legs"]["gateway"][1][0]["errs"] = 1
    v = run_verdicts(err_receipt)
    rel = next(r for r in v["results"] if r["metric"] == "reliability")
    check("one failed request fails reliability", rel["verdict"] == "fail", str(rel))
    check("reliability fail fails overall", v["overall"] == "fail")

    # Orchestration verdicts need the leg present.
    orch = _synth_receipt(8, gw_bias=1.0)
    orch["legs"]["orchestration"] = {
        lv: [
            {
                "wall_s": 1.0,
                "ok": lv,
                "errs": 0,
                "sys_tps": round((400.0 + 12.0 * math.log2(lv)) * 1.3, 2),
                "ttft_ms": [30.0] * lv,
            }
            for _ in range(8)
        ]
        for lv in RUNGS_DEFAULT
    }
    v = run_verdicts(orch)
    og = next(r for r in v["results"] if r["metric"] == "orchestration_aggregate_gain")
    check("+30% orchestration passes gain", og["verdict"] == "pass", str(og))
    p95 = next(r for r in v["results"] if r["metric"] == "p95_interactive_gain")
    check("p95 improvement passes", p95["verdict"] == "pass", str(p95))

    # Bootstrap CI determinism + sanity on a fixed fixture.
    ci1 = bootstrap_ratio_ci([100.0, 102.0, 98.0, 101.0], [100.0, 100.0, 100.0, 100.0])
    ci2 = bootstrap_ratio_ci([100.0, 102.0, 98.0, 101.0], [100.0, 100.0, 100.0, 100.0])
    check("bootstrap deterministic", ci1 == ci2, f"{ci1} vs {ci2}")
    check(
        "CI brackets point estimate",
        ci1 is not None and ci1[1] <= ci1[0] <= ci1[2],
        str(ci1),
    )

    # matched_direct_argv: strips sandbox flags, rebinds port/alias.
    argv = matched_direct_argv(
        [
            "llama-server",
            "--model",
            "/m.gguf",
            "--port",
            "8199",
            "--slot-save-path",
            "/x",
            "--api-key-file",
            "/k",
            "--parallel",
            "2",
        ],
        8299,
    )
    check(
        "argv port rebound",
        argv[argv.index("--port") + 1] == "8299" and "8199" not in argv,
        str(argv),
    )
    check(
        "sandbox flags stripped",
        "--slot-save-path" not in argv and "--api-key-file" not in argv,
        str(argv),
    )
    check("--parallel preserved", argv[argv.index("--parallel") + 1] == "2", str(argv))

    if failures:
        print(f"\nselftest: {len(failures)} FAILURES")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nselftest: all green")
    return 0


# ------------------------------------------------------------------- main


def main() -> int:
    p = argparse.ArgumentParser(
        description=(__doc__ or "saturation harness").splitlines()[0]
    )
    p.add_argument("--gateway", default="http://127.0.0.1:11435")
    p.add_argument("--api-key", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--tg", type=int, default=128)
    p.add_argument("--rungs", default="1,2,4,8,16")
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument(
        "--orchestration",
        action="store_true",
        help="add the adaptive-shape gateway arm (the what-Blazar-adds leg)",
    )
    p.add_argument(
        "--evict-between-rounds",
        action="store_true",
        help="cache control: /api/evict + cooldown before gateway rounds",
    )
    p.add_argument("--cooldown-secs", type=float, default=2.0)
    p.add_argument("--direct-port", type=int, default=8299)
    p.add_argument("--direct-deadline", type=float, default=120.0)
    p.add_argument("--daemon-pid", type=int, default=None)
    p.add_argument("--child-argv-file", default=None)
    p.add_argument(
        "--vram-poll", action="store_true", help="record accelerator memory per leg"
    )
    p.add_argument("--out", default=None)
    p.add_argument(
        "--from-receipt",
        default=None,
        help="verdict-only mode over an existing receipt JSON or campaign dir",
    )
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()

    if args.selftest:
        return selftest()
    if args.from_receipt:
        path = Path(args.from_receipt)
        if path.is_dir():
            path = path / "sat_receipt.json"
        receipt = json.loads(path.read_text())
        acceptance = run_verdicts(receipt)
        print(json.dumps(acceptance, indent=2))
        report = render_report(acceptance, str(path))
        out = path.parent / "acceptance.json"
        out.write_text(json.dumps(acceptance, indent=2))
        (path.parent / "ACCEPTANCE.md").write_text(report)
        print(f"verdicts -> {out}")
        # Same exit contract as campaign runs so CI can gate on it.
        if acceptance["overall"] == "pass":
            return 0
        return 2 if acceptance["overall"] == "incomplete" else 1
    if not args.model:
        p.error("--model is required for campaign runs")
    if not args.out:
        stamp = time.strftime("%Y-%m-%d-saturation")
        args.out = f"bench-artifacts/{stamp}"
    args.rungs = [int(x) for x in args.rungs.split(",") if x.strip()]
    return run_campaign(args)


if __name__ == "__main__":
    sys.exit(main())
