#!/usr/bin/env python3
"""pallama engine-vs-engine benchmark — llama-bench sweep across ALL installed engines.

scripts/bench_compare.py answers "how does pallama's orchestration compare to
direct llama-server / ollama / the engine ceiling?" using the ACTIVE engine.
This script answers the other question: "which installed ENGINE should I run?"

  every engine in <data>/engines that ships a llama-bench binary is benched
  with the SAME model, prompt-processing and generation sizes, and GPU
  offload — sequentially, one engine at a time, so the GPU is exclusively
  each bench's during its run (fairness contract; never parallel).

  Engines without llama-bench (e.g. mistral.rs — different binary family)
  are listed and skipped: that comparison is an HTTP-level measurement, not
  a llama-bench sweep.

Results: per-engine pp<N>/tg<N> tok/s with standard deviation, plus a
relative-to-best column, and raw llama-bench JSON artifacts under
~/.cache/pallama-bench-engines/<timestamp>/ for later re-analysis.

Usage:
  scripts/bench_engines.py                          # largest model, all engines
  scripts/bench_engines.py --model qwen3.5-9b       # substring of store name/file
  scripts/bench_engines.py --pp 2048 --tg 256       # heavier prefill/decode mix
  scripts/bench_engines.py b10809-cuda b10809       # only these engine tags
  scripts/bench_engines.py -- -fa on                # extra llama-bench passthrough

Exit codes: 0 clean · 1 bench failure (>=1 engine errored) · 2 environment
abort (no model found / no benchable engine).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

DEFAULT_DATA_DIR = os.environ.get(
    "PALLAMA_DATA_DIR",
    os.path.join(
        os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share")), "pallama"
    ),
)


def log(msg: str = "") -> None:
    print(msg, flush=True)


def find_bench(engine_dir: str) -> str | None:
    """Locate llama-bench inside one extracted engine tree (shallowest wins)."""
    for root, dirs, files in os.walk(engine_dir):
        dirs.sort()
        for name in ("llama-bench", "llama-bench.exe"):
            if name in files:
                return os.path.join(root, name)
    return None


def discover(data_dir: str) -> list[tuple[str, str]]:
    """All installed engines as (tag, llama-bench path) — benchable only."""
    engines_dir = os.path.join(data_dir, "engines")
    found: list[tuple[str, str]] = []
    if not os.path.isdir(engines_dir):
        return found
    for tag in sorted(os.listdir(engines_dir)):
        bench = find_bench(os.path.join(engines_dir, tag))
        if bench and os.access(bench, os.X_OK):
            found.append((tag, bench))
    return found


def list_all_engines(data_dir: str) -> list[str]:
    engines_dir = os.path.join(data_dir, "engines")
    return sorted(os.listdir(engines_dir)) if os.path.isdir(engines_dir) else []


def pick_model(models_dir: str, want: str | None) -> str:
    """Resolve the bench model.

    Default heuristic: the LARGEST .gguf in the shared models dir — test
    artifacts (imatrix, verify shards) are small; real models dominate.
    `want` matches as substring against filenames, case-insensitive.
    """
    if not os.path.isdir(models_dir):
        raise SystemExit(
            f"no models dir at {models_dir} — pull a model first (`pallama pull`)"
        )
    ggufs = [f for f in os.listdir(models_dir) if f.lower().endswith(".gguf")]
    if not ggufs:
        raise SystemExit(
            f"no .gguf files in {models_dir} — pull a model first (`pallama pull`)"
        )
    if want:
        matches = [f for f in ggufs if want.lower() in f.lower()]
        if len(matches) != 1:
            raise SystemExit(
                f"model '{want}' matches {matches or 'nothing'} — need exactly one"
            )
        return os.path.join(models_dir, matches[0])
    largest = max(ggufs, key=lambda f: os.path.getsize(os.path.join(models_dir, f)))
    return os.path.join(models_dir, largest)


def run_engine(
    tag: str,
    bench: str,
    model: str,
    pp: int,
    tg: int,
    ngl: int,
    extra: list[str],
    timeout_s: int,
) -> dict:
    """One engine's sweep. Returns parsed pp/tg numbers or {'error': ...}."""
    argv = [
        bench,
        "-m",
        model,
        "-p",
        str(pp),
        "-n",
        str(tg),
        "-ngl",
        str(ngl),
        "-o",
        "json",
        *extra,
    ]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return {"error": f"timed out after {timeout_s}s"}
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout)[-300:]
        return {"error": f"rc={proc.returncode}: {tail.strip()}"}
    try:
        entries = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"error": "unparsed json output"}
    pp_entry = None
    tg_entry = None
    for e in entries:
        if e.get("n_prompt", 0) > 0 and e.get("n_gen", 0) == 0:
            pp_entry = e
        elif e.get("n_gen", 0) > 0 and e.get("n_prompt", 0) == 0:
            tg_entry = e
    if pp_entry is None or tg_entry is None:
        return {"error": "missing pp/tg entries in output"}
    out: dict = {
        "backend": entries[0].get("backends", "?") if entries else "?",
        "gpu": entries[0].get("gpu_info", "-") if entries else "-",
    }
    try:
        for label, entry, tokens in (
            ("pp", pp_entry, pp_entry["n_prompt"]),
            ("tg", tg_entry, tg_entry["n_gen"]),
        ):
            avg = entry["avg_ns"]
            out[label] = tokens / avg * 1e9
            # tok/s stddev propagated from the time distribution
            out[f"{label}_sd"] = tokens * entry.get("stddev_ns", 0) / (avg * avg) * 1e9
    except (NameError, KeyError, ZeroDivisionError):
        return {"error": "missing pp/tg entries in output"}
    return out


def fmt_tps(v: float | None, sd: float | None) -> str:
    if v is None:
        return "-"
    return f"{v:8.1f} ±{sd:5.1f}" if sd is not None else f"{v:8.1f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument(
        "tags", nargs="*", help="engine tags to bench (default: all benchable)"
    )
    ap.add_argument(
        "--model", default=None, help="model substring (default: largest .gguf)"
    )
    ap.add_argument(
        "--pp", type=int, default=512, help="prompt-processing tokens (default 512)"
    )
    ap.add_argument(
        "--tg", type=int, default=128, help="generated tokens (default 128)"
    )
    ap.add_argument(
        "--ngl", type=int, default=99, help="GPU layers (default 99 = full offload)"
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=1200,
        help="per-engine timeout seconds (default 1200)",
    )
    ap.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help=f"pallama data dir (default {DEFAULT_DATA_DIR})",
    )
    ap.add_argument("extra", nargs="*", help=argparse.SUPPRESS)
    known, extra = ap.parse_known_args()
    if extra and extra and extra[0] == "--":
        extra = extra[1:]

    all_engines = list_all_engines(known.data_dir)
    benchable = discover(known.data_dir)
    if not benchable:
        log(f"no engine with a llama-bench binary under {known.data_dir}/engines")
        return 2
    skipped = [t for t in all_engines if t not in dict(benchable)]
    selected = [(t, b) for t, b in benchable if not known.tags or t in known.tags]
    if not selected:
        log(
            f"tags {known.tags} match none of {benchable and [t for t, _ in benchable]}"
        )
        return 2
    model = pick_model(os.path.join(known.data_dir, "models"), known.model)

    ts = time.strftime("%Y%m%d-%H%M%S")
    outdir = os.path.expanduser(f"~/.cache/pallama-bench-engines/{ts}")
    os.makedirs(outdir, exist_ok=True)

    log(
        f"engine sweep: {len(selected)}/{len(all_engines)} engines · model {os.path.basename(model)}"
    )
    log(
        f"              pp{known.pp} tg{known.tg} ngl {known.ngl} · sequential (GPU exclusive)"
    )
    if skipped:
        log(f"              skipped (no llama-bench): {', '.join(skipped)}")
    log(f"              artifacts: {outdir}")
    log()

    results: list[tuple[str, dict]] = []
    for i, (tag, bench) in enumerate(selected, 1):
        print(f"[{i}/{len(selected)}] {tag} ...", end=" ", flush=True)
        res = run_engine(
            tag, bench, model, known.pp, known.tg, known.ngl, extra, known.timeout
        )
        results.append((tag, res))
        if "error" in res:
            log(f"FAILED — {res['error']}")
        else:
            log(
                f"pp{known.pp} {res['pp']:.1f} t/s · tg{known.tg} {res['tg']:.1f} t/s · {res['backend']}"
            )

    ok = [(t, r) for t, r in results if "error" not in r]
    summary_path = os.path.join(outdir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(
            f"# engine sweep {ts} · {os.path.basename(model)} · pp{known.pp} tg{known.tg} ngl{known.ngl}\n"
        )
        f.write(json.dumps({t: r for t, r in results}, indent=2) + "\n")
    if len(ok) < 1:
        log("\nno engine produced numbers")
        return 1

    best_pp = max(r["pp"] for _, r in ok)
    best_tg = max(r["tg"] for _, r in ok)
    log()
    log(
        f"  {'engine':<14}{'backend':<8}{'pp' + str(known.pp) + ' t/s':>20}{'rel':>7}{'tg' + str(known.tg) + ' t/s':>20}{'rel':>7}"
    )
    log("  " + "-" * 74)
    for tag, r in ok:
        log(
            f"  {tag:<14}{r['backend'][:7]:<8}"
            f"{fmt_tps(r['pp'], r['pp_sd']):>20}{100 * r['pp'] / best_pp:>6.0f}%"
            f"{fmt_tps(r['tg'], r['tg_sd']):>20}{100 * r['tg'] / best_tg:>6.0f}%"
        )
    log("  " + "-" * 74)
    log(f"  gpu: {ok[0][1].get('gpu', '-')}")
    log(f"  summary + raw json: {outdir}")
    return 1 if len(ok) < len(results) else 0


if __name__ == "__main__":
    sys.exit(main())
