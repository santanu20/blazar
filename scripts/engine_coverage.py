#!/usr/bin/env python3
"""Engine flag coverage audit: how much of each engine's CLI surface
Pallama wires as first-class knobs vs leaves to passthrough.

Method (reproducible evidence, not a static table):
  1. Read each engine manifest's flag list from a pallama store DB
     (`engines.manifest` JSON, as probed by `pallama engine install`).
  2. A flag is WIRED when the literal string appears in the profile
     compiler or the argv translators (profile.rs / engine_impl.rs) —
     i.e. Pallama computes or passes it from a config knob.
  3. Everything else is LONG-TAIL: reachable through per-model
     `extra_args` (verbatim on llamacpp/mistralrs, manifest-gated on
     sglang) and bucketed by keyword families for the coverage doc.

Usage:
  python3 scripts/engine_coverage.py [store_db]
  (default store: ~/.local/share/pallama/pallama.db)
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EMISSION_SOURCES = [
    REPO / "crates/pallama-core/src/profile.rs",
    REPO / "crates/pallama-runtime/src/engine_impl.rs",
]

# Keyword families per engine kind. Order matters: first match wins.
BUCKETS: dict[str, list[tuple[str, str]]] = {
    "sglang": [
        (
            "multi-node/parallelism",
            r"tp-size|dp-size|pp-size|ep-size|-cp-size|dist-init|node-rank|custom-all-reduce|nccl|ptx|gpu-id",
        ),
        (
            "disaggregated/PD serving",
            r"disagg|pd-|prefill-server|decode-server|router|lookup-server|relay",
        ),
        ("MoE backends", r"moe-|deepep|flashinfer-mla|cutlass-"),
        ("RL/rollout tooling", r"rl|rollout|policy|reward|dummy|simpler-"),
        (
            "observability",
            r"metric|trace|otel|log-|bucket-|latency|profile|collect-salyut|telemetry",
        ),
        (
            "per-request/gateway-owned",
            r"api-key|host|port|cors|ssl|schedule-|banned|json|regex|allow-|overflow|reject",
        ),
        ("multimodal/ASR", r"multimodal|image|video|audio|asr|media|vision"),
        ("LoRA advanced", r"lora"),
        (
            "tokenizer/sampler internals",
            r"token|sampl|grammar|detoken|vocab|stop|whitespace",
        ),
        (
            "scheduler/memory detail",
            r"schedule|prefill|decode|batch|cache|kv|memory|graph|quant|offload|attention|chunk|radix|stream|watchdog|timeout|workspace",
        ),
    ],
    "llamacpp": [
        (
            "per-request sampling",
            r"temp|top-|min-|repeat|penalt|presence|frequency|seed|dynatemp|grammar|dry-|xtc|sampl|logit|banned|min-p|typical|mirostat",
        ),
        ("draft/speculative detail", r"draft|spec-|model-draft|beam"),
        ("CPU affinity", r"cpu-mask|cpu-range|poll-|thread|affinity"),
        ("observability", r"metric|log|verbose|debug|trace|props"),
        ("rpc/multi-node", r"rpc"),
        (
            "gateway-owned transport",
            r"api-key|host|port|ssl|cert|cors|prefix|webui|static|proxy|pool|endpoint|listener|unix|tcp",
        ),
        ("multimodal", r"mmproj|image|audio|video"),
        ("LoRA advanced", r"lora"),
        (
            "serving detail",
            r"cache|slot|batch|ctx|n-|keep|par-|flash|kv|prio|queue|cont|defrag|split|rope|yaRN|yarn|embed|rerank|iso|timeout|ping|repack|check|n-predict|escape|chat-template|resource|swap|no-|device|gpu-layer|main-gpu|tensor|override|version|help|apikey",
        ),
    ],
    "mistralrs": [
        (
            "agent family (their product)",
            r"agent|search|shell|code-exec|sandbox|mcp|tool|permission",
        ),
        ("ISQ/quantize tooling", r"calibration|imatrix|isq|quant|uqff|from-"),
        ("observability", r"log|metric|trace|verbose"),
        (
            "gateway-owned",
            r"host|port|api|token|tokeniz|chat-template|fail-on-err|chat",
        ),
        (
            "serving detail",
            r"batch|prefill|decode|cache|seq|ctx|model-len|num-|device|layer|lora|pa-|mtp|encoder|image|vision|prefix|max-|paged|memory|fraction",
        ),
    ],
}


def load_flags(db_path: Path) -> list[tuple[str, str, list[str]]]:
    db = sqlite3.connect(db_path)
    rows = db.execute("select tag, kind, manifest from engines").fetchall()
    out = []
    for tag, kind, manifest in rows:
        try:
            flags = json.loads(manifest).get("flags", [])
        except json.JSONDecodeError:
            continue
        out.append((tag, kind, flags))
    return out


def classify(flag: str, patterns: list[tuple[str, str]]) -> str:
    for name, pat in patterns:
        if re.search(pat, flag):
            return name
    return "uncategorized long-tail"


def main() -> int:
    db = Path(
        sys.argv[1]
        if len(sys.argv) > 1
        else Path.home() / ".local/share/pallama/pallama.db"
    )
    if not db.is_file():
        print(f"no store at {db}", file=sys.stderr)
        return 1
    src = "\n".join(p.read_text() for p in EMISSION_SOURCES)
    for tag, kind, flags in load_flags(db):
        wired = [f for f in flags if f in src]
        tail = [f for f in flags if f not in src]
        buckets: dict[str, list[str]] = {}
        for f in tail:
            buckets.setdefault(classify(f, BUCKETS.get(kind, [])), []).append(f)
        print(f"\n== {tag} ({kind}) — {len(flags)} flags probed")
        print(f"   wired first-class: {len(wired)}")
        for name in sorted(buckets, key=lambda n: -len(buckets[n])):
            ex = ", ".join(buckets[name][:4])
            print(f"   {name}: {len(buckets[name])}  (e.g. {ex})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
