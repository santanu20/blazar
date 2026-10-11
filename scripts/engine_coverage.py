#!/usr/bin/env python3
"""Engine capability coverage audit: one documented disposition for every
engine-advertised flag, anchored to emission truth instead of source-literal
grepping (the old method drifted within one release).

Disposition pipeline (first match wins):
  certified                 first-class AND backed by a live-engine receipt
                            in scripts/certification_receipts.json (a
                            benchmark or end-to-end run against a real
                            model on this hardware). Emission alone proves
                            the argv; the receipt proves the behavior.
  first-class               Blazar pins the flag from config knobs or
                            derivations — it appears in
                            scripts/first_class_flags.json, which is
                            generated from LIVE emission (profile-compiler
                            fixtures + spawn-time argv builders; see
                            crates/blazar-core/src/knob_registry.rs,
                            crates/blazar-runtime/src/knob_registry.rs)
                            and pinned in both directions by
                            crates/blazar-runtime/tests/knob_registry.rs.
  intentionally-unsupported the gateway owns the flag outright: extra_args
                            REFUSES it (reserved sets mirrored below from
                            the Rust gates, which stay the source of
                            truth). Typical: loopback bind, port, model
                            identity, child auth.
  passthrough               no first-class knob, but reachable today via
                            extra_args (verbatim on llamacpp;
                            manifest-gated strict on mistralrs / sglang /
                            sdcpp / mlx; reserved-checked on whisper).
                            Grouped into keyword families for readability
                            only — the disposition is uniform.
  not-applicable            the flag is not a capability Blazar could
                            integrate: CLI ergonomics, alternative input
                            surfaces, or an alias of a flag the lane
                            already emits. Per-flag reasons live in
                            NOT_APPLICABLE below and render in the doc.
  unverified                no verified way to reach the flag through
                            Blazar: either the lane has no
                            emission-truth registry entry yet, or the
                            flag is not first-class AND the lane has no
                            extra_args surface. Said out loud, not
                            guessed.

Policy: promotion from passthrough to a first-class knob requires
workload evidence — a receipt in scripts/certification_receipts.json
linking the flag to a live-engine run. Emission tests gate the "how",
receipts gate the "why".

A flag that is BOTH reserved and in the registry (e.g. mistral.rs
--max-model-len) lands first-class: Blazar derives it; the reserved
aspect only blocks duplicate user overrides.

Non-flag scope dispositions (documented, not flag-level):
  - SGLang router / disaggregated-PD tier: Blazar integrates the server
    launch path; router/disagg flags that the installed server advertises
    are passthrough-reachable where the manifest gates allow. A dedicated
    router tier is a scope decision, not a wiring gap.
  - sdcpp component settings: certification-pending — knob emission is
    pinned by the registry, per-component behavior against real models is
    tracked by the benchmark harness, not this audit.

Modes:
  (default)   human-readable matrix to stdout
  --json      machine-readable dump (disposition per flag)
  --markdown  regenerate docs/engine-coverage.md (byte-stable, no
              timestamps — the --check gate compares bytes)
  --refresh   snapshot the live store's engine manifests into
              scripts/manifest_flag_fixtures.json, then regenerate the
              doc from the new fixtures (commit both together)
  --check     CI gate: fixtures + registry + generated doc are present,
              valid, and mutually consistent (doc matches regeneration)

Store discovery for --refresh: $BLAZAR_DB, else XDG-style data dir
(XDG_DATA_HOME, falling back to the platform default), matching
crates/blazar-core/src/dirs.rs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REGISTRY = REPO / "scripts" / "first_class_flags.json"
FIXTURES = REPO / "scripts" / "manifest_flag_fixtures.json"
RECEIPTS_FILE = REPO / "scripts" / "certification_receipts.json"
DOC = REPO / "docs" / "engine-coverage.md"
DOC_HEADER = "# Engine capability coverage"

# Reserved sets mirrored from the Rust extra_args gates. The Rust consts
# are the source of truth; update this mirror in the same change that
# touches them (file:line anchors below).
RESERVED: dict[str, list[str]] = {
    # profile.rs, extra_args gate in the mistralrs compile path
    "mistralrs": [
        "--host",
        "--port",
        "-m",
        "-f",
        "--mmproj",
        "--max-model-len",
        "-np",
        "--max-seqs",
        "--max-num-batched-tokens",
        "--paged-attn",
        "--pa-memory-fraction",
        "--no-ui",
    ],
    # profile.rs, SGLANG_RESERVED_FLAGS (child-auth + derived pins)
    "sglang": [
        "--model-path",
        "--host",
        "--port",
        "--api-key",
        "--served-model-name",
        "--context-length",
        "--mem-fraction-static",
        "--cpu-offload-gb",
        "--enable-lora",
        "--lora-paths",
    ],
    # profile.rs, RESERVED in the sdcpp extra_args gate (component set)
    "sdcpp": [
        "--listen-ip",
        "--listen-port",
        "--diffusion-model",
        "--model",
        "--vae",
        "--llm",
        "--llm_vision",
        "--t5xxl",
        "--clip_l",
        "--clip_g",
        "--clip_vision",
        "--qwen2vl_vision",
        "-t",
        "--threads",
    ],
    # profile.rs, mlx_extra_args RESERVED (model + loopback listen pins)
    "mlx": ["--model", "--host", "--port"],
    # whisper.rs, RESERVED_WHISPER_FLAGS (loopback/model/VAD pins)
    "whisper": ["--host", "--port", "--model", "--vad", "--vad-model"],
    # llamacpp: no reserved set — extra_args ride verbatim.
}

# Lanes the emission-truth registry does not cover yet. knob_registry.rs
# owns llamacpp/mistralrs/sglang/sdcpp/mlx; whisper and piper ride
# knob_surface(). Empty today — kept so a future lane that composes its
# argv outside the registry is forced to say so here, out loud.
LANES_WITHOUT_REGISTRY: set[str] = set()

# Lanes with NO extra_args surface: argv is composed per call by Blazar,
# so a non-first-class flag is unreachable, not passthrough. Piper is a
# one-shot synthesizer spawn (no config knob, no user argv tail).
LANES_WITHOUT_EXTRA_ARGS: set[str] = {"piper"}

# Flags that are not capabilities Blazar could integrate — CLI ergonomics,
# alternative input surfaces, or long-form aliases of flags the lane
# already emits. The reason is rendered per flag in the doc. Tashkeel is
# the one deliberate hold-out: a real capability, but it needs a separate
# model asset and an Arabic voice, and this lane has no extra_args
# surface — revisit when tashkeel models enter the store.
NOT_APPLICABLE: dict[str, dict[str, str]] = {
    "piper": {
        "--config": ("config-file input competes with Blazar's composed argv"),
        "--debug": "interactive CLI ergonomics, not a synthesis capability",
        "--help": "interactive CLI ergonomics, not a synthesis capability",
        "--json-input": ("stdin JSON request surface; Blazar feeds plain text lines"),
        "--output_dir": (
            "directory batch output; Blazar always names explicit files "
            "(--output_file / --output_raw)"
        ),
        "--quiet": "long-form alias of the first-class -q the lane emits",
        "--tashkeel_model": (
            "Arabic diacritization; needs a separate model asset and an "
            "Arabic voice, and the lane has no extra_args surface"
        ),
    },
}

# Passthrough readability families (keyword grouping only). Order matters:
# first match wins.
FAMILIES: dict[str, list[tuple[str, str]]] = {
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
    "sdcpp": [
        (
            "component set (blazar-owned)",
            r"diffusion-model|vae|llm|clip|t5xxl|qwen2vl|vision|mmproj|te-",
        ),
        (
            "sampling/generation",
            r"cfg|step|seed|sampl|schedul|strength|guidance|shift|img|size|count",
        ),
        (
            "memory/backend placement",
            r"offload|vram|backend|device|thread|fa$|flash|cpu",
        ),
        ("gateway-owned transport", r"listen|host|port|webui|static|api-key"),
        ("input/output files", r"init-img|mask|output|prompt|control|ref|format|type"),
    ],
    "whisper": [
        ("transcription/translation", r"translat|language|detect|prompt"),
        ("decoding quality", r"beam|best|temperature|entropy|logprob|fallback"),
        (
            "segmentation/timing",
            r"timestamp|offset|duration|split|context|max.len|word",
        ),
    ],
    "mlx": [
        ("serving detail", r"cache|batch|concurr|template|adapter|draft|kv|quant"),
    ],
}

SCOPE_NOTES = {
    "sglang": (
        "Scope disposition: Blazar integrates the SGLang server launch path, "
        "not a separate router / disaggregated-PD tier. Router and disagg "
        "flags the installed server advertises are passthrough-reachable "
        "through extra_args where the manifest gate allows; a dedicated "
        "router tier is a documented scope decision, not a wiring gap."
    ),
    "sdcpp": (
        "Scope disposition: knob emission is pinned by the registry; "
        "per-component behavior against real diffusion models is "
        "certification-pending (tracked by the benchmark harness, not this "
        "audit)."
    ),
}


def default_store() -> Path:
    env = os.environ.get("BLAZAR_DB")
    if env:
        return Path(env)
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg) / "blazar" / "blazar.db"
    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library/Application Support/blazar/blazar.db"
    if sys.platform == "win32":
        return home / "AppData/Local/blazar/blazar.db"
    return home / ".local/share/blazar/blazar.db"


def load_registry() -> dict[str, list[str]]:
    doc = json.loads(REGISTRY.read_text())
    if doc.get("version") != 1:
        raise SystemExit(f"{REGISTRY}: unsupported schema version")
    lanes = doc["lanes"]
    for lane, flags in lanes.items():
        if not flags:
            raise SystemExit(f"{REGISTRY}: lane {lane} is empty")
        if flags != sorted(set(flags)):
            raise SystemExit(f"{REGISTRY}: lane {lane} is not sorted-unique")
    return lanes


def load_fixtures() -> list[dict[str, object]]:
    doc = json.loads(FIXTURES.read_text())
    if doc.get("version") != 1:
        raise SystemExit(f"{FIXTURES}: unsupported schema version")
    return doc["engines"]


def load_receipts() -> dict[str, dict[str, dict[str, str]]]:
    """Receipt index: lane -> flag -> {receipt, note}.

    A receipt is a repo-relative path to a live-engine run (benchmark
    artifact or research note) that exercised the flag end-to-end. It
    upgrades a first-class flag to certified; it can never upgrade a
    non-emitted flag.
    """
    doc = json.loads(RECEIPTS_FILE.read_text())
    if doc.get("version") != 1:
        raise SystemExit(f"{RECEIPTS_FILE}: unsupported schema version")
    index: dict[str, dict[str, dict[str, str]]] = {}
    for entry in doc["receipts"]:
        lane = str(entry["lane"])
        flag = str(entry["flag"])
        if lane in index and flag in index[lane]:
            raise SystemExit(f"{RECEIPTS_FILE}: duplicate {lane} {flag}")
        index.setdefault(lane, {})[flag] = {
            "receipt": str(entry["receipt"]),
            "note": str(entry.get("note", "")),
        }
    return index


def disposition(
    lane: str,
    flag: str,
    registry: dict[str, list[str]],
    receipts: dict[str, dict[str, dict[str, str]]],
) -> str:
    if lane in LANES_WITHOUT_REGISTRY:
        return "unverified"
    if flag in registry.get(lane, []):
        if flag in receipts.get(lane, {}):
            return "certified"
        return "first-class"
    if flag in NOT_APPLICABLE.get(lane, {}):
        return "not-applicable"
    if flag in RESERVED.get(lane, []):
        return "intentionally-unsupported"
    if lane in LANES_WITHOUT_EXTRA_ARGS:
        # No extra_args surface exists on this lane, so the flag is not
        # reachable through Blazar by any means — an honest "unverified",
        # not a phantom "passthrough".
        return "unverified"
    return "passthrough"


def family_of(lane: str, flag: str) -> str:
    for name, pat in FAMILIES.get(lane, []):
        if re.search(pat, flag):
            return name
    return "uncategorized long-tail"


@dataclass
class EngineAudit:
    tag: str
    kind: str
    total: int
    counts: dict[str, int]
    flags: dict[str, str]


def audit(
    registry: dict[str, list[str]],
    receipts: dict[str, dict[str, dict[str, str]]],
) -> list[EngineAudit]:
    out: list[EngineAudit] = []
    for engine in load_fixtures():
        lane = str(engine["kind"])
        raw_flags = engine["flags"]
        if not isinstance(raw_flags, list):
            raise SystemExit(f"fixture {engine['tag']}: flags must be a list")
        flags = sorted({str(f) for f in raw_flags})
        rows = {f: disposition(lane, f, registry, receipts) for f in flags}
        counts: dict[str, int] = {}
        for d in rows.values():
            counts[d] = counts.get(d, 0) + 1
        out.append(EngineAudit(str(engine["tag"]), lane, len(flags), counts, rows))
    return out


def render_markdown(
    audited: list[EngineAudit],
    registry: dict[str, list[str]],
    receipts: dict[str, dict[str, dict[str, str]]],
) -> str:
    lines = [
        DOC_HEADER,
        "",
        "Disposition of every engine-advertised flag against Blazar's",
        "emission-truth registry.",
        "",
        "- fixtures: `scripts/manifest_flag_fixtures.json` (engine manifest",
        "  snapshots, refreshed from a live store via `--refresh`)",
        "- registry: `scripts/first_class_flags.json` (live emission;",
        "  regenerate via `cargo run -p blazar-runtime --example",
        "  knob_registry`)",
        "- receipts: `scripts/certification_receipts.json` (live-engine runs",
        "  that upgrade first-class flags to certified)",
        "- gate: `python3 scripts/engine_coverage.py --check` fails CI when",
        "  the doc drifts from any input. No timestamps — the bytes are",
        "  the contract.",
        "",
        "| Disposition | Meaning |",
        "|---|---|",
        "| certified | first-class + live-engine receipt (behavior proven) |",
        "| first-class | Blazar pins it from a config knob or derivation |",
        "| passthrough | reachable via `extra_args`, no first-class knob |",
        "| intentionally-unsupported | gateway-owned; `extra_args` refuses it |",
        "| not-applicable | not a capability Blazar could integrate |",
        "| unverified | no verified reach through Blazar on this lane |",
        "",
        "Policy: promotion from passthrough to a first-class knob requires",
        "workload evidence — a receipt linking the flag to a live-engine",
        "run. Emission tests gate the *how*; receipts gate the *why*.",
        "",
    ]
    for engine in audited:
        lane = engine.kind
        tag = engine.tag
        counts = engine.counts
        flags = engine.flags
        lines.append(f"## {tag} ({lane}) — {engine.total} flags probed")
        lines.append("")
        summary = " · ".join(
            f"{d}: {counts.get(d, 0)}"
            for d in (
                "certified",
                "first-class",
                "passthrough",
                "intentionally-unsupported",
                "not-applicable",
                "unverified",
            )
        )
        lines.append(f"**{summary}**")
        lines.append("")
        if lane in registry:
            reg_n = len(registry[lane])
            lines.append(
                f"Registry lane `{lane}` pins {reg_n} first-class flags "
                "(compile + spawn-time emission)."
            )
            lines.append("")
        certified_hit = {
            f: receipts[lane][f] for f, d in sorted(flags.items()) if d == "certified"
        }
        if certified_hit:
            lines.append("### Certified against a live engine")
            lines.append("")
            for flag, meta in certified_hit.items():
                note = f" — {meta['note']}" if meta["note"] else ""
                lines.append(
                    f"- `{flag}` [`{meta['receipt']}`]({meta['receipt']}){note}"
                )
            lines.append("")
        by_fam: dict[str, list[str]] = {}
        for flag, d in sorted(flags.items()):
            if d == "passthrough":
                by_fam.setdefault(family_of(lane, flag), []).append(flag)
        if by_fam:
            lines.append("### Passthrough families")
            lines.append("")
            for fam in sorted(by_fam, key=lambda f: -len(by_fam[f])):
                members = by_fam[fam]
                preview = ", ".join(members[:6])
                more = "" if len(members) <= 6 else f" (+{len(members) - 6} more)"
                lines.append(f"- **{fam}** ({len(members)}): {preview}{more}")
            lines.append("")
        reserved_hit = [
            f for f, d in sorted(flags.items()) if d == "intentionally-unsupported"
        ]
        if reserved_hit:
            lines.append(
                "Intentionally-unsupported (gateway-owned, refused in "
                "`extra_args`): " + ", ".join(reserved_hit)
            )
            lines.append("")
        na_hit = [f for f, d in sorted(flags.items()) if d == "not-applicable"]
        if na_hit:
            lines.append("### Not applicable (deliberate, per-flag reasons)")
            lines.append("")
            for f in na_hit:
                lines.append(f"- {f}: {NOT_APPLICABLE[lane][f]}")
            lines.append("")
        if lane in LANES_WITHOUT_REGISTRY:
            lines.append(
                "Unverified: this lane has no emission-truth registry "
                "entry yet — every flag above is listed as unverified "
                "until it grows one (see knob_registry.rs)."
            )
            lines.append("")
        unverified_hit = [f for f, d in sorted(flags.items()) if d == "unverified"]
        if unverified_hit:
            lines.append(
                "Unverified (no first-class emission and no `extra_args` "
                "path on this lane): " + ", ".join(unverified_hit)
            )
            lines.append("")
        if lane in SCOPE_NOTES:
            lines.append(SCOPE_NOTES[lane])
            lines.append("")
    lines.append(
        "Fixtures refreshed from a live store via `--refresh`; they carry "
        "tag/kind/flags only (no machine paths). Dispositions are derived, "
        "never stored."
    )
    lines.append("")
    return "\n".join(lines)


def cmd_refresh(store: Path) -> int:
    if not store.is_file():
        print(f"no store at {store}", file=sys.stderr)
        return 1
    db = sqlite3.connect(store)
    rows = db.execute("select tag, kind, manifest from engines").fetchall()
    engines = []
    # Keep only flag-shaped tokens: the help parser occasionally emits
    # separators like "-----" which are not capabilities.
    flag_shape = re.compile(r"^-{1,2}[A-Za-z][\w.-]*$")
    for tag, kind, manifest in sorted(rows):
        try:
            raw = json.loads(manifest).get("flags", [])
        except json.JSONDecodeError:
            continue
        flags = sorted({f for f in raw if flag_shape.match(f)})
        engines.append({"tag": tag, "kind": kind, "flags": flags})
    if not engines:
        print(f"no engine manifests in {store}", file=sys.stderr)
        return 1
    FIXTURES.write_text(json.dumps({"version": 1, "engines": engines}, indent=2) + "\n")
    registry = load_registry()
    receipts = load_receipts()
    DOC.write_text(render_markdown(audit(registry, receipts), registry, receipts))
    print(f"{FIXTURES}: {len(engines)} engines")
    print(f"{DOC}: regenerated")
    return 0


def cmd_check() -> int:
    problems: list[str] = []
    try:
        registry = load_registry()
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
        print(f"FAIL registry: {e}", file=sys.stderr)
        return 1
    try:
        fixtures = load_fixtures()
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
        print(f"FAIL fixtures: {e}", file=sys.stderr)
        return 1
    if not fixtures:
        problems.append("fixtures hold no engines")
    try:
        receipts = load_receipts()
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
        print(f"FAIL receipts: {e}", file=sys.stderr)
        return 1
    for lane, flags in receipts.items():
        if lane not in registry:
            problems.append(f"receipt lane {lane} missing from registry")
            continue
        for flag, meta in flags.items():
            if flag not in registry[lane]:
                problems.append(
                    f"receipt {lane} {flag} is not emitted by the registry lane"
                )
            if not (REPO / meta["receipt"]).is_file():
                problems.append(
                    f"receipt {lane} {flag}: missing artifact {meta['receipt']}"
                )
    for engine in fixtures:
        lane = str(engine["kind"])
        if lane in LANES_WITHOUT_REGISTRY:
            continue
        if lane not in registry:
            problems.append(f"fixture kind {lane} missing from registry")
    if not DOC.is_file():
        problems.append(f"{DOC} missing (run --markdown)")
    else:
        want = render_markdown(audit(registry, receipts), registry, receipts)
        have = DOC.read_text()
        if have != want:
            problems.append(
                f"{DOC} is stale (run: python3 scripts/engine_coverage.py --markdown)"
            )
    if problems:
        for p in problems:
            print(f"FAIL {p}", file=sys.stderr)
        return 1
    print("engine coverage: fixtures, registry and doc are consistent")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument(
        "--json", action="store_true", help="machine-readable disposition dump"
    )
    ap.add_argument(
        "--markdown", action="store_true", help="regenerate docs/engine-coverage.md"
    )
    ap.add_argument(
        "--refresh",
        metavar="STORE",
        nargs="?",
        const=default_store(),
        help="snapshot live store manifests into fixtures",
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="CI gate: fixtures/registry/doc consistency",
    )
    args = ap.parse_args()

    if args.check:
        return cmd_check()
    if args.refresh is not None:
        return cmd_refresh(Path(args.refresh))

    registry = load_registry()
    receipts = load_receipts()
    audited = audit(registry, receipts)
    if args.markdown:
        DOC.write_text(render_markdown(audited, registry, receipts))
        print(f"{DOC}: regenerated")
        return 0
    if args.json:
        dump = [engine.__dict__ for engine in audited]
        print(json.dumps({"version": 1, "engines": dump}, indent=2))
        return 0
    for engine in audited:
        counts = engine.counts
        print(f"\n== {engine.tag} ({engine.kind}) — {engine.total} flags probed")
        for d in (
            "certified",
            "first-class",
            "passthrough",
            "intentionally-unsupported",
            "not-applicable",
            "unverified",
        ):
            print(f"   {d}: {counts.get(d, 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
