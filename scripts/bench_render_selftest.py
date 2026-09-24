"""Self-test for the bench publication renderer (no GPU, stdlib only).

Exercises the campaign-scoped findings/frontier/index paths that a
media-only artifact cannot reach: multi-level concurrency verdicts,
findings backing vs carried-over fallback, index idempotency, and the
full/sparse report renders. Run from the repo root:

    python3 scripts/bench_render_selftest.py

Exits nonzero on any regression.
"""

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

if not Path("scripts/bench_matrix.py").exists():
    sys.exit("run from the repo root (scripts/bench_matrix.py not found)")

spec = importlib.util.spec_from_file_location("bench_matrix", "scripts/bench_matrix.py")
bm = importlib.util.module_from_spec(spec)
sys.modules["bench_matrix"] = bm
spec.loader.exec_module(bm)


def rec(provider, params=None, tag="b11147-cuda", **kw):
    r = {
        "key": f"{tag}-{provider}-{json.dumps(params or {}, sort_keys=True)}",
        "tag": tag,
        "kind": "llamacpp",
        "provider": provider,
        "params": params or {},
        "model": "Qwen3.5-9B",
        "blazar_version": "blazar 0.11.0",
        "measured_at": "2026-09-24T12:00:00Z",
        "loadavg_5m": 1.0,
        "power_state": "AC",
    }
    r.update(kw)
    return r


# --- conc fixtures: blazar plateau at C=4 (gain 4->8 <10%), ollama 2 levels only, direct still gaining
conc = [
    rec(
        "conc-blazar",
        {"conc": 1, "rounds": 3},
        conc_level=1,
        conc_rounds=3,
        conc_ok=1,
        sys_tps=40.0,
        sum_stream_tps=40.2,
        conc_ttft_p99_ms=300,
        itl_p99_ms=25,
        conc_wall_s=3.2,
    ),
    rec(
        "conc-blazar",
        {"conc": 2, "rounds": 3},
        conc_level=2,
        conc_rounds=3,
        conc_ok=2,
        sys_tps=76.0,
        sum_stream_tps=78.0,
        conc_ttft_p99_ms=420,
        itl_p99_ms=27,
        conc_wall_s=3.4,
    ),
    rec(
        "conc-blazar",
        {"conc": 4, "rounds": 3},
        conc_level=4,
        conc_rounds=3,
        conc_ok=4,
        sys_tps=140.0,
        sum_stream_tps=150.0,
        conc_ttft_p99_ms=700,
        itl_p99_ms=30,
        conc_wall_s=3.7,
    ),
    rec(
        "conc-blazar",
        {"conc": 8, "rounds": 3},
        conc_level=8,
        conc_rounds=3,
        conc_ok=8,
        sys_tps=148.0,
        sum_stream_tps=160.0,
        conc_ttft_p99_ms=1500,
        itl_p99_ms=38,
        conc_wall_s=6.9,
    ),
    rec(
        "conc-ollama",
        {"conc": 1, "rounds": 3},
        conc_level=1,
        conc_rounds=3,
        conc_ok=1,
        sys_tps=35.0,
        sum_stream_tps=35.1,
        conc_ttft_p99_ms=330,
        itl_p99_ms=28,
        conc_wall_s=3.7,
    ),
    rec(
        "conc-ollama",
        {"conc": 2, "rounds": 3},
        conc_level=2,
        conc_rounds=3,
        conc_ok=2,
        sys_tps=60.0,
        sum_stream_tps=62.0,
        conc_ttft_p99_ms=500,
        itl_p99_ms=33,
        conc_wall_s=4.3,
    ),
    rec(
        "conc-direct",
        {"conc": 1},
        conc_level=1,
        conc_ok=1,
        sys_tps=41.0,
        sum_stream_tps=41.0,
        conc_ttft_p99_ms=290,
        itl_p99_ms=24,
    ),
    rec(
        "conc-direct",
        {"conc": 2},
        conc_level=2,
        conc_ok=2,
        sys_tps=80.0,
        sum_stream_tps=81.0,
        conc_ttft_p99_ms=400,
        itl_p99_ms=26,
    ),
    rec(
        "conc-direct",
        {"conc": 4},
        conc_level=4,
        conc_ok=4,
        sys_tps=155.0,
        sum_stream_tps=158.0,
        conc_ttft_p99_ms=680,
        itl_p99_ms=29,
    ),
]
table, verdicts = bm.conc_frontier(conc)
assert "| C |" in table and table.count("\n") >= 9, table
assert any(
    "blazar gateway" in v and "plateaus at C=8" in v and "148.0" in v for v in verdicts
), verdicts
assert any("ollama" in v.lower() and "insufficient levels" in v for v in verdicts), (
    verdicts
)
assert any("direct engine" in v and "still gaining at C=4" in v for v in verdicts), (
    verdicts
)
# eff: C=2 blazar = 76/(2*40) = 95%
assert "95%" in table, table
print(
    "conc_frontier: table rows OK, plateau/insufficient/gaining verdicts OK, eff 95% OK"
)

# --- text_findings: full backing fixture
gw = rec(
    "blazar",
    {"ctx": 16384, "np": 1},
    decode_tps_p50=41.2,
    ttft_ms_p50=280,
    ttft_ms_p99=340,
    itl_p50_ms=23.5,
    itl_p99_ms=30.0,
    prefill_tps_cold=420.0,
    prefill_tps_cached=2600.0,
    child_argv=["llama-server", "-np", "1", "--ctx-size", "16384"],
    daemon_boot_s=0.4,
    cold_ttft_ms=2900.0,
)
d16 = rec(
    "direct",
    {"ctx": 16384, "np": 1},
    decode_tps_p50=41.6,
    ttft_ms_p50=275,
    ttft_ms_p99=335,
    itl_p50_ms=23.4,
    itl_p99_ms=29.5,
    prefill_tps_cold=418.0,
    prefill_tps_cached=2580.0,
)
base4096 = rec("direct", {"ctx": 4096, "np": 1}, decode_tps_p50=42.0)
spec_var = rec("direct", {"ctx": 4096, "np": 1, "spec": "ngram"}, decode_tps_p50=34.0)
kv_var = rec("direct", {"ctx": 4096, "np": 1, "kv": "q8_0"}, decode_tps_p50=41.8)
greedy = rec("greedy", {}, exact_matches=20, prompts=20, ratio_mean=1.0, ratio_min=1.0)
greedy_gw = rec(
    "greedy_gw", {}, exact_matches=20, prompts=20, ratio_mean=1.0, ratio_min=1.0
)
mistral = rec(
    "blazar",
    {"ctx": 8192, "np": 1},
    tag="v0.9.3",
    decode_tps_p50=22.0,
    child_argv=["mistralrs-server"],
    note="pa-off",
)
gw4 = rec(
    "blazar",
    {"ctx": 16384, "np": 4},
    decode_tps_p50=38.0,
    ttft_ms_p50=300,
    itl_p50_ms=26.0,
    itl_p99_ms=34.0,
    prefill_tps_cold=410.0,
    prefill_tps_cached=2500.0,
    child_argv=["llama-server", "-np", "4", "--ctx-size", "16384"],
)

# --- tools lane: table + F12 backing + scoped marker
tools_ok = rec(
    "tools",
    {"tools": True},
    tools_scenarios=6,
    tools_wellformed=5,
    tools_selection=5,
    tools_args_valid=4,
    tools_false_positive=False,
    tools_ttft_p50_ms=412.0,
)
tools_ollama = rec(
    "tools-ollama",
    {"tools": True},
    tag="ollama-host",
    tools_scenarios=6,
    tools_wellformed=4,
    tools_selection=4,
    tools_args_valid=3,
    tools_false_positive=True,
    tools_ttft_p50_ms=655.0,
)
tools_err = rec("tools", {"tools": True}, tag="v0.9.3", error="engine refused tools")
ttab = bm.tools_table([tools_ok, tools_ollama, tools_err])
assert "| Runtime | scenarios |" in ttab and "5/6" in ttab and "5/5" in ttab, ttab
assert "ollama - qwen3.5:9b" in ttab and "4/5" in ttab, ttab
assert "v0.9.3" not in ttab, "error rows must not appear in the tools table"
assert bm.tools_table([]) == "_Not measured._"
out_t = bm.text_findings([tools_ok, tools_ollama])
backed_t = [b for b, _ in out_t if b]
assert any(
    "Tool-call quality" in b and "5/5" in b and "ollama: selection 4/5" in b
    for b in backed_t
), backed_t
sparse_t = bm.text_findings([])
assert any(c and "tools verbatim" in c for b, c in sparse_t if not b), (
    "tools carried text missing"
)
print(
    "tools_table + F12: rows, ollama compare, error exclusion, empty marker, carried OK"
)


recs = [
    gw,
    gw4,
    d16,
    base4096,
    spec_var,
    kv_var,
    greedy,
    greedy_gw,
    *conc,
    mistral,
    tools_ok,
]
out = bm.text_findings(recs)
backed = [b for b, _ in out if b]
carried = [c for b, c in out if not b]
print(f"text_findings full: backed={len(backed)} carried={len(carried)} (expect 8/0)")
assert len(backed) == 8 and len(carried) == 0, [x[:60] for x in backed]
assert any("noise" in b for b in backed), backed
assert any("mistral" in b.lower() for b in backed), backed
assert any("net loss" in b.lower() for b in backed), backed
assert any("6" in b and "x" in b.lower() for b in backed), backed  # prefill ratio ~6.2x
assert any("Tool-call quality" in b for b in backed), backed

# sparse: drop pair+greedy+variants -> F1/F5/F6 carried
sparse = [gw, *conc]
out2 = bm.text_findings(sparse)
backed2 = [b for b, _ in out2 if b]
carried2 = [c for b, c in out2 if not b]
print(f"text_findings sparse: backed={len(backed2)} carried={len(carried2)}")
assert len(carried2) >= 3, carried2
print("text_findings: full-backing + carried-fallback OK")

# --- index idempotency
with tempfile.TemporaryDirectory() as td:
    adir = Path(td) / "20260924-text-frontier"
    bm.update_campaign_index(adir, recs, "0.11.0")
    one = (adir.parent / "INDEX.md").read_text()
    bm.update_campaign_index(adir, recs, "0.11.0")
    two = (adir.parent / "INDEX.md").read_text()
    assert (
        one.count("20260924-text-frontier") == two.count("20260924-text-frontier") == 1
    ), "index not idempotent"
    print("update_campaign_index: idempotent (1 row after 2 calls) OK")

# --- full report render smoke on synthetic recs
with tempfile.TemporaryDirectory() as td:
    adir = Path(td) / "synthetic"
    adir.mkdir()
    (adir / "cells.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    outp = Path(td) / "REPORT.md"
    bm.write_publication_report(recs, adir, outp)
    md = outp.read_text()
    assert "## Findings (this campaign)" in md
    assert "Carried-over findings" not in md, (
        "carried section must be omitted when all findings are backed"
    )
    assert "Concurrency frontier" in md and "plateaus at C=8" in md
    assert (
        md.count("_Not measured in this campaign") >= 3
    )  # ppl/coldstart/idle etc unmeasured
    print(
        f"write_publication_report: synthetic full render OK ({md.count(chr(10))} lines)"
    )
    # sparse render: carried-over section MUST appear with provenance note
    outp2 = Path(td) / "SPARSE.md"
    bm.write_publication_report([gw], adir, outp2)
    md2 = outp2.read_text()
    assert "Carried-over findings" in md2 and "no receipt" in md2, md2[
        md2.find("Carried") - 50 : md2.find("Carried") + 200
    ]
    print("write_publication_report: sparse render carries unbacked findings OK")
# direct scorer pin: a correct tool-call outcome must score all-True
# (guards the set-subset type error class that crashed a live cell)
_scen = {
    "name": "x",
    "prompt": "p",
    "expected_fn": "get_weather",
    "required_args": ["city"],
}
_out = {
    "status": 200,
    "finish_reason": "tool_calls",
    "content": "",
    "calls": [{"name": "get_weather", "args": '{"city": "Tokyo"}'}],
}
_s = bm.score_tools_scenario(_scen, _out)
assert _s["wellformed"] and _s["selection"] and _s["args_valid"], _s

# receipts ship in-repo: the writer must strip the invoking user's
# home dir to `~` (a hard-coded /home/<user> path pins the artifact
# to one box), and leave path-free rows untouched
_home = str(Path.home())
assert _home != "/"
_row = json.dumps({"argv": [f"{_home}/.local/share/blazar/engines/b1/srv"]})
_san = bm.portable_path(_row)
assert f"{_home}" not in _san and "~/.local/share/blazar" in _san, _san
assert bm.portable_path("no paths here") == "no paths here"

print("ALL FIXTURE CHECKS GREEN")
