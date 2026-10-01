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
import struct
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

# --- F6e: 'ok streams' must decompose as 'ok of rounds x conc', never '12/4'
conc_cell = rec(
    "conc-blazar",
    {"conc": 4, "rounds": 3},
    conc_level=4,
    conc_ok=12,
    conc_rounds=3,
    sys_tps=67.5,
    sum_stream_tps=433.1,
    conc_wall_s=22.8,
    ttft_max_ms=4205,
    conc_ttft_p99_ms=4204,
    itl_p99_ms=81.3,
)
ctab = bm.conc_table([conc_cell])
assert "12 of 3x4" in ctab and "12/4" not in ctab, ctab
ftab, _ = bm.conc_frontier([conc_cell])
assert "12 of 3x4" in ftab, ftab
print("conc ok-streams: '12 of 3x4' decomposition in table + frontier OK")

# --- F6a: ollama reference row must carry the not-comparable caveat
ollama_speed = rec(
    "ollama",
    {},
    tag="ollama-host",
    ollama_model="qwen3.5:9b",
    decode_tps_p50=40.6,
    reference_note=(
        "reference serves 'qwen3.5:9b' (matrix model differs - t/s NOT comparable)"
    ),
)
stab = bm.speed_table([ollama_speed])
assert "NOT comparable" in stab and "matrix model differs" in stab, stab
stab_clean = bm.speed_table(
    [rec("ollama", {}, tag="ollama-host", ollama_model="same", decode_tps_p50=40.0)]
)
assert "NOT comparable" not in stab_clean, stab_clean
print("speed_table: reference_note caveat rendered when present, absent otherwise OK")

# --- watt efficiency: tok/s per W column, value or '-' when unmeasured
assert "tok/s per W" in stab_clean, stab_clean
watted = bm.speed_table(
    [
        rec(
            "ollama",
            {},
            tag="ollama-host",
            ollama_model="same",
            decode_tps_p50=40.0,
            decode_tps_per_w=0.312,
        )
    ]
)
assert "0.312" in watted, watted
wattless_row = next(ln for ln in stab_clean.splitlines() if ln.startswith("| ollama "))
assert wattless_row.split("|")[4].strip() == "-", (
    stab_clean,
    "unmeasured power must render '-' in the tok/s-per-W column, not a number",
)
hdr_cells = stab_clean.splitlines()[0].split("|")
assert len(hdr_cells) == len(wattless_row.split("|")), stab_clean
print("speed_table: tok/s per W column (value rendered, '-' unmeasured) OK")

# --- F6b: cold-start table drops cells with no cold fields and labels configs
cold_ok = rec(
    "blazar",
    {},
    daemon_boot_s=0.4,
    cold_first_request_s=3.9,
    cold_ttft_ms=2900.0,
    load_s=3.5,
    rss_peak_mib=6400.0,
)
cold_cfg = rec(
    "blazar",
    {"config": "pa_off"},
    daemon_boot_s=0.4,
    cold_ttft_ms=2950.0,
    load_s=3.6,
    rss_peak_mib=6400.0,
)
cold_none = rec(
    "blazar",
    {"conc": 8, "config": "conc_default"},
    rss_peak_mib=0.0,
)
cold_tab = bm.coldstart_table([cold_ok, cold_cfg, cold_none])
assert "first request (cold engine load)" in cold_tab
assert "(pa_off)" in cold_tab, "config variants must be labeled, not anonymous dups"
blazar_rows = [ln for ln in cold_tab.splitlines() if "blazar gateway" in ln]
assert len(blazar_rows) == 2, cold_tab
print("coldstart_table: cold-field filter + config labeling OK")

# --- F6d: ppl provenance column (own vs borrowed tool)
ppl_own = rec("ppl", {}, perplexity=17.35, ppl_err=0.92, ppl_tool="own")
ppl_borrowed = rec(
    "ppl",
    {},
    tag="v0.9.4",
    kind="mistralrs",
    perplexity=17.35,
    ppl_err=0.92,
    ppl_tool="borrowed:llama-b11202",
)
ptab = bm.ppl_table([ppl_own, ppl_borrowed])
assert "| ppl tool |" in ptab and "borrowed (llama-b11202)" in ptab, ptab
assert ptab.count("| own |") == 1, ptab
print("ppl_table: tool provenance column OK")

# --- F3 forensics: reshape table surfaces the daemon tail, ANSI-stripped
reshape_tail = rec(
    "reshape",
    {"reshape": True},
    reshape_observed=False,
    slots_from=2,
    slots_to=None,
    requests_before=116,
    requests_after=0,
    requests_failed=0,
    daemon_tail=[
        "2026-09-28T18:10:40Z \x1b[32mINFO\x1b[0m supervisor: slot pressure 2/2 sustained",
        "2026-09-28T18:10:41Z \x1b[32mINFO\x1b[0m evict: terminating child and releasing the slot",
    ],
)
rtail = bm.reshape_table([reshape_tail])
assert "<details>" in rtail and "slot pressure 2/2" in rtail, rtail
assert "\x1b" not in rtail, "ANSI escapes must not leak into the markdown"
print("reshape_table: daemon tail details block, ANSI-stripped OK")

# --- F3 forensics: a poller that never saw the slot shape renders a
# detection VOID, not a product "NO" (20260928-gguf-full: 150/150 samples
# read None against a guessed /api/ps schema)
reshape_void = rec(
    "reshape",
    {"reshape": True},
    reshape_observed=False,
    slots_from=None,
    slots_to=None,
    detection_error="api/ps never reported a slots reading for the model",
    daemon_tail=[],
)
rvoid = bm.reshape_table([reshape_void])
assert "NO (detection void)" in rvoid, rvoid
print("reshape_table: detection void distinguished from a true NO OK")

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
greedy_ollama = rec(
    "greedy_ollama",
    {},
    exact_matches=9,
    prompts=20,
    ratio_mean=0.812,
    ratio_min=0.514,
    ollama_model="qwen3.5:9b",
)

# --- E3: ollama greedy reference renders its run-to-run determinism row
gollama_tab = bm.greedy_table(
    [
        rec("greedy", {}, exact_matches=20, prompts=20, ratio_mean=1.0, ratio_min=1.0),
        greedy_ollama,
    ]
)
assert "run-to-run, temp 0" in gollama_tab, gollama_tab
assert "qwen3.5:9b" in gollama_tab and "9/20" in gollama_tab, gollama_tab
assert "0.812" in gollama_tab and "0.514" in gollama_tab, gollama_tab
print("greedy_table: ollama run-to-run reference row OK")


# --- E4: media image rows aggregate identical configs (median+n) and
# label the A/B knob so default/fa_off/vae_tiling/sage rows are distinct
def media_img_rec(cfg, med, lo=None, hi=None, cold=60.0):
    return rec(
        "media-image",
        {
            "size": "512x512",
            "steps": [4],
            "runs": 3,
            **({"config": cfg} if cfg else {}),
        },
        tag="master-920-2f88688",
        model="qwen-image-2.1",
        dims_seen=[512, 512],
        total_s_median=med,
        total_s_min=lo if lo is not None else med - 0.4,
        total_s_max=hi if hi is not None else med + 0.4,
        cold_request_s=cold,
    )


mtab = bm.media_table(
    [
        media_img_rec(None, 47.0),
        media_img_rec(None, 49.0),
        media_img_rec("fa_off", 52.2),
    ]
)
assert mtab.count("| image -") == 2, mtab
assert "(n=2)" in mtab and "48" in mtab, mtab
assert "[fa_off]" in mtab, mtab
print("media_table: identical-config aggregation (median+n) + knob labels OK")

# --- E4/G1: a knob the profiler skipped renders inert, not a fake Δ%
axes = bm.media_axes_table(
    [
        rec(
            "media-image",
            {"size": "512x512", "steps": [4], "runs": 3},
            tag="master-920-2f88688",
            total_s_median=47.3,
            total_s_min=46.9,
            total_s_max=49.2,
            cold_request_s=60.4,
        ),
        rec(
            "media-image",
            {"size": "512x512", "steps": [4], "runs": 3, "config": "sage_attn_on"},
            tag="master-920-2f88688",
            total_s_median=45.6,
            total_s_min=45.4,
            total_s_max=45.6,
            cold_request_s=57.6,
            note="config knob inert on this box: profiler skipped sage_attn "
            "(SageAttention requires a CUDA device) — measures the default "
            "posture",
        ),
    ]
)
assert "| inert |" in axes, axes
assert "profiler skipped sage_attn" in axes, axes
assert "-3.6%" not in axes, "inert row must not imply a sage effect"
print("media_axes_table: inert-knob row + note OK")
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

# --- reshape lane: table + F13 backing + scoped marker
reshape_ok = rec(
    "reshape",
    {"reshape": True},
    reshape_observed=True,
    slots_from=1,
    slots_to=8,
    time_to_reshape_s=142.0,
    requests_before=180,
    requests_after=95,
    ttft_p50_before_ms=11800.0,
    ttft_p50_after_ms=940.0,
    sys_tps_before=39.4,
    sys_tps_after=91.2,
    requests_failed=0,
    timeline_samples=150,
)
reshape_no = rec(
    "reshape",
    {"reshape": True},
    tag="v0.9.3",
    reshape_observed=False,
    slots_from=2,
    slots_to=None,
    requests_before=210,
    requests_after=0,
    requests_failed=0,
    timeline_samples=150,
)
rtab = bm.reshape_table([reshape_ok, reshape_no])
assert "| Runtime | reshape | slots |" in rtab and "1->8" in rtab and "142" in rtab, (
    rtab
)
assert "11800->940" in rtab and "39->91" in rtab and "| 0 |" in rtab, rtab
assert "NO" in rtab, "unobserved reshape must render NO"
assert bm.reshape_table([]) == "_Not measured._"
out_r = bm.text_findings([reshape_ok])
backed_r = [b for b, _ in out_r if b]
assert any(
    "Adaptive reshape" in b and "1->8" in b and "0 dropped" in b for b in backed_r
), backed_r
sparse_r = bm.text_findings([])
assert any(
    c and "reshape under sustained concurrency" in c for b, c in sparse_r if not b
), "reshape carried text missing"
print("reshape_table + F13: transition row, NO case, empty marker, carried OK")


recs = [
    reshape_ok,
    gw,
    gw4,
    d16,
    base4096,
    spec_var,
    kv_var,
    greedy,
    greedy_gw,
    greedy_ollama,
    *conc,
    mistral,
    tools_ok,
]
out = bm.text_findings(recs)
backed = [b for b, _ in out if b]
carried = [c for b, c in out if not b]
print(f"text_findings full: backed={len(backed)} carried={len(carried)} (expect 8/0)")
assert len(backed) == 9 and len(carried) == 0, [x[:60] for x in backed]
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
    (adir / "campaign_cmd.txt").write_text(
        "--model qwen3-1.7b --engines sglang-0.5.19\n"
    )
    outp = Path(td) / "REPORT.md"
    bm.write_publication_report(
        recs, adir, outp, "--model qwen3-1.7b --engines sglang-0.5.19"
    )
    md = outp.read_text()
    assert "## Findings (this campaign)" in md
    assert "Carried-over findings" not in md, (
        "carried section must be omitted when all findings are backed"
    )
    assert "Concurrency frontier" in md and "plateaus at C=8" in md
    assert (
        md.count("_Not measured in this campaign") >= 3
    )  # ppl/coldstart/idle etc unmeasured
    # exec summary must be a bullet list, one fact per line
    es = md.split("## Executive summary", 1)[1].split("##", 1)[0]
    es_lines = [ln for ln in es.splitlines() if ln.strip()]
    assert es_lines and all(ln.startswith("- ") for ln in es_lines), es_lines
    # Reproduce must carry the actual campaign invocation, not a generic one
    rep = md.split("## Reproduce", 1)[1].split("```bash", 1)[1].split("```", 1)[0]
    assert "--model qwen3-1.7b --engines sglang-0.5.19" in rep, rep
    # no accidental double-blank runs anywhere in the render
    assert "\n\n\n" not in md, "double blank-line run in render"
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

# --- sglang lane inclusion: dynamic, store-driven (no hardcoded box
# state). A safetensors DIR row flips the lane benchable; GGUF-only
# stores exclude it with the honest reason text.
import sqlite3 as _sq  # noqa: E402 (test-scoped, next to its fixture)


def _mini_store(td, model_rows, engine_rows):
    con = _sq.connect(Path(td) / "blazar.db")
    con.execute("CREATE TABLE engines (tag TEXT, kind TEXT)")
    con.execute(
        "CREATE TABLE models (path TEXT, name TEXT, mmproj_path TEXT, components TEXT)"
    )
    con.executemany("INSERT INTO engines VALUES (?,?)", engine_rows)
    con.executemany("INSERT INTO models VALUES (?,?,?,?)", model_rows)
    con.commit()
    con.close()


with tempfile.TemporaryDirectory() as td:
    st_dir = Path(td) / "qwen3-1.7b.d"
    st_dir.mkdir()
    (st_dir / "model-00001-of-00001.safetensors").write_bytes(b"x" * 16)
    gguf = Path(td) / "text.gguf"
    gguf.write_bytes(b"g" * 32)
    _mini_store(
        td,
        [(str(st_dir), "qwen3-1.7b", None, None), (str(gguf), "text9b", None, None)],
        [("sglang-0.5.19", "sglang"), ("b11202-cuda", "llamacpp")],
    )
    (Path(td) / "engines" / "sglang-0.5.19").mkdir(parents=True)
    (Path(td) / "engines" / "b11202-cuda").mkdir(parents=True)
    data = Path(td)
    assert bm.has_text_safetensors_model(data), "dir row must qualify"
    assert bm.engine_exclusion_reasons(data) == {}, "no exclusion with a dir row"
    engs = bm.load_engines(data)
    tags = {e.tag: e.kind for e in engs}
    assert tags.get("sglang-0.5.19") == "sglang", tags
    sg = next(e for e in engs if e.kind == "sglang")
    assert sg.server is None and sg.bench is None, "gateway-routed contract"
    print("sglang inclusion: dir row -> benchable gateway-routed Engine OK")

with tempfile.TemporaryDirectory() as td:
    gguf = Path(td) / "text.gguf"
    gguf.write_bytes(b"g" * 32)
    _mini_store(td, [(str(gguf), "text9b", None, None)], [("sglang-0.5.19", "sglang")])
    (Path(td) / "engines" / "sglang-0.5.19").mkdir(parents=True)
    data = Path(td)
    assert not bm.has_text_safetensors_model(data), "gguf-only store must not qualify"
    reasons = bm.engine_exclusion_reasons(data)
    assert "sglang" in reasons and "safetensors" in reasons["sglang"], reasons
    assert all(e.kind != "sglang" for e in bm.load_engines(data)), "must be dropped"
    print("sglang exclusion: gguf-only store -> dropped with reason OK")

# relative model path (product stores relative spellings)
with tempfile.TemporaryDirectory() as td:
    (Path(td) / "models" / "m.d").mkdir(parents=True)
    _mini_store(td, [("models/m.d", "m", None, None)], [])
    assert bm.has_text_safetensors_model(Path(td)), "relative dir row must resolve"
    print("sglang inclusion: relative dir row resolves OK")

# cold-cache file set: HF dirs expand to shard files (fadvise on a dir
# inode would leave every shard cached), GGUF files pass through
with tempfile.TemporaryDirectory() as td:
    d = Path(td) / "m.d"
    d.mkdir()
    (d / "a.safetensors").write_bytes(b"a")
    (d / "b.safetensors").write_bytes(b"b")
    (d / "config.json").write_bytes(b"c")
    mm = Path(td) / "proj.gguf"
    assert bm.cold_cache_files(d, mm) == [d / "a.safetensors", d / "b.safetensors", mm]
    f = Path(td) / "m.gguf"
    assert bm.cold_cache_files(f, None) == [f]
    assert bm.cold_cache_files(None, None) == []
    print("cold_cache_files: dir->shards, file passthrough, empty OK")


# GGUF hf-base-id derivation: mirror of the product's gguf.rs
# hf_base_model_id — general.basename + general.size_label -> org/repo.
# Synthetic GGUF v2 header (magic, version, tensor_count, kv_count, kvs).
def _synthetic_gguf(kvs: list[tuple[str, str]]) -> bytes:
    out = bytearray(b"GGUF")
    out += struct.pack("<I", 2)
    out += struct.pack("<Q", 0)  # tensor count
    out += struct.pack("<Q", len(kvs))
    for key, value in kvs:
        kb, vb = key.encode(), value.encode()
        out += struct.pack("<Q", len(kb)) + kb
        out += struct.pack("<I", 8)  # string value type
        out += struct.pack("<Q", len(vb)) + vb
    return bytes(out)


with tempfile.TemporaryDirectory() as td:
    g = Path(td) / "m.gguf"
    g.write_bytes(
        _synthetic_gguf(
            [
                ("general.architecture", "qwen35"),
                ("general.basename", "Qwen_Qwen3.5-9B"),
                ("general.size_label", "9B"),
            ]
        )
    )
    assert bm.gguf_hf_base_model_id(g) == "Qwen/Qwen3.5-9B", "size label already suffix"
    g3 = Path(td) / "m3.gguf"
    g3.write_bytes(_synthetic_gguf([("general.basename", "nosplit")]))
    assert bm.gguf_hf_base_model_id(g3) is None, "no org/repo split -> None"
    g4 = Path(td) / "m4.gguf"
    g4.write_bytes(b"not-gguf")
    assert bm.gguf_hf_base_model_id(g4) is None, "bad magic -> None"
    print("gguf_hf_base_model_id: derive, unsplittable, bad-magic OK")

# load_done attempts cap: a key with MAX_CELL_ATTEMPTS error rows becomes
# done (engine-reality crash stops retrying); fewer errors keep retrying;
# ok rows are done regardless.
with tempfile.TemporaryDirectory() as td:
    cp = Path(td) / "cells.jsonl"
    rows = [
        {"key": "crashed", "error": "sigkill"},
        {"key": "crashed", "error": "sigkill"},
        {"key": "flaky", "error": "once"},
        {"key": "good", "ttft_ms_p50": 5},
    ]
    cp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    d1 = bm.load_done(cp)
    assert "good" in d1, "ok row is done"
    assert "flaky" not in d1, "1 error row still retries"
    assert "crashed" not in d1, "2 error rows still retry (cap is 3)"
    cp.open("a").write(json.dumps({"key": "crashed", "error": "sigkill"}) + "\n")
    d2 = bm.load_done(cp)
    assert "crashed" in d2, "3 identical error rows cap the cell"
    print("load_done attempts cap: ok/1-err/2-err/3-err OK")

# manifest_flags: gateway-routed engines gate variant axes on the
# product's probed manifest flags — json parse keeps only long flags,
# missing manifest / missing db degrade to the empty set
with tempfile.TemporaryDirectory() as td:
    con = _sq.connect(Path(td) / "blazar.db")
    con.execute("CREATE TABLE engines (tag TEXT, kind TEXT, manifest TEXT)")
    mani = json.dumps(
        {"flags": ["--kv-cache-dtype", "--cuda-graph-max-bs", "short", 7]}
    )
    con.execute("INSERT INTO engines VALUES ('sglang-0.5.19','sglang',?)", (mani,))
    con.execute("INSERT INTO engines VALUES ('gone','sglang',NULL)")
    con.commit()
    con.close()
    got = bm.manifest_flags(Path(td), "sglang-0.5.19")
    assert got == {"--kv-cache-dtype", "--cuda-graph-max-bs"}, got
    assert bm.manifest_flags(Path(td), "gone") == set()
    assert bm.manifest_flags(Path(td) / "nope", "x") == set()
    print("manifest_flags: json/none/missing-db OK")

# build_local_corpus: selection must flow through git so gitignored
# notes (agent memory / scratch reports carrying operator credentials)
# and credential-pattern files stay out of bench artifacts; no git at
# all -> fail closed
import subprocess as sp  # noqa: E402 (test-scoped)
import unittest.mock as _mock  # noqa: E402 (test-scoped)

with tempfile.TemporaryDirectory() as td:
    repo = Path(td) / "r"
    (repo / "docs").mkdir(parents=True)
    (repo / "docs" / "a.md").write_text("alpha beta gamma " * 5000)
    (repo / "MEMORY.md").write_text("agent memory " * 200)
    (repo / "AGENTS.md").write_text("agent instructions " * 200)
    (repo / "secret.md").write_text("token ghp_" + "a" * 30 + " filler " * 2000)
    (repo / ".gitignore").write_text("MEMORY.md\n")
    sp.run(["git", "init", "-q", str(repo)], check=True)
    sp.run(
        [
            "git",
            "-C",
            str(repo),
            "add",
            "docs/a.md",
            "AGENTS.md",
            "secret.md",
            ".gitignore",
        ],
        check=True,
    )
    dest = Path(td) / "corpus.txt"
    assert bm.build_local_corpus(dest, repo=repo), "fixture corpus must build"
    text = dest.read_text()
    assert "alpha beta gamma" in text, "tracked text must be included"
    assert "agent memory" not in text, "gitignored MEMORY.md must be excluded"
    assert "agent instructions" not in text, "AGENTS.md must be excluded by name"
    assert "ghp_" not in text, "credential-pattern file must be excluded"
    with _mock.patch.object(bm.subprocess, "run", side_effect=OSError("no git")):
        assert not bm.build_local_corpus(dest), "git failure must fail closed"
    print("build_local_corpus: gitignore/denylist/credential-guard OK")

# demote_headings: appended-campaign chapters nest under their wrapper;
# headings inside fenced code blocks must NOT be shifted.
_demoted = bm.demote_headings(
    "# Title\n\ntext\n\n## Section\n\n```bash\n# not a heading\n```\n"
)
assert _demoted.startswith("## Title"), "top heading must gain one level"
assert "\n### Section" in _demoted, "section headings must gain one level"
assert "# not a heading" in _demoted, "fence content must stay untouched"
print("demote_headings: heading shift + fence immunity OK")

print("ALL FIXTURE CHECKS GREEN")
