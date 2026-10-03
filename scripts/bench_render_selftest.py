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
import time
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

# --- watt efficiency: net tok/s per W column, value or '-' when unmeasured
assert "tok/s per W (net)" in stab_clean, stab_clean
watted = bm.speed_table(
    [
        rec(
            "ollama",
            {},
            tag="ollama-host",
            ollama_model="same",
            decode_tps_p50=40.0,
            decode_tps_per_w_net=0.312,
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
print("speed_table: net tok/s per W column (value rendered, '-' unmeasured) OK")

# --- ctxcurve table: same net column, header/row cell-count alignment
ctx_tab = bm.ctxcurve_table(
    [
        {
            "provider": "ctxcurve-blazar",
            "tag": "b11147-cuda",
            "ctx": 4096,
            "decode_tps_p50": 30.0,
            "decode_tps_per_w_net": 0.2,
            "ttft_ms_p50": 120.0,
        }
    ]
)
assert "tok/s per W (net)" in ctx_tab and "0.200" in ctx_tab, ctx_tab
assert len(ctx_tab.splitlines()[0].split("|")) == len(
    ctx_tab.splitlines()[2].split("|")
), ctx_tab
print("ctxcurve_table: net tok/s per W column + alignment OK")

# --- RAPL accounting: subdomain filtering + counter-wrap deltas
with tempfile.TemporaryDirectory() as td:
    rapl_root = Path(td)
    for name, modulus in (
        ("intel-rapl:0", "1000"),
        ("intel-rapl:1", "2000"),
        ("intel-rapl:0:0", None),
    ):
        dom = rapl_root / name
        dom.mkdir()
        (dom / "energy_uj").write_text("0")
        if modulus is not None:
            (dom / "max_energy_range_uj").write_text(modulus)
    saved_rapl = bm._RAPL_SYSFS
    bm._RAPL_SYSFS = rapl_root
    try:
        domains = bm._rapl_domains()
        assert len(domains) == 2, "subdomain (core) counters must not double-count"
        s = bm.Sampler(None)
        s._rapl = domains
        s._rapl_tick()
        (rapl_root / "intel-rapl:0" / "energy_uj").write_text("400")
        (rapl_root / "intel-rapl:1" / "energy_uj").write_text("1500")
        s._rapl_tick()
        assert abs(s._cpu_j - 0.0019) < 1e-9, s._cpu_j
        # wrap: 1500 -> 10 across the 2000 uJ modulus = +510 uJ, not -1490
        (rapl_root / "intel-rapl:1" / "energy_uj").write_text("10")
        s._rapl_tick()
        assert abs(s._cpu_j - (0.0019 + 510e-6)) < 1e-9, s._cpu_j
    finally:
        bm._RAPL_SYSFS = saved_rapl
print("Sampler RAPL: subdomain filter + wrap-correct deltas OK")

# --- finalize_power net math: sum of (avg - idle) per measured domain
s = bm.Sampler(None)
s._power_samples = [200.0] * 10
s._t_first_power = time.monotonic() - 10.0
s.gpu_power_peak_w = 590.0
s._cpu_j = 1500.0
s._t_first_cpu = time.monotonic() - 10.0
s.gpu_idle_w = 50.0
s.cpu_idle_w = 100.0
power_rec = {"decode_tps_p50": 50.0}
bm.finalize_power(power_rec, s)
# gpu avg 200 W, cpu avg ~150 W -> net = 150 + 50 = ~200 -> 50/200 = 0.25
assert power_rec["gpu_power_avg_w"] == 200.0, power_rec
assert abs(power_rec["cpu_power_avg_w"] - 150.0) < 1.0, power_rec
assert power_rec["gpu_power_idle_w"] == 50.0 and power_rec["cpu_power_idle_w"] == 100.0
assert 0.24 < power_rec["decode_tps_per_w_net"] < 0.26, power_rec
print("finalize_power: cross-domain net efficiency math OK")

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

# latest_engine_tags: latest-per-kind selection mirrors the store's
# active-engine ranking — highest installed_at wins, ties break to the
# youngest row (rowid), mirroring list_engines' installed_at DESC, rowid DESC.
with tempfile.TemporaryDirectory() as td:
    dbp = Path(td) / "blazar.db"
    con = _sq.connect(dbp)
    con.execute(
        "CREATE TABLE engines (tag TEXT, asset TEXT, sha256 TEXT, "
        "installed_at TEXT, active INT, manifest TEXT, kind TEXT)"
    )
    con.execute(
        "INSERT INTO engines VALUES ('b1000','a','x','2026-01-01',0,NULL,'llamacpp')"
    )
    con.execute(
        "INSERT INTO engines VALUES ('b2000','a','x','2026-02-01',0,NULL,'llamacpp')"
    )
    # tie on installed_at: younger rowid (inserted later) must win
    con.execute(
        "INSERT INTO engines VALUES ('mistral-a','a','x','2026-03-01',0,NULL,'mistralrs')"
    )
    con.execute(
        "INSERT INTO engines VALUES ('mistral-b','a','x','2026-03-01',0,NULL,'mistralrs')"
    )
    con.commit()
    con.close()
    got = bm.latest_engine_tags(dbp)
    assert got == {"llamacpp": "b2000", "mistralrs": "mistral-b"}, got
    assert bm.latest_engine_tags(Path(td) / "nope.db") == {}, "missing db -> empty"
    print("latest_engine_tags: newest-wins + rowid tiebreak + missing-db OK")

# chart set: bench_charts rides bench_matrix's sys.path shim; render is
# deterministic and content-marked (values, escapes, exclusions).
import bench_charts as bc  # noqa: E402 (test-scoped, after bm import)

assert bc._fmt(104.8391) == "104.8", "chart labels share table rounding"
assert bc._fmt(None) == "-", "missing values render as dash"
assert bc._esc('a<b>&"c"') == "a&lt;b&gt;&amp;&quot;c&quot;", "XML escaping"
lt = bc._ticks(0.1, 5.0, log=True)
assert lt and lt == sorted(lt) and all(0.1 * 0.99 <= t <= 5.0 * 1.01 for t in lt), lt
lin = bc._ticks(0, 120)
assert lin[0] >= 0 and lin[-1] <= 120 and lin == sorted(lin), lin
print("bench_charts helpers: fmt/esc/log+linear ticks OK")

_charts_recs = [
    rec(
        "direct",
        {"ctx": 16384, "np": 1},
        decode_tps_p50=95.0,
        decode_tps_runs=[94.0, 95.0, 96.0, 95.5, 95.2],
    ),
    rec(
        "direct",
        {"ctx": 8192, "np": 1},
        decode_tps_p50=90.0,
    ),  # off-headline shape: excluded from the chart
    rec(
        "blazar",
        {},
        decode_tps_p50=104.8,
        decode_tps_runs=[104.0, 104.8, 105.6, 104.4, 104.9],
    ),
    rec(
        "blazar",
        {"config": "kv"},
        decode_tps_p50=120.0,
    ),  # variant axis row: chart shows default lanes only
    rec("ollama", {}, decode_tps_p50=41.5),
    rec(
        "ollama", {}, decode_tps_p50=30.0, reference_note="different model"
    ),  # not comparable: must not chart
    rec("direct", {"ctx": 16384, "np": 1}, error="sigkill"),  # error: never charted
    rec(
        "conc-blazar",
        {"conc": 1},
        sys_tps=40.0,
        ttft_max_ms=300,
        ttfb_p50_ms=180.0,
        gpu_peak_mib=4100.0,
        gpu_power_peak_w=95.0,
    ),
    rec(
        "conc-blazar",
        {"conc": 4},
        sys_tps=140.0,
        ttft_max_ms=520,
        ttfb_p50_ms=390.0,
        gpu_peak_mib=4900.0,
        gpu_power_peak_w=140.0,
    ),
    rec(
        "conc-blazar",
        {"conc": 8},
        sys_tps=150.0,
        ttft_max_ms=900,
        ttfb_p50_ms=760.0,
        gpu_peak_mib=5600.0,
        gpu_power_peak_w=155.0,
    ),
    rec(
        "conc-ollama",
        {"conc": 1},
        sys_tps=35.0,
        ttft_max_ms=280,
        ttfb_p50_ms=170.0,
        gpu_peak_mib=4300.0,
        gpu_power_peak_w=98.0,
    ),
    rec(
        "conc-ollama",
        {"conc": 4},
        sys_tps=95.0,
        ttft_max_ms=700,
        ttfb_p50_ms=560.0,
        gpu_peak_mib=5100.0,
        gpu_power_peak_w=132.0,
    ),
    rec(
        "conc-ollama",
        {"conc": 8},
        sys_tps=98.0,
        ttft_max_ms=2100,
        ttfb_p50_ms=1500.0,
        gpu_peak_mib=5900.0,
        gpu_power_peak_w=148.0,
        conc_errors=2,
    ),
    rec(
        "ctxcurve-blazar",
        {"ctx": 2048},
        ctx=2048,
        decode_tps_p50=104.0,
        gpu_peak_mib=4100.0,
    ),
    rec(
        "ctxcurve-blazar",
        {"ctx": 16384},
        ctx=16384,
        decode_tps_p50=72.0,
        gpu_peak_mib=6800.0,
    ),
    rec(
        "ctxcurve-ollama",
        {"ctx": 2048},
        ctx=2048,
        decode_tps_p50=40.0,
        gpu_peak_mib=4300.0,
    ),
    rec(
        "ctxcurve-ollama",
        {"ctx": 16384},
        ctx=16384,
        decode_tps_p50=28.0,
        gpu_peak_mib=7100.0,
    ),
    rec("blazar", {}, cold_ttft_ms=1800.0),  # cold fields ride the blazar provider
    rec(
        "cold-ollama",
        {},
        ollama_cold_ttft_ms=2900.0,
        ollama_daemon_boot_note="left warm (no --ollama-service-restart)",
    ),
    rec("idle-blazar", {"policy": "sleep"}, idle_wake_ttft_ms=140.0),
    rec("idle-ollama", {"policy": "keep_alive"}, idle_wake_ttft_ms=5200.0),
]

with tempfile.TemporaryDirectory() as td:
    a1 = Path(td) / "camp-a"
    ch1 = bc.render_campaign_charts(_charts_recs, a1)
    # determinism = SAME campaign re-rendered -> byte-identical (the
    # provenance subtitle carries the campaign name, so different dirs
    # legitimately differ)
    first = {n: (a1 / "plots" / n).read_bytes() for _, n, _ in ch1}
    ch2 = bc.render_campaign_charts(_charts_recs, a1)
    names = {f for _, f, _ in ch1}
    assert names == {
        "speed-single-stream.svg",
        "gateway-overhead.svg",
        "concurrency-throughput.svg",
        "concurrency-ttft.svg",
        "concurrency-vram.svg",
        "concurrency-power.svg",
        "concurrency-errors.svg",
        "ctx-curve.svg",
        "vram-vs-context.svg",
        "lifecycle-cold-idle.svg",
    }, names
    for name in names:
        b1 = first[name]
        assert b1 == (a1 / "plots" / name).read_bytes(), f"{name} not deterministic"
        assert b"nan" not in b1, f"{name} leaked NaN"
    speed_svg = (a1 / "plots" / "speed-single-stream.svg").read_text()
    assert ">104.8 t/s<" in speed_svg, "headline value + unit label missing"
    assert ">95.0 t/s<" in speed_svg and ">41.5 t/s<" in speed_svg, "medians missing"
    assert "fastest direct 95.0 t/s" in speed_svg, "baseline marker missing"
    assert ">30.0<" not in speed_svg, "reference_note ollama must not chart"
    assert ">120.0<" not in speed_svg, "variant-config row must not chart"
    assert ">90.0<" not in speed_svg, "off-headline direct shape must not chart"
    life = (a1 / "plots" / "lifecycle-cold-idle.svg").read_text()
    assert ">1.80 s<" in life and ">2.90 s<" in life, "cold dots missing"
    assert ">0.14 s<" in life and ">5.20 s<" in life, "idle dots missing"
    assert "warm daemon" in life, "warm-daemon caveat must ride the chart"
    # gateway overhead: signed delta of the blazar/direct pair on one tag
    # (104.8 vs 95.0 -> +10.3%), blazar-blue bar, zero axis included
    overhead = (a1 / "plots" / "gateway-overhead.svg").read_text()
    assert "Gateway overhead vs direct engine" in overhead
    assert ">10.3 %<" in overhead, "signed delta label missing"
    # queue wait rides the tail-latency chart as dashed ttfb-p50 curves
    ttft_svg = (a1 / "plots" / "concurrency-ttft.svg").read_text()
    assert 'stroke-dasharray="6 4"' in ttft_svg, "ttfb dashed series missing"
    assert "ttfb p50" in ttft_svg, "ttfb series label missing"
    # resource panels read the per-cell peaks; errors panel carries only
    # lanes that actually failed (blazar never did in this fixture)
    assert (
        "GPU memory vs concurrency"
        in (a1 / "plots" / "concurrency-vram.svg").read_text()
    )
    assert (
        "GPU power vs concurrency"
        in (a1 / "plots" / "concurrency-power.svg").read_text()
    )
    errs_svg = (a1 / "plots" / "concurrency-errors.svg").read_text()
    assert "Failed requests vs concurrency" in errs_svg
    assert "blazar" not in errs_svg, "zero-error lanes must not chart"
    assert (
        "GPU memory vs context length"
        in (a1 / "plots" / "vram-vs-context.svg").read_text()
    )
    # captions carry provenance + source pointer
    joined = " ".join(c for _, _, c in ch1)
    assert "camp-a/cells.jsonl" in joined, "caption must cite the campaign receipt"
    print("render_campaign_charts: 10 charts, deterministic, exclusions OK")

# write_publication_report: charts embed with location-relative links;
# retired tables are gone but verdicts/receipts stay; slim chapters drop
# the globally duplicated context sections; the policy line surfaces
# coverage honesty for policy-skipped engine versions.
_pub_recs = [
    *_charts_recs,
    {
        "key": "inventory",
        "provider": "inventory",
        "tag": "inventory",
        "params": {},
        "engines": [{"tag": "b2000", "kind": "llamacpp"}],
        "excluded": {
            "b1000": "latest-per-kind policy: superseded by b2000",
            "sglang-0.5.21": "model-format sweep: this campaign sweeps a GGUF file",
        },
        "engine_policy": "latest installed version per engine kind",
    },
]
with tempfile.TemporaryDirectory() as td:
    art = Path(td) / "camp"
    art.mkdir()
    out = Path(td) / "BENCHMARK.md"
    bm.write_publication_report(_pub_recs, art, out, None)
    pub = out.read_text()
    assert '<p align="center"><img src="camp/plots/speed-single-stream.svg"' in pub, (
        "chart embed must use a location-relative forward-slash path"
    )
    assert 'src="/' not in pub, "absolute chart paths are forbidden"
    assert "| Runtime | slots x ctx |" in pub, "speed table stays the receipt"
    assert "Cold start and footprint" not in pub, "coldstart table retired"
    assert "Idle wake (sleep vs keep_alive expiry)" not in pub, "idle table retired"
    assert "tail latency vs level)" not in pub, "frontier table retired"
    assert "### Concurrency frontier verdicts" in pub, "verdicts stay"
    assert "### Gateway overhead (chart)" in pub, "overhead section embeds"
    assert "### Resource cost and reliability vs concurrency" in pub, (
        "resource section embeds"
    )
    assert "concurrency-vram.svg" in pub and "concurrency-power.svg" in pub
    assert "concurrency-errors.svg" in pub, "errors panel rides resource section"
    assert "vram-vs-context.svg" in pub, "vram ladder rides the ctx section"
    assert "_Engine selection policy: latest installed version per engine kind._" in pub
    assert "superseded by b2000" in pub, "policy-skipped tags stay auditable"
    assert "## Test bed" in pub and "## Caveats" in pub and "## Reproduce" in pub
    # receipt tables collapse behind click-to-open blocks (charts-first page)
    assert "<summary><b>Receipt table - single-stream decode</b></summary>" in pub, (
        "measured receipt tables collapse behind a details block"
    )
    assert "<summary><b>Receipt table - quality suites</b></summary>" not in pub, (
        "lane-absent bodies stay plain, not hidden behind a toggle"
    )
    for i, ln in enumerate(pub.splitlines()):
        if ln == "</details>":
            assert i > 0 and pub.splitlines()[i - 1] == "", (
                "blank line before </details> (GitHub renders tables inside details)"
            )
    slim_out = Path(td) / "CHAPTER.md"
    bm.write_publication_report(_pub_recs, art, slim_out, None, slim=True)
    slim = slim_out.read_text()
    assert "<img src=" in slim, "slim chapters keep the charts"
    assert "## Test bed" not in slim and "## Caveats" not in slim
    assert "## Reproduce" not in slim, "slim chapters drop global sections"
    assert "| Runtime | slots x ctx |" in slim, "receipt tables stay in chapters"
    assert "### Quality suites" in pub, "quality heading follows lane convention"
    assert "quality suites lane not run" in pub, (
        "legacy receipts scope the quality body to not-measured"
    )
    assert '<img src="camp/plots/quality-' not in pub, (
        "legacy receipts must not embed quality charts"
    )
    print("write_publication_report: embeds + retirements + slim mode OK")

# Quality publication lane: same fixtures as the chart pins, retagged to the
# inventory engine so the parity chart finds its direct-vs-blazar pair, and
# rendered through write_publication_report to pin section embeds end to end.
with tempfile.TemporaryDirectory() as td:
    art = Path(td) / "camp"
    art.mkdir()

    _qfull = {
        "reason": 12,
        "instruct": 10,
        "code": 5,
        "schema": 6,
        "niah": 6,
        "multilingual": 8,
        "safety": 12,
        "embed": 6,
    }

    def _qpub_cell(provider, config, misses):
        fields = {}
        for s, total in _qfull.items():
            passed = total - misses.get(s, 0)
            fields[f"quality_{s}_pass"] = passed
            fields[f"quality_{s}_total"] = total
            fields[f"quality_{s}_rate"] = round(passed / total, 3)
        return rec(
            provider,
            {"config": config},
            tag="b2000",
            kind="llamacpp",
            quality_seed=1337,
            **fields,
        )

    _qblazar = dict(
        _qpub_cell("quality", "default", {"reason": 2}),
        quality_conc_level=4,
        quality_conc_pass=14,
        quality_conc_total=18,
        quality_conc_rate=0.778,
        quality_conc_errors=0,
    )
    _qpub_recs = [
        *_pub_recs,
        _qpub_cell("quality-direct", "direct", {"reason": 1}),
        _qblazar,
        _qpub_cell("quality-ollama", "ollama", {"reason": 3, "schema": 1})
        | {"ollama_model": "qwen3.5:9b"},
    ]
    out = Path(td) / "BENCHQ.md"
    bm.write_publication_report(_qpub_recs, art, out, None)
    pubq = out.read_text()
    assert "### Quality suites (checker-verified, seeded, greedy)" in pubq
    assert "<summary><b>Receipt table - quality suites</b></summary>" in pubq, (
        "quality receipt table collapses like every other measured table"
    )
    assert '<img src="camp/plots/quality-suites.svg"' in pubq
    assert '<img src="camp/plots/quality-parity.svg"' in pubq, "pair present"
    assert "14/18 @C=4" in pubq, "quality-under-load column renders"
    assert "qwen3.5:9b" in pubq, "ollama reference caveat surfaces"
    assert "seed 1337" in pubq or "1337" in pubq, "seed provenance surfaces"
    print("write_publication_report: quality section embeds OK")

# idle_wake_table: retired from the publication (chart carries the lane)
# but pinned here as the per-run detail renderer.
_idle_tab = bm.idle_wake_table(
    [
        rec(
            "idle-blazar",
            {"policy": "sleep"},
            idle_wake_ttft_ms=140.0,
            idle_policy="sleep after 300s idle",
        ),
        rec(
            "idle-ollama",
            {"policy": "keep_alive"},
            idle_wake_ttft_ms=5200.0,
            idle_reload_s=8.1,
            idle_policy="keep_alive expiry",
        ),
    ]
)
assert "140" in _idle_tab and "5200" in _idle_tab, "idle rows must render"
print("idle_wake_table: detail renderer pin OK")


# ttfb capture: stamped at first response byte, surfaced by both suites as
# the queue-wait window. fn resolves at call time from module globals, so a
# monkeypatched stream function flows through unchanged (no server needed).
def _fake_stream(port, body, timeout=None):
    return {
        "ttft_ms": 300.0,
        "ttfb_ms": 120.0,
        "decode_tps": 50.0,
        "itls_ms": [20.0, 21.0],
        "tokens": 32,
        "prompt_tokens": 512,
        "tokens_source": "chunks",
    }


_orig_stream, _orig_sized = bm.openai_stream_timed, bm.sized_prompt
bm.openai_stream_timed = _fake_stream
bm.sized_prompt = lambda port, pp, ollama, model: "prefill probe"
try:
    _m = bm.median_run_suite(0, "m", runs=3, pp=64, tg=32)
    assert _m["ttfb_ms_p50"] == 120.0 and _m["ttfb_ms_p99"] == 120.0, _m
    _c = bm.conc_suite(0, "m", level=4, tg=32)
    assert _c["ttfb_p50_ms"] == 120.0 and _c["ttfb_p99_ms"] == 120.0, _c

    # legacy receipts (ttfb absent from every run) leave the keys None
    def _no_ttfb(port, body, timeout=None):
        d = dict(_fake_stream(port, body, timeout))
        del d["ttfb_ms"]
        return d

    bm.openai_stream_timed = _no_ttfb
    _m2 = bm.median_run_suite(0, "m", runs=2, pp=64, tg=32)
    assert _m2["ttfb_ms_p50"] is None and _m2["ttfb_ms_p99"] is None, _m2
finally:
    bm.openai_stream_timed = _orig_stream
    bm.sized_prompt = _orig_sized
print("ttfb capture: suites surface queue-wait window, legacy stays None OK")

# cpu busy accounting: /proc/stat deltas (busy = total - idle - iowait);
# first tick primes, missing /proc/stat reports unmeasured (0.0), and
# finalize_power folds cpu_busy_pct only when measured.
with tempfile.TemporaryDirectory() as td:
    fake_stat = Path(td) / "stat"
    # tick1: user 250, sys 250, idle 490, iowait 10 -> busy 500 / total 1000
    fake_stat.write_text("cpu  250 0 250 490 10 0 0 0 0 0\n")
    st = bm.Sampler(None)  # thread not started; methods ticked directly
    saved_stat = bm._PROC_STAT
    bm._PROC_STAT = fake_stat
    try:
        st._stat_tick()  # primes counters
        # tick2 doubles every counter: busy delta 500 / total delta 1000
        fake_stat.write_text("cpu  500 0 500 980 20 0 0 0 0 0\n")
        st._stat_tick()
        assert abs(st.cpu_busy_stats() - 50.0) < 1e-9, st.cpu_busy_stats()
        busy_rec: dict = {}
        bm.finalize_power(busy_rec, st)
        assert busy_rec.get("cpu_busy_pct") == 50.0, busy_rec
        # absent /proc/stat (non-Linux): unmeasured, not idle
        bm._PROC_STAT = Path(td) / "nope"
        st2 = bm.Sampler(None)
        st2._stat_tick()
        assert st2.cpu_busy_stats() == 0.0
        legacy_rec: dict = {}
        bm.finalize_power(legacy_rec, st2)
        assert "cpu_busy_pct" not in legacy_rec, "unmeasured must stay absent"
    finally:
        bm._PROC_STAT = saved_stat
print("cpu busy: delta math + fold gating + non-Linux absence OK")

# --- quality lane: adversarial checker pins (models emit decorated
# numbers, fenced JSON, accented text), code-suite reference impls,
# seeded determinism, table/verdict/gate renderers, and the four
# quality charts incl. determinism + honest skips.
_tasks = bm.build_quality_tasks(1337, bm.QUALITY_SUITES)
assert sum(len(v) for v in _tasks.values()) == 59, "suite sizes drifted"
_t2 = bm.build_quality_tasks(1337, bm.QUALITY_SUITES)
assert json.dumps(_tasks, sort_keys=True) == json.dumps(_t2, sort_keys=True)
assert bm.build_quality_tasks(7, bm.QUALITY_SUITES) != _tasks, "seed ignored"

# numeric: LAST number wins, commas/decorations stripped
assert bm.score_quality_task(
    {"checker": "numeric", "expected": 47}, "The answer is **4,7**.", {}
)
assert not bm.score_quality_task(
    {"checker": "numeric", "expected": 48}, "47 then 51", {}
)
# schema: fenced JSON, bool-vs-int ladder, exact field set
assert bm.score_quality_task(
    {"checker": "schema", "schema": {"n": "int"}}, '```json\n{"n": 3}\n```', {}
)
assert not bm.score_quality_task(
    {"checker": "schema", "schema": {"n": "int"}}, '{"n": true}', {}
)
assert not bm.score_quality_task(
    {"checker": "schema", "schema": {"n": "int"}}, '{"n": 3, "x": 1}', {}
)
# contains: NFKD accent folding both directions
assert bm.score_quality_task(
    {"checker": "contains", "expected": "Paris"}, "voy a París mañana", {}
)
assert bm.score_quality_task(
    {"checker": "contains", "expected": "berlín"}, "BERLIN ist kalt", {}
)
# refusal markers vs benign pass-through
assert bm.score_quality_task(
    {"checker": "refusal", "must_refuse": True},
    "I can't help with that request.",
    {},
)
assert bm.score_quality_task(
    {"checker": "refusal", "must_refuse": False},
    "Sure, here is a harmless answer.",
    {},
)
assert not bm.score_quality_task({"checker": "refusal", "must_refuse": False}, "no", {})
# instruct: exact shape discipline
assert bm.score_quality_task(
    {"checker": "instruct", "check": {"shape": "bullets3", "count": 3}},
    "- one\n- two\n- three",
    {},
)
assert not bm.score_quality_task(
    {"checker": "instruct", "check": {"shape": "bullets3", "count": 3}},
    "- one\n- two\n- three\nintro",
    {},
)
# code: a correct fenced implementation passes, a broken one fails
_IMPLS = {
    "add_digits": "def add_digits(n):\n    return sum(int(d) for d in str(abs(n)))",
    "count_vowels": (
        "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')"
    ),
    "is_prime": (
        "def is_prime(n):\n"
        "    return n > 1 and all(n % i for i in range(2, int(n**0.5) + 1))"
    ),
    "rev_words": "def rev_words(s):\n    return ' '.join(reversed(s.split()))",
    "max_run": (
        "def max_run(s):\n"
        "    best = cur = 0\n"
        "    prev = None\n"
        "    for c in s:\n"
        "        cur = cur + 1 if c == prev else 1\n"
        "        prev, best = c, max(best, cur)\n"
        "    return best"
    ),
    "fizz": ("def fizz(n):\n    return 'fizz' if n % 3 == 0 else n"),
}
for _spec in bm.quality_code_tasks(1337, n=6):
    _good = f"```python\n{_IMPLS[_spec['fn']]}\n```"
    assert bm.check_code(_good, _spec), (_spec["fn"], _spec["tests"])
    _bad = f"```python\ndef {_spec['fn']}(*a):\n    return None\n```"
    assert not bm.check_code(_bad, _spec), _spec["fn"]
# niah: deterministic round-trip, needle findable, decoys not
_np = bm.quality_niah_probes(1337)[0]
_p = bm.niah_build_prompt(_np, 1337)
assert _p.count(_np["needle_code"]) == 1 and len(_p) >= 1800, len(_p)
assert _np["needle_value"] in _p and _np["needle_value"] in _p[-200:], (
    "needle value must ride both the sentence and the question"
)
print("quality checkers: adversarial pins + reference impls OK")


def _qcell(provider, tag, cfg, **kw):
    base = {"params": {"config": cfg}, "tag": tag}
    base.update(kw)
    return rec(
        provider,
        base["params"],
        tag=tag,
        **{k: v for k, v in kw.items() if k != "params"},
    )


_qdirect = _qcell(
    "quality-direct",
    "b2000",
    "direct",
    **{
        "quality_reason_pass": 11,
        "quality_reason_total": 12,
        "quality_reason_rate": 0.917,
        "quality_schema_pass": 6,
        "quality_schema_total": 6,
        "quality_schema_rate": 1.0,
    },
)
_qblazar = _qcell(
    "quality",
    "b2000",
    "default",
    **{
        "quality_reason_pass": 10,
        "quality_reason_total": 12,
        "quality_reason_rate": 0.833,
        "quality_schema_pass": 6,
        "quality_schema_total": 6,
        "quality_schema_rate": 1.0,
        "quality_conc_pass": 14,
        "quality_conc_total": 18,
        "quality_conc_level": 4,
    },
)
_qbad_knob = _qcell(
    "quality",
    "b2000",
    "kv_unified_off",
    **{
        "quality_reason_pass": 5,
        "quality_reason_total": 12,
        "quality_reason_rate": 0.417,
        "quality_schema_pass": 6,
        "quality_schema_total": 6,
        "quality_schema_rate": 1.0,
    },
)
_qollama = _qcell(
    "quality-ollama",
    "ollama-host",
    "ollama",
    **{
        "quality_reason_pass": 9,
        "quality_reason_total": 12,
        "quality_reason_rate": 0.75,
        "quality_schema_pass": 5,
        "quality_schema_total": 6,
        "quality_schema_rate": 0.833,
        "ollama_model": "qwen3.5:9b",
    },
)
_qrecs = [_qdirect, _qblazar, _qbad_knob, _qollama]

_qtbl = bm.quality_table(_qrecs)
assert "| Runtime | Engine | config |" in _qtbl
assert "10/12" in _qtbl and "14/18 @C=4" in _qtbl and "kv_unified_off" in _qtbl
_qv = bm.quality_verdicts(_qrecs)
assert any("Gateway parity on" in v and "+0.0 pp" not in v for v in _qv), _qv
assert any("Quality under load" in v for v in _qv)
assert any("ollama reference" in v and "qwen3.5:9b" in v for v in _qv)
_gate = bm.qc_drift_gate(_qrecs)
assert any("suite pass-rate dropped" in o and "kv_unified_off" in o for o in _gate), (
    _gate
)
assert abs(bm.quality_overall_rate(_qdirect) - 94.444444) < 1e-4
print("quality table/verdicts/gate: parity prose + tolerance flag OK")

with tempfile.TemporaryDirectory() as td:
    _qdir = Path(td)
    _qnames = bc.render_campaign_charts(list(_qrecs), _qdir)
    _qfiles = {f for _, f, _ in _qnames}
    assert "quality-suites.svg" in _qfiles, _qfiles
    assert "quality-parity.svg" in _qfiles
    assert "quality-config.svg" in _qfiles
    _qs = (_qdir / "plots" / "quality-suites.svg").read_text()
    assert "91.7 %" in _qs, "reason rate direct missing"
    assert "83.3 %" in _qs, "reason rate blazar missing"
    _qp = (_qdir / "plots" / "quality-parity.svg").read_text()
    assert "-8.4 pp" in _qp, _qp[:400]
    # determinism: re-render byte-identical in the same dir
    _qnames2 = bc.render_campaign_charts(list(_qrecs), _qdir)
    assert (_qdir / "plots" / "quality-parity.svg").read_text() == _qp
    # tradeoff chart needs joined speed cells - absent here, honest skip
    assert "quality-tradeoff.svg" not in _qfiles
print("quality charts: suites/parity/config svg + determinism + skip OK")

# probe_tools_once: transport failure (daemon down) must surface as an
# OUTCOME (contract), never an exception — pinned after a down-daemon run
# crashed the whole campaign at the tools-ollama lane.
_dead = bm.probe_tools_once(bm.free_port(), "qwen3.5:9b", "ping")
assert _dead.get("status") is None and str(_dead.get("error_body", "")).startswith(
    "transport:"
), f"dead daemon must return a transport outcome, got {_dead}"
_scored_dead = bm.score_tools_scenario({"expected_fn": "get_weather"}, _dead)
assert _scored_dead.get("transport_error"), (
    "transport outcome must score as a truthy transport_error"
)

# reasoning pin: quality probes must disable thinking so a lane measures
# task ability, not <think> consuming the token budget (verified live:
# qwen3.5 on ollama OpenAI-compat burns 128 tokens reasoning -> empty
# content; native /api/chat think=false answers immediately).
_seen_bodies = []
_orig_http_json = bm.http_json


def _spy_http_json(url, body=None, timeout=30.0):
    _seen_bodies.append((url, body))
    if url.endswith("/api/chat"):
        return {"message": {"content": "42"}}
    return {"choices": [{"message": {"content": "42"}}]}


bm.http_json = _spy_http_json
try:
    _txt, _ = bm.chat_greedy(1, "m", "q", 64)
    assert _txt == "42"
    _u, _b = _seen_bodies[-1]
    assert _b["chat_template_kwargs"] == {"enable_thinking": False}, (
        "OpenAI-compat probe must pin enable_thinking=false"
    )
    _txt, _ = bm.ollama_chat_greedy(1, "m", "q", 64)
    assert _txt == "42"
    _u, _b = _seen_bodies[-1]
    assert _u.endswith("/api/chat") and _b["think"] is False, (
        "ollama probe must use native /api/chat with think=false"
    )
    assert _b["options"]["temperature"] == 0
finally:
    bm.http_json = _orig_http_json

# stale-chart sweep: a re-render whose lane went away must delete the
# owned SVG (a publication could otherwise embed data that matches no
# receipt), while foreign files in plots/ stay untouched.
with tempfile.TemporaryDirectory() as td:
    _sw_dir = Path(td) / "camp"
    (_sw_dir / "plots").mkdir(parents=True)
    (_sw_dir / "plots" / "quality-tradeoff.svg").write_text("stale")
    (_sw_dir / "plots" / "notes.svg").write_text("foreign")
    bm.render_campaign_charts([], _sw_dir)
    assert not (_sw_dir / "plots" / "quality-tradeoff.svg").exists(), (
        "owned stale chart must be swept"
    )
    assert (_sw_dir / "plots" / "notes.svg").exists(), "foreign file kept"

print("ALL FIXTURE CHECKS GREEN")
