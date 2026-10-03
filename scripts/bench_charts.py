#!/usr/bin/env python3
"""Zero-dependency SVG charts for the benchmark publication.

Design follows professional benchmark-report conventions (MLPerf / LLMPerf
style): sorted horizontal bars with direct value labels, throughput and
tail-latency curves against concurrency, a long-context degradation curve,
and a cold/idle lifecycle panel. Okabe-Ito colorblind-safe palette, direct
labeling over legends, log scales for latency spans, and a provenance
subtitle (model / date / campaign) on every chart.

Pure functions, deterministic output (no timestamps, sorted iteration):
rendering the same cells twice produces byte-identical SVGs, which the
selftest pins.
"""

from __future__ import annotations

import math
from pathlib import Path

# ---------------------------------------------------------------------------
# design tokens

W = 880
SANS = "Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
TITLE_PX, TITLE_FILL = 16, "#111111"
SUB_PX, SUB_FILL = 11.5, "#777777"
TICK_PX, TICK_FILL = 11, "#666666"
VALUE_PX = 12.5
GRID, AXIS = "#EBEBEB", "#C9C9C9"
INK, WHISKER, BASELINE = "#222222", "#333333", "#555555"
FOOT_PX, FOOT_FILL = 10.5, "#999999"

# Okabe-Ito, one stable color per provider family; alternates cover extra
# engines within one family (multiple installed engine tags).
FAMILY_COLOR = {
    "blazar": "#0072B2",  # blue
    "direct": "#E69F00",  # orange
    "ollama": "#009E73",  # bluish green
}
ALT_COLORS = ("#CC79A7", "#56B4E9", "#999999")  # reddish purple, sky, grey


# ---------------------------------------------------------------------------
# small helpers


def _esc(s: str) -> str:
    """XML-escape every text node that reaches the SVG."""
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _fmt(v: float | None, nd: int = 1) -> str:
    """Shared decimal formatter: chart labels must round exactly like the
    markdown tables (pfmt-compatible explicit format spec, never str(round))."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    return f"{v:.{nd}f}"


def _pctl(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolation percentile over an already-sorted sample."""
    if not sorted_vals:
        raise ValueError("percentile of empty sample")
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = (len(sorted_vals) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(sorted_vals[lo])
    return float(sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo))


def _median(vals: list[float]) -> float:
    return _pctl(sorted(vals), 0.5)


def _text_w(s: str, px: float) -> float:
    """Rough mixed-case advance width; only sizes layout margins/labels."""
    return 0.62 * px * len(str(s))


def _nice_step(span: float) -> float:
    exp = math.floor(math.log10(span / 5))
    for mult in (1, 2, 2.5, 5, 10):
        step = mult * 10**exp
        if span / step <= 6:
            return step
    return 10 ** (exp + 1)


def _ticks(lo: float, hi: float, log: bool = False) -> list[float]:
    """Axis ticks: nice linear steps, or decades with 1-2-5 mantissas when
    the span is under two decades (latency charts often span 100 ms..5 s)."""
    if hi <= lo:
        return [lo]
    if not log:
        step = _nice_step(hi - lo)
        first = math.ceil(lo / step) * step
        out, v = [], first
        while v <= hi + step * 1e-9:
            out.append(round(v, 10))
            v += step
        return out
    d0, d1 = math.floor(math.log10(lo)), math.ceil(math.log10(hi))
    ticks: list[float] = []
    if d1 - d0 <= 2:
        for d in range(d0, d1 + 1):
            for m in (1, 2, 5):
                v = m * 10**d
                if lo * 0.999 <= v <= hi * 1.001:
                    ticks.append(v)
    else:
        ticks = [10.0**d for d in range(d0, d1 + 1)]
    return ticks or [lo, hi]


def _tick_label(v: float, log: bool) -> str:
    if log:
        if v > 0:
            exp = round(math.log10(v))
            if abs(v - 10.0**exp) < 1e-9:
                return f"1e{exp}"
        return _fmt(v, 1 if v < 10 else 0)
    if abs(v) >= 1000:
        return f"{v / 1000:g}k"
    return _fmt(v, 0 if abs(v) >= 10 else 1)


def _header(title: str, subtitle: str) -> list[str]:
    return [
        f'<text x="0" y="16" font-family="{SANS}" font-size="{TITLE_PX}" '
        f'font-weight="600" fill="{TITLE_FILL}">{_esc(title)}</text>',
        f'<text x="0" y="34" font-family="{SANS}" font-size="{SUB_PX}" '
        f'fill="{SUB_FILL}">{_esc(subtitle)}</text>',
    ]


def _series_color(family: str, index: int) -> str:
    base = FAMILY_COLOR.get(family, ALT_COLORS[2])
    return base if index == 0 else ALT_COLORS[(index - 1) % len(ALT_COLORS)]


def _svg(doc: list[str], w: float, h: float) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.0f}" '
        f'height="{h:.0f}" viewBox="0 0 {w:.0f} {h:.0f}" '
        f'role="img">\n' + "\n".join(doc) + "\n</svg>\n"
    )


# ---------------------------------------------------------------------------
# chart primitives


def hbar_svg(
    title: str,
    subtitle: str,
    rows: list[tuple[str, float, float, float, str]],
    unit: str,
    baseline: tuple[str, float] | None = None,
    signed: bool = False,
    footnote: str | None = None,
) -> str:
    """Sorted horizontal bars with direct value labels and IQR whiskers.

    rows: (label, value, iqr_lo, iqr_hi, color) — caller sorts. baseline:
    (label, x) rendered as a dashed reference line. signed: rows may carry
    negative values (gateway-overhead deltas); the scale then spans both
    sides of a drawn zero axis and value labels sit on the outer side of
    each bar.
    """
    if not rows:
        raise ValueError("hbar_svg needs at least one row")
    left = max(_text_w(lbl, TICK_PX) for lbl, *_ in rows) + 14
    right = W - 86
    plot_top, row_h, bar_h = 56, 30, 16
    vmax = max(hi for _, _, _, hi, _ in rows)
    if baseline:
        vmax = max(vmax, baseline[1])
    vmax *= 1.08
    vmin = 0.0
    if signed:
        vmin = min(0.0, min(lo for _, _, lo, _, _ in rows)) * 1.15
        vmax = max(vmax, 0.0)

    def x(v: float) -> float:
        if signed:
            return left + (right - left) * ((v - vmin) / (vmax - vmin))
        return left + (right - left) * (v / vmax)

    doc = _header(title, subtitle)
    for t in _ticks(vmin, vmax):
        doc.append(
            f'<line x1="{x(t):.1f}" y1="{plot_top}" x2="{x(t):.1f}" '
            f'y2="{plot_top + len(rows) * row_h}" stroke="{GRID}" '
            f'stroke-width="1"/>'
        )
        doc.append(
            f'<text x="{x(t):.1f}" y="{plot_top + len(rows) * row_h + 16}" '
            f'font-family="{SANS}" font-size="{TICK_PX}" fill="{TICK_FILL}" '
            f'text-anchor="middle">{_esc(_tick_label(t, False))}</text>'
        )
    if signed:
        doc.append(
            f'<line x1="{x(0):.1f}" y1="{plot_top}" x2="{x(0):.1f}" '
            f'y2="{plot_top + len(rows) * row_h}" stroke="{AXIS}" '
            f'stroke-width="1.2"/>'
        )
    if baseline:
        doc.append(
            f'<line x1="{x(baseline[1]):.1f}" y1="{plot_top - 4}" '
            f'x2="{x(baseline[1]):.1f}" y2="{plot_top + len(rows) * row_h}" '
            f'stroke="{BASELINE}" stroke-width="1.4" stroke-dasharray="5 4"/>'
        )
        doc.append(
            f'<text x="{x(baseline[1]):.1f}" y="{plot_top - 8}" '
            f'font-family="{SANS}" font-size="{FOOT_PX}" fill="{BASELINE}" '
            f'text-anchor="middle">{_esc(baseline[0])}</text>'
        )
    for i, (lbl, val, lo, hi, color) in enumerate(rows):
        cy = plot_top + i * row_h + bar_h / 2
        doc.append(
            f'<text x="{left - 8}" y="{cy + 4:.1f}" font-family="{SANS}" '
            f'font-size="{TICK_PX}" fill="{INK}" text-anchor="end">'
            f"{_esc(lbl)}</text>"
        )
        bx = min(x(0), x(val))
        bw = max(abs(x(val) - x(0)), 1)
        doc.append(
            f'<rect x="{bx:.1f}" y="{plot_top + i * row_h + 2:.1f}" '
            f'width="{bw:.1f}" height="{bar_h}" rx="2.5" '
            f'fill="{color}"/>'
        )
        if hi > lo:
            wl, wr = x(lo), x(hi)
            cap = 3.5
            doc.append(
                f'<line x1="{wl:.1f}" y1="{cy:.1f}" x2="{wr:.1f}" y2="{cy:.1f}" '
                f'stroke="{WHISKER}" stroke-width="1.2"/>'
            )
            for wx in (wl, wr):
                doc.append(
                    f'<line x1="{wx:.1f}" y1="{cy - cap:.1f}" x2="{wx:.1f}" '
                    f'y2="{cy + cap:.1f}" stroke="{WHISKER}" '
                    f'stroke-width="1.2"/>'
                )
        lbl_x = x(hi) + 8
        anchor = ""
        if signed and val < 0:
            lbl_x, anchor = x(hi) - 8, ' text-anchor="end"'
        doc.append(
            f'<text x="{lbl_x:.1f}" y="{cy + 4:.1f}"{anchor} '
            f'font-family="{MONO}" font-size="{VALUE_PX}" fill="{INK}" '
            f'font-weight="600">{_esc(_fmt(val))} {_esc(unit)}</text>'
        )
    fy = plot_top + len(rows) * row_h + 34
    doc.append(
        f'<line x1="{left:.1f}" y1="{fy - 18}" x2="{right:.1f}" y2="{fy - 18}" '
        f'stroke="{AXIS}" stroke-width="1"/>'
    )
    if footnote:
        doc.append(
            f'<text x="0" y="{fy + 2:.1f}" font-family="{SANS}" '
            f'font-size="{FOOT_PX}" fill="{FOOT_FILL}">{_esc(footnote)}</text>'
        )
    return _svg(doc, W, fy + (12 if footnote else 0))


def line_svg(
    title: str,
    subtitle: str,
    series: list[tuple],
    x_label: str,
    y_label: str,
    log_x: bool = False,
    log_y: bool = False,
    footnote: str | None = None,
) -> str:
    """Multi-series line panel. series: (label, points, color) with points
    sorted ascending on x, plus an optional 4th element: an SVG dash pattern
    that renders the polyline dashed (secondary series such as queue-wait
    TTFB ride the same color dashed). Few series get direct end labels; more
    fall back to a top-right legend."""
    if not series:
        raise ValueError("line_svg needs at least one series")
    series = [(s[0], s[1], s[2], s[3] if len(s) > 3 else None) for s in series]
    all_pts = [p for _, pts, _, _ in series for p in pts]
    if not all_pts:
        raise ValueError("line_svg series carry no points")
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]

    def lo(v: list[float], log: bool) -> float:
        m = min(v)
        return m / (1.35 if log else 1.0)

    def hi(v: list[float], log: bool) -> float:
        m = max(v)
        return m * (1.35 if log else 1.06)

    xlo, xhi = lo(xs, log_x), hi(xs, log_x)
    ylo, yhi = lo(ys, log_y), hi(ys, log_y)
    direct = len(series) <= 4
    top, left = 56, 58
    right = (
        W - (_text_w(max((s[0] for s in series), key=len), TICK_PX) + 18)
        if direct
        else W - 26
    )

    def x(v: float) -> float:
        if log_x:
            return left + (right - left) * (math.log10(v / xlo) / math.log10(xhi / xlo))
        return left + (right - left) * ((v - xlo) / (xhi - xlo))

    def y(v: float) -> float:
        if log_y:
            return top + (1 - math.log10(v / ylo) / math.log10(yhi / ylo)) * 240
        return top + (1 - (v - ylo) / (yhi - ylo)) * 240

    doc = _header(title, subtitle)
    for t in _ticks(ylo, yhi, log_y):
        doc.append(
            f'<line x1="{left}" y1="{y(t):.1f}" x2="{right}" y2="{y(t):.1f}" '
            f'stroke="{GRID}" stroke-width="1"/>'
        )
        doc.append(
            f'<text x="{left - 8}" y="{y(t) + 4:.1f}" font-family="{SANS}" '
            f'font-size="{TICK_PX}" fill="{TICK_FILL}" text-anchor="end">'
            f"{_esc(_tick_label(t, log_y))}</text>"
        )
    for t in _ticks(xlo, xhi, log_x):
        doc.append(
            f'<text x="{x(t):.1f}" y="{top + 240 + 16}" font-family="{SANS}" '
            f'font-size="{TICK_PX}" fill="{TICK_FILL}" text-anchor="middle">'
            f"{_esc(_tick_label(t, log_x))}</text>"
        )
    doc.append(
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + 240}" '
        f'stroke="{AXIS}" stroke-width="1"/>'
    )
    doc.append(
        f'<line x1="{left}" y1="{top + 240}" x2="{right}" y2="{top + 240}" '
        f'stroke="{AXIS}" stroke-width="1"/>'
    )
    doc.append(
        f'<text x="{left}" y="{top - 10}" font-family="{SANS}" '
        f'font-size="{TICK_PX}" fill="{SUB_FILL}">{_esc(y_label)}</text>'
    )
    doc.append(
        f'<text x="{(left + right) / 2:.1f}" y="{top + 240 + 34}" '
        f'font-family="{SANS}" font-size="{TICK_PX}" fill="{SUB_FILL}" '
        f'text-anchor="middle">{_esc(x_label)}</text>'
    )
    for lbl, pts, color, dash in series:
        if len(pts) > 1:
            path = " ".join(f"{x(px):.1f},{y(py):.1f}" for px, py in pts)
            dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
            doc.append(
                f'<polyline points="{path}" fill="none" stroke="{color}" '
                f'stroke-width="2" stroke-linejoin="round"{dash_attr}/>'
            )
        for px, py in pts:
            doc.append(
                f'<circle cx="{x(px):.1f}" cy="{y(py):.1f}" r="4" '
                f'fill="#FFFFFF" stroke="{color}" stroke-width="2"/>'
            )
        if direct:
            lx, ly = pts[-1]
            doc.append(
                f'<text x="{x(lx) + 9:.1f}" y="{y(ly) + 4:.1f}" '
                f'font-family="{SANS}" font-size="{TICK_PX}" fill="{color}" '
                f'font-weight="600">{_esc(lbl)}</text>'
            )
    if not direct:
        lx, ly = right, top + 6
        for lbl, _, color, _dash in series:
            doc.append(
                f'<rect x="{lx - _text_w(lbl, TICK_PX) - 22:.1f}" '
                f'y="{ly - 9:.1f}" width="10" height="10" rx="2" '
                f'fill="{color}"/>'
            )
            doc.append(
                f'<text x="{lx - _text_w(lbl, TICK_PX) - 8:.1f}" y="{ly:.1f}" '
                f'font-family="{SANS}" font-size="{TICK_PX}" fill="{INK}" '
                f'text-anchor="end">{_esc(lbl)}</text>'
            )
            ly += 17
    if footnote:
        doc.append(
            f'<text x="0" y="{top + 240 + 56}" font-family="{SANS}" '
            f'font-size="{FOOT_PX}" fill="{FOOT_FILL}">{_esc(footnote)}</text>'
        )
    return _svg(doc, W, top + 240 + 62)


def dumbbell_svg(
    title: str,
    subtitle: str,
    rows: list[tuple[str, float | None, float | None, str, str]],
    unit: str,
    name_a: str,
    name_b: str,
    footnote: str | None = None,
) -> str:
    """Paired-dot panel on a log value axis (cold/idle lifecycle): one dot
    per runtime per row, connected; no bar-from-zero (log axes have no
    zero). rows: (label, value_a, value_b, color_a, color_b)."""
    if not rows:
        raise ValueError("dumbbell_svg needs at least one row")
    vals = [v for _, a, b, _, _ in rows for v in (a, b) if v is not None]
    if not vals:
        raise ValueError("dumbbell_svg rows carry no values")
    vlo, vhi = min(vals) / 1.5, max(vals) * 1.5
    left = max(_text_w(lbl, TICK_PX) for lbl, *_ in rows) + 14
    right = W - 96
    plot_top, row_h = 56, 34

    def x(v: float) -> float:
        return left + (right - left) * (math.log10(v / vlo) / math.log10(vhi / vlo))

    doc = _header(title, subtitle)
    for t in _ticks(vlo, vhi, log=True):
        doc.append(
            f'<line x1="{x(t):.1f}" y1="{plot_top}" x2="{x(t):.1f}" '
            f'y2="{plot_top + len(rows) * row_h}" stroke="{GRID}" '
            f'stroke-width="1"/>'
        )
        doc.append(
            f'<text x="{x(t):.1f}" y="{plot_top + len(rows) * row_h + 16}" '
            f'font-family="{SANS}" font-size="{TICK_PX}" fill="{TICK_FILL}" '
            f'text-anchor="middle">{_esc(_tick_label(t, True))}</text>'
        )
    lx = right
    for nm, color in ((name_a, rows[0][3]), (name_b, rows[0][4])):
        doc.append(
            f'<rect x="{lx - _text_w(nm, TICK_PX) - 20:.1f}" y="40" width="10" '
            f'height="10" rx="2" fill="{color}"/>'
        )
        doc.append(
            f'<text x="{lx - _text_w(nm, TICK_PX) - 6:.1f}" y="49" '
            f'font-family="{SANS}" font-size="{TICK_PX}" fill="{INK}" '
            f'text-anchor="end">{_esc(nm)}</text>'
        )
        lx -= _text_w(nm, TICK_PX) + 40
    for i, (lbl, a, b, ca, cb) in enumerate(rows):
        cy = plot_top + i * row_h + row_h / 2
        doc.append(
            f'<text x="{left - 8}" y="{cy + 4:.1f}" font-family="{SANS}" '
            f'font-size="{TICK_PX}" fill="{INK}" text-anchor="end">'
            f"{_esc(lbl)}</text>"
        )
        pts = [(v, c, n) for v, c, n in ((a, ca, name_a), (b, cb, name_b)) if v]
        pts.sort(key=lambda p: p[0])
        if len(pts) == 2:
            doc.append(
                f'<line x1="{x(pts[0][0]):.1f}" y1="{cy:.1f}" '
                f'x2="{x(pts[1][0]):.1f}" y2="{cy:.1f}" stroke="#CCCCCC" '
                f'stroke-width="2.5"/>'
            )
        for j, (v, c, _n) in enumerate(pts):
            doc.append(f'<circle cx="{x(v):.1f}" cy="{cy:.1f}" r="5" fill="{c}"/>')
            side = "end" if (len(pts) == 2 and j == 0) else "start"
            dx = -9 if side == "end" else 9
            doc.append(
                f'<text x="{x(v) + dx:.1f}" y="{cy + 4:.1f}" '
                f'font-family="{MONO}" font-size="{VALUE_PX}" fill="{INK}" '
                f'text-anchor="{side}">{_esc(_fmt(v, 2))} {_esc(unit)}</text>'
            )
    fy = plot_top + len(rows) * row_h + 34
    doc.append(
        f'<line x1="{left:.1f}" y1="{fy - 18}" x2="{right:.1f}" y2="{fy - 18}" '
        f'stroke="{AXIS}" stroke-width="1"/>'
    )
    if footnote:
        doc.append(
            f'<text x="0" y="{fy + 2:.1f}" font-family="{SANS}" '
            f'font-size="{FOOT_PX}" fill="{FOOT_FILL}">{_esc(footnote)}</text>'
        )
    return _svg(doc, W, fy + (12 if footnote else 0))


# ---------------------------------------------------------------------------
# campaign renderer


def scatter_svg(
    title: str,
    subtitle: str,
    points: list[tuple[float, float, str, str]],
    x_label: str,
    y_label: str,
    footnote: str | None = None,
) -> str:
    """Labeled dot scatter for tradeoff quadrants.

    points: (x, y, color, label) — caller sorts for deterministic draw
    order. Dashed zero lines appear on whichever axis spans zero; point
    labels alternate above/below the dot to limit collisions.
    """
    if not points:
        raise ValueError("scatter_svg needs at least one point")
    left, right, top, ph = 64.0, W - 28, 56.0, 240.0
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xlo, xhi = min(xs), max(xs)
    ylo, yhi = min(ys), max(ys)
    if xhi - xlo < 1e-9:
        xhi = xlo + 1.0
    if yhi - ylo < 1e-9:
        yhi = ylo + 1.0
    xlo, xhi = xlo - (xhi - xlo) * 0.08, xhi + (xhi - xlo) * 0.08
    ylo, yhi = ylo - (yhi - ylo) * 0.08, min(100.0, yhi + (yhi - ylo) * 0.08)

    def px(v: float) -> float:
        return left + (right - left) * ((v - xlo) / (xhi - xlo))

    def py(v: float) -> float:
        return top + ph - ph * ((v - ylo) / (yhi - ylo))

    doc = _header(title, subtitle)
    for t in _ticks(ylo, yhi):
        doc.append(
            f'<line x1="{left:.1f}" y1="{py(t):.1f}" x2="{right:.1f}" '
            f'y2="{py(t):.1f}" stroke="{GRID}" stroke-width="1"/>'
        )
        doc.append(
            f'<text x="{left - 8:.1f}" y="{py(t) + 4:.1f}" font-family="{SANS}" '
            f'font-size="{TICK_PX}" fill="{TICK_FILL}" text-anchor="end">'
            f"{_esc(_tick_label(t, False))}</text>"
        )
    for t in _ticks(xlo, xhi):
        doc.append(
            f'<text x="{px(t):.1f}" y="{top + ph + 16:.1f}" '
            f'font-family="{SANS}" font-size="{TICK_PX}" fill="{TICK_FILL}" '
            f'text-anchor="middle">{_esc(_tick_label(t, False))}</text>'
        )
    if xlo <= 0.0 <= xhi:
        doc.append(
            f'<line x1="{px(0):.1f}" y1="{top:.1f}" x2="{px(0):.1f}" '
            f'y2="{top + ph:.1f}" stroke="{BASELINE}" stroke-width="1.2" '
            f'stroke-dasharray="5 4"/>'
        )
    if ylo <= 0.0 <= yhi:
        doc.append(
            f'<line x1="{left:.1f}" y1="{py(0):.1f}" x2="{right:.1f}" '
            f'y2="{py(0):.1f}" stroke="{BASELINE}" stroke-width="1.2" '
            f'stroke-dasharray="5 4"/>'
        )
    doc.append(
        f'<line x1="{left:.1f}" y1="{top + ph:.1f}" x2="{right:.1f}" '
        f'y2="{top + ph:.1f}" stroke="{AXIS}" stroke-width="1"/>'
    )
    doc.append(
        f'<line x1="{left:.1f}" y1="{top:.1f}" x2="{left:.1f}" '
        f'y2="{top + ph:.1f}" stroke="{AXIS}" stroke-width="1"/>'
    )
    for i, (x, y, color, lbl) in enumerate(
        sorted(points, key=lambda p: (p[0], p[1], p[3]))
    ):
        dy = -10 if i % 2 == 0 else 18
        doc.append(
            f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="5" fill="{color}" '
            f'stroke="#FFFFFF" stroke-width="1.5"/>'
        )
        doc.append(
            f'<text x="{px(x):.1f}" y="{py(y) + dy:.1f}" font-family="{SANS}" '
            f'font-size="10" fill="{INK}" text-anchor="middle">'
            f"{_esc(_fmt(x))}%, {_esc(_fmt(y))}pp — {_esc(lbl)}</text>"
        )
    doc.append(
        f'<text x="{left:.1f}" y="{top - 10:.1f}" font-family="{SANS}" '
        f'font-size="{TICK_PX}" fill="{TICK_FILL}">{_esc(y_label)}</text>'
    )
    doc.append(
        f'<text x="{(left + right) / 2:.1f}" y="{top + ph + 34:.1f}" '
        f'font-family="{SANS}" font-size="{TICK_PX}" fill="{TICK_FILL}" '
        f'text-anchor="middle">{_esc(x_label)}</text>'
    )
    fy = top + ph + 52
    if footnote:
        doc.append(
            f'<text x="0" y="{fy:.1f}" font-family="{SANS}" '
            f'font-size="{FOOT_PX}" fill="{FOOT_FILL}">{_esc(footnote)}</text>'
        )
        fy += 14
    return _svg(doc, W, fy)


def _provenance(recs: list[dict], campaign: str) -> str:
    models: dict[str, int] = {}
    date = ""
    for r in recs:
        if r.get("model"):
            m = str(r["model"]).removesuffix(".d")
            models[m] = models.get(m, 0) + 1
        if not date and r.get("measured_at"):
            date = str(r["measured_at"])[:10]
    model = max(models.items(), key=lambda kv: kv[1])[0] if models else "?"
    return f"{model} · {date or '?'} · {campaign}"


def _ok_speed_rows(recs: list[dict]) -> list[dict]:
    """Headline-comparable speed cells: error-free, default config, direct
    rows pinned to the 1x16384 headline shape; ollama only when it served
    the same model (no reference_note)."""
    out = []
    for r in recs:
        if "error" in r:
            continue
        p = r.get("provider")
        params = r.get("params", {})
        if p == "blazar":
            cfg = params.get("config")
            if cfg and cfg != "default":
                continue
        elif p == "direct":
            if params.get("ctx") != 16384 or params.get("np") != 1:
                continue
        elif p == "ollama":
            if r.get("reference_note"):
                continue
        else:
            continue
        if r.get("decode_tps_p50") is None:
            continue
        out.append(r)
    return out


# Every SVG this module owns in <campaign>/plots/. A re-render that skips
# a lane (receipts no longer carry it) must not leave the previous run's
# file behind: a publication embedding it would show stale data.
_CHART_FILES = frozenset(
    {
        "speed-single-stream.svg",
        "concurrency-throughput.svg",
        "concurrency-ttft.svg",
        "ctx-curve.svg",
        "lifecycle-cold-idle.svg",
        "gateway-overhead.svg",
        "concurrency-vram.svg",
        "concurrency-power.svg",
        "concurrency-errors.svg",
        "vram-vs-context.svg",
        "quality-suites.svg",
        "quality-parity.svg",
        "quality-config.svg",
        "quality-tradeoff.svg",
    }
)


def render_campaign_charts(
    recs: list[dict], out_dir: Path
) -> list[tuple[str, str, str]]:
    """Render the publication chart set into <out_dir>/plots/.

    Returns (section_title, filename, caption) triples for markdown
    embedding; charts whose lane has no data are skipped with a log line
    (a lane legitimately not measured is not an error). Chart files from
    an earlier render whose lane went away are removed so the plots dir
    never holds a stale image the publication could embed.
    """
    out_dir = Path(out_dir)
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    for stale in sorted(plots.glob("*.svg")):
        if stale.name in _CHART_FILES:
            stale.unlink()
    campaign = out_dir.name
    prov = _provenance(recs, campaign)
    src = f"Source: {campaign}/cells.jsonl."
    charts: list[tuple[str, str, str]] = []

    def emit(fname: str, svg: str) -> str:
        (plots / fname).write_text(svg, encoding="utf-8")
        return fname

    # (a) single-stream decode bars + IQR, baseline = fastest direct engine
    speed = _ok_speed_rows(recs)
    if speed:
        rows = []
        for r in speed:
            runs = sorted(float(v) for v in (r.get("decode_tps_runs") or []))
            v = float(r["decode_tps_p50"])
            lo = _pctl(runs, 0.25) if runs else v
            hi = _pctl(runs, 0.75) if runs else v
            fam = r.get("provider") or "?"
            rows.append((f"{fam} · {r.get('tag', '?')}", v, lo, hi, fam))
        fams = sorted({fam for _, _, _, _, fam in rows})
        color_of = {}
        for fam in fams:
            members = sorted({lbl for lbl, *_, f in rows if f == fam})
            color_of[fam] = {m: _series_color(fam, i) for i, m in enumerate(members)}
        bars = [(lbl, v, lo, hi, color_of[fam][lbl]) for lbl, v, lo, hi, fam in rows]
        bars.sort(key=lambda b: b[1], reverse=True)
        # baseline from the provider records — bars carry colors, not families
        directs = [
            float(r["decode_tps_p50"]) for r in speed if r.get("provider") == "direct"
        ]
        base = (
            (f"fastest direct {_fmt(max(directs))} t/s", max(directs))
            if directs
            else None
        )
        emit(
            "speed-single-stream.svg",
            hbar_svg(
                "Single-stream decode throughput",
                prov,
                bars,
                "t/s",
                baseline=base,
                footnote="Bars: median of 5 runs. Whiskers: interquartile range. Higher is better.",
            ),
        )
        charts.append(
            (
                "Single-stream decode throughput (chart)",
                "speed-single-stream.svg",
                "Median decode t/s per runtime and engine; whiskers span the "
                "interquartile range of the 5 runs; the dashed line marks the "
                f"fastest direct engine. Higher is better. {src}",
            )
        )
    else:
        print("charts: skip speed bars (no comparable single-stream cells)")

    # (a2) gateway overhead: signed % decode-t/s delta, blazar vs direct on
    # the same engine build (tag); negative = blazar faster
    def _tps_by_tag(fam: str) -> dict[str, float]:
        acc: dict[str, list[float]] = {}
        for r in speed:
            if r.get("provider") == fam and r.get("decode_tps_p50") is not None:
                acc.setdefault(str(r.get("tag")), []).append(float(r["decode_tps_p50"]))
        return {t: _median(v) for t, v in acc.items()}

    direct_med = _tps_by_tag("direct")
    blazar_med = _tps_by_tag("blazar")
    overhead = [
        (
            tag,
            (blazar_med[tag] - direct_med[tag]) / direct_med[tag] * 100.0,
        )
        for tag in sorted(set(direct_med) & set(blazar_med))
        if direct_med[tag] > 0
    ]
    if overhead:
        overhead.sort(key=lambda kv: kv[1])
        bars = [(tag, d, d, d, FAMILY_COLOR["blazar"]) for tag, d in overhead]
        emit(
            "gateway-overhead.svg",
            hbar_svg(
                "Gateway overhead vs direct engine",
                prov,
                bars,
                "%",
                signed=True,
                footnote="Signed decode t/s delta, blazar vs the direct "
                "engine on the same engine build and model. Left of zero "
                "= blazar faster.",
            ),
        )
        charts.append(
            (
                "Gateway overhead (chart)",
                "gateway-overhead.svg",
                "Decode t/s delta of routing through blazar relative to "
                "driving the same engine build directly; left of zero means "
                f"the gateway path won. {src}",
            )
        )
    else:
        print("charts: skip gateway overhead (no blazar/direct pair per tag)")

    # (b)/(c) concurrency panels: throughput and worst-case TTFT vs level
    conc = [
        r
        for r in recs
        if str(r.get("provider", "")).startswith("conc-") and "error" not in r
    ]
    if conc:
        keys = sorted(
            {
                (str(r.get("provider")).removeprefix("conc-"), str(r.get("tag", "?")))
                for r in conc
            }
        )
        fam_color: dict[tuple[str, str], str] = {}
        fam_count: dict[str, int] = {}
        for fam, tag in keys:
            fam_count[fam] = fam_count.get(fam, 0) + 1
            fam_color[(fam, tag)] = _series_color(fam, fam_count[fam] - 1)
        tp_pts: dict[tuple[str, str], list[tuple[float, float]]] = {k: [] for k in keys}
        ttft_pts: dict[tuple[str, str], list[tuple[float, float]]] = {
            k: [] for k in keys
        }
        ttfb_pts: dict[tuple[str, str], list[tuple[float, float]]] = {
            k: [] for k in keys
        }
        vram_pts: dict[tuple[str, str], list[tuple[float, float]]] = {
            k: [] for k in keys
        }
        power_pts: dict[tuple[str, str], list[tuple[float, float]]] = {
            k: [] for k in keys
        }
        err_pts: dict[tuple[str, str], list[tuple[float, float]]] = {
            k: [] for k in keys
        }
        for r in conc:
            fam = str(r.get("provider")).removeprefix("conc-")
            key = (fam, str(r.get("tag", "?")))
            lvl = r.get("params", {}).get("conc")
            if lvl is None:
                continue
            if r.get("sys_tps") is not None:
                tp_pts[key].append((float(lvl), float(r["sys_tps"])))
            if r.get("ttft_max_ms") is not None:
                ttft_pts[key].append((float(lvl), float(r["ttft_max_ms"])))
            if r.get("ttfb_p50_ms") is not None:
                ttfb_pts[key].append((float(lvl), float(r["ttfb_p50_ms"])))
            if r.get("gpu_peak_mib") is not None:
                vram_pts[key].append((float(lvl), float(r["gpu_peak_mib"])))
            if r.get("gpu_power_peak_w") is not None:
                power_pts[key].append((float(lvl), float(r["gpu_power_peak_w"])))
            if r.get("conc_errors"):
                err_pts[key].append((float(lvl), float(r["conc_errors"])))
        labels = {k: f"{k[0]} · {k[1]}" for k in keys}

        def series_of(pts: dict) -> list:
            out = []
            for k in keys:
                if pts[k]:
                    out.append((labels[k], sorted(pts[k]), fam_color[k]))
            return out

        if any(tp_pts.values()):
            emit(
                "concurrency-throughput.svg",
                line_svg(
                    "System throughput vs concurrency",
                    prov,
                    series_of(tp_pts),
                    "concurrent streams",
                    "system t/s",
                    footnote="Each point: one sustained-load lane cell. Higher is better.",
                ),
            )
            charts.append(
                (
                    "Concurrency scaling (chart)",
                    "concurrency-throughput.svg",
                    "Aggregate system tokens/s as parallel streams are added; "
                    "flat-to-rising means the scheduler keeps the device "
                    f"saturated. Higher is better. {src}",
                )
            )
        else:
            print("charts: skip concurrency throughput panel (no sys_tps cells)")
        if any(ttft_pts.values()):
            ttft_series = series_of(ttft_pts)
            queue_foot = "Worst stream per level, log scale. Lower is better."
            queue_cap = ""
            if any(ttfb_pts.values()):
                for k in keys:
                    if ttfb_pts[k]:
                        ttft_series.append(
                            (
                                f"{labels[k]} · ttfb p50",
                                sorted(ttfb_pts[k]),
                                fam_color[k],
                                "6 4",
                            )
                        )
                queue_foot += (
                    " Dashed: median time-to-first-byte across streams -"
                    " the upper bound on queue wait under burst arrival."
                )
                queue_cap = (
                    " Dashed curves carry the median time-to-first-byte, "
                    "which upper-bounds queue wait under burst arrival."
                )
            emit(
                "concurrency-ttft.svg",
                line_svg(
                    "Worst-case first-token latency vs concurrency",
                    prov,
                    ttft_series,
                    "concurrent streams",
                    "TTFT max ms (log)",
                    log_y=True,
                    footnote=queue_foot,
                ),
            )
            charts.append(
                (
                    "Concurrency tail latency (chart)",
                    "concurrency-ttft.svg",
                    "Worst-case first-token wait per stream as concurrency "
                    "rises (log scale) - the tail the scheduler must bound. "
                    f"Lower is better.{queue_cap} {src}",
                )
            )
        else:
            print("charts: skip concurrency TTFT panel (no ttft_max_ms cells)")

        # (c2) resource cost vs concurrency: VRAM, GPU power, error counts
        if any(vram_pts.values()):
            emit(
                "concurrency-vram.svg",
                line_svg(
                    "GPU memory vs concurrency",
                    prov,
                    series_of(vram_pts),
                    "concurrent streams",
                    "VRAM peak MiB",
                    footnote="Peak device memory per sustained-load cell.",
                ),
            )
            charts.append(
                (
                    "Resource cost vs concurrency (chart)",
                    "concurrency-vram.svg",
                    "Peak VRAM footprint as parallel streams (and their KV "
                    f"caches) stack up. {src}",
                )
            )
        else:
            print("charts: skip conc VRAM panel (no gpu_peak_mib in conc cells)")
        if any(power_pts.values()):
            emit(
                "concurrency-power.svg",
                line_svg(
                    "GPU power vs concurrency",
                    prov,
                    series_of(power_pts),
                    "concurrent streams",
                    "GPU power peak W",
                    footnote="Peak board power per sustained-load cell.",
                ),
            )
            charts.append(
                (
                    "Resource cost vs concurrency (chart)",
                    "concurrency-power.svg",
                    "Peak GPU board power per concurrency level - the energy "
                    f"price of keeping the device saturated. {src}",
                )
            )
        else:
            print("charts: skip conc power panel (no gpu_power_peak_w in conc cells)")
        if any(err_pts.values()):
            emit(
                "concurrency-errors.svg",
                line_svg(
                    "Failed requests vs concurrency",
                    prov,
                    series_of(err_pts),
                    "concurrent streams",
                    "failed requests",
                    footnote="Requests that returned an error or timed out "
                    "per sustained-load cell. Lower is better (zero is the "
                    "goal).",
                ),
            )
            charts.append(
                (
                    "Reliability vs concurrency (chart)",
                    "concurrency-errors.svg",
                    "Failed requests per concurrency level; lanes that never "
                    f"failed are omitted. Lower is better. {src}",
                )
            )
        else:
            n_conc_cells = len(conc)
            print(
                f"charts: zero failed requests across {n_conc_cells} conc "
                "cells - errors panel omitted"
            )
    else:
        print("charts: skip concurrency panels (no conc cells)")

    # (d) long-context degradation curve
    ctx = [
        r
        for r in recs
        if str(r.get("provider", "")).startswith("ctxcurve-")
        and "error" not in r
        and r.get("ctx") is not None
        and r.get("decode_tps_p50") is not None
    ]
    if ctx:
        keys = sorted(
            {
                (
                    str(r.get("provider")).removeprefix("ctxcurve-"),
                    str(r.get("tag", "?")),
                )
                for r in ctx
            }
        )
        pts: dict[tuple[str, str], list[tuple[float, float]]] = {k: [] for k in keys}
        for r in ctx:
            key = (
                str(r.get("provider")).removeprefix("ctxcurve-"),
                str(r.get("tag", "?")),
            )
            pts[key].append((float(r["ctx"]), float(r["decode_tps_p50"])))
        series = [
            (f"{k[0]} · {k[1]}", sorted(pts[k]), _series_color(k[0], 0))
            for k in keys
            if pts[k]
        ]
        if series:
            emit(
                "ctx-curve.svg",
                line_svg(
                    "Decode throughput vs context length",
                    prov,
                    series,
                    "context tokens (log)",
                    "decode t/s",
                    log_x=True,
                    footnote="Long-context points, log x-axis. Higher is better.",
                ),
            )
            charts.append(
                (
                    "Long-context degradation (chart)",
                    "ctx-curve.svg",
                    "Single-stream decode t/s as prompt context grows (log "
                    f"x-axis) - KV-cache pressure made visible. {src}",
                )
            )
        else:
            print("charts: skip ctx curve (no plottable points)")

        # (d2) memory pressure: peak VRAM along the same context ladder
        vram_ctx: dict[tuple[str, str], list[tuple[float, float]]] = {}
        for r in ctx:
            if r.get("gpu_peak_mib") is None:
                continue
            key = (
                str(r.get("provider")).removeprefix("ctxcurve-"),
                str(r.get("tag", "?")),
            )
            vram_ctx.setdefault(key, []).append(
                (float(r["ctx"]), float(r["gpu_peak_mib"]))
            )
        vram_series = [
            (f"{k[0]} · {k[1]}", sorted(vram_ctx[k]), _series_color(k[0], 0))
            for k in sorted(vram_ctx)
            if vram_ctx[k]
        ]
        if vram_series:
            emit(
                "vram-vs-context.svg",
                line_svg(
                    "GPU memory vs context length",
                    prov,
                    vram_series,
                    "context tokens (log)",
                    "VRAM peak MiB",
                    log_x=True,
                    footnote="Peak device memory per context cell, log "
                    "x-axis. The KV-cache slope is the capacity ceiling.",
                ),
            )
            charts.append(
                (
                    "Memory vs context (chart)",
                    "vram-vs-context.svg",
                    "Peak VRAM as prompt context grows (log x-axis) - the "
                    "KV-cache slope that sets the usable context ceiling. "
                    f"{src}",
                )
            )
        else:
            print("charts: skip vram-vs-context (no gpu_peak_mib in ctxcurve cells)")
    else:
        print("charts: skip ctx curve (no ctxcurve cells)")

    # (e) lifecycle panel: cold start + idle wake, log seconds
    def _med(recs_in: list[dict], field: str) -> float | None:
        vals = [r[field] for r in recs_in if r.get(field) is not None]
        return _median(vals) if vals else None

    cold_blazar = _med(
        [r for r in recs if r.get("provider") == "cold-blazar" and "error" not in r],
        "cold_ttft_ms",
    )
    if cold_blazar is None:
        cold_blazar = _med(
            [r for r in recs if r.get("provider") == "blazar" and "error" not in r],
            "cold_ttft_ms",
        )
    cold_ollama = _med(
        [r for r in recs if r.get("provider") == "cold-ollama" and "error" not in r],
        "ollama_cold_ttft_ms",
    )
    idle_blazar = _med(
        [r for r in recs if r.get("provider") == "idle-blazar" and "error" not in r],
        "idle_wake_ttft_ms",
    )
    idle_ollama = _med(
        [r for r in recs if r.get("provider") == "idle-ollama" and "error" not in r],
        "idle_wake_ttft_ms",
    )
    life_rows_spec = [
        ("cold start to first token", cold_blazar, cold_ollama),
        ("idle wake to first token", idle_blazar, idle_ollama),
    ]
    life_rows_spec = [(label, a, b) for label, a, b in life_rows_spec if a or b]
    if life_rows_spec:
        ca, cb = FAMILY_COLOR["blazar"], FAMILY_COLOR["ollama"]
        rows = [
            (
                lbl,
                (a / 1000) if a else None,
                (b / 1000) if b else None,
                ca,
                cb,
            )
            for lbl, a, b in life_rows_spec
        ]
        warm_note = any(
            "left warm" in str(r.get("ollama_daemon_boot_note", "")) for r in recs
        )
        foot = "Dots: median across cells, log scale. Lower is better."
        if warm_note:
            foot += " ollama cold row reflects a warm daemon (no service restart in this campaign)."
        emit(
            "lifecycle-cold-idle.svg",
            dumbbell_svg(
                "Cold start and idle wake",
                prov,
                rows,
                "s",
                "blazar",
                "ollama",
                footnote=foot,
            ),
        )
        charts.append(
            (
                "Lifecycle: cold start and idle wake (chart)",
                "lifecycle-cold-idle.svg",
                "Seconds to first token after a cold start (page cache "
                "dropped) and after idle-policy expiry; blazar keeps weights "
                "resident while ollama reloads from disk. "
                + ("Warm-daemon ollama caveat applies. " if warm_note else "")
                + f"Lower is better. {src}",
            )
        )
    else:
        print("charts: skip lifecycle panel (no cold/idle cells)")

    # (f) quality suites: checker-verified correctness on the same seeded
    # task set across providers. Embed is a per-engine parity metric
    # (vector spaces differ across engines), so it joins the parity chart
    # but never the absolute-rate or overall rows.
    Q_SUITES = (
        "reason",
        "instruct",
        "code",
        "schema",
        "niah",
        "multilingual",
        "safety",
    )
    q_fam = {
        "quality": "blazar",
        "quality-direct": "direct",
        "quality-ollama": "ollama",
    }
    qcells = [r for r in recs if r.get("provider") in q_fam and "error" not in r]

    def q_cfg(r: dict) -> str:
        return str(r.get("params", {}).get("config", "default"))

    def q_rate(r: dict, suite: str) -> float | None:
        v = r.get(f"quality_{suite}_rate")
        return float(v) * 100.0 if v is not None else None

    def q_overall(r: dict) -> float | None:
        p = t = 0
        for s in Q_SUITES:
            tt = r.get(f"quality_{s}_total")
            if tt:
                p += r.get(f"quality_{s}_pass") or 0
                t += tt
        return (p / t * 100.0) if t else None

    q_default = [
        r
        for r in qcells
        if r.get("provider") in ("quality-direct", "quality-ollama")
        or (r.get("provider") == "quality" and q_cfg(r) == "default")
    ]

    # (f1) absolute pass rates per suite x provider
    suites_seen = sorted(
        {
            s
            for r in q_default
            for s in Q_SUITES
            if r.get(f"quality_{s}_rate") is not None
        }
    )
    if suites_seen:
        rows = []
        for s in suites_seen:
            for pname in ("quality-direct", "quality", "quality-ollama"):
                vals = [q_rate(r, s) for r in q_default if r.get("provider") == pname]
                vals = [v for v in vals if v is not None]
                if vals:
                    rows.append(
                        (
                            f"{s} · {q_fam[pname]}",
                            _median(vals),
                            _median(vals),
                            _median(vals),
                            FAMILY_COLOR[q_fam[pname]],
                        )
                    )
        if rows:
            emit(
                "quality-suites.svg",
                hbar_svg(
                    "Quality suite pass rates",
                    prov,
                    rows,
                    "%",
                    baseline=("100 % pass", 100.0),
                    footnote="Checker-verified pass rate per suite and "
                    "runtime, same seeded task set. Higher is better; "
                    "ollama row is its own same-family model build.",
                ),
            )
            charts.append(
                (
                    "Quality suites (chart)",
                    "quality-suites.svg",
                    "Deterministic-checker pass rates per suite (reasoning, "
                    "instruction following, code, schema, needle-in-haystack, "
                    "multilingual, safety) across runtimes on the identical "
                    f"seeded task set. {src}",
                )
            )
        else:
            print("charts: skip quality suites (no rate fields)")
    else:
        print("charts: skip quality suites (no quality cells)")

    # (f2) gateway quality parity: blazar minus direct, percentage points
    q_pairs: list[tuple[str, float]] = []
    for suite in (*Q_SUITES, "embed"):
        for tag in sorted({str(r.get("tag")) for r in q_default}):
            b = [
                q_rate(r, suite)
                for r in q_default
                if r.get("provider") == "quality" and r.get("tag") == tag
            ]
            d = [
                q_rate(r, suite)
                for r in q_default
                if r.get("provider") == "quality-direct" and r.get("tag") == tag
            ]
            b = [v for v in b if v is not None]
            d = [v for v in d if v is not None]
            if b and d:
                q_pairs.append((f"{suite} · {tag}", _median(b) - _median(d)))
    if q_pairs:
        q_pairs.sort(key=lambda kv: kv[1])
        bars = [(lbl, d, d, d, FAMILY_COLOR["blazar"]) for lbl, d in q_pairs]
        emit(
            "quality-parity.svg",
            hbar_svg(
                "Gateway quality parity (blazar - direct)",
                prov,
                bars,
                "pp",
                signed=True,
                footnote="Pass-rate delta in percentage points through the "
                "gateway vs the same engine driven directly. Zero axis = "
                "transparent routing.",
            ),
        )
        charts.append(
            (
                "Gateway quality parity (chart)",
                "quality-parity.svg",
                "Per-suite pass-rate delta of routing through blazar vs the "
                "same engine build driven directly; bars hugging the zero "
                f"axis are the gateway being output-transparent. {src}",
            )
        )
    else:
        print("charts: skip quality parity (no blazar/direct pair)")

    # (f3) quality vs configuration knob: overall-rate delta per axis cell
    q_axes = [
        r for r in qcells if r.get("provider") == "quality" and q_cfg(r) != "default"
    ]
    cfg_rows: list[tuple[str, float]] = []
    for r in q_axes:
        tag = str(r.get("tag"))
        base = [
            q_overall(x)
            for x in q_default
            if x.get("provider") == "quality" and x.get("tag") == tag
        ]
        base = [v for v in base if v is not None]
        cur = q_overall(r)
        if base and cur is not None:
            cfg_rows.append((f"{tag} · {q_cfg(r)}", cur - _median(base)))
    if cfg_rows:
        cfg_rows.sort(key=lambda kv: kv[1])
        bars = [(lbl, d, d, d, FAMILY_COLOR["blazar"]) for lbl, d in cfg_rows]
        emit(
            "quality-config.svg",
            hbar_svg(
                "Quality vs gateway configuration",
                prov,
                bars,
                "pp",
                signed=True,
                footnote="Overall suite pass-rate delta per gateway config "
                "knob vs the default config, same seed and tasks. Left of "
                "zero = the knob costs correctness.",
            ),
        )
        charts.append(
            (
                "Quality vs configuration (chart)",
                "quality-config.svg",
                "Overall pass-rate delta per gateway knob (KV quantization, "
                "attention, batching) against the default config on the "
                f"identical seeded task set. {src}",
            )
        )
    else:
        print("charts: skip quality config (no axis cells)")

    # (f4) quality/latency tradeoff quadrant: speed delta vs quality delta
    trade: list[tuple[float, float, str, str]] = []
    for r in q_axes:
        tag = str(r.get("tag"))
        label = q_cfg(r)
        spd = [
            x
            for x in recs
            if x.get("provider") == "blazar"
            and x.get("tag") == tag
            and x.get("params", {}).get("config") == label
            and "error" not in x
            and x.get("decode_tps_p50") is not None
        ]
        spd_base = [
            x
            for x in recs
            if x.get("provider") == "blazar"
            and x.get("tag") == tag
            and x.get("params", {}).get("config") == "default"
            and "error" not in x
            and x.get("decode_tps_p50") is not None
        ]
        q_base = [
            q_overall(x)
            for x in q_default
            if x.get("provider") == "quality" and x.get("tag") == tag
        ]
        q_base = [v for v in q_base if v is not None]
        cur = q_overall(r)
        if not (spd and spd_base and q_base and cur is not None):
            continue
        s_now = float(spd[0]["decode_tps_p50"])
        s_base = float(spd_base[0]["decode_tps_p50"])
        if s_base > 0:
            trade.append(
                (
                    (s_now - s_base) / s_base * 100.0,
                    cur - _median(q_base),
                    FAMILY_COLOR["blazar"],
                    f"{tag} · {label}",
                )
            )
    if trade:
        emit(
            "quality-tradeoff.svg",
            scatter_svg(
                "Quality / latency tradeoff per configuration",
                prov,
                trade,
                "decode t/s delta vs default (%)",
                "overall pass-rate delta (pp)",
                footnote="Each dot is one gateway config knob: x = speed "
                "delta, y = quality delta vs the default. Top-right = "
                "strictly better; bottom-right = speed bought with "
                "correctness.",
            ),
        )
        charts.append(
            (
                "Quality / latency tradeoff (chart)",
                "quality-tradeoff.svg",
                "Per-knob tradeoff against the default configuration: "
                "horizontal axis is decode-speed delta, vertical axis is "
                "overall checker pass-rate delta. A configuration is "
                "production-ready only when its quality delta stays within "
                f"tolerance. {src}",
            )
        )
    else:
        print("charts: skip quality tradeoff (no joined speed+quality axis cells)")

    return charts
