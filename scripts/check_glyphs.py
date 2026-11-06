#!/usr/bin/env python3
"""Glyph gate: no new CJK-rendering content may land in this repo.

Two tiers, both reporting file:line:codepoint (never the glyph itself,
so gate output can never render CJK fallback in any terminal):

Tier 1 (repo-wide, committed content): CJK script - ideographs, kana,
hangul, fullwidth forms, CJK punctuation. Legitimate linguistic test
data and historical records are allowlisted below (each entry verified
to hold CJK as DATA, not prose).

Tier 2 (added lines only): the punctuation render class - dashes,
ellipsis, arrows, math operators, middle dot, smart quotes, micro,
multiplication/division signs. These render via CJK fallback fonts and
are banned from NEW lines everywhere (docs/research historical text is
grandfathered; new authoring is not). Sources of "added lines":
staged diff, else working-tree diff, plus untracked files. In CI
(GITHUB_SHA set), the diff is taken against the merge-base with
origin/main when available, else tier 2 is skipped with a notice.

Exempt classes everywhere: box drawing U+2500-257F and the check/cross
marks U+2713/U+2717/U+2705 (native terminal glyphs, render fine).
Binary files are skipped by content sniff.

Usage: check_glyphs.py            (repo-anchored, both tiers)
       check_glyphs.py PATH ...   (tier-1 scan of given paths only)
Exit 0 clean, 1 violations, 2 usage/infra error.
"""

from __future__ import annotations

import os
import subprocess
import sys

# CJK-as-data sites (verified holdings): gateway language-test sources,
# their integration fixtures, historical research record, brand assets.
CJK_DATA_ALLOWLIST = {
    "crates/blazar-gateway/src/prompt_recipe.rs",
    "crates/blazar-gateway/src/scrub.rs",
    "crates/blazar-gateway/src/translate.rs",
    "crates/blazar-gateway/tests/gateway.rs",
    "docs/research/2026-10-20-deepseek-deepinfer-comparison.md",
}
CJK_DATA_PREFIXES = ("docs/assets/", "bench-artifacts/")

# Historical records: tier 2 never applies to their existing content.
TIER2_GRANDFATHER_PREFIXES = ("docs/research/", "bench-artifacts/", "docs/assets/")
TIER2_GRANDFATHER_FILES = {"CHANGELOG.md"}

TIER1_RANGES = (
    (0x3000, 0x303F),  # CJK punctuation
    (0x3040, 0x30FF),  # kana
    (0x4E00, 0x9FFF),  # ideographs
    (0xAC00, 0xD7AF),  # hangul
    (0xFF00, 0xFFEF),  # fullwidth forms
)
TIER2_CODEPOINTS = frozenset(
    [
        0x2014,
        0x2013,
        0x2026,
        0x00B7,
        0x00D7,
        0x00F7,
        0x00B5,
        0x2248,
        0x2265,
        0x2264,
        0x2260,
        0x2261,
        0x00B1,
        # arrows (2190-21FF) and math operators (2200-22FF) covered as
        # ranges below; smart quotes join the explicit set
        0x2018,
        0x2019,
        0x201C,
        0x201D,
    ]
)
TIER2_RANGES = ((0x2190, 0x21FF), (0x2200, 0x22FF))
EXEMPT_CODEPOINTS = frozenset((0x2713, 0x2717, 0x2705))
EXEMPT_RANGES = ((0x2500, 0x257F),)


def in_ranges(ch: str, ranges) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in ranges)


def is_tier1(ch: str) -> bool:
    return in_ranges(ch, TIER1_RANGES)


def is_tier2(ch: str) -> bool:
    cp = ord(ch)
    return cp in TIER2_CODEPOINTS or in_ranges(ch, TIER2_RANGES)


def is_exempt(ch: str) -> bool:
    cp = ord(ch)
    return cp in EXEMPT_CODEPOINTS or in_ranges(ch, EXEMPT_RANGES)


def tier1_path_allowed(path: str) -> bool:
    return path in CJK_DATA_ALLOWLIST or path.startswith(CJK_DATA_PREFIXES)


def tier2_path_grandfathered(path: str) -> bool:
    return path in TIER2_GRANDFATHER_FILES or path.startswith(
        TIER2_GRANDFATHER_PREFIXES
    )


def looks_binary(path: str) -> bool:
    try:
        with open(path, "rb") as fh:
            return b"\x00" in fh.read(4096)
    except OSError:
        return True


def scan_text_lines(path: str, lines: list[str], tier2: bool) -> list[str]:
    findings: list[str] = []
    allow1 = tier1_path_allowed(path)
    gf2 = not tier2 or tier2_path_grandfathered(path)
    for lineno, line in enumerate(lines, 1):
        for ch in line:
            if ord(ch) <= 127 or is_exempt(ch):
                continue
            if is_tier1(ch):
                if not allow1:
                    findings.append(
                        f"{path}:{lineno}: U+{ord(ch):04X} (CJK script, not allowlisted)"
                    )
            elif is_tier2(ch) and tier2 and not gf2:
                findings.append(
                    f"{path}:{lineno}: U+{ord(ch):04X} (render-fallback punctuation on a new line)"
                )
    return findings


def tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, check=True
    ).stdout
    return [f for f in out.splitlines() if f]


def added_lines_from_diff(diff_text: str) -> dict[str, list[str]]:
    """Map path -> added lines from a unified diff."""
    result: dict[str, list[str]] = {}
    path = None
    for line in diff_text.splitlines():
        if line.startswith("+++ b/"):
            path = line[6:]
            result.setdefault(path, [])
        elif line.startswith("+") and not line.startswith("+++") and path:
            result[path].append(line[1:])
    return result


def main() -> int:
    findings: list[str] = []
    explicit = sys.argv[1:]
    if explicit:
        for path in explicit:
            if looks_binary(path):
                continue
            try:
                with open(path, encoding="utf-8", errors="replace") as fh:
                    findings.extend(
                        scan_text_lines(path, fh.read().splitlines(), False)
                    )
            except OSError as exc:
                print(f"glyph gate: cannot read {path}: {exc}", file=sys.stderr)
                return 2
    else:
        # Tier 1: every tracked file, committed content.
        for path in tracked_files():
            if looks_binary(path):
                continue
            try:
                with open(path, encoding="utf-8", errors="replace") as fh:
                    findings.extend(
                        scan_text_lines(path, fh.read().splitlines(), False)
                    )
            except OSError:
                continue
        # Tier 2: new lines only. CI diffs against origin/main when
        # that ref exists; locally prefers staged, else working tree,
        # and always includes untracked files wholesale.
        diff_text = ""
        if os.environ.get("GITHUB_SHA"):
            base = subprocess.run(
                check=False,
                args=[
                    "git",
                    "rev-parse",
                    "--verify",
                    "-q",
                    "origin/main",
                ],
                capture_output=True,
                text=True,
            ).stdout.strip()
            if base:
                merge_base = subprocess.run(
                    check=False,
                    args=["git", "merge-base", base, "HEAD"],
                    capture_output=True,
                    text=True,
                ).stdout.strip()
                if merge_base:
                    diff_text = subprocess.run(
                        check=False,
                        args=["git", "diff", f"{merge_base}..HEAD"],
                        capture_output=True,
                        text=True,
                    ).stdout
        else:
            staged = subprocess.run(
                ["git", "diff", "--cached"], capture_output=True, text=True, check=False
            ).stdout
            if staged:
                diff_text = staged
            else:
                diff_text = subprocess.run(
                    ["git", "diff"], capture_output=True, text=True, check=False
                ).stdout
        for path, lines in added_lines_from_diff(diff_text).items():
            if looks_binary(path):
                continue
            findings.extend(scan_text_lines(path, lines, True))
        untracked = subprocess.run(
            check=False,
            args=[
                "git",
                "ls-files",
                "--others",
                "--exclude-standard",
            ],
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        for path in untracked:
            if looks_binary(path):
                continue
            try:
                with open(path, encoding="utf-8", errors="replace") as fh:
                    findings.extend(scan_text_lines(path, fh.read().splitlines(), True))
            except OSError:
                continue

    if findings:
        for f in findings[:200]:
            print(f, file=sys.stderr)
        print(
            f"glyph gate: {len(findings)} violation(s) - CJK script or "
            f"render-fallback punctuation (see file:line:codepoint above)",
            file=sys.stderr,
        )
        return 1
    print("glyph gate: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
