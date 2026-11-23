#!/usr/bin/env python3
r"""Re-resolve the line references in docs/config-code-paths.md.

The doc's per-field rows cite consumption sites as
`` `path.rs:LINE` in `fn_name` ``. Field coverage is guarded by
check_config_doc_drift.py (names, not lines); what drifts is every
LINE number as the tree moves. This script re-anchors each citation
to the current line of the cited function:

- plain names: `` rg -n '^(\s*)(pub )?(async )?fn NAME\b' path `` and
  take the first match in the cited file;
- if the function left the cited file (moves like the config-edit
  relocation), it is searched across crates/ and the citation's path
  is rewritten too;
- non-resolvable anchors (generator artifacts such as `` the `` or
  `` ? ``, or deleted functions) are left untouched and reported, so
  the output stays honest about what it could not prove.

Idempotent; front matter, prose, tables' field/lane columns, and any
hand-written notes are preserved byte-for-byte. Exit code is 0 even
with unresolved anchors (run with --strict to fail on them).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DOC = REPO / "docs" / "config-code-paths.md"

CITE = re.compile(r"`(?P<path>[^`\s]+\.rs):(?P<line>\d+)` in `(?P<fn>[^`]+)`")
FN = "rg --no-heading --line-number -- '^\\s*(?:pub(?:\\(crate\\))? )?(?:async )?(?:unsafe )?fn {name}\\b' {path}"
JUNK_ANCHORS = {"the", "?", "", "a", "it"}
STAMP = re.compile(r"Generated \d{4}-\d{2}-\d{2} against working tree `[^`]+`")


def repo_rel(raw: str) -> Path:
    p = Path(raw)
    return (
        p
        if p.is_absolute()
        else (
            REPO / ("crates/" + raw if not raw.startswith("crates/") else raw)
        ).resolve()
    )


def fn_line(name: str, path: Path) -> int | None:
    if not path.is_file():
        return None
    out = subprocess.run(
        FN.format(name=re.escape(name), path=path),
        shell=True,
        capture_output=True,
        text=True,
        cwd=REPO,
    ).stdout.strip()
    if not out:
        return None
    return int(out.splitlines()[0].split(":", 1)[0])


def find_across_crates(name: str) -> Path | None:
    out = subprocess.run(
        FN.format(name=re.escape(name), path="crates"),
        shell=True,
        capture_output=True,
        text=True,
        cwd=REPO,
    ).stdout.strip()
    if not out:
        return None
    hit = Path(out.splitlines()[0].split(":", 1)[0])
    return hit.resolve()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero when any anchor stays unresolved",
    )
    args = ap.parse_args()

    text = DOC.read_text()

    # (path-as-cited, fn) -> resolved (repo-relative path, line)
    cache: dict[tuple[str, str], tuple[str, int] | None] = {}
    stats = {"updated": 0, "moved": 0, "junk": 0, "unresolved": 0, "unchanged": 0}

    def resolve(cited_path: str, fn: str) -> tuple[str, int] | None:
        key = (cited_path, fn)
        if key in cache:
            return cache[key]
        rel = cited_path if cited_path.startswith("crates/") else "crates/" + cited_path
        target = repo_rel(cited_path)
        hit = fn_line(fn, target) if fn not in JUNK_ANCHORS else None
        if hit is not None:
            cache[key] = (rel, hit)
            return cache[key]
        if fn not in JUNK_ANCHORS:
            elsewhere = find_across_crates(fn)
            if elsewhere is not None:
                hit = fn_line(fn, elsewhere)
                if hit is not None:
                    rel_new = str(elsewhere.relative_to(REPO))
                    cache[key] = (rel_new, hit)
                    return cache[key]
        cache[key] = None
        return None

    def sub(m: re.Match) -> str:
        cited, old_line, fn = m.group("path"), int(m.group("line")), m.group("fn")
        if fn in JUNK_ANCHORS:
            stats["junk"] += 1
            return m.group(0)
        got = resolve(cited, fn)
        if got is None:
            stats["unresolved"] += 1
            return m.group(0)
        new_rel, new_line = got
        changed_path = new_rel != cited and new_rel != "crates/" + cited
        if changed_path:
            stats["moved"] += 1
        elif new_line != old_line:
            stats["updated"] += 1
        else:
            stats["unchanged"] += 1
        return f"`{new_rel}:{new_line}` in `{fn}`"

    new_text = CITE.sub(sub, text)
    import datetime

    stamp = f"Generated {datetime.date.today().isoformat()} against working tree `regen via scripts/regen_config_code_paths.py`"
    if STAMP.search(new_text):
        new_text = STAMP.sub(stamp, new_text, count=1)

    DOC.write_text(new_text)
    print(f"citations: {stats}")
    if args.strict and stats["unresolved"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
