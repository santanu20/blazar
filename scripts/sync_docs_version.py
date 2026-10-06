#!/usr/bin/env python3
"""Keep the GitHub Pages site's version strings in lockstep with releases.

The site (docs/, served via GitHub Pages from main) carries the release
version in exactly five places: the hero eyebrow, the two install
command URLs, and the demo banner in index.html plus its SVG twin.
These are "current release" claims — a stale one is the site lying
about what shipping today looks like. Verification-date stamps inside
the markdown pages ("spot-verified against vX on DATE") are historical
records and are deliberately NOT touched.

Mode:
  default   rewrite the five spots to the workspace version
  --check   exit 1 if any spot disagrees with --expect (CI gate)

The workspace version in Cargo.toml is the single source of truth;
the release-prep commit runs this alongside the version bump, and the
release gate runs --check so a forgotten sync fails the tag build.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# (file, pattern with one version group, what it is) — every pattern
# must match exactly once; a drift in the surrounding markup is a
# rewrite-the-sync problem, never a silent skip.
VERSION_SPOTS: list[tuple[str, str, str]] = [
    (
        "docs/index.html",
        r'(hero-eyebrow">v)(\d+\.\d+\.\d+)( ·)',
        "hero eyebrow version",
    ),
    (
        "docs/index.html",
        r"(raw\.githubusercontent\.com/\S+?/blazar/v)(\d+\.\d+\.\d+)(/scripts/install\.sh)",
        "linux/macOS install URL",
    ),
    (
        "docs/index.html",
        r"(raw\.githubusercontent\.com/\S+?/blazar/v)(\d+\.\d+\.\d+)(/scripts/install\.ps1)",
        "windows install URL",
    ),
    (
        "docs/index.html",
        r"(blazar )(\d+\.\d+\.\d+)( — powered by)",
        "index.html demo banner",
    ),
    (
        "docs/assets/blazar-demo.svg",
        r"(blazar )(\d+\.\d+\.\d+)( — powered by)",
        "SVG demo banner",
    ),
]


def workspace_version() -> str:
    # Parsed by hand rather than with tomllib: the release-gate runners
    # ship Python 3.10, where tomllib does not exist yet. The root
    # manifest carries the version under [workspace.package]; a
    # plain-package layout would put it under [package] directly.
    text = (REPO / "Cargo.toml").read_text(encoding="utf-8")
    for section in ("workspace.package", "package"):
        header = re.search(
            rf"^\[{re.escape(section)}\]\s*$",
            text,
            re.MULTILINE,
        )
        if header is None:
            continue
        body = text[header.end() :]
        end = re.search(r"^\[", body, re.MULTILINE)
        version = re.search(
            r'^version\s*=\s*"([^"]+)"',
            body[: end.start()] if end else body,
            re.MULTILINE,
        )
        if version is not None:
            return version.group(1)
    raise SystemExit(
        "Cargo.toml: no version found under [workspace.package] or [package]"
    )


def check_or_sync(target: str, check_only: bool) -> int:
    failures: list[str] = []
    for rel, pattern, label in VERSION_SPOTS:
        path = REPO / rel
        try:
            body = path.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"FAIL {rel}: unreadable ({exc})")
            return 1
        rx = re.compile(pattern)
        matches = rx.findall(body)
        if len(matches) != 1:
            failures.append(
                f"{rel}: {label}: expected exactly 1 match, found {len(matches)}"
            )
            continue
        found = matches[0][1]
        if found == target:
            continue
        if check_only:
            failures.append(f"{rel}: {label}: shows v{found}, expected v{target}")
            continue
        body = rx.sub(lambda m: m.group(1) + target + m.group(3), body)
        path.write_text(body, encoding="utf-8")
        print(f"sync {rel}: {label}: v{found} -> v{target}")
    if failures:
        for line in failures:
            print(f"FAIL {line}")
        print(
            "run scripts/sync_docs_version.py in the release-prep commit "
            "(docs version must equal the tag)"
        )
        return 1
    if check_only:
        print(f"docs version spots all at v{target}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check",
        action="store_true",
        help="verify only; exit 1 when any spot disagrees with --expect",
    )
    ap.add_argument(
        "--expect",
        default=workspace_version(),
        help="version the spots must carry (default: workspace version)",
    )
    args = ap.parse_args()
    return check_or_sync(args.expect, args.check)


if __name__ == "__main__":
    sys.exit(main())
