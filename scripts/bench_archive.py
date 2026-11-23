#!/usr/bin/env python3
"""Keep ``bench-artifacts/`` in step with what the tree cites.

The repo tracks only the bench receipts that a registry, a doc page, the README
or a code comment cites, plus ``INDEX.md``. Everything else under
``bench-artifacts/`` belongs on the bench-archive GitHub release, not in git.

A run directory is *cited* when a tracked file outside ``bench-artifacts/``,
``docs/research/``, ``docs/issues/`` and ``CHANGELOG.md`` mentions
``bench-artifacts/<run>`` or a file inside it. Every tracked file of a cited
run stays tracked; every tracked file of an uncited run is archived. Markdown
reports that sit untracked inside a cited run are added, because the ignore
rules hide them from ``git status``.

Modes:
  (no flag)   print the plan
  --check     exit 1 when the plan is not empty (CI gate)
  --apply     write the archive tarball and its .sha256 sidecar, then update the
              index: ``git rm --cached`` for the archive set, ``git add -f``
              for the add set. Never uploads, never commits.
  --selftest  run the fixture test

After ``--apply`` commit the index and upload the tarball with its sidecar:

    gh release upload <archive release> <tarball> <sha256>

``scripts/engine_coverage.py --check`` fails when a citation points at a
receipt that is no longer tracked, so run it before committing.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

BENCH_DIR = "bench-artifacts"
INDEX = f"{BENCH_DIR}/INDEX.md"
# The GitHub release that carries the archived reports.
ARCHIVE_RELEASE = "bench-archive-2026-10"
# Mentions inside these files describe history or other working notes; they do
# not make a run part of the published record.
NON_CITING_PREFIXES = (f"{BENCH_DIR}/", "docs/research/", "docs/issues/")
NON_CITING_FILES = ("CHANGELOG.md",)
# Reports are tracked; charts only where a document embeds them by name.
REPORT_SUFFIX = ".md"
CITABLE_SUFFIXES = (".md", ".svg")
# Plain ERE on purpose: the same pattern is handed to ``git grep -E``.
MENTION_ER = f"{BENCH_DIR}/[A-Za-z0-9._+-]+(/[A-Za-z0-9._+-]+)*"
# Credential shapes are refused before a blob enters a public archive.
CREDENTIAL = re.compile(
    r"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"
    r"|(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]{32,}|AKIA[0-9A-Z]{16}"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----"
)


def git(
    repo: Path, *args: str, stdin: bytes | None = None, ok: tuple[int, ...] = (0,)
) -> bytes:
    """Run git in ``repo``; an exit code outside ``ok`` aborts the program."""
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        input=stdin,
        capture_output=True,
        check=False,
    )
    if proc.returncode not in ok:
        sys.stderr.write(proc.stderr.decode("utf-8", "replace"))
        raise SystemExit(f"git {' '.join(args)}: exit {proc.returncode}")
    return proc.stdout


def in_head(repo: Path, path: str) -> bool:
    """True when ``path`` has a blob in HEAD (staged-only files do not)."""
    proc = subprocess.run(
        ["git", "-C", str(repo), "cat-file", "-e", f"HEAD:{path}"],
        capture_output=True,
        check=False,
    )
    return proc.returncode == 0


def head_time(repo: Path) -> int:
    return int(git(repo, "log", "-1", "--format=%ct").decode().strip())


def tracked_receipts(repo: Path) -> list[str]:
    names = git(repo, "ls-files", "-z", "--", BENCH_DIR).decode().split("\0")
    return sorted(name for name in names if name)


def cited_paths(repo: Path) -> set[str]:
    """Run directories and receipt files named by files that count as citations."""
    excludes = [
        f":(exclude){name}" for name in (*NON_CITING_PREFIXES, *NON_CITING_FILES)
    ]
    out = git(
        repo,
        "grep",
        "--cached",
        "--full-name",
        "-I",
        "-o",
        "-z",
        "-E",
        MENTION_ER,
        "--",
        ".",
        *excludes,
        ok=(0, 1),
    )
    # Each match is its own record: ``path NUL match`` ended by a newline on
    # some git versions and by a NUL on others. Splitting on both reads either
    # form; a sentence-final period is not part of the path.
    tokens = re.split(r"[\0\n]", out.decode("utf-8", "replace"))
    return {t.rstrip(".") for t in tokens[1::2] if t.startswith(f"{BENCH_DIR}/")}


def run_of(path: str) -> str:
    """The run a receipt sits in; a top-level file is its own run."""
    parts = path.split("/")
    return parts[1] if len(parts) > 1 else path


def untracked_reports(repo: Path, run: str, tracked: list[str]) -> set[str]:
    """Markdown reports on disk inside ``run`` that git does not track."""
    root = repo / BENCH_DIR / run
    found: set[str] = set()
    if not root.is_dir():
        return found
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        # staging/ symlinks point at user-local models: never follow or descend.
        dirnames[:] = [
            name for name in dirnames if not (Path(dirpath) / name).is_symlink()
        ]
        for filename in filenames:
            path = Path(dirpath) / filename
            if filename.endswith(REPORT_SUFFIX) and not path.is_symlink():
                rel = path.relative_to(repo).as_posix()
                if rel not in tracked:
                    found.add(rel)
    return found


@dataclass
class Plan:
    tracked: list[str]
    kept: list[str]
    archive: list[str]
    add: list[str]
    # Cited runs with nothing tracked and nothing on disk to add.
    missing: list[str]
    cited: set[str] = field(default_factory=set)


def plan_for(repo: Path) -> Plan:
    cited = cited_paths(repo)
    # A directory-level mention ("bench-artifacts/<run>") and a file-level one
    # ("bench-artifacts/<run>/<file>") both cite the whole run.
    cited_runs = {run_of(name) for name in cited if name != INDEX}
    tracked = tracked_receipts(repo)
    tracked_set = set(tracked)
    kept_set = {
        path
        for path in tracked
        if path == INDEX or path in cited or run_of(path) in cited_runs
    }
    kept = [path for path in tracked if path in kept_set]
    archive = [path for path in tracked if path not in kept_set]
    add = {
        name
        for name in cited
        if name.count("/") >= 2
        and name not in tracked_set
        and name.endswith(CITABLE_SUFFIXES)
        and (repo / name).is_file()
    }
    for run in sorted(cited_runs):
        add |= untracked_reports(repo, run, tracked)
    present = {run_of(path) for path in tracked} | {run_of(path) for path in add}
    missing = [f"{BENCH_DIR}/{run}/" for run in sorted(cited_runs - present)]
    return Plan(tracked, kept, archive, sorted(add), missing, cited)


def show(plan: Plan) -> None:
    print(
        f"{BENCH_DIR}: {len(plan.tracked)} tracked, {len(plan.kept)} kept, "
        f"{len(plan.archive)} to archive, {len(plan.add)} to add"
    )
    for label, paths in (("archive", plan.archive), ("add", plan.add)):
        for path in paths:
            print(f"  {label}: {path}")
    for name in plan.missing:
        print(f"  cited, no receipt committed: {name}")


def sidecar_for(out: Path) -> Path:
    return out.with_name(re.sub(r"\.tar(\.gz)?$", "", out.name) + ".sha256")


def blob_at_head(repo: Path, path: str) -> bytes:
    blob = git(repo, "cat-file", "blob", f"HEAD:{path}")
    hit = CREDENTIAL.search(blob.decode("utf-8", "replace"))
    if hit:
        shape = hit.group(0)[:7]
        raise SystemExit(
            f"{path}: credential-shaped string starting {shape!r}; archive aborted"
        )
    return blob


def build_archive(blobs: dict[str, bytes], mtime: int, out: Path) -> str:
    """Write ``blobs`` as a reproducible tar.gz and return its sha256.

    Member order, mtimes, modes and ownership are fixed, so the same committed
    files always produce the same bytes; gzip stores no name or timestamp.
    """
    with (
        out.open("xb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz,
        tarfile.open(fileobj=gz, mode="w", format=tarfile.GNU_FORMAT) as tar,
    ):
        for path in sorted(blobs):
            blob = blobs[path]
            member = tarfile.TarInfo(path)
            member.size = len(blob)
            member.mtime = mtime
            member.mode = 0o644
            tar.addfile(member, io.BytesIO(blob))
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    with sidecar_for(out).open("x") as sidecar:
        sidecar.write(f"{digest}  {out.name}\n")
    return digest


def verify_archive(out: Path, blobs: dict[str, bytes]) -> None:
    with tarfile.open(out, "r:gz") as tar:
        members = tar.getmembers()
        if [member.name for member in members] != sorted(blobs):
            raise SystemExit(f"{out}: member list differs from the plan")
        for member in members:
            stream = tar.extractfile(member)
            if stream is None or stream.read() != blobs[member.name]:
                raise SystemExit(
                    f"{out}: {member.name} differs from the committed blob"
                )


def index_update(repo: Path, *args: str, paths: list[str]) -> None:
    """Run a git index command over ``paths`` (NUL-separated on stdin)."""
    listing = b"\0".join(path.encode() for path in paths) + b"\0"
    git(repo, *args, "--pathspec-from-file=-", "--pathspec-file-nul", stdin=listing)


def apply(repo: Path, plan: Plan, out: Path | None) -> int:
    if not plan.archive and not plan.add:
        print(
            "already in sync: every tracked receipt is cited and nothing cited is missing"
        )
        return 0
    digest = ""
    if plan.archive:
        # The archive is built from HEAD, so a file that was only staged would
        # be missing from it while the index change below still lands.
        uncommitted = [path for path in plan.archive if not in_head(repo, path)]
        if uncommitted:
            raise SystemExit(
                "staged but never committed (the archive is built from HEAD, commit first): "
                + ", ".join(uncommitted)
            )
        archive_path = out or default_archive(repo)
        if archive_path.exists():
            raise SystemExit(
                f"{archive_path}: already exists, pass --out to pick another name"
            )
        blobs = {path: blob_at_head(repo, path) for path in plan.archive}
        digest = build_archive(blobs, head_time(repo), archive_path)
        verify_archive(archive_path, blobs)
        print(f"archive: {archive_path} ({len(plan.archive)} files, sha256 {digest})")
    if plan.archive:
        index_update(repo, "rm", "--cached", "-q", paths=plan.archive)
        print(
            f"untracked from the index: {len(plan.archive)} file(s); they stay on disk"
        )
    if plan.add:
        index_update(repo, "add", "-f", paths=plan.add)
        print(f"tracked in the index: {len(plan.add)} file(s)")
    if plan.archive:
        print(
            "next: commit the index, then "
            f"`gh release upload {ARCHIVE_RELEASE} <tarball> <sha256>` "
            "and update the release notes"
        )
    else:
        print("next: commit the index")
    return 0


def default_archive(repo: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    return repo / BENCH_DIR / f"bench-artifacts-{stamp}.tar.gz"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Keep bench-artifacts/ in step with what the tree cites."
    )
    parser.add_argument(
        "--repo",
        default=str(REPO),
        help="repository to inspect (default: the one holding this script)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check", action="store_true", help="exit 1 when the plan is not empty"
    )
    mode.add_argument(
        "--apply", action="store_true", help="write the archive and update the index"
    )
    mode.add_argument("--selftest", action="store_true", help="run the fixture test")
    parser.add_argument(
        "--out",
        type=Path,
        help="archive path for --apply (default: bench-artifacts/bench-artifacts-<date>.tar.gz)",
    )
    args = parser.parse_args(argv)
    if args.selftest:
        return selftest()
    repo = Path(args.repo).resolve()
    plan = plan_for(repo)
    show(plan)
    if args.apply:
        return apply(repo, plan, args.out)
    if args.check and (plan.archive or plan.add):
        print(
            "run scripts/bench_archive.py --apply, commit the index, "
            "then upload the tarball",
            file=sys.stderr,
        )
        return 1
    return 0


def selftest() -> int:
    """Exercise the plan, the guards and the archive format on a throwaway repo."""
    failures = 0

    def check(name: str, cond: bool) -> None:
        nonlocal failures
        if cond:
            print(f"ok - {name}")
        else:
            failures += 1
            print(f"FAIL - {name}")

    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "repo"

        def put(rel: str, text: str = "fixture\n") -> None:
            path = repo / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)

        def commit(paths: list[str]) -> None:
            git(repo, "add", "-f", "--", *paths)
            git(
                repo,
                "-c",
                "user.name=fixture",
                "-c",
                "user.email=fixture@invalid",
                "commit",
                "-q",
                "-m",
                "fixture",
            )

        def run(stem: str) -> str:
            return f"{BENCH_DIR}/{stem}"

        put(".gitignore", f"{BENCH_DIR}/**\n!{BENCH_DIR}/**/\n!{INDEX}\n")
        put(
            "README.md",
            f"Proofs: {run('2026-01-cited/REPORT.md')}. "
            f"Newer: {run('2026-05-new/REPORT.md')}\n",
        )
        put(
            "src/registry.json",
            '{"lane": "'
            + run("2026-02-dir")
            + '", "raw": "'
            + run("2026-02-dir/payload.json")
            + '", "gone": "'
            + run("2026-04-ghost")
            + '"}\n',
        )
        put("docs/research/note.md", f"{run('2026-03-old/benchmark.md')}\n")
        put("CHANGELOG.md", f"{run('2026-03-old/benchmark.md')}\n")
        tracked = [
            INDEX,
            run("2026-01-cited/REPORT.md"),
            run("2026-01-cited/extra.md"),
            run("2026-02-dir/report.md"),
            run("2026-02-dir/plot.svg"),
            run("2026-03-old/benchmark.md"),
            run("2026-03-old/charts/a.svg"),
            run("orphan-2026-09.json"),
        ]
        for rel in tracked:
            put(rel)
        git(repo, "init", "-q")
        commit(
            [
                ".gitignore",
                "README.md",
                "src/registry.json",
                "docs/research/note.md",
                "CHANGELOG.md",
                *tracked,
            ]
        )
        for rel in (
            run("2026-01-cited/LATE.md"),
            run("2026-03-old/NEW.md"),
            run("2026-02-dir/payload.json"),
            run("2026-05-new/REPORT.md"),
        ):
            put(rel)

        plan = plan_for(repo)
        check("tracked count", plan.tracked == sorted(tracked))
        check(
            "keep: index, cited file's run (whole run), dir-cited run",
            plan.kept
            == sorted(
                [
                    INDEX,
                    run("2026-01-cited/REPORT.md"),
                    run("2026-01-cited/extra.md"),
                    run("2026-02-dir/report.md"),
                    run("2026-02-dir/plot.svg"),
                ]
            ),
        )
        check(
            "archive: uncited run and orphan json; research/changelog mentions do not count",
            plan.archive
            == sorted(
                [
                    run("orphan-2026-09.json"),
                    run("2026-03-old/benchmark.md"),
                    run("2026-03-old/charts/a.svg"),
                ]
            ),
        )
        check(
            "add: cited new report and untracked report inside a cited run; payload and uncited-run md excluded",
            plan.add
            == sorted([run("2026-01-cited/LATE.md"), run("2026-05-new/REPORT.md")]),
        )
        check("ghost run reported", plan.missing == [f"{run('2026-04-ghost')}/"])
        check(
            "cli --check fails while the plan is non-empty",
            main(["--repo", str(repo), "--check"]) == 1,
        )

        out = Path(tmp) / "art.tar.gz"
        check("apply returns 0", apply(repo, plan, out) == 0)
        with tarfile.open(out, "r:gz") as tar:
            members = tar.getmembers()
            names = [member.name for member in members]
            mtimes = {member.mtime for member in members}
            blobs: dict[str, bytes] = {}
            for member in members:
                stream = tar.extractfile(member)
                if stream is not None:
                    blobs[member.name] = stream.read()
        check("tarball members are the archive set, sorted", names == plan.archive)
        check(
            "tarball bytes equal the committed blobs",
            all(
                blob == git(repo, "cat-file", "blob", f"HEAD:{name}")
                for name, blob in blobs.items()
            ),
        )
        check("one commit timestamp for every member", len(mtimes) == 1)
        digest = hashlib.sha256(out.read_bytes()).hexdigest()
        check(
            "sidecar names the tarball and its sha256",
            Path(f"{tmp}/art.sha256").read_text() == f"{digest}  art.tar.gz\n",
        )
        check(
            "index now holds exactly kept + add",
            tracked_receipts(repo) == sorted(plan.kept + plan.add),
        )
        check(
            "archived files stay on disk",
            all((repo / path).is_file() for path in plan.archive),
        )
        check(
            "cli --check passes once in sync",
            main(["--repo", str(repo), "--check"]) == 0,
        )
        again = Path(tmp) / "again.tar.gz"
        check(
            "second apply is a no-op",
            apply(repo, plan_for(repo), again) == 0 and not again.exists(),
        )

        first = Path(tmp) / "a1.tar.gz"
        second = Path(tmp) / "a2.tar.gz"
        mtime = head_time(repo)
        head_blobs = {
            path: git(repo, "cat-file", "blob", f"HEAD:{path}") for path in plan.archive
        }
        check(
            "rebuilding from the same commit gives identical bytes",
            build_archive(head_blobs, mtime, first)
            == build_archive(head_blobs, mtime, second),
        )

        staged = run("2026-06-staged/x.md")
        put(staged)
        git(repo, "add", "-f", "--", staged)
        try:
            apply(repo, plan_for(repo), Path(tmp) / "staged.tar.gz")
            check("staged-only file refuses the archive", False)
        except SystemExit as exc:
            check(
                "staged-only file refuses the archive",
                "never committed" in str(exc.code),
            )
        git(repo, "rm", "--cached", "-q", "--", staged)
        (repo / staged).unlink()

        secret = run("2026-07-secret/report.md")
        put(secret, "token " + "gh" + "p_" + "A" * 36 + "\n")
        commit([secret])
        try:
            apply(repo, plan_for(repo), Path(tmp) / "secret.tar.gz")
            check("credential-shaped blob refuses the archive", False)
        except SystemExit as exc:
            check(
                "credential-shaped blob refuses the archive",
                "credential-shaped" in str(exc.code),
            )

    print(
        "bench_archive selftest: PASS"
        if not failures
        else f"bench_archive selftest: {failures} FAIL"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
