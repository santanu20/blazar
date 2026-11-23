#!/usr/bin/env python3
"""Curated capability-registry sync tool.

Keeps `registry/capability-lanes.json` honest and discovers new lane
candidates without ever mutating the registry automatically (it is a
trust anchor: every row lands through maintainer review).

Two modes:

  --check   CI gate. Verifies every registry row: the pinned commit
            still exists upstream (the registry once pointed at a SHA
            a fork had force-pushed away), the fork's tree at that SHA
            still declares each advertised architecture in
            src/llama-arch.cpp, and no advertised architecture has
            graduated into llama.cpp mainline (a graduated lane should
            be retired). Exits non-zero on any drift.

  --propose Discovery. Scans open ggml-org/llama.cpp pull requests for
             architecture additions, probes each PR head for a real
             LLM_ARCH_* delta versus mainline, checks that GGUF builds
             of the architecture circulate on Hugging Face, and emits
             proposed rows as JSON + Markdown for maintainer review.

  --verify  Signature gate. Checks the registry's ed25519 signature
            (registry/capability-lanes.json.sig) against the committed
            public key (registry/capability-lanes.pub) - the same pair
            `blazar registry sign` produces and the embedded runtime
            key verifies against. Exits non-zero when the signature is
            missing or does not verify, so a CI gate fails closed the
            moment a hand-edited (or unsigned) registry slips in.

Both modes are read-only with respect to the repository.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

UPSTREAM_REPO = "ggml-org/llama.cpp"
UPSTREAM_ARCH_FILE = "src/llama-arch.cpp"
HF_API = "https://huggingface.co/api/models"
GITHUB_API = "https://api.github.com"
RAW = "https://raw.githubusercontent.com"

# Titles that historically introduced a new LLM_ARCH_* enum; loose on
# purpose - every hit is verified by an actual enum delta before it
# can become a proposal.
PR_TITLE_RE = re.compile(
    r"(add|support|implement).{0,40}(arch|model|zamba|raon|mpnet|domyn|wedlm|instella|inkling|hy[_ -])",
    re.IGNORECASE,
)


def http_get(url: str, headers: dict[str, str] | None = None) -> bytes:
    """GET with token auth, retries and rate-limit backoff."""
    hdrs = {"User-Agent": "blazar-registry-sync"}
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        hdrs["Authorization"] = f"Bearer {token}"
    if headers:
        hdrs.update(headers)
    last_err: Exception | None = None
    for attempt in range(4):
        req = urllib.request.Request(url, headers=hdrs)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read()
        except urllib.error.HTTPError as err:
            if err.code in (403, 429) and attempt < 3:
                time.sleep(2**attempt * 5)
                continue
            raise
        except urllib.error.URLError as err:
            last_err = err
            if attempt < 3:
                time.sleep(2**attempt * 2)
                continue
            raise
    raise RuntimeError(f"unreachable: {url} ({last_err})")


def github_json(path: str) -> object:
    return json.loads(http_get(f"{GITHUB_API}{path}"))


def raw_file(repo: str, ref: str, path: str) -> str:
    return http_get(f"{RAW}/{repo}/{ref}/{path}").decode("utf-8", "replace")


def arch_enums(text: str) -> set[str]:
    """LLM_ARCH_* enum names present in a llama-arch.cpp."""
    names = set(re.findall(r"LLM_ARCH_([A-Z0-9_]+)", text))
    names.discard("UNKNOWN")
    return names


def upstream_arch_set() -> set[str]:
    return arch_enums(raw_file(UPSTREAM_REPO, "master", UPSTREAM_ARCH_FILE))


def enum_to_gguf_name(enum: str, fork_text: str) -> str | None:
    """The exact GGUF `general.architecture` string for an enum.

    Parsed from the arch-name table in the fork's llama-arch.cpp so
    registry rows never guess the wire format.
    """
    pat = re.compile(r"LLM_ARCH_" + enum + r'\s*,\s*"([^"]+)"')
    m = pat.search(fork_text)
    return m.group(1) if m else None


def sha_alive(repo: str, sha: str) -> bool:
    try:
        github_json(f"/repos/{repo}/commits/{sha}")
        return True
    except urllib.error.HTTPError as err:
        if err.code == 404:
            return False
        raise


def load_registry(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        rows = json.load(fh)
    if not isinstance(rows, list):
        raise SystemExit(f"{path}: expected a JSON array of lane rows")
    return rows


def cmd_check(registry_path: str) -> int:
    rows = load_registry(registry_path)
    upstream_text = raw_file(UPSTREAM_REPO, "master", UPSTREAM_ARCH_FILE)
    drift: list[str] = []

    for row in rows:
        lane, repo, sha = row["id"], row["repo"], row["ref_sha"]
        if not sha_alive(repo, sha):
            drift.append(
                f"{lane}: pinned commit {sha[:12]} no longer exists in {repo} (force-push?) - repoint or retire"
            )
            continue
        try:
            fork_text = raw_file(repo, sha, UPSTREAM_ARCH_FILE)
        except Exception as err:  # probe failure, not a registry verdict
            drift.append(
                f"{lane}: probe error fetching {UPSTREAM_ARCH_FILE} at {sha[:12]} ({type(err).__name__}) - network or rate limit, not a registry verdict"
            )
            continue
        for gguf_name in row.get("architectures", []):
            # The enum<->wire-name mapping is only reliable in the fork's
            # own arch-name table (underscores live in the enum, not the
            # wire string: raon1nano == RAON1_NANO), so verdicts key on
            # the exact quoted wire string, exactly what a GGUF
            # general.architecture carries.
            wire = f'"{gguf_name}"'
            if wire in upstream_text:
                drift.append(
                    f"{lane}: architecture '{gguf_name}' graduated into llama.cpp mainline - retire the lane"
                )
            elif wire not in fork_text:
                drift.append(
                    f"{lane}: fork at {sha[:12]} does not declare '{gguf_name}' - broken pin"
                )
        print(
            f"ok  {lane:14s} {repo}@{sha[:12]} archs={','.join(row.get('architectures', []))}"
        )

    if drift:
        print("\ndrift detected:")
        for line in drift:
            print(f"  - {line}")
        return 1
    print(f"\n{len(rows)} lanes verified, no drift")
    return 0


def hf_gguf_count(arch: str) -> tuple[int, str]:
    """(model-repo count, sample repo id) for GGUFs mentioning the arch."""
    try:
        q = urllib.parse.quote(f"sort=downloads&filter=gguf&search={arch}&limit=5")
        hits = json.loads(http_get(f"{HF_API}?{q}").decode())
        sample = hits[0]["id"] if hits else ""
        return len(hits), sample
    except Exception:  # discovery is advisory only
        return -1, ""


def cmd_propose() -> int:
    upstream = upstream_arch_set()
    query = urllib.parse.quote(
        f"repo:{UPSTREAM_REPO} is:pr is:open architecture in:title"
    )
    prs = github_json(f"/search/issues?q={query}&per_page=50").get("items", [])
    print(f"# registry-sync proposals ({len(prs)} candidate PRs scanned)\n")

    proposals: list[dict] = []
    for pr in prs:
        if not PR_TITLE_RE.search(pr["title"]):
            continue
        number = pr["number"]
        try:
            detail = github_json(f"/repos/{UPSTREAM_REPO}/pulls/{number}")
            head_repo = (detail.get("head") or {}).get("repo") or {}
            repo_full, head_sha = (
                head_repo.get("full_name"),
                (detail.get("head") or {}).get("sha"),
            )
            if not repo_full or not head_sha:
                continue
            fork_text = raw_file(repo_full, head_sha, UPSTREAM_ARCH_FILE)
            new_enums = sorted(a for a in arch_enums(fork_text) - upstream)
            if not new_enums:
                continue
            names = [n for e in new_enums if (n := enum_to_gguf_name(e, fork_text))]
            if not names:
                continue
            count, sample = max((hf_gguf_count(n) for n in names), default=(-1, ""))
            proposals.append(
                {
                    "id": names[0].replace("_", "-"),
                    "repo": repo_full,
                    "ref_sha": head_sha,
                    "upstream_pr": number,
                    "architectures": names,
                    "hf_gguf_repos": count,
                    "hf_sample": sample,
                    "title": pr["title"],
                }
            )
        except Exception as err:  # advisory mode must keep going
            print(f"  skip PR #{number}: {type(err).__name__}", file=sys.stderr)

    if not proposals:
        print("no new-architecture PRs with GGUF circulation found")
        return 0
    print("## proposed rows (maintainer review required - NOT applied)\n")
    print("| lane | PR | fork | architectures | HF GGUFs |")
    print("|------|----|------|---------------|----------|")
    for p in proposals:
        print(
            f"| {p['id']} | #{p['upstream_pr']} | {p['repo']}@{p['ref_sha'][:10]} | {','.join(p['architectures'])} | {p['hf_gguf_repos']} ({p['hf_sample'][:40]}) |"
        )
    print("\n```json")
    print(json.dumps(proposals, indent=2))
    print("```")
    return 0


def cmd_verify(registry_path: str) -> int:
    """Fail-closed ed25519 signature check of the registry file."""
    try:
        # PyNaCl is the only non-stdlib dependency this script has, and
        # only this mode needs it - import lazily so --check/--propose
        # keep working in minimal CI images.
        from nacl.exceptions import BadSignatureError
        from nacl.signing import VerifyKey
    except ImportError:
        print(
            "verify: PyNaCl is required for signature checks - pip install pynacl",
            file=sys.stderr,
        )
        return 1

    sig_path = registry_path + ".sig"
    pub_path = os.path.join(os.path.dirname(registry_path), "capability-lanes.pub")
    for path in (registry_path, sig_path, pub_path):
        if not os.path.isfile(path):
            print(
                f"verify: missing {path} - the registry must ship as a signed pair",
                file=sys.stderr,
            )
            return 1

    with open(registry_path, "rb") as fh:
        body = fh.read()
    with open(sig_path) as fh:
        sig_b64 = fh.read().strip()
    with open(pub_path) as fh:
        pub_b64 = fh.read().strip()
    try:
        sig = base64.b64decode(sig_b64, validate=True)
        pub = base64.b64decode(pub_b64, validate=True)
    except (ValueError, binascii.Error) as exc:
        print(
            f"verify: signature/public-key files must be pure base64: {exc}",
            file=sys.stderr,
        )
        return 1

    try:
        VerifyKey(pub).verify(body, sig)
    except BadSignatureError:
        print(
            "verify: registry signature does NOT verify against"
            " capability-lanes.pub - re-sign with"
            " `blazar registry sign` after review",
            file=sys.stderr,
        )
        return 1
    print(f"verify: {registry_path} signature OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--registry",
        default="registry/capability-lanes.json",
        help="registry JSON path",
    )
    parser.add_argument(
        "--check", action="store_true", help="verify every row; exit 1 on drift"
    )
    parser.add_argument(
        "--propose",
        action="store_true",
        help="scan open upstream PRs for lane candidates",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="check the registry's ed25519 signature (fail-closed)",
    )
    args = parser.parse_args()
    if not (args.check or args.propose or args.verify):
        parser.error("choose --check, --propose and/or --verify")
    rc = 0
    if args.verify:
        rc |= cmd_verify(args.registry)
    if args.check:
        rc |= cmd_check(args.registry)
    if args.propose:
        rc |= cmd_propose()
    return rc


if __name__ == "__main__":
    sys.exit(main())
