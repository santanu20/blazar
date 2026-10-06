#!/usr/bin/env python3
"""Config-doc drift guard.

Cross-checks docs/config-code-paths.md against the REAL struct surface
of crates/blazar-core/src/config.rs (brace-depth parse, same method the
nuclear audit used). Exits non-zero when a serializable field exists in
code but is missing from the doc's table (or the doc lists a field the
code dropped), so the "exhaustive" claim in the doc header stays true.

Run after any config.rs field change; regenerate rows by hand from the
consumption-site evidence (two-hop: field -> accessor -> callsite).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CONFIG_RS = REPO / "crates" / "blazar-core" / "src" / "config.rs"
DOC = REPO / "docs" / "config-code-paths.md"


def parse_struct_fields(src: str) -> dict[str, list[str]]:
    """struct Name { ... } -> field list (pub name: ...), brace-aware."""
    structs: dict[str, list[str]] = {}
    for m in re.finditer(r"^pub struct (\w+)\s*(?:#[^{]*)?\{", src, re.MULTILINE):
        name, i = m.group(1), m.end() - 1
        depth, fields = 0, []
        while i < len(src):
            c = src[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            elif depth == 1 and c == "\n":
                pass
            i += 1
        body = src[m.end() : i]
        for line in body.splitlines():
            fm = re.match(r"\s*pub (\w+):", line)
            if fm:
                fields.append(fm.group(1))
        if fields:
            structs[name] = fields
    return structs


def doc_rows(md: str) -> dict[str, set[str]]:
    """Section headers like '## Config (top-level keys) — 184 fields' map
    to the first-column field names of the table that follows."""
    out: dict[str, set[str]] = {}
    for m in re.finditer(r"^##\s+([^(—\n]+)", md, re.MULTILINE):
        sec = m.group(1).strip()
        # fields live in the table rows until the next header
        nxt = md.find("\n## ", m.end())
        chunk = md[m.end() : nxt if nxt != -1 else len(md)]
        rows = re.findall(r"^\|\s*(`?\w+`?)\s*\|", chunk, re.MULTILINE)
        # skip the table header itself ("| field | default | ...") and separator
        fields = {r.strip("`") for r in rows} - {"field"}
        if fields:
            out[sec] = fields
    return out


def main() -> int:
    src = CONFIG_RS.read_text()
    md = DOC.read_text()
    structs = parse_struct_fields(src)
    rows = doc_rows(md)

    # The doc tracks one table per config.rs struct (section == struct).
    tracked = {
        "Config": "Config",
        "MistralrsTuning": "MistralrsTuning",
        "SglangTuning": "SglangTuning",
        "ModelOverride": "ModelOverride",
        "SamplerDefaults": "SamplerDefaults",
        "Remote": "Remote",
        "ApiKey": "ApiKey",
        "WarmPeg": "WarmPeg",
        "EngineRouting": "EngineRouting",
        "SemanticCacheConfig": "SemanticCacheConfig",
    }
    failures = 0
    for sec, struct in tracked.items():
        code_fields = structs.get(struct, [])
        doc_fields = rows.get(sec)
        if doc_fields is None:
            print(f"[drift] section for {struct} not found in doc")
            failures += 1
            continue
        missing = [f for f in code_fields if f not in doc_fields]
        extra = [f for f in sorted(doc_fields) if f not in code_fields]
        # claimed count in the section header, if present
        hm = re.search(rf"^##\s+{re.escape(sec)}[^\n]*?(\d+)\s+fields", md, re.MULTILINE)
        claimed = int(hm.group(1)) if hm else None
        problems = []
        if missing:
            problems.append(f"missing from doc: {missing}")
        if extra:
            problems.append(f"in doc but not code: {extra}")
        if claimed is not None and claimed != len(code_fields):
            problems.append(
                f"header claims {claimed} fields, code has {len(code_fields)}"
            )
        if problems:
            failures += 1
            print(f"[drift] {struct}: " + "; ".join(problems))
        else:
            print(f"[ok] {struct}: {len(code_fields)} fields match doc")

    if failures:
        print(
            f"\n{failures} struct table(s) drifted — update docs/config-code-paths.md"
        )
        return 1
    print("\ndoc is in sync with config.rs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
