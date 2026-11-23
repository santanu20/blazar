#!/bin/sh
# Hygiene gates: machine-specific absolute paths + AI-agent artifacts +
# root-level working notes + packaging license integrity + CJK-rendering
# glyph control.
# Blocks the classes a code review easily misses - real home dirs and
# agent scratch paths baked into comments/fixtures, external project
# names leaking into shipped source, campaign or findings notes
# committed at the top level instead of beside their receipts, non-ASCII
# glyphs that render via CJK fallback fonts - at CI time. Exits non-zero
# listing every offending file:line.
#
# Usage: check_hygiene.sh [paths...]
#   With no args, scans every git-tracked file, rejects root-level
#   working notes and runs the packaging license gate. Explicit paths
#   are scanned as-is (used by the self-test below); the root-note and
#   license gates are repo-anchored and skipped in that mode.
#
# CHANGELOG.md is excluded: historical entries cite past paths and
# project names as record - rewriting history is not the gate's job.

set -u

# Each allowlisted hit must carry a functional justification:
# - /home/other (manifest.rs): invented-user fixture proving that a
#   manifest recorded on a foreign machine re-roots onto this install.
# - /home/other (blazar-cli main.rs, worker.env): invented-user fixture
#   proving the 0600 systemd EnvironmentFile line rides the unit verbatim.
# - /home/other (blazar-cli main.rs): invented-user fixture proving the
#   list rendering never truncates long foreign model paths.
ALLOWLIST='crates/blazar-runtime/src/engine/manifest\.rs:[0-9]*:.*"/home/other/\.local/share/blazar.*|crates/blazar-cli/src/main\.rs:[0-9]*:.*"/home/other/\.local/share/blazar.*|crates/blazar-cli/src/main\.rs:[0-9]*:.*EnvironmentFile=/home/other/\.config/blazar/worker\.env.*'

# Machine-specific absolute paths: real home dirs and agent scratch
# areas. Functionally-required system prefixes (/usr/local/bin,
# /usr/local/cuda, dscl) do not match these patterns.
PATH_PATTERNS='/home/[a-z]|/Users/[a-z]|/root/|/tmp/opencode'

# AI-agent session artifacts: unexplained external project names that
# leaked into shipped comments. Extend the denylist as new ones appear.
ARTIFACT_PATTERNS='geokit'

# Root-level markdown is product surface: the landing page, the release
# record, the published benchmark report. Any other top-level .md is a
# working note that belongs beside its receipts (bench-artifacts/<run>/)
# or under docs/research/. Name a new product document here to allow it.
ROOT_DOC_ALLOWLIST='BENCHMARK.md
CHANGELOG.md
README.md'

# license_field_present FILE PATTERN - true when FILE matches the
# license declaration pattern. Split out so the self-test can prove
# both directions (hit on a declaring file, silence on one without).
license_field_present() {
    grep -q -- "$2" "$1" 2>/dev/null
}

# Packaging license integrity: every distribution manifest declares
# the license and the repo root carries both license texts. This is
# the shipped-binaries-without-notices complaint class (ollama
# #3185); the gate turns it into a CI failure instead of a
# release-day scramble. Repo-anchored, so it runs in no-args mode only.
check_packaging_licenses() {
    fail=0
    [ -f LICENSE-MIT ] ||
        { echo "license gate: LICENSE-MIT missing at repo root" >&2; fail=1; }
    [ -f LICENSE-APACHE ] ||
        { echo "license gate: LICENSE-APACHE missing at repo root" >&2; fail=1; }
    license_field_present packaging/scoop/blazar.json '"license"' ||
        { echo 'license gate: packaging/scoop/blazar.json has no "license" field' >&2; fail=1; }
    license_field_present packaging/winget/blazar.yaml '^License:' ||
        { echo "license gate: packaging/winget/blazar.yaml has no License: field" >&2; fail=1; }
    license_field_present packaging/homebrew/blazar.rb 'license "' ||
        { echo 'license gate: packaging/homebrew/blazar.rb has no license stanza' >&2; fail=1; }
    return "$fail"
}

# scan_files - stdin is one path per line; prints every violating
# file:line:content.
scan_files() {
    while IFS= read -r f; do
        [ -f "$f" ] || continue
        [ "$f" = "CHANGELOG.md" ] && continue
        # The scanner's own pattern vocabulary is not a violation,
        # however the script and target were spelled (rel/abs).
        case "$f" in
            "$0"|*/"$0") continue ;;
        esac
        case "$0" in
            "$f"|*/"$f") continue ;;
        esac
        grep -HnE "$PATH_PATTERNS|$ARTIFACT_PATTERNS" -- "$f" 2>/dev/null |
            grep -vE "^$ALLOWLIST$" || true
    done
}

# root_note_violations - stdin is one tracked path per line; prints the
# root-level markdown files that are not named product documents.
# Repo-anchored, so it runs in no-args mode only.
root_note_violations() {
    while IFS= read -r f; do
        case "$f" in
            *.md) ;;
            *) continue ;;
        esac
        case "$f" in
            */*) continue ;;
        esac
        printf '%s\n' "$ROOT_DOC_ALLOWLIST" | grep -qxF -- "$f" && continue
        printf '%s\n' "$f"
    done
}

# Glyph gate: CJK script outside allowlisted data sites, plus
# render-fallback punctuation on NEW lines. Repo-anchored, so it runs
# in no-args mode only. Reports codepoints, never glyph bytes.
check_glyphs() {
    command -v python3 >/dev/null 2>&1 ||
        { echo "glyph gate: python3 required" >&2; return 1; }
    python3 "$(dirname "$0")/check_glyphs.py"
}

license_fail=0
glyph_fail=0
note_fail=0
if [ "$#" -gt 0 ]; then
    hits=$(printf '%s\n' "$@" | scan_files)
else
    tracked=$(git ls-files)
    hits=$(printf '%s\n' "$tracked" | scan_files)
    check_packaging_licenses || license_fail=1
    check_glyphs || glyph_fail=1
    root_notes=$(printf '%s\n' "$tracked" | root_note_violations)
    if [ -n "$root_notes" ]; then
        note_fail=1
        printf '%s\n' "$root_notes" |
            while IFS= read -r f; do
                echo "root-doc gate: $f is a working note at the repo root; keep it beside its receipts (bench-artifacts/<run>/) or under docs/research/ - product documents are named in ROOT_DOC_ALLOWLIST" >&2
            done
    fi
fi

if [ -n "$hits" ] || [ "$license_fail" -ne 0 ] || [ "$glyph_fail" -ne 0 ] ||
    [ "$note_fail" -ne 0 ]; then
    [ -n "$hits" ] && printf '%s\n' "$hits" >&2
    echo "hygiene gate: machine-specific paths / AI artifacts / root-level working notes / license gaps / glyphs found (see above)" >&2
    exit 1
fi
echo "hygiene gate: clean"

# Glyph-gate self-test: must fire on a CJK codepoint and stay silent on
# ASCII (codepoints only - the test never writes a literal glyph).
glyph_tmp=$(mktemp) || exit 1
printf 'plain ascii line\n' >"$glyph_tmp"
if python3 "$(dirname "$0")/check_glyphs.py" "$glyph_tmp" >/dev/null 2>&1; then
    :
else
    echo "hygiene gate: SELF-TEST FAILED (clean file flagged by glyph gate)" >&2
    rm -f "$glyph_tmp"
    exit 1
fi
printf 'bad \344\270\255 line\n' >"$glyph_tmp"
if python3 "$(dirname "$0")/check_glyphs.py" "$glyph_tmp" >/dev/null 2>&1; then
    echo "hygiene gate: SELF-TEST FAILED (CJK codepoint not caught)" >&2
    rm -f "$glyph_tmp"
    exit 1
fi
rm -f "$glyph_tmp"

# Self-test: the gate must fire on a violating file and stay silent on
# a clean one - a gate that cannot fail protects nothing.
tmp=$(mktemp) || exit 1
trap 'rm -f "$tmp"' EXIT

printf 'see /home/someone/x and geokit notes\n' >"$tmp"
if [ -z "$(printf '%s\n' "$tmp" | scan_files)" ]; then
    echo "hygiene gate: SELF-TEST FAILED (violation not caught)" >&2
    exit 1
fi

printf 'portable ~ and repo-relative paths only\n' >"$tmp"
if [ -n "$(printf '%s\n' "$tmp" | scan_files)" ]; then
    echo "hygiene gate: SELF-TEST FAILED (clean file flagged)" >&2
    exit 1
fi

# Root-note self-test: fires on a stray root-level note, stays silent on
# product documents, non-markdown files and notes filed under docs/.
root_notes=$(printf '%s\n' W4-FINDINGS.md README.md docs/research/note.md \
    Cargo.toml CHANGELOG.md BENCHMARK.md | root_note_violations)
if [ "$root_notes" != "W4-FINDINGS.md" ]; then
    echo "hygiene gate: SELF-TEST FAILED (root-note gate misclassified: $root_notes)" >&2
    exit 1
fi

# License-field helper self-test: fires on a manifest missing the
# declaration, stays silent on one carrying it.
printf 'License: MIT OR Apache-2.0\n' >"$tmp"
license_field_present "$tmp" '^License:' ||
    { echo "hygiene gate: SELF-TEST FAILED (license field not detected)" >&2; exit 1; }
printf 'PackageName: x\n' >"$tmp"
license_field_present "$tmp" '^License:' &&
    { echo "hygiene gate: SELF-TEST FAILED (missing license not caught)" >&2; exit 1; }

exit 0
