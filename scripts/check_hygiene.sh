#!/bin/sh
# Hygiene gates: machine-specific absolute paths + AI-agent artifacts +
# packaging license integrity.
# Blocks the classes a code review easily misses — real home dirs and
# agent scratch paths baked into comments/fixtures, external project
# names leaking into shipped source — at CI time. Exits non-zero
# listing every offending file:line.
#
# Usage: check_hygiene.sh [paths...]
#   With no args, scans every git-tracked file and runs the packaging
#   license gate. Explicit paths are scanned as-is (used by the
#   self-test below); the license gate is repo-anchored and skipped in
#   that mode.
#
# CHANGELOG.md is excluded: historical entries cite past paths and
# project names as record — rewriting history is not the gate's job.

set -u

# Each allowlisted hit must carry a functional justification:
# - /home/other (manifest.rs): invented-user fixture proving that a
#   manifest recorded on a foreign machine re-roots onto this install.
# - /home/other (blazar-cli main.rs): invented-user fixture proving the
#   list rendering never truncates long foreign model paths.
ALLOWLIST='crates/blazar-runtime/src/engine/manifest\.rs:[0-9]*:.*"/home/other/\.local/share/blazar.*|crates/blazar-cli/src/main\.rs:[0-9]*:.*"/home/other/\.local/share/blazar.*'

# Machine-specific absolute paths: real home dirs and agent scratch
# areas. Functionally-required system prefixes (/usr/local/bin,
# /usr/local/cuda, dscl) do not match these patterns.
PATH_PATTERNS='/home/[a-z]|/Users/[a-z]|/root/|/tmp/opencode'

# AI-agent session artifacts: unexplained external project names that
# leaked into shipped comments. Extend the denylist as new ones appear.
ARTIFACT_PATTERNS='geokit'

# license_field_present FILE PATTERN — true when FILE matches the
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

# scan_files — stdin is one path per line; prints every violating
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

license_fail=0
if [ "$#" -gt 0 ]; then
    hits=$(printf '%s\n' "$@" | scan_files)
else
    hits=$(git ls-files | scan_files)
    check_packaging_licenses || license_fail=1
fi

if [ -n "$hits" ] || [ "$license_fail" -ne 0 ]; then
    [ -n "$hits" ] && printf '%s\n' "$hits" >&2
    echo "hygiene gate: machine-specific paths / AI artifacts / license gaps found (see above)" >&2
    exit 1
fi
echo "hygiene gate: clean"

# Self-test: the gate must fire on a violating file and stay silent on
# a clean one — a gate that cannot fail protects nothing.
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

# License-field helper self-test: fires on a manifest missing the
# declaration, stays silent on one carrying it.
printf 'License: MIT OR Apache-2.0\n' >"$tmp"
license_field_present "$tmp" '^License:' ||
    { echo "hygiene gate: SELF-TEST FAILED (license field not detected)" >&2; exit 1; }
printf 'PackageName: x\n' >"$tmp"
license_field_present "$tmp" '^License:' &&
    { echo "hygiene gate: SELF-TEST FAILED (missing license not caught)" >&2; exit 1; }

exit 0
