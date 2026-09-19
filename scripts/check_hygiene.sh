#!/bin/sh
# Hygiene gates: machine-specific absolute paths + AI-agent artifacts.
# Blocks the classes a code review easily misses — real home dirs and
# agent scratch paths baked into comments/fixtures, external project
# names leaking into shipped source — at CI time. Exits non-zero
# listing every offending file:line.
#
# Usage: check_hygiene.sh [paths...]
#   With no args, scans every git-tracked file. Explicit paths are
#   scanned as-is (used by the self-test below).
#
# CHANGELOG.md is excluded: historical entries cite past paths and
# project names as record — rewriting history is not the gate's job.

set -u

# Each allowlisted hit must carry a functional justification:
# - /home/other (manifest.rs): invented-user fixture proving that a
#   manifest recorded on a foreign machine re-roots onto this install.
# - /home/other (pallama-cli main.rs): invented-user fixture proving the
#   list rendering never truncates long foreign model paths.
ALLOWLIST='crates/pallama-runtime/src/engine/manifest\.rs:[0-9]*:.*"/home/other/\.local/share/pallama.*|crates/pallama-cli/src/main\.rs:[0-9]*:.*"/home/other/\.local/share/pallama.*'

# Machine-specific absolute paths: real home dirs and agent scratch
# areas. Functionally-required system prefixes (/usr/local/bin,
# /usr/local/cuda, dscl) do not match these patterns.
PATH_PATTERNS='/home/[a-z]|/Users/[a-z]|/root/|/tmp/opencode'

# AI-agent session artifacts: unexplained external project names that
# leaked into shipped comments. Extend the denylist as new ones appear.
ARTIFACT_PATTERNS='geokit'

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

if [ "$#" -gt 0 ]; then
    hits=$(printf '%s\n' "$@" | scan_files)
else
    hits=$(git ls-files | scan_files)
fi

if [ -n "$hits" ]; then
    printf '%s\n' "$hits" >&2
    echo "hygiene gate: machine-specific paths / AI artifacts found (see above)" >&2
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

exit 0
