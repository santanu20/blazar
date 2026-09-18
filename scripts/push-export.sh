#!/bin/sh
# Publishes the dev branch with the full logical history behind the
# publish gates (the squash-mirror flow retired once history was
# purged of secrets). Hygiene, fmt, clippy, tests, and a
# credential-shape scan on the exact tree being pushed run as hard
# blockers — any failure stops before anything reaches the remote.
#
# Flow: clean-tree check -> hygiene -> fmt/clippy/tests ->
# credential scan on the HEAD tree -> push (fast-forward only; a
# diverged remote aborts loudly) -> remote-sha verification.
#
# Environment:
#   PALLAMA_PUSH_REMOTE  remote name (default: origin)
#   PALLAMA_PUSH_FROM    local branch to publish (default: master)
#   PALLAMA_PUSH_TO      remote branch (default: main)
#   PALLAMA_PUSH_FAST    set to 1 to skip the test suite (CI already
#                          ran it; never for release pushes)
#
# Tags ride separately: git push <remote> --tags

set -eu

remote=${PALLAMA_PUSH_REMOTE:-origin}
src_branch=${PALLAMA_PUSH_FROM:-master}
dst_branch=${PALLAMA_PUSH_TO:-main}

say() { printf 'push-export: %s\n' "$1"; }

# The gates validate HEAD; uncommitted tracked changes would publish
# something other than what the gates validated. Untracked files
# never enter the push and are fine to exist.
dirty=$(git status --porcelain | { grep -v '^??' || true; })
if [ -n "$dirty" ]; then
    printf '%s\n' "$dirty" >&2
    say "tracked changes not committed - commit or stash first" >&2
    exit 1
fi

# Gates, in fail-fast order (cheapest first).
sh scripts/check_hygiene.sh
cargo fmt --all --check
cargo clippy --workspace --all-targets --quiet -- -D warnings
if [ "${PALLAMA_PUSH_FAST:-0}" != "1" ]; then
    cargo test --workspace -- --test-threads=1
fi

# Credential-shaped scan on the exact tree being pushed (HEAD), not
# the working directory: untracked local files are out of scope.
# git grep exits 0 on a match, 1 when clean, >1 when the scan itself
# failed — everything but 1 aborts.
token_re='ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{20,}'
scan_rc=0
git grep -qE "$token_re" HEAD -- || scan_rc=$?
if [ "$scan_rc" -eq 0 ]; then
    say "credential-shaped token found in the tree to publish - aborting" >&2
    exit 1
elif [ "$scan_rc" -gt 1 ]; then
    say "credential scan could not run (rc=$scan_rc) - aborting" >&2
    exit 1
fi

# Push the real history. Plain push = fast-forward only: a diverged
# remote is an anomaly to reconcile by hand, never steamrolled.
git push "$remote" "$src_branch:$dst_branch"

# Remote proof: the pushed sha must be what the branch now holds.
remote_sha=$(git ls-remote "$remote" "refs/heads/${dst_branch}" | cut -f1)
[ "$remote_sha" = "$(git rev-parse "$src_branch")" ] || {
    say "remote ${dst_branch} does not point at the published commit" >&2
    exit 1
}
say "published $(git rev-parse --short "$src_branch") -> ${dst_branch} (verified via ls-remote)"
