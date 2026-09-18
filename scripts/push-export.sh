#!/bin/sh
# Publishes the current commit to the public repo as a fresh archive
# commit (squash-mirror flow): the public branch tracks releases, not
# development history. Gates run as hard blockers — any failure stops
# before anything is pushed.
#
# Flow: clean-tree check -> fmt/clippy/tests -> hygiene gate -> archive
# into the mirror clone -> secret-token + hygiene scan on the archived
# tree -> commit + push -> remote-sha verification.
#
# Environment:
#   PALLAMA_EXPORT_REPO   mirror URL (default: the pallama repo)
#   PALLAMA_EXPORT_DIR    mirror clone location (default: ~/.cache)
#   PALLAMA_EXPORT_FAST   set to 1 to skip the test suite (CI already
#                         ran it; never for release pushes)

set -eu

repo_url=${PALLAMA_EXPORT_REPO:-https://github.com/santanu20/pallama.git}
export_dir=${PALLAMA_EXPORT_DIR:-$HOME/.cache/pallama-export/repo}
branch=main
src=$(pwd)

say() { printf 'push-export: %s\n' "$1"; }

# The archive snapshots HEAD; uncommitted tracked changes would publish
# something other than what the gates validated. Untracked files never
# enter the archive and are fine to exist.
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
if [ "${PALLAMA_EXPORT_FAST:-0}" != "1" ]; then
    cargo test --workspace -- --test-threads=1
fi

# Mirror clone: self-healing — a wiped cache simply re-clones, and a
# directory left wrecked by an interrupted run is discarded first.
if [ -d "$export_dir" ] && [ ! -d "$export_dir/.git" ]; then
    rm -rf "$export_dir"
fi
if [ ! -d "$export_dir/.git" ]; then
    mkdir -p "$(dirname "$export_dir")"
    git clone --branch "$branch" "$repo_url" "$export_dir"
fi

# Refresh the mirror worktree from HEAD (history-free archive).
cd "$export_dir"
find . -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
git -C "$src" archive HEAD | tar -x -C "$export_dir"

# Scan what actually ships: hygiene gate + credential-token shapes.
sh scripts/check_hygiene.sh
if grep -rInE 'ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{20,}' \
    --exclude-dir=.git .; then
    say "credential-shaped token found in the archive - aborting" >&2
    exit 1
fi

git add -A
if git diff --cached --quiet; then
    say "nothing to publish (archive identical to ${branch})"
    exit 0
fi

msg=${1:-export: $(git -C "$src" log -1 --pretty=%s)}
git -c user.name=santanu20 \
    -c user.email=santanu20@users.noreply.github.com \
    commit --quiet -m "$msg"
git push origin "$branch"

# Remote proof: the pushed sha must be what the mirror HEAD is.
remote_sha=$(git ls-remote origin "refs/heads/${branch}" | cut -f1)
[ "$remote_sha" = "$(git rev-parse HEAD)" ] || {
    say "remote ${branch} does not point at the exported commit" >&2
    exit 1
}
say "published $(git rev-parse --short HEAD) -> ${branch} (verified via ls-remote)"
