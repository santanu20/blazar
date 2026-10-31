#!/usr/bin/env bash
# Build a native Debian package for the Blazar CLI.
#
# Usage: scripts/package-deb.sh <release-binary> <version> <arch> <out.deb>
#
# Stages the binary plus shell completions into a proper deb layout and
# assembles it with dpkg-deb (available on every Debian/Ubuntu runner
# and dev box — no cargo-deb install needed). The package installs the
# CLI only: `blazar serve` manages its own daemon lifecycle, so no
# maintainer scripts or service units ride along.
set -euo pipefail

bin="${1:?usage: package-deb.sh <release-binary> <version> <arch> <out.deb>}"
version="${2:?missing version}"
arch="${3:?missing arch (e.g. amd64)}"
out="${4:?missing output .deb path}"

if [[ ! -x "$bin" ]]; then
  echo "package-deb: $bin is not an executable release binary" >&2
  exit 1
fi
command -v dpkg-deb >/dev/null || {
  echo "package-deb: dpkg-deb not found (install dpkg)" >&2
  exit 1
}

stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT

# File layout: binary, completions, and the debian control files.
install -D -m 0755 "$bin" "$stage/usr/bin/blazar"
mkdir -p \
  "$stage/usr/share/bash-completion/completions" \
  "$stage/usr/share/zsh/site-functions" \
  "$stage/usr/share/fish/vendor_completions.d"
# NOTE: keep this heredoc UNquoted (version/arch expand) and keep the
# description free of backticks — an unquoted heredoc would execute
# them as command substitution.
"$stage/usr/bin/blazar" completions bash > "$stage/usr/share/bash-completion/completions/blazar" 2>/dev/null || true
"$stage/usr/bin/blazar" completions zsh   > "$stage/usr/share/zsh/site-functions/_blazar" 2>/dev/null || true
"$stage/usr/bin/blazar" completions fish > "$stage/usr/share/fish/vendor_completions.d/blazar.fish" 2>/dev/null || true

mkdir -p "$stage/DEBIAN"
cat > "$stage/DEBIAN/control" <<EOF
Package: blazar
Version: ${version}
Section: utils
Priority: optional
Architecture: ${arch}
Depends: libc6
Maintainer: Santanu <santanu20@users.noreply.github.com>
Homepage: https://github.com/santanu20/blazar
Description: Local model server and inference gateway
 Blazar runs local LLMs behind one daemon speaking the OpenAI,
 Ollama, and Anthropic wire dialects, with model management,
 routing, speculative decoding, and usage accounting built in.
 Engines are downloaded separately via "blazar engine install".
EOF

dpkg-deb --root-owner-group --build "$stage" "$out"
echo "package-deb: built $out ($(du -h "$out" | cut -f1))"
