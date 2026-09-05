#!/bin/sh
# pallama bootstrap installer — Linux and macOS.
#
#   curl --proto '=https' --tlsv1.2 -fsSL <raw-url-of-this-file> | sh
#
# Verifies the asset sha256 from the GitHub release API (the same source of
# truth as `pallama engine update`) before installing anything. No root:
# pallama is user-local (~/.local/share/pallama).
#
# Environment overrides:
#   PALLAMA_VERSION            pin a release tag (e.g. v0.1.0)
#   PALLAMA_REPO               GitHub owner/name hosting releases
#   PALLAMA_INSTALL_DIR        binary destination (default ~/.local/bin)
#   PALLAMA_INSTALL_BASE_URL   replace the GitHub API base (mirrors, tests)
#   GITHUB_TOKEN               optional API token (rate limits, private repos)
#
# Flag: --with-systemd-unit   also install a `systemctl --user` service.

# Wrap everything in main() so a truncated partial download cannot execute
# half a script (same guard technique as the ollama installer).
main() {

set -eu

REPO="${PALLAMA_REPO:-}"
API_BASE="${PALLAMA_INSTALL_BASE_URL:-https://api.github.com/repos/${REPO}}"

status() { echo ">>> $*" >&2; }
error() { echo "ERROR: $*" >&2; exit 1; }

# TLS pinning only when talking to GitHub (test/mirror bases may be http).
SECURE=
case "$API_BASE" in
    https://*) SECURE="--proto =https --tlsv1.2" ;;
esac

fetch() {
    if [ -n "${GITHUB_TOKEN:-}" ]; then
        curl --fail --silent --show-error --location $SECURE \
            -H "Authorization: Bearer ${GITHUB_TOKEN}" "$@"
    else
        curl --fail --silent --show-error --location $SECURE "$@"
    fi
}

# Pick the Linux libc variant: native musl systems use musl; glibc >= 2.35
# (the ubuntu-22.04 build floor) uses gnu; anything older or undetectable
# falls back to the static musl build.
pick_libc() {
    LDD_OUT=$(ldd --version 2>&1 || true)
    case "$LDD_OUT" in
        *musl*) echo musl; return ;;
    esac
    V=$(printf '%s\n' "$LDD_OUT" | sed -n '1s/.*[^0-9.]\([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p')
    MAJ=${V%%.*}
    MIN=${V#*.}
    MIN=${MIN%%.*}
    if [ -n "$V" ] && { [ "$MAJ" -gt 2 ] || { [ "$MAJ" -eq 2 ] && [ "$MIN" -ge 35 ]; }; }; then
        echo gnu
    else
        [ -n "$V" ] && status "glibc ${V} is older than the gnu build floor (2.35) — using the static musl build"
        echo musl
    fi
}

WITH_UNIT=0
SYSTEM=0
FROM_BIN=
FORCE_BUILD=0
while [ $# -gt 0 ]; do
    case "$1" in
        --with-systemd-unit) WITH_UNIT=1 ;;
        --system) SYSTEM=1 ;;
        --build) FORCE_BUILD=1 ;;
        --from) shift; [ $# -gt 0 ] || error "--from needs a binary path"; FROM_BIN=$1 ;;
        --from=*) FROM_BIN=${1#--from=} ;;
        *) error "unknown option: $1 (supported: --with-systemd-unit, --system, --from <binary>, --build)" ;;
    esac
    shift
done

# ollama-style sudo handling: empty when root, overridable for tests.
SUDO="${PALLAMA_SUDO-sudo}"
[ "$(id -u)" -eq 0 ] && SUDO=

# Zero-argument auto mode: no repo configured but a local build exists
# next to this script -> bootstrap it system-wide (binary + systemd unit),
# everything handled here. Explicit flags always win.
if [ -z "$FROM_BIN" ] && [ "$SYSTEM" = 0 ] &&
   [ -z "${PALLAMA_REPO:-}" ] && [ -z "${PALLAMA_INSTALL_BASE_URL:-}" ]; then
    SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
    if [ -x "$(dirname "$SCRIPT_DIR")/target/release/pallama" ]; then
        FROM_BIN="$(dirname "$SCRIPT_DIR")/target/release/pallama"
        SYSTEM=1
        status "auto: local build found, installing system-wide (binary + systemd service)"
    fi
fi

if [ -n "$FROM_BIN" ]; then
    # Bootstrap mode: install a locally built binary (no release needed).
    [ -f "$FROM_BIN" ] || error "--from: no such file: $FROM_BIN"
    [ -x "$FROM_BIN" ] || error "--from: not executable: $FROM_BIN"
    "$FROM_BIN" --version >/dev/null 2>&1 || error "--from: binary does not run: $FROM_BIN"
    status "Bootstrap install from $FROM_BIN (skipping release download)"
fi

# ---- source-build fallback --------------------------------------------
# Used when no prebuilt asset can be fetched (offline, rate-limited, exotic
# target): build the checkout with cargo and install that. Requires a
# toolchain — the installer never hides a build behind a download.

install_local() {
    # install_local <binary> <channel-label>
    INSTALL_DIR="${PALLAMA_INSTALL_DIR:-$HOME/.local/bin}"
    mkdir -p "$INSTALL_DIR" || error "cannot create ${INSTALL_DIR}"
    install -m 0755 "$1" "$INSTALL_DIR/pallama" || error "install to ${INSTALL_DIR} failed"
    VER=$("$INSTALL_DIR/pallama" --version 2>/dev/null || echo "(version check failed)")
    status "Installed pallama ${VER} to ${INSTALL_DIR}/pallama ($2)"
    status "Next: pallama engine update && pallama pull <model> && pallama run <model>"
}

find_checkout() {
    if [ -n "${PALLAMA_CHECKOUT:-}" ]; then
        [ -f "$PALLAMA_CHECKOUT/crates/pallama-cli/Cargo.toml" ] && { echo "$PALLAMA_CHECKOUT"; return 0; }
        return 1
    fi
    d=$(cd "$(dirname "$0")" && pwd -P)
    while [ "$d" != "/" ]; do
        [ -f "$d/crates/pallama-cli/Cargo.toml" ] && { echo "$d"; return 0; }
        d=$(dirname "$d")
    done
    return 1
}

build_from_checkout() {
    REASON="$1"
    status "fallback: building from source (${REASON})"
    command -v cargo >/dev/null 2>&1 ||
        error "${REASON}, and no cargo toolchain found. Install Rust (https://rustup.rs), or download a release asset manually."
    CK=$(find_checkout) ||
        error "${REASON}, and no pallama source checkout found near this script (set PALLAMA_CHECKOUT=<repo> to point at one)."
    status "cargo build --release -p pallama-cli (in ${CK})"
    (cd "$CK" && cargo build --release -p pallama-cli) ||
        error "source build failed (cargo output above)"
    [ -f "$CK/target/release/pallama" ] || error "build produced no target/release/pallama"
    install_local "$CK/target/release/pallama" "source build in ${CK}"
    status "All inference is upstream llama.cpp — ggml, ggerganov and contributors did the hard parts."
    exit 0
}

# --build: force the source path — audited/offline installs that never
# touch the release channel. Falls through to the same install tail.
if [ "$FORCE_BUILD" = 1 ]; then
    build_from_checkout "--build requested"
fi

# Repo guard is release-channel only: --from bootstrap and mirror/test
# base URLs never touch the GitHub release API.
if [ -z "${PALLAMA_INSTALL_BASE_URL:-}" ] && [ -z "${FROM_BIN:-}" ] && [ -z "$REPO" ]; then
    error "PALLAMA_REPO is not set. Export PALLAMA_REPO=owner/pallama (the GitHub repo hosting pallama releases) and re-run, or bootstrap a local build: sh scripts/install.sh --from target/release/pallama"
fi

# ---- bootstrap branch: --from needs no release, no OS/arch detection ----
if [ -n "$FROM_BIN" ]; then
    if [ "$SYSTEM" = 1 ]; then
        BIN_DIR="${PALLAMA_SYSTEM_BIN_DIR:-/usr/local/bin}"
        $SUDO mkdir -p "$BIN_DIR" || error "cannot create ${BIN_DIR} (need sudo?)"
        $SUDO install -m 0755 "$FROM_BIN" "$BIN_DIR/pallama" || error "install to ${BIN_DIR} failed"
        SVC_USER="${PALLAMA_SERVICE_USER:-$(id -un)}"
        UNIT_PATH="${PALLAMA_UNIT_PATH:-/etc/systemd/system/pallama.service}"
        SYSTEMCTL="${PALLAMA_SYSTEMCTL:-systemctl}"
        $SUDO mkdir -p "$(dirname "$UNIT_PATH")"
        UNIT=$(cat <<EOF
[Unit]
Description=Pallama daemon (llama.cpp orchestration)
After=network-online.target
Wants=network-online.target

[Service]
ExecStart=${BIN_DIR}/pallama serve
User=${SVC_USER}
Group=${PALLAMA_SERVICE_GROUP:-$(id -gn)}
SupplementaryGroups=render video
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF
)
        printf '%s\n' "$UNIT" | $SUDO tee "$UNIT_PATH" >/dev/null || error "writing $UNIT_PATH failed"
        $SUDO "$SYSTEMCTL" daemon-reload || error "systemctl daemon-reload failed"
        $SUDO "$SYSTEMCTL" enable --now pallama || error "enabling pallama.service failed"
        PORT=$(sed -n 's/^port[[:space:]]*=[[:space:]]*\([0-9]*\).*/\1/p' "$HOME/.config/pallama/config.toml" 2>/dev/null | head -1)
        PORT=${PORT:-11434}
        i=0
        while ! curl -s --max-time 2 "http://127.0.0.1:${PORT}/healthz" 2>/dev/null | grep -q ok; do
            i=$((i + 1)); [ "$i" -gt 30 ] && break
            sleep 1
        done
        if curl -s --max-time 2 "http://127.0.0.1:${PORT}/healthz" 2>/dev/null | grep -q ok; then
            status "systemd service active; pallama healthy on :${PORT} (logs: journalctl -u pallama)"
        else
            status "service enabled; healthz not answering on :${PORT} yet — check: journalctl -u pallama -n 30"
        fi
        VER=$("$BIN_DIR/pallama" --version 2>/dev/null || echo "(version check failed)")
        status "Installed pallama ${VER} system-wide (${BIN_DIR}/pallama + ${UNIT_PATH})"
        status "All inference is upstream llama.cpp — ggml, ggerganov and contributors did the hard parts."
        exit 0
    fi
    # --from without --system: plain user-local install of the local build.
    install_local "$FROM_BIN" "bootstrap, unverified channel"
    exit 0
fi

OS=$(uname -s)
ARCH=$(uname -m)
case "$ARCH" in
    x86_64) RUST_ARCH=x86_64 ;;
    aarch64 | arm64) RUST_ARCH=aarch64 ;;
    *) error "unsupported architecture: $ARCH (supported: x86_64/amd64, aarch64/arm64)" ;;
esac
case "$OS" in
    Linux) TARGET="${RUST_ARCH}-unknown-linux-$(pick_libc)" ;;
    Darwin) TARGET="${RUST_ARCH}-apple-darwin" ;;
    *) error "unsupported OS: $OS (this installer covers Linux/macOS; Windows uses install.ps1)" ;;
esac

RELEASE_PATH="releases/latest"
[ -n "${PALLAMA_VERSION:-}" ] && RELEASE_PATH="releases/tags/${PALLAMA_VERSION}"
status "Fetching release metadata from ${API_BASE}/${RELEASE_PATH}"
JSON=$(fetch "${API_BASE}/${RELEASE_PATH}") ||
    build_from_checkout "could not fetch release metadata from ${API_BASE} (offline or rate limited)"

TAG=$(printf '%s' "$JSON" | sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')
[ -n "$TAG" ] || error "could not parse tag_name from release metadata"

ASSET="pallama-${TAG}-${TARGET}.tar.gz"

# Extract one field of the exact asset record. The assets array is flattened
# one field per line; matching starts at the asset's own "name" line and ends
# at the next "name" line, so a sibling asset's digest can never leak in.
asset_field() {
    printf '%s' "$JSON" | tr ',' '\n' | awk -v field="$1" -v want="\"$ASSET\"" '
        /"name":/ {
            if (in_asset) exit
            if (index($0, "\"name\":" want) > 0 || index($0, "\"name\": " want) > 0) in_asset = 1
            next
        }
        in_asset && index($0, "\"" field "\":") > 0 { print }
    ' | sed -n "s/.*\"$1\"[[:space:]]*:[[:space:]]*\"\([^\"]*\)\".*/\1/p" | sed -n '1p'
}

asset_names() {
    printf '%s' "$JSON" | tr ',' '\n' |
        sed -n 's/.*"name"[[:space:]]*:[[:space:]]*"\(pallama-[^"]*\)".*/\1/p'
}

URL=$(asset_field browser_download_url)
[ -n "$URL" ] ||
    build_from_checkout "no prebuilt asset ${ASSET} in release ${TAG} (available: $(asset_names | tr '\n' ' '))"

DIGEST=$(asset_field digest)
case "${DIGEST:-}" in
    sha256:*) EXPECT=${DIGEST#sha256:} ;;
    *) error "release metadata has no sha256 digest for ${ASSET} yet — GitHub computes it shortly after upload; retry in a minute. Refusing unverified install." ;;
esac

TMP=$(mktemp -d) || error "mktemp failed"
cleanup() { rm -rf "$TMP"; }
trap cleanup EXIT INT TERM

status "Downloading ${ASSET} (${TAG})..."
TARBALL="$TMP/pallama.tar.gz"
fetch "$URL" -o "$TARBALL" || error "download failed: $URL"

if command -v sha256sum >/dev/null 2>&1; then
    GOT=$(sha256sum "$TARBALL" | cut -d' ' -f1)
elif command -v shasum >/dev/null 2>&1; then
    GOT=$(shasum -a 256 "$TARBALL" | cut -d' ' -f1)
else
    error "need sha256sum or shasum to verify the download"
fi
[ "$GOT" = "$EXPECT" ] ||
    error "sha256 mismatch for ${ASSET}: expected ${EXPECT}, got ${GOT} — download corrupted or tampered; not installing"
status "sha256 verified"

EXDIR="$TMP/extract"
mkdir -p "$EXDIR"
tar -xzf "$TARBALL" -C "$EXDIR" || error "failed to extract tarball"
[ -f "$EXDIR/pallama" ] || error "tarball did not contain a 'pallama' binary at its root"

INSTALL_DIR="${PALLAMA_INSTALL_DIR:-$HOME/.local/bin}"
mkdir -p "$INSTALL_DIR" || error "cannot create ${INSTALL_DIR}"
NEW="$INSTALL_DIR/pallama.new.$$"
install -m 0755 "$EXDIR/pallama" "$NEW" || error "cannot write into ${INSTALL_DIR}"
mv -f "$NEW" "$INSTALL_DIR/pallama"

PIDFILE="$HOME/.local/share/pallama/run/pallama.pid"
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; then
    status "the pallama daemon is still running the old binary; run 'pallama stop' and any command to restart on ${TAG}"
fi

case ":$PATH:" in
    *":$INSTALL_DIR:"*) ;;
    *) status "NOTE: ${INSTALL_DIR} is not on your PATH — add it: export PATH=\"${INSTALL_DIR}:\$PATH\"" ;;
esac

if [ "$WITH_UNIT" = 1 ]; then
    command -v systemctl >/dev/null 2>&1 || error "--with-systemd-unit requires systemctl"
    ABS_INSTALL_DIR=$(cd "$INSTALL_DIR" && pwd -P)
    UNIT_DIR="$HOME/.config/systemd/user"
    mkdir -p "$UNIT_DIR"
    cat > "$UNIT_DIR/pallama.service" <<EOF
[Unit]
Description=Pallama daemon (llama.cpp orchestration)

[Service]
ExecStart=${ABS_INSTALL_DIR}/pallama serve
Restart=on-failure
RestartSec=3

[Install]
WantedBy=default.target
EOF
    systemctl --user daemon-reload
    systemctl --user enable --now pallama.service ||
        error "failed to enable the user unit (no systemd user session? try: loginctl enable-linger \$USER)"
    status "systemd user service installed and started (logs: journalctl --user -u pallama)"
fi

VER=$("$INSTALL_DIR/pallama" --version 2>/dev/null || echo "(version check failed)")
status "Installed pallama ${VER} to ${INSTALL_DIR}/pallama"
status "Next: pallama engine update && pallama pull <model> && pallama run <model>"
status "All inference is upstream llama.cpp — ggml, ggerganov and contributors did the hard parts."

}

main "$@"
