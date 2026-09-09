#!/bin/sh
# pallama installer — Linux and macOS. System-wide, like ollama:
# root-owned binary in /usr/local/bin + systemd unit (Linux) or launchd
# service (macOS). There is NO user-path install mode; a second copy in
# ~/.local/bin is how stale-binary daemon races happen (doctor flags them).
#
#   curl --proto '=https' --tlsv1.2 -fsSL <raw-url-of-this-file> | sh
#
# From a checkout: builds fresh with cargo first (zero-arg runs never
# install a stale target/release), then installs system-wide. Missing
# compile toolchain (cc/rust)? scripts/bootstrap.sh --minimal provisions
# it automatically — announce-then-act, never silently.
#
# Verifies the asset sha256 from the GitHub release API (the same source
# of truth as `pallama engine update`) before installing anything.
#
# Flags:
#   --build           force the source path (no release channel contact;
#                     bootstraps the toolchain when missing)
#   --from <binary>   install a locally built binary (bootstrap/offline)
#   --uninstall       remove binary + units (models/config are user data,
#                     kept: ~/.local/share/pallama, ~/.config/pallama)
#
# Environment overrides:
#   PALLAMA_VERSION            pin a release tag (e.g. v0.3.0)
#   PALLAMA_REPO               GitHub owner/name hosting releases (unset:
#                              derived from the enclosing checkout's git
#                              origin when available)
#   PALLAMA_BOOTSTRAP          override the bootstrap script path (default:
#                              <checkout>/scripts/bootstrap.sh)
#   PALLAMA_AUTO_BOOTSTRAP     0 = never auto-install a toolchain; auto
#                              mode falls back to the release channel
#   PALLAMA_FORCE_BOOTSTRAP    1 = run the bootstrap even when a toolchain
#                              exists (test knob, like PALLAMA_SUDO)
#   PALLAMA_INSTALL_BASE_URL   replace the GitHub API base (mirrors, tests)
#   PALLAMA_INSTALL_ENGINE     0 = skip the engine bootstrap (default: install
#                              the llama.cpp engine so the box is infer-ready)
#   PALLAMA_INSTALL_MODEL      optional first model to pull (e.g.
#                              qwen2.5:0.5b) — opt-in, never defaulted
#   PALLAMA_SYSTEM_BIN_DIR     binary destination (default /usr/local/bin)
#   PALLAMA_SERVICE_USER/GROUP unit user/group (default: invoking user)
#   GITHUB_TOKEN               optional API token (rate limits, private repos)

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

# PALLAMA_REPO unset: derive owner/name from the enclosing checkout's
# git origin so a one-liner run inside a clone (or from the README of a
# fork) needs no exports. A git remote is the only trustworthy source —
# there is no hardcoded canonical home, so forks keep installing from
# their own releases.
derive_repo() {
    command -v git >/dev/null 2>&1 || return 1
    _url=$(cd "$(dirname "$0")" 2>/dev/null && git remote get-url origin 2>/dev/null) || return 1
    case "$_url" in
        git@*) _url=${_url#git@*} _url=${_url#*:} ;;
        https://*) _url=${_url#https://*/} ;;
        http://*) _url=${_url#http://*/} ;;
        ssh://git@*) _url=${_url#ssh://git@*} _url=${_url#*/} ;;
        ssh://*) _url=${_url#ssh://*/} ;;
        *) return 1 ;;
    esac
    _url=${_url%.git}
    case "$_url" in
        */*/*) return 1 ;; # extra path segments — not owner/name
        */*) printf '%s\n' "$_url" ;;
        *) return 1 ;;
    esac
}

# Provision a missing compile toolchain (cc + rust) via bootstrap.sh
# --minimal, announce-then-act. Returns 0 when a source build is
# possible (already present, or bootstrapped now); 1 when declined
# (PALLAMA_AUTO_BOOTSTRAP=0), impossible (no bootstrap script) or the
# bootstrap itself failed — NEVER a silent path: the caller reports.
ensure_toolchain() {
    if [ "${PALLAMA_FORCE_BOOTSTRAP:-0}" != 1 ]; then
        command -v cargo >/dev/null 2>&1 && command -v cc >/dev/null 2>&1 && return 0
    fi
    [ "${PALLAMA_AUTO_BOOTSTRAP:-1}" = 1 ] || return 1
    _bs="${PALLAMA_BOOTSTRAP:-}"
    if [ -z "$_bs" ]; then
        _ck=$(find_checkout 2>/dev/null) && [ -f "$_ck/scripts/bootstrap.sh" ] && _bs="$_ck/scripts/bootstrap.sh"
    fi
    [ -n "$_bs" ] || return 1
    status "toolchain missing — bootstrapping cc/make/rust via: sh $_bs --minimal"
    if ! sh "$_bs" --minimal; then
        status "WARN: toolchain bootstrap failed (output above) — source build unavailable on this box"
        return 1
    fi
    # rustup ran in a child process; its ~/.cargo/env can't reach us, so
    # refresh PATH manually.
    PATH="$HOME/.cargo/bin:$PATH"
    command -v cargo >/dev/null 2>&1 && command -v cc >/dev/null 2>&1
}

UNINSTALL=0
FROM_BIN=
FORCE_BUILD=0
while [ $# -gt 0 ]; do
    case "$1" in
        --uninstall) UNINSTALL=1 ;;
        --build) FORCE_BUILD=1 ;;
        --from) shift; [ $# -gt 0 ] || error "--from needs a binary path"; FROM_BIN=$1 ;;
        --from=*) FROM_BIN=${1#--from=} ;;
        *) error "unknown option: $1 (supported: --build, --from <binary>, --uninstall)" ;;
    esac
    shift
done

# ollama-style privilege: root runs plain, everyone else needs sudo.
# There is no user-local fallback — a second pallama in ~/.local/bin is
# precisely how stale-binary daemon races happen.
# PALLAMA_SUDO is a test/mirror knob: a pass-through wrapper, or empty to
# run everything unprivileged (unset-only default — sudo).
SUDO="${PALLAMA_SUDO-sudo}"
[ "$(id -u)" -eq 0 ] && SUDO=

# Privilege is enforced where it's needed: the privileged install command
# itself fails with "cannot create /usr/local/bin (need sudo?)" when root
# is genuinely unavailable — no fragile tty/sudo probing up front.

# --uninstall: remove what install.sh put here (binary + units).
# NEVER touches models or config — those are user data.
if [ "$UNINSTALL" = 1 ]; then
    if command -v pallama >/dev/null 2>&1; then pallama stop >/dev/null 2>&1 || true; fi
    for BIN in "${PALLAMA_SYSTEM_BIN_DIR:-/usr/local/bin}/pallama" "$HOME/.local/bin/pallama"; do
        if [ -e "$BIN" ]; then
            ([ -w "$(dirname "$BIN")" ] && rm -f "$BIN") || $SUDO rm -f "$BIN"
            status "removed $BIN"
        fi
    done
    if [ -f /etc/systemd/system/pallama.service ]; then
        $SUDO systemctl disable --now pallama 2>/dev/null || true
        $SUDO rm -f /etc/systemd/system/pallama.service && status "removed system unit"
        $SUDO systemctl daemon-reload 2>/dev/null || true
    fi
    if [ -f "$HOME/.config/systemd/user/pallama.service" ]; then
        systemctl --user disable --now pallama 2>/dev/null || true
        rm -f "$HOME/.config/systemd/user/pallama.service" && status "removed legacy user unit"
        systemctl --user daemon-reload 2>/dev/null || true
    fi
    if [ "$(uname -s)" = Darwin ]; then
        launchctl bootout "gui/$(id -u)/dev.pallama" 2>/dev/null ||
            launchctl unload "$HOME/Library/LaunchAgents/dev.pallama.plist" 2>/dev/null || true
        if [ -f "$HOME/Library/LaunchAgents/dev.pallama.plist" ]; then
            rm -f "$HOME/Library/LaunchAgents/dev.pallama.plist" && status "removed launch agent"
        fi
        if [ -f /Library/LaunchDaemons/dev.pallama.plist ]; then
            $SUDO launchctl bootout system/dev.pallama 2>/dev/null ||
                $SUDO launchctl unload /Library/LaunchDaemons/dev.pallama.plist 2>/dev/null || true
            $SUDO rm -f /Library/LaunchDaemons/dev.pallama.plist && status "removed launch daemon"
        fi
    fi
    status "uninstalled. models/config kept at ~/.local/share/pallama and ~/.config/pallama (delete manually if desired)"
    exit 0
fi

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
    # Prints the built binary path on success; returns non-zero when a
    # source build is impossible or fails (caller decides: fatal vs
    # loud fallback). NEVER exits from here — when called via $(...) an
    # exit only kills the subshell and the caller would silently degrade
    # to the release channel.
    command -v cargo >/dev/null 2>&1 || return 1
    CK=$(find_checkout) || return 1
    status "building from source: cargo build --release -p pallama-cli (in ${CK})"
    if ! (cd "$CK" && cargo build --release -p pallama-cli); then
        echo "ERROR: source build failed (cargo output above)" >&2
        return 1
    fi
    [ -f "$CK/target/release/pallama" ] || { echo "ERROR: build produced no target/release/pallama" >&2; return 1; }
    echo "$CK/target/release/pallama"
}

# Zero-argument auto mode: a checkout present -> compile FRESH (never a
# stale target/release), then install system-wide. Checkout-less runs
# (curl | sh) fall through to the release channel. A missing toolchain
# is bootstrapped first (announce-then-act); every failure falls back to
# the release channel LOUD — never silently.
if [ -z "$FROM_BIN" ] && [ "$FORCE_BUILD" = 0 ] &&
   [ -z "${PALLAMA_REPO:-}" ] && [ -z "${PALLAMA_INSTALL_BASE_URL:-}" ]; then
    if find_checkout >/dev/null 2>&1; then
        if ! ensure_toolchain; then
            status "auto: toolchain unavailable (bootstrap failed or PALLAMA_AUTO_BOOTSTRAP=0) — falling back to the release channel"
        elif FROM_BIN=$(build_from_checkout); then
            status "auto: installing the fresh build system-wide (binary + service)"
        else
            status "auto: source build failed — falling back to the release channel (errors above)"
        fi
    fi
fi

if [ -n "$FROM_BIN" ]; then
    # Bootstrap mode: install a locally built binary (no release needed).
    [ -f "$FROM_BIN" ] || error "--from: no such file: $FROM_BIN"
    [ -x "$FROM_BIN" ] || error "--from: not executable: $FROM_BIN"
    "$FROM_BIN" --version >/dev/null 2>&1 || error "--from: binary does not run: $FROM_BIN"
    status "Bootstrap install from $FROM_BIN (skipping release download)"
fi

# --build: force the source path (audited/offline installs; never touches
# the release channel). Toolchain bootstrapping is part of the deal —
# the user asked for a source build, so a missing cc/rust is provisioned,
# and a failed bootstrap is FATAL (explicit intent, no fallback).
if [ "$FORCE_BUILD" = 1 ] && [ -z "${FROM_BIN:-}" ]; then
    ensure_toolchain ||
        error "--build: no compile toolchain and bootstrap failed — install Rust (https://rustup.rs) + a C compiler, or set PALLAMA_BOOTSTRAP=<path to scripts/bootstrap.sh>"
    FROM_BIN=$(build_from_checkout) ||
        error "--build: source build failed (cargo output above)"
    status "--build: source path forced (no release channel contact)"
fi

# Repo guard is release-channel only: --from bootstrap and mirror/test
# base URLs never touch the GitHub release API. PALLAMA_REPO unset:
# derive owner/name from the checkout's git origin (README one-liners
# work inside a clone with zero exports).
if [ -z "${PALLAMA_INSTALL_BASE_URL:-}" ] && [ -z "${FROM_BIN:-}" ] && [ -z "$REPO" ]; then
    REPO=$(derive_repo) || REPO=
    [ -n "$REPO" ] && status "PALLAMA_REPO unset — derived from git origin: ${REPO}"
    API_BASE="${PALLAMA_INSTALL_BASE_URL:-https://api.github.com/repos/${REPO}}"
fi
if [ -z "${PALLAMA_INSTALL_BASE_URL:-}" ] && [ -z "${FROM_BIN:-}" ] && [ -z "$REPO" ]; then
    error "PALLAMA_REPO is not set and no git origin to derive it from. Either export PALLAMA_REPO=owner/pallama (the GitHub repo hosting releases) and re-run, or clone the repo and run scripts/install.sh from inside it (compiles from source, no release needed)"
fi

# macOS service via launchd (called from install_system when systemctl
# is absent but launchctl exists). Non-root: LaunchAgent at login. Root:
# LaunchDaemon at boot with UserName= so the daemon still runs
# unprivileged — mirroring the systemd unit's User=. KeepAlive is the
# launchd spelling of Restart=always.
install_launchd() {
    LABEL=dev.pallama
    if [ "$(id -u)" -eq 0 ]; then
        PLIST_DIR=/Library/LaunchDaemons
        USER_KEY="
    <key>UserName</key>
    <string>${SVC_USER}</string>"
    else
        PLIST_DIR="$HOME/Library/LaunchAgents"
        USER_KEY=
    fi
    PLIST_PATH="$PLIST_DIR/$LABEL.plist"
    PLIST_BODY=$(cat <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>${BIN_DIR}/pallama</string>
        <string>serve</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>ProcessType</key>
    <string>Background</string>${USER_KEY}
</dict>
</plist>
EOF
)
    if [ "$(id -u)" -eq 0 ]; then
        launchctl bootout "system/$LABEL" 2>/dev/null || true
        printf '%s\n' "$PLIST_BODY" > "$PLIST_PATH"
        launchctl bootstrap system "$PLIST_PATH" 2>/dev/null || launchctl load "$PLIST_PATH"
    else
        UID_N=$(id -u)
        launchctl bootout "gui/${UID_N}/$LABEL" 2>/dev/null ||
            launchctl unload "$PLIST_PATH" 2>/dev/null || true
        mkdir -p "$PLIST_DIR"
        printf '%s\n' "$PLIST_BODY" > "$PLIST_PATH"
        launchctl bootstrap "gui/$UID_N" "$PLIST_PATH" 2>/dev/null ||
            launchctl load "$PLIST_PATH" || error "loading $PLIST_PATH failed"
    fi
    poll_healthz "logs: log show --predicate 'process == \"pallama\"' --last 5m"
}

# Poll the configured host (quoted or bare TOML), not a hardcoded
# loopback — a config bound to a specific interface answers there.
# Shared by the systemd and launchd install paths.
poll_healthz() {
    # poll_healthz <log-hint>
    HOST=$(sed -n 's/^host[[:space:]]*=[[:space:]]*//p' "$HOME/.config/pallama/config.toml" 2>/dev/null | head -1 | tr -d '"')
    PORT=$(sed -n 's/^port[[:space:]]*=[[:space:]]*\([0-9]*\).*/\1/p' "$HOME/.config/pallama/config.toml" 2>/dev/null | head -1)
    HOST=${HOST:-127.0.0.1}
    PORT=${PORT:-11434}
    i=0
    while ! curl -s --max-time 2 "http://${HOST}:${PORT}/healthz" 2>/dev/null | grep -q ok; do
        i=$((i + 1)); [ "$i" -gt 30 ] && break
        sleep 1
    done
    if curl -s --max-time 2 "http://${HOST}:${PORT}/healthz" 2>/dev/null | grep -q ok; then
        status "service active; pallama healthy on :${PORT} ($1)"
    else
        status "service enabled; healthz not answering on :${PORT} yet — check: $1"
    fi
}

# ---- system-wide install: root-owned binary + systemd unit ----
# One implementation for every channel (build/auto/from/release). Like
# ollama: enable the unit and RESTART it on upgrade so the new binary is
# live immediately; stop any user-started daemon first so the unit never
# fights it for the port.
install_system() {
    # install_system <binary>
    BIN_DIR="${PALLAMA_SYSTEM_BIN_DIR:-/usr/local/bin}"
    $SUDO mkdir -p "$BIN_DIR" || error "cannot create ${BIN_DIR} (need sudo?)"
    # Replace a possibly-running binary without ETXTBSY: temp file + rename
    # (the running process keeps its inode; new execs get the new binary).
    # Root-owned like ollama when we have root; plain install otherwise
    # (mirrors/tests run through a pass-through "sudo").
    $SUDO install -o0 -g0 -m0755 "$1" "$BIN_DIR/pallama.new.$$" 2>/dev/null ||
    $SUDO install -m0755 "$1" "$BIN_DIR/pallama.new.$$" ||
    error "install to ${BIN_DIR} failed"
    $SUDO mv -f "$BIN_DIR/pallama.new.$$" "$BIN_DIR/pallama"
    # A user-started daemon owns the port; the unit would crash-loop.
    PIDFILE="$HOME/.local/share/pallama/run/pallama.pid"
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; then
        kill -TERM "$(cat "$PIDFILE")" 2>/dev/null || true
        i=0
        while kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null && [ "$i" -lt 10 ]; do
            i=$((i + 1)); sleep 1
        done
    fi
    # Under `sudo` the invoking user is SUDO_USER (id -un would say root);
    # the daemon must run as the data-owning user (engines/models live in
    # that home), so the unit never points at /root's empty store.
    SVC_USER="${PALLAMA_SERVICE_USER:-${SUDO_USER:-$(id -un)}}"
    SVC_GROUP="${PALLAMA_SERVICE_GROUP:-$(id -gn "$SVC_USER")}"
    UNIT_PATH="${PALLAMA_UNIT_PATH:-/etc/systemd/system/pallama.service}"
    SYSTEMCTL="${PALLAMA_SYSTEMCTL:-systemctl}"
    if command -v "$SYSTEMCTL" >/dev/null 2>&1; then
        # SupplementaryGroups only for groups that exist on this box —
        # systemd rejects the whole unit when a listed group is missing
        # (containers, WSL, minimal images ship without render/video).
        SG=
        for g in render video; do
            if getent group "$g" >/dev/null 2>&1 ||
               grep -q "^${g}:" /etc/group 2>/dev/null; then
                SG="${SG}${SG:+ }$g"
            fi
        done
        SG_LINE=
        [ -n "$SG" ] && SG_LINE="SupplementaryGroups=$SG"
        $SUDO mkdir -p "$(dirname "$UNIT_PATH")"
        UNIT=$(cat <<EOF
[Unit]
Description=Pallama daemon (llama.cpp orchestration)
After=network-online.target
Wants=network-online.target

[Service]
ExecStart=${BIN_DIR}/pallama serve
User=${SVC_USER}
Group=${SVC_GROUP}
${SG_LINE}
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF
)
        printf '%s\n' "$UNIT" | $SUDO tee "$UNIT_PATH" >/dev/null || error "writing $UNIT_PATH failed"
        $SUDO "$SYSTEMCTL" daemon-reload || error "systemctl daemon-reload failed"
        # Upgrade-in-place: restart an already-active unit (like ollama),
        # enable+start otherwise.
        if $SUDO "$SYSTEMCTL" is-active --quiet pallama 2>/dev/null; then
            $SUDO "$SYSTEMCTL" restart pallama || error "restarting pallama.service failed"
        else
            $SUDO "$SYSTEMCTL" enable --now pallama || error "enabling pallama.service failed"
        fi
        poll_healthz "logs: journalctl -u pallama"
        SERVICE_DESC=" + systemd unit ${UNIT_PATH}"
    elif [ "$(uname -s)" = Darwin ] && command -v launchctl >/dev/null 2>&1; then
        install_launchd
        SERVICE_DESC=" + launchd service ${PLIST_PATH}"
    else
        status "no service manager found — binary installed at ${BIN_DIR}/pallama; start it manually: pallama serve"
    fi
    VER=$("$BIN_DIR/pallama" --version 2>/dev/null || echo "(version check failed)")
    status "Installed pallama ${VER} system-wide (${BIN_DIR}/pallama${SERVICE_DESC:-})"
    # The stale-copy race: a leftover user-path copy gets resurrected by
    # services with their own PATH. Remove it as part of every install.
    # -ef (same inode, symlinks followed) works where `readlink -f` does
    # not (old macOS): skip removal only when the copy IS the system file.
    # shellcheck disable=SC3013 # XSI extension; dash/busybox/bash/BSD sh
    # all implement it, and the degraded path removes a copy policy wants
    # gone anyway.
    if [ -e "$HOME/.local/bin/pallama" ] &&
       ! [ "$HOME/.local/bin/pallama" -ef "$BIN_DIR/pallama" ]; then
        rm -f "$HOME/.local/bin/pallama" && status "removed stale user-path copy ~/.local/bin/pallama"
    fi
    # One-click readiness: persist config migrations (legacy api_keys ->
    # [[keys]] etc.) so the first `pallama` invocation never FAILs on an
    # old config. Best-effort: a missing config or an offline box must
    # not fail the install.
    if "$BIN_DIR/pallama" migrate >/dev/null 2>&1; then
        status "config migrated/verified (canonical form)"
    else
        status "config migration skipped (no config or parse issue — run: pallama migrate)"
    fi
    # One-click readiness: a pallama install without a llama.cpp engine
    # has ZERO inference capability. Bootstrap the engine now (the user
    # ran the installer — the download is sanctioned, never hidden).
    # Opt out: PALLAMA_INSTALL_ENGINE=0. Optional first model:
    # PALLAMA_INSTALL_MODEL=<repo> (pull lane, opt-in — model choice is
    # the user's call, not the installer's).
    if [ "${PALLAMA_INSTALL_ENGINE:-1}" != 0 ] &&
       ! "$BIN_DIR/pallama" engine list 2>/dev/null | grep -q '\[active\]'; then
        status "bootstrapping llama.cpp engine (pallama engine update — largest download of this install)..."
        if "$BIN_DIR/pallama" engine update --no-gate; then
            status "engine bootstrap complete"
        else
            status "WARN: engine bootstrap failed (offline?) — inference NOT ready. Run: pallama engine update"
        fi
    else
        status "engine already active (or bootstrap disabled) — skipping engine download"
    fi
    if [ -n "${PALLAMA_INSTALL_MODEL:-}" ]; then
        status "pulling first model: ${PALLAMA_INSTALL_MODEL}..."
        if "$BIN_DIR/pallama" pull "${PALLAMA_INSTALL_MODEL}"; then
            status "model ready: ${PALLAMA_INSTALL_MODEL}"
        else
            status "WARN: model pull failed — run: pallama pull ${PALLAMA_INSTALL_MODEL}"
        fi
    fi
    status "system ready — next steps:"
    if [ -n "${PALLAMA_INSTALL_MODEL:-}" ]; then
        status "  pallama run ${PALLAMA_INSTALL_MODEL}   # chat REPL (model pulled above)"
    else
        status "  pallama pull <model>    # e.g. pallama pull Qwen3-0.6B (find one: pallama search qwen3)"
    fi
    status "  pallama doctor          # health check with per-row hints"
    status "All inference is upstream llama.cpp — ggml, ggerganov and contributors did the hard parts."
}


# Bootstrap/--from/auto-build channels install directly.
if [ -n "$FROM_BIN" ]; then
    install_system "$FROM_BIN"
    exit 0
fi

OS=$(uname -s)
ARCH=$(uname -m)
case "$ARCH" in
    x86_64) RUST_ARCH=x86_64 ;;
    aarch64 | arm64) RUST_ARCH=aarch64 ;;
    # 32-bit ARM (armv8l = aarch64 kernel with 32-bit userland): the only
    # published build is the static musl hard-float one — no gnu variant
    # exists, so libc detection is skipped entirely.
    armv7l | armv7hl | armv8l) RUST_ARCH=armv7 ;;
    *) error "unsupported architecture: $ARCH (supported: x86_64/amd64, aarch64/arm64, armv7)" ;;
esac

case "$OS" in
    Linux)
        if [ "$RUST_ARCH" = armv7 ]; then
            LIBC=musleabihf
        else
            LIBC=$(pick_libc)
        fi
        STATUS_OS="linux"
        ASSET_TRIPLE_SUFFIX="unknown-linux-${LIBC}"
        ;;
    Darwin)
        LIBC="osx"
        STATUS_OS="macOS"
        ASSET_TRIPLE_SUFFIX="apple-darwin"
        ;;
    *) error "unsupported OS: $OS (this installer covers Linux and macOS)" ;;
esac

STATUS_OS_ARCH="${STATUS_OS} ${RUST_ARCH} (${LIBC})"
status "Looking for release ${PALLAMA_VERSION:-latest} for ${STATUS_OS_ARCH}..."

# Latest-or-pinned release metadata from the GitHub API.
if [ -n "${PALLAMA_VERSION:-}" ]; then
    RELEASE_PATH="releases/tags/${PALLAMA_VERSION}"
else
    RELEASE_PATH="releases/latest"
fi
RELEASE_JSON=$(fetch "${API_BASE}/${RELEASE_PATH}") ||
    error "failed to look up release ${PALLAMA_VERSION:-latest} in ${API_BASE} (PALLAMA_REPO set? network up?)"

TAG=$(printf '%s' "$RELEASE_JSON" | sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p' | head -1)
[ -n "$TAG" ] || error "could not parse tag_name from the release API response"

# Asset lines: pick the one matching this OS/arch (release-workflow naming
# contract: pallama-<tag>-<rust-triple>.tar.gz). Each sibling asset line
# carries its own digest, so a name/digest mix-up is impossible.
# Quoted variable = literal case match; a raw expansion in a case pattern
# would act as a glob (a hostile tag_name like "*" must not match).
WANTED="pallama-${TAG}-${RUST_ARCH}-${ASSET_TRIPLE_SUFFIX}.tar.gz"
printf '%s' "$RELEASE_JSON" | tr ',' '\n' | grep -E '"(name|digest)": *"' | \
    sed -e 's/.*"name": *"\([^"]*\)".*/name \1/' -e 's/.*"digest": *"\([^"]*\)".*/digest \1/' | \
    while read -r KIND VAL; do
        if [ "$KIND" = "name" ]; then
            case "$VAL" in
                "$WANTED") ASSET="$VAL" ;;
            esac
        elif [ "$KIND" = "digest" ] && [ -n "${ASSET:-}" ]; then
            printf '%s %s\n' "$ASSET" "$VAL"
            # Reset: a later asset's digest must never pair with this name
            # (only the pair emitted right after the matching name counts).
            ASSET=
        fi
    done > "${TMPDIR:-/tmp}/pallama-asset.$$"
read -r ASSET EXPECT < "${TMPDIR:-/tmp}/pallama-asset.$$" || true
rm -f "${TMPDIR:-/tmp}/pallama-asset.$$"
[ -n "$ASSET" ] || error "release ${TAG} has no asset matching pallama-${TAG}-${RUST_ARCH}-${ASSET_TRIPLE_SUFFIX}.tar.gz (available: $(printf '%s' "$RELEASE_JSON" | tr ',' '\n' | grep -o '"name": *"[^"]*"' | cut -d'"' -f4 | tr '\n' ' '))"
[ -n "$EXPECT" ] || error "release ${TAG} asset ${ASSET} carries no sha256 digest — refusing to install unverified"
EXPECT=${EXPECT#sha256:}

ASSET_URL=$(printf '%s' "$RELEASE_JSON" | tr ',' '\n' | grep -o '"browser_download_url": *"[^"]*"' |
    sed -e 's/.*"browser_download_url": *"\([^"]*\)".*/\1/' | grep -F "/$ASSET" | head -1)
ASSET_URL=${ASSET_URL:-${API_BASE}/releases/download/${TAG}/${ASSET}}
status "Downloading ${ASSET} (${TAG})..."
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
TARBALL="$TMP/${ASSET}"
fetch "$ASSET_URL" -o "$TARBALL" ||
    error "download failed: ${ASSET}"

status "Verifying sha256..."
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

install_system "$EXDIR/pallama"

}

main "$@"
