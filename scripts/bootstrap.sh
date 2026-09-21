#!/bin/sh
# blazar environment bootstrap. Two consumers:
#   1. CONTRIBUTORS (default mode): every dependency needed to build,
#      test and lint from source.
#   2. scripts/install.sh (--minimal): just the compile toolchain (cc,
#      make, rust) when a user asked to build from source on a box that
#      lacks it.
# The SHIPPED binary has no runtime dependencies (bundled SQLite, rustls)
# — end users on the release-binary lane never need this script.
#
# Idempotent: installs only what is missing. Linux (apt/dnf/pacman/zypper)
# + macOS (brew / Xcode CLT). Windows contributors: install Rust from
# https://rustup.rs, then `cargo test --workspace` (binary install:
# scripts/install.ps1).
#
# Flags:
#   --check   report status, change nothing
#   --minimal only cc + make + rust (the from-source install lane;
#             skips git/python3/shellcheck/xz dev extras)
#
# Environment: BLAZAR_SUDO — pass-through wrapper, or empty to run
# unprivileged (unset-only default — sudo), same knob as install.sh.

# Wrap everything in main() so a truncated partial download cannot execute
# half a script.
main() {

set -eu

status() { echo ">>> $*" >&2; }
error() { echo "ERROR: $*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

CHECK=0
MINIMAL=0
for arg in "$@"; do
    case "$arg" in
        --check) CHECK=1 ;;
        --minimal) MINIMAL=1 ;;
        *) error "unknown flag: $arg (supported: --check, --minimal)" ;;
    esac
done

SUDO="${BLAZAR_SUDO-sudo}"
[ "$(id -u)" -eq 0 ] && SUDO=

OS=$(uname -s)
case "$OS" in
    Linux | Darwin) ;;
    MINGW* | MSYS* | CYGWIN*)
        status "Windows: install Rust from https://rustup.rs, then: cargo test --workspace"
        status "binary install: scripts/install.ps1"
        exit 0
        ;;
    *) error "unsupported OS: $OS" ;;
esac

# Tool -> distro package for the detected package manager.
pkg_for() { # pkg_for <pm> <tool>; prints package name or returns 1 (skip)
    case "$1/$2" in
        apt-get/cc) echo build-essential ;;
        apt-get/xz) echo xz-utils ;;
        dnf/cc) echo gcc ;;
        dnf/shellcheck) echo ShellCheck ;;
        pacman/cc) echo base-devel ;;
        pacman/python3) echo python ;;
        zypper/cc) echo gcc ;;
        zypper/shellcheck) echo ShellCheck ;;
        brew/cc) return 1 ;; # Xcode Command Line Tools, not a formula
        *) echo "$2" ;;
    esac
}

if [ "$MINIMAL" = 1 ]; then
    TOOLS="cc make"
else
    TOOLS="cc curl xz git python3 shellcheck"
fi

MISSING_TOOLS=
for t in $TOOLS; do
    if have "$t"; then
        if [ "$CHECK" = 1 ]; then status "ok      $t"; fi
    else
        status "missing $t"
        MISSING_TOOLS="$MISSING_TOOLS $t"
    fi
done

RUST_OK=0
if have cargo && have rustc; then
    if [ "$CHECK" = 1 ]; then status "ok      rust (cargo + rustc)"; fi
else
    RUST_OK=1
    status "missing rust (cargo + rustc)"
fi

if [ "$CHECK" = 1 ]; then
    status "check complete (missing tools listed above run nothing here)"
    exit 0
fi

# --- install missing system tools --------------------------------------
if [ -n "$MISSING_TOOLS" ]; then
    if [ "$OS" = Darwin ]; then
        case "$MISSING_TOOLS" in
            *cc*)
                if [ "$MINIMAL" = 1 ]; then
                    error "cc missing: run 'xcode-select --install' for the Command Line Tools, then re-run the installer"
                fi
                status "cc missing: run 'xcode-select --install' for the Command Line Tools, then re-run this script"
                ;;
        esac
        have brew || error "brew missing — install from https://brew.sh, then re-run"
        PKGS=
        for t in $MISSING_TOOLS; do
            if p=$(pkg_for brew "$t"); then PKGS="$PKGS $p"; fi
        done
        [ -n "$PKGS" ] && brew install $PKGS
    else
        PM=
        for c in apt-get dnf pacman zypper; do
            if have "$c"; then PM=$c; break; fi
        done
        [ -n "$PM" ] || error "no supported package manager (apt-get/dnf/pacman/zypper); install manually: $MISSING_TOOLS"
        PKGS=
        for t in $MISSING_TOOLS; do
            if p=$(pkg_for "$PM" "$t"); then PKGS="$PKGS $p"; fi
        done
        [ -n "$PKGS" ] || error "nothing installable for:$MISSING_TOOLS (see notes above)"
        case "$PM" in
            apt-get) $SUDO apt-get update && $SUDO apt-get install -y $PKGS ;;
            dnf) $SUDO dnf install -y $PKGS ;;
            pacman) $SUDO pacman -Sy --noconfirm --needed $PKGS ;;
            zypper) $SUDO zypper --non-interactive install $PKGS ;;
        esac
    fi
    status "installed:$PKGS"
fi

# --- rust toolchain (opt-in flow, minimal profile) ---------------------
if [ "$RUST_OK" = 1 ]; then
    have curl || error "curl missing — needed to fetch rustup; install curl and re-run"
    status "installing rust via rustup (minimal profile)"
    curl --proto '=https' --tlsv1.2 -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal
    . "$HOME/.cargo/env"
    have cargo || error "rustup finished but cargo not on PATH — open a new shell and re-run"
    status "rust installed (persists in ~/.cargo/bin; source ~/.cargo/env in your profile)"
fi

if [ "$MINIMAL" = 1 ]; then
    status "compile toolchain ready (installer will continue the source build)"
else
    status "dev environment ready. Next:"
    status "  cargo test --workspace -- --test-threads=1"
    status "  cargo clippy --workspace --all-targets -- -D warnings"
    status "  sh tests/install_e2e.sh"
    status "  sh scripts/install.sh --build   # system-wide install from this checkout"
fi

}

main "$@"
