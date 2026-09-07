#!/bin/sh
# pallama DEVELOPER-environment bootstrap: every dependency a contributor
# needs to build, test and lint from source. The SHIPPED binary has no
# runtime dependencies (bundled SQLite, rustls) — end users never need
# this; scripts/install.sh stays dep-free on purpose.
#
# Idempotent: installs only what is missing. Linux (apt/dnf/pacman/zypper)
# + macOS (brew). Windows contributors: install Rust from https://rustup.rs,
# then `cargo test --workspace` (binary install: scripts/install.ps1).
#
# Flags:
#   --check   report status, change nothing
#
# Environment: PALLAMA_SUDO — pass-through wrapper, or empty to run
# unprivileged (unset-only default — sudo), same knob as install.sh.

# Wrap everything in main() so a truncated partial download cannot execute
# half a script.
main() {

set -eu

CHECK=0
[ "${1:-}" = "--check" ] && CHECK=1

status() { echo ">>> $*" >&2; }
error() { echo "ERROR: $*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

SUDO="${PALLAMA_SUDO-sudo}"
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

MISSING_TOOLS=
for t in cc curl xz git python3 shellcheck; do
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
    status "installing rust via rustup (minimal profile)"
    curl --proto '=https' --tlsv1.2 -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal
    . "$HOME/.cargo/env"
    have cargo || error "rustup finished but cargo not on PATH — open a new shell and re-run"
    status "rust installed (persists in ~/.cargo/bin; source ~/.cargo/env in your profile)"
fi

status "dev environment ready. Next:"
status "  cargo test --workspace -- --test-threads=1"
status "  cargo clippy --workspace --all-targets -- -D warnings"
status "  sh tests/install_e2e.sh"
status "  sh scripts/install.sh --build   # system-wide install from this checkout"

}

main "$@"
