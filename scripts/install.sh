#!/bin/sh
# blazar installer — Linux and macOS. System-wide, like ollama:
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
# of truth as `blazar engine update`) before installing anything.
#
# Flags:
#   --build           force the source path (no release channel contact;
#                     bootstraps the toolchain when missing)
#   --from <binary>   install a locally built binary (bootstrap/offline)
#   --uninstall       remove binary + units (models/config are user data,
#                     kept: ~/.local/share/blazar, ~/.config/blazar)
#
# Environment overrides:
#   BLAZAR_VERSION            pin a release tag (e.g. v0.3.0)
#   BLAZAR_REPO               GitHub owner/name hosting releases (unset:
#                              derived from the enclosing checkout's git
#                              origin when available)
#   BLAZAR_BOOTSTRAP          override the bootstrap script path (default:
#                              <checkout>/scripts/bootstrap.sh)
#   BLAZAR_AUTO_BOOTSTRAP     0 = never auto-install a toolchain; auto
#                              mode falls back to the release channel
#   BLAZAR_FORCE_BOOTSTRAP    1 = run the bootstrap even when a toolchain
#                              exists (test knob, like BLAZAR_SUDO)
#   BLAZAR_INSTALL_BASE_URL   replace the GitHub API base (mirrors, tests)
#   BLAZAR_INSTALL_ENGINE     0 = skip the engine bootstrap (default: install
#                              the llama.cpp engine so the box is infer-ready)
#   BLAZAR_UNIT_MEMORY_HIGH  systemd soft memory ceiling for the daemon
#                              cgroup (default: 85% of RAM, recomputed by
#                              systemd at every unit start — adapts to RAM
#                              changes). Soft = reclaim/throttle only, never
#                              an OOM kill. Set to an empty string to omit
#                              the line entirely.
#   BLAZAR_AUTO_DRIVER        0 = skip the GPU preflight (default: detect PCI
#                              GPUs and, when the driver userspace is missing,
#                              install it from FIRST-PARTY distro repos only)
#   BLAZAR_INSTALL_MODEL      optional first model to pull (e.g.
#                              qwen2.5:0.5b) — opt-in, never defaulted
#   BLAZAR_SYSTEM_BIN_DIR     binary destination (default /usr/local/bin)
#   BLAZAR_SERVICE_USER/GROUP unit user/group (default: invoking user)
#   GITHUB_TOKEN               optional API token (rate limits, private repos)

# Wrap everything in main() so a truncated partial download cannot execute
# half a script (same guard technique as the ollama installer).
main() {

set -eu

REPO="${BLAZAR_REPO:-}"
API_BASE="${BLAZAR_INSTALL_BASE_URL:-https://api.github.com/repos/${REPO}}"

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

# BLAZAR_REPO unset: derive owner/name from the enclosing checkout's
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

# ---- invoking-user scope (sudo one-click support) --------------------------
# `curl | sudo sh install.sh` runs everything as root: user-keyed paths
# (HOME, cargo/rustup, the blazar store) belong to the INVOKING user.
# Every user-environment action (toolchain probe, source build, migrate/
# engine/pull CLIs) goes through as_user so nothing populates /root's
# store or leaves root-owned files in the user's checkout.
USER_HOME="$HOME"
BUILD_USER=
if [ "$(id -u)" -eq 0 ] && [ "${SUDO_USER:-}" != "" ] && [ "$SUDO_USER" != "root" ]; then
    BUILD_USER="$SUDO_USER"
    _hm=$(getent passwd "$SUDO_USER" 2>/dev/null | cut -d: -f6)
    [ -z "$_hm" ] && [ "$(uname -s)" = Darwin ] &&
        _hm=$(dscl . -read "/Users/$SUDO_USER" NFSHomeDirectory 2>/dev/null | awk '{print $NF}')
    [ -n "$_hm" ] && USER_HOME="$_hm"
fi
USER_CARGO_BIN="$USER_HOME/.cargo/bin"

as_user() {
    # as_user <cmd...> — run in the invoking user's environment (HOME +
    # cargo on PATH). Unprivileged installs run in place.
    if [ -z "$BUILD_USER" ]; then
        HOME="$USER_HOME" PATH="$USER_CARGO_BIN:$PATH" "$@"
    elif command -v runuser >/dev/null 2>&1; then
        runuser -u "$BUILD_USER" -- env HOME="$USER_HOME" PATH="$USER_CARGO_BIN:$PATH" "$@"
    else
        sudo -H -u "$BUILD_USER" env PATH="$USER_CARGO_BIN:$PATH" "$@"
    fi
}

# Provision a missing compile toolchain (cc + rust) via bootstrap.sh
# --minimal, announce-then-act. Returns 0 when a source build is
# possible (already present, or bootstrapped now); 1 when declined
# (BLAZAR_AUTO_BOOTSTRAP=0), impossible (no bootstrap script) or the
# bootstrap itself failed — NEVER a silent path: the caller reports.
# Probes and rustup land in the INVOKING user's environment (a root-run
# probe used to miss ~/.cargo and re-install rustup into /root).
ensure_toolchain() {
    if [ "${BLAZAR_FORCE_BOOTSTRAP:-0}" != 1 ]; then
        as_user sh -c 'command -v cargo >/dev/null 2>&1 && command -v cc >/dev/null 2>&1' && return 0
    fi
    [ "${BLAZAR_AUTO_BOOTSTRAP:-1}" = 1 ] || return 1
    _bs="${BLAZAR_BOOTSTRAP:-}"
    if [ -z "$_bs" ]; then
        _ck=$(find_checkout 2>/dev/null) && [ -f "$_ck/scripts/bootstrap.sh" ] && _bs="$_ck/scripts/bootstrap.sh"
    fi
    [ -n "$_bs" ] || return 1
    status "toolchain missing — bootstrapping cc/make/rust via: sh $_bs --minimal"
    # bootstrap.sh mixes root work (cc via the package manager) with
    # user work (rustup into $HOME/.cargo): run it as root but with the
    # invoking user's HOME, then hand the fresh ~/.cargo/.rustup back to
    # that user — root-owned rustup files break every later user build.
    if [ -n "$BUILD_USER" ]; then
        if ! HOME="$USER_HOME" sh "$_bs" --minimal; then
            status "WARN: toolchain bootstrap failed (output above) — source build unavailable on this box"
            return 1
        fi
        chown -R "$BUILD_USER:$(id -gn "$BUILD_USER")" "$USER_HOME/.cargo" "$USER_HOME/.rustup" 2>/dev/null
    elif ! sh "$_bs" --minimal; then
        status "WARN: toolchain bootstrap failed (output above) — source build unavailable on this box"
        return 1
    fi
    # rustup ran in a child process; its ~/.cargo/env can't reach us, so
    # verify through the user environment explicitly.
    as_user sh -c 'command -v cargo >/dev/null 2>&1 && command -v cc >/dev/null 2>&1'
}

# ---- GPU preflight: the zero-touch last mile ------------------------------
# The engine bootstrap at the end of install_system picks its asset by
# driver presence — a driverless NVIDIA box silently serves on CPU and
# nobody is told why. This preflight detects PCI GPU hardware (lspci) and,
# when the driver userspace is missing, installs it from FIRST-PARTY
# distro repos only — announce-then-act, never fatal to the install.
# Third-party-only sources (RPMFusion on Fedora, NVIDIA's own repo on
# openSUSE) are PRINTED with exact commands, never executed: adding a
# third-party repository as root is a line this installer does not cross.
# Opt out: BLAZAR_AUTO_DRIVER=0. CUDA asset choice stays in blazar
# itself (`blazar engine update` picks the newest CUDA build the driver
# supports — resolve_cuda_asset; runtimes are bundled, no toolkit needed).
gpu_preflight() {
    [ "$(uname -s)" = Linux ] || return 0
    # An explicit user opt-out is acknowledged before any internal-lane
    # skip (e.g. BLAZAR_INSTALL_ENGINE=0) — the operator asked for silence
    # by name and gets the confirmation line regardless of what else is on.
    if [ "${BLAZAR_AUTO_DRIVER:-1}" != 1 ]; then
        status "GPU preflight skipped (BLAZAR_AUTO_DRIVER=0)"
        return 0
    fi
    [ "${BLAZAR_INSTALL_ENGINE:-1}" != 0 ] || return 0
    # Containers/WSL1 expose no PCI bus — nothing to detect.
    if [ ! -d /sys/bus/pci ]; then
        status "GPU preflight: no PCI bus (container/WSL?) — skipped"
        return 0
    fi

    PM=
    for c in apt-get dnf pacman zypper; do
        if command -v "$c" >/dev/null 2>&1; then PM=$c; break; fi
    done

    # lspci (pciutils) is the only hardware oracle; provision it when
    # missing via the same announce-then-act lane as bootstrap.sh.
    if ! command -v lspci >/dev/null 2>&1; then
        if [ -z "$PM" ]; then
            status "GPU preflight: lspci missing and no supported package manager — skipped"
            return 0
        fi
        status "lspci missing — installing pciutils (GPU hardware detection)"
        case "$PM" in
            apt-get) $SUDO apt-get update && $SUDO apt-get install -y pciutils ;;
            dnf) $SUDO dnf install -y pciutils ;;
            pacman) $SUDO pacman -Sy --noconfirm --needed pciutils ;;
            zypper) $SUDO zypper --non-interactive install pciutils ;;
        esac || { status "WARN: pciutils install failed — GPU preflight skipped"; return 0; }
    fi
    command -v lspci >/dev/null 2>&1 ||
        { status "GPU preflight: lspci unavailable — skipped"; return 0; }

    # PCI vendor census over VGA (0300) + 3D-controller (0302) classes.
    # `lspci -n` lines look like `0000:01:00.0 0300: 10de:28a0 (rev a1)`;
    # the space-prefixed ` 10de:` token anchors the vendor match (a bare
    # `10de:` glob would false-positive on device ids).
    PCI_GPUS=$(lspci -n -d ::0300 2>/dev/null; lspci -n -d ::0302 2>/dev/null)
    [ -n "$PCI_GPUS" ] ||
        { status "GPU preflight: no PCI display controllers — CPU lane"; return 0; }
    HW_NVIDIA=0 HW_AMD=0 HW_INTEL=0
    case "$PCI_GPUS" in *" 10de:"*) HW_NVIDIA=1 ;; esac
    case "$PCI_GPUS" in *" 1002:"*) HW_AMD=1 ;; esac
    case "$PCI_GPUS" in *" 8086:"*) HW_INTEL=1 ;; esac

    if [ "$HW_NVIDIA" = 1 ]; then
        if ! command -v nvidia-smi >/dev/null 2>&1; then
            status "NVIDIA GPU detected (PCI 10de:) but no NVIDIA driver userspace — installing from distro repos"
            NVIDIA_INSTALLED=0
            case "$PM" in
                apt-get)
                    if command -v ubuntu-drivers >/dev/null 2>&1; then
                        status "  running: ubuntu-drivers autoinstall (picks the right driver branch)"
                        if $SUDO ubuntu-drivers autoinstall; then NVIDIA_INSTALLED=1; fi
                    fi
                    if [ "$NVIDIA_INSTALLED" != 1 ]; then
                        status "  running: apt-get install -y nvidia-driver (Debian metapackage)"
                        if $SUDO apt-get update && $SUDO apt-get install -y nvidia-driver; then NVIDIA_INSTALLED=1; fi
                    fi
                    ;;
                dnf)
                    # akmod-nvidia lives in RPMFusion (third party) — only
                    # install when the user has already enabled it.
                    if $SUDO dnf repolist --enabled 2>/dev/null | grep -qi rpmfusion; then
                        status "  running: dnf install -y akmod-nvidia (kernel module compiles — takes minutes)"
                        if $SUDO dnf install -y akmod-nvidia; then NVIDIA_INSTALLED=1; fi
                    fi
                    ;;
                pacman)
                    status "  running: pacman -S nvidia nvidia-utils (official repo)"
                    if $SUDO pacman -Sy --noconfirm --needed nvidia nvidia-utils; then NVIDIA_INSTALLED=1; fi
                    ;;
            esac
            if [ "$NVIDIA_INSTALLED" = 1 ]; then
                if command -v mokutil >/dev/null 2>&1 &&
                   mokutil --sb-state 2>/dev/null | grep -qi enabled; then
                    status "Secure Boot ON: distro-signed packages (Ubuntu) load as-is; locally built modules (Fedora akmods) may prompt a MokManager key enrollment at next boot"
                fi
                status "NVIDIA driver installed — REBOOT REQUIRED, then run: blazar engine update"
                status "  (auto-picks the newest CUDA build this driver supports; CUDA runtimes are bundled — no toolkit install)"
            else
                status "WARN: NVIDIA driver not auto-installed on this distro — manual lanes:"
                status "  Ubuntu/Debian: sudo ubuntu-drivers autoinstall   (needs the 'universe' repo: sudo add-apt-repository universe)"
                status "  Fedora/RHEL:  sudo dnf install https://mirrors.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora 2>/dev/null || echo VERSION).noarch.rpm && sudo dnf install akmod-nvidia"
                status "  openSUSE:     add the NVIDIA repo (zypper ar -f https://download.nvidia.com/opensuse/leap nvidia) then zypper install nvidia-driver-G06"
                status "  any distro:   https://www.nvidia.com/drivers — after install + reboot: blazar engine update"
            fi
        elif ! timeout 10 nvidia-smi -L >/dev/null 2>&1; then
            status "NVIDIA driver installed but not communicating (module unloaded?) — a REBOOT usually brings it up; then: blazar engine update"
        else
            status "GPU preflight: NVIDIA driver present — CUDA engine lane eligible (blazar engine update picks the newest driver-compatible CUDA build)"
        fi
    fi

    if [ "$HW_AMD" = 1 ] || [ "$HW_INTEL" = 1 ]; then
        ICD_FOUND=0
        for d in /usr/share/vulkan/icd.d /etc/vulkan/icd.d; do
            if [ -d "$d" ] && ls "$d"/*.json >/dev/null 2>&1; then ICD_FOUND=1; fi
        done
        if [ "$ICD_FOUND" = 0 ]; then
            status "AMD/Intel GPU detected but no Vulkan ICD found — the Vulkan engine lane would fall back to CPU"
            ICD_PKGS=
            ICD_INSTALLED=0
            case "$PM" in
                apt-get | dnf)
                    ICD_PKGS=mesa-vulkan-drivers
                    status "  running: $PM install -y $ICD_PKGS"
                    if [ "$PM" = apt-get ]; then
                        if $SUDO apt-get update && $SUDO apt-get install -y "$ICD_PKGS"; then ICD_INSTALLED=1; fi
                    else
                        if $SUDO dnf install -y "$ICD_PKGS"; then ICD_INSTALLED=1; fi
                    fi
                    ;;
                pacman)
                    [ "$HW_AMD" = 1 ] && ICD_PKGS="vulkan-radeon"
                    [ "$HW_INTEL" = 1 ] && ICD_PKGS="${ICD_PKGS:+$ICD_PKGS }vulkan-intel"
                    if [ -n "$ICD_PKGS" ]; then
                        status "  running: pacman -S $ICD_PKGS"
                        if $SUDO pacman -Sy --noconfirm --needed $ICD_PKGS; then ICD_INSTALLED=1; fi
                    fi
                    ;;
                zypper)
                    ICD_PKGS=Mesa-vulkan-drivers
                    status "  running: zypper install $ICD_PKGS"
                    if $SUDO zypper --non-interactive install "$ICD_PKGS"; then ICD_INSTALLED=1; fi
                    ;;
            esac
            if [ "$ICD_INSTALLED" = 1 ]; then
                status "Vulkan ICDs installed ($ICD_PKGS) — GPU lane ready, no reboot normally needed"
            else
                status "WARN: Vulkan ICDs not installed — manual: install mesa-vulkan-drivers (apt/dnf), vulkan-radeon/vulkan-intel (pacman), Mesa-vulkan-drivers (zypper)"
            fi
        fi
    fi
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
# There is no user-local fallback — a second blazar in ~/.local/bin is
# precisely how stale-binary daemon races happen.
# BLAZAR_SUDO is a test/mirror knob: a pass-through wrapper, or empty to
# run everything unprivileged (unset-only default — sudo).
SUDO="${BLAZAR_SUDO-sudo}"
[ "$(id -u)" -eq 0 ] && SUDO=

# F151: under `sudo sh install.sh`, env_reset makes HOME=/root while the
# data-owning user is SUDO_USER — user-keyed paths (pidfile stop, config
# poll, legacy user unit, launch agents, stale ~/.local copies) must
# resolve through the INVOKING user's home or the unit crash-loops on a
# port the real user's daemon already owns.
# USER_HOME/BUILD_USER are derived in the early as_user block near the
# top — nothing user-scoped is resolved from $HOME past this point.

# Privilege is enforced where it's needed: the privileged install command
# itself fails with "cannot create /usr/local/bin (need sudo?)" when root
# is genuinely unavailable — no fragile tty/sudo probing up front.

# --uninstall: remove what install.sh put here (binary + units).
# NEVER touches models or config — those are user data.
if [ "$UNINSTALL" = 1 ]; then
    if command -v blazar >/dev/null 2>&1; then blazar stop >/dev/null 2>&1 || true; fi
    for BIN in "${BLAZAR_SYSTEM_BIN_DIR:-/usr/local/bin}/blazar" "$USER_HOME/.local/bin/blazar"; do
        if [ -e "$BIN" ]; then
            ([ -w "$(dirname "$BIN")" ] && rm -f "$BIN") || $SUDO rm -f "$BIN"
            status "removed $BIN"
        fi
    done
    if [ -f /etc/systemd/system/blazar.service ]; then
        $SUDO systemctl disable --now blazar 2>/dev/null || true
        $SUDO rm -f /etc/systemd/system/blazar.service && status "removed system unit"
        $SUDO systemctl daemon-reload 2>/dev/null || true
    fi
    if [ -f "$USER_HOME/.config/systemd/user/blazar.service" ]; then
        systemctl --user disable --now blazar 2>/dev/null || true
        rm -f "$USER_HOME/.config/systemd/user/blazar.service" && status "removed legacy user unit"
        systemctl --user daemon-reload 2>/dev/null || true
    fi
    if [ "$(uname -s)" = Darwin ]; then
        launchctl bootout "gui/$(id -u)/dev.blazar" 2>/dev/null ||
            launchctl unload "$USER_HOME/Library/LaunchAgents/dev.blazar.plist" 2>/dev/null || true
        if [ -f "$USER_HOME/Library/LaunchAgents/dev.blazar.plist" ]; then
            rm -f "$USER_HOME/Library/LaunchAgents/dev.blazar.plist" && status "removed launch agent"
        fi
        if [ -f /Library/LaunchDaemons/dev.blazar.plist ]; then
            $SUDO launchctl bootout system/dev.blazar 2>/dev/null ||
                $SUDO launchctl unload /Library/LaunchDaemons/dev.blazar.plist 2>/dev/null || true
            $SUDO rm -f /Library/LaunchDaemons/dev.blazar.plist && status "removed launch daemon"
        fi
    fi
    status "uninstalled. models/config kept at ~/.local/share/blazar and ~/.config/blazar (delete manually if desired)"
    exit 0
fi

find_checkout() {
    if [ -n "${BLAZAR_CHECKOUT:-}" ]; then
        [ -f "$BLAZAR_CHECKOUT/crates/blazar-cli/Cargo.toml" ] && { echo "$BLAZAR_CHECKOUT"; return 0; }
        return 1
    fi
    d=$(cd "$(dirname "$0")" && pwd -P)
    while [ "$d" != "/" ]; do
        [ -f "$d/crates/blazar-cli/Cargo.toml" ] && { echo "$d"; return 0; }
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
    as_user sh -c 'command -v cargo >/dev/null 2>&1' || return 1
    CK=$(find_checkout) || return 1
    status "building from source: cargo build --release -p blazar-cli (in ${CK})"
    # Always build in the invoking user's environment: a root-run build
    # leaves root-owned artifacts in the user's target/ and breaks every
    # later user build.
    if ! as_user sh -c "cd '$CK' && cargo build --release -p blazar-cli"; then
        echo "ERROR: source build failed (cargo output above)" >&2
        return 1
    fi
    [ -f "$CK/target/release/blazar" ] || { echo "ERROR: build produced no target/release/blazar" >&2; return 1; }
    echo "$CK/target/release/blazar"
}

# Zero-argument auto mode: a checkout present -> compile FRESH (never a
# stale target/release), then install system-wide. Checkout-less runs
# (curl | sh) fall through to the release channel. A missing toolchain
# is bootstrapped first (announce-then-act); every failure falls back to
# the release channel LOUD — never silently.
if [ -z "$FROM_BIN" ] && [ "$FORCE_BUILD" = 0 ] &&
   [ -z "${BLAZAR_REPO:-}" ] && [ -z "${BLAZAR_INSTALL_BASE_URL:-}" ]; then
    if find_checkout >/dev/null 2>&1; then
        if ! ensure_toolchain; then
            status "auto: toolchain unavailable (bootstrap failed or BLAZAR_AUTO_BOOTSTRAP=0) — falling back to the release channel"
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
        error "--build: no compile toolchain and bootstrap failed — install Rust (https://rustup.rs) + a C compiler, or set BLAZAR_BOOTSTRAP=<path to scripts/bootstrap.sh>"
    FROM_BIN=$(build_from_checkout) ||
        error "--build: source build failed (cargo output above)"
    status "--build: source path forced (no release channel contact)"
fi

# Repo guard is release-channel only: --from bootstrap and mirror/test
# base URLs never touch the GitHub release API. BLAZAR_REPO unset:
# derive owner/name from the checkout's git origin, else fall back to
# the published repo (fresh curl|sh needs zero exports).
if [ -z "${BLAZAR_INSTALL_BASE_URL:-}" ] && [ -z "${FROM_BIN:-}" ] && [ -z "$REPO" ]; then
    REPO=$(derive_repo) || REPO=
    if [ -n "$REPO" ]; then
        status "BLAZAR_REPO unset — derived from git origin: ${REPO}"
    else
        REPO=santanu20/blazar
        status "BLAZAR_REPO unset — using the published repo ${REPO}"
    fi
    API_BASE="${BLAZAR_INSTALL_BASE_URL:-https://api.github.com/repos/${REPO}}"
fi

# macOS service via launchd (called from install_system when systemctl
# is absent but launchctl exists). Non-root: LaunchAgent at login. Root:
# LaunchDaemon at boot with UserName= so the daemon still runs
# unprivileged — mirroring the systemd unit's User=. KeepAlive is the
# launchd spelling of Restart=always.
install_launchd() {
    LABEL=dev.blazar
    if [ "$(id -u)" -eq 0 ]; then
        PLIST_DIR=/Library/LaunchDaemons
        USER_KEY="
    <key>UserName</key>
    <string>${SVC_USER}</string>"
    else
        PLIST_DIR="$USER_HOME/Library/LaunchAgents"
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
        <string>${BIN_DIR}/blazar</string>
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
    poll_healthz "logs: log show --predicate 'process == \"blazar\"' --last 5m"
}

# Poll the configured host (quoted or bare TOML), not a hardcoded
# loopback — a config bound to a specific interface answers there.
# Shared by the systemd and launchd install paths.
poll_healthz() {
    # poll_healthz <log-hint>
    HOST=$(sed -n 's/^host[[:space:]]*=[[:space:]]*//p' "$USER_HOME/.config/blazar/config.toml" 2>/dev/null | head -1 | tr -d '"')
    PORT=$(sed -n 's/^port[[:space:]]*=[[:space:]]*\([0-9]*\).*/\1/p' "$USER_HOME/.config/blazar/config.toml" 2>/dev/null | head -1)
    HOST=${HOST:-127.0.0.1}
    # Blazar's own default port — NEVER 11434 (that is ollama's; a
    # fallback poll there would read a FOREIGN server's health).
    PORT=${PORT:-11435}
    i=0
    while ! curl -s --max-time 2 "http://${HOST}:${PORT}/healthz" 2>/dev/null | grep -q ok; do
        i=$((i + 1)); [ "$i" -gt 30 ] && break
        sleep 1
    done
    if curl -s --max-time 2 "http://${HOST}:${PORT}/healthz" 2>/dev/null | grep -q ok; then
        status "service active; blazar healthy on :${PORT} ($1)"
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
    BIN_DIR="${BLAZAR_SYSTEM_BIN_DIR:-/usr/local/bin}"
    $SUDO mkdir -p "$BIN_DIR" || error "cannot create ${BIN_DIR} (need sudo?)"
    # Replace a possibly-running binary without ETXTBSY: temp file + rename
    # (the running process keeps its inode; new execs get the new binary).
    # Unique temp name per run (mktemp): a leftover temp from an earlier
    # failed install (pid-recycled $$.suffix, or a full-disk partial copy)
    # must never be renamed into place as the installed binary.
    # Root-owned like ollama when we have root; plain install otherwise
    # (mirrors/tests run through a pass-through "sudo").
    NEW_BIN=$($SUDO mktemp "$BIN_DIR/blazar.new.XXXXXX") ||
        error "cannot create temp file in ${BIN_DIR} (disk full?)"
    $SUDO install -o0 -g0 -m0755 "$1" "$NEW_BIN" 2>/dev/null ||
    $SUDO install -m0755 "$1" "$NEW_BIN" ||
        { $SUDO rm -f "$NEW_BIN"; error "install to ${BIN_DIR} failed"; }
    $SUDO mv -f "$NEW_BIN" "$BIN_DIR/blazar"
    # A user-started daemon owns the port; the unit would crash-loop.
    PIDFILE="$USER_HOME/.local/share/blazar/run/blazar.pid"
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
    SVC_USER="${BLAZAR_SERVICE_USER:-${SUDO_USER:-$(id -un)}}"
    SVC_GROUP="${BLAZAR_SERVICE_GROUP:-$(id -gn "$SVC_USER")}"
    UNIT_PATH="${BLAZAR_UNIT_PATH:-/etc/systemd/system/blazar.service}"
    SYSTEMCTL="${BLAZAR_SYSTEMCTL:-systemctl}"
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
        # Soft memory ceiling for the daemon cgroup: protects the rest of
        # the box from runaway children without OOM-killing legit model
        # loads (mmap'd weights are reclaimable). Empty knob = no line.
        MEMORY_HIGH="${BLAZAR_UNIT_MEMORY_HIGH-85%}"
        MH_LINE=
        [ -n "$MEMORY_HIGH" ] && MH_LINE="MemoryHigh=$MEMORY_HIGH"
        SVC_HOME=$(getent passwd "$SVC_USER" | cut -d: -f6)
        SVC_DATA_DIR=${SVC_HOME}/.local/share/blazar
        $SUDO mkdir -p "$SVC_DATA_DIR"
        $SUDO mkdir -p "$(dirname "$UNIT_PATH")"
        UNIT=$(cat <<EOF
[Unit]
Description=Blazar daemon (llama.cpp orchestration)
After=network-online.target
Wants=network-online.target

[Service]
ExecStart=${BIN_DIR}/blazar serve
User=${SVC_USER}
Group=${SVC_GROUP}
# Engine children inherit the daemon cwd; upstream binaries that walk
# relative paths must never start at the filesystem root (symlink
# loops under /run).
WorkingDirectory=${SVC_DATA_DIR}
${SG_LINE}
${MH_LINE}
Restart=always
RestartSec=3
# blazar serve exits 3 on hard singleton conflicts — another server
# owns the port, or a live peer owns the daemon lock (e.g. a session
# blazar serve while the unit is active). Neither is transient, so
# do not restart-loop them.
# NOTE: no backticks and no literal dollar-parenthesis text anywhere in
# this heredoc body — dash executes both while parsing the enclosing
# command substitution at READ time (the installer hung for 40+ minutes
# running "blazar serve" from a comment).
RestartPreventExitStatus=3

[Install]
WantedBy=multi-user.target
EOF
)
        printf '%s\n' "$UNIT" | $SUDO tee "$UNIT_PATH" >/dev/null || error "writing $UNIT_PATH failed"
        $SUDO "$SYSTEMCTL" daemon-reload || error "systemctl daemon-reload failed"
        # Upgrade-in-place: restart an already-active unit (like ollama).
        # Fresh install: enable WITHOUT --now — starting the unit before
        # the engine bootstrap below crash-loops serve() ("no engine
        # installed", exit 1 + Restart=always) for the whole engine
        # download (measured: 86 restarts during a 28 MiB asset fetch).
        # The start is deferred to just after the engine exists.
        if $SUDO "$SYSTEMCTL" is-active --quiet blazar 2>/dev/null; then
            $SUDO "$SYSTEMCTL" restart blazar || error "restarting blazar.service failed"
            poll_healthz "logs: journalctl -u blazar"
        else
            $SUDO "$SYSTEMCTL" enable blazar || error "enabling blazar.service failed"
            DEFERRED_START=1
        fi
        SERVICE_DESC=" + systemd unit ${UNIT_PATH}"
    elif [ "$(uname -s)" = Darwin ] && command -v launchctl >/dev/null 2>&1; then
        install_launchd
        SERVICE_DESC=" + launchd service ${PLIST_PATH}"
    else
        status "no service manager found — binary installed at ${BIN_DIR}/blazar; start it manually: blazar serve"
    fi
    VER=$("$BIN_DIR/blazar" --version 2>/dev/null || echo "(version check failed)")
    status "Installed blazar ${VER} system-wide (${BIN_DIR}/blazar${SERVICE_DESC:-})"
    # The stale-copy race: a leftover user-path copy gets resurrected by
    # services with their own PATH. Remove it as part of every install.
    # -ef (same inode, symlinks followed) works where `readlink -f` does
    # not (old macOS): skip removal only when the copy IS the system file.
    # shellcheck disable=SC3013 # XSI extension; dash/busybox/bash/BSD sh
    # all implement it, and the degraded path removes a copy policy wants
    # gone anyway.
    if [ -e "$USER_HOME/.local/bin/blazar" ] &&
       ! [ "$USER_HOME/.local/bin/blazar" -ef "$BIN_DIR/blazar" ]; then
        rm -f "$USER_HOME/.local/bin/blazar" && status "removed stale user-path copy ~/.local/bin/blazar"
    fi
    # One-click readiness: persist config migrations (legacy api_keys ->
    # [[keys]] etc.) so the first `blazar` invocation never FAILs on an
    # old config. Best-effort: a missing config or an offline box must
    # not fail the install.
# Run a user-store CLI (migrate, engine update, pull) as the DATA-OWNING
# user — see the as_user block above for why a root-run installer must
# never populate /root's store (engine-less crash-looping daemon).
    if as_user "$BIN_DIR/blazar" migrate >/dev/null 2>&1; then
        status "config migrated/verified (canonical form)"
    else
        status "config migration skipped (no config or parse issue — run: blazar migrate)"
    fi
    # One-click readiness: a blazar install without a llama.cpp engine
    # has ZERO inference capability. Bootstrap the engine now (the user
    # ran the installer — the download is sanctioned, never hidden).
    # Opt out: BLAZAR_INSTALL_ENGINE=0. Optional first model:
    # BLAZAR_INSTALL_MODEL=<repo> (pull lane, opt-in — model choice is
    # the user's call, not the installer's).
    if [ "${BLAZAR_INSTALL_ENGINE:-1}" != 0 ] &&
       ! as_user "$BIN_DIR/blazar" engine list --json 2>/dev/null | grep -q '"active": *true'; then
        status "bootstrapping llama.cpp engine (blazar engine update — largest download of this install)..."
        if as_user "$BIN_DIR/blazar" engine update --no-gate; then
            status "engine bootstrap complete"
        else
            status "WARN: engine bootstrap failed (offline?) — inference NOT ready. Run: blazar engine update"
        fi
    else
        status "engine already active (or bootstrap disabled) — skipping engine download"
    fi
    # The bootstrap lane is llamacpp-only (zero-touch default: any GGUF,
    # fastest cold start). The other engines are one command away — say
    # so, every install, so the choice is discoverable without docs.
    status "other engines, one command each:"
    status "  blazar engine install --kind sglang     # SGLang: safetensors lane, best quality + batching (Linux + NVIDIA, ~6 GiB)"
    status "  blazar engine install --kind mistralrs  # mistral.rs: GGUF + safetensors (~0.8 GiB)"
    status "  blazar engine install --kind sdcpp      # sd.cpp: diffusion + video checkpoints — Qwen-Image/FLUX/Z-Image/Chroma/SDXL/SD1.5/Wan 2.1 T2V (any GPU via Vulkan, ~0.04-0.3 GiB)"
    status "  blazar engine install --kind whisper    # whisper: audio transcription + translation (CPU, ~10 MiB; ggml models)"
    status "  blazar engine list                      # what is installed; blazar engine use <tag> switches the serving engine"
    # Fresh-install start, deferred until the engine exists (see the
    # enable block above). Started even when bootstrap failed: a running
    # crash-looping unit still answers `systemctl status` diagnostics
    # better than a silent inactive one.
    if [ "${DEFERRED_START:-0}" = 1 ]; then
        $SUDO "$SYSTEMCTL" start blazar || error "starting blazar.service failed"
        poll_healthz "logs: journalctl -u blazar"
    fi
    if [ -n "${BLAZAR_INSTALL_MODEL:-}" ]; then
        status "pulling first model: ${BLAZAR_INSTALL_MODEL}..."
        if as_user "$BIN_DIR/blazar" pull "${BLAZAR_INSTALL_MODEL}"; then
            status "model ready: ${BLAZAR_INSTALL_MODEL}"
        else
            status "WARN: model pull failed — run: blazar pull ${BLAZAR_INSTALL_MODEL}"
        fi
    fi
    status "system ready — next steps:"
    if [ -n "${BLAZAR_INSTALL_MODEL:-}" ]; then
        status "  blazar run ${BLAZAR_INSTALL_MODEL}   # chat REPL (model pulled above)"
    else
        status "  blazar pull <model>    # e.g. blazar pull Qwen3-0.6B (find one: blazar search qwen3)"
    fi
    status "  blazar doctor          # health check with per-row hints"
    status "engines: llamacpp serves by default; the menu above installs SGLang or mistral.rs"
    status "All inference is upstream llama.cpp, mistral.rs and SGLang — the engine authors did the hard parts."
}


# GPU preflight runs before every install_system lane (bootstrap --from,
# auto-build, release download): the engine bootstrap inside picks its
# asset by driver presence, so the driver must be provisioned first.
gpu_preflight

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
status "Looking for release ${BLAZAR_VERSION:-latest} for ${STATUS_OS_ARCH}..."

# Latest-or-pinned release metadata from the GitHub API.
if [ -n "${BLAZAR_VERSION:-}" ]; then
    RELEASE_PATH="releases/tags/${BLAZAR_VERSION}"
else
    RELEASE_PATH="releases/latest"
fi
RELEASE_JSON=$(fetch "${API_BASE}/${RELEASE_PATH}") ||
    error "failed to look up release ${BLAZAR_VERSION:-latest} in ${API_BASE} (BLAZAR_REPO set? network up?)"

TAG=$(printf '%s' "$RELEASE_JSON" | sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p' | head -1)
[ -n "$TAG" ] || error "could not parse tag_name from the release API response"

# Asset lines: pick the one matching this OS/arch (release-workflow naming
# contract: blazar-<tag>-<rust-triple>.tar.gz). Each sibling asset line
# carries its own digest, so a name/digest mix-up is impossible.
# Quoted variable = literal case match; a raw expansion in a case pattern
# would act as a glob (a hostile tag_name like "*" must not match).
WANTED="blazar-${TAG}-${RUST_ARCH}-${ASSET_TRIPLE_SUFFIX}.tar.gz"
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
    done > "${TMPDIR:-/tmp}/blazar-asset.$$"
read -r ASSET EXPECT < "${TMPDIR:-/tmp}/blazar-asset.$$" || true
rm -f "${TMPDIR:-/tmp}/blazar-asset.$$"
[ -n "$ASSET" ] || error "release ${TAG} has no asset matching blazar-${TAG}-${RUST_ARCH}-${ASSET_TRIPLE_SUFFIX}.tar.gz (available: $(printf '%s' "$RELEASE_JSON" | tr ',' '\n' | grep -o '"name": *"[^"]*"' | cut -d'"' -f4 | tr '\n' ' '))"
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
[ -f "$EXDIR/blazar" ] || error "tarball did not contain a 'blazar' binary at its root"

install_system "$EXDIR/blazar"

}

main "$@"
