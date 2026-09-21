#!/bin/sh
# shellcheck disable=SC2086 # $SUDO is INTENTIONALLY unquoted throughout:
# empty means "run unprivileged" (zero words); quoted it would become
# argv[0]="". Same privilege convention as install.sh.
# Blazar clean uninstall — removes EVERYTHING install.sh put here plus
# per-user state (config, store, engines, caches). Models are the one
# exception: they are expensive re-downloads, so removal is ASKED, with
# a safe keep-by-default. For the quick binary+units-only path use:
# scripts/install.sh --uninstall
#
# Removed, in order:
#   1. services + processes   systemd system/user units, launchd plists,
#                             the running daemon (exact pidfile pid —
#                             NEVER pkill -f, which self-matches)
#   2. system artifacts       binary (BLAZAR_SYSTEM_BIN_DIR), unit file,
#                             unit drop-ins, stale ~/.local/bin copy
#   3. user state             ~/.local/share/blazar minus models;
#                             ~/.config/blazar is KEPT by default (it
#                             holds your port pin, keys and settings —
#                             losing it silently re-creates defaults on
#                             the next install and can collide with
#                             ollama on 11434). --purge removes it too.
#   4. models (ASKED)         GGUF models + whisper ggml models — prompt
#                             shows sizes first; default keep
#
# Options:
#   --dry-run         print every action, remove nothing
#   --remove-models   non-interactive full nuke (models deleted!)
#   --keep-models     non-interactive, models explicitly kept
#   --purge           also remove ~/.config/blazar (config + token)
#   --yes             skip prompts using SAFE defaults (models KEPT);
#                     never implies --remove-models — data deletion is
#                     opt-in only via the explicit flag
#   --help            this text
#
# Env overrides (testing/mirrors; defaults match install.sh):
#   BLAZAR_SYSTEM_BIN_DIR   binary location      (default /usr/local/bin)
#   BLAZAR_UNIT_PATH        systemd unit path    (default
#                             /etc/systemd/system/blazar.service)
#   BLAZAR_SYSTEMCTL        systemctl binary     (default systemctl)
#   BLAZAR_SUDO             sudo wrapper ("" = unprivileged; tests)
#
# NEVER touches: ~/.cache/blazar-validate-* (validation harness
# sandboxes), anything outside the paths above.

set -u

status() { printf '>>> %s\n' "$*"; }
error() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

usage() { sed -n '2,37p' "$0"; exit 0; }

DRY=0
MODELS=ask
PURGE=0
YES=0
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY=1 ;;
        --remove-models) MODELS=remove ;;
        --keep-models) MODELS=keep ;;
        --purge) PURGE=1 ;;
        --yes) YES=1 ;;
        --help | -h) usage ;;
        *) error "unknown option: $1 (supported: --dry-run, --remove-models, --keep-models, --yes, --purge, --help)" ;;
    esac
    shift
done
# --yes = skip prompts with SAFE defaults. It can never combine with an
# explicit model deletion (silent last-flag-wins here once destroyed
# 19 GiB of user models on 2026-09-13).
[ "$YES" -eq 1 ] && {
    [ "$MODELS" = remove ] &&
        error "--yes cannot combine with --remove-models (safe default: models kept)"
    MODELS=keep
}

# ollama-style privilege: root runs plain, everyone else needs sudo.
SUDO="${BLAZAR_SUDO-sudo}"
[ "$(id -u)" -eq 0 ] && SUDO=

# Privilege preflight BEFORE any mutation: privileged actions (disable
# unit, remove binary/unit) must be actionable or we refuse to start.
# The split-brain alternative (system state kept, user state deleted
# under a crash-looping restart) is far worse than a clean refusal.
# Allowed: root; a tty (sudo can prompt); a non-default BLAZAR_SUDO
# wrapper (caller owns its auth). Refused: default sudo, no tty, no
# cached credentials.
if [ "$(id -u)" -ne 0 ] && [ "$SUDO" = sudo ] && [ "$DRY" != 1 ] &&
    [ ! -t 0 ] && ! sudo -n true 2>/dev/null; then
    error "sudo cannot run non-interactively (no tty / no cached credentials) — refusing to start a partial uninstall. Re-run from a terminal for the password prompt, or pipe credentials to 'sudo -S', or set BLAZAR_SUDO for CI."
fi

# run <desc> <cmd...>: every mutation goes through here so --dry-run
# covers everything and the log doubles as an action transcript.
FAILURES=0
run() {
    _desc=$1
    shift
    if [ "$DRY" = 1 ]; then
        printf '  [dry-run] %s\n' "$_desc"
        return 0
    fi
    if "$@" >/dev/null 2>&1; then
        printf '  %s\n' "$_desc"
    else
        # F152: never print an action that did not happen — the
        # transcript must not claim success over a failed mutation.
        _rc=$?
        FAILURES=$((FAILURES + 1))
        printf '  [FAIL rc=%s] %s\n' "$_rc" "$_desc"
    fi
}

BIN_DIR="${BLAZAR_SYSTEM_BIN_DIR:-/usr/local/bin}"
UNIT_PATH="${BLAZAR_UNIT_PATH:-/etc/systemd/system/blazar.service}"
SYSTEMCTL="${BLAZAR_SYSTEMCTL:-systemctl}"
# Run-as-root support (parity with install.sh): under `sudo sh
# uninstall.sh` $HOME is /root — user-keyed state (config, data, models,
# user unit) belongs to the INVOKING user, resolved via SUDO_USER.
USER_HOME="$HOME"
if [ "$(id -u)" -eq 0 ] && [ "${SUDO_USER:-}" != "" ] && [ "$SUDO_USER" != "root" ]; then
    _hm=$(getent passwd "$SUDO_USER" 2>/dev/null | cut -d: -f6)
    [ -z "$_hm" ] && [ "$(uname -s)" = Darwin ] &&
        _hm=$(dscl . -read "/Users/$SUDO_USER" NFSHomeDirectory 2>/dev/null | awk '{print $NF}')
    [ -n "$_hm" ] && USER_HOME="$_hm"
fi
CONFIG_DIR="$USER_HOME/.config/blazar"
DATA_DIR="$USER_HOME/.local/share/blazar"
MODELS_DIR="$DATA_DIR/models"
WHISPER_MODELS_DIR="$DATA_DIR/whisper/models"
USER_UNIT="$USER_HOME/.config/systemd/user/blazar.service"
UNIT_DROPIN="$UNIT_PATH.d"

# ---------------------------------------------------------------- 1. stop
status "stopping services and daemons"

# Ask the CLI to stop engine children gracefully first (best-effort;
# the pidfile TERM below still applies when the CLI is already gone).
if [ "$DRY" != 1 ] && command -v "$BIN_DIR/blazar" >/dev/null 2>&1; then
    "$BIN_DIR/blazar" stop >/dev/null 2>&1 || true
fi

if command -v "$SYSTEMCTL" >/dev/null 2>&1; then
    if [ -e "$UNIT_PATH" ] || [ "$DRY" = 1 ]; then
        run "systemctl disable --now blazar" $SUDO "$SYSTEMCTL" disable --now blazar
        run "systemctl daemon-reload" $SUDO "$SYSTEMCTL" daemon-reload
    fi
    if [ -e "$USER_UNIT" ]; then
        run "systemctl --user disable --now blazar" systemctl --user disable --now blazar
        run "systemctl --user daemon-reload" systemctl --user daemon-reload
    fi
fi

if [ "$(uname -s)" = Darwin ]; then
    run "launchctl bootout gui/$(id -u)/dev.blazar" \
        launchctl bootout "gui/$(id -u)/dev.blazar"
    run "launchctl bootout system/dev.blazar" \
        $SUDO launchctl bootout system/dev.blazar
fi

# User-started daemon (not service-managed): TERM the exact pidfile pid.
# NEVER pkill/pgrep -f — the pattern self-matches this script's own
# invocation line (documented incident class).
PIDFILE="$DATA_DIR/run/blazar.pid"
if [ -f "$PIDFILE" ]; then
    _pid=$(cat "$PIDFILE" 2>/dev/null || true)
    case "$_pid" in
        '' | *[!0-9]*) ;; # malformed: skip, never guess
        *)
            if kill -0 "$_pid" 2>/dev/null; then
                run "TERM daemon pid $_pid (pidfile)" kill -TERM "$_pid"
                if [ "$DRY" != 1 ]; then
                    _i=0
                    while kill -0 "$_pid" 2>/dev/null && [ "$_i" -lt 10 ]; do
                        _i=$((_i + 1))
                        sleep 1
                    done
                fi
            fi
            ;;
    esac
fi

# Stray sweep (Linux): any process whose EXECUTABLE is the system binary
# itself — pidfile-less manual `blazar serve &`, a deleted-pidfile
# daemon. Exact /proc/<pid>/exe match only (no name patterns to
# self-match); foreign-XDG daemons running OTHER binaries are not ours
# to kill.
if [ "$(uname -s)" = Linux ] && [ -e "$BIN_DIR/blazar" ]; then
    _strays=
    for _p in /proc/[0-9]*/exe; do
        [ "$(readlink "$_p" 2>/dev/null)" = "$BIN_DIR/blazar" ] || continue
        _spid=${_p#/proc/}
        _spid=${_spid%/exe}
        [ "$_spid" = "$$" ] && continue
        _strays="$_strays $_spid"
    done
    for _spid in $_strays; do
        run "TERM stray daemon pid $_spid (exe match)" kill -TERM "$_spid"
    done
    if [ "$DRY" != 1 ] && [ -n "$_strays" ]; then
        _i=0
        while [ "$_i" -lt 10 ]; do
            _alive=0
            for _spid in $_strays; do
                kill -0 "$_spid" 2>/dev/null && _alive=1
            done
            [ "$_alive" = 0 ] && break
            _i=$((_i + 1))
            sleep 1
        done
    fi
fi

# ------------------------------------------------------- 2. system files
status "removing system artifacts"
if [ -e "$BIN_DIR/blazar" ] || [ "$DRY" = 1 ]; then
    run "remove $BIN_DIR/blazar" $SUDO rm -f "$BIN_DIR/blazar"
fi
if [ -e "$USER_HOME/.local/bin/blazar" ]; then
    run "remove stale user-path copy ~/.local/bin/blazar" \
        rm -f "$USER_HOME/.local/bin/blazar"
fi
if [ -e "$UNIT_PATH" ]; then
    run "remove unit $UNIT_PATH" $SUDO rm -f "$UNIT_PATH"
fi
if [ -d "$UNIT_DROPIN" ]; then
    run "remove unit drop-ins $UNIT_DROPIN" $SUDO rm -rf "$UNIT_DROPIN"
fi
if [ -e "$USER_UNIT" ]; then
    run "remove legacy user unit $USER_UNIT" rm -f "$USER_UNIT"
fi
if [ "$(uname -s)" = Darwin ]; then
    [ -e "$USER_HOME/Library/LaunchAgents/dev.blazar.plist" ] &&
        run "remove launch agent plist" rm -f "$USER_HOME/Library/LaunchAgents/dev.blazar.plist"
    [ -e /Library/LaunchDaemons/dev.blazar.plist ] &&
        run "remove launch daemon plist" $SUDO rm -f /Library/LaunchDaemons/dev.blazar.plist
fi

# --------------------------------------------------------- 3. user state
# Config is user settings (port pin, keys, overlays) — kept unless --purge.
if [ "$PURGE" -eq 1 ]; then
    if [ -e "$CONFIG_DIR" ]; then
        run "purge $CONFIG_DIR (config.toml, gh-token.env)" rm -rf "$CONFIG_DIR"
    fi
else
    status "kept $CONFIG_DIR (config.toml, gh-token.env, backups) — re-run with --purge to remove"
fi
status "removing regenerable user state (store + engines + caches)"

# Data dir minus models: remove named subpaths (store, engines, runtime,
# config snapshots, audit log, KV sessions, spec caches — all regenerable
# or re-downloadable) so a declined model removal leaves the directory
# itself intact.
# blazar.db plus its SQLite WAL sidecars (-wal, -shm) that can outlive
# the db file after a daemon shutdown — orphaned checkpoints, not data.
for _sub in blazar.db blazar.db-wal blazar.db-shm engines run snapshots log sessions speccache; do
    [ -e "$DATA_DIR/$_sub" ] &&
        run "remove $DATA_DIR/$_sub" rm -rf "$DATA_DIR/$_sub"
done
# whisper binary lane (server binaries, registry) — re-downloadable.
[ -e "$DATA_DIR/whisper" ] && [ ! -e "$WHISPER_MODELS_DIR" ] &&
    run "remove $DATA_DIR/whisper" rm -rf "$DATA_DIR/whisper"

# ----------------------------------------------------------- 4. models
_hum() { du -sh "$1" 2>/dev/null | cut -f1 || echo '?'; }
if [ "$MODELS" = ask ]; then
    if [ ! -d "$MODELS_DIR" ] && [ ! -d "$WHISPER_MODELS_DIR" ]; then
        status "no models present — nothing to ask"
        MODELS=none
    elif [ "$DRY" = 1 ]; then
        status "[dry-run] would prompt for model removal"
        MODELS=remove # describe the full path in a dry run
    else
        printf '>>> models found:\n'
        [ -d "$MODELS_DIR" ] && printf '    %s  %s\n' "$(_hum "$MODELS_DIR")" "$MODELS_DIR"
        [ -d "$WHISPER_MODELS_DIR" ] && printf '    %s  %s\n' "$(_hum "$WHISPER_MODELS_DIR")" "$WHISPER_MODELS_DIR"
        printf '    remove them? re-downloading later costs the full pull again [y/N] '
        read -r _ans || _ans=
        case "$_ans" in
            y | Y | yes | YES) MODELS=remove ;;
            *) MODELS=keep ;;
        esac
    fi
fi
case "$MODELS" in
    remove)
        [ -d "$MODELS_DIR" ] && run "remove $MODELS_DIR" rm -rf "$MODELS_DIR"
        [ -d "$WHISPER_MODELS_DIR" ] && run "remove $WHISPER_MODELS_DIR" rm -rf "$WHISPER_MODELS_DIR"
        ;;
    keep)
        # Server binaries are engine-class (cheap re-download) even when
        # models are kept: only the ggml models survive a "keep".
        [ -d "$DATA_DIR/whisper/bin" ] &&
            run "remove $DATA_DIR/whisper/bin" rm -rf "$DATA_DIR/whisper/bin"
        status "models kept"
        ;;
    none) ;;
esac

# Leftover empty parents: harmless, remove only when empty (rmdir chain
# clears whisper/ after its models went, then the data dir itself).
if [ "$DRY" != 1 ] && [ -d "$DATA_DIR" ]; then
    rmdir "$DATA_DIR/whisper" 2>/dev/null || true
    rmdir "$DATA_DIR" 2>/dev/null || true
fi

if [ "$FAILURES" -gt 0 ]; then
    status "INCOMPLETE: ${FAILURES} action(s) failed — re-run and investigate the [FAIL] lines above"
    exit 1
fi
status "done. blazar fully removed (validation sandboxes under ~/.cache untouched)"
# NOT `[ ... ] && status` as the last line: a false test there becomes the
# script's exit status (POSIX footgun; --remove-models runs used to exit 1
# after a flawless uninstall).
if [ "$MODELS" = keep ]; then
    status "models remain at $MODELS_DIR — delete manually when ready"
fi
exit 0
