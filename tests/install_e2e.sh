#!/bin/sh
# End-to-end test for scripts/install.sh against a local fake GitHub API.
#
# Proves, without touching GitHub or the real HOME:
#   1. happy path  — asset selected per-arch, digest verified, binary installed and runs
#   2. tamper      — corrupted digest in metadata -> hard failure, nothing installed
#   3. wrong arch  — release without a matching asset -> clear error naming the asset
#   4. one-click   — engine bootstrap: install.sh pulls the llama.cpp engine
#                   from the (faked) engine lane and leaves it ACTIVE — the
#                   box is infer-ready with zero manual steps
#   5. --build     — a failing toolchain bootstrap is LOUD and FATAL (the
#                   bootstrap really ran, with --minimal, nothing installed)
#   6. unit groups — SupplementaryGroups lists only render/video groups that
#                   exist on the box (systemd rejects units naming ghosts)
#   7. armv7 host  — fake uname armv7l maps to the static musleabihf asset
#
# The fake release JSON puts a decoy asset with a WRONG digest first, so a
# parser bug that grabs a sibling asset's digest fails this test.
#
# Requires: python3, curl, sha256sum, target/release/pallama (host build),
# and target/release/stub-llama-server (cargo build --release --features
# test-util --bin stub-llama-server) for case 4.

set -eu

ROOT=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
INSTALL_SH="$ROOT/scripts/install.sh"
BIN="$ROOT/target/release/pallama"

[ -f "$BIN" ] || { echo "FAIL: $BIN not built — run: cargo build --release"; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "FAIL: python3 required"; exit 1; }

# Same target derivation as install.sh (oracle duplication is intentional).
case "$(uname -m)" in
    x86_64) RUST_ARCH=x86_64 ;;
    aarch64) RUST_ARCH=aarch64 ;;
    *) echo "FAIL: unsupported host arch $(uname -m)"; exit 1 ;;
esac
LDD_OUT=$(ldd --version 2>&1 || true)
case "$LDD_OUT" in
    *musl*) LIBC=musl ;;
    *) LIBC=gnu ;;
esac
TARGET="${RUST_ARCH}-unknown-linux-${LIBC}"
TAG=v0.1.0
ASSET="pallama-${TAG}-${TARGET}.tar.gz"
DECOY="pallama-${TAG}-aarch64-unknown-linux-gnu.tar.gz"
[ "$RUST_ARCH" = aarch64 ] && DECOY="pallama-${TAG}-x86_64-unknown-linux-gnu.tar.gz"

TMP=$(mktemp -d)
SRV="$TMP/srv"
mkdir -p "$SRV" "$TMP/home"

# Fake privileged environment: "sudo" executes plainly, "systemctl" says
# the unit is inactive (so the enable path runs) and accepts everything.
cat > "$TMP/fakesudo" <<'EOF'
#!/bin/sh
exec "$@"
EOF
chmod +x "$TMP/fakesudo"
cat > "$TMP/fakesystemctl" <<EOF
#!/bin/sh
# Args log: lets tests assert call ORDER (enable-before-start, no --now).
printf '%s\n' "\$*" >> "$TMP/systemctl.log"
case "\$1" in
    is-active) exit 3 ;;
    *) exit 0 ;;
esac
EOF
chmod +x "$TMP/fakesystemctl"
: > "$TMP/systemctl.log"
UNIT_OUT="$TMP/pallama.service"
SYSTEM_BIN="$TMP/system-bin"
SERVER_PID=

# Package the host binary exactly like the release workflow: flat root.
STAGE="$TMP/stage"
mkdir -p "$STAGE"
cp "$BIN" "$STAGE/pallama"
cp "$ROOT/LICENSE-MIT" "$ROOT/LICENSE-APACHE" "$STAGE/"
tar -czf "$SRV/$ASSET" -C "$STAGE" .
SHA=$(sha256sum "$SRV/$ASSET" | cut -d' ' -f1)
echo "DECOY-BYTES-NOT-A-REAL-ASSET" > "$SRV/$DECOY"

cleanup() {
    [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null || true
    rm -rf "$TMP"
}
trap cleanup EXIT INT TERM
PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')

cat > "$TMP/serve.py" <<EOF
import http.server, os, sys
srv_dir, port = sys.argv[1], int(sys.argv[2])
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/releases/latest':
            meta = os.path.join(srv_dir, 'release.json')
            if os.path.exists(meta):
                self._send(open(meta, 'rb').read(), 'application/json')
            else:
                self.send_error(404)
        elif self.path.startswith('/repos/ggml-org/llama.cpp/releases'):
            # Engine lane (PALLAMA_GH_BASE points GhClient here): the
            # releases list is all latest_b_release needs.
            meta = os.path.join(srv_dir, 'llama-releases.json')
            if os.path.exists(meta):
                self._send(open(meta, 'rb').read(), 'application/json')
            else:
                self.send_error(404)
        elif self.path.startswith('/download/'):
            name = self.path[len('/download/'):]
            if '/' in name or '..' in name:
                self.send_error(404); return
            p = os.path.join(srv_dir, name)
            if os.path.exists(p):
                self._send(open(p, 'rb').read(), 'application/octet-stream')
            else:
                self.send_error(404)
        else:
            self.send_error(404)
    def _send(self, data, ctype):
        self.send_response(200)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def log_message(self, *a):
        pass
http.server.HTTPServer(('127.0.0.1', port), H).serve_forever()
EOF
python3 "$TMP/serve.py" "$SRV" "$PORT" &
SERVER_PID=$!

BASE="http://127.0.0.1:${PORT}"

INSTALL_ENV="HOME=$TMP/home PALLAMA_INSTALL_BASE_URL=$BASE PALLAMA_SUDO=$TMP/fakesudo PALLAMA_SYSTEMCTL=$TMP/fakesystemctl PALLAMA_SYSTEM_BIN_DIR=$SYSTEM_BIN PALLAMA_UNIT_PATH=$UNIT_OUT PALLAMA_INSTALL_ENGINE=0"

# Readiness probe: the decoy asset exists before the server starts; release
# metadata is written per test case below.
i=0
while ! curl --fail --silent --output /dev/null "$BASE/download/$DECOY" 2>/dev/null; do
    i=$((i + 1)); [ "$i" -gt 50 ] && { echo "FAIL: test server never became ready"; kill "$SERVER_PID" 2>/dev/null || true; exit 1; }
    sleep 0.1
done
PASS=0
FAIL=0

ok()   { echo "PASS: $1"; PASS=$((PASS + 1)); }
bad()  { echo "FAIL: $1"; FAIL=$((FAIL + 1)); }

# --- 1. happy path (decoy asset first with a wrong digest) -----------------
printf '{"tag_name":"%s","assets":[{"name":"%s","digest":"sha256:0000","browser_download_url":"%s/download/%s"},{"name":"%s","digest":"sha256:%s","browser_download_url":"%s/download/%s"}]}' \
    "$TAG" "$DECOY" "$BASE" "$DECOY" "$ASSET" "$SHA" "$BASE" "$ASSET" > "$SRV/release.json"

OUT=$(env $INSTALL_ENV sh "$INSTALL_SH" 2>&1) && RC=0 || RC=$?
if [ "$RC" = 0 ] && [ -x "$SYSTEM_BIN/pallama" ] && "$SYSTEM_BIN/pallama" --version >/dev/null 2>&1; then
    ok "system install succeeded, binary runs ($TARGET)"
else
    bad "install failed (rc=$RC)"; echo "$OUT" | sed 's/^/    /'
fi
[ -f "$UNIT_OUT" ] && ok "systemd unit written" || bad "no unit at $UNIT_OUT"
grep -q "Restart=always" "$UNIT_OUT" 2>/dev/null && ok "unit Restart=always" || bad "unit lacks Restart=always"
grep -q "ExecStart=$SYSTEM_BIN/pallama serve" "$UNIT_OUT" 2>/dev/null &&
    ok "unit ExecStart points at the installed binary" || bad "unit ExecStart wrong"
grep -q '^MemoryHigh=85%$' "$UNIT_OUT" 2>/dev/null &&
    ok "unit MemoryHigh default 85%" || bad "unit lacks MemoryHigh=85%"
# PALLAMA_UNIT_MEMORY_HIGH='' must omit the line (operator opt-out)
rm -rf "$UNIT_OUT"
OUT=$(env $INSTALL_ENV PALLAMA_UNIT_MEMORY_HIGH= sh "$INSTALL_SH" 2>&1) && RC=0 || RC=$?
if [ "$RC" = 0 ] && [ -f "$UNIT_OUT" ] && ! grep -q '^MemoryHigh=' "$UNIT_OUT"; then
    ok "empty memory knob omits MemoryHigh line"
else
    bad "empty memory knob did not omit MemoryHigh (rc=$RC)"
fi
echo "$OUT" | grep -q "sha256 verified" && ok "digest verified message" || bad "no 'sha256 verified' in output"
# --- 2. tampered digest ------------------------------------------------------
printf '{"tag_name":"%s","assets":[{"name":"%s","digest":"sha256:%s","browser_download_url":"%s/download/%s"}]}' \
    "$TAG" "$ASSET" "0${SHA#?}" "$BASE" "$ASSET" > "$SRV/release.json"

rm -rf "$SYSTEM_BIN" "$UNIT_OUT"
OUT=$(env $INSTALL_ENV sh "$INSTALL_SH" 2>&1) && RC=0 || RC=$?
if [ "$RC" != 0 ] && [ ! -e "$SYSTEM_BIN/pallama" ]; then
    ok "tampered digest rejected, nothing installed"
else
    bad "tampered digest NOT rejected (rc=$RC)"
fi
echo "$OUT" | grep -q "sha256 mismatch" && ok "mismatch error names the cause" || bad "error does not mention sha256 mismatch"

# --- 3. release without a matching asset ------------------------------------
printf '{"tag_name":"%s","assets":[{"name":"%s","digest":"sha256:0000","browser_download_url":"%s/download/%s"}]}' \
    "$TAG" "$DECOY" "$BASE" "$DECOY" > "$SRV/release.json"

rm -rf "$SYSTEM_BIN" "$UNIT_OUT"
OUT=$(env $INSTALL_ENV sh "$INSTALL_SH" 2>&1) && RC=0 || RC=$?
if [ "$RC" != 0 ] && [ ! -e "$SYSTEM_BIN/pallama" ]; then
    ok "missing-asset release rejected, nothing installed"
else
    bad "missing asset NOT rejected (rc=$RC)"
fi
echo "$OUT" | grep -q "$ASSET" && echo "$OUT" | grep -q "$DECOY" &&
    ok "error names wanted asset and lists available" || bad "error does not name wanted/available assets"

# --- 4. one-click engine bootstrap (fake llama.cpp release lane) --------------
# The install itself comes from the happy-path metadata (case 1 shape);
# the ENGINE comes from a faked llama.cpp releases list served by the same
# python server. The engine tarball carries the stub llama-server, so the
# whole download -> sha verify -> extract -> probe -> activate lane runs
# for real, offline. x86_64 host only (asset suffix is deterministic there).
STUB="$ROOT/target/release/stub-llama-server"
if [ "$(uname -m)" = x86_64 ] && [ -x "$STUB" ]; then
    rm -rf "$SYSTEM_BIN" "$UNIT_OUT" "${TMP:?}/home"
    printf '{"tag_name":"%s","assets":[{"name":"%s","digest":"sha256:0000","browser_download_url":"%s/download/%s"},{"name":"%s","digest":"sha256:%s","browser_download_url":"%s/download/%s"}]}' \
        "$TAG" "$DECOY" "$BASE" "$DECOY" "$ASSET" "$SHA" "$BASE" "$ASSET" > "$SRV/release.json"

    ETAG=b999
    EASSET="llama-${ETAG}-bin-ubuntu-x86_64.tar.gz"
    ESTAGE="$TMP/estage"
    mkdir -p "$ESTAGE/bin"
    cp "$STUB" "$ESTAGE/bin/llama-server"
    tar -czf "$SRV/$EASSET" -C "$ESTAGE" .
    ESA=$(sha256sum "$SRV/$EASSET" | cut -d' ' -f1)
    printf '[{"tag_name":"%s","prerelease":true,"published_at":"2026-09-08T00:00:00Z","assets":[{"name":"%s","digest":"sha256:%s","size":1,"browser_download_url":"%s/download/%s"}]}]' \
        "$ETAG" "$EASSET" "$ESA" "$BASE" "$EASSET" > "$SRV/llama-releases.json"
    # Pin the asset pick (config engine_asset = Exact candidate) and a
    # port that cannot clash with any real daemon.
    mkdir -p "$TMP/home/.config/pallama"
    printf 'engine_asset = "ubuntu-x86_64"\nport = 11499\n' > "$TMP/home/.config/pallama/config.toml"

    ENGINE_ENV="HOME=$TMP/home PALLAMA_INSTALL_BASE_URL=$BASE PALLAMA_GH_BASE=$BASE PALLAMA_SUDO=$TMP/fakesudo PALLAMA_SYSTEMCTL=$TMP/fakesystemctl PALLAMA_SYSTEM_BIN_DIR=$SYSTEM_BIN PALLAMA_UNIT_PATH=$UNIT_OUT"
    OUT=$(env $ENGINE_ENV sh "$INSTALL_SH" 2>&1) && RC=0 || RC=$?
    if [ "$RC" = 0 ] && echo "$OUT" | grep -q "engine bootstrap complete"; then
        ok "one-click: engine bootstrapped during install"
    else
        bad "engine bootstrap did not complete (rc=$RC)"; echo "$OUT" | sed 's/^/    /'
    fi
    LIST=$(env HOME=$TMP/home PALLAMA_GH_BASE= "$SYSTEM_BIN/pallama" engine list 2>/dev/null) || LIST=
    echo "$LIST" | grep -q "$ETAG.*\[active\]" &&
        ok "engine ${ETAG} installed and ACTIVE (persisted in store)" ||
        bad "engine not active after install: $LIST"
    echo "$OUT" | grep -q "system ready" && ok "final readiness status printed" || bad "no readiness status"
else
    echo "SKIP: case 4 needs x86_64 host + $STUB"
fi

# --- 5. --build with a failing bootstrap: loud + fatal ------------------------
# FORCE_BOOTSTRAP runs our fake bootstrap even though this box has cargo;
# the fake records its argv and fails, proving the --build lane (a) really
# calls the bootstrap, (b) invokes it with --minimal, (c) reports the
# failure loudly, (d) refuses to continue (explicit intent, no fallback).
rm -rf "$SYSTEM_BIN" "$UNIT_OUT"
MARKER="$TMP/bootstrap-called"
cat > "$TMP/fake-bootstrap" <<EOF
#!/bin/sh
printf '%s\n' "\$*" > "$MARKER"
exit 1
EOF
chmod +x "$TMP/fake-bootstrap"
BUILD_ENV="HOME=$TMP/home PALLAMA_SUDO=$TMP/fakesudo PALLAMA_SYSTEMCTL=$TMP/fakesystemctl PALLAMA_SYSTEM_BIN_DIR=$SYSTEM_BIN PALLAMA_UNIT_PATH=$UNIT_OUT PALLAMA_INSTALL_ENGINE=0 PALLAMA_CHECKOUT=$ROOT PALLAMA_BOOTSTRAP=$TMP/fake-bootstrap PALLAMA_FORCE_BOOTSTRAP=1"
OUT=$(env $BUILD_ENV sh "$INSTALL_SH" --build 2>&1) && RC=0 || RC=$?
if [ "$RC" != 0 ] && [ "$(cat "$MARKER" 2>/dev/null)" = "--minimal" ]; then
    ok "--build ran the bootstrap (--minimal) and failed hard"
else
    bad "--build bootstrap not run/not fatal (rc=$RC, argv=$(cat "$MARKER" 2>/dev/null))"
fi
echo "$OUT" | grep -q "toolchain bootstrap failed" && ok "bootstrap failure reported loudly" || bad "no bootstrap-failure warning"
[ ! -e "$SYSTEM_BIN/pallama" ] && ok "nothing installed after failed --build" || bad "binary installed despite failed --build"

# --- 6. unit SupplementaryGroups only for groups that exist ------------------
# install.sh filters render/video through /etc/group; the generated unit
# must match the box (oracle duplication of the existence predicate is
# intentional, same as the TARGET derivation at the top of this file).
rm -rf "$SYSTEM_BIN" "$UNIT_OUT" "${TMP:?}/home"
mkdir -p "$TMP/home"
printf '{"tag_name":"%s","assets":[{"name":"%s","digest":"sha256:%s","browser_download_url":"%s/download/%s"}]}' \
    "$TAG" "$ASSET" "$SHA" "$BASE" "$ASSET" > "$SRV/release.json"
OUT=$(env $INSTALL_ENV sh "$INSTALL_SH" 2>&1) && RC=0 || RC=$?
WANT_SG=
for g in render video; do
    if getent group "$g" >/dev/null 2>&1 || grep -q "^${g}:" /etc/group 2>/dev/null; then
        WANT_SG="${WANT_SG}${WANT_SG:+ }$g"
    fi
done
if [ -z "$WANT_SG" ]; then
    if ! grep -q "^SupplementaryGroups=" "$UNIT_OUT" 2>/dev/null; then
        ok "no SupplementaryGroups line when render/video groups are absent"
    else
        bad "unit lists SupplementaryGroups but no render/video group exists"
    fi
else
    grep -q "^SupplementaryGroups=${WANT_SG}$" "$UNIT_OUT" 2>/dev/null &&
        ok "SupplementaryGroups=${WANT_SG} matches existing groups" ||
        bad "unit SupplementaryGroups mismatch (want '${WANT_SG}': $(grep '^SupplementaryGroups' "$UNIT_OUT" 2>/dev/null))"
fi

# --- 7. armv7 host mapping: picks the static musleabihf asset -----------------
# A fake uname (first on PATH) reports armv7l/Linux; the release lane must
# derive pallama-<tag>-armv7-unknown-linux-musleabihf.tar.gz. The staged
# tarball carries the host binary so the install completes; the point is
# the asset-name mapping. x86_64 host only (binary must still run).
if [ "$(uname -m)" = x86_64 ]; then
    rm -rf "$SYSTEM_BIN" "$UNIT_OUT" "${TMP:?}/home"
    mkdir -p "$TMP/home" "$TMP/fakebin"
    cat > "$TMP/fakebin/uname" <<'EOF'
#!/bin/sh
case "$1" in
    -m) echo armv7l ;;
    *) echo Linux ;;
esac
EOF
    chmod +x "$TMP/fakebin/uname"
    ARM_ASSET="pallama-${TAG}-armv7-unknown-linux-musleabihf.tar.gz"
    tar -czf "$SRV/$ARM_ASSET" -C "$STAGE" .
    ARM_SHA=$(sha256sum "$SRV/$ARM_ASSET" | cut -d' ' -f1)
    printf '{"tag_name":"%s","assets":[{"name":"%s","digest":"sha256:%s","browser_download_url":"%s/download/%s"}]}' \
        "$TAG" "$ARM_ASSET" "$ARM_SHA" "$BASE" "$ARM_ASSET" > "$SRV/release.json"
    OUT=$(env PATH="$TMP/fakebin:$PATH" $INSTALL_ENV sh "$INSTALL_SH" 2>&1) && RC=0 || RC=$?
    if [ "$RC" = 0 ] && echo "$OUT" | grep -q "Downloading ${ARM_ASSET}"; then
        ok "armv7l host mapped to ${ARM_ASSET}"
    else
        bad "armv7l mapping failed (rc=$RC)"; echo "$OUT" | sed 's/^/    /'
    fi
    echo "$OUT" | grep -q "linux armv7 (musleabihf)" &&
        ok "status line reports the armv7 + musleabihf pick" ||
        bad "status line missing armv7 musleabihf"
else
    echo "SKIP: case 7 needs x86_64 host (fake-armv7 tarball carries the host binary)"
fi

# --- 8. GPU preflight: wiring, opt-out, driverless-NVIDIA lane ----------------
# (a) full-flow wiring: PALLAMA_AUTO_DRIVER=0 reaches the preflight and
#     skips it; (b) function-level: a driverless NVIDIA PCI census drives
#     the apt driver lane (fake apt-get records its argv) with the REBOOT
#     + engine-update messaging — no real package is touched.
rm -rf "$SYSTEM_BIN" "$UNIT_OUT" "${TMP:?}/home"
mkdir -p "$TMP/home"
printf '{"tag_name":"%s","assets":[{"name":"%s","digest":"sha256:%s","browser_download_url":"%s/download/%s"}]}' \
    "$TAG" "$ASSET" "$SHA" "$BASE" "$ASSET" > "$SRV/release.json"
OUT=$(env $INSTALL_ENV PALLAMA_AUTO_DRIVER=0 sh "$INSTALL_SH" 2>&1) && RC=0 || RC=$?
if [ "$RC" = 0 ] && echo "$OUT" | grep -q "GPU preflight skipped (PALLAMA_AUTO_DRIVER=0)"; then
    ok "full flow: GPU preflight wired and opt-out honored"
else
    bad "GPU preflight opt-out not observed in full flow (rc=$RC)"
fi

# Function-level scenario: extract gpu_preflight verbatim from install.sh
# and run it against a scratch PATH whose lspci reports a driverless
# NVIDIA card and whose apt-get is a recorder. nvidia-smi is absent from
# the scratch PATH — the driverless branch must fire.
GPDIR="$TMP/gpu"
mkdir -p "$GPDIR/scratch" "$GPDIR/fake"
sed -n '/^gpu_preflight() {/,/^}/p' "$INSTALL_SH" > "$GPDIR/fn.sh"
cat > "$GPDIR/fake/lspci" <<'EOF'
#!/bin/sh
case "$3" in ::0300|::0302) printf '0000:01:00.0 0300: 10de:28a0 (rev a1)\n' ;; esac
exit 0
EOF
cat > "$GPDIR/fake/apt-get" <<EOF
#!/bin/sh
echo "apt-get \$*" >> "$GPDIR/fake/apt.log"
exit 0
EOF
chmod +x "$GPDIR/fake/lspci" "$GPDIR/fake/apt-get"
for b in uname ls timeout grep; do
    ln -sf "$(command -v "$b")" "$GPDIR/scratch/$b"
done
ln -sf "$GPDIR/fake/lspci" "$GPDIR/scratch/lspci"
ln -sf "$GPDIR/fake/apt-get" "$GPDIR/scratch/apt-get"
cat > "$GPDIR/run.sh" <<EOF
status() { echo ">>> \$*"; }
SUDO="$TMP/fakesudo"
PATH="$GPDIR/scratch"
. "$GPDIR/fn.sh"
gpu_preflight
EOF
OUT=$(sh "$GPDIR/run.sh" 2>&1)
if echo "$OUT" | grep -q "NVIDIA GPU detected (PCI 10de:) but no NVIDIA driver userspace" &&
   grep -q "install -y nvidia-driver" "$GPDIR/fake/apt.log" 2>/dev/null; then
    ok "driverless NVIDIA: detected, driver lane executed via apt-get"
else
    bad "driverless NVIDIA lane did not fire"; echo "$OUT" | sed 's/^/    /'
fi
echo "$OUT" | grep -q "REBOOT REQUIRED, then run: pallama engine update" &&
    ok "driverless NVIDIA: REBOOT + engine-update chain in message" ||
    bad "missing REBOOT/engine-update guidance"
echo "$OUT" | grep -q "newest CUDA build this driver supports" &&
    ok "driverless NVIDIA: latest-compatible-CUDA messaging present" ||
    bad "missing latest-CUDA messaging"
# Opt-out at function level: nothing installed, skip announced.
rm -f "$GPDIR/fake/apt.log"
cat > "$GPDIR/run.sh" <<EOF
status() { echo ">>> \$*"; }
SUDO="$TMP/fakesudo"
PATH="$GPDIR/scratch"
PALLAMA_AUTO_DRIVER=0
. "$GPDIR/fn.sh"
gpu_preflight
EOF
OUT=$(sh "$GPDIR/run.sh" 2>&1)
if echo "$OUT" | grep -q "GPU preflight skipped" && [ ! -f "$GPDIR/fake/apt.log" ]; then
    ok "opt-out: preflight skipped, no package command run"
else
    bad "opt-out leaked a package install"; echo "$OUT" | sed 's/^/    /'
fi

echo
# --- 9. uninstall.sh flag matrix + install deferred-start ordering ---------

UNINSTALL="$ROOT/scripts/uninstall.sh"
UENV="HOME=$TMP/home PALLAMA_SUDO=$TMP/fakesudo PALLAMA_SYSTEMCTL=$TMP/fakesystemctl PALLAMA_SYSTEM_BIN_DIR=$SYSTEM_BIN PALLAMA_UNIT_PATH=$UNIT_OUT"

stage_installed() {
    rm -rf "${TMP:?}/home" "${SYSTEM_BIN:?}" ; mkdir -p "$TMP/home"
    D="$TMP/home/.local/share/pallama"
    mkdir -p "$D/models" "$D/engines/b1" "$D/whisper/models" "$D/whisper/bin" "$D/run" \
             "$TMP/home/.config/pallama" "$SYSTEM_BIN"
    echo gguf > "$D/models/qwen3-0.6b-q4_0.gguf"
    echo ggml > "$D/whisper/models/ggml-base.bin"
    echo bin  > "$D/whisper/bin/whisper-server"
    echo db   > "$D/pallama.db"
    echo wal  > "$D/pallama.db-wal"
    echo shm  > "$D/pallama.db-shm"
    echo pid  > "$D/run/pallama.pid"
    echo cfg  > "$TMP/home/.config/pallama/config.toml"
    printf '#!/bin/sh\nexit 0\n' > "$SYSTEM_BIN/pallama"
    chmod +x "$SYSTEM_BIN/pallama"
    printf '[Unit]\n' > "$UNIT_OUT"
}

# 9a. --dry-run removes nothing.
stage_installed
env $UENV sh "$UNINSTALL" --dry-run --keep-models >/dev/null 2>&1 </dev/null
[ $? -eq 0 ] && ok "uninstall --dry-run exits 0" || bad "uninstall --dry-run exits 0"
[ -f "$SYSTEM_BIN/pallama" ] && [ -f "$TMP/home/.local/share/pallama/pallama.db" ] \
    && ok "uninstall --dry-run removed nothing" || bad "uninstall --dry-run removed nothing"

# 9b. --keep-models --yes (the once-destructive combo): models KEPT, all
# regenerable state + WAL sidecars gone, config kept.
stage_installed
env $UENV sh "$UNINSTALL" --keep-models --yes >/dev/null 2>&1 </dev/null
D="$TMP/home/.local/share/pallama"
[ -f "$D/models/qwen3-0.6b-q4_0.gguf" ] && ok "uninstall --yes keeps gguf models" || bad "uninstall --yes keeps gguf models"
[ -f "$D/whisper/models/ggml-base.bin" ] && ok "uninstall --yes keeps whisper models" || bad "uninstall --yes keeps whisper models"
[ ! -e "$D/pallama.db" ] && [ ! -e "$D/pallama.db-wal" ] && [ ! -e "$D/pallama.db-shm" ] \
    && ok "uninstall removes db + WAL sidecars" || bad "uninstall removes db + WAL sidecars"
[ ! -d "$D/engines" ] && [ ! -d "$D/whisper/bin" ] && [ ! -e "$SYSTEM_BIN/pallama" ] && [ ! -e "$UNIT_OUT" ] \
    && ok "uninstall removes engines, whisper bins, system binary, unit" || bad "uninstall removes engines, whisper bins, system binary, unit"
[ -f "$TMP/home/.config/pallama/config.toml" ] && ok "uninstall keeps config (no --purge)" || bad "uninstall keeps config (no --purge)"

# 9c. --yes --remove-models conflicts hard BEFORE any mutation.
stage_installed
env $UENV sh "$UNINSTALL" --yes --remove-models >/dev/null 2>&1 </dev/null && RC=0 || RC=$?
[ "$RC" -ne 0 ] && ok "uninstall --yes --remove-models refused" || bad "uninstall --yes --remove-models refused"
[ -f "$TMP/home/.local/share/pallama/models/qwen3-0.6b-q4_0.gguf" ] && [ -f "$SYSTEM_BIN/pallama" ] \
    && ok "refused uninstall mutated nothing" || bad "refused uninstall mutated nothing"

# 9d. --remove-models alone is the explicit full nuke.
stage_installed
env $UENV sh "$UNINSTALL" --remove-models >/dev/null 2>&1 </dev/null && RC=0 || RC=$?
D="$TMP/home/.local/share/pallama"
[ "$RC" = 0 ] && ok "uninstall --remove-models completes" || bad "uninstall --remove-models completes (rc=$RC)"
[ ! -e "$D/models/qwen3-0.6b-q4_0.gguf" ] && ok "uninstall --remove-models removes gguf models" || bad "uninstall --remove-models removes gguf models"
[ ! -e "$D/whisper/models/ggml-base.bin" ] && ok "uninstall --remove-models removes whisper models" || bad "uninstall --remove-models removes whisper models"
stage_installed
env $UENV sh "$UNINSTALL" --remove-models --yes >/dev/null 2>&1 </dev/null && RC=0 || RC=$?
[ "$RC" -ne 0 ] && ok "uninstall --remove-models --yes (either order) refused" || bad "uninstall --remove-models --yes (either order) refused"

# 9e. privilege preflight: default sudo + non-tty + no cached creds refused
# BEFORE mutation. A PATH-shimmed failing sudo makes the credential probe
# deterministic on any host.
stage_installed
mkdir -p "$TMP/fakesbin"
printf '#!/bin/sh\nexit 1\n' > "$TMP/fakesbin/sudo"; chmod +x "$TMP/fakesbin/sudo"
env -u PALLAMA_SUDO HOME="$TMP/home" PATH="$TMP/fakesbin:$PATH" \
    PALLAMA_SYSTEMCTL="$TMP/fakesystemctl" PALLAMA_SYSTEM_BIN_DIR="$SYSTEM_BIN" \
    PALLAMA_UNIT_PATH="$UNIT_OUT" \
    sh "$UNINSTALL" --keep-models >/dev/null 2>&1 </dev/null && RC=0 || RC=$?
[ "$RC" -ne 0 ] && ok "uninstall preflight refuses non-tty default sudo" || bad "uninstall preflight refuses non-tty default sudo"
[ -f "$SYSTEM_BIN/pallama" ] && [ -f "$TMP/home/.local/share/pallama/pallama.db" ] \
    && ok "preflight refusal mutated nothing" || bad "preflight refusal mutated nothing"

# 9f. install deferred-start ordering: fresh install enables WITHOUT --now,
# then explicitly starts after the engine bootstrap (never crash-loops on a
# engine-less unit).
printf '{"tag_name":"%s","assets":[{"name":"%s","digest":"sha256:%s","browser_download_url":"%s/download/%s"}]}' \
    "$TAG" "$ASSET" "$SHA" "$BASE" "$ASSET" > "$SRV/release.json"
: > "$TMP/systemctl.log"
OUT=$(env $INSTALL_ENV sh "$INSTALL_SH" 2>&1) && RC=0 || RC=$?
if grep -qx 'enable pallama' "$TMP/systemctl.log" && ! grep -q 'enable --now pallama' "$TMP/systemctl.log"; then
    ok "fresh install enables without --now"
else
    bad "fresh install enables without --now"
fi
if grep -qx 'start pallama' "$TMP/systemctl.log"; then
    ok "fresh install explicitly starts the unit"
else
    bad "fresh install explicitly starts the unit"
fi
if [ "$(grep -nx 'enable pallama\|start pallama' "$TMP/systemctl.log" | head -1 | cut -d: -f1)" \
     -lt "$(grep -nx 'start pallama' "$TMP/systemctl.log" | head -1 | cut -d: -f1)" ]; then
    ok "enable precedes start"
else
    bad "enable precedes start"
fi

# 9g. health poll target pin: the fallback port must be pallama's 11435,
# never ollama's 11434 (it answers green while pallama is dead). Comments
# may name 11434 to document the hazard; functional code may not.
sed 's/#.*$//' "$ROOT/scripts/install.sh" | grep -q 11434 &&
    bad "install.sh functionally references 11434" ||
    ok "install.sh polls 11435, never 11434"

echo "install e2e: $PASS passed, $FAIL failed"
[ "$FAIL" = 0 ]
