#!/bin/sh
# End-to-end test for scripts/install.sh against a local fake GitHub API.
#
# Proves, without touching GitHub or the real HOME:
#   1. happy path  — asset selected per-arch, digest verified, binary installed and runs
#   2. tamper      — corrupted digest in metadata -> hard failure, nothing installed
#   3. wrong arch  — release without a matching asset -> clear error naming the asset
#
# The fake release JSON puts a decoy asset with a WRONG digest first, so a
# parser bug that grabs a sibling asset's digest fails this test.
#
# Requires: python3, curl, sha256sum, and target/release/pallama (host build).

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
cat > "$TMP/fakesystemctl" <<'EOF'
#!/bin/sh
case "$1" in
    is-active) exit 3 ;;
    *) exit 0 ;;
esac
EOF
chmod +x "$TMP/fakesystemctl"
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

INSTALL_ENV="HOME=$TMP/home PALLAMA_INSTALL_BASE_URL=$BASE PALLAMA_SUDO=$TMP/fakesudo PALLAMA_SYSTEMCTL=$TMP/fakesystemctl PALLAMA_SYSTEM_BIN_DIR=$SYSTEM_BIN PALLAMA_UNIT_PATH=$UNIT_OUT"

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

echo
echo "install e2e: $PASS passed, $FAIL failed"
[ "$FAIL" = 0 ]
