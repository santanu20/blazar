#!/bin/sh
# Wipe cargo's incremental-compilation cache (target/debug/incremental).
#
# Why: cargo NEVER garbage-collects target/ — stale per-session hash dirs
# pile up indefinitely (18G observed on this repo after heavy test waves;
# 818 orphaned session dirs). The cache is pure build speed state: wiping
# it is always safe when no build is running and costs one recompile of
# active crates. Run after heavy test sessions or from cron.
#
# Usage: sh scripts/clean-target.sh   (from anywhere inside the checkout)
set -eu

ROOT=$( unset CDPATH; cd -- "$(dirname -- "$0")/.." && pwd )
INC="$ROOT/target/debug/incremental"

# Never race a live build: cargo (or its rustc children) mid-run would
# recreate/rewrite files under us and the wipe could corrupt the session.
if command -v pgrep >/dev/null 2>&1; then
    if pgrep -x cargo >/dev/null 2>&1 || pgrep -x rustc >/dev/null 2>&1; then
        echo "ERROR: cargo/rustc is running — stop builds before wiping the cache" >&2
        exit 1
    fi
fi

if [ ! -d "$INC" ]; then
    echo "nothing to clean: $INC does not exist"
    exit 0
fi

kb_before=$(du -sk "$INC" 2>/dev/null | cut -f1) || kb_before=0
rm -rf "$INC"
freed_mb=$((kb_before / 1024))
echo "wiped $INC — freed ${freed_mb} MiB (next dev build recompiles from scratch once)"
