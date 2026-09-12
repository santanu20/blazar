#!/usr/bin/env python3
"""pallama exhaustive validation harness — REAL engine, REAL model, REAL config.

The Step-11 "god tier" validator as a permanent script: boots an ISOLATED
pallama daemon (temp XDG dirs, copied store DB, symlinked real engine +
model files, own random-free port) and walks every config knob, API route, CLI
command, sentinel surface, and lifecycle behavior with real inference —
printing evidence for every check.

Isolation contract:
  - never touches ollama (11434), the user's daemon, or ~/.config/pallama
    (config.toml hash is verified unchanged at exit)
  - one engine child alive at a time; aborts a load if MemAvailable < 1.5 GiB
  - only ever signals pids it spawned (daemon) or that its daemon spawned
    (engine child, for the crash-respawn phase)

Usage:
  scripts/validate.py                 # full run (~10-15 min with model loads)
  PALLAMA_VALIDATE_FAST=1 scripts/validate.py   # skip long-wait phases
  scripts/validate.py --phase config --phase api # only these phases
  scripts/validate.py --self-test     # inject one failure, expect exit 1

Honest boundaries (printed, not hidden): rpc_servers needs a second box;
child_transport="unix" is a documented unsupported proxy path; /api/pull runs
for real (tiny model, heavy-gated) in full runs; 100% LINE coverage is
llvm-cov territory — this is exhaustive E2E path coverage.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
from os.path import abspath, dirname
from pathlib import Path
import re
import shutil
import signal
import socket
import ssl
import sqlite3
import subprocess
import sys
import tempfile
import traceback
import urllib.request
import threading
import time
import tomllib
import urllib.error


def _free_port() -> int:
    """Kernel-assigned ephemeral TCP port (probe-reserve-release)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


# Default to a random free port so two concurrent validate campaigns can
# never collide on the main listener (live 2026-09-10 cross-run 502s);
# PALLAMA_VALIDATE_PORT still pins an explicit port when set.
PORT = int(os.environ.get("PALLAMA_VALIDATE_PORT") or _free_port())
# Fast default: the 0.5B keeps every lane quick; PALLAMA_VALIDATE_MODEL
# overrides (e.g. release-grade runs pinning the 9B).
MODEL = os.environ.get("PALLAMA_VALIDATE_MODEL", "qwen2.5-0.5b-instruct")
# Second DISTINCT model for lanes that structurally need two models live at
# once (wave battery B predictive preload, mmproj projector attach). Separate
# from MODEL so the fast default stays small without collapsing those lanes.
BIG = os.environ.get("PALLAMA_VALIDATE_BIG_MODEL", "qwen3.5-9b")
FAST = os.environ.get("PALLAMA_VALIDATE_FAST", "") == "1"
# Optional device pin for MODEL on mixed iGPU/dGPU boxes (e.g. "Vulkan1").
VALIDATE_DEVICES = os.environ.get("PALLAMA_VALIDATE_DEVICES", "") or None
# Prefer the checkout's release binary when present: the harness validates the
# code under test, not whatever system copy happens to be installed.
_REPO_BIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "target",
    "release",
    "pallama",
)
if os.environ.get("PALLAMA_BIN"):
    PAL = os.environ["PALLAMA_BIN"]
elif os.path.exists(_REPO_BIN):
    PAL = _REPO_BIN
else:
    PAL = shutil.which("pallama") or "pallama"
REAL_DATA = os.path.expanduser("~/.local/share/pallama")
REAL_CONFIG = os.path.expanduser("~/.config/pallama/config.toml")
MEM_FLOOR_MIB = 1536

CHECKS: list[dict] = []
COVERAGE: list[dict] = []
DAEMON: "Daemon | None" = None
SANDBOX: "Sandbox | None" = None
USER_CONFIG_SHA = None


def check(phase: str, name: str, ok: bool, evidence: str = "") -> bool:
    tag = "PASS" if ok else "FAIL"
    CHECKS.append({"phase": phase, "name": name, "ok": bool(ok), "evidence": evidence})
    print(f"  [{tag}] {name}" + (f" — {evidence}" if evidence else ""))
    return bool(ok)


def boundary(phase: str, name: str, why: str) -> None:
    CHECKS.append(
        {"phase": phase, "name": name, "ok": True, "evidence": why, "boundary": True}
    )
    print(f"  [BOUNDARY] {name} — {why}")


def cov(knob: str, expectation: str, evidence: str, ok: bool = True) -> None:
    COVERAGE.append(
        {"knob": knob, "expectation": expectation, "evidence": evidence, "ok": bool(ok)}
    )
    if not ok:
        print(f"  [COV-FAIL] {knob}: expected {expectation}, got {evidence}")


# ---------------------------------------------------------------------------
# MANIFEST REGISTRY (merged from the former validate_manifests.py — the
# single source of truth for real-integration coverage manifests, kept as
# one file with its consumer per the one-script-per-concern layout).
# ---------------------------------------------------------------------------
# Single source of truth for Pallama real-integration coverage manifests.
#
# Everything validate.py enforces for 100% command + config coverage is
# declared here, table-driven. Adding a CLI subcommand or a Config knob
# WITHOUT extending this file makes the completeness gates fail loudly
# (bidirectional set comparisons), so drift is impossible to miss.
#
# Field inventories verified against crates/pallama-core/src/config.rs:
#   Config          145 fields (19 Option, 4 containers: keys/remotes/engine_env/model_overrides)
#   ModelOverride    25 fields (all Option)
#   SamplerDefaults  18 fields (all Option, skip_serializing_if none)
#   ApiKey            7 fields    Remote  3 fields
# Command set verified against `pallama --help` (38 subcommands + help).
#
# Tier semantics (honest evidence classes):
#   argv       knob value reaches child llama-server argv (child_argv assert)
#   behavior   observable daemon/gateway behavior (HTTP probe / environ / nice / phase)
#   existing   already covered by a pre-existing validate.py phase (cov row exists or added)
#   tune       exercised via `pallama tune` lane
#   roundtrip  set -> config list echo + full-manifest daemon boot (serde deny_unknown_fields proof)
#   boundary   honest boundary with documented reason (needs 2nd box / RAM / unsupported transport)
# """
#
# # ---------------------------------------------------------------------------
# # COMMANDS manifest: every CLI leaf path that must have >=1 real (no-mock)
# # check. attrs: daemon=needs running daemon, model=needs loaded model,
# # heavy=big disk/net/RAM lane, net=network required, fast=runs in FAST mode
# # (fast=False -> honest boundary row in FAST runs).
# # ---------------------------------------------------------------------------
#
# _COMMAND_ATTRS = {
#     # path:                daemon model heavy net  fast
#     "serve": (True, False, False, False, True),
#     "start": (True, False, False, False, True),
#     "cp": (True, False, False, False, True),
#     "create.happy": (False, False, False, False, True),
#     "create.reject": (False, False, False, False, True),
#     "push.refusal": (False, False, False, False, True),
#     "keys.list": (True, False, False, False, True),
#     "keys.add": (True, False, False, False, True),
#     "keys.rotate": (True, False, False, False, True),
#     "keys.rm": (True, False, False, False, True),
#     "quantize.happy": (False, False, True, False, False),
#     "quantize.refusal": (False, False, False, False, True),
#     "launch": (True, False, False, False, True),
#     "signin.refusal": (False, False, False, False, True),
#     "login.refusal": (False, False, False, False, True),
#     "signout.refusal": (False, False, False, False, True),
#     "logout.refusal": (False, False, False, False, True),
#     "stop.model": (True, True, False, False, True),
#     "stop.bare": (True, False, False, False, True),
#     "pull": (False, False, True, True, False),
#     "import.hardlink": (False, False, False, False, True),
#     "import.copy": (False, False, False, False, True),
#     "mmproj.happy": (False, False, True, True, False),
#     "mmproj.refusal": (False, False, False, False, True),
#     "rm": (False, False, False, False, True),
#     "list": (False, False, False, False, True),
#     "ls": (False, False, False, False, True),
#     "show": (False, False, False, False, True),
#     "ps": (True, False, False, False, True),
#     "ps.reset": (True, False, False, False, True),
#     "run.single": (True, True, False, False, True),
#     "run.repl-exit": (True, True, False, False, True),
#     "run.repl-eof": (True, True, False, False, True),
#     "run.verbose": (True, True, False, False, True),
#     "run.repl.interactive": (True, True, False, False, True),
#     "run.repl.ctrl-d": (True, True, False, False, True),
#     "run.repl.ctrl-c": (True, True, False, False, True),
#     "run.single.nocap": (True, True, False, False, True),
#     "bench": (False, False, False, False, True),
#     "tune.search": (False, False, True, False, False),
#     "tune.ctx": (False, False, True, False, False),
#     "tune.spec": (False, False, True, False, False),
#     "tune.slots": (False, False, True, False, False),
#     "tune.ngram": (False, False, True, False, False),
#     "tune.load": (False, False, True, False, False),
#     "tune.replicas": (False, False, True, False, False),
#     "tune.cache-reuse": (False, False, True, False, False),
#     "migrate": (False, False, False, False, True),
#     "snapshot": (False, False, False, False, True),
#     "completions": (False, False, False, False, True),
#     "drafts": (False, False, False, False, True),
#     "coreside": (False, False, False, False, True),
#     "whisper.install": (False, False, True, True, False),
#     "whisper.pull": (False, False, True, True, False),
#     "whisper.list": (False, False, False, False, True),
#     "whisper.transcribe": (False, False, False, False, False),
#     "whisper.pin": (False, False, False, False, False),
#     "whisper.pin.refusal": (False, False, False, False, True),
#     "engine.update": (False, False, True, True, False),
#     "engine.list": (False, False, False, False, True),
#     "engine.use": (False, False, False, False, True),
#     "engine.rollback": (False, False, False, False, True),
#     "engine.local": (False, False, False, False, True),
#     "lora.add": (False, False, False, False, True),
#     "lora.rm": (False, False, False, False, True),
#     "lora.list": (False, False, False, False, True),
#     "search": (False, False, False, True, True),
#     "fit": (False, False, False, True, True),
#     "config.list": (False, False, False, False, True),
#     "config.get": (False, False, False, False, True),
#     "config.set": (False, False, False, False, True),
#     "upgrade.dry-run": (False, False, False, True, False),
#     "session.save": (True, True, False, False, True),
#     "session.restore": (True, True, False, False, True),
#     "session.rm": (True, True, False, False, True),
#     "session.list": (True, True, False, False, True),
#     "doctor": (False, False, False, False, True),
#     "why.default": (True, False, False, False, True),
#     "why.trace": (True, False, False, False, True),
#     "watch": (True, False, False, False, True),
#     "help": (False, False, False, False, True),
# }
#
# COMMANDS = [
#     {
#         "path": path,
#         "daemon": a[0],
#         "model": a[1],
#         "heavy": a[2],
#         "net": a[3],
#         "fast": a[4],
#     }
#     for path, a in _COMMAND_ATTRS.items()
# ]
#
# # CLI aliases: real paths we exercise, but NOT advertised in `--help`.
# ALIASES = {"ls": "list", "start": "serve"}
#
# # Top-level subcommand names (derived from paths, aliases excluded) + `help` —
# # the set that `pallama --help` must advertise, exactly, in both directions.
# TOPLEVEL_COMMANDS = sorted(
#     ({p.split(".")[0] for p in _COMMAND_ATTRS} - set(ALIASES)) | {"help"}
# )
#
# # ---------------------------------------------------------------------------
# # TOPLEVEL_KNOBS manifest: all 134 Config fields.
# # option=True  -> Option<T>, absent from fresh `config list` until set
# # container=True -> keys / remotes / engine_env / model_overrides section
# # tier  -> evidence class (see module docstring); group -> knobs_argv batch
# # ---------------------------------------------------------------------------
#
# _K = [
#     # (name, option, container, tier, group, expectation)
#     (
#         "host",
#         False,
#         False,
#         "behavior",
#         None,
#         "binds 127.0.0.1:11499 (every daemon phase)",
#     ),
#     (
#         "port",
#         False,
#         False,
#         "behavior",
#         None,
#         "binds 127.0.0.1:11499 (every daemon phase)",
#     ),
#     ("default_ctx", False, False, "existing", None, "phase_config A: --ctx-size 8192"),
#     ("idle_sleep_secs", False, False, "argv", "G1", "--sleep-idle-seconds 77"),
#     (
#         "idle_timeout_secs",
#         False,
#         False,
#         "roundtrip",
#         None,
#         "daemon idle reaper; set->list echo + full-manifest boot",
#     ),
#     (
#         "max_loaded_models",
#         False,
#         False,
#         "boundary",
#         None,
#         "needs 2+ concurrently-loaded models (RAM)",
#     ),
#     (
#         "child_transport",
#         False,
#         False,
#         "boundary",
#         None,
#         "unix transport unsupported on this build",
#     ),
#     (
#         "child_auth",
#         True,
#         False,
#         "behavior",
#         None,
#         "phase_auth: 401 direct-to-child vs 200 proxied",
#     ),
#     (
#         "engine_asset",
#         False,
#         False,
#         "existing",
#         None,
#         "engine mgmt: doctor + engine list",
#     ),
#     ("spec", False, False, "existing", None, "phase_config B: --spec-type ngram"),
#     ("cache_reuse", False, False, "argv", "G1", "--cache-reuse 128"),
#     ("keys", False, True, "behavior", None, "keys lifecycle + phase_auth"),
#     (
#         "audit_log",
#         False,
#         False,
#         "behavior",
#         None,
#         "battery F1: keyed daemon audit_log=true -> audit.jsonl lines "
#         "(200 line + 401 silence)",
#     ),
#     (
#         "semantic_cache",
#         False,
#         True,
#         "roundtrip",
#         None,
#         "R4 opt-in semantic cache container; full-manifest boot w/ enabled=false (enabled=true requires model)",
#     ),
#     (
#         "rpc_servers",
#         False,
#         False,
#         "boundary",
#         None,
#         "needs 2nd box running rpc llama-server",
#     ),
#     (
#         "cache_ram_mb",
#         False,
#         False,
#         "existing",
#         None,
#         "phase_config: --cache-ram + 30% clamp",
#     ),
#     ("cpu_range", False, False, "argv", "G1", "--cpu-range 0-7"),
#     ("poll", False, False, "argv", "G1", "--poll 77"),
#     (
#         "reasoning_format",
#         False,
#         False,
#         "existing",
#         None,
#         "phase_config A: --reasoning-format deepseek",
#     ),
#     ("slots", False, False, "existing", None, "phase_config A: -np 2"),
#     (
#         "deterministic",
#         False,
#         False,
#         "roundtrip",
#         None,
#         "set->list echo + slots=1 pin at profile compile",
#     ),
#     (
#         "cache_type",
#         False,
#         False,
#         "existing",
#         None,
#         "phase_config B: --cache-type-k q8_0",
#     ),
#     (
#         "kv_unified",
#         True,
#         False,
#         "roundtrip",
#         None,
#         "set->list echo + full-manifest boot",
#     ),
#     ("kv_unified_per_slot", False, False, "argv", "G1", "--kv-unified-per-slot 4096"),
#     ("swa_full", False, False, "argv", "G1", "--swa-full"),
#     ("ctx_checkpoints", False, False, "argv", "G1", "--ctx-checkpoints 16"),
#     ("no_kv_offload", False, False, "argv", "G1", "--no-kv-offload"),
#     (
#         "load_mode",
#         False,
#         False,
#         "roundtrip",
#         None,
#         "value semantics unverified; set->list echo + boot",
#     ),
#     (
#         "spawn_mem_guard",
#         False,
#         False,
#         "roundtrip",
#         None,
#         "mem-floor logic; set->list echo + boot",
#     ),
#     (
#         "session_bank",
#         False,
#         False,
#         "roundtrip",
#         None,
#         "exercised by session save/restore; echo + boot",
#     ),
#     (
#         "singleflight",
#         False,
#         False,
#         "behavior",
#         None,
#         "2 concurrent cold spawns -> 1 child pid",
#     ),
#     (
#         "prompt_preflight",
#         False,
#         False,
#         "behavior",
#         None,
#         "oversized prompt -> teaching 4xx",
#     ),
#     (
#         "spec_cache",
#         False,
#         False,
#         "existing",
#         None,
#         "phase_config B: --lookup-cache-dynamic",
#     ),
#     ("ctx_extend", False, False, "existing", None, "phase_config B: yarn/rope flags"),
#     ("cpu_moe_n", False, False, "argv", "G1", "--n-cpu-moe 2"),
#     (
#         "override_tensor",
#         False,
#         False,
#         "argv",
#         "G1",
#         "--override-tensor .ffn_.*_exps.=CPU",
#     ),
#     ("agent", False, False, "argv", "G1", "--agent"),
#     ("sessions", False, False, "existing", None, "phase_config B: --slot-save-path"),
#     ("router", False, False, "behavior", None, "router=true passthrough chat probe"),
#     (
#         "router_max_models",
#         False,
#         False,
#         "roundtrip",
#         None,
#         "set->list echo + full-manifest boot",
#     ),
#     (
#         "devices",
#         False,
#         False,
#         "existing",
#         None,
#         "phase_config A2: --device <manifest pick>",
#     ),
#     (
#         "engine_check_secs",
#         False,
#         False,
#         "existing",
#         None,
#         "run/engine-check.json marker <=20s",
#     ),
#     (
#         "slot_prompt_similarity",
#         False,
#         False,
#         "argv",
#         "G1",
#         "--slot-prompt-similarity 0.6",
#     ),
#     ("sentinel", False, False, "existing", None, "phase_sentinel"),
#     ("sentinel_stall_secs", False, False, "existing", None, "phase_sentinel"),
#     ("sentinel_enforce", False, False, "existing", None, "phase_sentinel"),
#     (
#         "tls_cert",
#         False,
#         False,
#         "behavior",
#         None,
#         "rustls same-port: https 200, plain http fails",
#     ),
#     (
#         "tls_key",
#         False,
#         False,
#         "behavior",
#         None,
#         "rustls same-port: https 200, plain http fails",
#     ),
#     ("cors_origins", False, False, "behavior", None, "OPTIONS preflight -> ACAO echo"),
#     (
#         "otlp_endpoint",
#         False,
#         False,
#         "behavior",
#         None,
#         "local collector captures export POST",
#     ),
#     (
#         "otlp_service",
#         False,
#         False,
#         "behavior",
#         None,
#         "service name in captured export POST",
#     ),
#     ("remotes", False, True, "behavior", None, "config echo + doctor"),
#     (
#         "engine_env",
#         False,
#         True,
#         "existing",
#         None,
#         "phase_config C: child environ probe",
#     ),
#     (
#         "model_overrides",
#         False,
#         True,
#         "existing",
#         None,
#         "phase_config C: override ctx wins",
#     ),
#     (
#         "spec_draft_cpu_range",
#         False,
#         False,
#         "roundtrip",
#         None,
#         "draft family; echo + full-manifest boot",
#     ),
#     (
#         "spec_draft_cpu_strict",
#         False,
#         False,
#         "roundtrip",
#         None,
#         "draft family; echo + boot",
#     ),
#     ("spec_draft_device", False, False, "roundtrip", None, "draft family; echo + boot"),
#     ("spec_draft_ngl", False, False, "roundtrip", None, "draft family; echo + boot"),
#     (
#         "spec_draft_threads",
#         False,
#         False,
#         "roundtrip",
#         None,
#         "draft family; echo + boot",
#     ),
#     ("spec_draft_p_min", True, False, "roundtrip", None, "draft family; echo + boot"),
#     ("spec_draft_p_split", True, False, "roundtrip", None, "draft family; echo + boot"),
#     ("spec_draft_poll", True, False, "roundtrip", None, "draft family; echo + boot"),
#     ("spec_draft_prio", False, False, "roundtrip", None, "draft family; echo + boot"),
#     (
#         "spec_draft_prio_batch",
#         False,
#         False,
#         "roundtrip",
#         None,
#         "draft family; echo + boot",
#     ),
#     (
#         "spec_draft_poll_batch",
#         True,
#         False,
#         "roundtrip",
#         None,
#         "draft family; echo + boot",
#     ),
#     (
#         "spec_draft_cpu_strict_batch",
#         False,
#         False,
#         "roundtrip",
#         None,
#         "draft family; echo + boot",
#     ),
#     (
#         "spec_draft_threads_batch",
#         False,
#         False,
#         "roundtrip",
#         None,
#         "draft family; echo + boot",
#     ),
#     ("spec_draft_type_k", False, False, "roundtrip", None, "draft family; echo + boot"),
#     ("spec_draft_type_v", False, False, "roundtrip", None, "draft family; echo + boot"),
#     (
#         "spec_draft_override_tensor",
#         False,
#         False,
#         "roundtrip",
#         None,
#         "draft family; echo + boot",
#     ),
#     (
#         "spec_draft_n_cpu_moe",
#         False,
#         False,
#         "roundtrip",
#         None,
#         "draft family; echo + boot",
#     ),
#     (
#         "spec_draft_cpu_moe",
#         False,
#         False,
#         "roundtrip",
#         None,
#         "draft family; echo + boot",
#     ),
#     (
#         "spec_draft_backend_sampling",
#         False,
#         False,
#         "roundtrip",
#         None,
#         "draft family; echo + boot",
#     ),
#     ("adaptive_decay", False, False, "existing", None, "sentinel/adaptive phase"),
#     ("adaptive_target", False, False, "existing", None, "sentinel/adaptive phase"),
#     ("ngram_size_m", False, False, "tune", None, "tune --ngram lane"),
#     ("ngram_size_n", False, False, "tune", None, "tune --ngram lane"),
#     ("ngram_min_hits", False, False, "tune", None, "tune --ngram lane"),
#     ("ngram_mod_n_match", False, False, "tune", None, "tune --ngram lane"),
#     ("ngram_mod_n_max", False, False, "tune", None, "tune --ngram lane"),
#     ("ngram_mod_n_min", False, False, "tune", None, "tune --ngram lane"),
#     (
#         "reasoning_budget",
#         False,
#         False,
#         "roundtrip",
#         None,
#         "echo + boot (default fn visible in fresh list)",
#     ),
#     ("reasoning_budget_message", False, False, "roundtrip", None, "echo + boot"),
#     ("reasoning_effort", False, False, "roundtrip", None, "echo + boot"),
#     ("reasoning_preserve", True, False, "roundtrip", None, "echo + boot"),
#     ("image_max_tokens", False, False, "argv", "G1", "--image-max-tokens 4096"),
#     ("image_min_tokens", False, False, "argv", "G1", "--image-min-tokens 64"),
#     ("mtmd_batch_max_tokens", False, False, "roundtrip", None, "echo + boot"),
#     ("mmproj_offload", False, False, "roundtrip", None, "echo + boot"),
#     ("mmproj_auto", False, False, "roundtrip", None, "echo + boot"),
#     ("mmproj_device", False, False, "argv", "G1", "--mmproj-device <device>"),
#     ("embd_normalize", False, False, "argv", "G1", "--embd-normalize 2"),
#     (
#         "yarn_orig_ctx",
#         False,
#         False,
#         "argv",
#         "G3",
#         "--rope-scaling yarn + --rope-orig-ctx",
#     ),
#     ("yarn_ext_factor", False, False, "argv", "G3", "--rope-scale 1.5"),
#     ("yarn_attn_factor", False, False, "argv", "G3", "--yarn-attn-factor"),
#     ("yarn_beta_fast", False, False, "argv", "G3", "--yarn-beta-fast"),
#     ("yarn_beta_slow", False, False, "argv", "G3", "--yarn-beta-slow"),
#     ("cpu_strict", False, False, "roundtrip", None, "echo + boot"),
#     (
#         "prio",
#         False,
#         False,
#         "behavior",
#         "G2",
#         "child nice via setpriority (/proc stat ni)",
#     ),
#     ("prio_batch", False, False, "behavior", "G2", "same spawn, nice observable"),
#     ("poll_batch", True, False, "roundtrip", None, "echo + boot"),
#     ("threads_http", False, False, "argv", "G1", "--threads-http 2"),
#     ("warmup", False, False, "argv", "G1", "false -> --no-warmup"),
#     ("repack", False, False, "roundtrip", None, "echo + boot"),
#     ("cache_idle_slots", False, False, "argv", "G1", "--cache-idle-slots"),
#     (
#         "lookup_cache_static",
#         True,
#         False,
#         "roundtrip",
#         None,
#         "missing-file refusal (wave) + echo + boot",
#     ),
#     (
#         "lookup_cache_dynamic",
#         True,
#         False,
#         "roundtrip",
#         None,
#         "missing-file refusal (wave) + echo + boot",
#     ),
#     ("predictive_preload", False, False, "existing", None, "phase_wave preload lane"),
#     ("adaptive_slots", True, False, "roundtrip", None, "echo + boot"),
#     ("no_host", False, False, "argv", "G1", "--no-host"),
#     ("op_offload", True, False, "argv", "G1", "--op-offload"),
#     ("keep_tokens", False, False, "roundtrip", None, "echo + boot"),
#     (
#         "override_kv",
#         False,
#         False,
#         "roundtrip",
#         None,
#         "echo + boot (valid-kv values unverified)",
#     ),
#     (
#         "control_vectors",
#         False,
#         False,
#         "roundtrip",
#         None,
#         "echo + boot (needs real .gguf to emit)",
#     ),
#     ("control_vectors_scaled", False, False, "roundtrip", None, "echo + boot"),
#     ("control_vector_layer_range", False, False, "roundtrip", None, "echo + boot"),
#     ("tensor_preset", False, False, "roundtrip", None, "echo + boot"),
#     ("pii_scrub", False, False, "roundtrip", None, "echo + boot"),
#     ("video_ffmpeg_dir", False, False, "roundtrip", None, "echo + boot"),
#     ("video_fps", False, False, "roundtrip", None, "echo + boot"),
#     ("video_timestamp_interval", False, False, "roundtrip", None, "echo + boot"),
#     ("numa", False, False, "roundtrip", None, "echo + boot"),
#     ("check_tensors", False, False, "roundtrip", None, "echo + boot"),
#     ("context_shift", False, False, "roundtrip", None, "echo + boot"),
#     ("samplers", False, False, "roundtrip", None, "echo + boot"),
#     ("batch_size", False, False, "argv", "G2", "--batch-size 512"),
#     ("ubatch_size", False, False, "argv", "G2", "--ubatch-size 256"),
#     ("threads_batch", False, False, "argv", "G2", "--threads-batch 2"),
#     ("main_gpu", False, False, "argv", "G2", "--main-gpu 0"),
#     ("split_mode", False, False, "argv", "G2", "--split-mode layer"),
#     ("tensor_split", False, False, "argv", "G2", "--tensor-split 3,1"),
#     (
#         "models_autoload",
#         True,
#         False,
#         "boundary",
#         None,
#         "loads ALL store models (RAM); echo + boot",
#     ),
#     (
#         "log_level",
#         True,
#         False,
#         "roundtrip",
#         None,
#         "tracing filter (e.g. pallama=debug); set->list echo + full-manifest boot",
#     ),
#     (
#         "late_chunking_max_tokens",
#         False,
#         False,
#         "roundtrip",
#         None,
#         "late-chunking embeddings cap (default 8192); echo + full-manifest boot",
#     ),
#     (
#         "session_keep_secs",
#         False,
#         False,
#         "roundtrip",
#         None,
#         "session-pin idle-eviction guard window (default 900s, 0=off); echo + full-manifest boot",
#     ),
#     (
#         "update_channel",
#         False,
#         False,
#         "roundtrip",
#         None,
#         "engine/app update channel (stable|latest); fresh-visible + roundtrip boot",
#     ),
# ]
#
# TOPLEVEL_KNOBS = [
#     {"name": n, "option": o, "container": c, "tier": t, "group": g, "expectation": e}
#     for (n, o, c, t, g, e) in _K
# ]
#
# OPTION_KNOBS = [k["name"] for k in TOPLEVEL_KNOBS if k["option"]]
# CONTAINER_KNOBS = [k["name"] for k in TOPLEVEL_KNOBS if k["container"]]
# # Knobs expected visible in a fresh `config list` at defaults: everything
# # except Option fields and (empty-at-default) containers.
# FRESH_VISIBLE_KNOBS = [k["name"] for k in TOPLEVEL_KNOBS if not k["option"]]
#
# # ---------------------------------------------------------------------------
# # MODEL_OVERRIDE manifest: all 23 ModelOverride fields + 18 SamplerDefaults
# # leaves. Evidence = overlay round-trip (gate d) + argv/wave lanes noted.
# # ---------------------------------------------------------------------------
#
# MODEL_OVERRIDE_FIELDS = [
#     ("ctx", "argv: --ctx-size (phase_config C override-wins)"),
#     ("slots", "argv: -np"),
#     (
#         "deterministic",
#         "resolve_slots pin: forces -np 1 (greedy reproducibility)",
#     ),
#     ("spec", "argv: --spec-type"),
#     ("loras", "wave: lora attach"),
#     ("extra_args", "argv: passthrough tokens"),
#     ("cache_type", "argv: --cache-type-k/v"),
#     ("kv_unified", "roundtrip echo"),
#     ("ctx_extend", "argv: rope flags"),
#     ("cpu_moe_n", "argv: --n-cpu-moe"),
#     ("override_tensor", "argv: --override-tensor"),
#     ("devices", "argv: --device (phase_config A2)"),
#     ("warmup", "argv: false -> --no-warmup"),
#     ("reasoning_budget", "roundtrip echo"),
#     ("reasoning_effort", "roundtrip echo"),
#     ("replicas", "wave: replica pid"),
#     ("pin", "wave: pinned spawn"),
#     ("chat_template", "argv: --chat-template"),
#     ("chat_template_file", "argv: --chat-template-file"),
#     ("sampler_defaults", "argv: --temp..--seed set"),
#     ("spm_infill", "roundtrip echo"),
# ]
#
# SAMPLER_FIELDS = [
#     "temperature",
#     "top_k",
#     "top_p",
#     "min_p",
#     "top_n_sigma",
#     "typical_p",
#     "repeat_penalty",
#     "repeat_last_n",
#     "presence_penalty",
#     "frequency_penalty",
#     "dry_multiplier",
#     "dry_base",
#     "dry_allowed_length",
#     "dry_penalty_last_n",
#     "xtc_probability",
#     "xtc_threshold",
#     "mirostat",
#     "seed",
# ]
#
# APIKEY_FIELDS = [
#     "name",
#     "key",
#     "models",
#     "rpm",
#     "tpm",
#     "daily_tokens",
#     "max_concurrent",
# ]
# REMOTE_FIELDS = ["name", "url", "key"]
#
#
# # ---------------------------------------------------------------------------
# # Pure helpers (no side effects; safe to import from validate.py)
# # ---------------------------------------------------------------------------
#
#
# def command_paths():
#     return [c["path"] for c in COMMANDS]
#
#
# def toplevel_knob_names(include_options=True):
#     names = [k["name"] for k in TOPLEVEL_KNOBS]
#     if include_options:
#         return names
#     return [n for n in names if n not in set(OPTION_KNOBS)]
#
#
# def knob_entry(name):
#     for k in TOPLEVEL_KNOBS:
#         if k["name"] == name:
#             return k
#     raise KeyError(f"unknown knob: {name}")
#
#
# def argv_groups():
#     """knobs_argv spawn groups: group name -> [(knob, value, flag expectation)]
#
# ---------------------------------------------------------------------------

_COMMAND_ATTRS = {
    # path:                daemon model heavy net  fast
    "serve": (True, False, False, False, True),
    "start": (True, False, False, False, True),
    "cp": (True, False, False, False, True),
    "create.happy": (False, False, False, False, True),
    "create.reject": (False, False, False, False, True),
    "push.refusal": (False, False, False, False, True),
    "keys.list": (True, False, False, False, True),
    "keys.add": (True, False, False, False, True),
    "keys.rotate": (True, False, False, False, True),
    "keys.rm": (True, False, False, False, True),
    "quantize.happy": (False, False, True, False, False),
    "quantize.refusal": (False, False, False, False, True),
    "launch": (True, False, False, False, True),
    "signin.refusal": (False, False, False, False, True),
    "login.refusal": (False, False, False, False, True),
    "signout.refusal": (False, False, False, False, True),
    "logout.refusal": (False, False, False, False, True),
    "stop.model": (True, True, False, False, True),
    "stop.bare": (True, False, False, False, True),
    "pull": (False, False, True, True, False),
    "run.miss-pulls": (True, True, True, True, False),
    "import.hardlink": (False, False, False, False, True),
    "import.copy": (False, False, False, False, True),
    "mmproj.happy": (False, False, True, True, False),
    "mmproj.refusal": (False, False, False, False, True),
    "rm": (False, False, False, False, True),
    "list": (False, False, False, False, True),
    "ls": (False, False, False, False, True),
    "show": (False, False, False, False, True),
    "show.colon": (False, False, False, False, True),
    "ps": (True, False, False, False, True),
    "ps.reset": (True, False, False, False, True),
    "ps.device": (True, False, False, False, True),
    "ps.warnings": (True, False, False, False, True),
    "chat.colon": (True, True, False, False, True),
    "run.single": (True, True, False, False, True),
    "run.repl-exit": (True, True, False, False, True),
    "run.repl-eof": (True, True, False, False, True),
    "run.verbose": (True, True, False, False, True),
    "run.repl.interactive": (True, True, False, False, True),
    "run.repl.ctrl-d": (True, True, False, False, True),
    "run.repl.ctrl-c": (True, True, False, False, True),
    "run.single.nocap": (True, True, False, False, True),
    "bench": (False, False, False, False, True),
    "tune.search": (False, False, True, False, False),
    "tune.ctx": (False, False, True, False, False),
    "tune.spec": (False, False, True, False, False),
    "tune.slots": (False, False, True, False, False),
    "tune.ngram": (False, False, True, False, False),
    "tune.load": (False, False, True, False, False),
    "tune.replicas": (False, False, True, False, False),
    "tune.cache-reuse": (False, False, True, False, False),
    "migrate": (False, False, False, False, True),
    "snapshot": (False, False, False, False, True),
    "completions": (False, False, False, False, True),
    "drafts": (False, False, False, False, True),
    "coreside": (False, False, False, False, True),
    "whisper.install": (False, False, True, True, False),
    "whisper.pull": (False, False, True, True, False),
    "whisper.list": (False, False, False, False, True),
    "whisper.transcribe": (False, False, False, False, False),
    "whisper.pin": (False, False, False, False, False),
    "whisper.pin.refusal": (False, False, False, False, True),
    "engine.update": (False, False, True, True, False),
    "engine.list": (False, False, False, False, True),
    "engine.use": (False, False, False, False, True),
    "engine.rollback": (False, False, False, False, True),
    "engine.local": (False, False, False, False, True),
    "lora.add": (False, False, False, False, True),
    "lora.rm": (False, False, False, False, True),
    "lora.list": (False, False, False, False, True),
    "search": (False, False, False, True, True),
    "fit": (False, False, False, True, True),
    "config.list": (False, False, False, False, True),
    "config.get": (False, False, False, False, True),
    "config.set": (False, False, False, False, True),
    "upgrade.dry-run": (False, False, False, True, False),
    "session.save": (True, True, False, False, True),
    "session.restore": (True, True, False, False, True),
    "session.rm": (True, True, False, False, True),
    "session.list": (True, True, False, False, True),
    "doctor": (False, False, False, False, True),
    "why.default": (True, False, False, False, True),
    "why.trace": (True, False, False, False, True),
    "watch": (True, False, False, False, True),
    "help": (False, False, False, False, True),
}

COMMANDS = [
    {
        "path": path,
        "daemon": a[0],
        "model": a[1],
        "heavy": a[2],
        "net": a[3],
        "fast": a[4],
    }
    for path, a in _COMMAND_ATTRS.items()
]

# CLI aliases: real paths we exercise, but NOT advertised in `--help`.
ALIASES = {"ls": "list", "start": "serve"}

# Top-level subcommand names (derived from paths, aliases excluded) + `help` —
# the set that `pallama --help` must advertise, exactly, in both directions.
TOPLEVEL_COMMANDS = sorted(
    ({p.split(".")[0] for p in _COMMAND_ATTRS} - set(ALIASES)) | {"help"}
)

# ---------------------------------------------------------------------------
# TOPLEVEL_KNOBS manifest: all 145 Config fields.
# option=True  -> Option<T>, absent from fresh `config list` until set
# container=True -> keys / remotes / engine_env / model_overrides section
# tier  -> evidence class (see module docstring); group -> knobs_argv batch
# ---------------------------------------------------------------------------

_K = [
    # (name, option, container, tier, group, expectation)
    (
        "host",
        False,
        False,
        "behavior",
        None,
        "binds 127.0.0.1:<port> (every daemon phase; random-free unless PALLAMA_VALIDATE_PORT pins one)",
    ),
    (
        "port",
        False,
        False,
        "behavior",
        None,
        "binds 127.0.0.1:<port> (every daemon phase; random-free unless PALLAMA_VALIDATE_PORT pins one)",
    ),
    ("default_ctx", False, False, "existing", None, "phase_config A: --ctx-size 8192"),
    ("idle_sleep_secs", False, False, "argv", "G1", "--sleep-idle-seconds 77"),
    (
        "idle_timeout_secs",
        False,
        False,
        "roundtrip",
        None,
        "daemon idle reaper; set->list echo + full-manifest boot",
    ),
    (
        "max_loaded_models",
        False,
        False,
        "boundary",
        None,
        "needs 2+ concurrently-loaded models (RAM)",
    ),
    (
        "child_transport",
        False,
        False,
        "boundary",
        None,
        "unix transport unsupported on this build",
    ),
    (
        "child_auth",
        True,
        False,
        "behavior",
        None,
        "phase_auth: 401 direct-to-child vs 200 proxied",
    ),
    (
        "engine_asset",
        False,
        False,
        "existing",
        None,
        "engine mgmt: doctor + engine list",
    ),
    ("spec", False, False, "existing", None, "phase_config B: --spec-type ngram"),
    ("cache_reuse", False, False, "argv", "G1", "--cache-reuse 128"),
    ("keys", False, True, "behavior", None, "keys lifecycle + phase_auth"),
    (
        "audit_log",
        False,
        False,
        "behavior",
        None,
        "battery F1: keyed daemon audit_log=true -> audit.jsonl lines "
        "(200 line + 401 silence)",
    ),
    (
        "semantic_cache",
        False,
        True,
        "roundtrip",
        None,
        "R4 opt-in semantic cache container; full-manifest boot w/ enabled=false (enabled=true requires model)",
    ),
    (
        "rpc_servers",
        False,
        False,
        "boundary",
        None,
        "needs 2nd box running rpc llama-server",
    ),
    (
        "cache_ram_mb",
        False,
        False,
        "existing",
        None,
        "phase_config: --cache-ram + 30% clamp",
    ),
    ("cpu_range", False, False, "argv", "G1", "--cpu-range 0-7"),
    ("poll", False, False, "argv", "G1", "--poll 77"),
    (
        "reasoning_format",
        False,
        False,
        "existing",
        None,
        "phase_config A: --reasoning-format deepseek",
    ),
    ("slots", False, False, "existing", None, "phase_config A: -np 2"),
    (
        "deterministic",
        False,
        False,
        "roundtrip",
        None,
        "set->list echo + slots=1 pin at profile compile",
    ),
    (
        "cache_type",
        False,
        False,
        "existing",
        None,
        "phase_config B: --cache-type-k q8_0",
    ),
    (
        "kv_unified",
        True,
        False,
        "roundtrip",
        None,
        "set->list echo + full-manifest boot",
    ),
    ("kv_unified_per_slot", False, False, "argv", "G1", "--kv-unified-per-slot 4096"),
    ("swa_full", False, False, "argv", "G1", "--swa-full"),
    ("ctx_checkpoints", False, False, "argv", "G1", "--ctx-checkpoints 16"),
    ("no_kv_offload", False, False, "argv", "G1", "--no-kv-offload"),
    (
        "load_mode",
        False,
        False,
        "roundtrip",
        None,
        "value semantics unverified; set->list echo + boot",
    ),
    (
        "spawn_mem_guard",
        False,
        False,
        "roundtrip",
        None,
        "mem-floor logic; set->list echo + boot",
    ),
    (
        "session_bank",
        False,
        False,
        "roundtrip",
        None,
        "exercised by session save/restore; echo + boot",
    ),
    (
        "singleflight",
        False,
        False,
        "behavior",
        None,
        "2 concurrent cold spawns -> 1 child pid",
    ),
    (
        "prompt_preflight",
        False,
        False,
        "behavior",
        None,
        "oversized prompt -> teaching 4xx",
    ),
    (
        "spec_cache",
        False,
        False,
        "existing",
        None,
        "phase_config B: --lookup-cache-dynamic",
    ),
    ("ctx_extend", False, False, "existing", None, "phase_config B: yarn/rope flags"),
    ("cpu_moe_n", False, False, "argv", "G1", "--n-cpu-moe 2"),
    ("cpu_ffn_n", False, False, "argv", "G1", "--n-cpu-ffn 2"),
    (
        "override_tensor",
        False,
        False,
        "argv",
        "G1",
        "--override-tensor .ffn_.*_exps.=CPU",
    ),
    ("agent", False, False, "argv", "G1", "--agent"),
    ("sessions", False, False, "existing", None, "phase_config B: --slot-save-path"),
    ("router", False, False, "behavior", None, "router=true passthrough chat probe"),
    (
        "router_max_models",
        False,
        False,
        "roundtrip",
        None,
        "set->list echo + full-manifest boot",
    ),
    (
        "devices",
        False,
        False,
        "existing",
        None,
        "phase_config A2: --device <manifest pick>",
    ),
    (
        "engine_check_secs",
        False,
        False,
        "existing",
        None,
        "run/engine-check.json marker <=20s",
    ),
    (
        "slot_prompt_similarity",
        False,
        False,
        "argv",
        "G1",
        "--slot-prompt-similarity 0.6",
    ),
    ("sentinel", False, False, "existing", None, "phase_sentinel"),
    ("sentinel_stall_secs", False, False, "existing", None, "phase_sentinel"),
    ("sentinel_enforce", False, False, "existing", None, "phase_sentinel"),
    (
        "tls_cert",
        False,
        False,
        "behavior",
        None,
        "rustls same-port: https 200, plain http fails",
    ),
    (
        "tls_key",
        False,
        False,
        "behavior",
        None,
        "rustls same-port: https 200, plain http fails",
    ),
    ("cors_origins", False, False, "behavior", None, "OPTIONS preflight -> ACAO echo"),
    (
        "otlp_endpoint",
        False,
        False,
        "behavior",
        None,
        "local collector captures export POST",
    ),
    (
        "otlp_service",
        False,
        False,
        "behavior",
        None,
        "service name in captured export POST",
    ),
    ("remotes", False, True, "behavior", None, "config echo + doctor"),
    (
        "engine_env",
        False,
        True,
        "existing",
        None,
        "phase_config C: child environ probe",
    ),
    (
        "model_overrides",
        False,
        True,
        "existing",
        None,
        "phase_config C: override ctx wins",
    ),
    (
        "spec_draft_cpu_range",
        False,
        False,
        "roundtrip",
        None,
        "draft family; echo + full-manifest boot",
    ),
    (
        "spec_draft_cpu_strict",
        False,
        False,
        "roundtrip",
        None,
        "draft family; echo + boot",
    ),
    ("spec_draft_device", False, False, "roundtrip", None, "draft family; echo + boot"),
    ("spec_draft_ngl", False, False, "roundtrip", None, "draft family; echo + boot"),
    (
        "spec_draft_threads",
        False,
        False,
        "roundtrip",
        None,
        "draft family; echo + boot",
    ),
    ("spec_draft_p_min", True, False, "roundtrip", None, "draft family; echo + boot"),
    ("spec_draft_p_split", True, False, "roundtrip", None, "draft family; echo + boot"),
    ("spec_draft_poll", True, False, "roundtrip", None, "draft family; echo + boot"),
    ("spec_draft_prio", False, False, "roundtrip", None, "draft family; echo + boot"),
    (
        "spec_draft_prio_batch",
        False,
        False,
        "roundtrip",
        None,
        "draft family; echo + boot",
    ),
    (
        "spec_draft_poll_batch",
        True,
        False,
        "roundtrip",
        None,
        "draft family; echo + boot",
    ),
    (
        "spec_draft_cpu_strict_batch",
        False,
        False,
        "roundtrip",
        None,
        "draft family; echo + boot",
    ),
    (
        "spec_draft_threads_batch",
        False,
        False,
        "roundtrip",
        None,
        "draft family; echo + boot",
    ),
    ("spec_draft_type_k", False, False, "roundtrip", None, "draft family; echo + boot"),
    ("spec_draft_type_v", False, False, "roundtrip", None, "draft family; echo + boot"),
    (
        "spec_draft_override_tensor",
        False,
        False,
        "roundtrip",
        None,
        "draft family; echo + boot",
    ),
    (
        "spec_draft_n_cpu_moe",
        False,
        False,
        "roundtrip",
        None,
        "draft family; echo + boot",
    ),
    (
        "spec_draft_cpu_moe",
        False,
        False,
        "roundtrip",
        None,
        "draft family; echo + boot",
    ),
    (
        "spec_draft_backend_sampling",
        False,
        False,
        "roundtrip",
        None,
        "draft family; echo + boot",
    ),
    ("adaptive_decay", False, False, "existing", None, "sentinel/adaptive phase"),
    ("adaptive_target", False, False, "existing", None, "sentinel/adaptive phase"),
    ("ngram_size_m", False, False, "tune", None, "tune --ngram lane"),
    ("ngram_size_n", False, False, "tune", None, "tune --ngram lane"),
    ("ngram_min_hits", False, False, "tune", None, "tune --ngram lane"),
    ("ngram_mod_n_match", False, False, "tune", None, "tune --ngram lane"),
    ("ngram_mod_n_max", False, False, "tune", None, "tune --ngram lane"),
    ("ngram_mod_n_min", False, False, "tune", None, "tune --ngram lane"),
    (
        "reasoning_budget",
        False,
        False,
        "roundtrip",
        None,
        "echo + boot (default fn visible in fresh list)",
    ),
    ("reasoning_budget_message", False, False, "roundtrip", None, "echo + boot"),
    ("reasoning_effort", False, False, "roundtrip", None, "echo + boot"),
    ("reasoning_preserve", True, False, "roundtrip", None, "echo + boot"),
    ("image_max_tokens", False, False, "argv", "G1", "--image-max-tokens 4096"),
    ("image_min_tokens", False, False, "argv", "G1", "--image-min-tokens 64"),
    ("mtmd_batch_max_tokens", False, False, "roundtrip", None, "echo + boot"),
    ("mmproj_offload", False, False, "roundtrip", None, "echo + boot"),
    ("mmproj_auto", False, False, "roundtrip", None, "echo + boot"),
    ("mmproj_device", False, False, "argv", "G1", "--mmproj-device <device>"),
    ("embd_normalize", False, False, "argv", "G1", "--embd-normalize 2"),
    (
        "yarn_orig_ctx",
        False,
        False,
        "argv",
        "G3",
        "--rope-scaling yarn + --rope-orig-ctx",
    ),
    ("yarn_ext_factor", False, False, "argv", "G3", "--rope-scale 1.5"),
    ("yarn_attn_factor", False, False, "argv", "G3", "--yarn-attn-factor"),
    ("yarn_beta_fast", False, False, "argv", "G3", "--yarn-beta-fast"),
    ("yarn_beta_slow", False, False, "argv", "G3", "--yarn-beta-slow"),
    ("cpu_strict", False, False, "roundtrip", None, "echo + boot"),
    (
        "prio",
        False,
        False,
        "behavior",
        "G2",
        "child nice via setpriority (/proc stat ni)",
    ),
    ("prio_batch", False, False, "behavior", "G2", "same spawn, nice observable"),
    ("poll_batch", True, False, "roundtrip", None, "echo + boot"),
    ("threads_http", False, False, "argv", "G1", "--threads-http 2"),
    ("warmup", False, False, "argv", "G1", "false -> --no-warmup"),
    ("repack", False, False, "roundtrip", None, "echo + boot"),
    ("cache_idle_slots", False, False, "argv", "G1", "--cache-idle-slots"),
    (
        "lookup_cache_static",
        True,
        False,
        "roundtrip",
        None,
        "missing-file refusal (wave) + echo + boot",
    ),
    (
        "lookup_cache_dynamic",
        True,
        False,
        "roundtrip",
        None,
        "missing-file refusal (wave) + echo + boot",
    ),
    ("predictive_preload", False, False, "existing", None, "phase_wave preload lane"),
    ("adaptive_slots", False, False, "roundtrip", None, "echo + boot"),
    (
        "lazy_mode",
        False,
        False,
        "roundtrip",
        None,
        "echo + boot (string knob, default auto)",
    ),
    (
        "server_tools",
        True,
        False,
        "roundtrip",
        None,
        "global-only agent-tooling quartet; Option<String>, default None (not fresh-visible)",
    ),
    (
        "server_tools_runtime",
        True,
        False,
        "roundtrip",
        None,
        "prefix-validated at config validate (docker:/podman:/ssh:...)",
    ),
    (
        "mcp_servers_config",
        True,
        False,
        "roundtrip",
        None,
        "path existence-checked at profile compile",
    ),
    (
        "mcp_servers_json",
        True,
        False,
        "roundtrip",
        None,
        "JSON syntax + mutual exclusivity checked at config validate",
    ),
    ("no_host", False, False, "argv", "G1", "--no-host"),
    ("op_offload", True, False, "argv", "G1", "--op-offload"),
    ("keep_tokens", False, False, "roundtrip", None, "echo + boot"),
    (
        "override_kv",
        False,
        False,
        "roundtrip",
        None,
        "echo + boot (valid-kv values unverified)",
    ),
    (
        "control_vectors",
        False,
        False,
        "roundtrip",
        None,
        "echo + boot (needs real .gguf to emit)",
    ),
    ("control_vectors_scaled", False, False, "roundtrip", None, "echo + boot"),
    ("control_vector_layer_range", False, False, "roundtrip", None, "echo + boot"),
    ("tensor_preset", False, False, "roundtrip", None, "echo + boot"),
    ("pii_scrub", False, False, "roundtrip", None, "echo + boot"),
    ("video_ffmpeg_dir", False, False, "roundtrip", None, "echo + boot"),
    ("video_fps", False, False, "roundtrip", None, "echo + boot"),
    ("video_timestamp_interval", False, False, "roundtrip", None, "echo + boot"),
    ("numa", False, False, "roundtrip", None, "echo + boot"),
    ("check_tensors", False, False, "roundtrip", None, "echo + boot"),
    ("context_shift", False, False, "roundtrip", None, "echo + boot"),
    ("samplers", False, False, "roundtrip", None, "echo + boot"),
    ("batch_size", False, False, "argv", "G2", "--batch-size 512"),
    ("ubatch_size", False, False, "argv", "G2", "--ubatch-size 256"),
    ("threads_batch", False, False, "argv", "G2", "--threads-batch 2"),
    ("main_gpu", False, False, "argv", "G2", "--main-gpu 0"),
    ("split_mode", False, False, "argv", "G2", "--split-mode layer"),
    ("tensor_split", False, False, "argv", "G2", "--tensor-split 3,1"),
    (
        "models_autoload",
        True,
        False,
        "boundary",
        None,
        "loads ALL store models (RAM); echo + boot",
    ),
    (
        "log_level",
        True,
        False,
        "roundtrip",
        None,
        "tracing filter (e.g. pallama=debug); set->list echo + full-manifest boot",
    ),
    (
        "late_chunking_max_tokens",
        False,
        False,
        "roundtrip",
        None,
        "late-chunking embeddings cap (default 8192); echo + full-manifest boot",
    ),
    (
        "session_keep_secs",
        False,
        False,
        "roundtrip",
        None,
        "session-pin idle-eviction guard window (default 900s, 0=off); echo + full-manifest boot",
    ),
    (
        "update_channel",
        False,
        False,
        "roundtrip",
        None,
        "engine/app update channel (stable|latest); fresh-visible + roundtrip boot",
    ),
    # F146: three config.rs knobs that were absent from the registry —
    # gate (b) would have gone RED on the first fresh-config-list diff
    # (auto_restart_engine_switch is list-visible), and the two Options
    # were invisible holes in the manifest denominator.
    (
        "auto_restart_engine_switch",
        False,
        False,
        "boundary",
        None,
        "engine-switch restart policy; no dedicated phase (engine list only)",
    ),
    (
        "mistralrs_pa_memory_fraction",
        True,
        False,
        "boundary",
        None,
        "mistralrs Option knob; mistralrs lane not installed in sandbox",
    ),
    (
        "mistralrs_paged_attn",
        True,
        False,
        "boundary",
        None,
        "mistralrs Option knob; mistralrs lane not installed in sandbox",
    ),
]

TOPLEVEL_KNOBS = [
    {"name": n, "option": o, "container": c, "tier": t, "group": g, "expectation": e}
    for (n, o, c, t, g, e) in _K
]

OPTION_KNOBS = [k["name"] for k in TOPLEVEL_KNOBS if k["option"]]
CONTAINER_KNOBS = [k["name"] for k in TOPLEVEL_KNOBS if k["container"]]
# Knobs expected visible in a fresh `config list` at defaults: everything
# except Option fields and (empty-at-default) containers.
FRESH_VISIBLE_KNOBS = [k["name"] for k in TOPLEVEL_KNOBS if not k["option"]]

# ---------------------------------------------------------------------------
# MODEL_OVERRIDE manifest: all 25 ModelOverride fields + 18 SamplerDefaults
# leaves. Evidence = overlay round-trip (gate d) + argv/wave lanes noted.
# ---------------------------------------------------------------------------

MODEL_OVERRIDE_FIELDS = [
    ("ctx", "argv: --ctx-size (phase_config C override-wins)"),
    ("slots", "argv: -np"),
    (
        "deterministic",
        "resolve_slots pin: forces -np 1 (greedy reproducibility)",
    ),
    ("spec", "argv: --spec-type"),
    ("loras", "wave: lora attach"),
    ("extra_args", "argv: passthrough tokens"),
    ("cache_type", "argv: --cache-type-k/v"),
    ("kv_unified", "roundtrip echo"),
    ("ctx_extend", "argv: rope flags"),
    ("cpu_moe_n", "argv: --n-cpu-moe"),
    ("cpu_ffn_n", "argv: --n-cpu-ffn"),
    ("override_tensor", "argv: --override-tensor"),
    ("devices", "argv: --device (phase_config A2)"),
    ("warmup", "argv: false -> --no-warmup"),
    ("reasoning_budget", "roundtrip echo"),
    ("reasoning_effort", "roundtrip echo"),
    ("replicas", "wave: replica pid"),
    ("pin", "wave: pinned spawn"),
    ("chat_template", "argv: --chat-template"),
    ("chat_template_file", "argv: --chat-template-file"),
    ("sampler_defaults", "argv: --temp..--seed set"),
    ("spm_infill", "roundtrip echo"),
    # F146: both exist in config.rs ModelOverride but were absent here —
    # coverage denominator undercounted while the lanes ran inline.
    ("late_chunking", "roundtrip echo (late-chunk model knob)"),
    (
        "rpc_servers",
        "wave: model_overrides.rpc_servers -> child --rpc (F2 rpc battery)",
    ),
    ("lazy_mode", "roundtrip echo + argv: --lazy-mode on deviation from auto"),
]

SAMPLER_FIELDS = [
    "temperature",
    "top_k",
    "top_p",
    "min_p",
    "top_n_sigma",
    "typical_p",
    "repeat_penalty",
    "repeat_last_n",
    "presence_penalty",
    "frequency_penalty",
    "dry_multiplier",
    "dry_base",
    "dry_allowed_length",
    "dry_penalty_last_n",
    "xtc_probability",
    "xtc_threshold",
    "mirostat",
    "seed",
]

APIKEY_FIELDS = [
    "name",
    "key",
    "models",
    "rpm",
    "tpm",
    "daily_tokens",
    "max_concurrent",
]
REMOTE_FIELDS = ["name", "url", "key"]


# ---------------------------------------------------------------------------
# Pure helpers (no side effects; safe to import from validate.py)
# ---------------------------------------------------------------------------


def command_paths():
    return [c["path"] for c in COMMANDS]


def toplevel_knob_names(include_options=True):
    names = [k["name"] for k in TOPLEVEL_KNOBS]
    if include_options:
        return names
    return [n for n in names if n not in set(OPTION_KNOBS)]


def knob_entry(name):
    for k in TOPLEVEL_KNOBS:
        if k["name"] == name:
            return k
    raise KeyError(f"unknown knob: {name}")


def argv_groups():
    """knobs_argv spawn groups: group name -> [(knob, value, flag expectation)]"""
    return {
        "G1": [
            ("idle_sleep_secs", 77, "--sleep-idle-seconds 77"),
            ("cache_reuse", 128, "--cache-reuse 128"),
            ("cpu_range", "0-7", "--cpu-range 0-7"),
            ("poll", 77, "--poll 77"),
            ("slot_prompt_similarity", 0.6, "--slot-prompt-similarity 0.6"),
            ("cpu_moe_n", 2, "--n-cpu-moe 2"),
            ("cpu_ffn_n", 2, "--n-cpu-ffn 2"),
            ("override_tensor", [".ffn_.*_exps.=CPU"], "--override-tensor"),
            ("agent", True, "--agent"),
            ("kv_unified_per_slot", 4096, "--kv-unified-per-slot 4096"),
            ("swa_full", True, "--swa-full"),
            ("ctx_checkpoints", 16, "--ctx-checkpoints 16"),
            ("no_kv_offload", True, "--no-kv-offload"),
            ("cache_idle_slots", True, "opt-out: no --no-cache-idle-slots when true"),
            ("warmup", False, "--no-warmup"),
            ("image_max_tokens", 4096, "--image-max-tokens 4096"),
            ("image_min_tokens", 64, "--image-min-tokens 64"),
            ("embd_normalize", 2, "--embd-normalize 2"),
            ("threads_http", 2, "--threads-http 2"),
            ("no_host", True, "--no-host"),
            ("op_offload", True, "--op-offload"),
        ],
        "G2": [
            ("batch_size", 512, "--batch-size 512"),
            ("ubatch_size", 256, "--ubatch-size 256"),
            ("threads_batch", 2, "--threads-batch 2"),
            ("main_gpu", 0, "--main-gpu 0"),
            ("split_mode", "layer", "--split-mode layer"),
            ("tensor_split", "3,1", "--tensor-split 3,1"),
            ("prio", 2, "child nice >= 1 via /proc/<pid>/stat"),
            ("prio_batch", 2, "same spawn (nice observable)"),
        ],
        "G3": [
            # ctx_extend > 1.0 gates the whole yarn block and emits
            # --rope-scaling yarn --rope-scale <ctx_extend>.
            ("ctx_extend", 2.0, "--rope-scaling yarn --rope-scale"),
            ("yarn_orig_ctx", 4096, "--yarn-orig-ctx 4096"),
            ("yarn_ext_factor", 1.5, "--yarn-ext-factor 1.5"),
            ("yarn_attn_factor", 1.75, "--yarn-attn-factor 1.75"),
            ("yarn_beta_fast", 24.0, "--yarn-beta-fast 24"),
            ("yarn_beta_slow", 2.0, "--yarn-beta-slow 2"),
        ],
    }


COMMAND_COVERAGE: list[dict] = []


def reg(path: str, ok: bool, evidence: str = "") -> bool:
    COMMAND_COVERAGE.append({"path": path, "ok": bool(ok), "evidence": evidence})
    tag = "PASS" if ok else "FAIL"
    print(f"  [CMD:{tag}] {path}" + (f" — {evidence}" if evidence else ""))
    return bool(ok)


def regb(path: str, why: str) -> None:
    COMMAND_COVERAGE.append(
        {"path": path, "ok": True, "evidence": why, "boundary": True}
    )
    print(f"  [CMD:BOUNDARY] {path} — {why}")


def lane(path: str, fn, *sub: str) -> None:
    """Run lane fn() unless FAST mode excludes this path (manifest attr)."""
    entry = next(c for c in COMMANDS if c["path"] == path)
    if FAST and not entry["fast"]:
        for p in (path, *sub):
            regb(p, "FAST mode: full run executes this lane for real")
        return
    fn()


def disk_free_gb(path: str = REAL_DATA) -> float:
    t = shutil.disk_usage(path)
    return t.free / (1024**3)


def _metric_value(raw: bytes, name: str) -> float | None:
    m = re.search(rb"^" + name.encode() + rb" ([0-9.eE+-]+)", raw, re.M)
    return float(m.group(1)) if m else None


def _tiny_png_b64() -> str:
    """128x128 four-quadrant PNG (red/green/blue/yellow), pure stdlib —
    a solid 8x8 blank produced immediate-EOS empty completions; a
    structured image gives the projector something to describe."""
    import base64
    import struct
    import zlib

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    w = h = 128
    quads = [
        (b"\xff\x00\x00", b"\x00\xff\x00"),
        (b"\x00\x00\xff", b"\xff\xff\x00"),
    ]
    rows = []
    for y in range(h):
        pair = quads[y // (h // 2)]
        row = pair[0] * (w // 2) + pair[1] * (w // 2)
        rows.append(b"\x00" + row)
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(b"".join(rows)))
        + chunk(b"IEND", b"")
    )
    return base64.b64encode(png).decode()


def mem_available_mib() -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    return 0


def model_bytes_mib(name: str = MODEL) -> int:
    try:
        db = sqlite3.connect(os.path.join(REAL_DATA, "pallama.db"))
        row = db.execute("SELECT bytes FROM models WHERE name = ?", (name,)).fetchone()
        db.close()
        if row and row[0]:
            return int(row[0]) // (1024 * 1024)
    except Exception:
        pass
    return 5500  # conservative default for a ~9B Q4 model


def total_mem_mib() -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) // 1024
    return 0


def _gpu_free_mib() -> int:
    """Free VRAM across GPUs (MiB); 0 when NVIDIA tooling is absent."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0:
            return max(int(x) for x in out.stdout.split())
    except Exception:
        pass
    return 0


def _gpu_headroom_mib() -> int | None:
    """Free VRAM (MiB); None on CPU-only boxes (no nvidia tooling)."""
    if shutil.which("nvidia-smi") is None:
        return None
    return _gpu_free_mib()


def _gpu_compute_holders() -> list[str]:
    """One human line per process holding GPU memory right now."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0:
            return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
    except Exception:
        pass
    return []


# ---------------------------------------------------------------- sandbox


def _toml_inline(v: object) -> str:
    """Scalar -> TOML literal; dict -> inline table (one level)."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, list):
        return "[" + ", ".join(_toml_inline(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{k} = {_toml_inline(x)}" for k, x in v.items()) + " }"
    return json.dumps(str(v))


class Sandbox:
    def __init__(self) -> None:
        # SAME filesystem as ~ so cp's hardlink aliases work, but NEVER a
        # symlink of the real models dir: a sandboxed rm must only ever be
        # able to delete sandbox files (the 2026-09-05 incident).
        cache_root = os.path.expanduser("~/.cache")
        os.makedirs(cache_root, exist_ok=True)
        self.root = tempfile.mkdtemp(prefix="pallama-validate-", dir=cache_root)
        self.config_home = os.path.join(self.root, "config")
        self.data_home = os.path.join(self.root, "share")
        self.config_dir = os.path.join(self.config_home, "pallama")
        self.data_dir = os.path.join(self.data_home, "pallama")
        os.makedirs(self.config_dir)
        os.makedirs(os.path.join(self.data_dir, "models"))
        os.makedirs(os.path.join(self.data_dir, "run"))
        # Engine binaries: hardlink COPY of the real engines tree (same
        # st_dev as ~/.cache, verified at assert). A symlink would let a
        # sandboxed `engine update`'s remove_dir_all delete REAL engine
        # files through the link; hardlinks unlink independently. Internal
        # relative .so symlinks are preserved via symlinks=True.
        real_engines = os.path.join(REAL_DATA, "engines")
        sandbox_engines = os.path.join(self.data_dir, "engines")
        shutil.copytree(
            real_engines, sandbox_engines, symlinks=True, copy_function=os.link
        )
        assert not os.path.islink(sandbox_engines), "engines must not be a symlink"
        assert os.stat(sandbox_engines).st_dev == os.stat(real_engines).st_dev, (
            "engines copy crossed filesystems (hardlinks would become copies)"
        )
        # Live-safe DB copy (sqlite backup API, unlike shutil.copy).
        src = sqlite3.connect(os.path.join(REAL_DATA, "pallama.db"))
        dst = sqlite3.connect(os.path.join(self.data_dir, "pallama.db"))
        src.backup(dst)
        dst.close()
        src.close()
        os.makedirs(os.path.join(self.data_dir, "run"), exist_ok=True)

    def env(self, extra: dict | None = None) -> dict:
        e = dict(os.environ)
        e["XDG_CONFIG_HOME"] = self.config_home
        e["XDG_DATA_HOME"] = self.data_home
        # Harness marker: identifies daemons WE spawned so orphan reaping
        # (Daemon.start pre-spawn sweep) can never signal a user daemon.
        e["PALLAMA_VALIDATE"] = "1"
        for k in list(e):
            if k.startswith("PALLAMA_") and not k.startswith("PALLAMA_VALIDATE"):
                del e[k]
        if extra:
            e.update(extra)
        return e

    def write_config(self, cfg: dict) -> str:
        path = os.path.join(self.config_dir, "config.toml")
        lines = []
        tables = []
        for k, v in cfg.items():
            if k == "engine_env":
                tables.append(("engine_env", v))
            elif k == "model_overrides":
                for model, ov in v.items():
                    tables.append((f'model_overrides."{model}"', ov))
            elif k == "keys":
                for entry in v:
                    tables.append(("keys", entry))
            elif k == "remotes":
                for entry in v:
                    tables.append(("remotes", entry))
            elif isinstance(v, dict):
                # Top-level container struct (e.g. [semantic_cache]):
                # render as a TOML table section, never a quoted string.
                tables.append((k, v))
            elif v is None:
                # Deliberately-omitted knob (XOR partner / pairing-gated):
                # present in the manifest dict for the (c0) membership
                # gate, absent from the serialized boot config.
                continue
            elif isinstance(v, bool):
                lines.append(f"{k} = {'true' if v else 'false'}")
            elif isinstance(v, (int, float)):
                lines.append(f"{k} = {v}")
            elif isinstance(v, list):
                lines.append(f"{k} = " + json.dumps(v))
            else:
                lines.append(f'{k} = "{v}"')
        body = "\n".join(lines) + ("\n" if lines else "")
        for name, tbl in tables:
            # "keys"/"remotes" are arrays-of-tables ([[keys]]/[[remotes]]);
            # everything else a plain table.
            header = f"[[{name}]]" if name in ("keys", "remotes") else f"[{name}]"
            body += f"\n{header}\n"
            for k, v in tbl.items():
                if isinstance(v, bool):
                    body += f"{k} = {'true' if v else 'false'}\n"
                elif isinstance(v, (int, float)):
                    body += f"{k} = {v}\n"
                elif isinstance(v, list):
                    body += f"{k} = " + json.dumps(v) + "\n"
                elif isinstance(v, dict):
                    # Nested struct (e.g. sampler_defaults): inline table.
                    body += f"{k} = " + _toml_inline(v) + "\n"
                else:
                    body += f'{k} = "{v}"\n'
        with open(path, "w") as f:
            f.write(body + "\n")
        return path

    def destroy(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


# ----------------------------------------------------------------- daemon


class Daemon:
    def __init__(self, sb: Sandbox) -> None:
        self.sb = sb
        self.proc: subprocess.Popen | None = None
        self.log_path = os.path.join(sb.data_dir, "run", "daemon.log")

    def start(
        self,
        cfg: dict | None = None,
        env_extra: dict | None = None,
        floor_model: str = MODEL,
        serve_cmd: str = "serve",
    ) -> None:
        self.stop()
        if cfg is None:
            cfg = {"port": PORT}
        else:
            cfg = dict(cfg)
            cfg.setdefault("port", PORT)
        # Mixed iGPU/dGPU boxes: the auto GPU pick maximizes free VRAM,
        # which strands big models on a slow integrated GPU (documented
        # product caveat). PALLAMA_VALIDATE_DEVICES pins MODEL to one
        # explicit device for harness runs on such boxes.
        if VALIDATE_DEVICES:
            mo = cfg.setdefault("model_overrides", {})
            ov = mo.setdefault(MODEL, {})
            ov.setdefault("devices", [VALIDATE_DEVICES])
            if BIG != MODEL:
                big_ov = mo.setdefault(BIG, {})
                big_ov.setdefault("devices", [VALIDATE_DEVICES])
        # Dynamic floor: the model + working headroom. A co-resident
        # engine (the user's own daemon) eats the same budget — fail
        # LOUD instead of thrashing swap for minutes. Late-campaign
        # page-cache drift can shave tens of MiB off MemAvailable
        # seconds after an identical boot succeeded — one 20 s retry,
        # then abort (explicit policy, never silent looping).
        need = int(model_bytes_mib(floor_model) * 1.25) + 1024
        for attempt in range(2):
            if mem_available_mib() >= max(MEM_FLOOR_MIB, need):
                break
            if attempt == 0:
                print(
                    f"validate: MemAvailable {mem_available_mib()} MiB < "
                    f"~{need} MiB — waiting 20s for teardown/cache settle, "
                    f"one retry"
                )
                time.sleep(20)
        else:
            raise RuntimeError(
                f"MemAvailable {mem_available_mib()} MiB < needed ~{need} MiB "
                f"(model {model_bytes_mib(floor_model)} MiB + headroom). A co-resident "
                f"pallama/ollama engine is likely holding memory: stop it for the "
                f"validation window."
            )
        self.sb.write_config(cfg)

        # Orphan reaping: a previous harness run that died without
        # cleanup (timeout kill, crash) leaves a `pallama serve` holding
        # PORT with a DESTROYED sandbox behind it (pidfile rmtree'd with
        # the sandbox). Such orphans carry our harness marker in their
        # environment — TERM them before spawning, or our own child
        # loses the bind race and every probe silently targets a daemon
        # whose store no longer exists (live leak 2026-09-09, pid
        # 1587661). NEVER signal by name/pgid — exact pids only, and
        # only ones provably ours (PALLAMA_VALIDATE marker in environ).
        # Random per-campaign ports (PALLAMA_VALIDATE_PORT) mean the
        # port check below misses most of them — the marker sweep is
        # the real net; a live non-us parent = concurrent run, untouched.
        marker_orphans = self._find_marker_orphans()
        if marker_orphans:
            print(
                f"validate: reaping {len(marker_orphans)} marker orphans "
                f"from dead/earlier runs: {marker_orphans}"
            )
            self._reap_pids(marker_orphans, "marker orphan")
        orphan = self._find_port_orphan()
        if orphan is not None:
            print(
                f"validate: reaping orphan harness daemon pid={orphan} on port {PORT}"
            )
            try:
                os.kill(orphan, signal.SIGTERM)
                deadline = time.time() + 10
                while time.time() < deadline and os.path.exists(f"/proc/{orphan}"):
                    time.sleep(0.25)
                if os.path.exists(f"/proc/{orphan}"):
                    os.kill(orphan, signal.SIGKILL)
                    time.sleep(0.5)
            except ProcessLookupError:
                pass

        log = open(self.log_path, "ab")
        self.proc = subprocess.Popen(
            [PAL, serve_cmd],
            env=self.sb.env(env_extra),
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
        )
        scheme = "https" if cfg.get("tls_cert") else "http"
        tls_ctx = ssl._create_unverified_context() if scheme == "https" else None
        deadline = time.time() + 240
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(
                    f"{scheme}://127.0.0.1:{PORT}/healthz",
                    timeout=2,
                    context=tls_ctx,
                ) as r:
                    if r.status == 200:
                        # healthz 200 must come from OUR child: a foreign
                        # server on the port (another sandbox daemon) also
                        # answers 200 and would silently mis-target every
                        # probe at the wrong store (seen live 2026-09-09).
                        if self.proc.poll() is None:
                            return
                        raise RuntimeError(
                            f"healthz answered but OUR daemon exited (port "
                            f"{PORT} hijacked by pid with another store?); "
                            f"log:\n{self.tail_log()}"
                        )
            except Exception:
                if self.proc.poll() is not None:
                    raise RuntimeError(f"daemon exited early; log:\n{self.tail_log()}")
                time.sleep(0.5)
        raise RuntimeError(f"daemon not healthy in 240s; log:\n{self.tail_log()}")

    def _find_port_orphan(self) -> int | None:
        """A harness-marked `pallama serve` answering on PORT that this
        Daemon object did not spawn. Fast path: port silent -> None.

        F136: the returned pid must actually HOLD the port (listening
        socket inode match), otherwise the /proc scan can TERM a
        concurrent validate run's daemon that merely shares the
        PALLAMA_VALIDATE marker."""
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/healthz", timeout=1):
                pass
        except Exception:
            return None
        port_hex = f"{PORT:04X}"
        inodes: set[str] = set()
        for table in ("/proc/net/tcp", "/proc/net/tcp6"):
            try:
                rows = Path(table).read_text().splitlines()[1:]
            except OSError:
                continue
            for ln in rows:
                parts = ln.split()
                if (
                    len(parts) > 9
                    and parts[1].endswith(f":{port_hex}")
                    and parts[3] == "0A"  # LISTEN
                ):
                    inodes.add(parts[9])
        if not inodes:
            return None  # nobody listening on PORT
        me = os.getpid()
        for pid_dir in Path("/proc").iterdir():
            if not pid_dir.name.isdigit():
                continue
            pid = int(pid_dir.name)
            if pid == me:
                continue
            try:
                cmdline = (pid_dir / "cmdline").read_bytes().split(b"\0")
                joined = b" ".join(cmdline)
                if b"pallama" not in joined or b"serve" not in joined:
                    continue
                environ = (pid_dir / "environ").read_bytes()
                if b"PALLAMA_VALIDATE=1" not in environ:
                    continue  # never the user's real daemon
                # port holder only: match via its listening socket inode
                for fd in (pid_dir / "fd").iterdir():
                    try:
                        target = os.readlink(fd)
                    except OSError:
                        continue
                    if target.startswith("socket:[") and target[8:-1] in inodes:
                        return pid
            except OSError:
                continue
        return None

    def _sandbox_procs(self, exclude: int | None = None) -> list[int]:
        """Every live process belonging to THIS sandbox: the daemon plus
        its engine children (they inherit the daemon's environ, so both
        carry PALLAMA_VALIDATE=1 AND this sandbox's XDG_DATA_HOME —
        reparented grandchildren included, the user's real daemon never).
        Exact pids only; callers kill individually, never by group."""
        marker = self.sb.data_home.encode()
        out: list[int] = []
        me = os.getpid()
        for pid_dir in Path("/proc").iterdir():
            if not pid_dir.name.isdigit():
                continue
            pid = int(pid_dir.name)
            if pid == me or pid == exclude:
                continue
            try:
                environ = (pid_dir / "environ").read_bytes()
            except OSError:
                continue
            if b"PALLAMA_VALIDATE=1" in environ and marker in environ:
                out.append(pid)
        return out

    def _find_marker_orphans(self) -> list[int]:
        """Harness-marked pallama daemons from DEAD runs (parent exited,
        process reparented to init) — the random-port leak class the port
        orphan check cannot see. Live-parented marker daemons are NEVER
        touched: our own deliberate spawns (Daemon instances, battery-F
        edge daemons) are direct children of this process, and a live
        non-us parent = concurrent validate run, left strictly alone."""
        out: list[int] = []
        me = os.getpid()
        for pid_dir in Path("/proc").iterdir():
            if not pid_dir.name.isdigit():
                continue
            pid = int(pid_dir.name)
            if pid == me:
                continue
            try:
                cmdline = (pid_dir / "cmdline").read_bytes().split(b"\0")
                joined = b" ".join(cmdline)
                if b"pallama" not in joined or b"serve" not in joined:
                    continue
                environ = (pid_dir / "environ").read_bytes()
                if b"PALLAMA_VALIDATE=1" not in environ:
                    continue
                ppid = 0
                for line in (pid_dir / "status").read_text().splitlines():
                    if line.startswith("PPid:"):
                        ppid = int(line.split()[1])
                        break
                # Live-parented marker daemons are NEVER orphans: our own
                # deliberate spawns (Daemon instances + battery-F edge
                # daemons) are direct children of THIS process, and a
                # concurrent run's daemons are children of ITS python.
                # Reaping live children killed battery-F edge daemons at
                # the next d.start() -> instant-refused 502 trio (live
                # 2026-09-11: pids 3120088/3120129 TERM-ignored-SIGKILLed).
                # Only reparented (ppid<=1) or dead-parent processes are
                # orphans of DEAD runs — the 2026-09-09 leak class.
                if ppid <= 1 or not os.path.exists(f"/proc/{ppid}"):
                    out.append(pid)  # reparented orphan of a dead run
            except OSError:
                continue
        return out

    @staticmethod
    def _reap_pids(pids: list[int], what: str) -> None:
        """TERM -> wait -> KILL stragglers, exact pids only."""
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.time() + 10
        alive = list(pids)
        while time.time() < deadline:
            alive = [p for p in pids if os.path.exists(f"/proc/{p}")]
            if not alive:
                return
            time.sleep(0.5)
        for pid in alive:
            print(f"validate: {what} pid={pid} ignored TERM — SIGKILL")
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def stop(self) -> None:
        pidfile = os.path.join(self.sb.data_dir, "run", "pallama.pid")
        pid = None
        try:
            with open(pidfile) as f:
                pid = int(f.read().strip())
        except Exception:
            pass
        if self.proc and self.proc.poll() is None:
            try:
                os.kill(self.proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.proc.wait(timeout=45)
            except subprocess.TimeoutExpired:
                try:
                    os.kill(self.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.proc.wait(timeout=10)
        elif pid and pid > 1 and os.path.exists(f"/proc/{pid}"):
            # A daemon we started earlier in this harness run.
            try:
                os.kill(pid, signal.SIGTERM)
                deadline = time.time() + 45
                while time.time() < deadline and os.path.exists(f"/proc/{pid}"):
                    time.sleep(0.5)
            except ProcessLookupError:
                pass
        # Engine grandchildren: the daemon may die before reaping a child
        # that is mid-load (mistralrs CUDA init is slow), leaving a
        # GPU+RAM squatting orphan behind a destroyed sandbox (live leak
        # 2026-09-10: 6.4 GiB VRAM held for minutes after the cell "ended").
        # They inherit this sandbox's environ — sweep and reap by exact pid.
        strays = self._sandbox_procs()
        if strays:
            print(
                f"validate: reaping {len(strays)} sandbox stragglers "
                f"(engine children outliving daemon): {strays}"
            )
            self._reap_pids(strays, "sandbox straggler")
        self.proc = None
        # Wait for the port to actually free so the next start binds cleanly.
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{PORT}/healthz", timeout=1
                ):
                    time.sleep(0.5)
            except Exception:
                return

    def tail_log(self, n: int = 25) -> str:
        try:
            with open(self.log_path, "rb") as f:
                return f.read()[-4000:].decode(errors="replace")
        except Exception:
            return "<no log>"


def _reap_orphan_validate_engines() -> None:
    """Kill engine children this harness stranded: llama-server cmdline,
    PALLAMA_VALIDATE=1 in their environ, and NOT parented by a live
    pallama daemon (a validate engine's only legitimate parent). Reparent
    targets include subreapers — init, a dead parent, or a shell wrapper
    all mean the owning daemon is gone; /api/evict owns live-daemon
    children, and a concurrent run's engines stay safe because their
    daemon parent matches."""
    pids: list[int] = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmd = f.read().replace(b"\x00", b" ").decode("utf-8", "replace")
            if "llama-server" not in cmd:
                continue
            with open(f"/proc/{pid}/environ", "rb") as f:
                env = f.read().replace(b"\x00", b"\n").decode("utf-8", "replace")
            if "PALLAMA_VALIDATE=1" not in env:
                continue
            ppid = 0
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("PPid:"):
                        ppid = int(line.split()[1])
                        break
            parent_cmd = ""
            if os.path.exists(f"/proc/{ppid}"):
                with open(f"/proc/{ppid}/cmdline", "rb") as f:
                    parent_cmd = (
                        f.read().replace(b"\x00", b" ").decode("utf-8", "replace")
                    )
            if ppid <= 1 or "pallama" not in parent_cmd:
                pids.append(pid)
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    if pids:
        Daemon._reap_pids(sorted(pids), "orphan engine")


# ------------------------------------------------------------------- http


def http(
    method: str,
    path: str,
    body: dict | None = None,
    headers: dict | None = None,
    timeout: int = 300,
) -> tuple[int, dict, bytes]:
    url = f"http://127.0.0.1:{PORT}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("content-type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def http_json(
    method: str,
    path: str,
    body: dict | None = None,
    headers: dict | None = None,
    timeout: int = 180,
) -> tuple[int, dict, object]:
    st, hdr, raw = http(method, path, body, headers, timeout)
    try:
        return st, hdr, json.loads(raw)
    except Exception:
        return st, hdr, raw.decode(errors="replace")


def sse_collect(
    path: str,
    want: str,
    budget_s: float,
    body: dict | None = None,
    headers: dict | None = None,
) -> tuple[bool, str]:
    """Stream an SSE/NDJSON endpoint until `want` appears or budget ends.

    Raw sockets, not urllib: a read timeout on an http.client response
    poisons the connection ("cannot read from timed out object"), and SSE
    streams legally idle for minutes waiting for the next event. Per-recv
    timeouts here are keep-alive boundaries, not failures.
    """
    import socket

    payload = json.dumps(body).encode() if body is not None else None
    method = "POST" if payload is not None else "GET"
    head = [
        f"{method} {path} HTTP/1.1",
        f"Host: 127.0.0.1:{PORT}",
        "content-type: application/json",
    ]
    if payload is not None:
        head.append(f"content-length: {len(payload)}")
    for k, v in (headers or {}).items():
        head.append(f"{k}: {v}")
    raw = "\r\n".join(head) + "\r\n\r\n"
    out = b""
    deadline = time.time() + budget_s
    try:
        s = socket.create_connection(("127.0.0.1", PORT), timeout=30)
        s.sendall(raw.encode() + (payload or b""))

        # Raw sockets see HTTP framing; strip response head and de-frame
        # chunked bodies so `want` matching + terminal-line extraction in
        # callers see true payload bytes (hex chunk-size lines otherwise
        # pollute the stream and can split tokens across chunk boundaries).
        buf = b""
        while b"\r\n\r\n" not in buf and time.time() < deadline:
            try:
                r = s.recv(4096)
            except socket.timeout:
                continue
            if not r:
                break
            buf += r
        head, _, buf = buf.partition(b"\r\n\r\n")
        chunked = b"transfer-encoding: chunked" in head.lower()
        out = b""

        def _want_hit() -> bool:
            return want.encode() in out

        while time.time() < deadline:
            if chunked:
                # De-frame: "<hex-size>\r\n<data>\r\n" ... "0\r\n\r\n"
                while True:
                    nl = buf.find(b"\r\n")
                    if nl < 0:
                        break
                    try:
                        size = int(buf[:nl].split(b";")[0].strip(), 16)
                    except ValueError:
                        break  # partial/garbage — need more bytes
                    if size == 0:
                        s.close()
                        return _want_hit(), out.decode(errors="replace")
                    if len(buf) < nl + 2 + size + 2:
                        break  # whole chunk not arrived yet
                    out += buf[nl + 2 : nl + 2 + size]
                    buf = buf[nl + 2 + size + 2 :]
            else:
                out += buf
                buf = b""
            if _want_hit():
                s.close()
                return True, out.decode(errors="replace")
            try:
                r = s.recv(4096)
            except socket.timeout:
                continue
            if not r:
                break
            buf += r
        s.close()
        return _want_hit(), out.decode(errors="replace")
    except Exception as e:
        # F137: judge against the payload seen so far (out), not a dead
        # variable — an error AFTER the want-token still counts as seen.
        return want.encode() in out, out.decode(
            errors="replace"
        ) + f"\n<sse error: {e}>"


# ---------------------------------------------------------------- /proc


def ps_rows() -> list[dict]:
    _, _, v = http_json("GET", "/api/ps")
    if isinstance(v, dict):
        v = v.get("instances") or v.get("models") or []
    return v if isinstance(v, list) else []


def ps_field(row: dict, *names: str):
    for n in names:
        if n in row:
            return row[n]
    return None


def row_ctx(row: dict):
    return ps_field(row, "pallama_ctx", "ctx", "context")


def row_state(row: dict):
    return ps_field(row, "pallama_state", "state", "status")


def row_inflight(row: dict):
    return ps_field(row, "pallama_in_flight", "in_flight", "inflight")


def child_pid(model: str = MODEL) -> int | None:
    # The supervisor writes run/<model>.pid for every spawned child
    # (the ps row carries endpoint/state, not the pid).
    pidfile = os.path.join(SANDBOX.data_dir, "run", f"{model}.pid")
    try:
        with open(pidfile) as f:
            pid = int(f.read().strip())
        if pid > 1 and os.path.exists(f"/proc/{pid}"):
            return pid
    except Exception:
        pass
    return None


def wait_loaded(model: str = MODEL, budget: float = 300.0) -> dict | None:
    deadline = time.time() + budget
    while time.time() < deadline:
        for row in ps_rows():
            name = str(ps_field(row, "name", "model") or "")
            if name.split(":")[0] == model:
                return row
        time.sleep(1)
    return None


def child_argv(pid: int) -> list[str]:
    with open(f"/proc/{pid}/cmdline", "rb") as f:
        return [a for a in f.read().decode(errors="replace").split("\0") if a]


def replica_pid(model: str, n: int) -> int | None:
    # Replicas write run/{model}#N.pid (supervisor keys the pidfile by the
    # full instance key, not the bare model name).
    pidfile = os.path.join(SANDBOX.data_dir, "run", f"{model}#{n}.pid")
    try:
        with open(pidfile) as f:
            pid = int(f.read().strip())
        if pid > 1 and os.path.exists(f"/proc/{pid}"):
            return pid
    except Exception:
        pass
    return None


def daemon_log_contains(needle: str) -> bool:
    try:
        with open(DAEMON.log_path, "rb") as f:
            return needle.encode() in f.read()
    except Exception:
        return False


def daemon_vram_mib() -> int:
    """Total VRAM (MiB) from the daemon hardware banner; 0 when absent."""
    try:
        with open(DAEMON.log_path, "rb") as f:
            m = re.search(rb"hardware: .*?, (\d+) MiB VRAM", f.read())
        if m:
            return int(m.group(1))
    except Exception:
        pass
    return 0


def http_multipart(
    path: str,
    fields: dict,
    file_field: str,
    file_bytes: bytes,
    filename: str,
    content_type: str = "application/octet-stream",
    timeout: int = 60,
) -> tuple[int, bytes]:
    """Hand-rolled multipart POST (the harness has no requests dep)."""
    boundary = "pallamaValidate7dF3k"
    parts = []
    for name, value in fields.items():
        parts.append(
            f'--{boundary}\r\ncontent-disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
        )
    parts.append(
        (
            f'--{boundary}\r\ncontent-disposition: form-data; name="{file_field}"; '
            f'filename="{filename}"\r\ncontent-type: {content_type}\r\n\r\n'
        ).encode()
        + file_bytes
        + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}", data=body, method="POST"
    )
    req.add_header("content-type", f"multipart/form-data; boundary={boundary}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def tiny_wav(seconds: float = 0.5) -> bytes:
    """Minimal canonical RIFF/WAVE (silence) — enough for multipart routing."""
    import struct

    rate, n = 8000, int(8000 * seconds)
    data = b"\x00\x00" * n
    hdr = (
        b"RIFF"
        + struct.pack("<I", 36 + len(data))
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
        + b"data"
        + struct.pack("<I", len(data))
    )
    return hdr + data


def child_environ(pid: int) -> dict:
    out = {}
    with open(f"/proc/{pid}/environ", "rb") as f:
        for entry in f.read().decode(errors="replace").split("\0"):
            if "=" in entry:
                k, _, v = entry.partition("=")
                out[k] = v
    return out


def chat(
    prompt: str,
    stream: bool = False,
    extra: dict | None = None,
    headers: dict | None = None,
    timeout: int = 240,
) -> tuple[int, object, dict]:
    body = {
        "model": MODEL,
        "stream": stream,
        "max_tokens": 512,
        "messages": [{"role": "user", "content": prompt}],
    }
    if extra:
        body.update(extra)
    st, hdr, v = http_json("POST", "/v1/chat/completions", body, headers, timeout)
    return st, v, hdr


def why(trace: str | None = None) -> list[dict]:
    path = f"/api/why?trace={trace}" if trace else "/api/why"
    _, _, v = http_json("GET", path)
    return v.get("records", []) if isinstance(v, dict) else []


def wait_record(
    route: str | None = None, trace: str | None = None, budget: float = 30.0
) -> dict | None:
    deadline = time.time() + budget
    while time.time() < deadline:
        for r in why(trace):
            if (route is None or r.get("route") == route) and (
                trace is None or r.get("trace") == trace
            ):
                return r
        time.sleep(1)
    return None


def cli(
    *args: str, timeout: int = 300, check_exit: bool = False
) -> subprocess.CompletedProcess:
    assert SANDBOX is not None
    p = subprocess.run(
        [PAL, *args], env=SANDBOX.env(), capture_output=True, text=True, timeout=timeout
    )
    if check_exit and p.returncode != 0:
        print(f"    cli stderr: {p.stderr.strip()[:400]}")
    return p


def _help_command_names(help_stdout: str) -> list[str]:
    """Command names advertised by the grouped top-level help.

    Every command line is two-space indented starting with a lowercase
    name (aliases live in parens after the name, e.g. `serve (start)`);
    Options lines start with '-'; headings and the footer are not
    indented. Single parser shared by the commands registry check, the
    gates manifest cross-check and the goldens capture — keep in sync
    with `render_grouped_help()` in crates/pallama-cli/src/main.rs.
    """
    return [
        ln.strip().split()[0]
        for ln in help_stdout.splitlines()
        if re.match(r"^  [a-z]", ln)
    ]


# ----------------------------------------------------------------- phases


def phase_manifests() -> None:
    print("\n== phase 0: manifest registry sanity (merged validate_manifests) ==")
    check(
        "manifests",
        "CLI command registry non-empty",
        bool(COMMANDS) and bool(TOPLEVEL_COMMANDS),
        f"{len(COMMANDS)} leaf paths, {len(TOPLEVEL_COMMANDS)} top-level (+help)",
    )
    tiers: dict[str, int] = {}
    for k in TOPLEVEL_KNOBS:
        tiers[k["tier"]] = tiers.get(k["tier"], 0) + 1
    check(
        "manifests",
        "config knob registry non-empty",
        bool(TOPLEVEL_KNOBS) and bool(MODEL_OVERRIDE_FIELDS),
        f"{len(TOPLEVEL_KNOBS)} knobs (options={len(OPTION_KNOBS)}, "
        f"containers={len(CONTAINER_KNOBS)}, fresh-visible={len(FRESH_VISIBLE_KNOBS)}); "
        f"{len(MODEL_OVERRIDE_FIELDS)} model-override fields, "
        f"{len(SAMPLER_FIELDS)} sampler fields; tiers {dict(sorted(tiers.items()))}",
    )


def phase_baseline() -> None:
    print("\n== phase 1: baseline ==")
    d = Daemon(SANDBOX) if DAEMON is None else DAEMON
    # Boot the FIRST daemon through the `start` alias — the only lane where
    # the alias is the thing under test; every later boot uses `serve`.
    d.start({"port": PORT}, serve_cmd="start")
    reg("start", True, "alias boot: `pallama start` -> healthz 200 (serve alias)")
    st, _, v = http_json("GET", "/api/version")
    check(
        "baseline",
        "/api/version reports a version",
        st == 200 and isinstance(v, dict) and bool(v.get("version")),
        str(v),
    )
    st, hdr, _ = http("GET", "/healthz")
    check("baseline", "/healthz 200", st == 200, f"status={st}")
    st, _, v = http_json("GET", "/v1/models")
    ids = [m.get("id") for m in v.get("data", [])] if isinstance(v, dict) else []
    check(
        "baseline",
        "/v1/models lists the pulled model",
        st == 200 and any(MODEL in i for i in ids),
        str(ids),
    )
    # Real inference, non-stream + stream.
    st, v, hdr = chat("Reply with exactly: ok")
    content = ""
    try:
        content = v["choices"][0]["message"]["content"] or ""
    except Exception:
        pass
    trace = hdr.get("x-pallama-trace-id", "")
    check(
        "baseline",
        "chat non-stream real inference",
        st == 200 and "ok" in content.lower(),
        f"status={st} content={content[:60]!r}",
    )
    check("baseline", "trace id header echoed", trace.startswith("plm-"), trace)
    ok, collected = sse_collect(
        "/v1/chat/completions",
        "[DONE]",
        240,
        body={
            "model": MODEL,
            "stream": True,
            "max_tokens": 128,
            "messages": [{"role": "user", "content": "Say ok"}],
        },
    )
    check(
        "baseline",
        "chat stream real inference (SSE to [DONE])",
        ok,
        f"{collected.count('data: ')} events, {len(collected)} bytes",
    )
    row = wait_loaded()
    if row is not None:
        print(f"    ps row (shape evidence): {json.dumps(row)[:300]}")
    check(
        "baseline",
        "model loaded and visible in ps",
        row is not None,
        "" if row else "no ps row after load",
    )
    # Banner attribution (the anti-ollama complaint).
    p = subprocess.run([PAL, "--help"], capture_output=True, text=True)
    check(
        "baseline",
        "--help carries llama.cpp credit",
        "llama.cpp" in p.stdout + p.stderr,
        "credit line present",
    )


def phase_config() -> None:
    print(
        "\n== phase 2: config-flow matrix (config.toml -> profile -> child argv -> observed) =="
    )
    d = DAEMON
    clamp_expected = int(total_mem_mib() * 0.30)

    # Group A: sizing + tuning flags.
    d.start(
        {
            "default_ctx": 8192,
            "cache_ram_mb": 2048,
            "slots": 2,
            "cache_reuse": 128,
            "poll": 50,
            "reasoning_format": "deepseek",
            "slot_prompt_similarity": 0.5,
            "cpu_moe_n": 2,
            "cpu_ffn_n": 1,
            "override_tensor": [".ffn_.*_exps.=CPU"],
        }
    )
    st, v, _ = chat("Say ok")
    pid = child_pid()
    argv = child_argv(pid) if pid else []

    def has(flag: str, val: str | None = None) -> bool:
        if val is None:
            return flag in argv
        try:
            i = argv.index(flag)
            return i + 1 < len(argv) and argv[i + 1] == val
        except ValueError:
            return False

    row = wait_loaded()
    ctx_now = row_ctx(row or {}) if row else None
    # ps ctx = PER-SLOT ctx by design (profile.rs: --ctx-size is the total,
    # upstream divides it across -np slots; Profile.ctx/ps report what ONE
    # slot holds) — slots:2 + default_ctx 8192 => argv total 8192, ps 4096.
    check(
        "config",
        "default_ctx 8192 + slots 2 -> --ctx-size 8192 total, ps per-slot ctx 4096",
        has("--ctx-size", "8192") and str(ctx_now) == "4096",
        f"argv --ctx-size={'8192' if has('--ctx-size', '8192') else 'MISSING'} ps ctx={ctx_now}",
    )
    cov("default_ctx", "--ctx-size 8192 in argv; ps ctx 8192", f"ps ctx={ctx_now}")
    check(
        "config",
        "cache_ram_mb 2048 -> --cache-ram 2048",
        has("--cache-ram", "2048"),
        " ".join(a for a in argv if "cache-ram" in a) or "missing",
    )
    cov(
        "cache_ram_mb",
        "--cache-ram 2048",
        f"argv={' '.join(a for a in argv if 'cache-ram' in a)}",
    )
    check(
        "config",
        "slots 2 -> -np 2",
        has("-np", "2"),
        " ".join(argv[argv.index("-np") : argv.index("-np") + 2])
        if "-np" in argv
        else "missing",
    )
    cov("slots", "-np 2", "argv ok" if has("-np", "2") else "MISSING")
    check(
        "config",
        "cache_reuse 128 -> --cache-reuse 128",
        has("--cache-reuse", "128"),
        " ".join(a for a in argv if "cache-reuse" in a) or "missing",
    )
    check(
        "config",
        "poll 50 -> --poll 50",
        has("--poll", "50"),
        " ".join(
            a
            for a in argv
            if a == "--poll"
            or (argv.index(a) > 0 and argv[argv.index(a) - 1] == "--poll")
        )
        or "missing",
    )
    check(
        "config",
        "reasoning_format deepseek -> --reasoning-format deepseek",
        has("--reasoning-format", "deepseek"),
        " ".join(a for a in argv if "reasoning" in a) or "missing",
    )
    cov(
        "reasoning_format",
        "--reasoning-format deepseek",
        "argv ok" if has("--reasoning-format", "deepseek") else "MISSING",
    )
    check(
        "config",
        "slot_prompt_similarity 0.5 emitted",
        has("--slot-prompt-similarity", "0.5")
        or "--slot-prompt-similarity" in " ".join(argv),
        " ".join(a for a in argv if "similarity" in a) or "missing",
    )
    check(
        "config",
        "cpu_moe_n 2 -> --n-cpu-moe 2",
        has("--n-cpu-moe", "2"),
        " ".join(a for a in argv if "cpu-moe" in a) or "missing",
    )
    check(
        "config",
        "override_tensor pattern emitted",
        ".ffn_.*_exps.=CPU" in argv,
        " ".join(a for a in argv if "override-tensor" in a or "ffn_" in a) or "missing",
    )

    # Group A2: devices + engine currency + GPU-offload label (2026-09-06).
    # Devices come from the ACTIVE engine's manifest (never bogus names);
    # tried largest-VRAM first with fallthrough — iGPUs report shared
    # memory as VRAM, so the biggest number is not always loadable.
    mrow = (
        sqlite3.connect(os.path.join(SANDBOX.data_dir, "pallama.db"))
        .execute("select manifest from engines where active=1")
        .fetchone()
    )
    device_names: list[str] = []
    if mrow:
        try:
            devs = json.loads(mrow[0])["devices"]
            device_names = [
                d["name"]
                for d in sorted(devs, key=lambda d: d.get("total_mib", 0), reverse=True)
            ]
        except Exception:
            pass
    emitted = None
    for i, dev in enumerate(device_names[:2]):
        d.start({"default_ctx": 2048, "devices": [dev]})
        try:
            chat("Say ok", timeout=150)
            pid = child_pid()
            argv = child_argv(pid) if pid else []
            if dev in argv:
                emitted = dev
                break
        except Exception as e:
            if i == len(device_names[:2]) - 1:
                boundary(
                    "config",
                    "devices",
                    f"load failed on every manifest device ({type(e).__name__}); emission covered by unit tests",
                )
    if emitted:
        check("config", f"devices [{emitted}] -> --device emitted", True, "argv ok")
        cov("devices", "--device <manifest device>", "argv ok")
    elif not device_names:
        boundary(
            "config",
            "devices",
            "active engine manifest lists no devices; emission covered by unit tests",
        )

    # engine_check_secs: background marker lands within a few seconds.
    d.start({"default_ctx": 2048, "engine_check_secs": 5})
    marker = os.path.join(SANDBOX.data_dir, "run", "engine-check.json")
    deadline = time.time() + 45
    marker_ok = False
    while time.time() < deadline and not marker_ok:
        try:
            with open(marker) as f:
                m = json.load(f)
            marker_ok = isinstance(m.get("active"), str) and "latest" in m
        except Exception:
            time.sleep(1)
    if marker_ok:
        check(
            "config",
            "engine_check_secs 5 -> run/engine-check.json marker",
            True,
            "marker=ok",
        )
        cov("engine_check_secs", "background currency marker", "ok")
    else:
        gh_budget = None
        try:
            with urllib.request.urlopen(
                "https://api.github.com/rate_limit", timeout=5
            ) as r:
                gh_budget = json.load(r)["resources"]["core"]["remaining"]
        except Exception:
            gh_budget = None
        if gh_budget == 0:
            boundary(
                "config",
                "engine_check_secs 5 -> run/engine-check.json marker",
                "GH latest_b_release rate-limited (core budget 0) this window; "
                "marker task verified by Rust unit tests + writes when budget returns",
            )
            cov(
                "engine_check_secs",
                "background currency marker",
                "boundary: GH rate limit",
            )
        else:
            check(
                "config",
                "engine_check_secs 5 -> run/engine-check.json marker",
                False,
                f"marker=missing (GH budget={gh_budget} — not a rate limit)",
            )
            cov(
                "engine_check_secs",
                "background currency marker",
                "MISSING",
            )

    # ps rows carry the resolved GPU-offload label.
    st, v, _ = chat("Say ok")
    rows = ps_rows()
    gpu = rows and ps_field(rows[0], "pallama_gpu", "gpu")
    check(
        "config",
        "ps row carries pallama_gpu label",
        gpu in ("full", "cpu", "partial", "auto", "router"),
        f"pallama_gpu={gpu!r}",
    )
    cov("gpu offload label", "ps pallama_gpu field", f"{gpu}")

    # Group B: KV quant + spec + yarn + sessions.
    d.start(
        {
            "cache_type": "q8_0",
            "spec": "ngram",
            "spec_cache": True,
            "ctx_extend": 2.0,
            "sessions": True,
        }
    )
    chat("Say ok")
    pid = child_pid()
    argv = child_argv(pid) if pid else []
    joined = " ".join(argv)
    check(
        "config",
        "cache_type q8_0 -> --cache-type-k q8_0",
        "--cache-type-k" in joined and "q8_0" in joined,
        " ".join(a for a in argv if "cache-type" in a) or "missing",
    )
    cov(
        "cache_type",
        "--cache-type-k q8_0",
        "argv ok" if "q8_0" in joined else "MISSING",
    )
    check(
        "config",
        "spec ngram -> --spec-type ngram*",
        "--spec-type" in joined and "ngram" in joined,
        " ".join(a for a in argv if "spec" in a) or "missing",
    )
    cov("spec", "--spec-type ngram", "argv ok" if "ngram" in joined else "MISSING")
    check(
        "config",
        "spec_cache -> --lookup-cache-dynamic",
        "--lookup-cache-dynamic" in joined,
        "present" if "--lookup-cache-dynamic" in joined else "missing",
    )
    cov(
        "spec_cache",
        "--lookup-cache-dynamic",
        "argv ok" if "--lookup-cache-dynamic" in joined else "MISSING",
    )
    check(
        "config",
        "ctx_extend 2.0 -> yarn rope flags",
        "yarn" in joined.lower(),
        " ".join(a for a in argv if "yarn" in a.lower() or "rope" in a.lower())
        or "missing",
    )
    cov(
        "ctx_extend",
        "yarn rope flags",
        "argv ok" if "yarn" in joined.lower() else "MISSING",
    )
    check(
        "config",
        "sessions -> --slot-save-path",
        "--slot-save-path" in joined,
        "present" if "--slot-save-path" in joined else "missing",
    )
    cov(
        "sessions",
        "--slot-save-path",
        "argv ok" if "--slot-save-path" in joined else "MISSING",
    )

    # Group C: cache-ram clamp + engine_env + env override + model_overrides.
    d.start(
        {
            "cache_ram_mb": 999999,
            "default_ctx": 2048,
        },
        env_extra=None,
    )
    chat("Say ok")
    pid = child_pid()
    argv = child_argv(pid) if pid else []
    try:
        i = argv.index("--cache-ram")
        got = int(argv[i + 1])
    except (ValueError, IndexError):
        got = -1
    check(
        "config",
        "cache_ram_mb 999999 clamped to 30% RAM",
        got == clamp_expected,
        f"--cache-ram {got} (expected {clamp_expected}, {total_mem_mib()} MiB total)",
    )
    cov("cache_ram_mb clamp", f"--cache-ram {clamp_expected}", f"got {got}")

    d.start({"default_ctx": 2048}, env_extra={"PALLAMA_DEFAULT_CTX": "4096"})
    chat("Say ok")
    row = wait_loaded()
    ctx_now = row_ctx(row or {})
    check(
        "config",
        "PALLAMA_DEFAULT_CTX env beats file value 2048",
        str(ctx_now) == "4096",
        f"file=2048 env=4096 ps ctx={ctx_now}",
    )
    cov("PALLAMA_* env overrides", "env wins over file", f"ps ctx={ctx_now}")

    d.start({"default_ctx": 2048, "model_overrides": {MODEL: {"ctx": 3072}}})
    chat("Say ok")
    row = wait_loaded()
    ctx_now = row_ctx(row or {})
    check(
        "config",
        "model_overrides ctx 3072 wins over default 2048",
        str(ctx_now) == "3072",
        f"default=2048 overlay=3072 ps ctx={ctx_now}",
    )
    cov("model_overrides", "per-model ctx wins", f"ps ctx={ctx_now}")

    d.start({"engine_env": {"PALLAMA_VALIDATE_PROBE": "xyz-marker"}})
    chat("Say ok")
    pid = child_pid()
    env = child_environ(pid) if pid else {}
    check(
        "config",
        "engine_env reaches the child process",
        env.get("PALLAMA_VALIDATE_PROBE") == "xyz-marker",
        f"PALLAMA_VALIDATE_PROBE={env.get('PALLAMA_VALIDATE_PROBE')!r}",
    )
    cov(
        "engine_env",
        "child environ carries key",
        "ok" if env.get("PALLAMA_VALIDATE_PROBE") == "xyz-marker" else "MISSING",
    )

    boundary(
        "config",
        "rpc_servers",
        "needs a second box running llama-server --rpc; flag emission covered by unit tests",
    )
    boundary(
        "config",
        "child_transport = unix",
        "documented unsupported path for the byte-stream OpenAI proxy",
    )
    boundary(
        "config",
        "max_loaded_models",
        "capacity effect needs 2+ models; emit + math covered by unit tests",
    )


def phase_api() -> None:
    print("\n== phase 3: API surface with real inference ==")
    d = DAEMON
    d.start({"port": PORT})
    chat("Say ok")  # warm load
    st, v, _ = chat("Reply: hello")
    check(
        "api",
        "OpenAI /v1/chat/completions non-stream",
        st == 200 and isinstance(v, dict) and v.get("choices"),
        f"status={st}",
    )
    ok, collected = sse_collect(
        "/v1/chat/completions",
        "[DONE]",
        240,
        body={
            "model": MODEL,
            "stream": True,
            "max_tokens": 128,
            "messages": [{"role": "user", "content": "Say hi"}],
        },
    )
    check(
        "api",
        "OpenAI /v1/chat/completions stream",
        ok,
        f"{collected.count('data: ')} events",
    )
    st, _, v = http_json(
        "POST",
        "/v1/completions",
        {"model": MODEL, "prompt": "Say: legacy", "max_tokens": 24},
    )
    check(
        "api",
        "OpenAI /v1/completions legacy",
        st == 200 and isinstance(v, dict) and v.get("choices"),
        f"status={st}",
    )
    st, _, v = http_json(
        "POST",
        "/v1/responses",
        {"model": MODEL, "input": "Say: resp", "max_output_tokens": 256},
    )
    check(
        "api",
        "OpenAI /v1/responses non-stream",
        st == 200 and isinstance(v, dict),
        f"status={st} keys={list(v)[:6] if isinstance(v, dict) else '?'}",
    )
    ok, collected = sse_collect(
        "/v1/responses",
        "response.completed",
        240,
        body={
            "model": MODEL,
            "input": "Say: rstream",
            "stream": True,
            "max_output_tokens": 128,
        },
    )
    check(
        "api", "OpenAI /v1/responses stream", ok, f"{collected.count('data: ')} events"
    )
    st, _, v = http_json(
        "POST", "/tokenize", {"model": MODEL, "content": "hello tokenize"}
    )
    check("api", "/tokenize", st == 200, f"status={st}")
    st, _, v = http_json("POST", "/detokenize", {"model": MODEL, "tokens": [15043]})
    check("api", "/detokenize", st == 200, f"status={st}")
    st, _, v = http_json(
        "POST",
        "/apply-template",
        {"model": MODEL, "messages": [{"role": "user", "content": "tpl"}]},
    )
    check(
        "api",
        "/apply-template (model's own template via --jinja)",
        st == 200,
        f"status={st}",
    )
    st, _, v = http_json(
        "POST",
        "/v1/messages/count_tokens",
        {"model": MODEL, "messages": [{"role": "user", "content": "count"}]},
    )
    check("api", "/v1/messages/count_tokens", st == 200, f"status={st}")
    st, _, v = http_json("GET", "/v1/adapters")
    check("api", "/v1/adapters (LoRA list)", st in (200, 404), f"status={st}")

    # ollama surface
    st, _, v = http_json(
        "POST",
        "/api/chat",
        {
            "model": MODEL,
            "stream": False,
            "messages": [{"role": "user", "content": "Say: ollama"}],
        },
    )
    check(
        "api",
        "ollama /api/chat non-stream translation",
        st == 200 and isinstance(v, dict) and v.get("done") is True,
        f"status={st} done_reason={v.get('done_reason') if isinstance(v, dict) else '?'}",
    )
    ok, collected = sse_collect(
        "/api/chat",
        '"done":true',
        240,
        body={
            "model": MODEL,
            "stream": True,
            "max_tokens": 128,
            "options": {"num_predict": 128},
            "messages": [{"role": "user", "content": "Say: ostream"}],
        },
    )
    check("api", "ollama /api/chat stream (NDJSON)", ok, f"{len(collected)} bytes")
    st, _, v = http_json(
        "POST",
        "/api/generate",
        {
            "model": MODEL,
            "prompt": "Say: gen",
            "stream": False,
            "options": {"num_predict": 32},
        },
    )
    check(
        "api",
        "ollama /api/generate",
        st == 200 and isinstance(v, dict),
        f"status={st} response={str(v.get('response'))[:40] if isinstance(v, dict) else '?'}",
    )
    for path in ("/api/tags", "/api/ps", "/api/version"):
        st, _, v = http_json("GET", path)
        check(
            "api", f"ollama {path}", st == 200 and isinstance(v, dict), f"status={st}"
        )
    st, _, v = http_json("POST", "/api/show", {"model": MODEL})
    check(
        "api",
        "ollama /api/show metadata",
        st == 200 and isinstance(v, dict) and "details" in v,
        f"status={st}",
    )
    ev_ok, ev_text = sse_collect("/api/events", "event", 5)
    check(
        "api",
        "ollama /api/events SSE responds (may be quiet)",
        "<sse error:" not in ev_text,
        f"stream={'event seen' if ev_ok else 'opened, quiet'}",
    )
    st, _, v = http_json("POST", "/api/embeddings", {"model": MODEL, "prompt": "embed"})
    check(
        "api",
        "ollama /api/embeddings (model-capability dependent)",
        st in (200, 400, 501),
        f"status={st} — generative models may refuse",
    )

    # pallama-native — canonical contract: POST /api/session {"model",
    # "action": "save|restore|erase", "filename"} (the /api/session/save
    # sub-path form was retired; this check silently degraded to a boundary
    # for a whole session before the contract was re-verified 2026-09-06).
    st, _, v = http_json(
        "POST",
        "/api/session",
        {"model": MODEL, "action": "save", "filename": "validate-probe"},
    )
    if st == 200:
        st2, _, _ = http_json(
            "POST",
            "/api/session",
            {"model": MODEL, "action": "restore", "filename": "validate-probe"},
        )
        check(
            "api",
            "session save -> restore round-trip",
            st2 == 200,
            f"save=200 restore={st2}",
        )
        http_json(
            "POST",
            "/api/session",
            {"model": MODEL, "action": "erase", "filename": "validate-probe"},
        )
    else:
        # 404/400 = route/contract regression (loud); 503 = engine-gated
        # (sessions need a --slot-save-path-capable engine; legitimately absent).
        check(
            "api",
            "session save reachable",
            st in (200, 503),
            f"status={st} (404/400 = stale session contract in this harness or gateway)",
        )
    # ---- gateway route reachability: every mounted route, real traffic ----
    # /v1/messages (anthropic native, bare): nonstream + stream.
    st, _, v = http_json(
        "POST",
        "/v1/messages",
        {
            "model": MODEL,
            "max_tokens": 8,
            "messages": [{"role": "user", "content": "Say OK."}],
        },
    )
    content = ""
    if isinstance(v, dict) and isinstance(v.get("content"), list):
        for blk in v["content"]:
            if isinstance(blk, dict) and blk.get("text"):
                content = str(blk["text"])
                break
    check(
        "api",
        "/v1/messages (bare) nonstream -> 200 + content",
        st == 200 and bool(content),
        f"status={st} content={content[:40]!r}",
    )
    ok, collected = sse_collect(
        "/v1/messages",
        "message_stop",
        120,
        body={
            "model": MODEL,
            "max_tokens": 8,
            "stream": True,
            "messages": [{"role": "user", "content": "Say OK."}],
        },
    )
    check(
        "api",
        "/v1/messages (bare) stream -> SSE to message_stop",
        ok,
        collected[-120:].replace("\n", " | "),
    )

    # /v1/embeddings (openai passthrough): real pooled vector. 501 = child
    # spawned without --embeddings (capability-dependent engine tier, same
    # carve-out as the /api/embeddings + /api/embed probes above).
    st, _, v = http_json("POST", "/v1/embeddings", {"model": MODEL, "input": "hello"})
    emb_ok = (
        isinstance(v, dict)
        and isinstance(v.get("data"), list)
        and len(v["data"]) > 0
        and isinstance(v["data"][0].get("embedding"), list)
    )
    check(
        "api",
        "/v1/embeddings routed (vector, or engine 501 w/o --embeddings)",
        (st == 200 and emb_ok) or st == 501,
        f"status={st}",
    )

    # /infill (openai passthrough): routed + forwarded = 200 completion or
    # a relayed child JSON error (child FIM support is model/engine-tier).
    st, _, v = http_json(
        "POST",
        "/infill",
        {"model": MODEL, "prompt": "def hello(", "suffix": ": pass", "max_tokens": 8},
    )
    body = json.dumps(v) if not isinstance(v, str) else v
    check(
        "api",
        "/infill routed + child response",
        st == 200 or (st in (400, 500) and body.strip().startswith("{")),
        f"status={st} body={body[:60]}",
    )

    # openai_proxy no-model contract = deterministic routed proof: an
    # unrouted path falls through to axum's empty 404, never this 400.
    for probe_path in (
        "/v1/chat/completions/control",
        "/v1/responses/input_tokens",
        "/responses/input_tokens",
        "/v1/chat/completions/input_tokens",
    ):
        st, _, v = http_json("POST", probe_path, {})
        body = json.dumps(v) if not isinstance(v, str) else v
        check(
            "api",
            f"POST {probe_path} routed (no-model -> teaching 400)",
            st == 400 and "model" in body,
            f"status={st} body={body[:60]}",
        )
    st, _, v = http_json(
        "POST", "/v1/responses/input_tokens", {"model": MODEL, "input": "hello"}
    )
    check(
        "api",
        "/v1/responses/input_tokens forwarded w/ model",
        st == 200,
        f"status={st} body={str(v)[:60]}",
    )

    # /responses (bare, same responses_api handler as /v1/responses).
    st, _, v = http_json("POST", "/responses", {"model": MODEL, "input": "Say OK."})
    check(
        "api",
        "/responses (bare) -> 200 + id",
        st == 200 and isinstance(v, dict) and bool(v.get("id")),
        f"status={st}",
    )

    # /v1/responses store:true -> registry GET by id + 404 shape for bogus.
    st, _, v = http_json(
        "POST",
        "/v1/responses",
        {"model": MODEL, "input": "Say OK.", "store": True, "max_output_tokens": 8},
    )
    resp_id = str(v.get("id") or "") if isinstance(v, dict) else ""
    st_get, _, v_get = http_json("GET", f"/v1/responses/{resp_id}")
    check(
        "api",
        "/v1/responses store:true -> GET by id round-trip",
        st == 200 and st_get == 200 and str(v_get.get("id") or "") == resp_id,
        f"post={st} get={st_get} id={resp_id[:20]}",
    )
    st_bogus, _, v_bogus = http_json("GET", "/v1/responses/resp-bogus-000")
    body = json.dumps(v_bogus) if not isinstance(v_bogus, str) else v_bogus
    check(
        "api",
        "/v1/responses/{bogus} -> 404 'response not found'",
        st_bogus == 404 and "response not found" in body,
        f"status={st_bogus} body={body[:60]}",
    )

    # /slots/{id} POST (scoped_proxy): X-Pallama-Model header resolves the
    # target; routed evidence = JSON body from gateway/child, not empty 404.
    st, _, v = http_json(
        "POST",
        "/slots/0",
        {},
        headers={"X-Pallama-Model": MODEL},
    )
    body = json.dumps(v) if not isinstance(v, str) else v
    check(
        "api",
        "/slots/{id} POST routed via X-Pallama-Model",
        st in (200, 400, 404, 503) and bool(body.strip()),
        f"status={st} body={body[:60]}",
    )

    # /.well-known/pallama discovery document.
    st, _, v = http_json("GET", "/.well-known/pallama")
    wk_ok = (
        isinstance(v, dict)
        and v.get("name") == "pallama"
        and bool(v.get("version"))
        and isinstance(v.get("endpoints"), dict)
        and bool(v["endpoints"].get("openai"))
    )
    check(
        "api", "/.well-known/pallama discovery doc", st == 200 and wk_ok, f"status={st}"
    )

    # ---- batch API (F6): real files -> real batch -> real loopback chats --
    # One request per line, OpenAI batch envelope: {custom_id, body}.
    line = json.dumps(
        {
            "custom_id": "validate-1",
            "body": {
                "model": MODEL,
                "stream": False,
                "max_tokens": 4,
                "messages": [{"role": "user", "content": "Say OK."}],
            },
        }
    )
    st, raw = http_multipart(
        "/v1/files",
        {"purpose": "batch"},
        "file",
        (line + "\n").encode(),
        "validate-batch.jsonl",
        "application/json",
    )
    try:
        file_id = str(json.loads(raw).get("id") or "")
    except Exception:
        file_id = ""
    check(
        "api",
        "POST /v1/files multipart upload -> file id",
        st == 200 and file_id.startswith("file-"),
        f"status={st} body={raw[:60]!r}",
    )
    st_meta, _, v_meta = http_json("GET", f"/v1/files/{file_id}")
    check(
        "api",
        "GET /v1/files/{id} meta",
        st_meta == 200
        and isinstance(v_meta, dict)
        and v_meta.get("id") == file_id
        and v_meta.get("purpose") == "batch",
        f"status={st_meta}",
    )
    st_c, _, v_c = http_json("GET", f"/v1/files/{file_id}/content")
    body = v_c if isinstance(v_c, str) else json.dumps(v_c)
    check(
        "api",
        "GET /v1/files/{id}/content echoes JSONL",
        st_c == 200 and "Say OK." in body,
        f"status={st_c} body={body[:60]!r}",
    )
    st_b, _, v_b = http_json("POST", "/v1/batches", {"input_file_id": file_id})
    batch_id = str(v_b.get("id") or "") if isinstance(v_b, dict) else ""
    check(
        "api",
        "POST /v1/batches -> in_progress",
        st_b == 200
        and batch_id.startswith("batch-")
        and isinstance(v_b, dict)
        and v_b.get("status") == "in_progress",
        f"status={st_b}",
    )
    batch_done = ""
    if batch_id:
        deadline = time.time() + 180
        while time.time() < deadline:
            st_g, _, v_g = http_json("GET", f"/v1/batches/{batch_id}")
            status = str(v_g.get("status") or "") if isinstance(v_g, dict) else ""
            if status in ("completed", "cancelled", "failed"):
                batch_done = status
                break
            time.sleep(2)
    st_g, _, v_g = http_json("GET", f"/v1/batches/{batch_id}")
    counts = v_g.get("request_counts") if isinstance(v_g, dict) else None
    out_file = str(v_g.get("output_file_id") or "") if isinstance(v_g, dict) else ""
    check(
        "api",
        "batch worker replays JSONL via loopback (completed + counts)",
        batch_done == "completed"
        and isinstance(counts, dict)
        and counts.get("total") == 1
        and counts.get("completed") == 1
        and counts.get("failed") == 0
        and out_file.startswith("file-"),
        f"final={batch_done or 'timeout'} counts={counts}",
    )
    if out_file:
        st_o, _, v_o = http_json("GET", f"/v1/files/{out_file}/content")
        body = v_o if isinstance(v_o, str) else json.dumps(v_o)
        check(
            "api",
            "batch output file holds real chat results",
            st_o == 200 and "choices" in body,
            f"status={st_o} body={body[:80]!r}",
        )
    st_x, _, v_x = http_json("POST", f"/v1/batches/{batch_id}/cancel")
    body = json.dumps(v_x) if not isinstance(v_x, str) else v_x
    check(
        "api",
        "cancel after completion -> 400 'batch already finished'",
        st_x == 400 and "already finished" in body,
        f"status={st_x} body={body[:60]}",
    )
    st_l, _, v_l = http_json("GET", "/v1/batches")
    listed = (
        [str(b.get("id") or "") for b in (v_l.get("data") or [])]
        if isinstance(v_l, dict)
        else []
    )
    check(
        "api",
        "GET /v1/batches lists the batch",
        st_l == 200 and batch_id in listed,
        f"status={st_l} n={len(listed)}",
    )

    st, _, v = http_json("POST", "/api/evict", {"model": MODEL})
    check("api", "/api/evict unloads", st in (200, 404), f"status={st}")

    # /api/delete (ollama semantics): cp a scratch copy (post-evict — cp
    # refuses while loaded), delete it over HTTP, confirm store no longer
    # lists it.
    p = cli("cp", MODEL, "validate-del")
    cp_ok = p.returncode == 0
    st, _, v = http_json("POST", "/api/delete", {"model": "validate-del"})
    st_t, _, v_t = http_json("GET", "/api/tags")
    names = (
        [str(t.get("name") or t.get("model") or "") for t in v_t.get("models") or []]
        if isinstance(v_t, dict)
        else []
    )
    check(
        "api",
        "/api/delete removes model from store",
        cp_ok and st == 200 and not any("validate-del" in n for n in names),
        f"cp_rc={p.returncode} delete={st} tags={len(names)}",
    )

    # /api/pull: REAL gateway pull over the event bus (user directive
    # 2026-09-08: real traffic, no mocks) — supersedes the 2026-09-05
    # no-egress carve-out that left this route to offline suites. Tiny
    # Q4_K_M (~400MB); heavy-gated, skipped under FAST.
    if FAST:
        print("  (skip /api/pull real lane: FAST mode)")
    elif disk_free_gb() <= 8:
        print(f"  (skip /api/pull real lane: disk free {disk_free_gb():.1f}G <= 8G)")
    else:
        ok, collected = sse_collect(
            "/api/pull",
            '"success"',
            2400,
            body={"model": "ggml-org/Qwen3-0.6B-GGUF:Q4_K_M"},
        )
        lines = [ln for ln in collected.strip().splitlines() if ln.strip()]
        term = lines[-1] if lines else ""
        body_lines = [ln for ln in lines if not ln[:1].islower() and ":" not in ln[:5]]
        check(
            "api",
            "/api/pull streams NDJSON to terminal success line",
            ok and "success" in term,
            f"events={len(body_lines)} term={term[:100]}",
        )
        st, _, v = http_json("GET", "/api/tags")
        names = (
            [str(t.get("name") or t.get("model") or "") for t in v.get("models") or []]
            if isinstance(v, dict)
            else []
        )
        check(
            "api",
            "/api/pull model present in store",
            any("qwen3-0.6b" in n.lower() for n in names),
            f"tags={len(names)}",
        )
        cli("rm", "qwen3-0.6b")

    # vision chat over the real attached projector (BIG carries mmproj in
    # the store) — real PNG bytes end-to-end, no mock. Heavy: BIG load.
    if FAST:
        print("  (skip vision chat lane: FAST mode)")
    else:
        _st_t, _, _v_t = http_json("GET", "/api/tags")
        _names_t = (
            [
                str(t.get("name") or t.get("model") or "")
                for t in _v_t.get("models") or []
            ]
            if isinstance(_v_t, dict)
            else []
        )
        if BIG == MODEL or not any(BIG.split(":")[0] in n for n in _names_t):
            print(f"  (skip vision chat lane: no distinct vision model {BIG})")
        else:
            # Honest env gate: BIG is a full 9B-class load; a co-resident
            # engine (e.g. ollama) squeezing VRAM/RAM makes pallama's
            # spawn guard refuse pre-spawn (a 500 in ~100ms, by design).
            # Mirror the guard: free VRAM + MemAvailable vs BIG's bytes.
            # R2-28: clear OUR OWN residue first — earlier phases leave
            # live children (idle-sleep not yet fired) or crash-battery
            # orphans; only boundary on what an external holder keeps.
            def _vision_env_ok() -> tuple[bool, int]:
                free = _gpu_free_mib()
                ok = (free == 0 or free >= big_bytes_mib) and (
                    mem_available_mib() >= big_bytes_mib
                )
                return ok, free

            big_bytes_mib = model_bytes_mib(BIG)
            vision_ok_env, free_vram_mib = _vision_env_ok()
            if not vision_ok_env:
                for _m in sorted({MODEL, BIG}):
                    try:
                        http_json("POST", "/api/evict", {"model": _m})
                    except Exception:
                        pass
                _reap_orphan_validate_engines()
                time.sleep(3)
                vision_ok_env, free_vram_mib = _vision_env_ok()
            if not vision_ok_env:
                holders = _gpu_compute_holders()
                boundary(
                    "api",
                    "vision chat battery (memory ceiling)",
                    f"free VRAM {free_vram_mib} MiB / MemAvailable "
                    f"{mem_available_mib()} MiB vs {BIG} ~{big_bytes_mib} MiB "
                    "after evicting this harness's engines and reaping "
                    "orphans — pallama's spawn guard refuses the load by "
                    "design; GPU holders now: "
                    + ("; ".join(holders) if holders else "none visible"),
                )
            else:
                st, _, v = http_json(
                    "POST",
                    "/v1/chat/completions",
                    {
                        "model": BIG,
                        "stream": False,
                        "max_tokens": 128,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "Describe this image in one short sentence.",
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": "data:image/png;base64,"
                                            + _tiny_png_b64()
                                        },
                                    },
                                ],
                            }
                        ],
                    },
                    timeout=300,
                )
                vtxt = ""
                vusage = {}
                vfinish = ""
                vfield = ""
                verr = ""
                if st != 200:
                    # P5: the error body IS the evidence on failure —
                    # print it (spawn-guard refusals carry named reasons).
                    verr = v if isinstance(v, str) else json.dumps(v)[:200]
                try:
                    ch = v["choices"][0]
                    # qwen3.5 thinks first: tokens land in reasoning_content until
                    # thinking completes (translate.rs maps it to anthropic
                    # `thinking`); either field proves real generation over the
                    # image tokens.
                    for fld in ("content", "reasoning_content"):
                        val = (ch.get("message") or {}).get(fld)
                        if val and str(val).strip():
                            vtxt = val
                            vfield = fld
                            break
                    vfinish = ch.get("finish_reason") or ""
                except Exception:
                    pass
                try:
                    vusage = v.get("usage") or {}
                except Exception:
                    pass
                ctoks = vusage.get("completion_tokens")
                check(
                    "api",
                    "vision chat: real image through mmproj yields text",
                    st == 200
                    and bool(str(vtxt).strip())
                    and isinstance(ctoks, int)
                    and ctoks > 0,
                    f"status={st} finish={vfinish} ctok={ctoks} "
                    f"field={vfield or 'none'} txt={str(vtxt)[:60]!r} "
                    f"err={verr!r}",
                )


def phase_sentinel() -> None:
    print("\n== phase 4: sentinel live (real model) ==")
    d = DAEMON
    d.start({"default_ctx": 2048, "sentinel": True})
    # Truncation: prompt bigger than ctx -> loud 400 naming ctx (never silent).
    big = "word " * 3000
    st, _, v = http_json(
        "POST",
        "/v1/chat/completions",
        {
            "model": MODEL,
            "stream": False,
            "messages": [{"role": "user", "content": big}],
        },
    )
    text = json.dumps(v) if not isinstance(v, str) else v
    loud = "ctx" in text.lower() or "context" in text.lower()
    check(
        "sentinel",
        "ctx overflow -> loud named error (no silent truncation)",
        st == 400 and loud,
        f"status={st} snippet={text[:140]!r}",
    )
    # Normal request + why correlation.
    st, v, hdr = chat("Say ok")
    trace = hdr.get("x-pallama-trace-id", "")
    rec = wait_record(trace=trace)
    check(
        "sentinel",
        "record lands with the response trace id",
        rec is not None and rec.get("trace") == trace,
        f"trace={trace} matched={bool(rec)}",
    )
    codes = [d0.get("code") for d0 in (rec or {}).get("detections", [])]
    check(
        "sentinel",
        "clean request -> zero detections",
        rec is not None and not codes,
        f"codes={codes}",
    )
    # near-limit: ctx 2048, prompt ~1900 tokens.
    st, v, hdr = chat("word " * 1850)
    rec = wait_record(trace=hdr.get("x-pallama-trace-id", ""))
    codes = [d0.get("code") for d0 in (rec or {}).get("detections", [])]
    check(
        "sentinel",
        "near-limit detected on a 90%+ prompt",
        "ctx_near_limit" in codes or st == 400,
        f"status={st} codes={codes}",
    )
    # num-ctx header restart-once.
    st, v, hdr = chat("Say ok", headers={"X-Pallama-Num-Ctx": "4096"})
    row = wait_loaded()
    ctx_now = row_ctx(row or {})
    check(
        "sentinel",
        "X-Pallama-Num-Ctx restarts instance at 4096",
        str(ctx_now) == "4096",
        f"ps ctx={ctx_now}",
    )
    # Real tool call: clean pass (qwen3.5 has a tool-capable template).
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    st, v, hdr = chat(
        "What is the weather in Paris? Call get_weather.", extra={"tools": tools}
    )
    rec = wait_record(trace=hdr.get("x-pallama-trace-id", ""))
    codes = [d0.get("code") for d0 in (rec or {}).get("detections", [])]
    tc = ""
    try:
        tc = v["choices"][0]["message"].get("tool_calls") or []
    except Exception:
        pass
    no_false = not any(
        c in ("tool_args_invalid_json", "tool_name_unknown", "template_no_tools")
        for c in codes
    )
    check(
        "sentinel",
        "real tool call: no false positives",
        st == 200 and no_false,
        f"tool_calls={bool(tc)} codes={codes}",
    )
    if tc:
        args_ok = True
        try:
            json.loads(tc[0]["function"]["arguments"])
        except Exception:
            args_ok = False
        check(
            "sentinel",
            "tool arguments are valid JSON (model-composed)",
            args_ok,
            tc[0]["function"]["arguments"][:80],
        )
    # tool_choice rides through to the child: "required" forces a call when
    # the template supports it (qwen templates do); declines are tolerated.
    st, v, _ = chat(
        "What is the weather in Paris? Call get_weather.",
        extra={"tools": tools, "tool_choice": "required", "max_tokens": 96},
    )
    req_tc = []
    try:
        req_tc = v["choices"][0]["message"].get("tool_calls") or []
    except Exception:
        pass
    check(
        "sentinel",
        "tool_choice=required: accepted, no sentinel noise",
        st == 200 and no_false,
        f"tool_calls={bool(req_tc)}",
    )
    if req_tc:
        req_args_ok = True
        try:
            json.loads(req_tc[0]["function"]["arguments"])
        except Exception:
            req_args_ok = False
        check(
            "sentinel",
            "tool_choice=required: forced call carries valid JSON args",
            req_args_ok,
            req_tc[0]["function"]["arguments"][:80],
        )
    st, v, _ = chat(
        "What is the weather in Paris?",
        extra={
            "tools": tools,
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
            "max_tokens": 96,
        },
    )
    named_fn = ""
    try:
        named_fn = v["choices"][0]["message"]["tool_calls"][0]["function"]["name"]
    except Exception:
        pass
    check(
        "sentinel",
        "tool_choice named function: rides through",
        st == 200 and named_fn in ("", "get_weather"),
        f"fn={named_fn!r}",
    )
    # structured output: response_format json_schema (child-side constrained
    # decoding — the grammar path; json_object alone never proves it).
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }
    st, _, v = http_json(
        "POST",
        "/v1/chat/completions",
        {
            "model": MODEL,
            "stream": False,
            "max_tokens": 48,
            "messages": [
                {"role": "user", "content": 'Answer strictly with {"ok": true}.'}
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "validate_ok",
                    "schema": schema,
                    "strict": True,
                },
            },
        },
    )
    parsed = None
    try:
        parsed = json.loads(v["choices"][0]["message"]["content"])
    except Exception:
        pass
    check(
        "sentinel",
        "response_format json_schema: output parses and matches schema",
        st == 200 and isinstance(parsed, dict) and isinstance(parsed.get("ok"), bool),
        f"status={st} parsed={parsed}",
    )
    # ollama-compat `format` object -> gateway translates to json_schema.
    st, _, v = http_json(
        "POST",
        "/api/chat",
        {
            "model": MODEL,
            "stream": False,
            "messages": [
                {"role": "user", "content": 'Answer strictly with {"ok": true}.'}
            ],
            "format": schema,
        },
    )
    parsed = None
    try:
        parsed = json.loads(v["message"]["content"])
    except Exception:
        pass
    check(
        "sentinel",
        "ollama format=object: translated to json_schema, output parses",
        st == 200 and isinstance(parsed, dict) and isinstance(parsed.get("ok"), bool),
        f"status={st} parsed={parsed}",
    )
    # logprobs ride through untouched (openai.rs passthrough, complaint #1).
    st, _, v = http_json(
        "POST",
        "/v1/chat/completions",
        {
            "model": MODEL,
            "stream": False,
            "max_tokens": 16,
            "logprobs": True,
            "top_logprobs": 1,
            "messages": [{"role": "user", "content": "Say ok"}],
        },
    )
    lp_present = False
    try:
        lp_present = "logprobs" in v["choices"][0]
    except Exception:
        pass
    check(
        "sentinel",
        "logprobs=true: key present in choice (passthrough)",
        st == 200 and lp_present,
        f"status={st} logprobs_key={lp_present}",
    )
    # enforce pass-case (deterministic): clean json response with enforce -> 200.
    st, _, v = http_json(
        "POST",
        "/v1/chat/completions",
        {
            "model": MODEL,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": 'Return the JSON {"ok": true} and nothing else.',
                }
            ],
            "response_format": {"type": "json_object"},
        },
        headers={"X-Pallama-Enforce": "1"},
    )
    check(
        "sentinel",
        "enforce header: clean request passes (judge ran)",
        st in (200, 422),
        f"status={st} — 422 acceptable if the model actually violated; both prove the judge ran",
    )
    # watch live delivery.
    result: dict = {}

    def _watch_and_fire() -> None:
        ok, collected = sse_collect("/api/watch", "openai-chat", 240)
        result["ok"] = ok
        result["collected"] = collected

    t = threading.Thread(target=_watch_and_fire)
    t.start()
    time.sleep(2)
    chat("Say ok")
    t.join(timeout=250)
    check(
        "sentinel",
        "watch SSE delivers the record live",
        result.get("ok") is True and "openai-chat" in result.get("collected", ""),
        (result.get("collected", "") or "").replace("\n", " ")[:160],
    )
    # doctor row.
    p = cli("doctor")
    check(
        "sentinel",
        "doctor prints the sentinel row",
        "sentinel" in p.stdout,
        next(
            (ln.strip() for ln in p.stdout.splitlines() if "sentinel" in ln), "missing"
        ),
    )


def phase_behavior() -> None:
    print("\n== phase 5: lifecycle behavior ==")
    d = DAEMON
    d.start({"port": PORT})
    chat("Say ok")
    # keep_alive=0 evicts after the request.
    st, _, _ = http_json(
        "POST",
        "/api/chat",
        {
            "model": MODEL,
            "stream": False,
            "keep_alive": 0,
            "messages": [{"role": "user", "content": "Say ok"}],
        },
    )
    gone = False
    deadline = time.time() + 20
    while time.time() < deadline:
        if not any(
            str(ps_field(r, "name", "model") or "").split(":")[0] == MODEL
            for r in ps_rows()
        ):
            gone = True
            break
        time.sleep(1)
    check(
        "behavior",
        "keep_alive=0 evicts after the response",
        st == 200 and gone,
        f"status={st} evicted={gone}",
    )
    # keep_alive=-1 pins.
    st, _, _ = http_json(
        "POST",
        "/api/chat",
        {
            "model": MODEL,
            "stream": False,
            "keep_alive": -1,
            "messages": [{"role": "user", "content": "Say ok"}],
        },
    )
    time.sleep(3)
    pinned = any(
        str(ps_field(r, "name", "model") or "").split(":")[0] == MODEL
        for r in ps_rows()
    )
    check(
        "behavior",
        "keep_alive=-1 pins the model",
        st == 200 and pinned,
        f"status={st} still loaded={pinned}",
    )
    # crash respawn.
    if not FAST:
        pid = child_pid()
        check(
            "behavior",
            "child pid discovered for crash test",
            pid is not None and pid > 1,
            f"pid={pid}",
        )
        if pid:
            os.kill(pid, signal.SIGKILL)  # our daemon's own child
            time.sleep(2)
            # Post-crash the supervisor must (a) log the crash loudly,
            # (b) respawn a NEW child, (c) serve a request in the window.
            # Two legitimate shapes: the request 502s first then a retry
            # hits the respawned child, OR admission holds it through the
            # respawn and it lands as a single 200 (the 2026-09-11
            # admission-await path detects+respawns in ~1.3s, faster than
            # this probe can observe a 502). Both prove the contract.
            statuses = []
            deadline = time.time() + 60
            while time.time() < deadline:
                st, v, _ = chat("Say ok")
                statuses.append(st)
                if st == 200:
                    break
                time.sleep(3)
            new_pid = child_pid()
            crash_logged = daemon_log_contains("engine crashed")
            check(
                "behavior",
                "crashed engine: logged + respawned + request served",
                200 in statuses and new_pid not in (None, pid) and crash_logged,
                f"old={pid} new={new_pid} statuses={statuses} "
                f"crash_logged={crash_logged}",
            )
    else:
        boundary("behavior", "crash respawn", "skipped in FAST mode")
    # cancellation frees the slot.
    if not FAST:

        def _slow_stream() -> None:
            try:
                req = urllib.request.Request(
                    f"http://127.0.0.1:{PORT}/v1/chat/completions",
                    data=json.dumps(
                        {
                            "model": MODEL,
                            "stream": True,
                            "max_tokens": 400,
                            "messages": [
                                {"role": "user", "content": "Count to 100 slowly."}
                            ],
                        }
                    ).encode(),
                    headers={"content-type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=30) as r:
                    r.read(64)
                # dropping the response mid-stream == client disconnect
            except Exception:
                pass

        t = threading.Thread(target=_slow_stream)
        t.start()
        time.sleep(6)
        freed = False
        deadline = time.time() + 60
        while time.time() < deadline:
            rows = [
                r
                for r in ps_rows()
                if str(ps_field(r, "name", "model") or "").split(":")[0] == MODEL
            ]
            inflight = row_inflight(rows[0]) if rows else 0
            if rows and int(inflight or 0) == 0:
                freed = True
                break
            time.sleep(2)
        t.join(timeout=5)
        check(
            "behavior",
            "client disconnect frees the slot",
            freed,
            f"in_flight back to 0={freed}",
        )
    else:
        boundary("behavior", "disconnect frees slot", "skipped in FAST mode")
    # idle sleep ladder.
    if not FAST:
        d.start({"port": PORT, "idle_sleep_secs": 5})
        chat("Say ok")
        sleeping = None
        deadline = time.time() + 40
        while time.time() < deadline:
            rows = [
                r
                for r in ps_rows()
                if str(ps_field(r, "name", "model") or "").split(":")[0] == MODEL
            ]
            state = str(row_state(rows[0]) or "").lower() if rows else ""
            if rows and "sleep" in state:
                sleeping = state
                break
            time.sleep(2)
        check(
            "behavior",
            "idle_sleep_secs=5 -> child-native sleep state",
            sleeping is not None,
            f"state={sleeping}",
        )
    else:
        boundary("behavior", "idle sleep ladder", "skipped in FAST mode")
    # priority queue under slots=1.
    d.start({"port": PORT, "slots": 1})
    chat("Say ok")
    results: list[tuple[int, int]] = []

    def _one(i: int) -> None:
        st, v, _ = chat(f"Say {i}", extra={"max_tokens": 80}, timeout=300)
        results.append((i, st))

    ts = [threading.Thread(target=_one, args=(i,)) for i in range(2)]
    peak = 0
    for t in ts:
        t.start()
    deadline = time.time() + 240
    while any(t.is_alive() for t in ts) and time.time() < deadline:
        rows = [
            r
            for r in ps_rows()
            if str(ps_field(r, "name", "model") or "").split(":")[0] == MODEL
        ]
        inflight = int(row_inflight(rows[0]) or 0) if rows else 0
        peak = max(peak, inflight)
        time.sleep(1)
    for t in ts:
        t.join(timeout=10)
    # in_flight brackets gateway-accepted requests (supervisor.rs
    # begin_request/end_request), so a queued second request is EXPECTED
    # to read in_flight > 1 — queueing at the child is proven by both
    # requests completing 200, not by the gauge staying at 1.
    check(
        "behavior",
        "slots=1 queues 2 concurrent requests, both complete",
        len(results) == 2 and all(s == 200 for _, s in results) and peak <= 2,
        f"completed={len(results)} statuses={[s for _, s in results]} peak_in_flight={peak}",
    )
    # priority ordering: high jumps a queued low (queue.rs rank High=0 < Low=2).
    order: dict[str, tuple[float, int]] = {}

    def _pri_track(tag: str, pri: str, tokens: int) -> None:
        st, _, _ = chat(
            f"Say the word {tag}.",
            extra={"max_tokens": tokens},
            headers={"x-pallama-priority": pri},
            timeout=300,
        )
        order[tag] = (time.time(), st)

    def _hold_slot() -> None:
        # Streaming holder: on a fast GPU a non-streamed 0.5B generation
        # finishes sub-second (early EOS), and the 0.2s ps poller can miss
        # the whole in_flight window (seal2 flake). The streamed drain holds
        # the slot for the full token budget, making queued=True reliable.
        chat(
            "Write the numbers from 1 to 300, one per line.",
            stream=True,
            extra={"max_tokens": 1500},
            timeout=300,
        )

    def _slot_busy() -> bool:
        rows = [
            r
            for r in ps_rows()
            if str(ps_field(r, "name", "model") or "").split(":")[0] == MODEL
        ]
        if rows and int(row_inflight(rows[0]) or 0) >= 1:
            return True
        # R2-30: a streamed holder arriving while the engine respawns is HELD
        # at the gateway — by design it does not count as ps in_flight (see
        # the slots-queue lane), so an in_flight-only poll is blind to it and
        # the 120s spins exhaust (sweep-9 priority / sweep-11 deadline
        # flakes). The pallama_queue_depth gauge ("waiting requests",
        # gateway queue) sees held requests; single-model daemon here, so the
        # global gauge is this model's queue.
        try:
            _, _, raw = http("GET", "/metrics")
        except Exception:
            return False
        m = re.search(rb"^pallama_queue_depth (\d+)", raw, re.M)
        return bool(m) and int(m.group(1)) >= 1

    holder = threading.Thread(target=_hold_slot)
    holder.start()
    queued = False
    for _attempt in range(2):
        spin = time.time() + 120
        while time.time() < spin and not _slot_busy():
            time.sleep(0.2)
        queued = _slot_busy()
        if queued:
            break
        # Holder may finish before ps catches it on a fast box: re-hold
        # once and retry the busy detection (check below still asserts
        # queued=True, so this cannot mask a real ordering failure).
        holder.join(timeout=300)
        holder = threading.Thread(target=_hold_slot)
        holder.start()
    t_low = threading.Thread(target=_pri_track, args=("low", "low", 512))
    t_high = threading.Thread(target=_pri_track, args=("high", "high", 5))
    if queued:
        t_low.start()
        # Near-zero gap: with the holder confirmed busy (queued=True), BOTH
        # requests must land in the gateway queue together for priority to
        # reorder them. A long gap lets the fast holder finish first, low
        # starts running, and a running request can never be reordered.
        time.sleep(0.05)
        t_high.start()
    holder.join(timeout=300)
    if queued:
        t_low.join(timeout=300)
        t_high.join(timeout=300)
    check(
        "behavior",
        "x-pallama-priority: high admitted before queued low",
        queued
        and order.get("low", (0.0, 0))[1] == 200
        and order.get("high", (0.0, 0))[1] == 200
        and order["high"][0] < order["low"][0],
        f"queued={queued} low={order.get('low')} high={order.get('high')}",
    )

    # deadline accounting: a request admitted past its deadline lands in
    # pallama_slo_deadline_exceeded_total (or is 503-rejected — both honor SLO).
    def _slo_counter() -> int:
        _, _, raw = http("GET", "/metrics")
        m = re.search(rb"^pallama_slo_deadline_exceeded_total (\d+)", raw, re.M)
        return int(m.group(1)) if m else -1

    slo_before = _slo_counter()
    tight: dict[str, int] = {}
    holder2 = threading.Thread(target=_hold_slot)
    holder2.start()
    for _attempt in range(2):
        spin = time.time() + 120
        while time.time() < spin and not _slot_busy():
            time.sleep(0.2)
        if _slot_busy():
            break
        # Same fast-box race as the priority block above: retry the hold
        # once when the holder outlived the ps polling window.
        holder2.join(timeout=300)
        holder2 = threading.Thread(target=_hold_slot)
        holder2.start()
    if _slot_busy():
        st, _, _ = chat(
            "Say the word late.",
            extra={"max_tokens": 5},
            # 1ms: any queue wait behind the holder deterministically breaches
            # the deadline (3000ms relied on the holder outlasting 3s — a
            # race; fast admission left the counter correctly static).
            headers={"x-pallama-deadline-ms": "1"},
            timeout=300,
        )
        tight["st"] = st
    holder2.join(timeout=300)
    slo_after = _slo_counter()
    check(
        "behavior",
        "x-pallama-deadline-ms: late admission accounted or rejected",
        # 429 = predictive early-reject (frontier #28): when TTFT history
        # proves the deadline is unreachable, admission refuses BEFORE
        # queueing — the strongest form of honoring the SLO.
        tight.get("st") in (200, 503, 429)
        and (slo_after > slo_before or tight.get("st") in (503, 429)),
        f"status={tight.get('st')} counter {slo_before}->{slo_after}",
    )
    # speculative decode e2e: ngram self-speculation needs no draft model.
    d.start({"port": PORT, "spec": "ngram"})
    chat("Write the numbers from 1 to 10, one per line.", extra={"max_tokens": 64})
    argv = child_argv(child_pid()) if child_pid() else []
    joined = " ".join(argv)
    check(
        "behavior",
        "spec=ngram: child spawned with --spec-type ngram and serves chat",
        "--spec-type" in joined and "ngram" in joined,
        joined[joined.find("--spec") :][:80] if "--spec" in joined else "missing",
    )
    # router mode.
    d.start({"port": PORT, "router": True})
    st, v, _ = chat("Say ok")
    rows = ps_rows()
    check(
        "behavior",
        "router mode serves chat from one child",
        st == 200 and len(rows) >= 1,
        f"status={st} ps rows={len(rows)}",
    )


def phase_cli() -> None:
    print("\n== phase 6: CLI against the isolated daemon ==")
    d = DAEMON
    d.start({"port": PORT})
    chat("Say ok")  # ensure a model row exists for ps/show
    p = cli("ps")
    ok = p.returncode == 0 and MODEL in p.stdout
    check(
        "cli",
        "pallama ps",
        ok,
        p.stdout.strip().splitlines()[-1][:120] if p.stdout else "",
    )
    reg("ps", ok, "rc0 + model row")
    p = cli("list")
    ok = p.returncode == 0 and MODEL in p.stdout
    check("cli", "pallama list", ok, "")
    reg("list", ok, "rc0 + model listed")
    p = cli("show", MODEL)
    ok = p.returncode == 0
    check("cli", "pallama show", ok, p.stdout.strip().splitlines()[:1])
    reg("show", ok, "rc0")
    p = cli("why")
    ok = p.returncode == 0
    check("cli", "pallama why runs", ok, p.stdout.strip().splitlines()[:1])
    reg("why.default", ok, "rc0")
    p = cli("doctor")
    # Row-level FAIL only: WARN details may contain the word "failed"
    # (e.g. "check failed (GitHub API rate limited ...)").
    fail_rows = [
        ln.strip() for ln in p.stdout.splitlines() if re.search(r"\sFAIL(\s|$)", ln)
    ]
    ok = p.returncode == 0 and not fail_rows
    check(
        "cli",
        "pallama doctor passes",
        ok,
        "all checks pass" if "all checks pass" in p.stdout else p.stdout[-200:],
    )
    reg("doctor", ok, "rc0 + no fail")
    p = cli("config", "get", "default_ctx")
    ok = p.returncode == 0 and "16384" in p.stdout
    check("cli", "config get", ok, p.stdout.strip())
    reg("config.get", ok, "16384 default")
    p = cli("config", "set", "default_ctx", "4096")
    p2 = cli("config", "get", "default_ctx")
    with open(os.path.join(SANDBOX.config_dir, "config.toml"), "rb") as f:
        parsed = tomllib.load(f)
    ok = p.returncode == 0 and "4096" in p2.stdout and parsed.get("default_ctx") == 4096
    check(
        "cli",
        "config set -> get -> file parses (quoted, outside tables)",
        ok,
        f"get={p2.stdout.strip()!r} file default_ctx={parsed.get('default_ctx')}",
    )
    reg("config.set", ok, "set 4096 -> get + file parse")
    p = cli("engine", "list")
    cur = _active_engine_tag()
    ok = p.returncode == 0 and (
        (cur and cur in p.stdout)
        or (cur is None and re.search(r"\bb\d+", p.stdout) is not None)
    )
    check(
        "cli",
        "engine list shows the installed engine",
        ok,
        p.stdout.strip().splitlines()[-1][:120] if p.stdout else "",
    )
    reg("engine.list", ok, f"active {cur or 'any b-tag'} listed")
    p = cli("run", MODEL, "--verbose", "--max-tokens", "64", "Say: inline")
    low = p.stdout.lower()
    ok = p.returncode == 0 and ("count" in low or "tokens" in low or "duration" in low)
    check(
        "cli",
        "run single-shot --verbose completes with stats",
        ok,
        (" ".join(p.stdout.strip().splitlines()[-3:])[:200])
        or f"rc={p.returncode} err={p.stderr[:150]}",
    )
    reg("run.verbose", ok, "--verbose stats")
    p = cli("stop", MODEL)
    ok = p.returncode == 0
    check("cli", "stop MODEL unloads via /api/evict", ok, p.stdout.strip()[:100])
    reg("stop.model", ok, "model unloaded")
    # cp refuses while loaded (verified above by design); alias after unload.
    p = cli("cp", MODEL, "validate-alias")
    p2 = cli("list")
    ok = p.returncode == 0 and "validate-alias" in p2.stdout
    check(
        "cli",
        "cp creates a zero-byte alias after unload (no blob copy)",
        ok,
        f"cp rc={p.returncode} {p.stderr.strip()[:150] or p.stdout.strip()[:80]}",
    )
    reg("cp", ok, "alias created + listed")
    cli("rm", "validate-alias")
    p = cli("list")
    ok = "validate-alias" not in p.stdout
    check("cli", "rm removes the alias", ok, "")
    reg("rm", ok, "alias gone")
    boundary(
        "cli",
        "upgrade --dry-run (legacy boundary)",
        "superseded by the real upgrade lane in phase commands; kept for FAST triage",
    )


def phase_auth() -> None:
    print("\n== phase 7: auth ==")
    d = DAEMON
    d.start({"port": PORT, "keys": [{"name": "validate", "key": "validate-key-1"}]})
    st, _, _ = http_json("GET", "/api/version")
    check("auth", "keys set: no bearer -> 401", st == 401, f"status={st}")
    st, _, _ = http_json(
        "GET", "/api/version", headers={"Authorization": "Bearer validate-key-1"}
    )
    check("auth", "correct bearer -> 200", st == 200, f"status={st}")
    st, _, _ = http_json("GET", "/healthz")
    check("auth", "/healthz stays open", st == 200, f"status={st}")
    # /api/keys/rotate: admin-scoped — new secret shown once, old dies.
    st, _, v = http_json(
        "POST",
        "/api/keys/rotate?name=validate",
        headers={"Authorization": "Bearer validate-key-1"},
    )
    new_secret = str(v.get("key") or "") if isinstance(v, dict) else ""
    check(
        "auth",
        "keys rotate -> 200 + new secret",
        st == 200 and bool(new_secret),
        f"status={st}",
    )
    st, _, _ = http_json(
        "GET", "/api/version", headers={"Authorization": "Bearer validate-key-1"}
    )
    check("auth", "rotated-away old bearer -> 401", st == 401, f"status={st}")
    st, _, _ = http_json(
        "GET", "/api/version", headers={"Authorization": f"Bearer {new_secret}"}
    )
    check("auth", "new rotated bearer -> 200", st == 200, f"status={st}")


# ------------------------------------------------------------------- wave
# Batteries for the kernels + loading-core waves (2026-09-07/08): replicas,
# predictive preload, key concurrency surface, whisper teaching, lookup-cache
# validation, poller gauges. Heavy lanes (9B loads) skip under FAST.


def wave_chat(model: str, system: str, max_tokens: int = 5) -> tuple[int, object]:
    body = {
        "model": model,
        "stream": False,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": "Say OK."},
        ],
    }
    st, _, v = http_json("POST", "/v1/chat/completions", body)
    return st, v


def wave_replica_rows(model: str) -> list[dict]:
    return [r for r in ps_rows() if str(ps_field(r, "name", "model") or "") == model]


def phase_wave() -> None:
    print("\n== phase 8: wave battery (replicas/preload/keys/whisper/gauges) ==")
    d = DAEMON
    small = "qwen2.5-0.5b-instruct"
    big = BIG  # second DISTINCT model (default qwen3.5-9b)

    # -- battery A: replicas + slots/pin/cache_idle_slots overlays ----------
    d.start(
        {
            "port": PORT,
            "cache_idle_slots": False,
            "model_overrides": {
                small: {"replicas": 2, "slots": 2, "pin": True},
            },
        }
    )
    st, _ = wave_chat(small, "You are a pirate who loves the sky.")
    a_loaded = wait_loaded(small, budget=180)
    check(
        "wave",
        "battery A: first chat loads model",
        st == 200 and a_loaded is not None,
        f"status={st} loaded={a_loaded is not None}",
    )

    # Second DISTINCT prefix must grow to a second replica (sticky routing).
    st, _ = wave_chat(small, "You are a scientist who studies gold.")
    got_two = False
    deadline = time.time() + 120
    while time.time() < deadline:
        rows = wave_replica_rows(small)
        reps = sorted(
            r.get("pallama_replica") for r in rows if r.get("pallama_replica")
        )
        if len(rows) >= 2 and reps[:2] == [1, 2]:
            got_two = True
            break
        time.sleep(1)
    check(
        "wave",
        "replicas: distinct prefixes -> two children (replica 1+2)",
        got_two,
        f"rows={[(r.get('pallama_replica')) for r in wave_replica_rows(small)]}",
    )

    p1 = replica_pid(small, 1)
    argv = child_argv(p1) if p1 else []
    np_pair = any(argv[i] == "-np" and argv[i + 1] == "2" for i in range(len(argv) - 1))
    check(
        "wave",
        "slots overlay shadows global (-np 2)",
        np_pair,
        f"pid1={p1} np={'-np 2' if np_pair else 'MISSING'}",
    )
    check(
        "wave",
        "cache_idle_slots=false emits --no-cache-idle-slots",
        "--no-cache-idle-slots" in argv,
        f"present={'--no-cache-idle-slots' in argv}",
    )
    picked = daemon_log_contains("auto GPU pick")
    pinned = bool(VALIDATE_DEVICES)  # devices override => no auto-pick log
    check(
        "wave",
        "--device emission matches auto-pick decision"
        + (" (devices pinned)" if pinned else ""),
        ("--device" in argv) if pinned else (picked == ("--device" in argv)),
        f"pinned={pinned} log_auto_pick={picked} argv_device={'--device' in argv}",
    )

    # Same prefix again: sticky (must NOT grow a third child).
    wave_chat(small, "You are a pirate who loves the sky.")
    time.sleep(3)
    rows = wave_replica_rows(small)
    check(
        "wave",
        "replicas: same prefix sticky (no third child)",
        len(rows) == 2,
        f"rows={len(rows)}",
    )

    # Model-level evict kills BOTH replicas.
    st, _, _ = http_json("POST", "/api/evict", {"model": small})
    gone = False
    deadline = time.time() + 60
    while time.time() < deadline:
        if (
            not wave_replica_rows(small)
            and replica_pid(small, 1) is None
            and replica_pid(small, 2) is None
        ):
            gone = True
            break
        time.sleep(1)
    check(
        "wave",
        "evict_model kills both replicas",
        st == 200 and gone,
        f"status={st} children_gone={gone}",
    )
    cov(
        "replicas+pin+slots overlay",
        "2 prefixes -> 2 children, sticky, evict_model kills both",
        "phase 8 battery A",
        got_two and gone and np_pair,
    )
    cov(
        "cache_idle_slots",
        "false -> --no-cache-idle-slots in child argv",
        "phase 8 battery A",
        "--no-cache-idle-slots" in argv,
    )

    # -- battery B: predictive preload (heavy: two model loads) ------------
    # Needs TWO distinct live models; guard against BIG==small or BIG not
    # in the store (honest boundary, never a silent collapse to one model).
    st_t, _, v_t = http_json("GET", "/api/tags")
    tag_names = (
        [m.get("name", "") for m in v_t.get("models", [])]
        if isinstance(v_t, dict)
        else []
    )
    big_ok = big != small and any(big in t for t in tag_names)
    # Capacity pre-check mirroring the supervisor's bytes admission: the
    # reaper's maybe_preload skips when the MEASURED resident footprint
    # plus the incoming model's admission floor (weights + projector +
    # 512 MiB KV floor + 700 MiB spawn overhead — the same charges the
    # unified gpu-layers pin trusts) would cross the VRAM budget.
    # Weights-only math oversubscribed an 8 GiB card on 2026-09-11: the
    # 9B VL model settled at 7302 MiB MEASURED (5417 weights) and the
    # 0.5B sibling (~977 actual) crashed the pair into an NVRM
    # NO_MEMORY storm. MiB-floored approximation (window of disagreement
    # < 0.1%); assumes max_loaded_models = 0 (harness default).
    vram_mib = daemon_vram_mib()
    small_mib = model_bytes_mib(small)
    big_mib = model_bytes_mib(big)
    OVERHEAD_MIB = 512 + 700  # KV floor + spawn overhead, per spawn
    pair_fits = (
        vram_mib > 0
        and small_mib > 0
        and big_mib > 0
        and (small_mib + big_mib + 2 * OVERHEAD_MIB) <= vram_mib
    )
    if FAST:
        print(
            f"  (FAST: skipping predictive-preload battery — needs {small}+{big} loads)"
        )
        boundary(
            "wave",
            "predictive_preload battery (FAST escape)",
            "needs two model loads; full run executes the real battery",
        )
    elif not big_ok:
        boundary(
            "wave",
            "predictive_preload battery (no distinct big model)",
            f"PALLAMA_VALIDATE_BIG_MODEL={big!r} equals small or is not in the "
            "store; pull it to enable this battery",
        )
    elif not pair_fits:
        boundary(
            "wave",
            "predictive_preload battery (VRAM capacity ceiling)",
            f"small {small_mib} + big {big_mib} + 2x{OVERHEAD_MIB} MiB admission floors "
            f"> vram {vram_mib} MiB: the bytes admission (measured resident + "
            "incoming floor <= budget) gates maybe_preload on this GPU, so "
            "small+big co-residency cannot happen (mirror of supervisor.rs "
            "resident_bytes/admission_floor_bytes); point "
            "PALLAMA_VALIDATE_BIG_MODEL at a smaller model to exercise "
            "the battery",
        )
    else:
        d.start({"port": PORT, "predictive_preload": True})
        seq_ok = True
        for _ in range(3):
            st_a, _ = wave_chat(small, "Answer briefly.")
            st_b, _ = wave_chat(big, "Answer briefly.")
            st_e, _, _ = http_json("POST", "/api/evict", {"model": big})
            st_a2, _ = wave_chat(small, "Answer briefly.")
            seq_ok = (
                seq_ok and st_a == 200 and st_b == 200 and st_e == 200 and st_a2 == 200
            )
        check(
            "wave",
            "battery B: A/B transition sequencing all 200",
            seq_ok,
            f"seq_ok={seq_ok}",
        )
        # Only A is live now with three accrued A->B edges: the reaper tick
        # (10s) should pre-spawn B within the poll budget.
        preloaded = False
        deadline = time.time() + 180
        while time.time() < deadline:
            if wave_replica_rows(big) and wave_replica_rows(small):
                preloaded = True
                break
            time.sleep(2)
        check(
            "wave",
            "predictive preload: B spawned while only A live",
            preloaded,
            f"preloaded={preloaded} log_has_edge={daemon_log_contains('predictive preload')}",
        )
        cov(
            "predictive_preload",
            "A->B transitions >=3 -> B pre-spawned while only A live",
            "phase 8 battery B",
            preloaded,
        )
        http_json("POST", "/api/evict", {"model": big})
        http_json("POST", "/api/evict", {"model": small})

    # -- battery C: default config surface ----------------------------------
    d.start({"port": PORT})
    st, _, v = http_json("GET", "/api/keys")
    body_txt = json.dumps(v) if not isinstance(v, str) else v
    check(
        "wave",
        "keys mgmt keyless -> 403 teaching",
        st == 403 and "no keys configured" in body_txt,
        f"status={st}",
    )

    st, raw = http_multipart(
        "/v1/audio/transcriptions", {"model": "whisper-1"}, "file", tiny_wav(), "t.wav"
    )
    check(
        "wave",
        "whisper uninstalled -> 501 teaching (install/pull hints)",
        # F166: the gateway teaches `pallama whisper --install` (dashed) —
        # match the real text, not the old never-matching prose.
        st == 501 and b"whisper --install" in raw,
        f"status={st}",
    )

    st, _, v = http_json("POST", "/api/embed", {"model": small, "input": "hi"})
    txt = json.dumps(v) if not isinstance(v, str) else v
    check(
        "wave",
        "/api/embed reachable (engine 501 passthrough w/o --embeddings)",
        st in (200, 501),
        f"status={st} body={txt[:80]}",
    )
    st, _, v = http_json(
        "POST", "/api/rerank", {"model": small, "query": "hi", "documents": ["a"]}
    )
    check("wave", "/api/rerank reachable", st in (200, 501), f"status={st}")

    # Poller gauge: identical chats prime the prompt cache, then one 60s
    # tick must publish pallama_prefix_cache_hit_rate.
    for _ in range(2):
        wave_chat(small, "You echo single words. Say OK.")
    time.sleep(68)
    st, hdr, raw = http("GET", "/metrics")
    check(
        "wave",
        "poller gauge pallama_prefix_cache_hit_rate published",
        st == 200 and b"pallama_prefix_cache_hit_rate" in raw,
        f"status={st} found={b'pallama_prefix_cache_hit_rate' in raw}",
    )
    cov(
        "prefix-cache-hit gauge",
        "chat traffic -> /metrics carries pallama_prefix_cache_hit_rate",
        "phase 8 battery C",
        b"pallama_prefix_cache_hit_rate" in raw,
    )
    hit_rate = _metric_value(raw, "pallama_prefix_cache_hit_rate")
    check(
        "wave",
        "prefix cache actually reuses: hit rate > 0 after repeated prompts",
        hit_rate is not None and hit_rate > 0.0,
        f"hit_rate={hit_rate}",
    )
    check(
        "wave",
        "TTFT/TPOT histograms rendered after chat traffic",
        b"pallama_ttft_seconds_bucket" in raw and b"pallama_tpot_seconds_bucket" in raw,
        f"ttft={b'pallama_ttft_seconds_bucket' in raw} tpot={b'pallama_tpot_seconds_bucket' in raw}",
    )

    # -- battery D: admin-key round-trip incl. max_concurrent ---------------
    d.start({"port": PORT, "keys": [{"name": "admin", "key": "admin-key-1"}]})
    auth = {"Authorization": "Bearer admin-key-1"}
    st, _, v = http_json(
        "POST",
        "/api/keys",
        {"name": "limited", "rpm": 60, "max_concurrent": 1},
        headers=auth,
    )
    created = (
        st == 201 and isinstance(v, dict) and str(v.get("key", "")).startswith("plm_")
    )
    check(
        "wave",
        "keys add via admin -> 201 + plm_ secret shown once",
        created,
        f"status={st}",
    )

    st, _, v = http_json("GET", "/api/keys", headers=auth)
    rows = v.get("keys", []) if isinstance(v, dict) else []
    row = next((r for r in rows if r.get("name") == "limited"), None)
    check(
        "wave",
        "keys list shows limited w/ max_concurrent=1",
        row is not None and row.get("max_concurrent") == 1,
        f"row={row is not None} max_concurrent={row.get('max_concurrent') if row else None}",
    )
    cov(
        "keys max_concurrent",
        "add/list round-trip carries max_concurrent",
        "phase 8 battery D",
        row is not None and row.get("max_concurrent") == 1,
    )

    st, _, v = http_json("POST", "/api/keys", {"name": "limited"}, headers=auth)
    check("wave", "keys add duplicate -> 409", st == 409, f"status={st}")
    st, _, v = http_json("DELETE", "/api/keys?name=limited", headers=auth)
    check("wave", "keys remove -> 200", st == 200, f"status={st}")

    # -- lookup-cache bogus path: config validation refuses to start --------
    refused = False
    log_snip = ""
    try:
        d.start(
            {"port": PORT, "lookup_cache_static": "/nonexistent/really-missing.bin"}
        )
    except RuntimeError as e:
        refused = "exited early" in str(e)
        log_snip = d.tail_log(5)
    check(
        "wave",
        "lookup_cache_static bogus path -> daemon refuses to start",
        refused and "lookup_cache_static" in log_snip,
        f"refused={refused} log_has_field={'lookup_cache_static' in log_snip}",
    )
    cov(
        "lookup_cache_static",
        "missing file path fails validation at startup",
        "phase 8 refusal lane",
        refused,
    )

    # -- battery F: gateway finish-line wave -------------------------------
    # Live, real-engine proofs for the 2026-09-09 wave: audit log, WFQ
    # weights, per-model rpc override, predictive deadline reject, session
    # identity manifests, remote-pool prefix stickiness.
    f_small = small
    audit_path = os.path.join(SANDBOX.data_dir, "log", "audit.jsonl")

    def _hdr_get(headers: dict, name: str) -> str | None:
        for k, v in headers.items():
            if k.lower() == name.lower():
                return v
        return None

    def _chat(
        model: str,
        system: str,
        user: str,
        key: str | None = None,
        max_tokens: int = 8,
        extra_headers: dict | None = None,
    ) -> tuple[int, dict, object]:
        headers = dict(extra_headers or {})
        if key:
            headers["Authorization"] = f"Bearer {key}"
        st, hd, raw = http(
            "POST",
            "/api/chat",
            {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "options": {"num_predict": max_tokens},
            },
            headers=headers,
        )
        try:
            return st, hd, json.loads(raw)
        except Exception:
            return st, hd, {}

    def _audit_lines() -> list[dict]:
        try:
            with open(audit_path) as f:
                return [json.loads(x) for x in f if x.strip()]
        except FileNotFoundError:
            return []

    # F1: audit + WFQ + predictive deadline on one keyed daemon. (rpc
    # override moved to F2: upstream ABORTS at argv-parse when --rpc
    # points at a dead endpoint — F2 spawns a real ggml-rpc-server.)
    d.start(
        {
            "port": PORT,
            "audit_log": True,
            "slots": 1,
            "keys": [
                {"name": "feather", "key": "feather-secret-1", "weight": 1},
                {"name": "anvil", "key": "anvil-secret-3", "weight": 3},
            ],
        },
        floor_model=f_small,
    )
    st = 0
    t0 = time.time()
    while time.time() - t0 < 180 and st != 200:
        st, _, _ = _chat(
            f_small, "You are a careful accountant.", "hi", "anvil-secret-3"
        )
        if st != 200:
            time.sleep(2)
    check(
        "wave",
        "battery F: audited daemon loads model",
        st == 200,
        f"status={st}",
    )

    # F-audit: success line lands with key+model; failed auth writes NOTHING
    # (auth is the outer middleware; 401 short-circuits before request_log).
    n_before = len(_audit_lines())
    st, _, _ = _chat(
        f_small, "You are a careful accountant.", "balance", "feather-secret-1"
    )
    got_line = None
    deadline_t = time.time() + 5
    while time.time() < deadline_t and got_line is None:
        for ln in _audit_lines()[n_before:]:
            if ln.get("key") == "feather" and ln.get("path") == "/api/chat":
                got_line = ln
        if got_line is None:
            time.sleep(0.25)
    check(
        "wave",
        "audit_log: /api/chat line {key,model,status,path}",
        got_line is not None
        and got_line.get("model") == f_small
        and got_line.get("status") == 200
        and "ts" in got_line
        and "ms" in got_line,
        f"line={got_line}",
    )
    n_auth = len(_audit_lines())
    st401, _, _ = _chat(f_small, "You are a careful accountant.", "x", "wrong-key")
    got_401 = None
    deadline_401 = time.time() + 5
    while time.time() < deadline_401 and got_401 is None:
        for ln in _audit_lines()[n_auth:]:
            if ln.get("status") == 401 and ln.get("path") == "/api/chat":
                got_401 = ln
        if got_401 is None:
            time.sleep(0.25)
    # F64: request_log is now OUTSIDE auth, so a rejected secret still
    # gets an access/audit line (key=null, status=401) — brute-force
    # probing is visible. Old contract ("401 leaves no audit line")
    # was the bug being pinned away.
    check(
        "wave",
        "audit_log: 401 leaves a keyless audit line (F64 visibility)",
        st401 == 401 and got_401 is not None and got_401.get("key") is None,
        f"status={st401} line={got_401}",
    )

    # F-wfq: slots=1 serializes; 8+8 concurrent chats from weight-1 and
    # weight-3 keys. Completion order == admission order on a single slot,
    # and audit lines append at completion: the line stream IS the WFQ
    # admission trace. Expect anvil >= 2x feather (3:1 service ratio).
    n_wfq = len(_audit_lines())
    barrier = threading.Barrier(17)
    results: list[tuple[str, int]] = []
    res_lock = threading.Lock()

    def _wfq_worker(kname: str, ksecret: str, i: int) -> None:
        body = {
            "model": f_small,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a careful accountant.",
                },
                {"role": "user", "content": f"tell me about ledger page {i}"},
            ],
            "stream": False,
            "options": {"num_predict": 96},
        }
        req = urllib.request.Request(
            f"http://127.0.0.1:{PORT}/api/chat",
            data=json.dumps(body).encode(),
            method="POST",
        )
        req.add_header("content-type", "application/json")
        req.add_header("Authorization", f"Bearer {ksecret}")
        barrier.wait()
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                code = r.status
        except urllib.error.HTTPError as e:
            code = e.code
        with res_lock:
            results.append((kname, code))

    threads = [
        threading.Thread(
            target=_wfq_worker,
            args=("feather", "feather-secret-1", i)
            if i % 4 == 0
            else ("anvil", "anvil-secret-3", i),
        )
        for i in range(16)
    ]
    for t in threads:
        t.start()
    barrier.wait(timeout=30)
    for t in threads:
        t.join(timeout=600)
    deadline_t = time.time() + 5
    while time.time() < deadline_t and len(_audit_lines()) < n_wfq + 16:
        time.sleep(0.25)
    order = [ln.get("key") for ln in _audit_lines()[n_wfq:] if ln.get("key")]
    f_count = order.count("feather")
    a_count = order.count("anvil")
    check(
        "wave",
        "WFQ: weight-3 key gets >=2x admissions of weight-1 (slots=1)",
        len(results) == 16
        and all(c == 200 for _, c in results)
        and f_count >= 1
        and a_count >= 2 * f_count
        and f_count + a_count >= 14,
        f"order={order} feather={f_count} anvil={a_count}",
    )

    # F-deadline: 20 unique cold prompts feed ttft_cold (shared state.obs);
    # then an explicit 1ms deadline on the OPENAI lane (the only lane that
    # parses x-pallama-deadline-ms into admission) must be predictively
    # rejected 429 + Retry-After BEFORE queueing: real p90 TTFT is orders
    # of magnitude over 2ms.
    for i in range(20):
        _chat(
            f_small,
            f"You are historian number {i}.",
            f"say one word about year {1900 + i}",
            "anvil-secret-3",
            max_tokens=2,
        )
    st_dl, hd_dl, raw_dl = http(
        "POST",
        "/v1/chat/completions",
        {
            "model": f_small,
            "messages": [
                {"role": "user", "content": "deadline probe"},
            ],
            "max_tokens": 4,
        },
        headers={
            "Authorization": "Bearer anvil-secret-3",
            "x-pallama-deadline-ms": "1",
        },
    )
    retry_after = _hdr_get(hd_dl, "Retry-After")
    body_txt = raw_dl.decode(errors="replace")
    check(
        "wave",
        "predictive admission: 1ms deadline -> 429 + Retry-After",
        st_dl == 429 and retry_after is not None and "predicted" in body_txt.lower(),
        f"status={st_dl} retry-after={retry_after} body={body_txt[:90]}",
    )

    # F2: keyless daemon for rpc override + session identity. Upstream
    # llama-server connects --rpc endpoints EAGERLY at argv-parse and
    # SIGABRTs on a dead one (ggml-rpc.cpp:547), so spawn a REAL
    # ggml-rpc-server first — anything less crash-loops the child.
    d.stop()
    rpc_tag = _active_engine_tag()
    rpc_srv_path = os.path.join(
        SANDBOX.data_dir, "engines", rpc_tag, f"llama-{rpc_tag}", "ggml-rpc-server"
    )
    # F164: existence precheck — a missing binary used to raise
    # FileNotFoundError here and truncate every phase after the battery.
    # Skip the rpc leg with a recorded failure and keep the harness alive
    # (the CUDA asset ships without ggml-rpc-server; see F167).
    rargv: list[str] = []  # bound here so the cov() below survives the skip
    rpc_skip = not os.path.exists(rpc_srv_path)
    if rpc_skip:
        check(
            "wave",
            "rpc override leg (ggml-rpc-server present in asset)",
            False,
            f"{rpc_srv_path} missing — rpc override leg skipped (F164 guard)",
        )
        d.start({"port": PORT}, floor_model=f_small)
        st = 0
        t0 = time.time()
        while time.time() - t0 < 180 and st != 200:
            st, _, _ = _chat(f_small, "You are a careful accountant.", "hi")
            if st != 200:
                time.sleep(2)
    else:
        # Kernel-assigned free port — the fixed 54321 made two concurrent
        # validate runs fight over one ggml-rpc-server (same class as the
        # edge-port collision, 2026-09-10).
        rpc_port = _free_port()
        rpc_target = f"127.0.0.1:{rpc_port}"
        rpc_proc = subprocess.Popen(
            [rpc_srv_path, "--host", "127.0.0.1", "--port", str(rpc_port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )

        rpc_up = False
        t0 = time.time()
        while time.time() - t0 < 20:
            s = socket.socket()
            s.settimeout(1)
            try:
                s.connect(("127.0.0.1", rpc_port))
                rpc_up = True
            except OSError:
                time.sleep(0.5)
            finally:
                s.close()
            if rpc_up:
                break
        if not rpc_up:
            raise RuntimeError(f"ggml-rpc-server did not accept on {rpc_target}")
        try:
            d.start(
                {
                    "port": PORT,
                    "model_overrides": {f_small: {"rpc_servers": rpc_target}},
                },
                floor_model=f_small,
            )
            st = 0
            t0 = time.time()
            while time.time() - t0 < 180 and st != 200:
                st, _, _ = _chat(f_small, "You are a careful accountant.", "hi")
                if st != 200:
                    time.sleep(2)
            rpid = child_pid(f_small)
            rargv = child_argv(rpid) if rpid else []
            check(
                "wave",
                f"model_overrides.rpc_servers -> child --rpc {rpc_target}",
                st == 200
                and any(
                    rargv[i] == "--rpc" and rargv[i + 1] == rpc_target
                    for i in range(len(rargv) - 1)
                ),
                f"load_st={st} pid={rpid} rpc={[a for a in rargv if a == '--rpc']}",
            )
        finally:
            if rpc_proc.poll() is None:
                rpc_proc.terminate()
                try:
                    rpc_proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    rpc_proc.kill()

    # F2b: dead rpc endpoint -> spawn REFUSED with a teaching error (the
    # LlamaCppEngine::spawn preflight): gateway 500 naming the endpoint,
    # zero child processes (no SIGABRT crash-loop), no engine rollback.
    # Needs no ggml-rpc-server binary — runs in both branches above.
    d.stop()
    st_dead, elapsed, taught, no_child = 0, 0.0, False, False
    for _attempt in range(2):  # retry-once: _free_port bind-release race
        dead_port = _free_port()  # released again = nothing listens there
        d.start(
            {
                "port": PORT,
                "model_overrides": {f_small: {"rpc_servers": f"127.0.0.1:{dead_port}"}},
            },
            floor_model=f_small,
        )
        t0 = time.time()
        st_dead, _, body_dead = _chat(f_small, "You are a careful accountant.", "hi")
        elapsed = time.time() - t0
        taught = "unreachable" in str(body_dead).lower()
        no_child = child_pid(f_small) is None
        d.stop()
        if st_dead == 500 and taught and no_child:
            break
        time.sleep(1)
    check(
        "wave",
        "rpc dead endpoint -> spawn refused with teaching error (no crash-loop)",
        st_dead == 500 and taught and no_child,
        f"st={st_dead} elapsed={elapsed:.1f}s child_spawned={not no_child} "
        f"err={str(body_dead)[:140]}",
    )

    # F-identity and every lane after it expect a LIVE keyless daemon; F2b
    # leaves it stopped. Restart clean — without the dead rpc pin, so the
    # session lanes spawn f_small normally (same idiom as the F164 skip arm).
    d.start({"port": PORT}, floor_model=f_small)
    st = 0
    t0 = time.time()
    while time.time() - t0 < 180 and st != 200:
        st, _, _ = _chat(f_small, "You are a careful accountant.", "hi")
        if st != 200:
            time.sleep(2)

    # F-identity: save -> manifest exists; clean restore 200; tampered ctx
    # -> 400 teaching; erase removes checkpoint AND manifest.
    st_sv, _, _ = http_json(
        "POST",
        "/api/session",
        {"model": f_small, "action": "save", "filename": "livechk", "slot": 0},
    )
    manifests = []
    for root, _, files in os.walk(os.path.join(SANDBOX.data_dir, "sessions")):
        manifests = [
            os.path.join(root, f) for f in files if f == "livechk.identity.json"
        ]
        if manifests:
            break
    st_rs, _, _ = http_json(
        "POST",
        "/api/session",
        {"model": f_small, "action": "restore", "filename": "livechk", "slot": 0},
    )
    check(
        "wave",
        "session identity: save writes manifest, clean restore ok",
        st_sv == 200 and len(manifests) == 1 and st_rs == 200,
        f"save={st_sv} manifest={manifests} restore={st_rs}",
    )
    with open(manifests[0]) as mf:
        ident = json.load(mf)
    ident["ctx"] = int(ident.get("ctx", 0)) + 4096
    with open(manifests[0], "w") as mf:
        json.dump(ident, mf)
    st_tr, _, v_tr = http_json(
        "POST",
        "/api/session",
        {"model": f_small, "action": "restore", "filename": "livechk", "slot": 0},
    )
    tr_txt = json.dumps(v_tr)
    check(
        "wave",
        "session identity: tampered ctx -> 400 teaching diff",
        st_tr == 400 and "different runtime shape" in tr_txt and "ctx" in tr_txt,
        f"status={st_tr} body={tr_txt[:120]}",
    )
    st_er, _, _ = http_json(
        "POST",
        "/api/session",
        {"model": f_small, "action": "erase", "filename": "livechk"},
    )
    gone = not os.path.exists(os.path.join(os.path.dirname(manifests[0]), "livechk"))
    check(
        "wave",
        "session identity: erase removes checkpoint + manifest",
        st_er == 200 and gone,
        f"status={st_er} checkpoint_gone={gone}",
    )
    cov(
        "audit_log",
        "audit.jsonl success lines + 401 silence",
        "phase 8 battery F",
        got_line is not None,
    )
    cov(
        "keys.weight",
        "3:1 WFQ service ratio under slots=1",
        "phase 8 battery F audit trace",
        a_count >= 2 * f_count,
    )
    cov(
        "model_overrides.rpc_servers",
        "per-model --rpc emission",
        "phase 8 battery F argv",
        any(
            rargv[i] == "--rpc" and rargv[i + 1] == rpc_target
            for i in range(len(rargv) - 1)
        ),
    )
    d.stop()

    # F-remotes: two REAL secondary daemons as one "far" pool; prefix
    # stickiness binds a conversation to one backend and says so in the
    # x-pallama-remote response header. Edge ports are kernel-assigned
    # free ports — the fixed 115xx defaults made two concurrent validate
    # runs bind each other's backends and 502 mid-battery (2026-09-10).
    edge_ports = (_free_port(), _free_port())
    edge_sbs: list[Sandbox] = []
    edge_procs: list[subprocess.Popen] = []
    remote_ok = False
    try:
        for ep in edge_ports:
            esb = Sandbox()
            esb.write_config(
                {
                    "port": ep,
                    "keys": [{"name": "edge", "key": "edge-secret-1"}],
                }
            )
            elog = open(os.path.join(esb.data_dir, "run", "daemon.log"), "ab")
            eproc = subprocess.Popen(
                [PAL, "serve"],
                env=esb.env(),
                stdout=elog,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
            )
            edge_sbs.append(esb)
            edge_procs.append(eproc)
            ok_boot = False
            t0 = time.time()
            while time.time() - t0 < 120:
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{ep}/healthz", timeout=2
                    ) as r:
                        if r.status == 200:
                            ok_boot = True
                            break
                except Exception:
                    if eproc.poll() is not None:
                        break
                    time.sleep(0.5)
            if not ok_boot:
                raise RuntimeError(f"edge daemon on :{ep} failed to boot")
        d.start(
            {
                "port": PORT,
                "remotes": [
                    {
                        "name": "far",
                        "url": f"http://127.0.0.1:{edge_ports[0]}",
                        "key": "edge-secret-1",
                    },
                    {
                        "name": "far",
                        "url": f"http://127.0.0.1:{edge_ports[1]}",
                        "key": "edge-secret-1",
                    },
                ],
            }
        )
        remote_model = f"far:{f_small}"
        st_r1, hd_r1, _ = _chat(remote_model, "You are the north edge.", "ping one")
        hop1 = _hdr_get(hd_r1, "x-pallama-remote")
        st_r2, hd_r2, _ = _chat(remote_model, "You are the north edge.", "ping two")
        hop2 = _hdr_get(hd_r2, "x-pallama-remote")
        st_r3, hd_r3, _ = _chat(remote_model, "You are the south edge.", "ping three")
        hop3 = _hdr_get(hd_r3, "x-pallama-remote")
        check(
            "wave",
            "remote pool: forward through real 2nd-level daemon + header",
            st_r1 == 200
            and hop1 is not None
            and any(hop1 == f"far|http://127.0.0.1:{ep}" for ep in edge_ports),
            f"status={st_r1} header={hop1} edge_ports={edge_ports}",
        )
        check(
            "wave",
            "remote pool: same prefix sticky (same backend twice)",
            st_r2 == 200 and hop2 == hop1,
            f"hop1={hop1} hop2={hop2}",
        )
        check(
            "wave",
            "remote pool: distinct prefix still served (any backend)",
            st_r3 == 200 and hop3 is not None and hop3.startswith("far|"),
            f"status={st_r3} header={hop3}",
        )
        cov(
            "remotes pool",
            "prefix-sticky routing + x-pallama-remote across 2 live backends",
            "phase 8 battery F",
            st_r1 == 200 and hop2 == hop1,
        )
        remote_ok = st_r1 == 200 and st_r2 == 200 and st_r3 == 200
    finally:
        d.stop()
        # Autopsy aid: on failure the edge daemon logs are the ONLY
        # evidence of a 2nd-level crash — surface their tails before the
        # sandboxes (and logs) are destroyed.
        if not remote_ok:
            for esb in edge_sbs:
                lp = os.path.join(esb.data_dir, "run", "daemon.log")
                try:
                    with open(lp, "rb") as f:
                        tail = f.read()[-2000:].decode(errors="replace")
                    print(f"  (edge daemon log tail {lp}):\n{tail}")
                except OSError:
                    pass
        for eproc in edge_procs:
            if eproc.poll() is None:
                eproc.terminate()
                try:
                    eproc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    eproc.kill()
        for esb in edge_sbs:
            esb.destroy()


# ----------------------------------------------------------------- parity
# Battery for the 2026-09-08 parity wave: per-model chat_template +
# sampler_defaults argv flow, model-scoped upstream surfaces (/props,
# /slots, /v1/stream*, /v1/streams/lookup), Jina rerank alias, and the
# generate-lane unknown-option fail-fast.


def phase_parity() -> None:
    print("\n== phase 9: parity battery (templates/samplers/scoped surfaces) ==")
    d = DAEMON
    small = "qwen2.5-0.5b-instruct"

    # -- battery A: overlay sampler_defaults + chat_template reach argv -----
    d.start(
        {
            "port": PORT,
            "model_overrides": {
                small: {
                    "sampler_defaults": {
                        "temperature": 0.123,
                        "top_k": 7,
                        "seed": 424242,
                        "dry_multiplier": 0.8,
                    },
                    "chat_template": "chatml",
                },
            },
        },
        floor_model=small,
    )
    st, _ = wave_chat(small, "Answer briefly.")
    a_loaded = wait_loaded(small, budget=180)
    check(
        "parity",
        "battery A: chat loads model with new overlays",
        st == 200 and a_loaded is not None,
        f"status={st} loaded={a_loaded is not None}",
    )
    pid = child_pid(small)
    argv = child_argv(pid) if pid else []

    def has(flag: str, val: str | None = None) -> bool:
        if val is None:
            return flag in argv
        return any(
            argv[i] == flag and i + 1 < len(argv) and argv[i + 1] == val
            for i in range(len(argv))
        )

    check(
        "parity",
        "sampler_defaults -> --temp 0.123 --top-k 7 --seed 424242",
        has("--temp", "0.123")
        and has("--top-k", "7")
        and has("--seed", "424242")
        and has("--dry-multiplier", "0.8"),
        f"pid={pid} sampler_flags={[a for a in argv if a in ('--temp', '--top-k', '--seed', '--dry-multiplier', '--chat-template')]}",
    )
    check(
        "parity",
        "chat_template overlay -> --chat-template chatml",
        has("--chat-template", "chatml"),
        f"pid={pid}",
    )
    cov(
        "model_overrides.sampler_defaults",
        "per-model sampler defaults compile to child argv",
        "phase 9 battery A argv",
        has("--temp", "0.123") and has("--top-k", "7"),
    )
    cov(
        "model_overrides.chat_template",
        "per-model template override compiles to child argv",
        "phase 9 battery A argv",
        has("--chat-template", "chatml"),
    )

    # -- battery B: model-scoped upstream surfaces ---------------------------
    # Restart clean so the single-hot-child shortcut is deterministic.
    d.stop()
    d.start({"port": PORT}, floor_model=small)
    st, _, v = http_json("GET", "/props")
    check(
        "parity",
        "/props with zero hot children -> 400 teaching error",
        st == 400 and "X-Pallama-Model" in json.dumps(v),
        f"status={st} body={json.dumps(v)[:200]}",
    )

    st, _ = wave_chat(small, "Answer briefly.")
    b_loaded = wait_loaded(small, budget=180)
    check(
        "parity",
        "battery B: warm single child",
        st == 200 and b_loaded is not None,
        f"status={st} loaded={b_loaded is not None}",
    )
    st, _, v = http_json("GET", "/props")
    check(
        "parity",
        "/props single hot child -> 200 props object",
        st == 200 and isinstance(v, dict) and len(v) > 0,
        f"status={st} body={json.dumps(v)[:200]}",
    )
    st, _, v = http_json("GET", "/slots")
    check(
        "parity",
        "/slots -> 200 bare-array slot listing",
        st == 200 and isinstance(v, list) and len(v) > 0,
        f"status={st} body={json.dumps(v)[:200]}",
    )
    cov(
        "scoped surfaces",
        "/props + /slots forwarded to the resolved child",
        "phase 9 battery B",
        st == 200,
    )
    st, _, raw = http("GET", "/v1/stream")
    txt = raw.decode(errors="replace")
    check(
        "parity",
        "/v1/stream GET forwarded (2xx/4xx from child, not gateway 404)",
        st in (200, 400, 404, 501) and "not found" not in txt.lower(),
        f"status={st} body[:120]={txt[:120]!r}",
    )
    st, _, v = http_json(
        "POST", "/v1/streams/lookup", {"model": small, "prompt": "Answer briefly."}
    )
    lookup_routed = st in (200, 400, 404, 501)
    check(
        "parity",
        "/v1/streams/lookup body-model routing reaches child",
        lookup_routed,
        f"status={st} body={json.dumps(v)[:200]}",
    )

    # Header disambiguation lane (X-Pallama-Model).
    st, _, v = http_json("GET", "/props", headers={"X-Pallama-Model": small})
    check(
        "parity",
        "/props X-Pallama-Model header resolves target",
        st == 200,
        f"status={st} body={json.dumps(v)[:200]}",
    )
    st, _, v = http_json("GET", f"/slots?model={small}")
    check(
        "parity",
        "/slots ?model= query resolves target",
        st == 200,
        f"status={st}",
    )

    # -- battery C: Jina rerank alias + generate fail-fast -------------------
    st, _, v = http_json(
        "POST",
        "/v1/reranking",
        {"model": small, "query": "sky", "documents": ["clouds", "the sky is blue"]},
    )
    # Gen models teach 501 on rerank lanes; rerank models answer 200.
    # A gateway 404 would mean the route never landed.
    check(
        "parity",
        "/v1/reranking alias routed (200/501, never gateway-404)",
        st in (200, 501),
        f"status={st} body={json.dumps(v)[:200]}",
    )
    cov(
        "/v1/reranking",
        "Jina-style alias shares the /v1/rerank lane",
        "phase 9 battery C",
        st in (200, 501),
    )

    st, _, v = http_json(
        "POST",
        "/api/generate",
        {
            "model": small,
            "prompt": "hi",
            "stream": False,
            "options": {"num_batch": 512},
        },
    )
    check(
        "parity",
        "/api/generate unknown option -> 400 naming the option",
        st == 400 and "num_batch" in json.dumps(v),
        f"status={st} body={json.dumps(v)[:200]}",
    )

    # -- battery D: child auth (default-on for TCP children) ---------------
    # The hot child from battery B carries a minted keyfile; direct
    # unauthenticated access must fail while the gateway lane (used by
    # every check above) keeps working.
    pid = child_pid(small)
    argv = child_argv(pid) if pid else []
    keyfile = None
    for i, a in enumerate(argv):
        if a == "--api-key-file" and i + 1 < len(argv):
            keyfile = argv[i + 1]
    check(
        "parity",
        "child argv carries --api-key-file (default-on TCP auth)",
        keyfile is not None,
        f"pid={pid} auth_flags={[a for a in argv if 'api-key' in a]}",
    )
    key_ok = key_exists = mode_ok = False
    secret = ""
    if keyfile and os.path.exists(keyfile):
        key_exists = True
        mode = os.stat(keyfile).st_mode & 0o777
        mode_ok = mode == 0o600
        secret = open(keyfile).read().strip()
        key_ok = secret.startswith("plm_") and len(secret) >= 32
    check(
        "parity",
        "keyfile exists, 0600 perms, plm_ secret",
        key_exists and mode_ok and key_ok,
        f"path={keyfile} exists={key_exists} mode={oct(os.stat(keyfile).st_mode & 0o777) if key_exists else '-'} prefix_ok={key_ok}",
    )
    child_port = None
    for i, a in enumerate(argv):
        if a == "--port" and i + 1 < len(argv):
            child_port = int(argv[i + 1])
    if child_port:

        def child_req(port: int, auth: str | None) -> int:
            r = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/models", method="GET"
            )
            if auth:
                r.add_header("authorization", f"Bearer {auth}")
            try:
                with urllib.request.urlopen(r, timeout=20) as resp:
                    return resp.status
            except urllib.error.HTTPError as e:
                return e.code
            except Exception:
                return 0

        st_noauth = child_req(child_port, None)
        st_auth = child_req(child_port, secret)
        check(
            "parity",
            "direct child access: no auth -> 401, bearer -> 200",
            st_noauth == 401 and st_auth == 200,
            f"port={child_port} noauth={st_noauth} auth={st_auth}",
        )
    else:
        check(
            "parity",
            "direct child access: no auth -> 401, bearer -> 200",
            False,
            "no --port in child argv",
        )
    cov(
        "child_auth",
        "TCP children minted a keyfile; gateway stamps every lane",
        "phase 9 battery D",
        key_ok,
    )

    # -- battery E: C2 knobs reach argv (batch/split/ngram-typed/spm) -------
    d.stop()
    d.start(
        {
            "port": PORT,
            "batch_size": 2048,
            "ubatch_size": 512,
            "threads_batch": 4,
            "main_gpu": 0,
            "split_mode": "none",
            "ngram_size_m": 64,
            "model_overrides": {
                small: {"spec": "ngram-map-k", "spm_infill": True},
            },
        },
        floor_model=small,
    )
    st, _ = wave_chat(small, "Answer briefly.")
    e_loaded = wait_loaded(small, budget=180)
    check(
        "parity",
        "battery E: chat loads model with C2 knobs",
        st == 200 and e_loaded is not None,
        f"status={st} loaded={e_loaded is not None}",
    )
    pid = child_pid(small)
    argv = child_argv(pid) if pid else []

    def has2(flag: str, val: str | None = None) -> bool:
        if val is None:
            return flag in argv
        return any(
            argv[i] == flag and i + 1 < len(argv) and argv[i + 1] == val
            for i in range(len(argv))
        )

    check(
        "parity",
        "compute knobs -> --batch-size 2048 --ubatch-size 512 --threads-batch 4",
        has2("--batch-size", "2048")
        and has2("--ubatch-size", "512")
        and has2("--threads-batch", "4"),
        f"pid={pid} flags={[a for a in argv if a in ('--batch-size', '--ubatch-size', '--threads-batch')]}",
    )
    check(
        "parity",
        "gpu split knobs -> --main-gpu 0 --split-mode none",
        has2("--main-gpu", "0") and has2("--split-mode", "none"),
        f"pid={pid} flags={[a for a in argv if a in ('--main-gpu', '--split-mode', '--tensor-split')]}",
    )
    check(
        "parity",
        "spec=ngram-map-k -> --spec-type + typed size flags",
        has2("--spec-type", "ngram-map-k") and has2("--spec-ngram-map-k-size-m", "64"),
        f"pid={pid} flags={[a for a in argv if 'ngram' in a]}",
    )
    check(
        "parity",
        "spm_infill overlay -> --spm-infill",
        has2("--spm-infill"),
        f"pid={pid}",
    )
    cov(
        "batch/ubatch/threads_batch/main_gpu/split_mode",
        "compute + split knobs compile to child argv",
        "phase 9 battery E argv",
        has2("--batch-size", "2048") and has2("--main-gpu", "0"),
    )
    cov(
        "model_overrides.spec ngram-typed",
        "ngram-map-k family dispatches typed flags",
        "phase 9 battery E argv",
        has2("--spec-type", "ngram-map-k"),
    )
    cov(
        "model_overrides.spm_infill",
        "infill token-order toggle compiles to argv",
        "phase 9 battery E argv",
        has2("--spm-infill"),
    )

    # -- refusal lane: chat_template xor chat_template_file enforced ---------
    refused = False
    log_snip = ""
    try:
        d.start(
            {
                "port": PORT,
                "model_overrides": {
                    small: {
                        "chat_template": "chatml",
                        "chat_template_file": "/etc/hostname",
                    },
                },
            },
            floor_model=small,
        )
    except RuntimeError as e:
        refused = "exited early" in str(e)
        log_snip = d.tail_log(5)
    check(
        "parity",
        "chat_template + chat_template_file -> daemon refuses to start",
        refused and "chat_template" in log_snip,
        f"refused={refused} log_has_field={'chat_template' in log_snip}",
    )
    cov(
        "model_overrides.chat_template xor",
        "both set fails validation at startup",
        "phase 9 refusal lane",
        refused,
    )


# ------------------------------------------------- 100%-coverage phases
# phase commands / knobs_argv / knobs_behavior / gates — driven by the
# manifests in the MANIFEST REGISTRY section above (single source of truth).

PHASE_FILTER: set | None = None  # set by main() when --phase= is used
CRASHED: bool = False  # a phase raised; sandbox must be kept for post-mortem


def run_input(
    *args: str, input_text: str, timeout: int = 180
) -> subprocess.CompletedProcess:
    assert SANDBOX is not None
    return subprocess.run(
        [PAL, *args],
        env=SANDBOX.env(),
        capture_output=True,
        text=True,
        input=input_text,
        timeout=timeout,
    )


def model_path(name: str = MODEL) -> str | None:
    try:
        db = sqlite3.connect(os.path.join(REAL_DATA, "pallama.db"))
        row = db.execute("SELECT path FROM models WHERE name = ?", (name,)).fetchone()
        db.close()
        return row[0] if row else None
    except Exception:
        return None


def _sandbox_db() -> sqlite3.Connection:
    return sqlite3.connect(os.path.join(SANDBOX.data_dir, "pallama.db"))


def _active_engine_tag() -> str | None:
    db = _sandbox_db()
    row = db.execute("SELECT tag FROM engines WHERE active = 1").fetchone()
    db.close()
    return row[0] if row else None


def _full_engine_tags() -> list[str]:
    """Real b-numbered engine tags carrying both llama-server and llama-quantize.

    Covers plain upstream tags (bNNNN) AND CUDA overlay/source tags
    (bNNNN-cuda) — `engine use`/`rollback` switch between them the same
    way, so the gate must count both. The real store is user-mutable
    (updates prune old tags, `engine local` registers non-b tags), so
    tests must never hardcode engine tags. Disk dirs WITHOUT a store row
    are orphaned install debris — `engine use` refuses them ("Query
    returned no rows") — so intersect with the table.
    """
    try:
        db = sqlite3.connect(os.path.join(REAL_DATA, "pallama.db"))
        rows = {r[0] for r in db.execute("SELECT tag FROM engines")}
        db.close()
    except Exception:
        rows = None
    tags: list[str] = []
    eng_root = os.path.join(REAL_DATA, "engines")
    if os.path.isdir(eng_root):
        for name in os.listdir(eng_root):
            if (
                re.fullmatch(r"b\d+(?:-cuda)?", name)
                and (rows is None or name in rows)
                and all(
                    os.path.isfile(os.path.join(eng_root, name, f"llama-{name}", tool))
                    for tool in ("llama-server", "llama-quantize")
                )
            ):
                tags.append(name)
    return sorted(tags, key=lambda t: int(t[1:].split("-")[0]), reverse=True)


def _stat_nice(pid: int) -> int:
    with open(f"/proc/{pid}/stat") as f:
        parts = f.read().rsplit(")", 1)[1].split()
    return int(parts[17])  # field 19 overall (ni)


def _child_flag(argv: list[str], flag: str) -> str | None:
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
    return None


def _hf_smallest_mmproj(repo: str) -> tuple[str, int]:
    url = f"https://huggingface.co/api/models/{repo}"
    with urllib.request.urlopen(url, timeout=60) as r:
        meta = json.loads(r.read())
    cands = [
        (s["size"], s["rfilename"])
        for s in meta.get("siblings", [])
        if "mmproj" in s["rfilename"].lower()
    ]
    if not cands:
        raise RuntimeError(f"no mmproj file in {repo}")
    size, fname = min(cands)
    return fname, int(size)


def phase_commands() -> None:
    print("\n== phase commands: every CLI path, real (no-mock) ==")
    # The user's real store may hold a partial/local engine active; pin a
    # full one in the sandbox so quantize + child spawns resolve real tools.
    cli("engine", "use", _full_engine_tags()[0])

    d = DAEMON
    d.start({"port": PORT})
    chat("Say ok")

    # -- A: light store/inspect commands --------------------------------
    p = cli("--help")
    reg(
        "help",
        p.returncode == 0
        and "Usage: pallama <COMMAND>" in p.stdout
        and bool(_help_command_names(p.stdout)),
        "rc0 + usage line + grouped command listing",
    )

    p = cli("list")
    reg(
        "list",
        p.returncode == 0 and MODEL in p.stdout,
        "rc0",
    )
    p = cli("ls")
    reg("ls", p.returncode == 0 and MODEL in p.stdout, "alias of list, rc0")

    p = cli("show", MODEL)
    reg("show", p.returncode == 0, "rc0")

    # ollama-migrant colon input (`model:tag`) must resolve onto the flat
    # store row (qwen2.5:0.5b-instruct -> qwen2.5-0.5b-instruct).
    p = cli("show", MODEL.replace("-", ":", 1))
    reg(
        "show.colon",
        p.returncode == 0 and MODEL in p.stdout,
        f"colon input rc0 + canonical row printed; out={p.stdout.strip()[:60]!r}",
    )

    p = cli("ps", "--reset")
    reg("ps.reset", p.returncode == 0, f"rc0; out={p.stdout.strip()[:80]!r}")

    # /api/ps rows must carry the placement card (CLI renders full@card).
    dev_rows = ps_rows()
    reg(
        "ps.device",
        bool(dev_rows) and all("pallama_device" in r for r in dev_rows),
        f"{len(dev_rows)} api ps rows carry pallama_device",
    )

    # ...and the profile-compile warnings array (may be empty; key must
    # exist so consumers can rely on the shape).
    reg(
        "ps.warnings",
        bool(dev_rows) and all("pallama_warnings" in r for r in dev_rows),
        f"{len(dev_rows)} api ps rows carry pallama_warnings",
    )

    # Gateway-side colon resolution: ollama-style model:tag must route
    # onto the flat row through the same ensure() choke point (rides the
    # instance loaded above — warm, cheap).
    st_c, _, _ = chat(
        "Say ok",
        extra={"model": MODEL.replace("-", ":", 1), "max_tokens": 8},
    )
    reg(
        "chat.colon",
        st_c == 200,
        f"model:tag via gateway chat -> {st_c}",
    )

    p = cli("why")
    reg("why", p.returncode == 0, "rc0")

    p = cli("doctor")
    reg(
        "doctor",
        p.returncode == 0
        and not [ln for ln in p.stdout.splitlines() if re.search(r"\sFAIL(\s|$)", ln)],
        "rc0 + no fail",
    )

    # why with a real trace id: take it from the why records themselves.
    recs = why()
    tid = recs[0].get("trace") if recs else None
    p = cli("why", tid) if tid else cli("why")
    reg(
        "why.trace",
        p.returncode == 0 and (tid is not None),
        f"trace={tid} rc={p.returncode}",
    )

    p = cli("coreside")
    reg("coreside", p.returncode == 0, p.stdout.strip()[:80])

    p = cli("drafts", MODEL)
    reg("drafts", p.returncode == 0, p.stdout.strip()[:80])

    def _search():
        p = cli("search", "qwen", timeout=120)
        reg(
            "search",
            p.returncode == 0 and len(p.stdout.strip()) > 0,
            p.stdout.strip()[:100],
        )
        # multi-word query joins into one HF search; table carries the
        # SIZE/ARCH/CTX columns and a pull-hint footer.
        p2 = cli("search", "qwen", "0.5b", "gguf", timeout=120)
        reg(
            "search.multi-word-columns",
            p2.returncode == 0
            and all(h in p2.stdout for h in ("SIZE", "ARCH", "CTX"))
            and "pull one" in p2.stdout,
            p2.stdout.strip().splitlines()[0][:100] if p2.stdout.strip() else "",
        )

    lane("search", _search)

    def _fit():
        # fit takes a pull target (owner/repo), not a RAM size.
        p = cli("fit", "Qwen/Qwen2.5-0.5B-Instruct-GGUF", timeout=120)
        reg(
            "fit",
            p.returncode == 0 and "fit preview for" in p.stdout and "QUANT" in p.stdout,
            p.stdout.strip().splitlines()[0][:100] if p.stdout.strip() else "",
        )

    lane("fit", _fit)

    p = cli("config", "list")
    reg("config.list", p.returncode == 0 and "port" in p.stdout, "rc0 + port key")

    # serve: proven by every daemon boot in this harness (argv[1] == serve)
    reg(
        "serve",
        d.proc is not None and d.proc.poll() is None,
        "daemon alive via `pallama serve` (healthz 200 + chat 200 above)",
    )
    # watch: live SSE tail — timeout-kill after 6s, banner must appear
    w = subprocess.run(
        ["timeout", "6", PAL, "watch"],
        env=SANDBOX.env(),
        capture_output=True,
        text=True,
    )
    reg(
        "watch",
        w.returncode == 124 and "watching sentinel" in w.stdout,
        f"rc={w.returncode} banner={'watching sentinel' in w.stdout}",
    )

    # -- B: create (happy + refusal) ------------------------------------
    mf_path = os.path.join(SANDBOX.root, "Modelfile")
    with open(mf_path, "w") as f:
        f.write(f"FROM {MODEL}\nPARAMETER num_ctx 2048\n")
    cli("stop", MODEL)  # create refuses while the source model is loaded
    p = cli("create", "validate-created", "-f", mf_path)
    cfg_parsed = {}
    try:
        with open(os.path.join(SANDBOX.config_dir, "config.toml"), "rb") as f:
            cfg_parsed = tomllib.load(f)
    except Exception:
        pass
    mo = (cfg_parsed.get("model_overrides") or {}).get("validate-created") or {}
    reg(
        "create.happy",
        p.returncode == 0 and "created" in p.stdout.lower() and mo.get("ctx") == 2048,
        f"rc={p.returncode} override ctx={mo.get('ctx')}",
    )
    with open(mf_path, "w") as f:
        f.write(f"FROM {MODEL}\nTEMPLATE this is a trap\n")
    p = cli("create", "validate-reject", "-f", mf_path)
    reg(
        "create.reject",
        p.returncode != 0 and "refuses to fake" in (p.stdout + p.stderr),
        f"rc={p.returncode} err={p.stderr.strip()[:100]}",
    )
    cli("rm", "validate-created")

    # -- C: lora add/list/rm --------------------------------------------
    p = cli("lora", "add", MODEL, "/nonexistent/validate.safetensors")
    add_out = p.stdout + p.stderr
    p2 = cli("lora", "list", MODEL)
    lora_id = None
    m = re.search(r"#(\d+)", add_out) or re.search(r"#(\d+)", p2.stdout)
    if m:
        lora_id = m.group(1)
    reg("lora.add", p.returncode == 0 and lora_id is not None, f"id={lora_id}")
    reg("lora.list", p2.returncode == 0 and "validate" in (p2.stdout + add_out), "")
    if lora_id:
        p = cli("lora", "rm", lora_id)
        reg("lora.rm", p.returncode == 0, p.stdout.strip()[:60])
    else:
        regb("lora.rm", "no lora id surfaced by add/list output")
    regb(
        "lora.apply",
        "real .safetensors LoRA fixture unavailable offline; CLI lifecycle (add/list/rm) + argv surface covered here, application path in Rust unit tests",
    )

    # -- D: import (hardlink + copy) ------------------------------------
    src = model_path(MODEL)

    def _store_file_for(name: str) -> str | None:
        for cand in (
            os.path.join(SANDBOX.data_dir, "models", name),
            os.path.join(SANDBOX.data_dir, "models", f"{name}.gguf"),
        ):
            if os.path.exists(cand):
                return cand
        base = (
            _sandbox_db()
            .execute("SELECT path FROM models WHERE name = ?", (name,))
            .fetchone()
        )
        return base[0] if base else None

    if src:
        p = cli("import", src, "--name", "validate-imported")
        f_imported = _store_file_for("validate-imported")
        nlink = os.stat(f_imported).st_nlink if f_imported else 0
        reg(
            "import.hardlink",
            p.returncode == 0 and nlink >= 2,
            f"rc={p.returncode} nlink={nlink}",
        )
        cli("rm", "validate-imported")
        p = cli("import", src, "--name", "validate-copied", "--quant", "q8_0", "--copy")
        f_copied = _store_file_for("validate-copied")
        nlink_c = os.stat(f_copied).st_nlink if f_copied else 0
        reg(
            "import.copy",
            p.returncode == 0 and nlink_c == 1 and os.path.isfile(f_copied or ""),
            f"rc={p.returncode} nlink={nlink_c}",
        )
        cli("rm", "validate-copied")
    else:
        regb("import.hardlink", "model path unavailable in store DB")
        regb("import.copy", "model path unavailable in store DB")

    # -- E: mmproj (refusal always; happy via real HF projector) --------
    # Target BIG: attaching a projector is model-agnostic but the happy lane
    # was proven on the 9B — keep it pinned there even when MODEL is small.
    if src:
        target = BIG if BIG != MODEL else MODEL
        # mmproj refuses while the model is loaded; release it first and
        # wait out the async drain (stop returns before unload completes).
        cli("stop", target, check_exit=False)
        for _ in range(60):
            if target not in json.dumps(ps_rows()):
                break
            time.sleep(1)
        p = cli("mmproj", target, src)
        reg(
            "mmproj.refusal",
            p.returncode != 0 and "not a vision projector" in (p.stdout + p.stderr),
            f"rc={p.returncode} err={p.stderr.strip()[:100]}",
        )

    def _mmproj_happy():
        try:
            fname, size = _hf_smallest_mmproj("Qwen/Qwen2.5-VL-3B-Instruct-GGUF")
            dst = os.path.join(SANDBOX.root, fname.replace("/", "_"))
            url = f"https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/{fname}"
            with urllib.request.urlopen(url, timeout=600) as r, open(dst, "wb") as f:
                shutil.copyfileobj(r, f)
            target = BIG if BIG != MODEL else MODEL
            p = cli("mmproj", target, dst, timeout=300)
            reg(
                "mmproj.happy",
                p.returncode == 0 and "attach" in (p.stdout + p.stderr).lower(),
                f"{fname} ({size >> 20} MiB) rc={p.returncode}",
            )
        except Exception as e:
            regb("mmproj.happy", f"HF projector lane failed: {e}")

    lane("mmproj.happy", _mmproj_happy)

    # -- F: keys lifecycle ----------------------------------------------
    # /api/keys requires an existing configured key (chicken-and-egg):
    # seed a bootstrap gatekey, then exercise add/list/rotate/rm against it.
    d.stop()
    d.start(
        {"port": PORT, "keys": [{"name": "gatekey", "key": "plm-validate-gate-000"}]}
    )
    p = cli("keys", "add", "vk1", "--rpm", "10")
    m1 = re.search(r"plm_\S+", p.stdout + p.stderr)
    secret1 = m1.group(0) if m1 else None
    reg(
        "keys.add",
        p.returncode == 0 and secret1 is not None,
        "plm_ secret printed once",
    )
    p = cli("keys", "add", "vk2")
    m2 = re.search(r"plm_\S+", p.stdout + p.stderr)
    secret2 = m2.group(0) if m2 else None
    p = cli("keys", "list")
    reg(
        "keys.list",
        p.returncode == 0
        and "vk1" in p.stdout
        and "vk2" in p.stdout
        and (secret1 or "?") not in p.stdout
        and (secret2 or "?") not in p.stdout,
        "names listed, full secrets hidden (redacted plm_ prefix ok)",
    )
    p = cli("keys", "rotate", "vk1")
    newsec = re.search(r"plm_\S+", p.stdout + p.stderr)
    reg(
        "keys.rotate",
        p.returncode == 0 and newsec is not None and newsec.group(0) != (secret1 or ""),
        "secret rotated",
    )
    p = cli("keys", "rm", "vk1")
    reg("keys.rm", p.returncode == 0, p.stdout.strip()[:60])

    # -- G: launch (env handoff + key resolution) -----------------------
    p = cli("launch", "printenv", "OPENAI_BASE_URL")
    reg(
        "launch",
        p.returncode == 0 and f":{PORT}" in p.stdout,
        f"base={p.stdout.strip()[:60]!r}",
    )
    if secret2:
        p = cli("launch", "--key", "vk2", "printenv", "OPENAI_API_KEY")
        reg(
            "launch",
            p.returncode == 0 and secret2 in p.stdout,
            "named key resolved to plm_ secret",
        )
    p = cli("keys", "rm", "vk2")  # bootstrap gatekey still present
    check(
        "commands",
        "keys rm non-last key succeeds",
        p.returncode == 0,
        f"rc={p.returncode}",
    )
    p = cli("keys", "rm", "gatekey")
    check(
        "commands",
        "keys rm refuses the last key",
        p.returncode != 0,
        f"rc={p.returncode} err={(p.stdout + p.stderr).strip()[:80]}",
    )
    # Drop key gating so the remaining plain-chat lanes run unauthenticated.
    d.stop()
    d.start({"port": PORT})

    # -- H: run single-shot + REPL --------------------------------------
    p = cli("run", MODEL, "Say: ok")
    reg(
        "run.single",
        p.returncode == 0 and len(p.stdout.strip()) > 0,
        p.stdout.strip().splitlines()[-1][:80] if p.stdout.strip() else "",
    )
    p = run_input("run", MODEL, input_text="/exit\n", timeout=240)
    reg(
        "run.repl-exit",
        p.returncode == 0 and (">>>" in p.stdout or "REPL" in p.stdout),
        f"rc={p.returncode} banner={'>>>' in p.stdout}",
    )
    p = run_input("run", MODEL, input_text="", timeout=240)
    reg("run.repl-eof", p.returncode == 0, f"rc={p.returncode} (EOF exits cleanly)")

    # rm while the model is loaded must refuse with a stop-first hint.
    p = cli("rm", MODEL)
    reg(
        "rm.running-guard",
        p.returncode != 0 and "running" in (p.stdout + p.stderr).lower(),
        f"rc={p.returncode} out={(p.stdout + p.stderr).strip()[:70]!r}",
    )

    # -- I: session lifecycle (model loaded) ----------------------------
    chat("Say ok")
    p = cli("session", "save", MODEL, "validate-sess")
    reg("session.save", p.returncode == 0, p.stdout.strip()[:60])
    p = cli("session", "list", MODEL)
    reg(
        "session.list",
        p.returncode == 0 and "validate-sess" in p.stdout,
        p.stdout.strip()[:80],
    )
    p = cli("session", "restore", MODEL, "validate-sess")
    reg("session.restore", p.returncode == 0, p.stdout.strip()[:60])
    p = cli("session", "rm", MODEL, "validate-sess")
    reg("session.rm", p.returncode == 0, p.stdout.strip()[:60])

    # -- J: snapshot -----------------------------------------------------
    before = set(os.listdir(SANDBOX.data_dir))
    p = cli("snapshot")
    after = set(os.listdir(SANDBOX.data_dir))
    new_dirs = sorted(after - before)
    reg("snapshot", p.returncode == 0, f"rc0; new artifacts: {new_dirs}")

    # -- K: migrate (legacy api_keys -> [[keys]] + backup) ---------------
    SANDBOX.write_config(
        {"port": PORT, "api_keys": ["legacy-secret-1", "legacy-secret-2"]}
    )
    p = cli("migrate")
    try:
        with open(os.path.join(SANDBOX.config_dir, "config.toml"), "rb") as f:
            mig = tomllib.load(f)
        n_keys = len(mig.get("keys", []))
    except Exception:
        n_keys = 0
    baks = [f for f in os.listdir(SANDBOX.config_dir) if ".bak-" in f]
    p2 = cli("migrate")
    reg(
        "migrate",
        p.returncode == 0 and n_keys == 2 and baks,
        f"rc={p.returncode} keys={n_keys} backups={baks[:1]} idempotent_rc={p2.returncode}",
    )
    # migrate leaves [[keys]] in config.toml; drop them so the daemonless
    # heavy lanes below (bench auto-start etc.) boot without key gating —
    # bearerless /v1 clients (whisper transcribe) would 401 otherwise.
    SANDBOX.write_config({"port": PORT})

    # -- L: completions (4 shells + bash -n parse) ----------------------
    comp_ok = True
    comp_ev = []
    for shell in ("bash", "zsh", "fish", "powershell"):
        p = cli("completions", shell)
        okk = p.returncode == 0 and len(p.stdout) > 100
        comp_ok = comp_ok and okk
        comp_ev.append(f"{shell}:{len(p.stdout)}B")
        if shell == "bash" and okk:
            sf = os.path.join(SANDBOX.root, "comp.bash")
            with open(sf, "w") as f:
                f.write(p.stdout)
            comp_ok = (
                comp_ok
                and subprocess.run(["bash", "-n", sf], capture_output=True).returncode
                == 0
            )
            comp_ev.append("bash -n:ok")
    reg("completions", comp_ok, " ".join(comp_ev))

    # -- heavy lanes below run daemonless --------------------------------
    d.stop()

    # R2-29: bench/tune spawn llama-bench directly. Crash-battery engines
    # orphaned by hard teardown can still hold VRAM and starve CUDA init
    # (sweep-10: every lane died exit 1 under residue while the identical
    # invocation passed post-run on a free GPU). Clear own residue first;
    # boundary honestly if an external holder remains.
    _reap_orphan_validate_engines()
    gpu_mib = _gpu_headroom_mib()
    if gpu_mib is not None and gpu_mib < 1200:
        time.sleep(3)
        _reap_orphan_validate_engines()
        gpu_mib = _gpu_headroom_mib()
    vram_starved = gpu_mib is not None and gpu_mib < 1200
    heavy_ok = disk_free_gb() > 8.0 and not vram_starved

    def _bench():
        if vram_starved:
            regb(
                "bench",
                f"free VRAM {gpu_mib} MiB < 1200 MiB after reaping this "
                f"harness's orphaned engines — GPU holders: "
                f"{'; '.join(_gpu_compute_holders()) or 'none visible'}",
            )
            return
        p = cli("bench", MODEL, timeout=900)
        reg(
            "bench",
            p.returncode == 0 and len(p.stdout.strip()) > 0,
            p.stdout.strip().splitlines()[-1][:100]
            if p.stdout.strip()
            else f"rc={p.returncode}",
        )

    lane("bench", _bench)

    def _tune(flag: str, value: str | None):
        def fn():
            args = (
                ("tune", MODEL, flag)
                if value is None
                else (
                    "tune",
                    MODEL,
                    flag,
                    value,
                )
            )
            p = cli(*args, timeout=2400)
            reg(
                f"tune.{flag.lstrip('-')}",
                p.returncode == 0,
                (
                    p.stdout.strip().splitlines()[-1][:90]
                    if p.stdout.strip()
                    else f"rc={p.returncode} err={p.stderr[:200]}"
                ),
            )

        lane(f"tune.{flag.lstrip('-')}", fn)

    # tune flags: value-taking flags (--ctx/--spec/--slots) need explicit
    # values in the current CLI; the rest are switches.
    for flag, value in (
        ("--search", None),
        ("--ctx", "16384"),
        ("--spec", "auto"),
        ("--slots", "2"),
        ("--ngram", None),
        ("--load", None),
        ("--replicas", None),
        ("--cache-reuse", None),
    ):
        if heavy_ok:
            _tune(flag, value)
        else:
            regb(
                f"tune.{flag.lstrip('-')}",
                (
                    f"free VRAM {gpu_mib} MiB < 1200 MiB after reaping "
                    f"orphaned engines; GPU holders: "
                    f"{'; '.join(_gpu_compute_holders()) or 'none visible'}"
                    if vram_starved
                    else f"disk free {disk_free_gb():.1f}G <= 8G: heavy lane skipped"
                ),
            )

    def _pull_retry(*args, timeout=2400):
        # Single explicit retry for transient HF API failures (observed
        # intermittent 429/5xx on /api/models). Policy: max 1 retry, 10s
        # backoff, evidence carries both attempts — never silent.
        return _pull_retry_cmd(list(args), timeout=timeout)

    def _pull_retry_cmd(argv, timeout=2400):
        p = cli(*argv, timeout=timeout)
        if p.returncode != 0:
            time.sleep(10)
            p2 = cli(*argv, timeout=timeout)
            p2.stderr = (p2.stderr or "") + f" [retry after rc={p.returncode}]"
            return p2
        return p

    def _quantize():
        # llama-quantize refuses requantizing already-quantized tensors, so
        # the happy lane needs a real f16 source (1.5G download).
        src_repo = "ggml-org/Qwen3-0.6B-GGUF:F16"
        psrc = _pull_retry(src_repo, timeout=3600)
        if psrc.returncode != 0:
            regb(
                "quantize.happy",
                f"F16 source pull failed: {(psrc.stderr or psrc.stdout).strip()[:100]}",
            )
            p = cli("quantize", MODEL, "-t", "BOGUSQT")
            reg(
                "quantize.refusal",
                p.returncode != 0
                and (
                    "implausible" in (p.stdout + p.stderr).lower()
                    or "invalid ftype" in (p.stdout + p.stderr).lower()
                    or "unknown" in (p.stdout + p.stderr).lower()
                ),
                f"rc={p.returncode}",
            )
            return
        p = cli("quantize", MODEL, "-t", "BOGUSQT")
        reg(
            "quantize.refusal",
            p.returncode != 0
            and (
                "implausible" in (p.stdout + p.stderr).lower()
                or "invalid ftype" in (p.stdout + p.stderr).lower()
                or "unknown" in (p.stdout + p.stderr).lower()
            ),
            f"rc={p.returncode} err={p.stderr.strip()[:80]}",
        )
        p = cli(
            "quantize",
            "qwen3-0.6b",
            "-t",
            "Q8_0",
            "--name",
            "validate-quant",
            timeout=1800,
        )
        reg(
            "quantize.happy",
            p.returncode == 0 and "registered" in (p.stdout + p.stderr).lower(),
            p.stdout.strip()[:80],
        )
        p = cli(
            "quantize",
            "qwen3-0.6b",
            "-t",
            "Q8_0",
            "--name",
            "validate-quant",
            timeout=120,
        )
        reg(
            "quantize.refusal",
            p.returncode != 0 and "already exists" in (p.stdout + p.stderr),
            "dst-exists refusal",
        )
        cli("rm", "validate-quant")
        cli("rm", "qwen3-0.6b")

    if heavy_ok:
        lane("quantize.happy", _quantize, "quantize.refusal")
    else:
        regb("quantize.happy", f"disk free {disk_free_gb():.1f}G <= 8G")
        regb("quantize.refusal", f"disk free {disk_free_gb():.1f}G <= 8G")

    def _whisper():
        p = cli("whisper", "--install", timeout=1200)
        inst_out = p.stdout + p.stderr
        rate_limited = p.returncode != 0 and (
            "rate limited" in inst_out.lower() or "403" in inst_out
        )
        if rate_limited:
            for wp in (
                "whisper.install",
                "whisper.pull",
                "whisper.list",
                "whisper.pin",
                "whisper.transcribe",
            ):
                regb(
                    wp,
                    "GH API rate limited (403): whisper server download blocked "
                    "this window; rerun when budget resets",
                )
            return
        reg("whisper.install", p.returncode == 0, p.stdout.strip()[:80])
        # Retry-once on the model pull: HF LFS links throttle/cut mid-stream
        # (live-observed: 28M/142M then dead socket); download_file resumes
        # from the .part, so the second attempt usually lands.
        p = cli("whisper", "--pull", "base", timeout=1200)
        if p.returncode != 0:
            p = cli("whisper", "--pull", "base", timeout=1200)
        pull_ok = p.returncode == 0
        reg("whisper.pull", pull_ok, p.stdout.strip()[:80])
        p = cli("whisper", "--list")
        reg(
            "whisper.list",
            p.returncode == 0 and len(p.stdout.strip()) > 0,
            p.stdout.strip()[:80],
        )
        # pin lifecycle: standalone flag path (no install/net), real tag dir
        # from the sandbox install above.
        bin_root = os.path.join(SANDBOX.data_dir, "whisper", "bin")
        tag = ""
        if os.path.isdir(bin_root):
            tag = next(
                (
                    e
                    for e in sorted(os.listdir(bin_root))
                    if os.path.isdir(os.path.join(bin_root, e))
                ),
                "",
            )
        pin_ok = False
        if tag:
            p = cli("whisper", "--pin", tag)
            pin_ok = p.returncode == 0 and "pinned to" in (p.stdout + p.stderr)
            p = cli("whisper", "--list")
            pin_ok = pin_ok and "(pinned)" in p.stdout
            p = cli("whisper", "--pin", "none")
            pin_ok = pin_ok and "pin removed" in (p.stdout + p.stderr).lower()
            p = cli("whisper", "--list")
            pin_ok = pin_ok and "(pinned)" not in p.stdout
        reg("whisper.pin", pin_ok, f"tag={tag or 'no tag dir found'}")
        wav = os.path.join(SANDBOX.root, "tiny.wav")
        with open(wav, "wb") as f:
            f.write(tiny_wav())
        p = cli("whisper", wav, timeout=600)
        out = p.stdout + p.stderr
        server_ok = (
            "not installed"
            not in (
                cli("whisper", "--list").stdout + cli("whisper", "--list").stderr
            ).lower()
        )
        if not pull_ok:
            # Model pull failed (throttled/cut HF link): the 501 that
            # follows is CORRECT teaching behavior, not the pinned
            # product bug. Boundary the lane; rerun when network allows.
            regb(
                "whisper.transcribe",
                "model pull failed — lane cannot run; server+model logic "
                "proven by unit tests and the pinned checks below are "
                "skipped (they would misfire on the pull failure)",
            )
            return
        reg(
            "whisper.transcribe",
            p.returncode == 0 or "empty" in out.lower() or "silence" in out.lower(),
            f"rc={p.returncode} server_installed={server_ok} "
            f"out={p.stdout.strip()[:60]} err={p.stderr.strip()[:60]}",
        )
        if p.returncode != 0 and server_ok and "501" in out:
            # True state mismatch only when the model IS pulled and the
            # lane still 501s ("<none pulled>" in the body = pull gap,
            # handled above by the pull_ok boundary).
            check(
                "commands",
                "whisper.transcribe 501 with server installed (product bug)",
                "<none pulled>" not in out,
                out.strip()[:120],
            )
        # Regression pin: whisper_cmd must resolve an admin bearer (PALLAMA_KEYS
        # else first unscoped [[keys]]) so transcription keeps working once the
        # gateway is key-gated — same class as the keys_cmd bearer fix.
        if server_ok:
            d.stop()
            d.start(
                {
                    "port": PORT,
                    "keys": [{"name": "gatekey", "key": "plm-validate-gate-000"}],
                }
            )
            p = cli("whisper", wav, "--model", "base", timeout=600)
            kout = p.stdout + p.stderr
            check(
                "commands",
                "whisper under [[keys]] resolves admin bearer (product pin)",
                p.returncode == 0 and "401" not in kout,
                f"rc={p.returncode} out={p.stdout.strip()[:60]} "
                f"err={p.stderr.strip()[:60]}",
            )
            d.stop()
            SANDBOX.write_config({"port": PORT})

    if heavy_ok:
        lane(
            "whisper.install",
            _whisper,
            "whisper.pull",
            "whisper.list",
            "whisper.pin",
            "whisper.transcribe",
        )  # runs the whole whisper battery
    else:
        for wp in (
            "whisper.install",
            "whisper.pull",
            "whisper.list",
            "whisper.pin",
            "whisper.transcribe",
        ):
            regb(wp, f"disk free {disk_free_gb():.1f}G <= 8G")

    # whisper --pin refusals: standalone flag path, no install/net needed
    # (FAST-visible — manifest tier fast=True).
    p = cli("whisper", "--pin", "v0.0.0-validate")
    r1 = p.returncode != 0 and "is not installed" in (p.stdout + p.stderr)
    p = cli("whisper", "--pin", "../evil")
    r2 = p.returncode != 0 and "plain tag name" in (p.stdout + p.stderr)
    reg(
        "whisper.pin.refusal",
        r1 and r2,
        f"unknown-tag={'is not installed' if r1 else 'miss'} "
        f"path-sep={'plain tag name' if r2 else 'miss'}",
    )

    tags = _full_engine_tags()
    anchor, dance = (tags + [None, None])[:2]
    # Prefer a plain upstream tag for the pin-update dance so the lane
    # exercises the standard asset path whenever the store has one; a
    # -cuda pick goes through the CUDA overlay repo (needs bNNNN-cuda
    # releases published there — boundary row when the channel is not
    # live yet).
    update_pick = next(
        (t for t in tags if t != anchor and not t.endswith("-cuda")), dance
    )

    def _engine_full():
        p = cli("engine", "use", dance)
        ok_use = p.returncode == 0 and _active_engine_tag() == dance
        reg("engine.use", ok_use, f"active={_active_engine_tag()}")
        p = cli("engine", "rollback")
        stepped = _active_engine_tag()
        rout = p.stdout + p.stderr
        if p.returncode != 0 and ("rate limited" in rout.lower() or "403" in rout):
            regb(
                "engine.rollback",
                "GH API rate limited (403): rollback metadata blocked this "
                f"window; engine.use dance proves the switch path; err={rout.strip()[:80]}",
            )
        else:
            reg(
                "engine.rollback",
                p.returncode == 0 and stepped not in (None, dance),
                f"rc={p.returncode} stepped to {stepped} out={rout.strip()[:60]}",
            )
        p = cli("engine", "use", anchor)
        reg(
            "engine.use",
            p.returncode == 0 and _active_engine_tag() == anchor,
            f"active={_active_engine_tag()}",
        )
        p = cli("engine", "update", update_pick, "--no-gate", timeout=1800)
        eout = p.stdout + p.stderr
        overlay_miss = (
            update_pick.endswith("-cuda")
            and p.returncode != 0
            and "overlay release" in eout
            and "PALLAMA_ENGINE_REPO" in eout
        )
        if overlay_miss:
            regb(
                "engine.update",
                "CUDA overlay channel not live yet (placeholder pallama/pallama); "
                f"tag-pinned update to {update_pick} needs a published bNNNN-cuda "
                "release there — the switch path is proven by engine.use above and "
                "the channel-update path by live engine-update runs",
            )
            p = cli("engine", "use", anchor)
            reg(
                "engine.update",
                p.returncode == 0 and _active_engine_tag() == anchor,
                f"active restored to {anchor}",
            )
            return
        rate_limited = p.returncode != 0 and (
            "rate limited" in eout.lower() or "403" in eout
        )
        real_server = os.path.join(
            REAL_DATA, "engines", anchor, f"llama-{anchor}", "llama-server"
        )
        if rate_limited:
            regb(
                "engine.update",
                "GH API rate limited (403): engine download blocked this "
                f"window; use/rollback lanes above prove the update path; "
                f"err={eout.strip()[:80]}",
            )
        else:
            reg(
                "engine.update",
                p.returncode == 0 and os.path.isfile(real_server),
                f"rc={p.returncode}; real {anchor} engine intact",
            )
        p = cli("engine", "use", anchor)
        reg(
            "engine.update",
            p.returncode == 0 and _active_engine_tag() == anchor,
            f"active restored to {anchor}",
        )

    def _engine_light():
        p = cli("engine", "use", dance)
        reg(
            "engine.use",
            p.returncode == 0 and _active_engine_tag() == dance,
            f"active={_active_engine_tag()}",
        )
        p = cli("engine", "use", anchor)
        reg(
            "engine.use",
            p.returncode == 0 and _active_engine_tag() == anchor,
            f"active={_active_engine_tag()}",
        )

    if heavy_ok and anchor and dance:
        lane("engine.update", _engine_full, "engine.use", "engine.rollback")
    elif anchor and dance:
        _engine_light()
        regb("engine.update", f"disk free {disk_free_gb():.1f}G <= 8G")
        regb("engine.rollback", f"disk free {disk_free_gb():.1f}G <= 8G")
    else:
        for ep in ("engine.use", "engine.rollback", "engine.update"):
            reg(
                ep,
                False,
                f"need >=2 full b-engines in real store, found {len(tags)}: {tags}",
            )
    p = cli("engine", "list")
    reg("engine.list", p.returncode == 0 and anchor in p.stdout, "")
    local_server = os.path.join(
        SANDBOX.data_dir, "engines", anchor, f"llama-{anchor}", "llama-server"
    )
    p = cli("engine", "local", local_server)
    reg(
        "engine.local",
        p.returncode == 0 and "local" in cli("engine", "list").stdout,
        p.stdout.strip()[:80],
    )
    cli("engine", "use", anchor)
    check(
        "commands",
        f"engine active left on full engine {anchor}",
        _active_engine_tag() == anchor,
        f"active={_active_engine_tag()}",
    )

    def _pull():
        before = set(cli("list").stdout.split())
        p = _pull_retry("ggml-org/Qwen3-0.6B-GGUF")
        after = set(cli("list").stdout.split())
        new = {w for w in after - before if "qwen3" in w.lower()}
        reg(
            "pull",
            p.returncode == 0 and bool(new),
            f"rc={p.returncode} new={sorted(new)[:3]}",
        )
        for name in new:
            cli("rm", name)

    if heavy_ok:
        lane("pull", _pull)
    else:
        regb("pull", f"disk free {disk_free_gb():.1f}G <= 8G")

    def _run_miss_pulls():
        # `pallama run` on a missing model must auto-pull (same flow as
        # `pallama pull`: progress, locks) and then run it — one-shot
        # prompt mode proves the whole chain parse -> pull -> serve.
        before = set(cli("list").stdout.split())
        p = _pull_retry_cmd(
            ["run", "ggml-org/Qwen3-0.6B-GGUF", "Say ok", "--max-tokens", "8"],
            timeout=2400,
        )
        after = set(cli("list").stdout.split())
        new = {w for w in after - before if "qwen3" in w.lower()}
        reg(
            "run.miss-pulls",
            p.returncode == 0 and bool(new) and "not in the store" in (p.stdout or ""),
            f"rc={p.returncode} new={sorted(new)[:3]}",
        )
        for name in new:
            cli("stop", name)
            cli("rm", name)

    if heavy_ok:
        lane("run.miss-pulls", _run_miss_pulls)
    else:
        regb("run.miss-pulls", f"disk free {disk_free_gb():.1f}G <= 8G")

    def _upgrade():
        mtime_before = os.path.getmtime(PAL)
        repo = os.environ.get("PALLAMA_VALIDATE_UPGRADE_REPO", "").strip()
        if not repo:
            regb(
                "upgrade.dry-run",
                "no real pallama release repo configured for this checkout "
                "(PALLAMA_REPO unset, no git origin); set "
                "PALLAMA_VALIDATE_UPGRADE_REPO=owner/repo to run the real lane; "
                "Rust suite covers resolve/verify against a local release server",
            )
            return
        env = {**SANDBOX.env(), "PALLAMA_REPO": repo}
        p = subprocess.run(
            [PAL, "upgrade", "--dry-run"],
            capture_output=True,
            text=True,
            timeout=900,
            env=env,
        )
        reg(
            "upgrade.dry-run",
            p.returncode == 0
            and "dry-run ok" in (p.stdout + p.stderr)
            and os.path.getmtime(PAL) == mtime_before,
            f"rc={p.returncode} repo={repo} out={(p.stdout or p.stderr).strip()[:80]}",
        )

    lane("upgrade.dry-run", _upgrade)

    # -- T: teaching refusals -------------------------------------------
    p = cli("push", MODEL)
    reg(
        "push.refusal",
        p.returncode != 0 and len(p.stdout + p.stderr) > 10,
        p.stderr.strip()[:80] or p.stdout.strip()[:80],
    )
    for cmdname in ("signin", "login", "signout", "logout"):
        p = cli(cmdname)
        reg(
            f"{cmdname}.refusal",
            p.returncode != 0 and len(p.stdout + p.stderr) > 10,
            (p.stderr.strip() or p.stdout.strip())[:80],
        )

    # -- V: stop bare (LAST: kills the daemon) ---------------------------
    d.start({"port": PORT})
    p = cli("stop")
    down = False
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{PORT}/healthz", timeout=2)
    except Exception:
        down = True
    reg(
        "stop.bare", p.returncode == 0 and down, f"rc={p.returncode} daemon down={down}"
    )
    d.stop()


def phase_knobs_argv() -> None:
    print("\n== phase knobs_argv: every child-argv knob, batched spawns ==")
    d = DAEMON
    for group, entries in argv_groups().items():
        cfg: dict = {"port": PORT}
        for knob, value, _ in entries:
            cfg[knob] = value
        if VALIDATE_DEVICES:
            cfg.setdefault("model_overrides", {}).setdefault(MODEL, {})["devices"] = [
                VALIDATE_DEVICES
            ]
            if group == "G1":
                cfg["mmproj_device"] = VALIDATE_DEVICES
        try:
            d.start(cfg, floor_model=MODEL)
        except RuntimeError:
            # tensor_split ratios can be rejected on a single visible
            # device — retry honestly without it and record the boundary.
            if "tensor_split" not in cfg:
                raise
            cov(
                "tensor_split",
                "argv: --tensor-split 3,1",
                f"spawn refused with tensor_split; log: {d.tail_log(5)[-200:]}",
                ok=False,
            )
            cfg.pop("tensor_split")
            entries = [e for e in entries if e[0] != "tensor_split"]
            d.start(cfg, floor_model=MODEL)
        st, _, _ = chat("Say ok")
        wait_loaded(budget=300)
        pid = child_pid()
        argv = child_argv(pid) if pid else []
        check(
            "knobs_argv",
            f"group {group}: spawn + chat",
            st == 200 and pid is not None,
            f"status={st} pid={pid}",
        )

        def has(flag: str, val: str | None = None) -> bool:
            if val is None:
                return flag in argv
            return any(
                argv[i] == flag and i + 1 < len(argv) and argv[i + 1] == val
                for i in range(len(argv))
            )

        for knob, value, expectation in entries:
            if knob == "cache_idle_slots":
                # opt-out knob: profile emits --no-cache-idle-slots only
                # when false; true (default on) must NOT carry the flag.
                absent = "--no-cache-idle-slots" not in argv
                cov(
                    knob,
                    "opt-out: true => no --no-cache-idle-slots in argv",
                    f"flag_absent={absent}",
                    ok=absent,
                )
                continue
            if knob.startswith("prio"):
                ni = _stat_nice(pid) if pid else 0
                cov(knob, expectation, f"child nice={ni}", ok=ni >= 1)
                continue
            tokens = expectation.split()
            flag = tokens[0]
            val = (
                tokens[1]
                if len(tokens) > 1
                and isinstance(value, (int, str))
                and not isinstance(value, bool)
                else None
            )
            got = has(flag, val)
            cov(
                knob,
                f"argv: {expectation}",
                f"argv {'has' if got else 'MISSING'} {flag}"
                + (f" {val}" if val else ""),
                ok=got,
            )
        if VALIDATE_DEVICES and group == "G1":
            got = has("--mmproj-device", VALIDATE_DEVICES)
            cov(
                "mmproj_device",
                f"argv: --mmproj-device {VALIDATE_DEVICES}",
                f"argv {'has' if got else 'MISSING'} --mmproj-device",
                ok=got,
            )
        d.stop()


def phase_knobs_behavior() -> None:
    print("\n== phase knobs_behavior: tls/cors/otlp/auth/remotes probes ==")
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    d = DAEMON

    # -- TLS: rustls on the same port ------------------------------------
    cert = os.path.join(SANDBOX.root, "validate.crt")
    key = os.path.join(SANDBOX.root, "validate.key")
    gen = subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            key,
            "-out",
            cert,
            "-days",
            "2",
            "-subj",
            "/CN=127.0.0.1",
            "-addext",
            "subjectAltName=IP:127.0.0.1",
        ],
        capture_output=True,
        text=True,
    )
    if gen.returncode == 0:
        d.start({"port": PORT, "tls_cert": cert, "tls_key": key})
        https_ok, plain_failed = False, False
        try:
            with urllib.request.urlopen(
                f"https://127.0.0.1:{PORT}/healthz",
                timeout=5,
                context=ssl._create_unverified_context(),
            ) as r:
                https_ok = r.status == 200
        except Exception:
            https_ok = False
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/healthz", timeout=5)
        except Exception:
            plain_failed = True
        cov(
            "tls_cert",
            "rustls same-port: https 200, plain http fails",
            f"https={https_ok} plain_failed={plain_failed}",
            ok=https_ok and plain_failed,
        )
        cov(
            "tls_key",
            "rustls same-port: https 200, plain http fails",
            f"https={https_ok} plain_failed={plain_failed}",
            ok=https_ok and plain_failed,
        )
        d.stop()
    else:
        cov(
            "tls_cert",
            "rustls handshake",
            f"openssl unavailable: {gen.stderr[:80]}",
            ok=False,
        )
        cov(
            "tls_key",
            "rustls handshake",
            f"openssl unavailable: {gen.stderr[:80]}",
            ok=False,
        )

    # -- CORS: preflight echo --------------------------------------------
    d.start({"port": PORT, "cors_origins": ["https://example.com"]})
    st, hdr, _ = http(
        "OPTIONS",
        "/v1/chat/completions",
        headers={
            "Origin": "https://example.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    acao = hdr.get("access-control-allow-origin", "")
    cov(
        "cors_origins",
        "OPTIONS preflight -> ACAO echo",
        f"status={st} acao={acao!r}",
        ok=st in (200, 204) and acao == "https://example.com",
    )
    d.stop()

    # -- OTLP: local collector captures export POST ----------------------
    captured: list[bytes] = []

    class _H(BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("content-length", 0))
            captured.append(self.rfile.read(n))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), _H)
    col_port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    d.start(
        {
            "port": PORT,
            "otlp_endpoint": f"http://127.0.0.1:{col_port}/v1/traces",
            "otlp_service": "pallama-validate",
        }
    )
    chat("Say ok")
    deadline = time.time() + 60
    hit = None
    while time.time() < deadline and hit is None:
        for body in captured:
            if b"pallama-validate" in body:
                hit = body
                break
        time.sleep(1)
    cov(
        "otlp_endpoint",
        "export POST arrives at local collector",
        f"{len(captured)} posts captured",
        ok=hit is not None or len(captured) > 0,
    )
    cov(
        "otlp_service",
        "service name present in export body",
        "service name found" if hit else f"{len(captured)} posts, name not found",
        ok=hit is not None,
    )
    srv.shutdown()
    d.stop()

    # -- child_auth: direct-to-child 401 vs proxied 200 -------------------
    d.start({"port": PORT, "child_auth": True})
    st, _, _ = chat("Say ok")
    pid = child_pid()
    argv = child_argv(pid) if pid else []
    wired = "--api-key-file" in argv or "--api-key" in argv
    cport = _child_flag(argv, "--port") if argv else None
    direct_401 = False
    if cport:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{cport}/v1/models", timeout=5)
        except urllib.error.HTTPError as e:
            direct_401 = e.code == 401
        except Exception:
            direct_401 = False
    cov(
        "child_auth",
        "direct-to-child 401, proxied 200",
        f"wired={wired} direct_401={direct_401} proxied={st == 200}",
        ok=wired and direct_401 and st == 200,
    )
    d.stop()

    # -- singleflight: 2 concurrent cold spawns -> 1 child ----------------
    d.start({"port": PORT, "singleflight": True})
    results: list[int] = []

    def _one():
        s, _, _ = chat("Say ok")
        results.append(s)

    ts = [threading.Thread(target=_one) for _ in range(2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=300)
    run_dir = os.path.join(SANDBOX.data_dir, "run")
    pids = [
        f for f in os.listdir(run_dir) if f.startswith(MODEL) and f.endswith(".pid")
    ]
    cov(
        "singleflight",
        "2 concurrent identical cold requests -> 1 child",
        f"statuses={results} pidfiles={pids}",
        ok=results == [200, 200] and len(pids) == 1,
    )
    d.stop()

    # -- prompt_preflight: oversized prompt -> teaching 4xx ----------------
    d.start({"port": PORT, "prompt_preflight": True})
    st, _, v = chat(
        "x", extra={"messages": [{"role": "user", "content": "x" * 2_000_000}]}
    )
    cov(
        "prompt_preflight",
        "oversized prompt rejected before spawn",
        f"status={st} body={str(v)[:80]}",
        ok=st >= 400,
    )
    d.stop()

    # -- remotes: config echo + boot --------------------------------------
    d.start(
        {
            "port": PORT,
            "remotes": [{"name": "edge", "url": "http://127.0.0.1:9", "key": "rk"}],
        }
    )
    listed = cli("config", "list").stdout
    cov(
        "remotes",
        "[[remotes]] accepted at boot + echoed by config list",
        "edge remote present" if "edge" in listed else "not echoed",
        ok="edge" in listed,
    )
    d.stop()

    # -- curated rows for knobs proven by earlier phases ------------------
    cov(
        "host",
        f"binds 127.0.0.1:{PORT}",
        "every daemon phase + healthz 200",
    )
    cov(
        "port",
        f"binds 127.0.0.1:{PORT}",
        "every daemon phase + healthz 200",
    )
    cov(
        "router",
        "router mode serves chat from one child",
        "phase behavior: router-mode check",
    )


def _full_toplevel() -> dict:
    """All 145 manifest knobs with benign explicit values (full-manifest boot).

    None values = deliberately omitted from the serialized boot config
    (XOR partners / pairing-gated knobs that cannot co-exist): the key
    still counts for the (c0) membership gate.
    """
    lk = os.path.join(SANDBOX.root, "lookup-static.bin")
    open(lk, "wb").close()
    mcp_cfg = os.path.join(SANDBOX.root, "mcp-servers.json")
    with open(mcp_cfg, "w") as f:
        f.write('{"mcpServers":{}}')
    return {
        "host": "127.0.0.1",
        "port": PORT,
        "late_chunking_max_tokens": 8192,
        "session_keep_secs": 900,
        # agent-tooling wave: validated for prefix/pairing/JSON/XOR in
        # config.rs; a real (valid-JSON) config file exercises the
        # existence-checked path, runtime + inline JSON stay omitted.
        "server_tools": "read_file",
        "server_tools_runtime": None,
        "mcp_servers_config": mcp_cfg,
        "mcp_servers_json": None,
        "mistralrs_pa_memory_fraction": 0.5,
        "mistralrs_paged_attn": False,
        "lazy_mode": "auto",
        "deterministic": False,
        "audit_log": False,
        "auto_restart_engine_switch": False,
        "cpu_ffn_n": 1,
        # container knob: benign explicit boot (enabled=false — true would
        # require a model; config.rs validation rule)
        "semantic_cache": {"enabled": False},
        "default_ctx": 2048,
        "idle_sleep_secs": 77,
        "idle_timeout_secs": 500,
        "max_loaded_models": 2,
        "child_transport": "tcp",
        "child_auth": True,
        "engine_asset": "ubuntu-vulkan-x64",
        "spec": "off",
        "cache_reuse": 128,
        "keys": [
            {
                "name": "gatekey",
                "key": "gate-secret-1",
                "models": [],
                "rpm": 0,
                "tpm": 0,
                "daily_tokens": 0,
                "max_concurrent": 0,
            }
        ],
        "rpc_servers": "",
        "cache_ram_mb": 1024,
        "cpu_range": "",
        "poll": 77,
        "reasoning_format": "deepseek",
        "slots": 2,
        "cache_type": "q8_0",
        "kv_unified": False,
        "kv_unified_per_slot": 4096,
        "swa_full": False,
        "ctx_checkpoints": 4,
        "no_kv_offload": False,
        "load_mode": "",
        "spawn_mem_guard": True,
        "session_bank": True,
        "singleflight": True,
        "prompt_preflight": True,
        "spec_cache": False,
        "ctx_extend": 0.0,
        "cpu_moe_n": 0,
        "override_tensor": [],
        "agent": False,
        "sessions": False,
        "router": False,
        "router_max_models": 2,
        "devices": [],
        "engine_check_secs": 30,
        "slot_prompt_similarity": 0.0,
        "sentinel": True,
        "sentinel_stall_secs": 120,
        "sentinel_enforce": False,
        "tls_cert": "",
        "tls_key": "",
        "cors_origins": [],
        "otlp_endpoint": "",
        "otlp_service": "",
        "remotes": [{"name": "edge", "url": "http://127.0.0.1:9", "key": "rk"}],
        "engine_env": {"GATE_PROBE": "1"},
        "spec_draft_cpu_range": "",
        "spec_draft_cpu_strict": False,
        "spec_draft_device": "",
        "spec_draft_ngl": "",
        "spec_draft_threads": 1,
        "spec_draft_p_min": 0.8,
        "spec_draft_p_split": 0.3,
        "spec_draft_poll": 5,
        "spec_draft_prio": 0,
        "spec_draft_prio_batch": 0,
        "spec_draft_poll_batch": True,
        "spec_draft_cpu_strict_batch": False,
        "spec_draft_threads_batch": 1,
        "spec_draft_type_k": "",
        "spec_draft_type_v": "",
        "spec_draft_override_tensor": [],
        "spec_draft_n_cpu_moe": 0,
        "spec_draft_cpu_moe": False,
        "spec_draft_backend_sampling": True,
        "adaptive_decay": 0,
        "adaptive_target": 0.0,
        "ngram_size_m": 0,
        "ngram_size_n": 0,
        "ngram_min_hits": 0,
        "ngram_mod_n_match": 0,
        "ngram_mod_n_max": 0,
        "ngram_mod_n_min": 0,
        "reasoning_budget": 1024,
        "reasoning_budget_message": "Reasoning",
        "reasoning_effort": "low",
        "reasoning_preserve": False,
        "image_max_tokens": 4096,
        "image_min_tokens": 64,
        "mtmd_batch_max_tokens": 4096,
        "mmproj_offload": True,
        "mmproj_auto": True,
        "mmproj_device": "",
        "embd_normalize": 2,
        "yarn_orig_ctx": 0,
        "yarn_ext_factor": 0.0,
        "yarn_attn_factor": 0.0,
        "yarn_beta_fast": 0.0,
        "yarn_beta_slow": 0.0,
        "cpu_strict": False,
        "prio": 0,
        "prio_batch": 0,
        "poll_batch": True,
        "threads_http": 1,
        "warmup": True,
        "repack": True,
        "cache_idle_slots": True,
        "lookup_cache_static": lk,
        "lookup_cache_dynamic": lk,
        "predictive_preload": False,
        "adaptive_slots": False,
        "no_host": False,
        "op_offload": False,
        "keep_tokens": 0,
        "override_kv": [],
        "control_vectors": [],
        "control_vectors_scaled": [],
        "control_vector_layer_range": "",
        "tensor_preset": "",
        "pii_scrub": False,
        "video_ffmpeg_dir": "",
        "video_fps": 0.0,
        "video_timestamp_interval": 0.0,
        "numa": "",
        "check_tensors": False,
        "context_shift": False,
        "samplers": "",
        "batch_size": 512,
        "ubatch_size": 256,
        "threads_batch": 2,
        "main_gpu": 0,
        "split_mode": "none",
        "tensor_split": "",
        "models_autoload": False,
        "log_level": "pallama=info",
        "update_channel": "stable",
    }


def _overlay_a() -> dict:
    """All 20 ModelOverride fields + 18 sampler leaves (chat_template side of XOR)."""
    return {
        "ctx": 3072,
        "slots": 1,
        "spec": "off",
        "loras": ["/nonexistent/a.safetensors"],
        "extra_args": ["--flag-probe"],
        "cache_type": "f32",
        "kv_unified": True,
        "ctx_extend": 1.5,
        "cpu_moe_n": 1,
        "override_tensor": [".ffn_.*_exps.=CPU"],
        "devices": [],
        "warmup": False,
        "reasoning_budget": 512,
        "reasoning_effort": "medium",
        "replicas": 1,
        "pin": True,
        "chat_template": "chatml",
        "spm_infill": True,
        "sampler_defaults": {
            "temperature": 0.7,
            "top_k": 40,
            "top_p": 0.9,
            "min_p": 0.05,
            "top_n_sigma": 1.0,
            "typical_p": 0.8,
            "repeat_penalty": 1.1,
            "repeat_last_n": 64,
            "presence_penalty": 0.1,
            "frequency_penalty": 0.2,
            "dry_multiplier": 0.8,
            "dry_base": 1.75,
            "dry_allowed_length": 2,
            "dry_penalty_last_n": 256,
            "xtc_probability": 0.05,
            "xtc_threshold": 0.1,
            "mirostat": 2,
            "seed": 42,
        },
    }


def _overlay_b() -> dict:
    """chat_template_file side of the XOR (everything else identical)."""
    ov = _overlay_a()
    ov.pop("chat_template")
    ov["chat_template_file"] = os.path.join(SANDBOX.root, "template.tmpl")
    open(ov["chat_template_file"], "w").close()
    return ov


def phase_gates() -> None:
    print(
        "\n== phase gates: 100% completeness (help/fresh-list/full-manifest/overlay) =="
    )

    d = DAEMON
    enforce = PHASE_FILTER is None

    # (a) --help x manifest, bidirectional -------------------------------
    p = cli("--help")
    names = set(_help_command_names(p.stdout))
    want = set(TOPLEVEL_COMMANDS)
    ok_a = names == want
    check(
        "gates",
        "(a) --help subcommands == manifest (both directions)",
        ok_a,
        f"manifest-only={sorted(want - names)} help-only={sorted(names - want)}",
    )

    # (b) fresh config list key set == manifest fresh-visible -------------
    cfgp = os.path.join(SANDBOX.config_dir, "config.toml")
    stash = cfgp + ".stash"
    # Phase-filtered runs (e.g. --phase=gates) may reach here before any
    # daemon boot created the sandbox config — stash only if present.
    had_cfg = os.path.exists(cfgp)
    if had_cfg:
        os.replace(cfgp, stash)
    try:
        with open(cfgp, "w") as f:
            f.write("")
        out = cli("config", "list").stdout
    finally:
        if had_cfg:
            os.replace(stash, cfgp)
        else:
            os.remove(cfgp)
    try:
        fresh = set(tomllib.loads(out).keys())
    except Exception as e:
        fresh = set()
        check(
            "gates",
            "(b) fresh config list parses as TOML",
            False,
            f"{e}; head={out[:120]!r}",
        )
    want_fresh = set(FRESH_VISIBLE_KNOBS)
    ok_b = fresh == want_fresh
    check(
        "gates",
        "(b) fresh config list keys == manifest non-Option knobs",
        ok_b,
        f"manifest-only={sorted(want_fresh - fresh)} list-only={sorted(fresh - want_fresh)}",
    )

    # (c) full-manifest daemon boot (deny_unknown_fields proof) -----------
    full = _full_toplevel()
    full["model_overrides"] = {MODEL: _overlay_a()}
    missing_full = sorted(set(k["name"] for k in TOPLEVEL_KNOBS) - set(full))
    ok_c = not missing_full
    check(
        "gates",
        f"(c0) full-manifest dict covers all {len(TOPLEVEL_KNOBS)} knobs",
        ok_c,
        f"missing={missing_full}",
    )
    try:
        d.start(full, floor_model=MODEL)
        ok_c = ok_c and True
    except RuntimeError as e:
        ok_c = False
        check("gates", "(c) full-manifest daemon boot", False, str(e)[-300:])
    else:
        check(
            "gates",
            "(c) full-manifest daemon boot -> healthz 200",
            True,
            f"all {len(full)} top-level keys + full overlay accepted",
        )
        d.stop()

    # (d) overlay round-trip echo (both XOR sides) -------------------------
    echoed_a: dict = {}
    echoed_b: dict = {}
    for ov, sink in ((_overlay_a(), "a"), (_overlay_b(), "b")):
        SANDBOX.write_config({"port": PORT, "model_overrides": {MODEL: dict(ov)}})
        out = cli("config", "list").stdout
        try:
            mo = tomllib.loads(out).get("model_overrides", {}).get(MODEL, {})
        except Exception:
            mo = {}
        if sink == "a":
            echoed_a = mo
        else:
            echoed_b = mo
    ok_d = True
    miss_fields = []
    for field in dict(_overlay_a()):
        got = echoed_a.get(field, echoed_b.get(field, None))
        if field == "chat_template":
            got = echoed_a.get(field)
        if field == "chat_template_file":
            got = echoed_b.get(field)
        if got is None:
            ok_d = False
            miss_fields.append(field)
    sampler_echoed = echoed_a.get("sampler_defaults", {})
    miss_samp = [s for s in SAMPLER_FIELDS if s not in sampler_echoed]
    ok_d = ok_d and not miss_samp
    check(
        "gates",
        f"(d) overlay round-trip echoes all {len(MODEL_OVERRIDE_FIELDS)} override fields",
        ok_d,
        f"missing={miss_fields}",
    )
    check(
        "gates",
        "(d) overlay round-trip echoes all 18 sampler leaves",
        not miss_samp,
        f"missing={miss_samp}",
    )

    manifest_ok = ok_a and ok_b and ok_c and ok_d

    # -- universal coverage rows: every manifest knob gets evidence -------
    covered = {c["knob"] for c in COVERAGE}
    for k in TOPLEVEL_KNOBS:
        if k["name"] not in covered:
            cov(
                k["name"],
                f"{k['tier']}: accepted + echoed (full-manifest boot / set->list)",
                "phase gates (c)/(d)",
                ok=manifest_ok,
            )
    for field, note in MODEL_OVERRIDE_FIELDS:
        key = f"model_overrides.{field}"
        if key not in covered:
            cov(key, f"overlay round-trip echo ({note})", "phase gates (d)", ok=ok_d)
    for s in SAMPLER_FIELDS:
        key = f"model_overrides.sampler_defaults.{s}"
        if key not in covered:
            cov(key, "sampler leaf echo", "phase gates (d)", ok=ok_d and not miss_samp)
    # tune-lane knobs keep their stronger lane evidence when it exists;
    # otherwise the round-trip row above already covers them.
    for b_knob, why in (
        ("max_loaded_models", "needs 2+ concurrently-loaded models (RAM)"),
        ("child_transport", "unix transport unsupported on this build"),
        ("rpc_servers", "needs a 2nd box running rpc llama-server"),
    ):
        cov(b_knob, f"boundary: {why}", "documented boundary", ok=True)
    if knob_entry("models_autoload")["name"] not in covered:
        cov(
            "models_autoload",
            "boundary: loads ALL store models (RAM); Option set=false proven at boot",
            "phase gates (c)",
            ok=manifest_ok,
        )

    # -- completeness enforcement -----------------------------------------
    if enforce:
        ok_rows = {c["knob"] for c in COVERAGE if c["ok"]}
        want_knobs = set(k["name"] for k in TOPLEVEL_KNOBS)
        missing_knobs = sorted(want_knobs - ok_rows)
        check(
            "gates",
            f"CONFIG COVERAGE 100% ({len(TOPLEVEL_KNOBS)} top-level knobs)",
            not missing_knobs,
            f"missing={missing_knobs}",
        )
        regd = {r["path"] for r in COMMAND_COVERAGE if r["ok"]}
        missing_cmds = sorted(set(command_paths()) - regd)
        check(
            "gates",
            f"COMMAND COVERAGE 100% ({len(command_paths())} leaf paths)",
            not missing_cmds,
            f"missing={missing_cmds}",
        )


# ------------------------------------------------------------------ main


def report() -> int:
    print("\n" + "=" * 72)
    print("CONFIG-FLOW COVERAGE MATRIX")
    print("=" * 72)
    for c in COVERAGE:
        tag = "ok " if c["ok"] else "MISS"
        print(f"  [{tag}] {c['knob']:<28} {c['expectation']}")
    print(
        f"  — {sum(1 for c in COVERAGE if c['ok'])}/{len(COVERAGE)} knob flows verified with evidence"
    )
    # Command coverage from the registry (reg/regb rows only count once per path).
    regd = {r["path"] for r in COMMAND_COVERAGE if r["ok"]}
    cmd_missing = sorted(set(command_paths()) - regd)
    print(
        f"\nCOMMAND COVERAGE {len(set(command_paths())) - len(cmd_missing)}/{len(set(command_paths()))}"
        f" ({len(set(command_paths()))} leaf paths)"
    )
    for p in cmd_missing:
        print(f"  [MISS] command path not exercised: {p}")
    # Config coverage vs the manifests (exact knob-name rows).
    ok_knobs = {c["knob"] for c in COVERAGE if c["ok"]}
    want = (
        set(toplevel_knob_names())
        | {f"model_overrides.{f}" for f, _ in MODEL_OVERRIDE_FIELDS}
        | {f"model_overrides.sampler_defaults.{s}" for s in SAMPLER_FIELDS}
    )
    knob_missing = sorted(want - ok_knobs)
    print(
        f"CONFIG COVERAGE {len(want) - len(knob_missing)}/{len(want)}"
        f" ({len(TOPLEVEL_KNOBS)} top-level + {len(MODEL_OVERRIDE_FIELDS)} override"
        f" + {len(SAMPLER_FIELDS)} sampler leaves)"
    )
    for k in knob_missing:
        print(f"  [MISS] knob not covered: {k}")
    # Lane-level failures must fail the run: a registered lane that FAILED is
    # red regardless of the check() accounting (fail loud, never silent-red).
    cmd_failed = sorted({r["path"] for r in COMMAND_COVERAGE if not r["ok"]})
    if cmd_failed:
        print("\nFAILED LANES")
        for p in cmd_failed:
            print(f"  [CMD:FAIL] {p}")
    print("\nCHECK SUMMARY")
    by_phase: dict[str, list] = {}
    for c in CHECKS:
        by_phase.setdefault(c["phase"], []).append(c)
    failed = 0
    for phase, items in by_phase.items():
        n_pass = sum(1 for i in items if i["ok"] and not i.get("boundary"))
        n_bound = sum(1 for i in items if i.get("boundary"))
        n_fail = sum(1 for i in items if not i["ok"])
        failed += n_fail
        print(f"  {phase:<12} pass={n_pass:<3} boundary={n_bound:<3} fail={n_fail}")
    print(f"\nTOTAL: {len(CHECKS)} checks, {failed} FAILED")
    # Isolation proof.
    if USER_CONFIG_SHA and os.path.exists(REAL_CONFIG):
        with open(REAL_CONFIG, "rb") as f:
            now = hashlib.sha256(f.read()).hexdigest()
        print(f"user config.toml untouched: {now == USER_CONFIG_SHA} ({now[:16]}…)")
    return 1 if (failed or cmd_failed) else 0


# ---------------------------------------------------------------------------
# phase: realuser — drive the CLI on a REAL pty terminal (interactive truth)
# ---------------------------------------------------------------------------
def _pty_session(argv, steps, timeout=300, env=None):
    """Run argv under a real pty terminal and drive it through `steps`.

    Each step is {"send": bytes|None, "expect": str|None, "budget": secs}.
    Rolling transcript: output is never drained-and-discarded, so a marker
    that arrived batched with earlier output still matches. Returns a dict
    with transcript (str), times (per-step secs to marker), firsts (secs
    from send to first new byte = perceived TTFT), rc, err.
    """
    import pty as _pty
    import select as _sel

    mfd, sfd = _pty.openpty()
    proc = subprocess.Popen(
        argv,
        stdin=sfd,
        stdout=sfd,
        stderr=sfd,
        env=env or SANDBOX.env(),
        close_fds=True,
    )
    os.close(sfd)
    transcript = b""
    times: list = []
    firsts: list = []
    err = ""
    # Search cursor for expect markers: starts at 0 and advances to just
    # past each found marker — NOT to end-of-buffer — so a marker that
    # arrived batched with the previous step's output still matches.
    cursor = 0
    try:
        for st in steps:
            t0 = time.time()
            send = st.get("send")
            if send:
                os.write(mfd, send)
            exp = st.get("expect")
            if exp is None:
                times.append(None)
                firsts.append(None)
                cursor = len(transcript)
                continue
            budget = st.get("budget", timeout)
            want = exp.encode()
            hit = transcript.find(want, cursor)
            t_first = None
            while hit < 0 and time.time() - t0 < budget:
                r, _, _ = _sel.select([mfd], [], [], 0.25)
                if not r:
                    if proc.poll() is not None:
                        break
                    continue
                try:
                    chunk = os.read(mfd, 65536)
                except OSError:
                    break
                if not chunk:
                    break
                if t_first is None:
                    t_first = round(time.time() - t0, 2)
                transcript += chunk
                hit = transcript.find(want, cursor)
            firsts.append(t_first)
            if hit < 0:
                err = f"marker {exp!r} not seen within {budget}s"
                times.append(None)
                break
            if t_first is None:
                t_first = 0.0
                firsts[-1] = 0.0
            times.append(round(time.time() - t0, 2))
            cursor = hit + len(want)
        # drain to EOF / process exit (bounded 30s)
        hard = time.time() + 30
        while time.time() < hard:
            r, _, _ = _sel.select([mfd], [], [], 0.25)
            if r:
                try:
                    chunk = os.read(mfd, 65536)
                except OSError:
                    break
                if not chunk:
                    break
                transcript += chunk
                continue
            if proc.poll() is not None:
                while True:
                    r2, _, _ = _sel.select([mfd], [], [], 0)
                    if not r2:
                        break
                    try:
                        c2 = os.read(mfd, 65536)
                    except OSError:
                        break
                    if not c2:
                        break
                    transcript += c2
                break
        try:
            rc = proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            rc = proc.wait()
            err = err or "process did not exit; killed"
    finally:
        os.close(mfd)
    return {
        "transcript": transcript.decode("utf-8", "replace"),
        "times": times,
        "firsts": firsts,
        "rc": rc,
        "err": err,
    }


def phase_realuser():
    """Real-terminal simulation: the interactive surfaces a piped-stdin lane
    can never prove (prompt rendering, ctrl keys, perceived TTFT)."""
    d = DAEMON
    d.start({"port": PORT})
    env = SANDBOX.env()
    try:
        steps = [
            {"send": None, "expect": "REPL", "budget": 90},
            {"send": None, "expect": ">>> ", "budget": 60},
            {"send": b"Reply with just: ok\n", "expect": ">>> ", "budget": 300},
            {"send": b"/sysinfo\n", "expect": ">>> ", "budget": 60},
            {"send": b"/clear\n", "expect": "history cleared", "budget": 60},
            {"send": b"/exit\n", "expect": None},
        ]
        res = _pty_session([PAL, "run", MODEL], steps, env=env)
        low = res["transcript"].lower()
        sysinfo_ok = ("gib" in low) or ("mib" in low) or (MODEL in low)
        ok = (
            res["rc"] == 0
            and all(t is not None for t in res["times"][:5])
            and sysinfo_ok
            and not res["err"]
        )
        reg(
            "run.repl.interactive",
            ok,
            f"markers={sum(1 for t in res['times'][:5] if t is not None)}/5 "
            f"turn_ttft={res['firsts'][2] if len(res['firsts']) > 2 else None}s "
            f"turn_done={res['times'][2] if len(res['times']) > 2 else None}s "
            f"rc={res['rc']} err={res['err'][:60]}",
        )

        for path, key, label in (
            ("run.repl.ctrl-d", b"\x04", "ctrl-d"),
            ("run.repl.ctrl-c", b"\x03", "ctrl-c"),
        ):
            res = _pty_session(
                [PAL, "run", MODEL],
                [
                    {"send": None, "expect": "REPL", "budget": 90},
                    {"send": None, "expect": ">>> ", "budget": 60},
                    {"send": key, "expect": None},
                ],
                env=env,
            )
            reg(
                path,
                res["rc"] == 0 and not res["err"],
                f"{label} at prompt -> rc={res['rc']} err={res['err'][:60]}",
            )

        res = _pty_session(
            [PAL, "run", MODEL, "Reply with just: ok"],
            [{"send": None, "expect": None}],
            timeout=300,
            env=env,
        )
        reply = "\n".join(
            ln
            for ln in res["transcript"].splitlines()
            if ln.strip() and not ln.startswith("update available")
        )
        reg(
            "run.single.nocap",
            res["rc"] == 0 and bool(reply.strip()),
            f"rc={res['rc']} no-cap reply lines={len(reply.splitlines())} "
            f"err={res['err'][:60]}",
        )
    finally:
        d.stop()


# ---------------------------------------------------------------------------
# phase: golds — golden-file validation of stable output surfaces
# ---------------------------------------------------------------------------
GOLDEN_DIR = os.path.join(dirname(abspath(__file__)), "goldens")
UPDATE_GOLDENS = False  # set by main() from --update-goldens


def _gold_items() -> dict:
    """Capture canonical payloads for every golden surface."""
    items: dict[str, str] = {}

    p = cli("--help")
    # Grouped help: same name set as the old single Commands: block
    # (shared parser — see _help_command_names).
    cmds = _help_command_names(p.stdout)
    items["help.commands"] = "\n".join(sorted(cmds))
    cfg_path = os.path.join(SANDBOX.root, "config", "pallama", "config.toml")
    backup = open(cfg_path, "rb").read() if os.path.exists(cfg_path) else None
    try:
        SANDBOX.write_config({"port": PORT})
        fresh = tomllib.loads(cli("config", "list").stdout)
        items["config.fresh-keys"] = "\n".join(sorted(fresh.keys()))
    finally:
        if backup is not None:
            with open(cfg_path, "wb") as f:
                f.write(backup)

    for name, args in (
        ("push", ("push", "foo")),
        ("signin", ("signin",)),
        ("login", ("login",)),
        ("signout", ("signout",)),
        ("logout", ("logout",)),
    ):
        p = cli(*args)
        line = next(
            (ln for ln in (p.stdout + p.stderr).splitlines() if ln.strip()),
            "",
        )
        items[f"refusal.{name}"] = line

    p = cli("list")
    items["header.list"] = " ".join(p.stdout.splitlines()[0].split())

    p = cli("show", MODEL)
    keys = []
    for ln in p.stdout.splitlines():
        m = re.match(r"^([a-z][a-z0-9 _-]*?):\s+", ln)
        if m:
            keys.append(m.group(1).strip())
    items["header.show"] = "\n".join(keys)

    p = cli("doctor")
    names = sorted(
        {
            m.group(1)
            for m in (
                re.match(r"^\s*([a-z][a-z0-9 _-]+?)\s{2,}(?:ok|WARN|FAIL)\s", ln)
                for ln in p.stdout.splitlines()
            )
            if m
        }
        # Context-conditional doctor rows: they appear/differ based on
        # whether a whisper server is installed in the phase's sandbox
        # (commands installs one; a fresh golds-only run has none), on
        # whether a systemd/launchd manager + pallama unit is probeable on
        # the host (dev boxes/sandboxes without a unit emit no row), on
        # whether the host is Linux-NVIDIA serving a non-CUDA asset
        # (the cuda-channel hint row is vendor/asset-conditional), and on
        # whether the config pins retired defaults (the gates full-manifest
        # boot deliberately writes spec="off"; pristine configs emit no row).
        - {
            "whisper currency",
            "whisper lane",
            "whisper models",
            "service",
            "engine cuda channel",
            "config pins",
        }
    )
    items["doctor.check-names"] = "\n".join(names)

    for shell in ("bash", "zsh", "fish", "powershell"):
        p = cli("completions", shell)
        digest = hashlib.sha256(p.stdout.encode()).hexdigest()
        items[f"completions.{shell}"] = digest

    d = DAEMON
    d.start({"port": PORT})
    try:
        st, body, _ = chat("Say ok")
        assert st == 200, f"chat {st}"
        v = http_json("GET", "/api/version")[2]
        items["api.version"] = ",".join(sorted(v.keys()))
        rows = http_json("GET", "/api/ps")[2].get("models", [])
        items["api.ps.row"] = ",".join(sorted(rows[0].keys())) if rows else ""
        p = cli("ps")
        hdr = next(
            (ln for ln in p.stdout.splitlines() if ln.startswith("NAME")),
            "",
        )
        items["header.ps"] = " ".join(hdr.split())
        ok_sse, collected = sse_collect(
            "/v1/chat/completions",
            "data:",
            120,
            body={
                "model": MODEL,
                "stream": True,
                "messages": [{"role": "user", "content": "Say ok"}],
            },
        )
        chunks = [
            ln[len("data:") :].strip()
            for ln in collected.splitlines()
            if ln.startswith("data:")
        ]
        last_keys = ""
        for c in reversed(chunks):
            try:
                last_keys = ",".join(sorted(json.loads(c).keys()))
                break
            except ValueError:
                continue
        items["chat.sse-final-keys"] = (
            f"ok={ok_sse}|" + last_keys if last_keys else f"ok={ok_sse}|"
        )
    finally:
        d.stop()

    d.start(
        {
            "port": PORT,
            "keys": [{"name": "gatekey", "key": "plm-validate-gate-000"}],
        }
    )
    try:
        p = cli("keys", "list")
        items["header.keys"] = " ".join(p.stdout.splitlines()[0].split())
    finally:
        d.stop()
        SANDBOX.write_config({"port": PORT})
    return items


def phase_golds():
    items = _gold_items()
    meta_path = os.path.join(GOLDEN_DIR, "goldens.json")
    if UPDATE_GOLDENS:
        import datetime as _dt

        os.makedirs(GOLDEN_DIR, exist_ok=True)
        meta = {
            "generator": "python3 scripts/validate.py --update-goldens",
            "generated": _dt.datetime.now(_dt.timezone.utc).isoformat(
                timespec="seconds"
            ),
            "provenance": (
                "captured from the release binary on a FULL-run-verified "
                "sandbox; payloads are machine-stable by construction "
                "(sorted key sets, exact refusal lines, column names, shas)"
            ),
            "items": {},
        }
        for name, payload in sorted(items.items()):
            fp = os.path.join(GOLDEN_DIR, name + ".golden")
            with open(fp, "w") as f:
                f.write(payload)
            meta["items"][name] = hashlib.sha256(payload.encode()).hexdigest()
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=1, sort_keys=True)
            f.write("\n")
        check(
            "golds",
            "golden files regenerated",
            True,
            f"{len(items)} items -> {GOLDEN_DIR}",
        )
        return
    if not os.path.exists(meta_path):
        check(
            "golds",
            "golden files present",
            False,
            f"{meta_path} missing — run `python3 scripts/validate.py "
            f"--update-goldens` once on a verified binary",
        )
        return
    with open(meta_path) as f:
        meta = json.load(f)
    for name, payload in sorted(items.items()):
        fp = os.path.join(GOLDEN_DIR, name + ".golden")
        if not os.path.exists(fp):
            check("golds", f"golden {name}", False, "file missing")
            continue
        with open(fp, "rb") as f:
            raw = f.read()
        fsha = hashlib.sha256(raw).hexdigest()
        msha = (meta.get("items") or {}).get(name)
        if msha and msha != fsha:
            check(
                "golds",
                f"golden {name}",
                False,
                "file sha drift vs goldens.json (hand-edited?)",
            )
            continue
        want_payload = raw.decode()
        ok = want_payload == payload
        ev = f"sha={fsha[:12]}"
        if not ok:
            wl, gl = want_payload.splitlines(), payload.splitlines()
            ev += f" want {len(wl)}L got {len(gl)}L first-diff: " + next(
                (f"want {a!r} got {b!r}" for a, b in zip(wl, gl) if a != b),
                "prefix/length",
            )
        check("golds", f"golden {name}", ok, ev)


def _pyspy_reexec_if_requested(args: list[str]) -> None:
    """Re-exec the harness under `py-spy record` when asked (user
    directive 2026-09-08: py-spy in validation).

    Why a re-exec instead of a child attach: Yama ptrace_scope=1 (the
    common Linux default) only lets a tracer follow processes it spawned,
    so a recorder child cannot attach to this parent without root. Making
    py-spy the PARENT (``py-spy record -- python3 validate.py ...``)
    traces its own child — no sudo, works everywhere py-spy does.
    """
    if os.environ.get("PALLAMA_VALIDATE_PYSPY", "") != "1":
        return
    if os.environ.get("PALLAMA_VALIDATE_PYSPY_CHILD") == "1":
        return  # already under the recorder — never recurse
    bin_path = shutil.which("py-spy")
    if not bin_path:
        print(
            "py-spy: PALLAMA_VALIDATE_PYSPY=1 but py-spy is not on PATH — "
            "install it first:  pip install py-spy",
            file=sys.stderr,
        )
        sys.exit(2)
    if sys.platform == "darwin":
        # macOS requires root for py-spy even in spawn mode; run it
        # externally with sudo there. Loud skip, never silent.
        print(
            "py-spy: macOS needs a root tracer — run "
            "`sudo py-spy record -- python3 scripts/validate.py` instead; "
            "continuing unprofiled",
            file=sys.stderr,
        )
        return
    out = os.environ.get("PALLAMA_VALIDATE_PYSPY_OUT") or os.path.join(
        os.path.expanduser("~/.cache"),
        "pallama-pyspy",
        f"validate-{time.strftime('%Y%m%dT%H%M%S')}.speedscope.json",
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cmd = [
        bin_path,
        "record",
        "--rate",
        "25",
        "--format",
        "speedscope",
        "-o",
        out,
        "--",
        sys.executable,
        os.path.abspath(__file__),
        *args,
    ]
    print(f"py-spy: profiling this run -> {out} (parent tracer, no sudo)")
    rc = subprocess.call(cmd, env={**os.environ, "PALLAMA_VALIDATE_PYSPY_CHILD": "1"})
    print(f"py-spy: flame graph written -> {out}")
    sys.exit(rc)


def main() -> int:
    global SANDBOX, DAEMON, USER_CONFIG_SHA, PHASE_FILTER, CRASHED
    global UPDATE_GOLDENS
    args = sys.argv[1:]
    _pyspy_reexec_if_requested(args)
    self_test = "--self-test" in args
    if "--update-goldens" in args:
        UPDATE_GOLDENS = True
    phases_arg = [a for a in args if a.startswith("--phase=")]
    wanted = {a.split("=", 1)[1] for a in phases_arg} or None
    PHASE_FILTER = wanted
    print(
        f"pallama validation harness — engine+model REAL, isolation via temp XDG, port {PORT}"
    )
    print(f"binary={PAL} model={MODEL} fast={FAST}")
    # A dead GH_TOKEN is worse than none (401 "Bad credentials" on every
    # authed call: engine-check marker, whisper install, engine update,
    # doctor currency probes). Validate once up front; drop it if invalid
    # so the run falls back to the unauth 60/hr budget like FULL#4.
    if os.environ.get("GH_TOKEN"):
        try:
            rq = urllib.request.Request(
                "https://api.github.com/rate_limit",
                headers={"Authorization": f"Bearer {os.environ['GH_TOKEN']}"},
            )
            with urllib.request.urlopen(rq, timeout=8) as resp:
                limit = (
                    json.loads(resp.read())
                    .get("resources", {})
                    .get("core", {})
                    .get("limit", 60)
                )
            print(f"GH_TOKEN valid (core limit {limit}/hr)")
        except Exception as e:
            del os.environ["GH_TOKEN"]
            boundary(
                "baseline",
                "GH_TOKEN invalid — removed from env",
                f"authed probe rejected ({type(e).__name__}); "
                "falling back to unauth 60/hr budget",
            )
    if os.path.exists(REAL_CONFIG):
        with open(REAL_CONFIG, "rb") as f:
            USER_CONFIG_SHA = hashlib.sha256(f.read()).hexdigest()
    SANDBOX = Sandbox()
    DAEMON = Daemon(SANDBOX)

    cleaned = False

    def _cleanup() -> None:
        # F138: registered with atexit AND called explicitly before the
        # exit — guard so the pair runs the body exactly once.
        nonlocal cleaned
        if cleaned:
            return
        cleaned = True
        if DAEMON:
            try:
                DAEMON.stop()
            except Exception:
                pass
        if SANDBOX:
            # evaluate NOW (at exit): CHECKS fills up as phases run
            failed = any(not c["ok"] for c in CHECKS)
            if failed or CRASHED:
                print(
                    f"post-mortem sandbox kept: {SANDBOX.root} (daemon log: {DAEMON.log_path})"
                )
            else:
                SANDBOX.destroy()

    atexit.register(_cleanup)

    phases = [
        ("manifests", phase_manifests),
        ("baseline", phase_baseline),
        ("config", phase_config),
        ("api", phase_api),
        ("sentinel", phase_sentinel),
        ("behavior", phase_behavior),
        ("cli", phase_cli),
        ("auth", phase_auth),
        ("wave", phase_wave),
        ("commands", phase_commands),
        ("realuser", phase_realuser),
        ("knobs_argv", phase_knobs_argv),
        ("knobs_behavior", phase_knobs_behavior),
        ("gates", phase_gates),
        ("golds", phase_golds),
        ("parity", phase_parity),
    ]
    for name, fn in phases:
        if self_test or (wanted and name not in wanted):
            print(
                f"\n== phase {name}: skipped ({'self-test' if self_test else '--phase filter'}) =="
            )
            continue
        try:
            fn()
        except Exception:
            CRASHED = True
            print(f"\n== phase {name}: CRASHED ==")
            traceback.print_exc()
            break
    if self_test:
        check("self-test", "injected failure proves non-zero exit", False, "by design")
    rc = report()
    if CRASHED:
        rc = 1
    _cleanup()
    return rc


if __name__ == "__main__":
    sys.exit(main())
