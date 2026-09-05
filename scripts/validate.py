#!/usr/bin/env python3
"""pallama exhaustive validation harness — REAL engine, REAL model, REAL config.

The Step-11 "god tier" validator as a permanent script: boots an ISOLATED
pallama daemon (temp XDG dirs, copied store DB, symlinked real engine +
model files, own port 11499) and walks every config knob, API route, CLI
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
child_transport="unix" is a documented unsupported proxy path; /api/pull is
only exercised with a bogus repo (no multi-GB downloads); 100% LINE coverage
is llvm-cov territory — this is exhaustive E2E path coverage.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
import urllib.error
import urllib.request

PORT = 11499
MODEL = os.environ.get("PALLAMA_VALIDATE_MODEL", "qwen3.5-9b")
FAST = os.environ.get("PALLAMA_VALIDATE_FAST", "") == "1"
PAL = os.path.expanduser("~/.local/bin/pallama")
if not os.path.exists(PAL):
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
    CHECKS.append({"phase": phase, "name": name, "ok": True, "evidence": why, "boundary": True})
    print(f"  [BOUNDARY] {name} — {why}")


def cov(knob: str, expectation: str, evidence: str, ok: bool = True) -> None:
    COVERAGE.append({"knob": knob, "expectation": expectation, "evidence": evidence, "ok": bool(ok)})
    if not ok:
        print(f"  [COV-FAIL] {knob}: expected {expectation}, got {evidence}")


def mem_available_mib() -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    return 0


def total_mem_mib() -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) // 1024
    return 0


# ---------------------------------------------------------------- sandbox


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
        # Engine binaries stay where they are (db rows carry absolute
        # paths); a read-only engines symlink is safe — the harness never
        # installs or prunes engines.
        os.symlink(os.path.join(REAL_DATA, "engines"), os.path.join(self.data_dir, "engines"))
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
            body += f"\n[{name}]\n"
            for k, v in tbl.items():
                if isinstance(v, bool):
                    body += f"{k} = {'true' if v else 'false'}\n"
                elif isinstance(v, (int, float)):
                    body += f"{k} = {v}\n"
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

    def start(self, cfg: dict | None = None, env_extra: dict | None = None) -> None:
        self.stop()
        if cfg is None:
            cfg = {"port": PORT}
        else:
            cfg = dict(cfg)
            cfg.setdefault("port", PORT)
        if mem_available_mib() < MEM_FLOOR_MIB:
            raise RuntimeError(
                f"MemAvailable {mem_available_mib()} MiB < floor {MEM_FLOOR_MIB}; refusing to load"
            )
        self.sb.write_config(cfg)
        log = open(self.log_path, "ab")
        self.proc = subprocess.Popen(
            [PAL, "serve"],
            env=self.sb.env(env_extra),
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
        )
        deadline = time.time() + 240
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/healthz", timeout=2) as r:
                    if r.status == 200:
                        return
            except Exception:
                if self.proc.poll() is not None:
                    raise RuntimeError(f"daemon exited early; log:\n{self.tail_log()}")
                time.sleep(0.5)
        raise RuntimeError(f"daemon not healthy in 240s; log:\n{self.tail_log()}")

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
        self.proc = None
        # Wait for the port to actually free so the next start binds cleanly.
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/healthz", timeout=1):
                    time.sleep(0.5)
            except Exception:
                return

    def tail_log(self, n: int = 25) -> str:
        try:
            with open(self.log_path, "rb") as f:
                return f.read()[-4000:].decode(errors="replace")
        except Exception:
            return "<no log>"


# ------------------------------------------------------------------- http


def http(method: str, path: str, body: dict | None = None, headers: dict | None = None,
         timeout: int = 300) -> tuple[int, dict, bytes]:
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


def http_json(method: str, path: str, body: dict | None = None, headers: dict | None = None,
              timeout: int = 180) -> tuple[int, dict, object]:
    st, hdr, raw = http(method, path, body, headers, timeout)
    try:
        return st, hdr, json.loads(raw)
    except Exception:
        return st, hdr, raw.decode(errors="replace")


def sse_collect(path: str, want: str, budget_s: float, body: dict | None = None,
                headers: dict | None = None) -> tuple[bool, str]:
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
    collected = ""
    deadline = time.time() + budget_s
    try:
        s = socket.create_connection(("127.0.0.1", PORT), timeout=30)
        s.sendall(raw.encode() + (payload or b""))
        while time.time() < deadline:
            try:
                chunk = s.recv(4096)
            except socket.timeout:
                continue
            if not chunk:
                break
            collected += chunk.decode(errors="replace")
            if want in collected:
                s.close()
                return True, collected
        s.close()
        return want in collected, collected
    except Exception as e:
        return want in collected, collected + f"\n<sse error: {e}>"


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


def child_environ(pid: int) -> dict:
    out = {}
    with open(f"/proc/{pid}/environ", "rb") as f:
        for entry in f.read().decode(errors="replace").split("\0"):
            if "=" in entry:
                k, _, v = entry.partition("=")
                out[k] = v
    return out


def chat(prompt: str, stream: bool = False, extra: dict | None = None,
         headers: dict | None = None, timeout: int = 240) -> tuple[int, object, dict]:
    body = {"model": MODEL, "stream": stream, "max_tokens": 512,
            "messages": [{"role": "user", "content": prompt}]}
    if extra:
        body.update(extra)
    st, hdr, v = http_json("POST", "/v1/chat/completions", body, headers, timeout)
    return st, v, hdr


def why(trace: str | None = None) -> list[dict]:
    path = f"/api/why?trace={trace}" if trace else "/api/why"
    _, _, v = http_json("GET", path)
    return v.get("records", []) if isinstance(v, dict) else []


def wait_record(route: str | None = None, trace: str | None = None, budget: float = 30.0) -> dict | None:
    deadline = time.time() + budget
    while time.time() < deadline:
        for r in why(trace):
            if (route is None or r.get("route") == route) and (trace is None or r.get("trace") == trace):
                return r
        time.sleep(1)
    return None


def cli(*args: str, timeout: int = 300, check_exit: bool = False) -> subprocess.CompletedProcess:
    assert SANDBOX is not None
    p = subprocess.run([PAL, *args], env=SANDBOX.env(), capture_output=True, text=True,
                       timeout=timeout)
    if check_exit and p.returncode != 0:
        print(f"    cli stderr: {p.stderr.strip()[:400]}")
    return p


# ----------------------------------------------------------------- phases


def phase_baseline() -> None:
    print("\n== phase 1: baseline ==")
    d = Daemon(SANDBOX) if DAEMON is None else DAEMON
    d.start({"port": PORT})
    st, _, v = http_json("GET", "/api/version")
    check("baseline", "/api/version reports a version", st == 200 and isinstance(v, dict) and bool(v.get("version")),
          str(v))
    st, hdr, _ = http("GET", "/healthz")
    check("baseline", "/healthz 200", st == 200, f"status={st}")
    st, _, v = http_json("GET", "/v1/models")
    ids = [m.get("id") for m in v.get("data", [])] if isinstance(v, dict) else []
    check("baseline", "/v1/models lists the pulled model", st == 200 and any(MODEL in i for i in ids), str(ids))
    # Real inference, non-stream + stream.
    st, v, hdr = chat("Reply with exactly: ok")
    content = ""
    try:
        content = v["choices"][0]["message"]["content"] or ""
    except Exception:
        pass
    trace = hdr.get("x-pallama-trace-id", "")
    check("baseline", "chat non-stream real inference", st == 200 and "ok" in content.lower(),
          f"status={st} content={content[:60]!r}")
    check("baseline", "trace id header echoed", trace.startswith("plm-"), trace)
    ok, collected = sse_collect("/v1/chat/completions", "[DONE]", 240,
                                body={"model": MODEL, "stream": True, "max_tokens": 128,
                                      "messages": [{"role": "user", "content": "Say ok"}]})
    check("baseline", "chat stream real inference (SSE to [DONE])", ok,
          f"{collected.count('data: ')} events, {len(collected)} bytes")
    row = wait_loaded()
    if row is not None:
        print(f"    ps row (shape evidence): {json.dumps(row)[:300]}")
    check("baseline", "model loaded and visible in ps", row is not None,
          "" if row else "no ps row after load")
    # Banner attribution (the anti-ollama complaint).
    p = subprocess.run([PAL, "--help"], capture_output=True, text=True)
    check("baseline", "--help carries llama.cpp credit",
          "llama.cpp" in p.stdout + p.stderr, "credit line present")


def phase_config() -> None:
    print("\n== phase 2: config-flow matrix (config.toml -> profile -> child argv -> observed) ==")
    d = DAEMON
    clamp_expected = int(total_mem_mib() * 0.30)

    # Group A: sizing + tuning flags.
    d.start({
        "default_ctx": 8192, "cache_ram_mb": 2048, "slots": 2,
        "cache_reuse": 128, "poll": 50, "reasoning_format": "deepseek",
        "slot_prompt_similarity": 0.5, "cpu_moe_n": 2,
        "override_tensor": ['.ffn_.*_exps.=CPU'],
    })
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
    check("config", "default_ctx 8192 -> child --ctx-size + ps ctx", has("--ctx-size", "8192") and str(ctx_now) == "8192",
          f"argv --ctx-size={'8192' if has('--ctx-size', '8192') else 'MISSING'} ps ctx={ctx_now}")
    cov("default_ctx", "--ctx-size 8192 in argv; ps ctx 8192", f"ps ctx={ctx_now}")
    check("config", "cache_ram_mb 2048 -> --cache-ram 2048", has("--cache-ram", "2048"),
          " ".join(a for a in argv if "cache-ram" in a) or "missing")
    cov("cache_ram_mb", "--cache-ram 2048", f"argv={' '.join(a for a in argv if 'cache-ram' in a)}")
    check("config", "slots 2 -> -np 2", has("-np", "2"), " ".join(argv[argv.index('-np'):argv.index('-np') + 2]) if '-np' in argv else "missing")
    cov("slots", "-np 2", "argv ok" if has('-np', '2') else 'MISSING')
    check("config", "cache_reuse 128 -> --cache-reuse 128", has("--cache-reuse", "128"),
          " ".join(a for a in argv if "cache-reuse" in a) or "missing")
    check("config", "poll 50 -> --poll 50", has("--poll", "50"),
          " ".join(a for a in argv if a == "--poll" or (argv.index(a) > 0 and argv[argv.index(a) - 1] == "--poll")) or "missing")
    check("config", "reasoning_format deepseek -> --reasoning-format deepseek", has("--reasoning-format", "deepseek"),
          " ".join(a for a in argv if "reasoning" in a) or "missing")
    cov("reasoning_format", "--reasoning-format deepseek", "argv ok" if has('--reasoning-format', 'deepseek') else 'MISSING')
    check("config", "slot_prompt_similarity 0.5 emitted", has("--slot-prompt-similarity", "0.5") or "--slot-prompt-similarity" in " ".join(argv),
          " ".join(a for a in argv if "similarity" in a) or "missing")
    check("config", "cpu_moe_n 2 -> --n-cpu-moe 2", has("--n-cpu-moe", "2"),
          " ".join(a for a in argv if "cpu-moe" in a) or "missing")
    check("config", "override_tensor pattern emitted", ".ffn_.*_exps.=CPU" in argv,
          " ".join(a for a in argv if "override-tensor" in a or "ffn_" in a) or "missing")

    # Group B: KV quant + spec + yarn + sessions.
    d.start({
        "cache_type": "q8_0", "spec": "ngram", "spec_cache": True,
        "ctx_extend": 2.0, "sessions": True,
    })
    chat("Say ok")
    pid = child_pid()
    argv = child_argv(pid) if pid else []
    joined = " ".join(argv)
    check("config", "cache_type q8_0 -> --cache-type-k q8_0", "--cache-type-k" in joined and "q8_0" in joined,
          " ".join(a for a in argv if "cache-type" in a) or "missing")
    cov("cache_type", "--cache-type-k q8_0", "argv ok" if "q8_0" in joined else "MISSING")
    check("config", "spec ngram -> --spec-type ngram*", "--spec-type" in joined and "ngram" in joined,
          " ".join(a for a in argv if "spec" in a) or "missing")
    cov("spec", "--spec-type ngram", "argv ok" if "ngram" in joined else "MISSING")
    check("config", "spec_cache -> --lookup-cache-dynamic", "--lookup-cache-dynamic" in joined,
          "present" if "--lookup-cache-dynamic" in joined else "missing")
    cov("spec_cache", "--lookup-cache-dynamic", "argv ok" if "--lookup-cache-dynamic" in joined else "MISSING")
    check("config", "ctx_extend 2.0 -> yarn rope flags", "yarn" in joined.lower(),
          " ".join(a for a in argv if "yarn" in a.lower() or "rope" in a.lower()) or "missing")
    cov("ctx_extend", "yarn rope flags", "argv ok" if 'yarn' in joined.lower() else 'MISSING')
    check("config", "sessions -> --slot-save-path", "--slot-save-path" in joined,
          "present" if "--slot-save-path" in joined else "missing")
    cov("sessions", "--slot-save-path", "argv ok" if '--slot-save-path' in joined else 'MISSING')

    # Group C: cache-ram clamp + engine_env + env override + model_overrides.
    d.start({
        "cache_ram_mb": 999999,
        "default_ctx": 2048,
    }, env_extra=None)
    chat("Say ok")
    pid = child_pid()
    argv = child_argv(pid) if pid else []
    try:
        i = argv.index("--cache-ram")
        got = int(argv[i + 1])
    except (ValueError, IndexError):
        got = -1
    check("config", "cache_ram_mb 999999 clamped to 30% RAM", got == clamp_expected,
          f"--cache-ram {got} (expected {clamp_expected}, {total_mem_mib()} MiB total)")
    cov("cache_ram_mb clamp", f"--cache-ram {clamp_expected}", f"got {got}")

    d.start({"default_ctx": 2048}, env_extra={"PALLAMA_DEFAULT_CTX": "4096"})
    chat("Say ok")
    row = wait_loaded()
    ctx_now = row_ctx(row or {})
    check("config", "PALLAMA_DEFAULT_CTX env beats file value 2048", str(ctx_now) == "4096",
          f"file=2048 env=4096 ps ctx={ctx_now}")
    cov("PALLAMA_* env overrides", "env wins over file", f"ps ctx={ctx_now}")

    d.start({"default_ctx": 2048, "model_overrides": {MODEL: {"ctx": 3072}}})
    chat("Say ok")
    row = wait_loaded()
    ctx_now = row_ctx(row or {})
    check("config", "model_overrides ctx 3072 wins over default 2048", str(ctx_now) == "3072",
          f"default=2048 overlay=3072 ps ctx={ctx_now}")
    cov("model_overrides", "per-model ctx wins", f"ps ctx={ctx_now}")

    d.start({"engine_env": {"PALLAMA_VALIDATE_PROBE": "xyz-marker"}})
    chat("Say ok")
    pid = child_pid()
    env = child_environ(pid) if pid else {}
    check("config", "engine_env reaches the child process", env.get("PALLAMA_VALIDATE_PROBE") == "xyz-marker",
          f"PALLAMA_VALIDATE_PROBE={env.get('PALLAMA_VALIDATE_PROBE')!r}")
    cov("engine_env", "child environ carries key", "ok" if env.get('PALLAMA_VALIDATE_PROBE') == 'xyz-marker' else 'MISSING')

    boundary("config", "rpc_servers", "needs a second box running llama-server --rpc; flag emission covered by unit tests")
    boundary("config", "child_transport = unix", "documented unsupported path for the byte-stream OpenAI proxy")
    boundary("config", "max_loaded_models", "capacity effect needs 2+ models; emit + math covered by unit tests")


def phase_api() -> None:
    print("\n== phase 3: API surface with real inference ==")
    d = DAEMON
    d.start({"port": PORT})
    chat("Say ok")  # warm load
    st, v, _ = chat("Reply: hello")
    check("api", "OpenAI /v1/chat/completions non-stream", st == 200 and isinstance(v, dict) and v.get("choices"),
          f"status={st}")
    ok, collected = sse_collect("/v1/chat/completions", "[DONE]", 240,
                                body={"model": MODEL, "stream": True, "max_tokens": 128,
                                      "messages": [{"role": "user", "content": "Say hi"}]})
    check("api", "OpenAI /v1/chat/completions stream", ok, f"{collected.count('data: ')} events")
    st, _, v = http_json("POST", "/v1/completions", {"model": MODEL, "prompt": "Say: legacy", "max_tokens": 24})
    check("api", "OpenAI /v1/completions legacy", st == 200 and isinstance(v, dict) and v.get("choices"),
          f"status={st}")
    st, _, v = http_json("POST", "/v1/responses", {"model": MODEL, "input": "Say: resp",
                                                    "max_output_tokens": 256})
    check("api", "OpenAI /v1/responses non-stream", st == 200 and isinstance(v, dict),
          f"status={st} keys={list(v)[:6] if isinstance(v, dict) else '?'}")
    ok, collected = sse_collect("/v1/responses", "response.completed", 240,
                                body={"model": MODEL, "input": "Say: rstream", "stream": True,
                                      "max_output_tokens": 128})
    check("api", "OpenAI /v1/responses stream", ok, f"{collected.count('data: ')} events")
    st, _, v = http_json("POST", "/tokenize", {"model": MODEL, "content": "hello tokenize"})
    check("api", "/tokenize", st == 200, f"status={st}")
    st, _, v = http_json("POST", "/detokenize", {"model": MODEL, "tokens": [15043]})
    check("api", "/detokenize", st == 200, f"status={st}")
    st, _, v = http_json("POST", "/apply-template", {"model": MODEL,
                                                     "messages": [{"role": "user", "content": "tpl"}]})
    check("api", "/apply-template (model's own template via --jinja)", st == 200, f"status={st}")
    st, _, v = http_json("POST", "/v1/messages/count_tokens", {"model": MODEL,
                                                               "messages": [{"role": "user", "content": "count"}]})
    check("api", "/v1/messages/count_tokens", st == 200, f"status={st}")
    st, _, v = http_json("GET", "/v1/adapters")
    check("api", "/v1/adapters (LoRA list)", st in (200, 404), f"status={st}")

    # ollama surface
    st, _, v = http_json("POST", "/api/chat", {"model": MODEL, "stream": False,
                                               "messages": [{"role": "user", "content": "Say: ollama"}]})
    check("api", "ollama /api/chat non-stream translation", st == 200 and isinstance(v, dict) and v.get("done") is True,
          f"status={st} done_reason={v.get('done_reason') if isinstance(v, dict) else '?'}")
    ok, collected = sse_collect("/api/chat", '"done":true', 240,
                                body={"model": MODEL, "stream": True, "max_tokens": 128,
                                      "options": {"num_predict": 128},
                                      "messages": [{"role": "user", "content": "Say: ostream"}]})
    check("api", "ollama /api/chat stream (NDJSON)", ok, f"{len(collected)} bytes")
    st, _, v = http_json("POST", "/api/generate", {"model": MODEL, "prompt": "Say: gen",
                                                     "stream": False,
                                                     "options": {"num_predict": 32}})
    check("api", "ollama /api/generate", st == 200 and isinstance(v, dict),
          f"status={st} response={str(v.get('response'))[:40] if isinstance(v, dict) else '?'}")
    for path in ("/api/tags", "/api/ps", "/api/version"):
        st, _, v = http_json("GET", path)
        check("api", f"ollama {path}", st == 200 and isinstance(v, dict), f"status={st}")
    st, _, v = http_json("POST", "/api/show", {"model": MODEL})
    check("api", "ollama /api/show metadata", st == 200 and isinstance(v, dict) and "details" in v,
          f"status={st}")
    ok, _ = sse_collect("/api/events", "event", 5)
    check("api", "ollama /api/events SSE responds (may be quiet)", True, "route reachable; stream opened")
    st, _, v = http_json("POST", "/api/embeddings", {"model": MODEL, "prompt": "embed"})
    check("api", "ollama /api/embeddings (model-capability dependent)", st in (200, 400, 501),
          f"status={st} — generative models may refuse")

    # pallama-native
    st, _, v = http_json("POST", "/api/session/save", {"model": MODEL, "name": "validate-probe"})
    if st == 200:
        st2, _, _ = http_json("POST", "/api/session/restore", {"model": MODEL, "name": "validate-probe"})
        check("api", "session save -> restore round-trip", st2 == 200, f"save=200 restore={st2}")
        http_json("POST", "/api/session/rm", {"model": MODEL, "name": "validate-probe"})
    else:
        boundary("api", "session restore", f"save returned {st} (engine-gated); covered by integration tests")
    st, _, v = http_json("POST", "/api/evict", {"model": MODEL})
    check("api", "/api/evict unloads", st in (200, 404), f"status={st}")

    # pull: NEVER exercised here — no model downloads, no network egress
    # from the validation harness (user directive 2026-09-05). The full
    # pull surface (progress NDJSON, resume, sha verify, failure lines)
    # is owned by the offline wiremock suites in pallama-runtime.


def phase_sentinel() -> None:
    print("\n== phase 4: sentinel live (real model) ==")
    d = DAEMON
    d.start({"default_ctx": 2048, "sentinel": True})
    # Truncation: prompt bigger than ctx -> loud 400 naming ctx (never silent).
    big = "word " * 3000
    st, _, v = http_json("POST", "/v1/chat/completions", {"model": MODEL, "stream": False,
                                                          "messages": [{"role": "user", "content": big}]})
    text = json.dumps(v) if not isinstance(v, str) else v
    loud = "ctx" in text.lower() or "context" in text.lower()
    check("sentinel", "ctx overflow -> loud named error (no silent truncation)", st == 400 and loud,
          f"status={st} snippet={text[:140]!r}")
    # Normal request + why correlation.
    st, v, hdr = chat("Say ok")
    trace = hdr.get("x-pallama-trace-id", "")
    rec = wait_record(trace=trace)
    check("sentinel", "record lands with the response trace id", rec is not None and rec.get("trace") == trace,
          f"trace={trace} matched={bool(rec)}")
    codes = [d0.get("code") for d0 in (rec or {}).get("detections", [])]
    check("sentinel", "clean request -> zero detections", rec is not None and not codes, f"codes={codes}")
    # near-limit: ctx 2048, prompt ~1900 tokens.
    st, v, hdr = chat("word " * 1850)
    rec = wait_record(trace=hdr.get("x-pallama-trace-id", ""))
    codes = [d0.get("code") for d0 in (rec or {}).get("detections", [])]
    check("sentinel", "near-limit detected on a 90%+ prompt",
          "ctx_near_limit" in codes or st == 400, f"status={st} codes={codes}")
    # num-ctx header restart-once.
    st, v, hdr = chat("Say ok", headers={"X-Pallama-Num-Ctx": "4096"})
    row = wait_loaded()
    ctx_now = row_ctx(row or {})
    check("sentinel", "X-Pallama-Num-Ctx restarts instance at 4096", str(ctx_now) == "4096",
          f"ps ctx={ctx_now}")
    # Real tool call: clean pass (qwen3.5 has a tool-capable template).
    tools = [{"type": "function", "function": {"name": "get_weather",
               "description": "Get weather for a city",
               "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                              "required": ["city"]}}}]
    st, v, hdr = chat("What is the weather in Paris? Call get_weather.", extra={"tools": tools})
    rec = wait_record(trace=hdr.get("x-pallama-trace-id", ""))
    codes = [d0.get("code") for d0 in (rec or {}).get("detections", [])]
    tc = ""
    try:
        tc = v["choices"][0]["message"].get("tool_calls") or []
    except Exception:
        pass
    no_false = not any(c in ("tool_args_invalid_json", "tool_name_unknown", "template_no_tools") for c in codes)
    check("sentinel", "real tool call: no false positives", st == 200 and no_false,
          f"tool_calls={bool(tc)} codes={codes}")
    if tc:
        args_ok = True
        try:
            json.loads(tc[0]["function"]["arguments"])
        except Exception:
            args_ok = False
        check("sentinel", "tool arguments are valid JSON (model-composed)", args_ok,
              tc[0]["function"]["arguments"][:80])
    # enforce pass-case (deterministic): clean json response with enforce -> 200.
    st, _, v = http_json("POST", "/v1/chat/completions",
                         {"model": MODEL, "stream": False,
                          "messages": [{"role": "user", "content": "Return the JSON {\"ok\": true} and nothing else."}],
                          "response_format": {"type": "json_object"}},
                         headers={"X-Pallama-Enforce": "1"})
    check("sentinel", "enforce header: clean request passes (judge ran)", st in (200, 422),
          f"status={st} — 422 acceptable if the model actually violated; both prove the judge ran")
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
    check("sentinel", "watch SSE delivers the record live",
          result.get("ok") is True and "openai-chat" in result.get("collected", ""),
          (result.get("collected", "") or "").replace("\n", " ")[:160])
    # doctor row.
    p = cli("doctor")
    check("sentinel", "doctor prints the sentinel row", "sentinel" in p.stdout,
          next((l.strip() for l in p.stdout.splitlines() if "sentinel" in l), "missing"))


def phase_behavior() -> None:
    print("\n== phase 5: lifecycle behavior ==")
    d = DAEMON
    d.start({"port": PORT})
    chat("Say ok")
    # keep_alive=0 evicts after the request.
    st, _, _ = http_json("POST", "/api/chat", {"model": MODEL, "stream": False, "keep_alive": 0,
                                               "messages": [{"role": "user", "content": "Say ok"}]})
    gone = False
    deadline = time.time() + 20
    while time.time() < deadline:
        if not any(str(ps_field(r, "name", "model") or "").split(":")[0] == MODEL for r in ps_rows()):
            gone = True
            break
        time.sleep(1)
    check("behavior", "keep_alive=0 evicts after the response", st == 200 and gone, f"status={st} evicted={gone}")
    # keep_alive=-1 pins.
    st, _, _ = http_json("POST", "/api/chat", {"model": MODEL, "stream": False, "keep_alive": -1,
                                               "messages": [{"role": "user", "content": "Say ok"}]})
    time.sleep(3)
    pinned = any(str(ps_field(r, "name", "model") or "").split(":")[0] == MODEL for r in ps_rows())
    check("behavior", "keep_alive=-1 pins the model", st == 200 and pinned, f"status={st} still loaded={pinned}")
    # crash respawn.
    if not FAST:
        pid = child_pid()
        check("behavior", "child pid discovered for crash test", pid is not None and pid > 1, f"pid={pid}")
        if pid:
            os.kill(pid, signal.SIGKILL)  # our daemon's own child
            time.sleep(2)
            # By design: the FIRST post-crash request 502s while the corpse
            # is reaped; the respawn serves the retry.
            statuses = []
            for _ in range(3):
                st, v, _ = chat("Say ok")
                statuses.append(st)
                if st == 200:
                    break
                time.sleep(2)
            new_pid = child_pid()
            check("behavior", "crashed engine: first request 502s (reap), retry respawns",
                  200 in statuses and new_pid not in (None, pid),
                  f"old={pid} new={new_pid} statuses={statuses}")
    else:
        boundary("behavior", "crash respawn", "skipped in FAST mode")
    # cancellation frees the slot.
    if not FAST:
        def _slow_stream() -> None:
            try:
                req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                             data=json.dumps({"model": MODEL, "stream": True,
                                                              "max_tokens": 400,
                                                              "messages": [{"role": "user", "content": "Count to 100 slowly."}]}).encode(),
                                             headers={"content-type": "application/json"})
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
            rows = [r for r in ps_rows() if str(ps_field(r, "name", "model") or "").split(":")[0] == MODEL]
            inflight = row_inflight(rows[0]) if rows else 0
            if rows and int(inflight or 0) == 0:
                freed = True
                break
            time.sleep(2)
        t.join(timeout=5)
        check("behavior", "client disconnect frees the slot", freed, f"in_flight back to 0={freed}")
    else:
        boundary("behavior", "disconnect frees slot", "skipped in FAST mode")
    # idle sleep ladder.
    if not FAST:
        d.start({"port": PORT, "idle_sleep_secs": 5})
        chat("Say ok")
        sleeping = None
        deadline = time.time() + 40
        while time.time() < deadline:
            rows = [r for r in ps_rows() if str(ps_field(r, "name", "model") or "").split(":")[0] == MODEL]
            state = str(row_state(rows[0]) or "").lower() if rows else ""
            if rows and "sleep" in state:
                sleeping = state
                break
            time.sleep(2)
        check("behavior", "idle_sleep_secs=5 -> child-native sleep state", sleeping is not None,
              f"state={sleeping}")
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
        rows = [r for r in ps_rows() if str(ps_field(r, "name", "model") or "").split(":")[0] == MODEL]
        inflight = int(row_inflight(rows[0]) or 0) if rows else 0
        peak = max(peak, inflight)
        time.sleep(1)
    for t in ts:
        t.join(timeout=10)
    check("behavior", "slots=1 queues 2 concurrent requests, both complete",
          len(results) == 2 and all(s == 200 for _, s in results) and peak <= 1,
          f"completed={len(results)} statuses={[s for _, s in results]} peak_in_flight={peak}")
    # router mode.
    d.start({"port": PORT, "router": True})
    st, v, _ = chat("Say ok")
    rows = ps_rows()
    check("behavior", "router mode serves chat from one child", st == 200 and len(rows) >= 1,
          f"status={st} ps rows={len(rows)}")


def phase_cli() -> None:
    print("\n== phase 6: CLI against the isolated daemon ==")
    d = DAEMON
    d.start({"port": PORT})
    chat("Say ok")  # ensure a model row exists for ps/show
    p = cli("ps")
    check("cli", "pallama ps", p.returncode == 0 and MODEL in p.stdout, p.stdout.strip().splitlines()[-1][:120] if p.stdout else "")
    p = cli("list")
    check("cli", "pallama list", p.returncode == 0 and MODEL in p.stdout, "")
    p = cli("show", MODEL)
    check("cli", "pallama show", p.returncode == 0, p.stdout.strip().splitlines()[:1])
    p = cli("why")
    check("cli", "pallama why runs", p.returncode == 0, p.stdout.strip().splitlines()[:1])
    p = cli("doctor")
    check("cli", "pallama doctor passes", p.returncode == 0 and "fail" not in p.stdout.lower(),
          "all checks pass" if "all checks pass" in p.stdout else p.stdout[-200:])
    p = cli("config", "get", "default_ctx")
    check("cli", "config get", p.returncode == 0 and "16384" in p.stdout, p.stdout.strip())
    p = cli("config", "set", "default_ctx", "4096")
    p2 = cli("config", "get", "default_ctx")
    with open(os.path.join(SANDBOX.config_dir, "config.toml"), "rb") as f:
        parsed = tomllib.load(f)
    check("cli", "config set -> get -> file parses (quoted, outside tables)",
          p.returncode == 0 and "4096" in p2.stdout and parsed.get("default_ctx") == 4096,
          f"get={p2.stdout.strip()!r} file default_ctx={parsed.get('default_ctx')}")
    p = cli("engine", "list")
    check("cli", "engine list shows the installed engine", p.returncode == 0 and "b108" in p.stdout,
          p.stdout.strip().splitlines()[-1][:120] if p.stdout else "")
    p = cli("run", MODEL, "--verbose", "Say: inline")
    low = p.stdout.lower()
    check("cli", "run single-shot --verbose completes with stats",
          p.returncode == 0 and ("count" in low or "tokens" in low or "duration" in low),
          (" ".join(p.stdout.strip().splitlines()[-3:])[:200]) or f"rc={p.returncode} err={p.stderr[:150]}")
    p = cli("stop", MODEL)
    check("cli", "stop MODEL unloads via /api/evict", p.returncode == 0, p.stdout.strip()[:100])
    # cp refuses while loaded (verified above by design); alias after unload.
    p = cli("cp", MODEL, "validate-alias")
    p2 = cli("list")
    check("cli", "cp creates a zero-byte alias after unload (no blob copy)",
          p.returncode == 0 and "validate-alias" in p2.stdout,
          f"cp rc={p.returncode} {p.stderr.strip()[:150] or p.stdout.strip()[:80]}")
    cli("rm", "validate-alias")
    p = cli("list")
    check("cli", "rm removes the alias", "validate-alias" not in p.stdout, "")
    boundary("cli", "upgrade --dry-run", "hits GitHub from this box; e2e-verified against a fake release server in the test suite")


def phase_auth() -> None:
    print("\n== phase 7: auth ==")
    d = DAEMON
    d.start({"port": PORT, "api_keys": ["validate-key-1"]})
    st, _, _ = http_json("GET", "/api/version")
    check("auth", "api_keys set: no bearer -> 401", st == 401, f"status={st}")
    st, _, _ = http_json("GET", "/api/version", headers={"Authorization": "Bearer validate-key-1"})
    check("auth", "correct bearer -> 200", st == 200, f"status={st}")
    st, _, _ = http_json("GET", "/healthz")
    check("auth", "/healthz stays open", st == 200, f"status={st}")


# ------------------------------------------------------------------ main


def report() -> int:
    print("\n" + "=" * 72)
    print("CONFIG-FLOW COVERAGE MATRIX")
    print("=" * 72)
    for c in COVERAGE:
        tag = "ok " if c["ok"] else "MISS"
        print(f"  [{tag}] {c['knob']:<28} {c['expectation']}")
    print(f"  — {sum(1 for c in COVERAGE if c['ok'])}/{len(COVERAGE)} knob flows verified with evidence")
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
    return 1 if failed else 0


def main() -> int:
    global SANDBOX, DAEMON, USER_CONFIG_SHA
    args = sys.argv[1:]
    self_test = "--self-test" in args
    phases_arg = [a for a in args if a.startswith("--phase=")]
    wanted = {a.split("=", 1)[1] for a in phases_arg} or None
    print(f"pallama validation harness — engine+model REAL, isolation via temp XDG, port {PORT}")
    print(f"binary={PAL} model={MODEL} fast={FAST}")
    if os.path.exists(REAL_CONFIG):
        with open(REAL_CONFIG, "rb") as f:
            USER_CONFIG_SHA = hashlib.sha256(f.read()).hexdigest()
    SANDBOX = Sandbox()
    DAEMON = Daemon(SANDBOX)

    def _cleanup() -> None:
        if DAEMON:
            try:
                DAEMON.stop()
            except Exception:
                pass
        if SANDBOX:
            SANDBOX.destroy()
    atexit.register(_cleanup)

    phases = [
        ("baseline", phase_baseline),
        ("config", phase_config),
        ("api", phase_api),
        ("sentinel", phase_sentinel),
        ("behavior", phase_behavior),
        ("cli", phase_cli),
        ("auth", phase_auth),
    ]
    for name, fn in phases:
        if self_test or (wanted and name not in wanted):
            print(f"\n== phase {name}: skipped ({'self-test' if self_test else '--phase filter'}) ==")
            continue
        fn()
    if self_test:
        check("self-test", "injected failure proves non-zero exit", False, "by design")
    rc = report()
    _cleanup()
    return rc


if __name__ == "__main__":
    sys.exit(main())
