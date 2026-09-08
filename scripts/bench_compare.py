#!/usr/bin/env python3
"""pallama comparison benchmark — pallama vs direct llama-server vs ollama vs llama-bench.

Fair head-to-head with REAL model loads, printing each contender's effective
config (the actual child argv from /proc) before timing anything:

  pallama   sandboxed daemon (validate.py isolation contract), default config,
            port 11499. The engine child's argv is captured via /proc/<pid>/cmdline.
  direct    the SAME upstream llama-server binary relaunched with the argv
            CLONED from pallama's child (only --host/--port/--slot-save-path
            differ) on port 11501 -> measures pure orchestration tax.
  ollama    the live system service on 11434, API-only (never signaled, never
            restarted); num_ctx matched to pallama's for KV fairness; unloaded
            afterwards via keep_alive=0.
  ceiling   llama-bench (pp512 / tg128) at matched threads+ctx = engine limit.

Metrics per contender: cold load, warm TTFT p50, decode tok/s, prompt
processing tok/s (approx: includes the first decode step), wall tok/s, peak
child RSS, peak GPU memory. Warmup discarded, median of --runs.

Isolation contract (same as scripts/validate.py):
  - never binds or touches port 11434 (ollama) — asserted at startup
  - ollama is driven over HTTP only; a failed unload is reported, never forced
  - only pids this script spawned are signaled (single-pid TERM, no groups)
  - XDG sandbox under ~/.cache with its own models dir; user config untouched
  - memory floor checked before every model load — abort loud, not swap-thrash

Usage:
  scripts/bench_compare.py                       # full run, default model
  scripts/bench_compare.py --fast                # 0.5b model, 1 run (plumbing proof)
  scripts/bench_compare.py --runs 5 --model qwen2.5-0.5b-instruct
  scripts/bench_compare.py --skip ollama,ceiling # pallama vs direct only

Exit codes: 0 clean · 1 verdict failure (proxy tax > 5% or contender
hard-fail) · 2 environment abort (memory floor / port conflict).

Full config-knob and code-path validation lives in scripts/validate.py;
this harness is its performance-comparison peer.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import socket
import sqlite3
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import validate as V  # noqa: E402 — incident-hardened Sandbox/Daemon/mem helpers

PAL_PORT = V.PORT  # 11499 (validate.py's sandbox port)
DIRECT_PORT = 11501
OLLAMA_PORT = 11434
ENGINE_BIN_DIRECTIVE = "llama-server"

DECODE_PROMPT = (
    "Explain in one paragraph why the sky is blue at noon and red at sunset."
)
DECODE_MAX_TOKENS = 192
TTFT_NOTE = "first delta carrying content or reasoning_content"
PP_PARA = (
    "The history of computing machinery spans mechanical calculators, "
    "electromechanical relays, vacuum tubes, discrete transistors, and "
    "integrated circuits. Each generation traded reliability and speed for "
    "scale, and each shift rewrote what ordinary people could compute. "
    "Stored-program architectures separated instructions from data, time "
    "sharing turned machines into conversational resources, and compilers "
    "raised the abstraction floor until high-level languages read like prose. "
)
PP_PROMPT = PP_PARA * 8 + "\n\nSummarize the above in exactly one sentence."
PP_MAX_TOKENS = 4


def log(msg: str) -> None:
    print(msg, flush=True)


TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"type": "string", "enum": ["c", "f"]},
            },
            "required": ["city"],
        },
    },
}
TOOL_PROMPT = "What is the weather in Tokyo right now? Use the get_weather tool."


# ------------------------------------------------------------------ helpers


def req(
    port: int, method: str, path: str, body: dict | None = None, timeout: int = 120
) -> tuple[int, object]:
    """Non-streaming request; returns (status, parsed-json-or-raw-string)."""
    url = f"http://127.0.0.1:{port}{path}"
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(url, data=data, method=method)
    r.add_header("content-type", "application/json")
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            raw = resp.read()
            st = resp.status
    except urllib.error.HTTPError as e:
        raw, st = e.read(), e.code
    try:
        return st, json.loads(raw)
    except Exception:
        return st, raw.decode(errors="replace")


def gpu_used_mib() -> int | None:
    try:
        out = (
            subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            .stdout.strip()
            .splitlines()
        )
        return int(out[0].strip())
    except Exception:
        return None


def rss_mib(pid: int | None) -> int:
    if not pid:
        return 0
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return 0


def proc_exe(pid: int | None) -> str | None:
    """Real executable path of a pid (None when unreadable/dead)."""
    if not pid:
        return None
    try:
        return os.path.realpath(f"/proc/{pid}/exe")
    except OSError:
        return None


def wait_port_closed(port: int, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            s = socket.create_connection(("127.0.0.1", port), timeout=1)
            s.close()
            time.sleep(0.5)
        except OSError:
            return
    log(f"  ! port {port} still open after {timeout}s")


def wait_pid_gone(pid: int | None, timeout: float = 45.0) -> None:
    if not pid:
        return
    deadline = time.time() + timeout
    while time.time() < deadline and os.path.exists(f"/proc/{pid}"):
        time.sleep(0.5)


def wait_gpu_settled(baseline_mib: int | None, timeout: float = 90.0) -> None:
    """Wait until GPU use is at/near baseline so contenders never share VRAM."""
    if baseline_mib is None:
        return
    last = None
    stable = 0
    deadline = time.time() + timeout
    while time.time() < deadline:
        now = gpu_used_mib()
        if now is None:
            return
        if now <= baseline_mib + 60:
            stable += 1
            if stable >= 2:
                return
        else:
            stable = 0
        last = now
        time.sleep(2.0)
    log(
        f"  ! GPU did not settle to baseline {baseline_mib} MiB (last={last}); continuing"
    )


def mem_guard(model_bytes_mib: float, what: str) -> None:
    need = int(model_bytes_mib * 1.25) + 1024
    avail = V.mem_available_mib()
    if avail < max(V.MEM_FLOOR_MIB, need):
        raise EnvironmentError(
            f"memory floor for {what}: MemAvailable {avail} MiB < needed ~{need} MiB "
            f"(model {int(model_bytes_mib)} MiB + headroom). Free memory or skip "
            f"contenders and retry."
        )


def release_ollama_resident(ctx: dict) -> None:
    """Ask the live ollama service to unload resident models before a
    contender claims the machine (one-engine-resident discipline). Without
    this, a warm ollama runner holds ~1-2 GiB RSS + VRAM and mem_guard
    aborts pallama/direct on boxes that ran fine cold.
    """
    try:
        ps = req(OLLAMA_PORT, "GET", "/api/ps", timeout=5)[1]
        loaded = (
            [
                m.get("name") or m.get("model", "")
                for m in (ps.get("models") or [])
                if isinstance(m, dict)
            ]
            if isinstance(ps, dict)
            else []
        )
    except Exception:
        return  # ollama not reachable — nothing to release
    for name in loaded:
        if name:
            log(f"  releasing ollama resident model {name} (keep_alive=0)")
            ollama_unload(name)
    if loaded:
        wait_gpu_settled(ctx["gpu_idle_mib"])


# ----------------------------------------------------------- stream timing


def _recv_stream(
    port: int,
    payload: bytes,
    path: str,
    budget: float,
    on_obj,
    method: str | None = None,
    extra_headers: dict | None = None,
    stop_on_first: bool = False,
) -> dict:
    """Raw-socket HTTP POST/GET; on_obj(obj) per SSE `data:` / NDJSON line.

    Raw sockets, not urllib: a read timeout poisons http.client streams and
    SSE legally idles between events (per-recv timeouts are keepalives).
    """
    method = method or ("POST" if payload else "GET")
    head = (
        f"{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
        "content-type: application/json\r\nconnection: close\r\n"
        f"content-length: {len(payload)}\r\n"
        + "".join(f"{k}: {v}\r\n" for k, v in (extra_headers or {}).items())
        + "\r\n"
    ).encode()
    t0 = t_last = time.perf_counter()
    status, first_tok_t, first_reason_t, first_content_t = None, None, None, None
    usage, done = None, False
    buf = b""
    err = ""
    s = None
    deadline = time.time() + budget
    try:
        s = socket.create_connection(("127.0.0.1", port), timeout=30)
        s.sendall(head + payload)
        s.settimeout(10)
        while time.time() < deadline and not done:
            try:
                chunk = s.recv(65536)
            except socket.timeout:
                continue
            if not chunk:
                break
            t_last = time.perf_counter()
            buf += chunk
            if status is None and b"\r\n" in buf:
                parts = buf.split(b"\r\n", 1)
                bits = parts[0].split(b" ")
                if len(bits) >= 2:
                    try:
                        status = int(bits[1])
                    except ValueError:
                        status = 0
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line.startswith(b"data:"):
                    if line.startswith(b"{"):  # NDJSON (ollama)
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        r = on_obj(obj, first_tok_t is None)
                        if (
                            r in ("first", "reasoning", "content")
                            and first_tok_t is None
                        ):
                            first_tok_t = time.perf_counter()
                        if r == "reasoning" and first_reason_t is None:
                            first_reason_t = time.perf_counter()
                        elif r == "content" and first_content_t is None:
                            first_content_t = time.perf_counter()
                        elif r == "done":
                            done = True
                        elif isinstance(r, dict):
                            usage = r
                    continue
                data = line[len(b"data:") :].strip()
                if data == b"[DONE]":
                    done = True
                    break
                try:
                    obj = json.loads(data)
                except Exception:
                    continue
                r = on_obj(obj, first_tok_t is None)
                if r in ("first", "reasoning", "content") and first_tok_t is None:
                    first_tok_t = time.perf_counter()
                    if stop_on_first:
                        done = True
                if r == "reasoning" and first_reason_t is None:
                    first_reason_t = time.perf_counter()
                elif r == "content" and first_content_t is None:
                    first_content_t = time.perf_counter()
                elif r == "done":
                    done = True
                elif isinstance(r, dict):
                    usage = r
    except Exception as e:
        err = str(e)
    finally:
        if s:
            try:
                s.close()
            except OSError:
                pass
    return {
        "status": status,
        "ttft_ms": (first_tok_t - t0) * 1000 if first_tok_t else None,
        "ttft_reasoning_ms": (first_reason_t - t0) * 1000 if first_reason_t else None,
        "ttft_content_ms": (first_content_t - t0) * 1000 if first_content_t else None,
        "total_ms": (t_last - t0) * 1000,
        "usage": usage,
        "error": err,
    }


def openai_stream_timed(
    port: int,
    model: str,
    prompt: str,
    max_tokens: int,
    budget: float = 300.0,
    headers: dict | None = None,
    think: bool | None = None,
) -> dict:
    """POST /v1/chat/completions (stream). Times first content-bearing delta.

    Also records the reasoning/answer split when the model emits
    `reasoning_content`: separate TTFTs and accumulated chars per channel,
    so reasoning economics (burn rate, answer latency) are measurable.
    `think=False` flips the template's think switch (qwen-style).
    """
    body: dict = {
        "model": model,
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if think is False:
        body["chat_template_kwargs"] = {"enable_thinking": False}

    split = {"reasoning_chars": 0, "content_chars": 0}

    def on_obj(obj: dict, want_first: bool):
        u = obj.get("usage")
        if isinstance(u, dict) and (
            u.get("completion_tokens") or u.get("prompt_tokens")
        ):
            usage = dict(u)
        else:
            usage = None
        for ch in obj.get("choices") or []:
            d = ch.get("delta") or {}
            r = d.get("reasoning_content") or ""
            c = d.get("content") or ""
            split["reasoning_chars"] += len(r)
            split["content_chars"] += len(c)
            if want_first and r:
                return "reasoning"
            if want_first and c:
                return "content"
        if obj.get("usage") is not None and not (obj.get("choices")):
            return usage or "done"
        return usage

    res = _recv_stream(
        port,
        json.dumps(body).encode(),
        "/v1/chat/completions",
        budget,
        on_obj,
        extra_headers=headers,
    )
    res["reasoning_chars"] = split["reasoning_chars"]
    res["content_chars"] = split["content_chars"]
    return res


def ollama_stream_timed(
    port: int,
    model: str,
    prompt: str,
    max_tokens: int,
    num_ctx: int | None,
    keep_alive: str = "10m",
    budget: float = 300.0,
) -> dict:
    """POST /api/chat (NDJSON stream). Times first content-bearing message."""
    body: dict = {
        "model": model,
        "stream": True,
        "keep_alive": keep_alive,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"num_predict": max_tokens},
    }
    if num_ctx:
        body["options"]["num_ctx"] = num_ctx

    def on_obj(obj: dict, want_first: bool):
        msg = obj.get("message") or {}
        if want_first and (
            msg.get("content") or msg.get("thinking") or msg.get("reasoning_content")
        ):
            return "first"
        if obj.get("done"):
            return {
                "prompt_tokens": obj.get("prompt_eval_count"),
                "completion_tokens": obj.get("eval_count"),
            }
        return None

    return _recv_stream(port, json.dumps(body).encode(), "/api/chat", budget, on_obj)


def metric_from(res: dict) -> dict:
    """Normalize one stream result into ttft/decode/pp/wall metrics."""
    u = res.get("usage") or {}
    pt, ct = u.get("prompt_tokens"), u.get("completion_tokens")
    out = {
        "ok": res["status"] == 200 and res["ttft_ms"] is not None and ct,
        "status": res["status"],
        "ttft_ms": res["ttft_ms"],
        "total_ms": res["total_ms"],
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "error": res["error"],
    }
    if out["ok"]:
        ttft_s = res["ttft_ms"] / 1000
        total_s = res["total_ms"] / 1000
        out["decode_tps"] = (
            (ct - 1) / (total_s - ttft_s) if total_s > ttft_s and ct > 1 else None
        )
        out["wall_tps"] = ct / total_s if total_s > 0 else None
        out["pp_tps"] = (pt / ttft_s) if pt and ttft_s > 0 else None
    return out


# ---------------------------------------------------------------- sampler


class Sampler:
    """Peak child RSS + GPU memory observer while a contender runs."""

    def __init__(self) -> None:
        self.pid: int | None = None
        self.rss_peak = 0
        self.gpu_peak: int | None = None
        self.gpu_base: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, pid: int | None) -> None:
        self._stop.clear()
        self.pid = pid
        self.gpu_base = gpu_used_mib()
        self.gpu_peak = self.gpu_base
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        tick = 0
        while not self._stop.is_set():
            self.rss_peak = max(self.rss_peak, rss_mib(self.pid))
            if tick % 3 == 0:  # nvidia-smi is ~100ms; sample it sparsely
                g = gpu_used_mib()
                if g is not None:
                    self.gpu_peak = max(self.gpu_peak or 0, g)
            tick += 1
            self._stop.wait(0.4)

    def stop(self) -> dict:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        return {
            "rss_peak_mib": self.rss_peak,
            "gpu_peak_mib": self.gpu_peak,
            "gpu_base_mib": self.gpu_base,
            "gpu_delta_mib": (self.gpu_peak - self.gpu_base)
            if self.gpu_peak is not None and self.gpu_base is not None
            else None,
        }


def pp_nonce_prompt() -> str:
    """Unique prefill prompt: a random nonce paragraph FIRST, then the body.

    The nonce leads so no prefix can hit the unified cache (--cache-reuse
    matches prefixes); every run forces a true full prefill. Without this,
    identical re-sent prompts make pp_tps = tokens/TTFT report cache-hit
    speed (~3x the engine's real prefill ceiling) — a fake number.
    """
    nonce = (
        f"Reference {secrets.token_hex(24)}: disregard this marker, "
        f"session {time.time_ns()}. "
    )
    return nonce + PP_PARA * 8 + "\n\nSummarize the above in exactly one sentence."


def run_suite(stream_fn, runs: int, tag: str) -> dict:
    """Warmup + N decode runs + N pp runs -> median metrics.

    Decode runs reuse one fixed short prompt (negligible prefix). PP runs
    each use a fresh unique prompt so pp_tps measures real prefill.
    """
    warm = metric_from(stream_fn(DECODE_PROMPT, DECODE_MAX_TOKENS))
    if not warm["ok"]:
        raise RuntimeError(
            f"{tag}: warmup failed status={warm['status']} err={warm['error']!r} "
            f"(tokens={warm['completion_tokens']})"
        )
    decode, pp = [], []
    for _ in range(runs):
        decode.append(metric_from(stream_fn(DECODE_PROMPT, DECODE_MAX_TOKENS)))
        pp.append(metric_from(stream_fn(pp_nonce_prompt(), PP_MAX_TOKENS)))
    bad = [m for m in decode + pp if not m["ok"]]
    if bad:
        raise RuntimeError(
            f"{tag}: {len(bad)}/{runs * 2} runs failed "
            f"(status={bad[0]['status']} err={bad[0]['error']!r})"
        )

    def med(vals, key):
        xs = sorted(v[key] for v in vals if v[key] is not None)
        return statistics.median(xs) if xs else None

    return {
        "ttft_ms_med": med(decode, "ttft_ms"),
        "ttft_ms_min": min(
            (m["ttft_ms"] for m in decode if m["ttft_ms"] is not None), default=None
        ),
        "decode_tps_med": med(decode, "decode_tps"),
        "decode_tps_min": min(
            (m["decode_tps"] for m in decode if m["decode_tps"] is not None),
            default=None,
        ),
        "decode_tps_max": max(
            (m["decode_tps"] for m in decode if m["decode_tps"] is not None),
            default=None,
        ),
        "pp_tps_med": med(pp, "pp_tps"),
        "wall_tps_med": med(decode, "wall_tps"),
        "pp_prompt_tokens": decode and pp[0]["prompt_tokens"],
        "decode_completion_tokens": decode[0]["completion_tokens"],
        "runs": runs,
    }


# -------------------------------------------------------------- contenders


def find_child_pid(data_dir: str, model: str, timeout: float = 300.0) -> int | None:
    deadline = time.time() + timeout
    pidfile = os.path.join(data_dir, "run", f"{model}.pid")
    while time.time() < deadline:
        try:
            with open(pidfile) as f:
                pid = int(f.read().strip())
            if os.path.exists(f"/proc/{pid}"):
                return pid
        except (OSError, ValueError):
            pass
        time.sleep(0.5)
    return None


def read_argv(pid: int | None) -> list[str]:
    if not pid:
        return []
    with open(f"/proc/{pid}/cmdline", "rb") as f:
        return [a for a in f.read().decode(errors="replace").split("\0") if a]


def clone_direct_argv(argv: list[str]) -> list[str]:
    """Copy pallama's child argv; swap only host/port and drop slot-save-path.

    Index-walk, never string surgery: flags and values are argv pairs.
    --slot-save-path points into the sandbox run dir, which is destroyed
    while the direct server still runs; sessions persist only on explicit
    save, so dropping it changes no performance characteristic.
    """
    drop_with_value = {"--host", "--port", "--slot-save-path"}
    out: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in drop_with_value:
            i += 2
            continue
        out.append(a)
        i += 1
    out += ["--host", "127.0.0.1", "--port", str(DIRECT_PORT)]
    return out


def bench_pallama(ctx: dict) -> dict:
    log(f"\n--- contender: pallama (sandbox daemon :{PAL_PORT}, default config) ---")
    sb, model, runs = ctx["sandbox"], ctx["model"], ctx["runs"]
    release_ollama_resident(ctx)
    mem_guard(ctx["model_mib"], "pallama")
    d = V.Daemon(sb)
    ctx["daemon"] = d
    d.start({"port": PAL_PORT})

    try:
        # Cold load: first request pays child spawn + model load (lazy loader).
        # Sampler starts before the request so the load-time VRAM/RSS ramp is
        # captured; a watcher attaches the child pid the moment its pidfile
        # appears (mid-request) so RSS is sampled through the load too.
        sampler = Sampler()
        sampler.start(None)

        def watch_pidfile() -> None:
            while sampler.pid is None:
                pid = find_child_pid(sb.data_dir, model, timeout=0.2)
                if pid:
                    sampler.pid = pid
                    return

        watcher = threading.Thread(target=watch_pidfile, daemon=True)
        watcher.start()
        cold = openai_stream_timed(PAL_PORT, model, DECODE_PROMPT, DECODE_MAX_TOKENS)
        watcher.join(timeout=5)
        pid = sampler.pid or find_child_pid(sb.data_dir, model, timeout=30)
        sampler.pid = pid
        cold_m = metric_from(cold)
        if not cold_m["ok"]:
            raise RuntimeError(f"pallama cold request failed: {cold_m}")
        ps = req(PAL_PORT, "GET", "/api/ps")[1]
        ctx_rows = (
            (ps.get("models") or ps.get("instances") or [])
            if isinstance(ps, dict)
            else []
        )
        row = next(
            (r for r in ctx_rows if isinstance(r, dict) and model in json.dumps(r)), {}
        )
        argv = read_argv(pid)
        # Fairness evidence: the exact engine invocation pallama compiled.
        ctx_val = row.get("pallama_ctx") or row.get("ctx")
        log(f"  child pid={pid} ctx={ctx_val}")
        log("  child argv: " + " ".join(argv))
        ctx["pallama_argv"] = argv
        if ctx_val:
            ctx["pallama_ctx"] = int(ctx_val) if str(ctx_val).isdigit() else ctx_val

        def stream(prompt: str, max_tokens: int) -> dict:
            return openai_stream_timed(PAL_PORT, model, prompt, max_tokens)

        suite = run_suite(stream, runs, "pallama")

        # ollama-compat translation layer tax (same model, /api/chat).
        compat = metric_from(
            ollama_stream_timed(
                PAL_PORT, model, DECODE_PROMPT, DECODE_MAX_TOKENS, num_ctx=None
            )
        )
        peaks = sampler.stop()
        return {
            "suite": suite,
            "cold_request_to_first_tok_s": (cold_m["ttft_ms"] or 0) / 1000,
            "compat_decode_tps": compat.get("decode_tps"),
            "child_pid": pid,
            "engine_argv": argv,
            "ctx": ctx_val,
            **peaks,
        }
    finally:
        d.stop()
        wait_pid_gone(find_child_pid(sb.data_dir, model, timeout=1), timeout=30)
        wait_gpu_settled(ctx["gpu_idle_mib"])


def bench_direct(ctx: dict) -> dict:
    argv_src = ctx.get("pallama_argv")
    if not argv_src:
        return {
            "skipped": "no pallama child argv captured (run pallama contender first)"
        }
    log(
        f"\n--- contender: direct llama-server :{DIRECT_PORT} (argv cloned from pallama) ---"
    )
    engine_bin = argv_src[0]
    engine_dir = os.path.dirname(engine_bin)
    argv = clone_direct_argv(argv_src)
    log(f"  engine: {engine_bin}")
    log("  argv  : " + " ".join(argv))
    diff = [a for a in argv_src if a not in argv]
    log(f"  stripped from pallama argv: {diff} (host/port/slot-save-path only)")
    release_ollama_resident(ctx)
    mem_guard(ctx["model_mib"], "direct")
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = engine_dir + (
        ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
    )
    log_path = os.path.join(ctx["outdir"], "direct-llama-server.log")
    proc = None
    t_spawn = time.time()
    sampler = Sampler()
    sampler.start(None)  # GPU baseline must predate the model load
    try:
        with open(log_path, "wb") as lf:
            proc = subprocess.Popen(
                argv,
                stdout=lf,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                env=env,
                cwd=engine_dir,
            )
        sampler.pid = proc.pid
        deadline = time.time() + 300
        ready = None
        while time.time() < deadline:
            try:
                st, _ = req(DIRECT_PORT, "GET", "/health", timeout=5)
                if st == 200:
                    ready = time.time()
                    break
            except Exception:
                if proc.poll() is not None:
                    raise RuntimeError(
                        f"direct llama-server exited rc={proc.returncode}; log tail:\n"
                        + open(log_path, errors="replace").read()[-1500:]
                    )
                time.sleep(0.5)
        if ready is None:
            raise RuntimeError("direct llama-server not healthy in 300s")
        spawn_to_ready = ready - t_spawn

        model_name = ctx["model"]

        def stream(prompt: str, max_tokens: int) -> dict:
            return openai_stream_timed(DIRECT_PORT, model_name, prompt, max_tokens)

        suite = run_suite(stream, ctx["runs"], "direct")
        peaks = sampler.stop()
        return {
            "suite": suite,
            "cold_spawn_to_ready_s": spawn_to_ready,
            "engine_argv": argv,
            **peaks,
        }
    finally:
        if proc and proc.poll() is None:
            proc.terminate()  # single pid we spawned; never a group
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        wait_gpu_settled(ctx["gpu_idle_mib"])


def ollama_unload(model: str, timeout: float = 90.0) -> bool:
    req(OLLAMA_PORT, "POST", "/api/generate", {"model": model, "keep_alive": 0})
    deadline = time.time() + timeout
    while time.time() < deadline:
        ps = req(OLLAMA_PORT, "GET", "/api/ps", timeout=10)[1]
        names = (
            [m.get("name", "") + m.get("model", "") for m in (ps.get("models") or [])]
            if isinstance(ps, dict)
            else []
        )
        if not any(model in n for n in names):
            return True
        time.sleep(1.5)
    return False


def ollama_runner_pid(model: str) -> int | None:
    """Locate the ollama runner pid by scanning /proc (its API exposes none).

    ollama >=0.6 spawns its llama-server fork as the runner: argv is
    `<...>/ollama/llama-server --model <blobs>/sha256-<hash> ...` — no
    literal 'runner' token (the pre-0.6 `ollama runner` name is gone, and
    matching on it silently returned None = the RSS=0 column bug). Match
    instead on: exe path contains 'ollama' (or argv[0] does when the exe
    link is unreadable) AND '--model' present; prefer the argv that
    mentions this model's tag family, else the first match.
    """
    fallback = None
    stem = model.split(":")[0].replace("-", "").replace(".", "").replace("_", "")
    for pid_s in os.listdir("/proc"):
        if not pid_s.isdigit() or int(pid_s) == os.getpid():
            continue
        try:
            with open(f"/proc/{pid_s}/cmdline", "rb") as f:
                cmd = f.read()
        except OSError:
            continue
        if b"--model" not in cmd:
            continue
        exe = proc_exe(int(pid_s)) or ""
        argv0 = cmd.split(b"\0", 1)[0].decode(errors="replace")
        if "ollama" not in exe and "ollama" not in argv0:
            continue
        pid = int(pid_s)
        if fallback is None:
            fallback = pid
        # prefer a runner whose blob name or argv mentions this model's tag
        joined = cmd.decode(errors="replace")
        norm = joined.replace("-", "").replace(".", "").replace("_", "")
        if stem and stem in norm:
            return pid
    return fallback


def resolve_ollama_model(ctx: dict) -> str | None:
    """Match a pallama store name to an ollama tag via /api/tags."""
    if ctx.get("ollama_model"):
        return ctx["ollama_model"]
    tags = req(OLLAMA_PORT, "GET", "/api/tags", timeout=10)[1]
    if not isinstance(tags, dict):
        return None

    def norm(s: str) -> str:
        return s.replace("-", "").replace(".", "").replace(":", "").replace("_", "")

    want = norm(ctx["model"])
    for m in tags.get("models") or []:
        name = m.get("name", "")
        base = norm(name.split(":")[0])
        if base == want or (want and want.startswith(base)):
            return name
    return None


def bench_ollama(ctx: dict) -> dict:
    log(f"\n--- contender: ollama (live service :{OLLAMA_PORT}, API-only) ---")
    model = resolve_ollama_model(ctx)
    if not model:
        return {"skipped": f"no ollama model matching {ctx['model']!r} in /api/tags"}
    log(f"  ollama model: {model}")
    show = req(OLLAMA_PORT, "POST", "/api/show", {"model": model}, timeout=30)[1]
    mi = show.get("model_info", {}) if isinstance(show, dict) else {}
    quant_ft = mi.get("general.file_type")
    log(
        f"  ollama engine: {req(OLLAMA_PORT, 'GET', '/api/version', timeout=10)[1]}"
        f"  file_type={quant_ft}"
    )
    mem_guard(ctx["model_mib"], "ollama")
    if not ollama_unload(model):
        log(
            "  ! ollama unload request did not free the model; continuing "
            "(VRAM may be shared -> results marked)"
        )
    wait_gpu_settled(ctx["gpu_idle_mib"])

    ctx_match = ctx.get("pallama_ctx")
    num_ctx_used = ctx_match
    ctx_mismatch = False

    def try_stream(num_ctx: int | None):
        return ollama_stream_timed(
            OLLAMA_PORT, model, DECODE_PROMPT, DECODE_MAX_TOKENS, num_ctx=num_ctx
        )

    sampler = Sampler()
    sampler.start(None)  # GPU baseline must predate the model load

    # Attach the runner pid MID-LOAD (same semantics as the pallama
    # contender): the cold request runs in-flight while a watcher polls
    # /proc, so the RSS ramp through model load is captured. Attaching
    # only after cold completes missed the load peak entirely.
    cold_box: dict = {}

    def run_cold() -> None:
        cold_box["res"] = try_stream(num_ctx_used)

    ct = threading.Thread(target=run_cold, daemon=True)
    ct.start()
    watcher_deadline = time.time() + 240
    pid = None
    while time.time() < watcher_deadline and not sampler.pid:
        pid = ollama_runner_pid(model)
        if pid:
            sampler.pid = pid
            break
        time.sleep(0.2)
    ct.join(timeout=300)
    cold = metric_from(cold_box.get("res") or {})
    if not sampler.pid:
        pid = ollama_runner_pid(model)
        sampler.pid = pid
        if pid is None:
            log(
                "  ! ollama runner pid not found — RSS column will read 0 "
                "(GPU delta unaffected)"
            )
    if not cold["ok"] and num_ctx_used:
        log(
            f"  ! ollama failed at matched num_ctx={num_ctx_used} "
            f"(status={cold['status']}); retrying at ollama default ctx"
        )
        ctx_mismatch = True
        num_ctx_used = None
        cold = metric_from(try_stream(None))
    if not cold["ok"]:
        raise RuntimeError(
            f"ollama cold request failed: status={cold['status']} err={cold['error']!r}"
        )
    log(f"  runner pid={sampler.pid} num_ctx={num_ctx_used or 'ollama-default'}")
    engine_exe = proc_exe(sampler.pid) or "ollama-runner (exe unreadable)"

    def stream(prompt: str, max_tokens: int) -> dict:
        return ollama_stream_timed(
            OLLAMA_PORT, model, prompt, max_tokens, num_ctx=num_ctx_used
        )

    suite = run_suite(stream, ctx["runs"], "ollama")
    peaks = sampler.stop()
    if not ollama_unload(model):
        log(
            "  ! ollama model still resident after keep_alive=0 (its own idle "
            "policy will unload it)"
        )
    wait_gpu_settled(ctx["gpu_idle_mib"])
    return {
        "model": model,
        "suite": suite,
        "cold_request_to_first_tok_s": (cold["ttft_ms"] or 0) / 1000,
        "num_ctx": num_ctx_used,
        "ctx_mismatch": ctx_mismatch,
        "runner_pid": sampler.pid,
        "engine_exe": engine_exe,
        "file_type": quant_ft,
        **peaks,
    }


def bench_ceiling(ctx: dict) -> dict:
    argv_src = ctx.get("pallama_argv") or []
    engine_dir = os.path.dirname(argv_src[0]) if argv_src else None
    if not engine_dir:
        return {"skipped": "no engine dir (pallama contender must run first)"}
    log("\n--- contender: llama-bench ceiling (engine limit, matched t/ctx) ---")
    bench = os.path.join(engine_dir, "llama-bench")
    threads = "16"
    for i, a in enumerate(argv_src):
        if a == "--threads" and i + 1 < len(argv_src):
            threads = argv_src[i + 1]
    cmd = [
        bench,
        "-m",
        ctx["gguf_path"],
        "-p",
        "512",
        "-n",
        "128",
        "-t",
        threads,
        "-ngl",
        "999",
        "-fa",
        "auto",
        "-o",
        "json",
    ]
    log("  " + " ".join(cmd))
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = engine_dir + (
        ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
    )
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=1200, env=env)
    except subprocess.TimeoutExpired:
        return {"skipped": "llama-bench timed out (20m)"}
    if p.returncode != 0:
        return {"skipped": f"llama-bench rc={p.returncode}: {p.stderr[-300:]}"}
    with open(os.path.join(ctx["outdir"], "llama-bench.json"), "w") as f:
        f.write(p.stdout)
    pp = tg = None
    try:
        rows = json.loads(p.stdout)
        for row in rows:
            if row.get("n_gen") in (0, None) and row.get("n_prompt") == 512:
                pp = row.get("avg_ts")
            elif row.get("n_prompt") in (0, None) and row.get("n_gen") == 128:
                tg = row.get("avg_ts")
    except Exception:
        pass
    if pp is None or tg is None:
        return {"skipped": f"unparsed llama-bench json (raw saved): {p.stdout[:200]}"}
    return {"pp512_tps": pp, "tg128_tps": tg, "argv": cmd}


# ------------------------------------------------------ feature/architecture


class DirectServer:
    """Owns a direct llama-server launched with pallama's cloned argv."""

    def __init__(self, ctx: dict) -> None:
        self.argv_src = ctx.get("pallama_argv") or []
        self.proc: subprocess.Popen | None = None
        self.spawn_to_ready_s: float | None = None
        self.log_path = os.path.join(ctx["outdir"], "direct-llama-server.log")

    def __enter__(self) -> "DirectServer":
        if not self.argv_src:
            raise RuntimeError("no pallama child argv captured")
        argv = clone_direct_argv(self.argv_src)
        engine_dir = os.path.dirname(self.argv_src[0])
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = engine_dir + (
            ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
        )
        t_spawn = time.time()
        with open(self.log_path, "ab") as lf:
            self.proc = subprocess.Popen(
                argv,
                stdout=lf,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                env=env,
                cwd=engine_dir,
            )
        deadline = time.time() + 300
        while time.time() < deadline:
            try:
                st, _ = req(DIRECT_PORT, "GET", "/health", timeout=5)
                if st == 200:
                    self.spawn_to_ready_s = time.time() - t_spawn
                    return self
            except Exception:
                if self.proc.poll() is not None:
                    raise RuntimeError(
                        f"direct llama-server exited rc={self.proc.returncode}; log tail:\n"
                        + open(self.log_path, errors="replace").read()[-1500:]
                    )
                time.sleep(0.5)
        raise RuntimeError("direct llama-server not healthy in 300s")

    def __exit__(self, *exc) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()  # single pid we spawned; never a group
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
        wait_port_closed(DIRECT_PORT)


def png_b64(w: int = 64, h: int = 64, rgb: tuple = (180, 40, 40)) -> str:
    """Hand-crafted PNG (stdlib zlib/struct) so vision probes need no deps."""
    import base64
    import struct
    import zlib

    row = bytes(rgb) * w
    raw = b"".join(b"\x00" + row for _ in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    return base64.b64encode(png).decode()


def probe_routes(port: int, model: str, ollama: bool = False) -> dict:
    """Capability probe: which routes does this server actually serve?"""
    tiny_chat = {
        "model": model,
        "max_tokens": 1,
        "messages": [{"role": "user", "content": "hi"}],
    }
    if ollama:
        tiny_chat = {
            "model": model,
            "stream": False,
            "keep_alive": 0,
            "messages": [{"role": "user", "content": "hi"}],
        }
    probes = [
        ("GET /health", "GET", "/health", None),
        ("GET /v1/models", "GET", "/v1/models", None),
        ("GET /api/tags", "GET", "/api/tags", None),
        ("GET /api/ps", "GET", "/api/ps", None),
        ("GET /api/version", "GET", "/api/version", None),
        ("GET /metrics", "GET", "/metrics", None),
        ("POST /api/show", "POST", "/api/show", {"model": model}),
        ("POST /v1/chat/completions", "POST", "/v1/chat/completions", tiny_chat),
        (
            "POST /v1/completions",
            "POST",
            "/v1/completions",
            {"model": model, "max_tokens": 1, "prompt": "hi"},
        ),
        (
            "POST /v1/embeddings",
            "POST",
            "/v1/embeddings",
            {"model": model, "input": "hello"},
        ),
        (
            "POST /v1/rerank",
            "POST",
            "/v1/rerank",
            {"model": model, "query": "hi", "documents": ["a", "b"]},
        ),
        ("POST /tokenize", "POST", "/tokenize", {"model": model, "content": "hello"}),
        ("POST /detokenize", "POST", "/detokenize", {"model": model, "tokens": [9707]}),
        (
            "POST /apply-template",
            "POST",
            "/apply-template",
            {"model": model, "messages": [{"role": "user", "content": "hi"}]},
        ),
        (
            "POST /v1/messages",
            "POST",
            "/v1/messages",
            {
                "model": model,
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "hi"}],
            },
        ),
        (
            "POST /v1/responses",
            "POST",
            "/v1/responses",
            {"model": model, "max_output_tokens": 1, "input": "hi"},
        ),
        (
            "POST /infill",
            "POST",
            "/infill",
            {
                "model": model,
                "input_prefix": "def ",
                "input_suffix": "():",
                "max_tokens": 1,
            },
        ),
        ("GET /v1/adapters", "GET", "/v1/adapters", None),
        ("POST /api/chat", "POST", "/api/chat", dict(tiny_chat, stream=False)),
        (
            "POST /api/generate",
            "POST",
            "/api/generate",
            {
                "model": model,
                "prompt": "hi",
                "stream": False,
                **({"keep_alive": 0} if ollama else {}),
            },
        ),
        (
            "POST /api/embed",
            "POST",
            "/api/embed",
            {"model": model, "input": "hello", **({"keep_alive": 0} if ollama else {})},
        ),
    ]
    out: dict[str, str] = {}
    for label, method, path, body in probes:
        try:
            st, _ = req(port, method, path, body, timeout=60)
        except Exception:
            out[label] = "err"
            continue
        if st == 200:
            out[label] = "ok"
        elif st in (400, 405, 422):
            out[label] = (
                "route"  # exists; probe body rejected (e.g. not an embed model)
            )
        else:
            out[label] = f"—{st}"
    # SSE endpoints: prove they stream at all.
    ok_events, _ = sse_probe(port, "/api/events")
    out["GET /api/events (SSE)"] = "ok" if ok_events else "—"
    if not ollama:
        ok_watch, _ = sse_probe(port, "/api/watch")
        out["GET /api/watch (SSE)"] = "ok" if ok_watch else "—"
    return out


def sse_probe(port: int, path: str, budget: float = 4.0) -> tuple[bool, float]:
    """GET an SSE endpoint; report whether any bytes arrived."""
    t0 = time.perf_counter()
    got = {"any": False}

    def on_obj(obj, want_first):
        got["any"] = True
        return None

    r = _recv_stream(port, b"", path, budget, on_obj, method="GET")
    ok = r["status"] == 200 and (r["total_ms"] or 0) > 0 and not r["error"]
    return bool(ok and r["status"] == 200), r["total_ms"]


def _timed(fn, n: int = 2) -> dict:
    """Run an op n times; return {ms_p50, ok, status, note}."""
    xs, status, note = [], None, ""
    for _ in range(n):
        t = time.perf_counter()
        try:
            ok, status, note = fn()
        except Exception as e:
            ok, status, note = False, None, str(e)[:80]
        if ok:
            xs.append((time.perf_counter() - t) * 1000)
    return {
        "ms": statistics.median(xs) if xs else None,
        "ok": bool(xs),
        "status": status,
        "note": note,
    }


def time_feature_ops(
    port: int, model: str, ollama: bool = False, vision: bool = False
) -> dict:
    """Latency of every user-facing feature, one dict per op."""
    ops: dict[str, dict] = {}

    if ollama:

        def chat32():
            st, v = req(
                port,
                "POST",
                "/api/chat",
                {
                    "model": model,
                    "stream": False,
                    "keep_alive": "10m",
                    "messages": [
                        {"role": "user", "content": "Say hello in one sentence."}
                    ],
                    "options": {"num_predict": 32},
                },
                timeout=120,
            )
            return st == 200, st, ""
    else:

        def chat32():
            st, v = req(
                port,
                "POST",
                "/v1/chat/completions",
                {
                    "model": model,
                    "max_tokens": 32,
                    "messages": [
                        {"role": "user", "content": "Say hello in one sentence."}
                    ],
                },
                timeout=120,
            )
            return st == 200, st, ""

    ops["chat non-stream 32tok"] = _timed(chat32)

    def completions32():
        st, _ = req(
            port,
            "POST",
            "/v1/completions",
            {"model": model, "max_tokens": 32, "prompt": "The capital of France is"},
            timeout=120,
        )
        return st == 200, st, ""

    ops["completions 32tok"] = _timed(completions32)

    if ollama:

        def embed():
            st, v = req(
                port,
                "POST",
                "/api/embed",
                {"model": model, "input": "hello", "keep_alive": 0},
                timeout=60,
            )
            return st == 200 and isinstance(v, dict) and v.get("embeddings"), st, ""
    else:

        def embed():
            st, v = req(
                port,
                "POST",
                "/v1/embeddings",
                {"model": model, "input": "hello"},
                timeout=60,
            )
            return st == 200, st, ""

    ops["embeddings"] = _timed(embed, n=2)
    if ollama:
        # capability probes used keep_alive:0 bodies — warm the model back
        # in so op timings are not contaminated by a reload.
        req(
            port,
            "POST",
            "/api/chat",
            {
                "model": model,
                "stream": False,
                "keep_alive": "10m",
                "messages": [{"role": "user", "content": "hi"}],
                "options": {"num_predict": 1},
            },
            timeout=180,
        )

    if not ollama:

        def tok():
            st, v = req(
                port,
                "POST",
                "/tokenize",
                {"model": model, "content": "hello world"},
                timeout=30,
            )
            return st == 200, st, ""

        ops["tokenize"] = _timed(tok, n=3)

        def detok():
            st, v = req(
                port,
                "POST",
                "/detokenize",
                {"model": model, "tokens": [9707, 1519]},
                timeout=30,
            )
            return st == 200, st, ""

        ops["detokenize"] = _timed(detok, n=3)

    def _tool_result(st: int, calls: list) -> tuple[bool, int, str]:
        """Args may arrive as a JSON string OR a pre-parsed object."""
        ok = st == 200 and bool(calls)
        if not calls:
            return ok, st, ("no-call" if st == 200 else "")
        args = (calls[0].get("function") or {}).get("arguments")
        if isinstance(args, str):
            try:
                json.loads(args)
                return ok, st, "args-valid"
            except Exception:
                return ok, st, "args-INVALID"
        return ok, st, "args-valid" if isinstance(args, dict) else "args-INVALID"

    # TOOL_SPEC/TOOL_PROMPT are module-level (shared with reasoning_matrix)
    tool_spec = TOOL_SPEC
    tool_prompt = TOOL_PROMPT

    if ollama:

        def tools():
            # 256-token budget: reasoning models burn budget in thinking
            # before emitting the call (content='' at small budgets).
            st, v = req(
                port,
                "POST",
                "/api/chat",
                {
                    "model": model,
                    "stream": False,
                    "keep_alive": "10m",
                    "messages": [{"role": "user", "content": tool_prompt}],
                    "tools": [tool_spec],
                    "options": {"num_predict": 256},
                },
                timeout=180,
            )
            msg = (v.get("message") or {}) if isinstance(v, dict) else {}
            calls = msg.get("tool_calls") or []
            return _tool_result(st, calls)
    else:

        def tools():
            st, v = req(
                port,
                "POST",
                "/v1/chat/completions",
                {
                    "model": model,
                    "max_tokens": 256,
                    "tools": [tool_spec],
                    "messages": [{"role": "user", "content": tool_prompt}],
                },
                timeout=180,
            )
            msg = (v.get("choices") or [{}])[0].get("message") or {}
            calls = msg.get("tool_calls") or []
            return _tool_result(st, calls)

    ops["tool call round-trip"] = _timed(tools, n=2)

    person_schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
        "additionalProperties": False,
    }

    if ollama:

        def jschema():
            st, v = req(
                port,
                "POST",
                "/api/chat",
                {
                    "model": model,
                    "stream": False,
                    "keep_alive": "10m",
                    "format": "json",
                    "think": False,  # reasoning models: thinking eats the budget (qwen: enable_thinking)
                    "messages": [
                        {"role": "user", "content": "Return JSON: name Alice, age 30."}
                    ],
                    "options": {"num_predict": 256},
                },
                timeout=120,
            )
            try:
                obj = json.loads((v.get("message") or {}).get("content", ""))
                return st == 200 and obj.get("name") == "Alice", st, "parsed"
            except Exception:
                return False, st, "unparseable"
    else:

        def jschema():
            st, v = req(
                port,
                "POST",
                "/v1/chat/completions",
                {
                    "model": model,
                    "max_tokens": 256,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "response_format": {
                        "type": "json_schema",
                        "strict": True,
                        "json_schema": {"name": "person", "schema": person_schema},
                    },
                    "messages": [
                        {"role": "user", "content": "Return JSON: name Alice, age 30."}
                    ],
                },
                timeout=120,
            )
            try:
                obj = json.loads(
                    (v.get("choices") or [{}])[0].get("message", {}).get("content", "")
                )
                return st == 200 and obj.get("name") == "Alice", st, "parsed"
            except Exception:
                return False, st, "unparseable"

    ops["structured output (schema)"] = _timed(jschema, n=2)

    if vision:
        b64 = png_b64()
        if ollama:

            def vis():
                st, v = req(
                    port,
                    "POST",
                    "/api/chat",
                    {
                        "model": model,
                        "stream": False,
                        "keep_alive": "10m",
                        "messages": [
                            {
                                "role": "user",
                                "content": "One word: what color?",
                                "images": [b64],
                            }
                        ],
                        "options": {"num_predict": 16},
                    },
                    timeout=180,
                )
                return st == 200, st, ""
        else:

            def vis():
                st, v = req(
                    port,
                    "POST",
                    "/v1/chat/completions",
                    {
                        "model": model,
                        "max_tokens": 16,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "One word: what color?"},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/png;base64,{b64}"
                                        },
                                    },
                                ],
                            }
                        ],
                    },
                    timeout=180,
                )
                return st == 200, st, ""

        ops["vision round-trip"] = _timed(vis, n=1)

    if not ollama:

        def messages():
            st, v = req(
                port,
                "POST",
                "/v1/messages",
                {
                    "model": model,
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": "hi"}],
                },
                timeout=120,
            )
            return st == 200, st, ""

        ops["anthropic /v1/messages"] = _timed(messages)

        def responses():
            st, v = req(
                port,
                "POST",
                "/v1/responses",
                {"model": model, "max_output_tokens": 16, "input": "hi"},
                timeout=120,
            )
            return st == 200, st, ""

        ops["responses API"] = _timed(responses)

    return ops


def queue_probe(port: int, model: str, ollama: bool = False) -> dict:
    """Fire a long stream, then a second small one: queue vs reject."""
    holder: dict = {}

    def long_stream():
        if ollama:
            r = ollama_stream_timed(port, model, DECODE_PROMPT, 256, num_ctx=None)
        else:
            r = openai_stream_timed(port, model, DECODE_PROMPT, 256)
        holder["long"] = metric_from(r)

    t = threading.Thread(target=long_stream, daemon=True)
    t.start()
    time.sleep(1.0)
    if ollama:
        small = metric_from(
            ollama_stream_timed(port, model, "Say ok.", 8, num_ctx=None)
        )
    else:
        small = metric_from(openai_stream_timed(port, model, "Say ok.", 8))
    t.join(timeout=120)
    return {
        "second_status": small["status"],
        "second_ttft_ms": small["ttft_ms"],
        "queued_not_rejected": small["status"] == 200,
    }


def native_timings(ctx: dict, daemon) -> dict:
    """Pallama-native architecture surfaces, each timed."""
    sb, model = ctx["sandbox"], ctx["model"]
    out: dict[str, dict] = {}

    def timed_api(label, method, path, body=None, n=1, ok_codes=(200,)):
        def call():
            st, _ = req(PAL_PORT, method, path, body, timeout=120)
            return st in ok_codes, st, ""

        out[label] = _timed(call, n)

    # sessions: save -> restore -> erase (slot KV checkpoint architecture).
    # Canonical contract: POST /api/session {"model", "action", "filename"}.
    timed_api(
        "session save",
        "POST",
        "/api/session",
        {"model": model, "action": "save", "filename": "bench-probe"},
    )
    timed_api(
        "session restore",
        "POST",
        "/api/session",
        {"model": model, "action": "restore", "filename": "bench-probe"},
    )
    timed_api(
        "session erase",
        "POST",
        "/api/session",
        {"model": model, "action": "erase", "filename": "bench-probe"},
    )

    # sentinel: why list + a trace lookup (needs a fresh trace id)
    timed_api("why (list)", "GET", "/api/why")
    r = urllib.request.Request(
        f"http://127.0.0.1:{PAL_PORT}/v1/chat/completions",
        data=json.dumps(
            {
                "model": model,
                "max_tokens": 8,
                "messages": [{"role": "user", "content": "Say ok."}],
            }
        ).encode(),
        method="POST",
    )
    r.add_header("content-type", "application/json")
    try:
        with urllib.request.urlopen(r, timeout=120) as resp:
            trace = resp.headers.get("x-pallama-trace-id", "")
    except Exception:
        trace = ""
    if trace:
        time.sleep(0.2)  # record commit is async
        timed_api("why (trace lookup)", "GET", f"/api/why?trace={trace}")

    # watch SSE: time to first event while chats complete
    watch: dict = {}

    def watch_stream():
        def on_obj(obj, want_first):
            if want_first:
                return "first"

        rr = _recv_stream(
            PAL_PORT, b"", "/api/watch", 90, on_obj, method="GET", stop_on_first=True
        )
        watch["ttft_ms"] = rr["ttft_ms"]
        watch["status"] = rr["status"]
        watch["err"] = rr["error"]

    wt = threading.Thread(target=watch_stream, daemon=True)
    wt.start()
    time.sleep(1.0)
    # Two triggers: records commit at request end; a second chat guards
    # against a first-response edge (empty reasoning stream etc.).
    openai_stream_timed(PAL_PORT, model, "Say ok.", 16)
    if not watch.get("ttft_ms"):
        time.sleep(1.0)
        openai_stream_timed(PAL_PORT, model, "Say ok again.", 16)
    wt.join(timeout=45)
    out["watch SSE first event"] = {
        "ms": watch.get("ttft_ms"),
        "ok": watch.get("ttft_ms") is not None,
        "status": watch.get("status"),
        "note": watch.get("err", ""),
    }

    # per-request num_ctx restart-once (X-Pallama-Num-Ctx)
    rr = openai_stream_timed(
        PAL_PORT, model, "Say ok.", 8, headers={"X-Pallama-Num-Ctx": "4096"}
    )
    out["num-ctx header restart (4096)"] = {
        "ms": rr["ttft_ms"],
        "ok": rr["status"] == 200,
        "status": rr["status"],
        "note": "restart-once + first tok",
    }
    # restore ctx
    r2 = urllib.request.Request(
        f"http://127.0.0.1:{PAL_PORT}/v1/chat/completions",
        data=json.dumps(
            {
                "model": model,
                "max_tokens": 8,
                "messages": [{"role": "user", "content": "Say ok."}],
            }
        ).encode(),
        method="POST",
    )
    r2.add_header("content-type", "application/json")
    r2.add_header("X-Pallama-Num-Ctx", str(ctx.get("pallama_ctx") or 16384))
    try:
        urllib.request.urlopen(r2, timeout=120).read()
    except Exception:
        pass

    # prefix-cache hit (cache_reuse): same long prompt twice
    a = metric_from(openai_stream_timed(PAL_PORT, model, PP_PROMPT, 1))
    b = metric_from(openai_stream_timed(PAL_PORT, model, PP_PROMPT, 1))
    out["prefix-cache 2nd-pass TTFT"] = {
        "ms": b["ttft_ms"],
        "ok": b["ok"],
        "status": b["status"],
        "note": f"1st={a['ttft_ms'] and round(a['ttft_ms'])}ms -> 2nd={b['ttft_ms'] and round(b['ttft_ms'])}ms",
    }

    # evict + full reload cycle
    t0 = time.perf_counter()
    st, _ = req(PAL_PORT, "POST", "/api/evict", {"model": model}, timeout=60)
    out["evict"] = {
        "ms": (time.perf_counter() - t0) * 1000,
        "ok": st == 200,
        "status": st,
        "note": "",
    }
    rr = openai_stream_timed(PAL_PORT, model, "Say ok.", 8)
    out["reload after evict (req→1st tok)"] = {
        "ms": rr["ttft_ms"],
        "ok": rr["status"] == 200,
        "status": rr["status"],
        "note": "",
    }

    # poller gauges (informational): cache-hit + spec-accept EWMA after the
    # traffic above; published only once a 60s poller tick saw traffic.
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{PAL_PORT}/metrics", timeout=30
        ) as resp:
            mtext = resp.read().decode(errors="replace")

        def gauge(gname):
            for ln in mtext.splitlines():
                if ln.startswith(gname + " "):
                    return ln.split()[-1]
            return None

        out["poller gauges"] = {
            "ms": None,
            "ok": True,
            "status": 200,
            "note": (
                f"prefix_cache_hit_rate={gauge('pallama_prefix_cache_hit_rate')} "
                f"spec_accept_rate={gauge('pallama_spec_accept_rate')} "
                "(blank = no poller tick with traffic yet)"
            ),
        }
    except Exception as e:
        out["poller gauges"] = {"ms": None, "ok": False, "status": 0, "note": str(e)}

    # singleflight dedup (informational): two IDENTICAL concurrent non-stream
    # chats vs one — coalesced execution should cost ~1x wall, not 2x.
    def _one_chat():
        t = time.perf_counter()
        req(
            PAL_PORT,
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "max_tokens": 24,
                "stream": False,
                "messages": [{"role": "user", "content": "Count from 1 to 8."}],
            },
            timeout=120,
        )
        return time.perf_counter() - t

    single = _one_chat()
    t0 = time.perf_counter()
    _ths = [threading.Thread(target=_one_chat) for _ in range(2)]
    for _t in _ths:
        _t.start()
    for _t in _ths:
        _t.join(timeout=130)
    pair = time.perf_counter() - t0
    out["singleflight pair-vs-single"] = {
        "ms": pair * 1000,
        "ok": True,
        "status": 200,
        "note": (
            f"single={single * 1000:.0f}ms pair={pair * 1000:.0f}ms "
            f"ratio={(pair / single):.2f}x (singleflight serializes the twin behind the leader, then re-serves from warm prefix cache — ~2x is two decodes; far above 2x would mean queue pathology)"
        ),
    }

    # CLI timings (daemon paths)
    for cmd in ("list", "ps", f"show {model}", "config list"):

        def cli_call(cmd=cmd):
            t = time.perf_counter()
            p = subprocess.run(
                [V.PAL, *cmd.split()],
                env=sb.env(),
                capture_output=True,
                text=True,
                timeout=60,
            )
            return (
                p.returncode == 0,
                p.returncode,
                f"{(time.perf_counter() - t) * 1000:.0f}ms rc=0",
            )

        out[f"cli: pallama {cmd}"] = _timed(cli_call, n=3)
    return out


def variant_timings(ctx: dict) -> dict:
    """Config-variant effects: short decode bench per variant vs baseline."""
    sb, model = ctx["sandbox"], ctx["model"]
    base_tps = (ctx.get("baseline_suite") or {}).get("decode_tps_med")
    out: dict[str, dict] = {}

    def short_bench(cfg) -> dict:
        d = V.Daemon(sb)
        d.start({**cfg, "port": PAL_PORT})
        try:

            def stream(prompt, max_tokens):
                return openai_stream_timed(PAL_PORT, model, prompt, max_tokens)

            warm = metric_from(stream(DECODE_PROMPT, 32))
            if not warm["ok"]:
                return {"failed": f"warmup status={warm['status']}"}
            runs = [
                metric_from(stream(DECODE_PROMPT, DECODE_MAX_TOKENS)) for _ in range(2)
            ]
            tps = statistics.median(
                [r["decode_tps"] for r in runs if r.get("decode_tps")]
            )
            ttft = statistics.median([r["ttft_ms"] for r in runs if r.get("ttft_ms")])
            pid = find_child_pid(sb.data_dir, model, timeout=60)
            argv = " ".join(read_argv(pid))
            return {
                "decode_tps": tps,
                "ttft_ms": ttft,
                "argv": argv,
                "tps_delta_pct": ((tps / base_tps - 1) * 100)
                if tps and base_tps
                else None,
            }
        finally:
            pid = find_child_pid(sb.data_dir, model, timeout=1)
            d.stop()
            wait_pid_gone(pid, timeout=30)
            wait_gpu_settled(ctx["gpu_idle_mib"])

    for name, cfg in (
        ("sentinel=false", {"sentinel": False}),
        ("spec=ngram", {"spec": "ngram"}),
        ("cache_type=q8_0", {"cache_type": "q8_0"}),
    ):
        log(f"\n  variant {name} ...")
        try:
            out[name] = {"cfg": cfg, **short_bench(cfg)}
        except Exception as e:
            out[name] = {"cfg": cfg, "failed": str(e)}

    # router mode: one child serves the whole store via a generated preset INI
    log("\n  variant router=true ...")
    try:
        d = V.Daemon(sb)
        d.start({"router": True, "port": PAL_PORT})
        try:
            cold = metric_from(openai_stream_timed(PAL_PORT, model, DECODE_PROMPT, 32))
            runs = [
                metric_from(
                    openai_stream_timed(
                        PAL_PORT, model, DECODE_PROMPT, DECODE_MAX_TOKENS
                    )
                )
                for _ in range(2)
            ]
            tps = statistics.median(
                [r["decode_tps"] for r in runs if r.get("decode_tps")]
            )
            t0 = time.perf_counter()
            st, _ = req(PAL_PORT, "POST", "/api/evict", {"model": model}, timeout=60)
            evict_ms = (time.perf_counter() - t0) * 1000
            ps = req(PAL_PORT, "GET", "/api/ps")[1]
            out["router=true"] = {
                "cold_first_chat_s": (cold["ttft_ms"] or 0) / 1000
                if cold["ok"]
                else None,
                "decode_tps": tps,
                "tps_delta_pct": ((tps / base_tps - 1) * 100)
                if tps and base_tps
                else None,
                "evict_ms": evict_ms,
                "evict_status": st,
                "ps_rows": len(
                    (ps.get("models") or []) if isinstance(ps, dict) else []
                ),
            }
        finally:
            d.stop()
            wait_gpu_settled(ctx["gpu_idle_mib"])
    except Exception as e:
        out["router=true"] = {"failed": str(e)}
    return out


def reasoning_matrix(port: int, model: str) -> dict:
    """Reasoning economics: the think switch ON vs OFF, measured.

    - chat stream: TTFT-to-first-THOUGHT vs TTFT-to-ANSWER vs no-think
      TTFT; reasoning/answer char split; wall time both modes
    - tool call round-trip: latency + arg validity with thinking ON/OFF
    - strict json_schema: the reasoning trap (budget eaten by thinking,
      empty content — the sentinel `reasoning_no_answer` case) vs the
      clean think:false parse
    Skipped with a reason when the model emits no reasoning channel.
    """
    probe = openai_stream_timed(
        port, model, "Answer with one short sentence: what is 2+2?", 96
    )
    if probe.get("ttft_reasoning_ms") is None and probe.get("reasoning_chars", 0) == 0:
        return {"skipped": "model emits no reasoning_content (not a reasoning model)"}

    prompt = "In one short paragraph: is 13 prime? Explain briefly."
    on_runs = [openai_stream_timed(port, model, prompt, 192) for _ in range(2)]
    off_runs = [
        openai_stream_timed(port, model, prompt, 192, think=False) for _ in range(2)
    ]

    def agg(runs, key):
        xs = [r.get(key) for r in runs if r.get(key) is not None]
        return statistics.median(xs) if xs else None

    out = {
        "chat_on": {
            "ttft_reasoning_ms": agg(on_runs, "ttft_reasoning_ms"),
            "ttft_answer_ms": agg(on_runs, "ttft_content_ms"),
            "reasoning_chars": agg(on_runs, "reasoning_chars"),
            "answer_chars": agg(on_runs, "content_chars"),
            "completion_tokens": agg(
                [
                    {
                        "t": r["usage"].get("completion_tokens")
                        if r.get("usage")
                        else None
                    }
                    for r in on_runs
                ],
                "t",
            ),
            "total_ms": agg(on_runs, "total_ms"),
        },
        "chat_off": {
            "ttft_answer_ms": agg(off_runs, "ttft_content_ms"),
            "answer_chars": agg(off_runs, "content_chars"),
            "total_ms": agg(off_runs, "total_ms"),
        },
    }

    def tool_run(think: bool | None):
        body: dict = {
            "model": model,
            "max_tokens": 384,
            "tools": [TOOL_SPEC],
            "messages": [{"role": "user", "content": TOOL_PROMPT}],
        }
        if think is False:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        st, v = req(port, "POST", "/v1/chat/completions", body, timeout=240)
        calls = (
            ((v.get("choices") or [{}])[0].get("message") or {}).get("tool_calls") or []
            if isinstance(v, dict)
            else []
        )
        valid = bool(calls) and isinstance(
            (calls[0].get("function") or {}).get("arguments"), (str, dict)
        )
        return st == 200 and valid

    t0 = time.perf_counter()
    tool_on_ok = tool_run(None)
    tool_on_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    tool_off_ok = tool_run(False)
    tool_off_ms = (time.perf_counter() - t0) * 1000
    out["tools_on"] = {"ms": tool_on_ms, "ok": tool_on_ok}
    out["tools_off"] = {"ms": tool_off_ms, "ok": tool_off_ok}

    person_schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
        "additionalProperties": False,
    }

    def schema_run(think: bool | None, max_tokens: int):
        body: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "response_format": {
                "type": "json_schema",
                "strict": True,
                "json_schema": {"name": "person", "schema": person_schema},
            },
            "messages": [
                {"role": "user", "content": "Return JSON: name Alice, age 30."}
            ],
        }
        if think is False:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        st, v = req(port, "POST", "/v1/chat/completions", body, timeout=240)
        try:
            obj = json.loads(
                (v.get("choices") or [{}])[0].get("message", {}).get("content", "")
            )
            return st == 200 and obj.get("name") == "Alice"
        except Exception:
            return False

    out["schema_on"] = {"parsed": schema_run(None, 512)}
    out["schema_off"] = {"parsed": schema_run(False, 256)}
    return out


def phase_features(ctx: dict) -> dict:
    """Every feature, every architecture surface, timed — sequentially."""
    log("\n" + "=" * 78)
    log("FEATURE & ARCHITECTURE MATRIX (sequential; one engine resident at a time)")
    log("=" * 78)
    sb, model = ctx["sandbox"], ctx["model"]
    feat: dict = {
        "capability": {},
        "ops": {},
        "native": {},
        "queue": {},
        "variants": {},
    }
    vision = bool(
        sqlite3.connect(os.path.join(sb.data_dir, "pallama.db"))
        .execute("select mmproj_path is not null from models where name=?", (model,))
        .fetchone()[0]
    )

    # --- pallama segment (daemon + model loaded)
    release_ollama_resident(ctx)
    mem_guard(ctx["model_mib"], "features/pallama")
    d = V.Daemon(sb)
    ctx["daemon"] = d
    d.start({"port": PAL_PORT})
    warm = metric_from(openai_stream_timed(PAL_PORT, model, "Say ok.", 8))
    if not warm["ok"]:
        d.stop()
        raise RuntimeError(f"features: pallama warm request failed {warm}")
    if not ctx.get("pallama_argv"):
        pid = find_child_pid(sb.data_dir, model, timeout=60)
        ctx["pallama_argv"] = read_argv(pid)
    log("\n  probing pallama routes + timing features ...")
    feat["capability"]["pallama"] = probe_routes(PAL_PORT, model)
    feat["ops"]["pallama"] = time_feature_ops(PAL_PORT, model, vision=vision)
    log("  timing pallama-native surfaces ...")
    feat["native"] = native_timings(ctx, d)
    log("  reasoning matrix (think on/off economics) ...")
    feat["reasoning"] = reasoning_matrix(PAL_PORT, model)
    log("  queue probe (slots=1, second request must queue not reject) ...")
    feat["queue"]["pallama"] = queue_probe(PAL_PORT, model)
    d.stop()
    wait_gpu_settled(ctx["gpu_idle_mib"])

    # --- direct segment
    try:
        with DirectServer(ctx) as srv:
            log("\n  probing direct llama-server routes + timing features ...")
            feat["capability"]["direct"] = probe_routes(DIRECT_PORT, model)
            feat["ops"]["direct"] = time_feature_ops(DIRECT_PORT, model, vision=vision)
            feat["queue"]["direct"] = queue_probe(DIRECT_PORT, model)
    except Exception as e:
        feat["capability"]["direct"] = {}
        feat["ops"]["direct"] = {}
        feat["queue"]["direct"] = {"error": str(e)}
    wait_gpu_settled(ctx["gpu_idle_mib"])

    # --- ollama segment
    om = resolve_ollama_model(ctx)
    if om:
        try:
            mem_guard(ctx["model_mib"], "features/ollama")
            log("\n  probing ollama routes + timing features ...")
            feat["capability"]["ollama"] = probe_routes(OLLAMA_PORT, om, ollama=True)
            feat["ops"]["ollama"] = time_feature_ops(
                OLLAMA_PORT, om, ollama=True, vision=vision
            )
            feat["queue"]["ollama"] = queue_probe(OLLAMA_PORT, om, ollama=True)
            ollama_unload(om)
        except Exception as e:
            feat["queue"]["ollama"] = {"error": str(e)}
        wait_gpu_settled(ctx["gpu_idle_mib"])
    else:
        log("\n  ollama: no matching model; feature segment skipped")

    # --- config variants
    feat["variants"] = variant_timings(ctx)
    return feat


def report_features(feat: dict) -> None:
    log("\n" + "=" * 78)
    log(
        "CAPABILITY MATRIX (ok = served · route = exists, probe body rejected · — = missing)"
    )
    log("=" * 78)
    caps = feat.get("capability") or {}
    labels: list[str] = []
    for c in caps.values():
        for k in c:
            if k not in labels:
                labels.append(k)
    hdr = f"  {'endpoint':<32}{'pallama':>10}{'direct':>10}{'ollama':>10}"
    log(hdr)
    log("  " + "-" * 62)
    for label in labels:
        cells = "".join(
            f"{(caps.get(name) or {}).get(label, '·'):>10}"
            for name in ("pallama", "direct", "ollama")
        )
        log(f"  {label:<32}{cells}")

    log("\n" + "=" * 78)
    log("FEATURE LATENCY (p50 ms; ok=op succeeded, note carries validity)")
    log("=" * 78)
    ops = feat.get("ops") or {}
    op_labels: list[str] = []
    for o in ops.values():
        for k in o:
            if k not in op_labels:
                op_labels.append(k)
    log(f"  {'feature':<32}{'pallama':>12}{'direct':>12}{'ollama':>12}")
    log("  " + "-" * 68)
    for label in op_labels:
        cells = []
        for name in ("pallama", "direct", "ollama"):
            op = (ops.get(name) or {}).get(label)
            if op is None:
                cells.append(f"{'·':>12}")
            elif not op.get("ok"):
                # 501 = route served, capability disabled for this model
                # (e.g. embeddings without --embeddings): a capability fact,
                # not a contender failure.
                cells.append(f"{'n/a' if op.get('status') == 501 else 'FAIL':>12}")
            else:
                cells.append(f"{op['ms']:>11.0f} ")
        log(f"  {label:<32}{''.join(cells)}")
    # notes under the table (validity of tools/json)
    for name in ("pallama", "direct", "ollama"):
        for label, op in (ops.get(name) or {}).items():
            if op and not op.get("ok"):
                kind = "n/a" if op.get("status") == 501 else "FAIL"
                why = (
                    (
                        " 501 = server cannot serve this capability for this "
                        "model (e.g. embeddings need --embeddings)"
                    )
                    if op.get("status") == 501
                    else ""
                )
                log(
                    f"    [{name}] {label}: {kind} status={op.get('status')}{why} {op.get('note', '')}"
                )
            elif op and op.get("note"):
                log(f"    [{name}] {label}: {op['note']}")

    log("\n" + "=" * 78)
    log("PALLAMA-NATIVE ARCHITECTURE SURFACES (timed)")
    log("=" * 78)
    for label, v in (feat.get("native") or {}).items():
        ms = v.get("ms")
        ok = "ok " if v.get("ok") else "FAIL"
        ms_s = f"{ms:>9.1f} ms" if isinstance(ms, (int, float)) else f"{'—':>11}"
        log(f"  [{ok}] {label:<40}{ms_s}  {v.get('note', '')}")

    rm = feat.get("reasoning") or {}
    if rm.get("skipped"):
        log(f"\n  reasoning matrix: {rm['skipped']}")
    elif rm:
        log("\n" + "=" * 78)
        log("REASONING MATRIX (think switch ON vs OFF — economics)")
        log("=" * 78)
        on, off = rm.get("chat_on") or {}, rm.get("chat_off") or {}
        log(f"  TTFT first thought      {fmt(on.get('ttft_reasoning_ms'), ' ms', 0)}")
        log(f"  TTFT first ANSWER (on)  {fmt(on.get('ttft_answer_ms'), ' ms', 0)}")
        log(f"  TTFT first token (off)  {fmt(off.get('ttft_answer_ms'), ' ms', 0)}")
        log(
            f"  reasoning chars (on)    {fmt(on.get('reasoning_chars'), '', 0)}   answer chars (on) {fmt(on.get('answer_chars'), '', 0)}"
        )
        log(
            f"  wall total on/off       {fmt(on.get('total_ms'), ' ms', 0)} / {fmt(off.get('total_ms'), ' ms', 0)}"
        )
        t_on, t_off = rm.get("tools_on") or {}, rm.get("tools_off") or {}
        log(
            f"  tool call on/off        {fmt(t_on.get('ms'), ' ms', 0)} ({'valid' if t_on.get('ok') else 'INVALID'})"
            f"  /  {fmt(t_off.get('ms'), ' ms', 0)} ({'valid' if t_off.get('ok') else 'INVALID'})"
        )
        s_on, s_off = rm.get("schema_on") or {}, rm.get("schema_off") or {}
        log(
            f"  strict schema on/off    {'parsed' if s_on.get('parsed') else 'TRAP: thinking ate the budget (sentinel: reasoning_no_answer)'}"
            f" / {'parsed' if s_off.get('parsed') else 'unparseable'}"
        )
        if on.get("ttft_answer_ms") and off.get("ttft_answer_ms"):
            speedup = (on["ttft_answer_ms"] / off["ttft_answer_ms"] - 1) * 100
            log(
                f"  VERDICT: think:false answers {speedup:+.0f}% faster to first token on this workload"
            )

    log("\n" + "=" * 78)
    log("QUEUE BEHAVIOR (slots=1: long stream + concurrent small request)")
    log("=" * 78)
    for name, q in (feat.get("queue") or {}).items():
        if "error" in q:
            log(f"  {name:<10} error: {q['error']}")
        else:
            verdict = (
                "QUEUED (200)"
                if q.get("queued_not_rejected")
                else f"REJECTED ({q.get('second_status')})"
            )
            ttft = q.get("second_ttft_ms")
            ttft_s = f"{ttft:.0f} ms" if isinstance(ttft, (int, float)) else "—"
            log(f"  {name:<10} second request: {verdict:<16} ttft={ttft_s}")

    log("\n" + "=" * 78)
    log("CONFIG-VARIANT EFFECTS (short decode bench vs default-config baseline)")
    log("=" * 78)
    for name, v in (feat.get("variants") or {}).items():
        if v.get("failed"):
            log(f"  {name:<22} FAILED: {v['failed']}")
            continue
        tps = v.get("decode_tps")
        delta = v.get("tps_delta_pct")
        ttft = v.get("ttft_ms")
        line = f"  {name:<22} decode {fmt(tps)} t/s"
        if delta is not None:
            line += f"  ({delta:+.1f}% vs default)"
        if ttft:
            line += f"  ttft {ttft:.0f}ms"
        log(line)
        if name == "router=true":
            log(
                f"  {'':22} cold first chat {fmt(v.get('cold_first_chat_s'), ' s', 2)}"
                f"  evict {fmt(v.get('evict_ms'), 'ms', 0)} (status {v.get('evict_status')})"
                f"  ps rows {v.get('ps_rows')}"
            )


# ------------------------------------------------------------------ report


def fmt(v, unit="", digits=1):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:,.{digits}f}{unit}"
    return f"{v}{unit}"


def report(ctx: dict, results: dict) -> int:
    pal, direct, oll, ceil = (
        results.get(k) or {} for k in ("pallama", "direct", "ollama", "ceiling")
    )
    log("\n" + "=" * 78)
    log("FAIRNESS TABLE (what each contender actually ran)")
    log("=" * 78)
    if pal.get("engine_argv"):
        log(f"  pallama engine : {' '.join(pal['engine_argv'])}")
    if direct.get("engine_argv"):
        log(f"  direct  engine : {' '.join(direct['engine_argv'])}")
        stripped = sorted(
            {
                a
                for a in (ctx.get("pallama_argv") or [])
                if a not in direct["engine_argv"]
            }
        )
        log(f"  argv delta     : {stripped}  (must be host/port/slot-save-path only)")
    if oll:
        note = " (CTX MISMATCH — see notes)" if oll.get("ctx_mismatch") else ""
        log(
            f"  ollama         : model={oll.get('model', '?')} "
            f"num_ctx={oll.get('num_ctx') or 'default'}{note} "
            f"pallama_ctx={ctx.get('pallama_ctx')} engine={oll.get('engine_exe') or 'ollama (exe unreadable)'}"
        )
    if ceil:
        log(
            f"  llama-bench    : pp512={fmt(ceil.get('pp512_tps'))} tg128={fmt(ceil.get('tg128_tps'))} t/s "
            f"(matched threads/ctx, full offload)"
        )

    log("\n" + "=" * 78)
    log("BENCHMARK (warm, median of runs; pp = unique-prompt true prefill)")
    log("=" * 78)
    cols = [
        ("pallama", pal),
        ("direct llama", direct),
        ("ollama", oll),
        ("llama-bench", ceil),
    ]

    def sget(r, key):
        return (r.get("suite") or {}).get(key)

    def rng(r):
        lo, hi = sget(r, "decode_tps_min"), sget(r, "decode_tps_max")
        if lo is None:
            return None
        return f"{lo:.1f}–{hi:.1f}"

    def dec(r):
        v = sget(r, "decode_tps_med")
        return v if v is not None else r.get("tg128_tps")

    def ppv(r):
        v = sget(r, "pp_tps_med")
        return v if v is not None else r.get("pp512_tps")

    rows = [
        ("cold req→1st tok (s)", lambda r: r.get("cold_request_to_first_tok_s"), 2),
        ("cold spawn→ready (s)", lambda r: r.get("cold_spawn_to_ready_s"), 2),
        ("TTFT warm p50 (ms)", lambda r: sget(r, "ttft_ms_med"), 0),
        ("decode t/s (median)", dec, 1),
        ("decode t/s (min–max)", rng, 1),
        ("pp t/s (unique prompt)", ppv, 0),
        ("wall t/s", lambda r: sget(r, "wall_tps_med"), 1),
        ("peak child RSS (MiB)", lambda r: r.get("rss_peak_mib"), 0),
        ("peak GPU delta (MiB)", lambda r: r.get("gpu_delta_mib"), 0),
    ]
    log(
        f"  {'metric':<24}{'pallama':>15}{'direct llama':>15}{'ollama':>15}{'llama-bench':>15}"
    )
    log("  " + "-" * 84)
    for label, extract, digits in rows:
        cells = []
        for _, r in cols:
            if r.get("skipped"):
                cells.append("skip")
            elif r.get("failed"):
                cells.append("FAIL")
            else:
                cells.append(fmt(extract(r), "", digits))
        log(f"  {label:<24}" + "".join(f"{c:>15}" for c in cells))

    log("")
    if pal.get("suite"):
        compat = pal.get("compat_decode_tps")
        base = pal["suite"].get("decode_tps_med")
        tax = ((1 - compat / base) * 100) if compat and base else None
        log(
            f"  pallama /api/chat compat decode t/s: {fmt(compat)}  "
            f"(translation tax vs its own /v1: {fmt(tax, '%')})"
        )
    log("\n" + "=" * 78)
    log("VERDICTS")
    log("=" * 78)
    rc = 0
    for name, r in (("pallama", pal), ("direct", direct), ("ollama", oll)):
        if r.get("failed"):
            log(f"  {name} HARD-FAILED: {r['failed']}")
            if name in ("pallama", "direct"):
                rc = 1  # the head-to-head is unmeasurable without both
    pt = (pal.get("suite") or {}).get("decode_tps_med")
    dt = (direct.get("suite") or {}).get("decode_tps_med")
    ot = (oll.get("suite") or {}).get("decode_tps_med")
    if pt and dt:
        tax = (1 - pt / dt) * 100
        verdict = "OK" if tax <= 5.0 else "FAIL"
        if tax > 5.0:
            rc = 1
        log(
            f"  orchestration tax (pallama vs direct, decode t/s): {tax:+.1f}%  [{verdict}]"
        )
    if pt and ot:
        d = (pt / ot - 1) * 100
        pal_exe = ctx.get("pallama_argv") and ctx["pallama_argv"][0]
        oll_exe = oll.get("engine_exe")
        same_engine = bool(
            pal_exe
            and oll_exe
            and pal_exe in oll_exe
            or pal_exe
            and oll_exe
            and oll_exe in pal_exe
        )
        if same_engine:
            log(f"  pallama vs ollama (decode t/s): {d:+.1f}%")
        else:
            log(
                f"  pallama vs ollama (decode t/s): {d:+.1f}%  [BACKEND-BIASED: "
                f"pallama={pal_exe} vs ollama={oll_exe} — different engine "
                f"binaries; the delta is engine/backend-structural, NOT "
                f"orchestration. Orchestration verdict is the pallama-vs-direct "
                f"row (same binary).]"
            )
    if pt and (ceil or {}).get("tg128_tps"):
        c = pt / ceil["tg128_tps"] * 100
        log(f"  pallama vs engine ceiling (tg128): {c:.0f}% of llama-bench")
    if oll.get("ctx_mismatch"):
        log(
            "  NOTE: ollama could not run at pallama's ctx — its row used its "
            "default; KV/memory footprint differs"
        )
    for name, r in (
        ("pallama", pal),
        ("direct", direct),
        ("ollama", oll),
        ("ceiling", ceil),
    ):
        if r.get("skipped"):
            log(f"  skipped {name}: {r['skipped']}")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=os.environ.get("PALLAMA_BENCH_MODEL", V.MODEL))
    ap.add_argument("--runs", type=int, default=None)
    ap.add_argument(
        "--fast", action="store_true", help="0.5b model, 1 run, no feature phase"
    )
    ap.add_argument(
        "--features",
        dest="features",
        action="store_true",
        default=None,
        help="run the feature/architecture matrix (default: on unless --fast)",
    )
    ap.add_argument("--no-features", dest="features", action="store_false")
    ap.add_argument(
        "--skip", default="", help="csv subset of pallama,direct,ollama,ceiling"
    )
    ap.add_argument(
        "--out", default=None, help="artifact dir (default ~/.cache/pallama-bench/<ts>)"
    )
    ap.add_argument(
        "--ollama-model",
        default=None,
        help="explicit ollama model name (default: auto-match from /api/tags)",
    )
    args = ap.parse_args()
    fast = args.fast or os.environ.get("PALLAMA_BENCH_FAST", "") == "1"
    model = "qwen2.5-0.5b-instruct" if fast and args.model == V.MODEL else args.model
    runs = args.runs or (1 if fast else 3)
    want_features = (not fast) if args.features is None else args.features
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    for p in (PAL_PORT, DIRECT_PORT):
        if p == 11434:
            log(f"REFUSING: configured port {p} collides with ollama's 11434")
            return 2

    ts = time.strftime("%Y%m%d-%H%M%S")
    outdir = args.out or os.path.expanduser(f"~/.cache/pallama-bench/{ts}")
    os.makedirs(outdir, exist_ok=True)

    # Sandbox FIRST (live-safe sqlite copy of the real store), then read
    # model facts from the copy — never from the live DB.
    sandbox = V.Sandbox()
    db = sqlite3.connect(os.path.join(sandbox.data_dir, "pallama.db"))
    row = db.execute(
        "select path, bytes, mmproj_path from models where name=?", (model,)
    ).fetchone()
    db.close()
    if not row:
        log(f"model {model!r} not in store; pallama list shows:")
        subprocess.run([V.PAL, "list"], env=sandbox.env())
        sandbox.destroy()
        return 2
    gguf_path, bytes_, mmproj = row
    engine_row = (
        sqlite3.connect(os.path.join(sandbox.data_dir, "pallama.db"))
        .execute("select tag from engines where active=1")
        .fetchone()
    )
    engine_tag = engine_row[0] if engine_row else "?"

    gpu_idle = gpu_used_mib()
    ctx = {
        "model": model,
        "runs": runs,
        "sandbox": sandbox,
        "outdir": outdir,
        "gguf_path": gguf_path,
        "model_mib": bytes_ / 2**20,
        "ollama_model": args.ollama_model,
        "gpu_idle_mib": gpu_idle,
        "engine_tag": engine_tag,
    }
    log(f"pallama comparison benchmark — model={model} runs={runs} fast={fast}")
    log(
        f"gguf={gguf_path} ({bytes_ / 2**20:,.0f} MiB, mmproj={'yes' if mmproj else 'no'})"
    )
    log(f"engine={engine_tag} gpu_idle={gpu_idle} MiB artifacts={outdir}")
    results: dict[str, dict] = {}
    failed_env = None
    try:
        order = [
            ("pallama", bench_pallama),
            ("direct", bench_direct),
            ("ollama", bench_ollama),
            ("ceiling", bench_ceiling),
        ]
        for name, fn in order:
            if name in skip:
                results[name] = {"skipped": "--skip"}
                continue
            try:
                results[name] = fn(ctx)
            except EnvironmentError as e:
                log(f"  ! environment abort during {name}: {e}")
                results[name] = {"skipped": f"low memory: {e}"}
                failed_env = failed_env or str(e)
            except Exception as e:
                log(f"  ! {name} contender FAILED: {e}")
                results[name] = {"failed": str(e)}
        if want_features:
            ctx["baseline_suite"] = (results.get("pallama") or {}).get("suite")
            try:
                results["features"] = phase_features(ctx)
            except EnvironmentError as e:
                log(f"  ! environment abort during features: {e}")
                results["features"] = {"skipped": f"low memory: {e}"}
                failed_env = failed_env or str(e)
            except Exception as e:
                log(f"  ! feature phase FAILED: {e}")
                results["features"] = {"failed": str(e)}
    finally:
        # ollama is never signaled (API-only unload already attempted in its
        # bench fn); the daemon and sandbox are ours to tear down.
        if ctx.get("daemon"):
            try:
                ctx["daemon"].stop()
            except Exception:
                pass
        sandbox.destroy()

    rc = report(ctx, results)
    if (
        isinstance(results.get("features"), dict)
        and "skipped" not in results["features"]
        and "failed" not in results["features"]
    ):
        report_features(results["features"])
    with open(os.path.join(outdir, "results.json"), "w") as f:
        json.dump(
            {
                "ctx": {k: v for k, v in ctx.items() if k not in ("sandbox", "daemon")},
                "results": results,
            },
            f,
            indent=2,
            default=str,
        )
    log(
        f"\nartifacts: {outdir}/results.json (+ llama-bench.json, direct-llama-server.log)"
    )
    if failed_env:
        return 2
    return rc


if __name__ == "__main__":
    sys.exit(main())
