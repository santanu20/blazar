#!/usr/bin/env python3
"""Professional cross-engine, cross-provider benchmark matrix for Pallama.

Sweeps every installed engine (llama.cpp builds AND mistral.rs) across
server providers (direct child spawn, pallama gateway, ollama reference),
measuring:

  speed      TTFT / decode t/s / prefill t/s per cell (+ llama-bench
             ceiling for engines that ship it)
  resources  peak RSS, peak GPU memory, teardown-verified VRAM return
  serving    greedy-parity text quality vs the llama.cpp-direct
             reference (raw /v1/completions, sampler-pinned) and
             llama-perplexity parity on a fixed corpus
  features   capability matrix (grammar, slots, tokenize, embeddings,
             vision, spec-decode, ...) from --help probes + documented
             constants for engines without introspectable CLIs

Cell model: (engine_tag, provider, params) -> one record in cells.jsonl
(append + resume; a cell key hashes engine/provider/params/model and the
harness version so stale records never masquerade as current).

Exit codes: 0 = all cells ok, 1 = some cells failed (recorded, campaign
continued), 2 = environment abort (no model / no engines / corpus fetch
failure).

Never touches the user's daemon, config, or ollama service: pallama
cells run inside the validate.py Sandbox (hardlinked engines + DB
backup), direct cells spawn our own PIDs on probed free ports, the
ollama cell is HTTP-only against an already-running service.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import sqlite3
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))


HARNESS_VERSION = 1
ARTIFACTS_ROOT = Path.home() / ".cache" / "pallama-bench-matrix"

# Default sweep (C1: fixed, CLI-tunable, no config matrix).
DIRECT_CTX_SWEEP = (4096, 16384)
DIRECT_NP_SWEEP = (1, 4)
DEFAULT_PP = 512
DEFAULT_TG = 128
DEFAULT_RUNS = 3
DEFAULT_NGL = 99
DEFAULT_TIMEOUT = 1800

# Ports (F2: direct lanes probe a free port themselves; these are the
# preferred starting points only).
OLLAMA_PORT = 11434

# Quality lane constants (F4/F5).
PPL_CTX = 2048
PPL_TOKENS_LIMIT = 32768
GREEDY_MAX_TOKENS = 64
GREEDY_SAMPLER = {"temperature": 0, "top_k": 1, "seed": 42}

# 20 raw-completion prompts (no chat markup: quality divergence must not
# be chat-template noise). Plain continuation style, varied domains.
GREEDY_PROMPTS = [
    "The capital city of France is",
    "Water boils at a temperature of",
    "The three primary colors of light are",
    "In 1969, humans first landed on",
    "The chemical formula of table salt is",
    "A square has exactly four",
    "The largest planet in the solar system is",
    "Photosynthesis converts sunlight into",
    "The speed of light in vacuum is approximately",
    "The first President of the United States was",
    "Mount Everest is located in the",
    "The Pacific Ocean is the largest ocean on",
    "Shakespeare wrote Romeo and Juliet during the",
    "The human skeleton has 206",
    "DNA carries the genetic",
    "The Great Wall of China was built to",
    "Honey is produced by",
    "The freezing point of water in Fahrenheit is",
    "A triangle's interior angles sum to",
    "The currency of Japan is the",
]

# Feature matrix: documented constants for engines without an
# introspectable CLI (ollama; source: docs.ollama.com cap pages, and the
# mistral.rs docs serve reference for anything --help misses).
OLLAMA_FEATURES = {
    "grammar-gbnf": False,
    "json-schema": True,
    "slots-sessions": False,
    "tokenize-endpoint": True,
    "embeddings": True,
    "rerank": False,
    "vision-mmproj": True,
    "spec-decode": False,
    "kv-quant": False,
    "lora-adapter": True,
    "quant-on-load": False,
    "paged-attn": False,
    "parallel-np": True,
    "anthropic-api": False,
    "metrics-endpoint": False,
    "ctx-override": True,
}
# llamacpp: feature -> (flag or None-if-documented-constant, kind)
FEATURE_FLAG_MAP_LLAMACPP = {
    "grammar-gbnf": "--grammar",
    "json-schema": None,  # response_format json_schema is served API-side
    "slots-sessions": "--slot-save-path",
    "tokenize-endpoint": None,  # server route, not a flag; constant True
    "embeddings": "--embeddings",
    "rerank": "--rerank",
    "vision-mmproj": "--mmproj",
    "spec-decode": "--model-draft",
    "kv-quant": "--cache-type-k",
    "lora-adapter": "--lora",
    "quant-on-load": None,  # llama-quantize tool ships with builds
    "paged-attn": "--flash-attn",
    "parallel-np": "-np",
    "anthropic-api": None,  # absent by design
    "metrics-endpoint": "--metrics",
    "ctx-override": "--ctx-size",
}
FEATURE_FLAG_MAP_MISTRALRS = {
    "grammar-gbnf": None,  # no GBNF support (docs)
    "json-schema": None,  # response_format only via chat API
    "slots-sessions": None,  # no /slots (HTTP reference)
    "tokenize-endpoint": None,  # no /tokenize (HTTP reference)
    "embeddings": None,  # /v1/embeddings exists (model-type driven)
    "rerank": None,  # absent (HTTP reference)
    "vision-mmproj": "--mmproj",
    "spec-decode": "--mtp",
    "kv-quant": "--pa-cache-type",
    "lora-adapter": "--lora",
    "quant-on-load": "--isq",
    "paged-attn": "--paged-attn",
    "parallel-np": "--max-seqs",
    "anthropic-api": None,  # /v1/messages exists (HTTP reference)
    "metrics-endpoint": "--disable-metrics",
    "ctx-override": "--max-model-len",
}
# Features whose truth is the INVERSE of the flag existing
# (--disable-metrics means metrics exist; json-schema via API not flag).
FEATURE_INVERSE = {"metrics-endpoint"}
FEATURE_CONSTANT = {
    ("llamacpp", "tokenize-endpoint"): True,
    ("llamacpp", "quant-on-load"): True,
    ("llamacpp", "anthropic-api"): False,
    # short flags (-np) are invisible to the long-flag --help regex; the
    # slot count IS llama-server's parallelism surface (documented)
    ("llamacpp", "parallel-np"): True,
    # response_format json_schema is served by the llama-server API
    ("llamacpp", "json-schema"): True,
    ("mistralrs", "grammar-gbnf"): False,
    ("mistralrs", "slots-sessions"): False,
    ("mistralrs", "tokenize-endpoint"): False,
    ("mistralrs", "rerank"): False,
    ("mistralrs", "json-schema"): True,
    ("mistralrs", "anthropic-api"): True,
}


def log(msg: str = "") -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# cell bookkeeping


def cell_key(tag: str, provider: str, params: dict, model: str) -> str:
    blob = json.dumps([tag, provider, params, model, HARNESS_VERSION], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        # errored cells retry on resume — only successes count as done
        if "error" not in rec:
            done.add(rec["key"])
    return done


def append_record(path: Path, record: dict) -> None:
    with path.open("a") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# engine discovery (DB kinds + binaries)


@dataclass
class Engine:
    tag: str
    kind: str  # llamacpp | mistralrs
    dir: Path
    server: Path  # llama-server or mistralrs binary
    bench: Path | None = None
    perplexity: Path | None = None


def load_engines(data_dir: Path) -> list[Engine]:
    db = data_dir / "pallama.db"
    kinds: dict[str, str] = {}
    if db.exists():
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            for tag, kind in con.execute("SELECT tag, kind FROM engines"):
                kinds[tag] = kind or "llamacpp"
        finally:
            con.close()
    out: list[Engine] = []
    engines_root = data_dir / "engines"
    if not engines_root.is_dir():
        return out
    for edir in sorted(engines_root.iterdir()):
        if not edir.is_dir():
            continue
        tag = edir.name
        if tag not in kinds:
            # orphan dir without a store row (install debris): not a
            # real engine — a pallama cell for it would flip NO row
            # active and the sandbox daemon exits "no engine installed"
            continue
        kind = kinds[tag]
        if kind == "mistralrs":
            server = edir / "mistralrs"
            if not server.exists():
                continue
            out.append(Engine(tag, kind, edir, server, None, None))
        else:
            server = edir / "llama-server"
            if not server.exists():
                # release layout nests one level: <tag>/llama-<tag>/
                nested = [p for p in edir.glob("*/llama-server") if p.is_file()]
                if not nested:
                    continue
                edir = nested[0].parent
                server = nested[0]
            bench = edir / "llama-bench"
            ppl = edir / "llama-perplexity"
            out.append(
                Engine(
                    tag,
                    kind,
                    edir,
                    server,
                    bench if bench.exists() else None,
                    ppl if ppl.exists() else None,
                )
            )
    return out


# ---------------------------------------------------------------------------
# process + measurement plumbing


def free_port(preferred: int | None = None) -> int:
    if preferred is not None:
        with socket.socket() as s:
            s.bind(("127.0.0.1", preferred))
            return preferred
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Sampler(threading.Thread):
    """Peak RSS + GPU memory sampler for one PID (bench_compare pattern)."""

    def __init__(self, pid: int, interval: float = 0.4):
        super().__init__(daemon=True)
        self.pid = pid
        self.interval = interval
        self.stop_evt = threading.Event()
        self.rss_peak_mib = 0.0
        self.gpu_peak_mib = 0.0
        self.gpu_base_mib = -1.0
        self._tick = 0

    def _rss(self) -> float:
        try:
            txt = Path(f"/proc/{self.pid}/status").read_text()
            m = re.search(r"VmRSS:\s+(\d+) kB", txt)
            return float(m.group(1)) / 1024 if m else 0.0
        except OSError:
            return 0.0

    def _gpu(self) -> float:
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            vals = [float(x) for x in out.stdout.strip().splitlines() if x]
            return vals[0] if vals else 0.0
        except (OSError, ValueError, subprocess.TimeoutExpired):
            return 0.0

    def run(self) -> None:
        while not self.stop_evt.is_set():
            self.rss_peak_mib = max(self.rss_peak_mib, self._rss())
            if self._tick % 3 == 0:
                g = self._gpu()
                if self.gpu_base_mib < 0:
                    self.gpu_base_mib = g
                self.gpu_peak_mib = max(self.gpu_peak_mib, g)
            self._tick += 1
            self.stop_evt.wait(self.interval)


def gpu_used_mib() -> float:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        vals = [float(x) for x in out.stdout.strip().splitlines() if x]
        return vals[0] if vals else 0.0
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return 0.0


def mem_guard(floor_mib: float, what: str) -> bool:
    g = gpu_used_mib()
    if g > floor_mib:
        log(
            f"  ! mem_guard: {what}: GPU {g:.0f} MiB > floor {floor_mib:.0f} MiB — waiting"
        )
        for _ in range(30):
            time.sleep(1.0)
            g = gpu_used_mib()
            if g <= floor_mib:
                return True
        return False
    return True


def http_json(url: str, payload: dict | None = None, timeout: float = 30.0):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"null")


def openai_stream_timed(port: int, body: dict, timeout: float = 300.0) -> dict:
    """POST /v1/chat/completions (stream) -> ttft/decode/wall metrics."""
    payload = dict(body)
    payload["stream"] = True
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    ttft = None
    tokens = 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            chunk = line[5:].strip()
            if chunk == "[DONE]":
                break
            try:
                j = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            choices = j.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            # thinking models (qwen3.5...) stream reasoning_content while
            # content stays empty — count BOTH as tokens: throughput is
            # token-speed regardless of which field carries them
            if delta.get("content") or delta.get("reasoning_content"):
                tokens += 1
                if ttft is None:
                    ttft = time.perf_counter() - t0
    total = time.perf_counter() - t0
    return {
        "ttft_ms": (ttft or total) * 1000,
        "decode_tps": (tokens - 1) / (total - ttft) if ttft and tokens > 1 else 0.0,
        "wall_tps": tokens / total if total > 0 else 0.0,
        "tokens": tokens,
    }


def ollama_stream_timed(port: int, body: dict, timeout: float = 300.0) -> dict:
    """POST /api/chat (stream) with the same metric extraction."""
    payload = dict(body)
    payload["stream"] = True
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    ttft = None
    tokens = 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            try:
                j = json.loads(line)
            except json.JSONDecodeError:
                continue
            if j.get("done"):
                break
            msg = j.get("message") or {}
            if msg.get("content"):
                tokens += 1
                if ttft is None:
                    ttft = time.perf_counter() - t0
    total = time.perf_counter() - t0
    return {
        "ttft_ms": (ttft or total) * 1000,
        "decode_tps": (tokens - 1) / (total - ttft) if ttft and tokens > 1 else 0.0,
        "wall_tps": tokens / total if total > 0 else 0.0,
        "tokens": tokens,
    }


def median_run_suite(
    port: int, model: str, runs: int, pp: int, tg: int, ollama: bool = False
) -> dict:
    """Warmup + N decode + N prefill runs -> metric medians."""
    fn = ollama_stream_timed if ollama else openai_stream_timed

    def body(max_tokens: int, prompt: str) -> dict:
        if ollama:
            return {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "options": {"num_ctx": 8192},
            }
        return {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }

    fn(port, body(tg, "warmup — reply with one word"), timeout=300.0)
    decode = []
    prefill = []
    for _ in range(runs):
        decode.append(
            fn(port, body(tg, "List fun facts about the ocean, one per line."))
        )
    para = " ".join(
        f"Paragraph {i}: summarize global maritime history." for i in range(24)
    )
    for _ in range(runs):
        prefill.append(fn(port, body(4, para)))  # max_tokens=4 -> prefill-dominated
    return {
        "ttft_ms_p50": statistics.median(x["ttft_ms"] for x in decode),
        "decode_tps_p50": statistics.median(x["decode_tps"] for x in decode),
        "prefill_tps_p50": statistics.median(x["wall_tps"] for x in prefill),
        "runs": runs,
    }


# ---------------------------------------------------------------------------
# direct provider (own PID, free port, teardown-verified)


def stage_mistralrs_view(
    model_path: Path, mmproj: Path | None, stage_root: Path
) -> Path:
    """F3: mistral.rs scans the model's whole directory for projectors —
    a flat shared models dir poisons every text model. Stage a private
    view (symlinks) containing ONLY this model's files, exactly like the
    Pallama serving lane does."""
    d = stage_root / re.sub(r"[^A-Za-z0-9_.-]", "_", model_path.stem)[:80]
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    os.symlink(model_path, d / model_path.name)
    if mmproj is not None and mmproj.exists():
        os.symlink(mmproj, d / mmproj.name)
    return d / model_path.name


def wait_healthy(
    kind: str, port: int, timeout: float, proc: subprocess.Popen | None = None
) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        # fail fast: a dead child can never become healthy (a crashed
        # mistral.rs cell otherwise burns the full 600s timeout)
        if proc is not None and proc.poll() is not None:
            return False
        try:
            if kind == "mistralrs":
                j = http_json(f"http://127.0.0.1:{port}/v1/models", timeout=5.0)
                for m in j.get("data", []):
                    if m.get("status") in (None, "loaded"):
                        return True
            else:
                j = http_json(f"http://127.0.0.1:{port}/health", timeout=5.0)
                if j.get("status") == "ok":
                    return True
        except (urllib.error.URLError, json.JSONDecodeError, OSError):
            pass
        time.sleep(0.5)
    return False


def teardown_proc(proc: subprocess.Popen, sampler: Sampler) -> dict:
    sampler.stop_evt.set()
    sampler.join(timeout=2.0)
    base = gpu_used_mib()
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
        proc.wait(timeout=10)
    ok = True
    for _ in range(15):
        if gpu_used_mib() <= base + 512:
            break
        time.sleep(1.0)
    else:
        ok = False
    return {"teardown_ok": ok}


def direct_argv(
    eng: Engine,
    model: Path,
    mmproj: Path | None,
    port: int,
    ctx: int,
    np_: int,
    ngl: int,
    staged: Path | None,
) -> list[str]:
    if eng.kind == "mistralrs":
        argv = [
            str(eng.server),
            "serve",
            "-f",
            str(staged or model),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--no-ui",
            "--max-model-len",
            str(ctx),
            "--max-seqs",
            str(np_),
        ]
        if staged is not None and mmproj is not None and mmproj.exists():
            # staged view holds only this model's projector — pass it
            # explicitly so discovery cannot pick anything else
            argv += ["--mmproj", str(staged.parent / mmproj.name)]
        return argv
    return [
        str(eng.server),
        "-m",
        str(model),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "-c",
        str(ctx),
        "-np",
        str(np_),
        "-ngl",
        str(ngl),
        "--jinja",
    ]


def run_direct_cell(
    eng: Engine,
    model: Path,
    mmproj: Path | None,
    params: dict,
    model_name: str,
    cfg: dict,
    stage_root: Path,
) -> dict:
    port = free_port()
    staged = None
    if eng.kind == "mistralrs":
        staged = stage_mistralrs_view(model, mmproj, stage_root)
    argv = direct_argv(
        eng,
        model,
        mmproj,
        port,
        params["ctx"],
        params["np"],
        cfg["ngl"],
        staged,
    )
    log(f"  spawn: {' '.join(argv)}")
    errlog = (
        stage_root.parent
        / "cells-stderr"
        / (f"{eng.tag}-ctx{params.get('ctx', 0)}-np{params.get('np', 0)}.log")
    )
    errlog.parent.mkdir(parents=True, exist_ok=True)
    with open(errlog, "wb") as errfh:
        proc = subprocess.Popen(
            argv,
            cwd=str(eng.dir),
            stdout=subprocess.DEVNULL,
            stderr=errfh,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    # errfh closed in the parent: the child inherited the fd
    sampler = Sampler(proc.pid)
    sampler.start()
    rec: dict = {"argv": argv, "stderr_log": str(errlog)}
    try:
        if not wait_healthy(eng.kind, port, 600.0, proc=proc):
            tail = ""
            try:
                tail = errlog.read_text(errors="replace")[-400:].replace("\n", " | ")
            except OSError:
                pass
            rec["error"] = (
                f"child exited rc={proc.poll()} during load"
                if proc.poll() is not None
                else "child failed to become healthy within 600s"
            )
            if tail:
                rec["error"] += f"; last output: {tail}"
            return rec
        # mistral.rs children register the served model as "default";
        # the pallama gateway rewrites at the proxy — we do it here.
        body_model = "default" if eng.kind == "mistralrs" else model_name
        rec.update(
            median_run_suite(port, body_model, cfg["runs"], cfg["pp"], cfg["tg"])
        )
        rec["rss_peak_mib"] = round(sampler.rss_peak_mib, 1)
        rec["gpu_peak_mib"] = round(sampler.gpu_peak_mib, 1)
        rec["gpu_base_mib"] = round(sampler.gpu_base_mib, 1)
    finally:
        rec.update(teardown_proc(proc, sampler))
    return rec


# ---------------------------------------------------------------------------
# pallama provider (validate.py Sandbox; per-engine active flip)


def run_pallama_cell(
    eng: Engine, model_name: str, cfg: dict, skip_ollama_note: str
) -> dict:
    # per-campaign unique port BEFORE the lazy import: validate.py reads
    # PALLAMA_VALIDATE_PORT once at import time — a shared fixed port is
    # exactly how orphaned sandbox daemons hijacked campaigns (leak class
    # fixed in validate.py; this makes collisions structurally impossible)
    os.environ["PALLAMA_VALIDATE_PORT"] = str(free_port())
    import validate as V

    rec: dict = {}
    sb = V.Sandbox()
    try:
        con = sqlite3.connect(Path(sb.data_home) / "pallama" / "pallama.db")
        con.execute("UPDATE engines SET active = (tag = ?)", (eng.tag,))
        con.commit()
        con.close()
        daemon = V.Daemon(sb)
        daemon.start(floor_model=model_name)
        port: int | None = None
        try:
            port = V.PORT
            deadline = time.time() + 600
            healthy = False
            while time.time() < deadline:
                try:
                    http_json(f"http://127.0.0.1:{port}/healthz", timeout=5.0)
                    healthy = True
                    break
                except json.JSONDecodeError:
                    # /healthz answers 2xx with a non-JSON body — up is up
                    healthy = True
                    break
                except (urllib.error.URLError, OSError):
                    time.sleep(0.5)
            if not healthy:
                return {"error": "sandbox daemon failed to boot"}
            assert port is not None
            # child spawns on first request; openai_stream_timed drives it
            rec.update(
                median_run_suite(port, model_name, cfg["runs"], cfg["pp"], cfg["tg"])
            )
            rec["provider_note"] = skip_ollama_note
        finally:
            daemon.stop()
            # forensic tail: the sandbox is destroyed below — keep the
            # last daemon lines so failed cells can be diagnosed from
            # cells.jsonl alone
            dlog = Path(sb.data_dir) / "run" / "daemon.log"
            if dlog.exists():
                rec["daemon_log_tail"] = "\n".join(
                    dlog.read_text(errors="replace").splitlines()[-12:]
                )
            # teardown verify: the port must go dark, else the daemon
            # outlived its cell — record loudly, never silently
            dark = False
            if port is not None:
                for _ in range(20):
                    try:
                        urllib.request.urlopen(
                            f"http://127.0.0.1:{port}/healthz", timeout=1.0
                        )
                        time.sleep(0.5)
                    except (urllib.error.URLError, OSError):
                        dark = True
                        break
            rec["teardown_ok"] = dark
            if not dark and port is not None:
                rec["teardown_warn"] = f"sandbox daemon still on :{port} after stop()"
    finally:
        sb.destroy()
    return rec


# ---------------------------------------------------------------------------
# ollama reference (HTTP-only, one cell)


def run_ollama_cell(cfg: dict, args_model: str | None = None) -> dict:
    try:
        tags = http_json(f"http://127.0.0.1:{OLLAMA_PORT}/api/tags", timeout=5.0)
    except (urllib.error.URLError, OSError):
        return {"error": "ollama not reachable on 11434 (skipped, not started)"}
    models = [m["name"] for m in tags.get("models", [])]
    if not models:
        return {"error": "ollama reachable but no models pulled"}
    # SAME-model reference first: substring match against the matrix
    # model (e.g. "Qwen3.5" -> "qwen3.5:9b"). Junk-heuristic only when
    # nothing matches — a same-family registry default is the honest
    # reference; an arbitrary small model is not.
    pick = None
    if args_model:
        needle = args_model.lower()
        hits = [m for m in models if needle in m.lower()]
        if hits:
            pick = min(hits, key=len)
    if pick is None:
        pick = min(models, key=lambda n: (0 if "0.5b" in n or "0.6b" in n else 1, n))
    out = {
        "ollama_model": pick,
        **median_run_suite(
            OLLAMA_PORT, pick, cfg["runs"], cfg["pp"], cfg["tg"], ollama=True
        ),
    }
    # the reference row sits in the same table as the matrix model —
    # name it loudly when it differs or the t/s columns mislead
    if args_model and args_model.lower() not in pick.lower():
        out["reference_note"] = (
            f"reference serves '{pick}' (matrix model differs — t/s NOT comparable)"
        )
    return out


# ---------------------------------------------------------------------------
# quality lane


def build_local_corpus(dest: Path, cap_bytes: int = 1_500_000) -> bool:
    """Deterministic offline ppl corpus: sorted repo text files, capped.
    Parity only needs the SAME text for every engine; a fixed local
    source removes the network flake entirely (R5)."""
    repo = Path(__file__).resolve().parent.parent
    parts: list[str] = []
    taken = 0
    files = sorted(
        p
        for p in repo.rglob("*")
        if p.suffix in {".rs", ".toml", ".md", ".py"}
        and "target" not in p.parts
        and p.stat().st_size < 200_000
    )
    for p in files:
        if taken >= cap_bytes:
            break
        try:
            # ASCII-only: a repo corpus can carry codepoints that abort
            # some tokenizers ("invalid codepoint" SIGABRT, proven live
            # on b10809 + qwen3.5) — parity needs identical text, not
            # exotic glyphs
            parts.append(
                p.read_text(errors="replace").encode("ascii", "ignore").decode()
            )
            taken += p.stat().st_size
        except OSError:
            continue
    if taken < 64_000:  # < ~16k tokens: ctx 2048 needs 2x ctx tokens
        return False
    dest.write_text("\n".join(parts))
    return True


def run_perplexity(eng: Engine, model: Path, corpus: Path, cfg: dict) -> dict:
    if eng.perplexity is None:
        return {"error": "engine ships no llama-perplexity (mistral.rs: unsupported)"}
    # F5: pinned identical args across engines; recorded in the artifact.
    # -f data file is REQUIRED by llama-perplexity (0 tokens => exit 1,
    # proven live on b10809); ctx 2048 demands >= 4096 corpus tokens.
    argv = [
        str(eng.perplexity),
        "-m",
        str(model),
        "-f",
        str(corpus),
        "--ctx-size",
        str(PPL_CTX),
        "--gpu-layers",
        str(cfg["ngl"]),
        "--chunks",
        "4",
    ]
    t0 = time.time()
    p = subprocess.run(
        argv,
        cwd=str(eng.dir),
        capture_output=True,
        text=True,
        timeout=cfg["timeout"],
        check=False,
    )
    out = p.stdout + p.stderr
    # b10809 shape (live-verified): "Final estimate: PPL = 22.9764 +/- 1.91745"
    m = re.findall(r"PPL = ([0-9.]+)(?: \+/- ([0-9.]+))?", out)
    vals = [a for a, _ in m if a]
    if not vals:
        return {
            "error": f"no PPL line in llama-perplexity output (exit {p.returncode})",
            "argv": argv,
        }
    return {
        "perplexity": float(vals[-1]),
        "ppl_error": float(m[-1][1]) if m[-1][1] else None,
        "argv": argv,
        "wall_s": round(time.time() - t0, 1),
    }


def greedy_completions(port: int, prompt: str, model: str) -> str:
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": GREEDY_MAX_TOKENS,
        "stream": False,
        **GREEDY_SAMPLER,
    }
    try:
        j = http_json(f"http://127.0.0.1:{port}/v1/completions", body, timeout=300.0)
    except urllib.error.HTTPError:
        # strict OpenAI implementations may reject top_k/seed in the
        # body — retry with temperature-only greed (still deterministic)
        body = {
            "model": model,
            "prompt": prompt,
            "max_tokens": GREEDY_MAX_TOKENS,
            "stream": False,
            "temperature": 0,
        }
        j = http_json(f"http://127.0.0.1:{port}/v1/completions", body, timeout=300.0)
    return (j.get("choices") or [{}])[0].get("text", "")


def run_greedy_parity(
    eng: Engine,
    model: Path,
    mmproj: Path | None,
    model_name: str,
    reference: dict[str, str],
    stage_root: Path,
) -> dict:
    """Spawn engine directly, sampler-pinned RAW completions, diff vs the
    llama.cpp-direct reference texts."""
    port = free_port()
    staged = (
        stage_mistralrs_view(model, mmproj, stage_root)
        if eng.kind == "mistralrs"
        else None
    )
    argv = direct_argv(eng, model, mmproj, port, 4096, 1, DEFAULT_NGL, staged)
    proc = subprocess.Popen(
        argv,
        cwd=str(eng.dir),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    sampler = Sampler(proc.pid)
    sampler.start()
    try:
        if not wait_healthy(eng.kind, port, 600.0, proc=proc):
            return {"error": "child failed to become healthy"}
        body_model = "default" if eng.kind == "mistralrs" else model_name
        ratios = []
        exact = 0
        first_div = []
        for i, prompt in enumerate(GREEDY_PROMPTS):
            got = greedy_completions(port, prompt, body_model)
            ref = reference.get(prompt, "")
            if got == ref:
                exact += 1
            ratios.append(difflib.SequenceMatcher(None, ref, got).ratio())
            fd = next(
                (k for k, (a, b) in enumerate(zip(ref, got)) if a != b),
                min(len(ref), len(got)),
            )
            first_div.append(fd)
        return {
            "exact_matches": exact,
            "prompts": len(GREEDY_PROMPTS),
            "ratio_mean": round(statistics.mean(ratios), 4),
            "ratio_min": round(min(ratios), 4),
            "first_divergence_median_chars": statistics.median(first_div),
        }
    finally:
        teardown_proc(proc, sampler)


def build_greedy_reference(
    eng: Engine, model: Path, mmproj: Path | None, model_name: str, stage_root: Path
) -> dict[str, str] | None:
    port = free_port()
    staged = (
        stage_mistralrs_view(model, mmproj, stage_root)
        if eng.kind == "mistralrs"
        else None
    )
    argv = direct_argv(eng, model, mmproj, port, 4096, 1, DEFAULT_NGL, staged)
    proc = subprocess.Popen(
        argv,
        cwd=str(eng.dir),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    sampler = Sampler(proc.pid)
    sampler.start()
    try:
        if not wait_healthy(eng.kind, port, 600.0, proc=proc):
            return None
        out = {}
        for prompt in GREEDY_PROMPTS:
            out[prompt] = greedy_completions(port, prompt, model_name)
        return out
    finally:
        teardown_proc(proc, sampler)


# ---------------------------------------------------------------------------
# features lane


def cli_flags(binary: Path, sub: list[str]) -> set[str]:
    try:
        p = subprocess.run(
            [str(binary), *sub],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(binary.parent),
            check=False,
        )
        return {
            tok.strip()
            for tok in re.findall(r"--[a-z0-9][a-z0-9-]*", p.stdout + p.stderr)
        }
    except (OSError, subprocess.TimeoutExpired):
        return set()


def features_row(kind: str, flags: set[str]) -> dict[str, bool]:
    fmap = (
        FEATURE_FLAG_MAP_LLAMACPP if kind == "llamacpp" else FEATURE_FLAG_MAP_MISTRALRS
    )
    row = {}
    for feat, flag in fmap.items():
        const = FEATURE_CONSTANT.get((kind, feat))
        if const is not None:
            row[feat] = const
        elif flag is None:
            row[feat] = False
        elif feat in FEATURE_INVERSE:
            row[feat] = flag in flags
        else:
            row[feat] = flag in flags
    return row


# ---------------------------------------------------------------------------
# reporting


def fmt(v, suffix=""):
    return "-" if v is None else f"{v}{suffix}"


def gpu_name() -> str:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        line = out.stdout.strip().splitlines()
        return line[0].replace(", ", " / driver ") if line else "n/a"
    except (OSError, subprocess.TimeoutExpired):
        return "n/a"


def write_markdown_report(
    path: Path,
    records: list[dict],
    model: Path,
    engines: list,
    feat_rows: dict[str, dict[str, bool]] | None,
    argv_summary: str,
) -> None:
    """Human-first markdown report: environment, speed, quality, features."""
    md: list[str] = []
    md.append("# Pallama benchmark matrix\n")
    md.append(f"- **date**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    md.append(f"- **model**: `{model.name}` ({model.stat().st_size // (1 << 20)} MiB)")
    md.append(f"- **gpu**: {gpu_name()}")
    md.append("- **engines**: " + ", ".join(f"{e.tag} ({e.kind})" for e in engines))
    md.append(f"- **harness**: bench_matrix v{HARNESS_VERSION} — `{argv_summary}`\n")

    speed = [r for r in records if "ttft_ms_p50" in r]
    if speed:
        md.append("## Speed (serving, streaming)\n")
        md.append(
            "| engine | kind | provider | params | ttft p50 (ms) | decode t/s | prefill t/s | GPU peak (MiB) |"
        )
        md.append("|---|---|---|---|---|---|---|---|")
        for r in speed:
            pa = (
                " ".join(f"{kk}={vv}" for kk, vv in (r.get("params") or {}).items())
                or "-"
            )
            if r.get("reference_note"):
                # honesty: only flagged when the reference serves a
                # DIFFERENT model than the matrix (same-model rows are
                # directly comparable)
                pa += f" ⚠ serves '{r['ollama_model']}' — t/s NOT comparable"
            md.append(
                "| {t} | {k} | {p} | {pa} | {a} | {b} | {c} | {g} |".format(
                    t=r.get("tag", "-"),
                    k=r.get("kind", "-"),
                    p=r.get("provider", "-"),
                    pa=pa,
                    a=fmt(
                        round(r["ttft_ms_p50"])
                        if r.get("ttft_ms_p50") is not None
                        else None
                    ),
                    b=fmt(
                        round(r["decode_tps_p50"], 1)
                        if r.get("decode_tps_p50") is not None
                        else None
                    ),
                    c=fmt(
                        round(r["prefill_tps_p50"], 1)
                        if r.get("prefill_tps_p50") is not None
                        else None
                    ),
                    g=fmt(round(r.get("gpu_peak_mib", 0))),
                )
            )
        md.append("")

    ppl = [r for r in records if r.get("provider") == "ppl"]
    if ppl:
        md.append("## Quality — perplexity (identical pinned args)\n")
        md.append("| engine | perplexity | wall (s) | note |")
        md.append("|---|---|---|---|")
        for r in ppl:
            note = r.get("error", "lower = better text fit")
            md.append(
                f"| {r.get('tag', '-')} | {fmt(r.get('perplexity'))} |"
                f" {fmt(r.get('wall_s'))} | {note} |"
            )
        md.append("")

    greedy = [r for r in records if r.get("provider") == "greedy"]
    if greedy:
        md.append("## Quality — greedy parity vs llama.cpp-direct reference\n")
        md.append(
            "| engine | exact matches | ratio mean | ratio min | first divergence (median chars) |"
        )
        md.append("|---|---|---|---|---|")
        for r in greedy:
            md.append(
                f"| {r.get('tag', '-')} | {r.get('exact_matches', '-')}/{r.get('prompts', '-')} |"
                f" {fmt(r.get('ratio_mean'))} | {fmt(r.get('ratio_min'))} |"
                f" {fmt(r.get('first_divergence_median_chars'))} |"
            )
        md.append("")

    if feat_rows:
        md.append("## Feature matrix\n")
        cols = list(feat_rows.keys())
        feats = sorted({f for r in feat_rows.values() for f in r})
        md.append("| feature | " + " | ".join(cols) + " |")
        md.append("|---|" + "---|" * len(cols))
        for f in feats:
            md.append(
                f"| `{f}` | "
                + " | ".join("Y" if feat_rows[c].get(f) else "-" for c in cols)
                + " |"
            )
        md.append("")

    failed = [r for r in records if "error" in r]
    if failed:
        md.append("## Failed cells\n")
        for r in failed:
            md.append(
                f"- `{r.get('tag')}` / {r.get('provider')} /"
                f" {r.get('params')}: {r['error']}"
            )
        md.append("")

    md.append("## Reading this report\n")
    md.append("- `direct` = raw child spawn on a probed free port (no gateway).")
    md.append(
        "- `pallama` = full gateway path inside a sandboxed daemon"
        " (profile compiler, routing, auth)."
    )
    md.append("- `ollama` = HTTP-only reference against the host service, one cell.")
    md.append(
        "- GPU peaks are sampled at ~1.2 s cadence; very short bursts may undersample."
    )
    md.append(
        "- Cells append to `cells.jsonl` and resume across reruns (keyed on"
        " engine/provider/params/model/harness-version)."
    )
    path.write_text("\n".join(md) + "\n")


def write_speed_table(records: list[dict], path: Path) -> None:
    cols = [
        "engine",
        "kind",
        "provider",
        "params",
        "ttft_ms",
        "decode_tps",
        "prefill_tps",
        "gpu_peak_mib",
        "rss_peak_mib",
    ]
    lines = [" | ".join(cols)]
    lines.append("-" * 100)
    for r in records:
        params = r.get("params", "")
        lines.append(
            " | ".join(
                [
                    r.get("tag", "-"),
                    r.get("kind", "-"),
                    r.get("provider", "-"),
                    str(params),
                    fmt(
                        round(r["ttft_ms_p50"])
                        if r.get("ttft_ms_p50") is not None
                        else None
                    ),
                    fmt(
                        round(r["decode_tps_p50"], 1)
                        if r.get("decode_tps_p50") is not None
                        else None
                    ),
                    fmt(
                        round(r["prefill_tps_p50"], 1)
                        if r.get("prefill_tps_p50") is not None
                        else None
                    ),
                    fmt(round(r.get("gpu_peak_mib", 0))),
                    fmt(round(r.get("rss_peak_mib", 0))),
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--data-dir", default=str(Path.home() / ".local/share/pallama"))
    ap.add_argument("--model", help="model name substring (default: largest .gguf)")
    ap.add_argument("--engines", nargs="*", help="engine tags (default: all)")
    ap.add_argument(
        "--providers",
        nargs="*",
        default=["direct", "pallama", "ollama"],
        choices=["direct", "pallama", "ollama"],
    )
    ap.add_argument("--pp", type=int, default=DEFAULT_PP)
    ap.add_argument("--tg", type=int, default=DEFAULT_TG)
    ap.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    ap.add_argument("--ngl", type=int, default=DEFAULT_NGL)
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    ap.add_argument("--skip-ppl", action="store_true")
    ap.add_argument("--skip-greedy", action="store_true")
    ap.add_argument("--skip-features", action="store_true")
    ap.add_argument("--corpus", help="local corpus .parquet/.txt for perplexity")
    ap.add_argument(
        "--fresh", action="store_true", help="ignore+replace existing cells.jsonl"
    )
    ap.add_argument("--artifacts-dir", help="override artifacts location")
    ap.add_argument(
        "--md",
        help="also write the markdown report to this path (e.g. BENCHMARK.md)",
    )
    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    models_dir = data_dir / "models"
    ggufs = sorted(
        models_dir.glob("*.gguf"), key=lambda p: p.stat().st_size, reverse=True
    )
    # exclude projectors + scratch files
    ggufs = [
        p
        for p in ggufs
        if "mmproj" not in p.name and not p.name.startswith(("imx-", "r5-"))
    ]
    if not ggufs:
        log("no candidate .gguf models in data dir")
        return 2
    model = ggufs[0]
    if args.model:
        hits = [p for p in ggufs if args.model.lower() in p.name.lower()]
        if not hits:
            log(f"no model matching {args.model!r}")
            return 2
        model = hits[0]
    model_name = model.stem.lower().removesuffix("-q4_k_m").removesuffix("-q4_0")
    # mmproj ownership is a per-model DB column, NOT dir proximity —
    # mistral.rs scans the model dir for projectors, so attaching a
    # stray one would poison a text model (the Bug-C class).
    own_mmproj: Path | None = None
    db = data_dir / "pallama.db"
    if db.exists():
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            for (mp,) in con.execute(
                "SELECT mmproj_path FROM models WHERE path = ?", (str(model),)
            ):
                if mp:
                    own_mmproj = Path(mp)
        finally:
            con.close()

    engines = load_engines(data_dir)
    if args.engines:
        engines = [e for e in engines if e.tag in args.engines]
    if not engines:
        log("no benchable engines discovered")
        return 2

    if args.artifacts_dir:
        art = Path(args.artifacts_dir)
    else:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        art = ARTIFACTS_ROOT / stamp
    art.mkdir(parents=True, exist_ok=True)
    cells_path = art / "cells.jsonl"
    if args.fresh and cells_path.exists():
        cells_path.unlink()
    done = load_done(cells_path)
    stage_root = art / "staging"
    stage_root.mkdir(exist_ok=True)

    cfg = {
        "pp": args.pp,
        "tg": args.tg,
        "runs": args.runs,
        "ngl": args.ngl,
        "timeout": args.timeout,
    }
    log(f"model: {model.name} ({model.stat().st_size // (1 << 20)} MiB)")
    log(f"engines: {', '.join(f'{e.tag}({e.kind})' for e in engines)}")
    log(f"providers: {', '.join(args.providers)}  artifacts: {art}")

    failures = 0
    records: list[dict] = []

    def emit(tag, kind, provider, params, key, rec):
        nonlocal failures
        # R4 guarantee: a cell crash is a recorded failure, never a
        # campaign kill (the b10826 orphan taught this the hard way).
        if "error" not in rec and rec.get("decode_tps_p50") == 0.0:
            rec["error"] = "zero tokens served (model failed to load/serve)"
        rec2 = {
            "key": key,
            "tag": tag,
            "kind": kind,
            "provider": provider,
            "params": params,
            "model": model.name,
            **rec,
        }
        append_record(cells_path, rec2)
        records.append(rec2)
        if "error" in rec:
            failures += 1
            log(f"  CELL FAILED: {rec['error']}")
        else:
            log(
                f"  ok: ttft {rec.get('ttft_ms_p50', 0):.0f}ms "
                f"decode {rec.get('decode_tps_p50', 0):.1f} t/s "
                f"gpu {rec.get('gpu_peak_mib', 0):.0f}MiB"
            )

    # ---- direct provider sweep
    if "direct" in args.providers:
        for eng in engines:
            for ctx in DIRECT_CTX_SWEEP:
                for np_ in DIRECT_NP_SWEEP:
                    params = {"ctx": ctx, "np": np_}
                    key = cell_key(eng.tag, "direct", params, model.name)
                    if key in done:
                        log(f"[direct {eng.tag} {params}] resumed — skipping")
                        continue
                    log(f"[direct {eng.tag} ctx={ctx} np={np_}]")
                    if not mem_guard(2048, f"pre-cell {eng.tag}"):
                        emit(
                            eng.tag,
                            eng.kind,
                            "direct",
                            params,
                            key,
                            {"error": "GPU memory floor exceeded before cell"},
                        )
                        continue
                    try:
                        rec = run_direct_cell(
                            eng, model, own_mmproj, params, model_name, cfg, stage_root
                        )
                    except Exception as exc:  # noqa: BLE001 — campaign continues (R4)
                        rec = {"error": f"direct cell crashed: {exc}"}
                    emit(eng.tag, eng.kind, "direct", params, key, rec)

    # ---- pallama provider (one default-config cell per engine)
    if "pallama" in args.providers:
        for eng in engines:
            params = {"config": "default"}
            key = cell_key(eng.tag, "pallama", params, model.name)
            if key in done:
                log(f"[pallama {eng.tag}] resumed — skipping")
                continue
            log(f"[pallama {eng.tag}] (sandbox, gateway, default profile)")
            try:
                rec = run_pallama_cell(eng, model_name, cfg, "sandboxed gateway cell")
            except Exception as exc:  # noqa: BLE001 — campaign continues (R4)
                rec = {"error": f"pallama cell crashed: {exc}"}
            rec.pop("rss_peak_mib", None)
            emit(eng.tag, eng.kind, "pallama", params, key, rec)

    # ---- ollama reference (one cell)
    if "ollama" in args.providers:
        params = {"reference": True}
        key = cell_key("ollama-host", "ollama", params, model.name)
        if key in done:
            log("[ollama] resumed — skipping")
        else:
            log("[ollama reference]")
            try:
                rec = run_ollama_cell(cfg, args.model)
            except Exception as exc:  # noqa: BLE001 — campaign continues (R4)
                rec = {"error": f"ollama cell crashed: {exc}"}
            emit("ollama-host", "ollama", "ollama", params, key, rec)

    # ---- quality: perplexity parity (llama.cpp engines)
    if not args.skip_ppl:
        corpus_needed = any(e.kind == "llamacpp" for e in engines)
        corpus: Path | None = Path(args.corpus) if args.corpus else None
        if corpus_needed and corpus is None:
            corpus = art / "corpus.txt"
            if not corpus.exists():
                log("building deterministic local ppl corpus (repo text)...")
                if not build_local_corpus(corpus):
                    log("local corpus build failed — aborting (exit 2)")
                    return 2
        for eng in engines:
            params = {"ppl": PPL_CTX}
            key = cell_key(eng.tag, "ppl", params, model.name)
            if key in done:
                continue
            log(f"[perplexity {eng.tag}]")
            if eng.kind != "llamacpp":
                emit(
                    eng.tag,
                    eng.kind,
                    "ppl",
                    params,
                    key,
                    {"error": "llama-perplexity is llama-server-family only"},
                )
                continue
            assert corpus is not None  # corpus_needed fetched or aborted above
            try:
                rec = run_perplexity(eng, model, corpus, cfg)
            except Exception as exc:  # noqa: BLE001 — campaign continues (R4)
                rec = {"error": f"ppl cell crashed: {exc}"}
            emit(eng.tag, eng.kind, "ppl", params, key, rec)

    # ---- quality: greedy parity vs llama.cpp-direct reference
    if not args.skip_greedy:
        ref_eng = next((e for e in engines if e.kind == "llamacpp"), None)
        reference = None
        if ref_eng is not None:
            log(f"[greedy reference from {ref_eng.tag}]")
            reference = build_greedy_reference(
                ref_eng, model, own_mmproj, model_name, stage_root
            )
        if reference is None:
            log("no llamacpp engine for greedy reference — skipping parity")
        else:
            for eng in engines:
                params = {"greedy": True}
                key = cell_key(eng.tag, "greedy", params, model.name)
                if key in done:
                    continue
                log(f"[greedy parity {eng.tag}]")
                try:
                    rec = run_greedy_parity(
                        eng, model, own_mmproj, model_name, reference, stage_root
                    )
                except Exception as exc:  # noqa: BLE001 — campaign continues (R4)
                    rec = {"error": f"greedy cell crashed: {exc}"}
                emit(eng.tag, eng.kind, "greedy", params, key, rec)

    # ---- features matrix
    feat_rows: dict[str, dict[str, bool]] | None = None
    if not args.skip_features:
        log("[features]")
        rows: dict[str, dict[str, bool]] = {}
        for eng in engines:
            if eng.kind == "mistralrs":
                flags = cli_flags(eng.server, ["serve", "--help"])
            else:
                flags = cli_flags(eng.server, ["--help"])
            rows[eng.tag] = features_row(eng.kind, flags)
        rows["ollama(documented)"] = OLLAMA_FEATURES
        feat_rows = rows
        feats = sorted({f for r in rows.values() for f in r})
        lines = ["feature | " + " | ".join(rows.keys())]
        lines.append("-" * 90)
        for f in feats:
            lines.append(
                f + " | " + " | ".join("Y" if rows[t].get(f) else "-" for t in rows)
            )
        (art / "features.txt").write_text("\n".join(lines) + "\n")
        log(f"features -> {art / 'features.txt'}")

    argv_summary = " ".join(sys.argv[1:])
    # the report reflects the FULL campaign (cells.jsonl), not just this
    # process's additions — resumed cells are earlier records
    hist: list[dict] = []
    if cells_path.exists():
        for line in cells_path.read_text().splitlines():
            try:
                hist.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    # LAST record per key wins: a retried cell supersedes its earlier
    # failure (hist order = append order)
    by_key: dict[str, dict] = {}
    for r in hist + records:
        k = r.get("key") or f"_noid_{id(r)}"
        by_key[k] = r
    all_records = list(by_key.values())
    md_path = art / "benchmark.md"
    write_markdown_report(md_path, all_records, model, engines, feat_rows, argv_summary)
    log(f"report  -> {md_path}")
    if args.md:
        write_markdown_report(
            Path(args.md), all_records, model, engines, feat_rows, argv_summary
        )
        log(f"report  -> {args.md}")
    write_speed_table(all_records, art / "summary.txt")
    log(f"summary -> {art / 'summary.txt'}")
    log(f"cells   -> {cells_path} ({len(records)} new, {len(all_records)} total)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
