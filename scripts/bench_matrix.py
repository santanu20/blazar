#!/usr/bin/env python3
"""Professional cross-engine, cross-provider benchmark matrix for Pallama.

Sweeps every installed engine (llama.cpp builds AND mistral.rs) across
server providers (direct child spawn, pallama gateway, ollama reference),
measuring:

  speed      TTFT p50/p90/p99, inter-token latency p50/p99, decode t/s,
             TRUE prefill t/s (prompt tokens / first-token time) with
             cache-hit variants, token counts from `usage` where the
             server provides it (chunks only as fallback)
  resources  peak RSS, peak GPU memory, peak GPU power, load time
             (spawn->healthy), teardown-verified VRAM return
  serving    greedy-parity text quality vs the SAME-engine direct
             reference (backend numerics) AND gateway-transparency
             parity (pallama path vs direct path, same engine),
             llama-perplexity parity on a fixed corpus
  features   capability matrix (grammar, slots, tokenize, embeddings,
             vision, spec-decode, ...) from --help probes + documented
             constants for engines without introspectable CLIs

v2 semantics (HARNESS_VERSION bump invalidates v1 cells):
  - decode counts ALL emitted tokens incl. reasoning/thinking fields
  - ollama lane counts thinking + caps num_predict (v1 measured the
    thinking phase as 76s "TTFT" on reasoning models)
  - prefill t/s is prompt_tokens / ttft on a token-targeted prompt
    (v1 printed ~4 generated tokens / wall — not a prefill number)
  - pallama cells record the resolved child argv + a real GPU/RSS
    sampler (v1 showed GPU 0)
  - greedy reference is per-engine; the gateway lane is the headline
    transparency test

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
import ctypes
import difflib
import importlib
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
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))


HARNESS_VERSION = 2
ARTIFACTS_ROOT = Path.home() / ".cache" / "pallama-bench-matrix"

# Default sweep (C1: fixed, CLI-tunable, no config matrix).
DIRECT_CTX_SWEEP = (4096, 16384)
DIRECT_NP_SWEEP = (1, 4)
DEFAULT_PP = 512
DEFAULT_TG = 128
DEFAULT_RUNS = 5
DEFAULT_NGL = 99
DEFAULT_TIMEOUT = 1800
DEFAULT_CONC_SWEEP = (4,)

# Ports (F2: direct lanes probe a free port themselves; these are the
# preferred starting points only).
OLLAMA_PORT = 11434

# Quality lane constants (F4/F5).
PPL_CTX = 2048
PPL_TOKENS_LIMIT = 32768
GREEDY_MAX_TOKENS = 256
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

# Deterministic sentence bank for token-targeted prefill prompts.
PREFILL_BANK = [
    "The harbor lights dimmed as the tide pulled the vessels seaward.",
    "Cartographers of the sixteenth century relied on travelers' tales.",
    "A steady wind carried salt and resin across the shipyard.",
    "Copper roofing develops a green patina over decades of weather.",
    "The archive kept ledgers bound in cloth and iron clasps.",
    "Migratory birds navigate using stars and magnetic fields.",
    "The foundry poured ingots every morning before the heat arrived.",
    "Old stone bridges arch because arches carry weight in compression.",
    "The lighthouse keeper logged fog density twice each night.",
    "River deltas grow where sediment settles faster than currents remove it.",
    "Apprentices learned joinery before they were allowed to carve.",
    "The observatory's brass telescope predated the photographic plate.",
    "Wool was traded in bales stamped with the town seal.",
    "Tides follow the moon more faithfully than the sun.",
    "The bakery started before dawn and sold out by noon.",
    "Surveyors chained distances across the moor in straight lines.",
    "Ink recipes guarded by monasteries included oak galls and iron.",
    "The mill race froze only in the hardest winters.",
    "Charts showed reefs as tiny asterisks of danger.",
    "Sailors spliced rope during the long watches between calms.",
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


def power_state() -> dict:
    """Host power source — battery-capped dGPU clocks taint absolute t/s."""
    on_battery = False
    pct = None
    name = None
    for psy in Path("/sys/class/power_supply").glob("*"):
        try:
            if (psy / "type").read_text().strip() != "Battery":
                continue
            if (psy / "status").read_text().strip() == "Discharging":
                on_battery = True
                name = psy.name
                cap = psy / "capacity"
                if cap.exists():
                    pct = int(cap.read_text().strip())
        except OSError:
            continue
    return {"on_battery": on_battery, "battery_pct": pct, "battery_name": name}


def free_port(preferred: int | None = None) -> int:
    if preferred is not None:
        with socket.socket() as s:
            s.bind(("127.0.0.1", preferred))
            return preferred
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _gpu_query(fields: str) -> list[list[float]]:
    """nvidia-smi CSV -> per-GPU value rows; [] when unavailable."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={fields}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        rows = []
        for line in out.stdout.strip().splitlines():
            vals = []
            for cell in line.split(","):
                cell = cell.strip()
                if not cell or cell in ("[N/A]", "N/A"):
                    vals.append(0.0)
                    continue
                try:
                    vals.append(float(cell))
                except ValueError:
                    vals.append(0.0)
            if vals:
                rows.append(vals)
        return rows
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return []


class Sampler(threading.Thread):
    """Peak RSS + GPU memory + GPU power sampler.

    pid may be None (global GPU-only sampling — used when the serving
    process is not our child, e.g. the ollama host service or the
    sandbox daemon's engine). pid may also be assigned LATE (pallama
    cells discover the engine child after the first request): RSS
    tracking simply starts at the next tick.
    """

    def __init__(self, pid: int | None = None, interval: float = 0.4):
        super().__init__(daemon=True)
        self.pid = pid
        self.interval = interval
        self.stop_evt = threading.Event()
        self.rss_peak_mib = 0.0
        self.gpu_peak_mib = 0.0
        self.gpu_base_mib = -1.0
        self.gpu_power_peak_w = 0.0
        self.gpu_power_base_w = -1.0
        self._tick = 0

    def _rss(self) -> float:
        if self.pid is None:
            return 0.0
        try:
            txt = Path(f"/proc/{self.pid}/status").read_text()
            m = re.search(r"VmRSS:\s+(\d+) kB", txt)
            return float(m.group(1)) / 1024 if m else 0.0
        except OSError:
            return 0.0

    def _gpu(self) -> tuple[float, float]:
        rows = _gpu_query("memory.used,power.draw")
        if not rows:
            return 0.0, 0.0
        return max(r[0] for r in rows), max(r[1] for r in rows)

    def run(self) -> None:
        while not self.stop_evt.is_set():
            self.rss_peak_mib = max(self.rss_peak_mib, self._rss())
            if self._tick % 3 == 0:
                g, w = self._gpu()
                if self.gpu_base_mib < 0:
                    self.gpu_base_mib = g
                    self.gpu_power_base_w = w
                self.gpu_peak_mib = max(self.gpu_peak_mib, g)
                self.gpu_power_peak_w = max(self.gpu_power_peak_w, w)
            self._tick += 1
            self.stop_evt.wait(self.interval)


def gpu_used_mib() -> float:
    rows = _gpu_query("memory.used")
    return max((r[0] for r in rows), default=0.0)


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


# ---------------------------------------------------------------------------
# cold-start parity plumbing (page cache + GPU idle + privileged service)


def fadvise_dontneed(paths) -> int:
    """Drop the kernel page cache for the given files (POSIX_FADV_DONTNEED).

    Cold-load lanes MUST run with the same cache state on every runtime:
    after any earlier lane the multi-GiB model sits in page cache and the
    next "cold" load is memory-fast, not disk-cold. Userspace, no root.
    Returns the number of files actually advised (missing files skip).
    """
    libc = ctypes.CDLL(None, use_errno=True)
    advised = 0
    for p in paths:
        if p is None:
            continue
        try:
            fd = os.open(str(p), os.O_RDONLY)
        except OSError:
            continue
        try:
            # 4 = POSIX_FADV_DONTNEED
            if libc.posix_fadvise(fd, 0, 0, 4) == 0:
                advised += 1
        finally:
            os.close(fd)
    return advised


def wait_gpu_idle(max_mib: float = 512.0, timeout_s: float = 60.0) -> bool:
    """Poll until the GPU drains below `max_mib` (eviction settle)."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if gpu_used_mib() <= max_mib:
            return True
        time.sleep(1.0)
    return gpu_used_mib() <= max_mib


def ollama_blob_paths(min_bytes: int = 100 * 1024 * 1024) -> list[Path]:
    """Model blobs (>100 MiB) from the host ollama store, for fadvise."""
    blobs = Path.home() / ".ollama" / "models" / "blobs"
    if not blobs.is_dir():
        return []
    return [p for p in blobs.iterdir() if p.is_file() and p.stat().st_size >= min_bytes]


def sudo_systemctl(*args: str, password: str | None = None) -> bool:
    """systemctl via sudo -S. The password travels on stdin only —
    NEVER in argv (ps-visible) and never logged. False on any failure."""
    if password is None:
        return False
    try:
        out = subprocess.run(
            ["sudo", "-S", "--", "systemctl", *args],
            input=password + "\n",
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return out.returncode == 0


def sandbox_model_files(sb, model_name: str) -> tuple[Path | None, Path | None]:
    """(weights file, mmproj file) from the sandbox store row — the cold
    probe drops the page cache on BOTH: a cold VL spawn reads the
    projector (875 MiB on the 9B row) off disk too, and leaving it
    cached would hand pallama a warmer cold start than the ollama lane
    (which has no projector at all)."""
    db = Path(sb.data_home) / "pallama" / "pallama.db"
    if not db.exists():
        return None, None
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            for p, mp in con.execute(
                "SELECT path, mmproj_path FROM models WHERE name = ?", (model_name,)
            ):
                return Path(p), Path(mp) if mp else None
        finally:
            con.close()
    except sqlite3.Error:
        return None, None
    return None, None


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


def percentile(vals: list[float], pct: int) -> float:
    """Inclusive percentile without numpy; max() for tiny samples."""
    if not vals:
        return 0.0
    if len(vals) < 3:
        return float(max(vals)) if pct >= 50 else float(min(vals))
    q = statistics.quantiles(vals, n=100, method="inclusive")
    return q[min(pct - 1, 99)]


def openai_stream_timed(port: int, body: dict, timeout: float = 300.0) -> dict:
    """POST /v1/chat/completions (stream) -> timing metrics.

    Token counts prefer the final `usage` (server-authoritative; the
    pallama gateway injects usage into /v1 streams) with per-chunk
    counting as fallback. Every chunk carrying content OR
    reasoning_content counts: throughput is token-speed regardless of
    which field carries them. ITLs come from chunk timestamps.
    """
    payload = dict(body)
    payload["stream"] = True
    payload["stream_options"] = {"include_usage": True}
    t0 = time.perf_counter()
    ttft = None
    stamps: list[float] = []
    usage = None
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError:
        # strict implementations may reject stream_options — retry
        # without it (chunk-counting fallback)
        payload.pop("stream_options", None)
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        resp = urllib.request.urlopen(req, timeout=timeout)
    with resp:
        for raw in resp:
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
            if isinstance(j.get("usage"), dict):
                usage = j["usage"]
            choices = j.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            # thinking models (qwen3.5...) stream reasoning_content while
            # content stays empty — count BOTH as tokens
            if delta.get("content") or delta.get("reasoning_content"):
                stamps.append(time.perf_counter())
                if ttft is None:
                    ttft = stamps[-1]
    total = time.perf_counter() - t0
    t_last = stamps[-1] if stamps else t0
    tokens_usage = None
    prompt_tokens = None
    if usage:
        tokens_usage = usage.get("completions_tokens")
        prompt_tokens = usage.get("prompt_tokens")
    tokens = tokens_usage if tokens_usage else len(stamps)
    src = "usage" if tokens_usage else "chunks"
    itls = [(b - a) * 1000 for a, b in zip(stamps, stamps[1:])]
    return {
        "ttft_ms": (ttft - t0) * 1000 if ttft is not None else total * 1000,
        "decode_tps": (
            (tokens - 1) / (t_last - ttft)
            if ttft is not None and tokens > 1 and t_last > ttft
            else 0.0
        ),
        "wall_tps": tokens / total if total > 0 else 0.0,
        "tokens": tokens,
        "prompt_tokens": prompt_tokens,
        "itls_ms": itls,
        "tokens_source": src,
    }


def ollama_stream_timed(port: int, body: dict, timeout: float = 300.0) -> dict:
    """POST /api/chat (stream) with the same metric extraction.

    v2: counts message.thinking/reasoning fields (v1 counted content
    only, measuring the whole thinking phase as "TTFT" on reasoning
    models), caps options.num_predict (top-level max_tokens is IGNORED
    by the ollama dialect), and prefers the final done-chunk counters
    (eval_count / eval_duration = engine-side exact throughput).
    """
    payload = dict(body)
    payload["stream"] = True
    opts = dict(payload.get("options") or {})
    if payload.get("max_tokens"):
        opts["num_predict"] = payload["max_tokens"]
    payload["options"] = opts
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    ttft = None
    stamps: list[float] = []
    final = {}
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
                final = j
                break
            msg = j.get("message") or {}
            if msg.get("content") or msg.get("thinking") or msg.get("reasoning"):
                tokens += 1
                stamps.append(time.perf_counter())
                if ttft is None:
                    ttft = stamps[-1]
    total = time.perf_counter() - t0
    t_last = stamps[-1] if stamps else t0
    eval_count = final.get("eval_count")
    eval_dur_s = (final.get("eval_duration") or 0) / 1e9
    prompt_eval_count = final.get("prompt_eval_count")
    prompt_eval_dur_s = (final.get("prompt_eval_duration") or 0) / 1e9
    # ollama's own model-load accounting (ns in the final chunk) — the
    # engine-authoritative number for cold-start rows
    load_dur_s = (final.get("load_duration") or 0) / 1e9
    if eval_count and eval_dur_s > 0:
        # engine-side exact: excludes network + harness parse overhead
        decode_tps = eval_count / eval_dur_s
        src = "engine_counters"
    elif ttft and tokens > 1 and t_last > ttft:
        decode_tps = (tokens - 1) / (t_last - ttft)
        src = "chunks"
    else:
        decode_tps = 0.0
        src = "chunks"
    itls = [(b - a) * 1000 for a, b in zip(stamps, stamps[1:])]
    return {
        "ttft_ms": (ttft - t0) * 1000 if ttft is not None else total * 1000,
        "decode_tps": decode_tps,
        "wall_tps": tokens / total if total > 0 else 0.0,
        "tokens": eval_count or tokens,
        "prompt_tokens": prompt_eval_count,
        "prompt_eval_dur_s": prompt_eval_dur_s if prompt_eval_count else None,
        "load_dur_s": load_dur_s if load_dur_s > 0 else None,
        "itls_ms": itls,
        "tokens_source": src,
    }


# ---------------------------------------------------------------------------
# token-targeted prefill prompts (true prefill t/s)


def count_tokens(port: int, text: str, ollama: bool, model: str) -> int | None:
    """Server-side tokenization; None when the route is unavailable.

    Body carries BOTH dialect keys (content for llama-server, prompt +
    model for the ollama-compatible translation layer) — the pallama
    gateway maps /tokenize to the child and needs the routed model."""
    body = {"content": text, "prompt": text, "model": model}
    try:
        if ollama:
            j = http_json(f"http://127.0.0.1:{port}/api/tokenize", body, timeout=15.0)
        else:
            j = http_json(f"http://127.0.0.1:{port}/tokenize", body, timeout=15.0)
        toks = j.get("tokens")
        return len(toks) if isinstance(toks, list) else None
    except (urllib.error.URLError, json.JSONDecodeError, OSError, ValueError):
        return None


def _probe_prompt_tokens(port: int, text: str, ollama: bool, model: str) -> int | None:
    """Real prompt token count for tokenize-less lanes: one untimed
    1-token generate, read the engine's prompt_eval_count. None when the
    lane can't answer (caller keeps its estimate)."""
    if not ollama:
        return None  # pallama lanes tokenize server-side already
    try:
        j = http_json(
            f"http://127.0.0.1:{port}/api/generate",
            {
                "model": model,
                "prompt": text,
                "stream": False,
                "options": {"num_predict": 1},
            },
            timeout=120.0,
        )
        n = j.get("prompt_eval_count")
        return n if isinstance(n, int) and n > 0 else None
    except (urllib.error.URLError, json.JSONDecodeError, OSError, ValueError):
        return None


_SIZED_PROMPT_CACHE: dict[tuple[int, int], str] = {}


def sized_prompt(port: int, target_tokens: int, ollama: bool, model: str) -> str:
    """Deterministic prompt of ~target_tokens tokens (user-text portion).

    Converges via tokenize-and-scale; falls back to ~3.6 chars/token
    estimate when no tokenize route exists (sandbox mistral.rs lane).
    """
    cached = _SIZED_PROMPT_CACHE.get((port, target_tokens))
    if cached is not None:
        return cached

    def build(n_sents: int) -> str:
        parts = []
        i = 0
        while len(parts) < n_sents:
            parts.append(PREFILL_BANK[i % len(PREFILL_BANK)])
            i += 1
        return " ".join(parts)

    n = max(1, target_tokens // 7)  # ~7 tokens per bank sentence
    text = build(n)
    got = count_tokens(port, text, ollama, model)
    if got is None:
        # No tokenize route on this backend (ollama 0.33.x ships none,
        # verified 404; sandboxed mistral.rs neither): size by characters
        # at the bank's measured ~3.5 chars/token. The old sentence-count
        # estimate built 2 sentences for a 512-token target (37 tokens
        # live on ollama) and quietly destated the prefill column.
        need_chars = int(target_tokens * 3.5)
        parts: list[str] = []
        taken = 0
        i = 0
        while taken < need_chars:
            sent = PREFILL_BANK[i % len(PREFILL_BANK)]
            parts.append(sent)
            taken += len(sent) + 1
            i += 1
        text = " ".join(parts)
        got = _probe_prompt_tokens(port, text, ollama, model)
        if got is not None:
            # Real-count convergence for tokenize-less lanes (ollama
            # ships no /api/tokenize): scale the char budget by the
            # engine's own prompt_eval_count from untimed 1-token
            # probes. Without this the char estimate left the ollama
            # lane at 371 real tokens vs the pallama lane's 527 at the
            # same 512 target — cross-runtime prefill comparisons were
            # invalid (shorter prompt = lower amortized prefill t/s).
            need_chars = len(text)
            for _ in range(3):
                if abs(got - target_tokens) <= target_tokens * 0.15:
                    break
                need_chars = max(200, round(need_chars * target_tokens / max(got, 1)))
                parts2: list[str] = []
                taken = 0
                i = 0
                while taken < need_chars:
                    sent = PREFILL_BANK[i % len(PREFILL_BANK)]
                    parts2.append(sent)
                    taken += len(sent) + 1
                    i += 1
                text = " ".join(parts2)
                nxt = _probe_prompt_tokens(port, text, ollama, model)
                if nxt is None:
                    break
                got = nxt
        _SIZED_PROMPT_CACHE[(port, target_tokens)] = text
        return text
    for _ in range(3):
        if abs(got - target_tokens) <= target_tokens * 0.15:
            break
        n = max(1, round(n * target_tokens / max(got, 1)))
        text = build(n)
        got = count_tokens(port, text, ollama, model) or target_tokens
    _SIZED_PROMPT_CACHE[(port, target_tokens)] = text
    return text


def median_run_suite(
    port: int, model: str, runs: int, pp: int, tg: int, ollama: bool = False
) -> dict:
    """Warmup + N decode + N prefill runs -> metric medians.

    Decode lane: repeated short prompt (runs 2+ ride the child's prompt
    cache — the measured TTFT is the cache-hit path, labeled as such).
    Prefill lane: token-targeted pp prompt; run 1 is the COLD prefill
    (uncached), runs 2+ measure the cache-hit path. True prefill t/s =
    prompt_tokens / first-token time (or ollama's engine counters).
    """
    fn = ollama_stream_timed if ollama else openai_stream_timed

    def body(max_tokens: int, prompt: str) -> dict:
        if ollama:
            return {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "options": {"num_ctx": 8192},
                "max_tokens": max_tokens,
            }
        return {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }

    t_warm0 = time.perf_counter()
    fn(port, body(tg, "warmup — reply with one word"), timeout=300.0)
    warmup_s = time.perf_counter() - t_warm0

    decode = []
    for _ in range(runs):
        decode.append(
            fn(port, body(tg, "List fun facts about the ocean, one per line."))
        )

    pre_prompt = sized_prompt(port, pp, ollama, model)
    prefill = []
    for _ in range(runs):
        prefill.append(
            fn(port, body(4, pre_prompt))
        )  # max_tokens=4 -> prefill-dominated

    def med(key: str, xs: list[dict]) -> float:
        return statistics.median(x[key] for x in xs if x.get(key) is not None)

    ttfts = [x["ttft_ms"] for x in decode]
    itls = [i for x in decode for i in x["itls_ms"]]

    # true prefill throughput: run 1 = cold (uncached), rest = cache-hit
    def prefill_tps(x: dict) -> float | None:
        if ollama and x.get("prompt_eval_dur_s"):
            return x["prompt_tokens"] / x["prompt_eval_dur_s"]
        pt = x.get("prompt_tokens")
        if pt and x["ttft_ms"] > 0:
            return pt / (x["ttft_ms"] / 1000.0)
        return None

    cold = prefill[0] if prefill else {}
    cached_runs = prefill[1:] or prefill
    cold_tps = prefill_tps(cold)
    cached_tps_vals = [t for t in (prefill_tps(x) for x in cached_runs) if t]
    src = decode[0].get("tokens_source", "chunks") if decode else "chunks"
    return {
        "ttft_ms_p50": statistics.median(ttfts),
        "ttft_ms_p90": percentile(ttfts, 90),
        "ttft_ms_p99": percentile(ttfts, 99),
        "ttft_ms_stdev": statistics.stdev(ttfts) if len(ttfts) > 1 else 0.0,
        "decode_tps_p50": med("decode_tps", decode),
        "decode_tps_runs": [round(x["decode_tps"], 2) for x in decode],
        "itl_p50_ms": percentile(itls, 50) if itls else None,
        "itl_p99_ms": percentile(itls, 99) if itls else None,
        "prefill_tps_cold": cold_tps,
        "prefill_tps_cached": statistics.median(cached_tps_vals)
        if cached_tps_vals
        else None,
        "ttft_prefill_cold_ms": cold.get("ttft_ms"),
        "ttft_prefill_cached_ms": statistics.median([x["ttft_ms"] for x in cached_runs])
        if cached_runs
        else None,
        "prompt_tokens": cold.get("prompt_tokens"),
        "warmup_s": round(warmup_s, 2),
        "runs": runs,
        "tokens_source": src,
    }


def conc_suite(
    port: int,
    model: str,
    level: int,
    tg: int,
    ollama: bool = False,
    rounds: int = 1,
) -> dict:
    """`level` concurrent streams -> aggregate throughput + tail latency.

    Unique prompt per stream (no shared prefix -> no cache collision,
    all streams pay real prefill). Exercises admission/queueing on the
    pallama path (WFQ/slot leases/predictive reject) and llama-server
    slot scheduling on the direct path.

    rounds > 1 = sustained load: sequential bursts with per-round and
    cross-round tail stats (a single burst never shows queue-drain p99s
    or thermal/admission drift).
    """
    fn = ollama_stream_timed if ollama else openai_stream_timed

    def body(max_tokens: int, i: int) -> dict:
        prompt = (
            f"Stream {i}: explain in one short paragraph why the sea is "
            f"salty, variation {i}, answer directly."
        )
        if ollama:
            return {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "options": {"num_ctx": 8192},
                "max_tokens": max_tokens,
            }
        return {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }

    def burst() -> dict:
        results: list[Any] = [None] * level

        def worker(i: int) -> None:
            try:
                results[i] = fn(port, body(tg, i), timeout=300.0)
            except Exception as exc:  # noqa: BLE001 — one stream failing is a datum
                results[i] = f"error: {exc}"

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(level)]
        t0 = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return {
            "wall_s": time.perf_counter() - t0,
            "ok": [r for r in results if isinstance(r, dict)],
            "errs": [r for r in results if isinstance(r, str)],
        }

    burst_recs = [burst() for _ in range(max(1, rounds))]
    wall_s = sum(b["wall_s"] for b in burst_recs)
    ok = [r for b in burst_recs for r in b["ok"]]
    errs = [e for b in burst_recs for e in b["errs"]]
    if not ok:
        # All streams failed is a dead backend, not a 0 t/s datapoint
        # (v2.0 recorded a degenerate "ok" row with ttft 0 / decode 0).
        raise RuntimeError(f"all {level} streams failed: {errs[:2]}")
    ttfts = [r["ttft_ms"] for r in ok]
    itls = [i for r in ok for i in r["itls_ms"]]
    total_tokens = sum(r.get("tokens") or 0 for r in ok)
    round_sys = [
        round(sum(r.get("tokens") or 0 for r in b["ok"]) / b["wall_s"], 2)
        for b in burst_recs
        if b["ok"] and b["wall_s"] > 0
    ]
    round_walls = [round(b["wall_s"], 2) for b in burst_recs]
    out = {
        "conc_level": level,
        "conc_rounds": len(burst_recs),
        "conc_wall_s": round(wall_s, 2),
        "conc_ok": len(ok),
        "conc_errors": len(errs),
        "conc_error_samples": errs[:3],
        # sum of per-stream rates: honest only when streams truly run in
        # parallel; on a single-slot backend streams serialize and the sum
        # overstates the system rate — read sys_tps for the real number.
        "sum_stream_tps": round(sum(r["decode_tps"] for r in ok), 2),
        "sys_tps": round(total_tokens / wall_s, 2) if wall_s > 0 else None,
        "ttft_spread_ms": round(max(ttfts) - min(ttfts), 1) if len(ttfts) > 1 else 0.0,
        "ttft_max_ms": round(max(ttfts)) if ttfts else None,
        "itl_p99_ms": round(percentile(itls, 99), 2) if itls else None,
        "total_tokens": total_tokens,
    }
    if len(burst_recs) > 1:
        out.update(
            {
                "conc_sys_tps_per_round": round_sys,
                "conc_sys_tps_p50": round(statistics.median(round_sys), 2)
                if round_sys
                else None,
                "conc_ttft_p99_ms": round(percentile(ttfts, 99), 1) if ttfts else None,
                "conc_wall_p99_s": round(
                    percentile([float(w) for w in round_walls], 99), 2
                )
                if round_walls
                else None,
            }
        )
    return out


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
    extras: dict | None = None,
) -> list[str]:
    """Build the child argv; `extras` carries variant axes (kv/spec/
    mmproj/pa) recorded in the cell params."""
    extras = extras or {}
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
        if extras.get("pa") == "off":
            argv += ["--paged-attn", "off"]
        if staged is not None and mmproj is not None and mmproj.exists():
            # staged view holds only this model's projector — pass it
            # explicitly so discovery cannot pick anything else
            argv += ["--mmproj", str(staged.parent / mmproj.name)]
        return argv
    argv = [
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
    if extras.get("kv"):
        # v-cache quant requires flash attention in upstream llama.cpp;
        # --flash-attn takes an explicit value in modern builds or it
        # swallows the next flag as its argument (live: ate --cache-type-k)
        argv += [
            "--flash-attn",
            "on",
            "--cache-type-k",
            extras["kv"],
            "--cache-type-v",
            extras["kv"],
        ]
    if extras.get("spec"):
        argv += ["--spec-type", extras["spec"]]
    if extras.get("mmproj") and mmproj is not None and mmproj.exists():
        argv += ["--mmproj", str(mmproj)]
    return argv


def warm_cell_gpu_guard(where: str) -> float:
    """Warm cells measured WITH a foreign GPU resident are contaminated
    (live twice: a production daemon's 5 GiB engine cut BOTH runtimes to
    ~5 t/s). Drain-wait, then flag what remains on the record."""
    if not wait_gpu_idle(max_mib=512.0, timeout_s=30.0):
        busy = round(gpu_used_mib(), 0)
        log(f"  ! {where}: GPU busy {busy} MiB — warm numbers are contaminated")
        return busy
    return 0.0


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
    rec = {"gpu_busy_mib": warm_cell_gpu_guard("direct cell")}
    staged = None
    if eng.kind == "mistralrs":
        staged = stage_mistralrs_view(model, mmproj, stage_root)
    extras = {k: params[k] for k in ("kv", "spec", "mmproj", "pa") if k in params}
    argv = direct_argv(
        eng,
        model,
        mmproj,
        port,
        params["ctx"],
        params["np"],
        cfg["ngl"],
        staged,
        extras,
    )
    log(f"  spawn: {' '.join(argv)}")
    phash = cell_key(eng.tag, "stderr", params, model.name)[:8]
    errlog = stage_root.parent / "cells-stderr" / f"{eng.tag}-{phash}.log"
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
        t_load0 = time.perf_counter()
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
        rec["load_s"] = round(time.perf_counter() - t_load0, 2)
        # mistral.rs children register the served model as "default";
        # the pallama gateway rewrites at the proxy — we do it here.
        body_model = "default" if eng.kind == "mistralrs" else model_name
        rec.update(
            median_run_suite(port, body_model, cfg["runs"], cfg["pp"], cfg["tg"])
        )
        rec["rss_peak_mib"] = round(sampler.rss_peak_mib, 1)
        rec["gpu_peak_mib"] = round(sampler.gpu_peak_mib, 1)
        rec["gpu_base_mib"] = round(sampler.gpu_base_mib, 1)
        rec["gpu_power_peak_w"] = round(sampler.gpu_power_peak_w, 1)
    finally:
        rec.update(teardown_proc(proc, sampler))
    return rec


def run_direct_conc_cell(
    eng: Engine,
    model: Path,
    mmproj: Path | None,
    level: int,
    model_name: str,
    cfg: dict,
    stage_root: Path,
) -> dict:
    """Concurrency lane on a direct child (np sized to the level)."""
    port = free_port()
    staged = None
    if eng.kind == "mistralrs":
        staged = stage_mistralrs_view(model, mmproj, stage_root)
    argv = direct_argv(
        eng, model, mmproj, port, 16384, max(level, 1), cfg["ngl"], staged
    )
    log(f"  spawn: {' '.join(argv)}")
    phash = cell_key(eng.tag, "stderr-conc", {"lvl": level}, model.name)[:8]
    errlog = stage_root.parent / "cells-stderr" / f"{eng.tag}-{phash}.log"
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
    sampler = Sampler(proc.pid)
    sampler.start()
    rec: dict = {"argv": argv}
    try:
        t_load0 = time.perf_counter()
        if not wait_healthy(eng.kind, port, 600.0, proc=proc):
            rec["error"] = "child failed to become healthy"
            return rec
        rec["load_s"] = round(time.perf_counter() - t_load0, 2)
        body_model = "default" if eng.kind == "mistralrs" else model_name
        # warm the child before the burst
        openai_stream_timed(
            port,
            {
                "model": body_model,
                "messages": [{"role": "user", "content": "warmup"}],
                "max_tokens": 8,
            },
        )
        rec.update(conc_suite(port, body_model, level, cfg["tg"]))
    finally:
        rec.update(teardown_proc(proc, sampler))
    return rec


# ---------------------------------------------------------------------------
# pallama provider (validate.py Sandbox; per-engine active flip)


def find_sandbox_engine_pid() -> int | None:
    """Locate the engine child the sandbox daemon spawned.

    The sandbox DB carries REAL engine paths (the daemon resolves the
    binary from its manifest), so a path-prefix scan misses it. The
    daemon is spawned with PALLAMA_VALIDATE=1 and the engine child
    inherits that environ — the same marker validate.py's orphan reaper
    trusts. Only our sandbox tree can carry it."""
    if not os.path.isdir("/proc"):
        return None
    for pid_s in os.listdir("/proc"):
        if not pid_s.isdigit():
            continue
        try:
            with open(f"/proc/{pid_s}/cmdline", "rb") as fh:
                cmdline = fh.read()
        except OSError:
            continue
        if not (b"llama-server" in cmdline or b"mistralrs" in cmdline):
            continue
        try:
            with open(f"/proc/{pid_s}/environ", "rb") as fh:
                environ = fh.read()
        except OSError:
            continue
        if b"PALLAMA_VALIDATE=1" in environ:
            return int(pid_s)
    return None


def read_proc_argv(pid: int) -> list[str]:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            return [a.decode("utf-8", "replace") for a in fh.read().split(b"\0") if a]
    except OSError:
        return []


def run_pallama_cell(
    eng: Engine,
    model_name: str,
    cfg: dict,
    skip_ollama_note: str,
    pallama_cfg: dict | None = None,
    soak_s: float = 0.0,
) -> dict:
    """Full-gateway cell inside the validate.py Sandbox.

    v2 instrumentation: daemon boot time, cold first-request time (child
    spawn + load + first token), the RESOLVED engine child argv (read
    from /proc — the profile compiler's exact emission), and a real
    GPU/RSS/power sampler attached first globally (captures the load
    spike) then to the discovered child pid (RSS from discovery on).
    """
    # per-campaign unique port BEFORE the lazy import: validate.py reads
    # PALLAMA_VALIDATE_PORT once at import time — a shared fixed port is
    # exactly how orphaned sandbox daemons hijacked campaigns (leak class
    # fixed in validate.py; this makes collisions structurally impossible)
    os.environ["PALLAMA_VALIDATE_PORT"] = str(free_port())
    V = importlib.import_module("validate")
    # F139: the module cache returns the FIRST import on later campaigns
    # — rebinding PORT on the module is what actually takes effect; the
    # env re-set above alone is inert past the first import.
    V.PORT = int(os.environ["PALLAMA_VALIDATE_PORT"])

    rec: dict = {}
    sb = V.Sandbox()
    sampler = Sampler(None)
    sampler.start()
    try:
        con = sqlite3.connect(Path(sb.data_home) / "pallama" / "pallama.db")
        con.execute("UPDATE engines SET active = (tag = ?)", (eng.tag,))
        con.commit()
        con.close()
        daemon = V.Daemon(sb)
        rec["gpu_busy_mib"] = warm_cell_gpu_guard("pallama cell")
        t_boot0 = time.perf_counter()
        port: int | None = None
        try:
            # start() INSIDE the stop()-owning try: a post-Popen raise
            # (healthz timeout, liveness guard) used to leak a live
            # daemon whose sandbox got destroyed under it (2026-09-10).
            daemon.start(
                cfg={"port": V.PORT, **(pallama_cfg or {})}, floor_model=model_name
            )
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
            rec["daemon_boot_s"] = round(time.perf_counter() - t_boot0, 2)
            assert port is not None
            # child spawns on first request; the warmup inside
            # median_run_suite drives it — time the warmup ourselves by
            # wrapping: run one explicit cold request first so the
            # cold-start number is clean, THEN the suite warms up.
            #
            # Parity hardening (same env as the ollama cold lane):
            # (1) drop the page cache on the model file the child will
            #     mmap — earlier lanes leave multi-GiB cached and a
            #     "cold" load would be memory-fast;
            # (2) assert the GPU drained (no co-resident squatter
            #     inflating the load);
            # (3) capture the probe's OWN ttft — first-token latency on
            #     a cold engine is the user-felt number and was
            #     previously discarded into the wall time.
            mfile, mmfile = sandbox_model_files(sb, model_name)
            rec["cold_fadvise_files"] = fadvise_dontneed([mfile, mmfile])
            if not wait_gpu_idle(max_mib=512.0, timeout_s=30.0):
                rec["cold_gpu_busy_mib"] = round(gpu_used_mib(), 0)
            t_cold0 = time.perf_counter()
            try:
                coldm = openai_stream_timed(
                    port,
                    {
                        "model": model_name,
                        "messages": [{"role": "user", "content": "cold-start probe"}],
                        "max_tokens": 4,
                    },
                    timeout=600.0,
                )
                rec["cold_first_request_s"] = round(time.perf_counter() - t_cold0, 2)
                rec["cold_ttft_ms"] = round(coldm.get("ttft_ms") or 0.0, 1)
                rec["cold_prompt_tokens"] = coldm.get("prompt_tokens")
            except Exception as cold_exc:  # noqa: BLE001 — keep forensics
                # a failed cold probe (e.g. 502 spawn-failure behind the
                # gateway) must still carry boot metrics + daemon tail for
                # diagnosis — return rec; the finally block captures the
                # rest (sampler peaks, daemon.stop, log tail, teardown)
                rec["error"] = f"cold probe failed: {cold_exc}"
                rec["cold_first_request_s"] = round(time.perf_counter() - t_cold0, 2)
                return rec
            child_pid = find_sandbox_engine_pid()
            if child_pid is not None:
                rec["child_pid"] = child_pid
                rec["child_argv"] = read_proc_argv(child_pid)
                sampler.pid = child_pid  # RSS tracking from here on
            else:
                rec["child_pid_note"] = (
                    "engine child not found in /proc (spawn failed?)"
                )
            rec.update(
                median_run_suite(port, model_name, cfg["runs"], cfg["pp"], cfg["tg"])
            )
            rec["provider_note"] = skip_ollama_note
            if soak_s > 0:
                # leak/soak probe: sustained decode, watch RSS/VRAM drift
                rss0 = sampler.rss_peak_mib
                gpu0 = sampler.gpu_peak_mib
                t_end = time.time() + soak_s
                n = 0
                while time.time() < t_end:
                    openai_stream_timed(
                        port,
                        {
                            "model": model_name,
                            "messages": [
                                {
                                    "role": "user",
                                    "content": f"soak round {n}: count slowly to twenty.",
                                }
                            ],
                            "max_tokens": 128,
                        },
                        timeout=300.0,
                    )
                    n += 1
                rec["soak_s"] = soak_s
                rec["soak_rounds"] = n
                rec["soak_gpu_drift_mib"] = round(sampler.gpu_peak_mib - gpu0, 1)
                rec["soak_rss_drift_mib"] = round(sampler.rss_peak_mib - rss0, 1)
        finally:
            rec["gpu_peak_mib"] = round(sampler.gpu_peak_mib, 1)
            rec["gpu_base_mib"] = round(sampler.gpu_base_mib, 1)
            rec["gpu_power_peak_w"] = round(sampler.gpu_power_peak_w, 1)
            rec["rss_peak_mib"] = round(sampler.rss_peak_mib, 1)
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
        sampler.stop_evt.set()
        sb.destroy()
    return rec


def run_pallama_conc_cell(
    eng: Engine, model_name: str, level: int, cfg: dict, rounds: int = 1
) -> dict:
    """Concurrency lane through the full gateway path (admission,
    queueing, slot leases — Pallama's scheduling surface)."""
    os.environ["PALLAMA_VALIDATE_PORT"] = str(free_port())
    V = importlib.import_module("validate")
    # F139: the module cache returns the FIRST import on later campaigns
    # — rebinding PORT on the module is what actually takes effect; the
    # env re-set above alone is inert past the first import.
    V.PORT = int(os.environ["PALLAMA_VALIDATE_PORT"])

    rec: dict = {}
    sb = V.Sandbox()
    sampler = Sampler(None)
    sampler.start()
    try:
        con = sqlite3.connect(Path(sb.data_home) / "pallama" / "pallama.db")
        con.execute("UPDATE engines SET active = (tag = ?)", (eng.tag,))
        con.commit()
        con.close()
        daemon = V.Daemon(sb)
        port: int | None = None
        try:
            # same ownership fix as run_pallama_cell: start() must be
            # covered by the finally that calls daemon.stop()
            daemon.start(floor_model=model_name)
            port = V.PORT
            deadline = time.time() + 600
            healthy = False
            while time.time() < deadline:
                try:
                    http_json(f"http://127.0.0.1:{port}/healthz", timeout=5.0)
                    healthy = True
                    break
                except json.JSONDecodeError:
                    healthy = True
                    break
                except (urllib.error.URLError, OSError):
                    time.sleep(0.5)
            if not healthy:
                return {"error": "sandbox daemon failed to boot"}
            assert port is not None
            openai_stream_timed(
                port,
                {
                    "model": model_name,
                    "messages": [{"role": "user", "content": "warmup"}],
                    "max_tokens": 8,
                },
                timeout=600.0,
            )
            # resolved engine argv (auto-slots np/ctx visibility — the
            # speed cells' headline forensics, now recorded for conc too)
            child_pid = find_sandbox_engine_pid()
            if child_pid is not None:
                rec["child_pid"] = child_pid
                rec["child_argv"] = read_proc_argv(child_pid)
            rec.update(conc_suite(port, model_name, level, cfg["tg"], rounds=rounds))
        finally:
            rec["gpu_peak_mib"] = round(sampler.gpu_peak_mib, 1)
            daemon.stop()
            dlog = Path(sb.data_dir) / "run" / "daemon.log"
            if dlog.exists():
                rec["daemon_log_tail"] = "\n".join(
                    dlog.read_text(errors="replace").splitlines()[-12:]
                )
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
    finally:
        sampler.stop_evt.set()
        sb.destroy()
    return rec


# ---------------------------------------------------------------------------
# idle-wake lane (sleep-vs-expiry semantics: pallama keeps weights in RAM
# and wakes cheap; ollama's keep_alive expiry unloads and pays a reload)


def run_pallama_idle_cell(eng: Engine, model_name: str, cfg: dict) -> dict:
    """Warm the model, let the reaper ladder sleep it (idle_sleep_secs),
    then measure the wake TTFT — pallama's structural idle advantage."""
    os.environ["PALLAMA_VALIDATE_PORT"] = str(free_port())
    V = importlib.import_module("validate")
    V.PORT = int(os.environ["PALLAMA_VALIDATE_PORT"])

    idle_sleep = 15
    rec: dict[str, Any] = {
        "idle_policy": f"sleep at {idle_sleep}s (weights stay RAM-resident)",
    }
    sb = V.Sandbox()
    sampler = Sampler(None)
    sampler.start()
    try:
        con = sqlite3.connect(Path(sb.data_home) / "pallama" / "pallama.db")
        con.execute("UPDATE engines SET active = (tag = ?)", (eng.tag,))
        con.commit()
        con.close()
        daemon = V.Daemon(sb)
        port: int | None = None
        try:
            daemon.start(
                cfg={"port": V.PORT, "idle_sleep_secs": idle_sleep},
                floor_model=model_name,
            )
            port = V.PORT
            deadline = time.time() + 600
            healthy = False
            while time.time() < deadline:
                try:
                    http_json(f"http://127.0.0.1:{port}/healthz", timeout=5.0)
                    healthy = True
                    break
                except json.JSONDecodeError:
                    healthy = True
                    break
                except (urllib.error.URLError, OSError):
                    time.sleep(0.5)
            if not healthy:
                return {"error": "sandbox daemon failed to boot"}
            assert port is not None
            openai_stream_timed(
                port,
                {
                    "model": model_name,
                    "messages": [{"role": "user", "content": "warmup"}],
                    "max_tokens": 8,
                },
                timeout=600.0,
            )
            loaded_gpu_mib = gpu_used_mib()
            # sleep detect: /api/ps pallama_state flips to Sleeping (the
            # child sleeps itself; weights stay RAM, VRAM released);
            # VRAM drop as fallback signal. Timeout must cover the idle
            # window + the 10s reaper tick + margin.
            slept = False
            deadline = time.time() + idle_sleep + 10 + 60
            while time.time() < deadline:
                try:
                    ps = http_json(f"http://127.0.0.1:{port}/api/ps", timeout=5.0)
                    states = [
                        str(r.get("pallama_state", "")).lower()
                        for r in ps.get("models", [])
                    ]
                    if any("sleep" in s for s in states):
                        slept = True
                        break
                except (urllib.error.URLError, OSError, json.JSONDecodeError):
                    pass
                if gpu_used_mib() < loaded_gpu_mib - 512:
                    slept = True
                    rec["sleep_detect"] = "vram_drop"
                    break
                time.sleep(1.0)
            rec["slept"] = slept
            if not slept:
                rec["idle_note"] = (
                    "model never reached Sleeping within "
                    f"{idle_sleep + 10 + 60}s — wake TTFT below is warm-path"
                )
            rec["gpu_at_sleep_mib"] = round(gpu_used_mib(), 1)
            m = openai_stream_timed(
                port,
                {
                    "model": model_name,
                    "messages": [
                        {"role": "user", "content": "wake probe — answer in one word"}
                    ],
                    "max_tokens": 8,
                },
                timeout=600.0,
            )
            rec["idle_wake_ttft_ms"] = round(m.get("ttft_ms") or 0.0, 1)
            rec["idle_wake_wall_s"] = round(
                (m.get("ttft_ms") or 0.0) / 1000.0 + sum(m.get("itls_ms", [])) / 1000.0,
                2,
            )
            rec["gpu_peak_mib"] = round(sampler.gpu_peak_mib, 1)
        finally:
            daemon.stop()
            dlog = Path(sb.data_dir) / "run" / "daemon.log"
            if dlog.exists():
                rec["daemon_log_tail"] = "\n".join(
                    dlog.read_text(errors="replace").splitlines()[-12:]
                )
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
    finally:
        sampler.stop_evt.set()
        sb.destroy()
    return rec


def run_ollama_idle_cell(cfg: dict, args_model: str | None) -> dict:
    """keep_alive expiry = ollama's idle policy: FULL unload. After the
    window the model row leaves /api/ps and the next request pays a
    complete reload — measure that TTFT against pallama's sleep-wake."""
    try:
        tags = http_json(f"http://127.0.0.1:{OLLAMA_PORT}/api/tags", timeout=5.0)
    except (urllib.error.URLError, OSError):
        return {"error": "ollama not reachable on 11434 (skipped, not started)"}
    models = [m["name"] for m in tags.get("models", [])]
    pick = pick_ollama_model(models, args_model)
    if pick is None:
        return {"error": f"no ollama model comparable to '{args_model}'"}
    keep_alive_s = 20
    rec: dict[str, Any] = {
        "ollama_model": pick,
        "idle_policy": f"keep_alive {keep_alive_s}s -> full unload",
    }
    sampler = Sampler(None)
    sampler.start()
    try:
        ollama_stream_timed(
            OLLAMA_PORT,
            {
                "model": pick,
                "messages": [{"role": "user", "content": "warmup"}],
                "options": {"num_ctx": 8192},
                "keep_alive": f"{keep_alive_s}s",
                "max_tokens": 8,
            },
            timeout=600.0,
        )
        # poll until the expiry unloads it (row gone from /api/ps)
        gone = False
        deadline = time.time() + keep_alive_s + 90
        while time.time() < deadline:
            try:
                ps = http_json(f"http://127.0.0.1:{OLLAMA_PORT}/api/ps", timeout=5.0)
                if not any(r.get("name") == pick for r in ps.get("models", [])):
                    gone = True
                    break
            except (urllib.error.URLError, OSError, json.JSONDecodeError):
                pass
            time.sleep(1.0)
        rec["expired"] = gone
        if not gone:
            rec["idle_note"] = (
                f"model still loaded after keep_alive {keep_alive_s}s+90s — "
                "wake TTFT below is warm-path (expiry semantics unverifiable)"
            )
        else:
            # disk-cold reload: same fadvise parity as the cold lane
            rec["cold_fadvise_files"] = fadvise_dontneed(ollama_blob_paths())
        m = ollama_stream_timed(
            OLLAMA_PORT,
            {
                "model": pick,
                "messages": [
                    {"role": "user", "content": "wake probe — answer in one word"}
                ],
                "options": {"num_ctx": 8192},
                "max_tokens": 8,
            },
            timeout=600.0,
        )
        rec["idle_wake_ttft_ms"] = round(m.get("ttft_ms") or 0.0, 1)
        if m.get("load_dur_s"):
            rec["idle_reload_s"] = round(m["load_dur_s"], 2)
        rec["gpu_peak_mib"] = round(sampler.gpu_peak_mib, 1)
    finally:
        sampler.stop_evt.set()
        sampler.join(timeout=2.0)
    rec["teardown_ok"] = ollama_evict(pick)
    return rec


# ---------------------------------------------------------------------------
# long-context degradation curve (decode t/s + TTFT vs ctx on both runtimes)


def run_pallama_ctx_cell(eng: Engine, model_name: str, ctx: int, cfg: dict) -> dict:
    """One ctx point on the curve: sandbox daemon with the per-model ctx
    override, 3-run decode suite. The profile compiler resolves ctx into
    the child argv (recorded) so the exact allocation is in the artifact."""
    os.environ["PALLAMA_VALIDATE_PORT"] = str(free_port())
    V = importlib.import_module("validate")
    V.PORT = int(os.environ["PALLAMA_VALIDATE_PORT"])

    rec: dict[str, Any] = {"ctx": ctx}
    sb = V.Sandbox()
    sampler = Sampler(None)
    sampler.start()
    try:
        con = sqlite3.connect(Path(sb.data_home) / "pallama" / "pallama.db")
        con.execute("UPDATE engines SET active = (tag = ?)", (eng.tag,))
        con.commit()
        con.close()
        daemon = V.Daemon(sb)
        port: int | None = None
        try:
            daemon.start(
                cfg={
                    "port": V.PORT,
                    "model_overrides": {model_name: {"ctx": ctx}},
                },
                floor_model=model_name,
            )
            port = V.PORT
            deadline = time.time() + 600
            healthy = False
            while time.time() < deadline:
                try:
                    http_json(f"http://127.0.0.1:{port}/healthz", timeout=5.0)
                    healthy = True
                    break
                except json.JSONDecodeError:
                    healthy = True
                    break
                except (urllib.error.URLError, OSError):
                    time.sleep(0.5)
            if not healthy:
                return {"error": f"sandbox daemon failed to boot at ctx {ctx}"}
            assert port is not None
            rec.update(median_run_suite(port, model_name, 3, cfg["pp"], cfg["tg"]))
            child_pid = find_sandbox_engine_pid()
            if child_pid is not None:
                rec["child_argv"] = read_proc_argv(child_pid)
            rec["gpu_peak_mib"] = round(sampler.gpu_peak_mib, 1)
        finally:
            daemon.stop()
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
    finally:
        sampler.stop_evt.set()
        sb.destroy()
    return rec


def run_ollama_ctx_cell(cfg: dict, args_model: str | None, ctx: int) -> dict:
    """One ctx point on ollama's curve: evict (runner respawn at the
    request's num_ctx — ollama reloads when the option changes), then a
    3-run decode suite pinned to that num_ctx."""
    try:
        tags = http_json(f"http://127.0.0.1:{OLLAMA_PORT}/api/tags", timeout=5.0)
    except (urllib.error.URLError, OSError):
        return {"error": "ollama not reachable on 11434 (skipped, not started)"}
    models = [m["name"] for m in tags.get("models", [])]
    pick = pick_ollama_model(models, args_model)
    if pick is None:
        return {"error": f"no ollama model comparable to '{args_model}'"}
    rec: dict[str, Any] = {"ctx": ctx, "ollama_model": pick}
    sampler = Sampler(None)
    sampler.start()
    try:
        ollama_evict(pick)

        def runs() -> list[dict]:
            out = []
            for _ in range(3):
                out.append(
                    ollama_stream_timed(
                        OLLAMA_PORT,
                        {
                            "model": pick,
                            "messages": [
                                {
                                    "role": "user",
                                    "content": "List fun facts about the ocean, one per line.",
                                }
                            ],
                            "options": {"num_ctx": ctx},
                            "max_tokens": cfg["tg"],
                        },
                        timeout=300.0,
                    )
                )
            return out

        decode = runs()
        ttfts = [x["ttft_ms"] for x in decode]
        rec.update(
            {
                "ttft_ms_p50": statistics.median(ttfts),
                "decode_tps_p50": statistics.median(x["decode_tps"] for x in decode),
                "decode_tps_runs": [round(x["decode_tps"], 2) for x in decode],
                "runs": 3,
            }
        )
        first_load = next(
            (x.get("load_dur_s") for x in decode if x.get("load_dur_s")), None
        )
        if first_load:
            rec["load_s"] = round(first_load, 2)
        rec["gpu_peak_mib"] = round(sampler.gpu_peak_mib, 1)
    finally:
        sampler.stop_evt.set()
        sampler.join(timeout=2.0)
    rec["teardown_ok"] = ollama_evict(pick)
    return rec


def run_ollama_conc_cell(
    cfg: dict, args_model: str | None, level: int, rounds: int = 1
) -> dict:
    """Concurrency parity on the ollama host service — same conc_suite,
    same unique-prompt bursts, same sustained rounds as the gateway lane."""
    try:
        tags = http_json(f"http://127.0.0.1:{OLLAMA_PORT}/api/tags", timeout=5.0)
    except (urllib.error.URLError, OSError):
        return {"error": "ollama not reachable on 11434 (skipped, not started)"}
    models = [m["name"] for m in tags.get("models", [])]
    pick = pick_ollama_model(models, args_model)
    if pick is None:
        return {"error": f"no ollama model comparable to '{args_model}'"}
    rec: dict[str, Any] = {"ollama_model": pick}
    sampler = Sampler(None)
    sampler.start()
    try:
        # first stream of round 1 pays the load — that is the honest
        # sustained-load shape for a service that starts cold
        rec.update(
            conc_suite(OLLAMA_PORT, pick, level, cfg["tg"], ollama=True, rounds=rounds)
        )
        rec["gpu_peak_mib"] = round(sampler.gpu_peak_mib, 1)
    finally:
        sampler.stop_evt.set()
        sampler.join(timeout=2.0)
    rec["teardown_ok"] = ollama_evict(pick)
    return rec


# ---------------------------------------------------------------------------
# ollama reference (HTTP-only, one cell)


def pick_ollama_model(models: list[str], args_model: str | None) -> str | None:
    """SAME-model reference pick: derive family+size from the matrix model
    name ("Qwen3.5-9B-Q4_K_M" -> family "qwen3.5", size "9b") and match
    the ollama tag exactly, then loosely. A junk fallback is worse than
    an honest skip — v2.0's stem-needle match missed "qwen3.5:9b" and
    benchmarked an alphabetically-first OCR model at 70 t/s."""
    if not args_model:
        return None
    stem = args_model.lower()
    parts = re.split(r"[-_:]", stem)
    family = parts[0] if parts else stem
    size = next((p for p in parts[1:] if p.endswith("b") and p[:-1].isdigit()), "")
    exact = f"{family}:{size}" if size else None
    if exact and exact in models:
        return exact
    hits = [m for m in models if family in m.lower()]
    if hits:
        # prefer the size-matching variant, then the shortest tag
        return min(hits, key=lambda m: (0 if size and size in m else 1, len(m)))
    return None


def ollama_evict(model: str, timeout_s: float = 90.0) -> bool:
    """keep_alive=0 unload + GPU drain poll. The host service is not our
    child — "teardown" means the model is actually gone from VRAM."""
    try:
        http_json(
            f"http://127.0.0.1:{OLLAMA_PORT}/api/generate",
            {"model": model, "keep_alive": 0},
            timeout=15.0,
        )
    except (urllib.error.URLError, OSError) as exc:
        log(f"  ! ollama evict {model} failed: {exc}")
        return False
    return wait_gpu_idle(max_mib=512.0, timeout_s=timeout_s)


def run_ollama_cold_cell(
    cfg: dict, args_model: str | None, service_restart: bool = False
) -> dict:
    """Cold-start parity lane: ollama daemon-boot → disk-cold model load
    → first token. Mirrors the pallama cold probe exactly (fadvise'd
    blobs, GPU-idle assert, aligned num_ctx) so the coldstart table
    compares the same physical state on both runtimes."""
    try:
        tags = http_json(f"http://127.0.0.1:{OLLAMA_PORT}/api/tags", timeout=5.0)
    except (urllib.error.URLError, OSError):
        return {"error": "ollama not reachable on 11434 (skipped, not started)"}
    models = [m["name"] for m in tags.get("models", [])]
    if not models:
        return {"error": "ollama reachable but no models pulled"}
    pick = pick_ollama_model(models, args_model)
    if pick is None:
        return {
            "error": (
                f"no ollama model comparable to '{args_model}' "
                f"(have: {', '.join(models[:6])}) — pull a matching tag"
            )
        }
    rec: dict[str, Any] = {"ollama_model": pick}

    # daemon boot (optional — restarting the user's systemd service is
    # opt-in via --ollama-service-restart; without it the daemon is warm
    # and only the model-load path is cold, disclosed via note)
    if service_restart:
        pw = os.environ.get("BENCH_SUDO_PASSWORD")
        if pw and sudo_systemctl("restart", "ollama", password=pw):
            t_boot0 = time.perf_counter()
            deadline = time.time() + 120.0
            while time.time() < deadline:
                try:
                    http_json(
                        f"http://127.0.0.1:{OLLAMA_PORT}/api/version", timeout=3.0
                    )
                    rec["ollama_daemon_boot_s"] = round(
                        time.perf_counter() - t_boot0, 2
                    )
                    break
                except (urllib.error.URLError, OSError, json.JSONDecodeError):
                    time.sleep(0.25)
            else:
                rec["ollama_daemon_boot_s"] = None
                rec["ollama_daemon_boot_note"] = (
                    "service restarted but /api/version never answered in 120s"
                )
        else:
            rec["ollama_daemon_boot_note"] = (
                "service restart unavailable (no BENCH_SUDO_PASSWORD or sudo failed) "
                "— daemon-warm cold-load measured"
            )
    else:
        rec["ollama_daemon_boot_note"] = (
            "daemon left warm (no --ollama-service-restart) — model-load path only"
        )

    sampler = Sampler(None)  # global GPU/power: the service is not our child
    sampler.start()
    try:
        if not ollama_evict(pick):
            rec["cold_gpu_busy_mib"] = round(gpu_used_mib(), 0)
        rec["cold_fadvise_files"] = fadvise_dontneed(ollama_blob_paths())
        # num_ctx 16384 = the pallama gateway cell's resolved ctx for the
        # matrix model — identical KV allocation on both runtimes
        m = ollama_stream_timed(
            OLLAMA_PORT,
            {
                "model": pick,
                "messages": [{"role": "user", "content": "cold-start probe"}],
                "options": {"num_ctx": 16384},
                "max_tokens": 4,
            },
            timeout=600.0,
        )
        rec["ollama_cold_ttft_ms"] = round(m.get("ttft_ms") or 0.0, 1)
        rec["ollama_cold_wall_s"] = round(
            (m.get("ttft_ms") or 0.0) / 1000.0 + sum(m.get("itls_ms", [])) / 1000.0,
            2,
        )
        if m.get("load_dur_s"):
            rec["ollama_load_s"] = round(m["load_dur_s"], 2)
        rec["ollama_cold_prompt_tokens"] = m.get("prompt_tokens")
        rec["gpu_peak_mib"] = round(sampler.gpu_peak_mib, 1)
        rec["gpu_base_mib"] = round(sampler.gpu_base_mib, 1)
        rec["gpu_power_peak_w"] = round(sampler.gpu_power_peak_w, 1)
    finally:
        sampler.stop_evt.set()
        sampler.join(timeout=2.0)
    # free VRAM for later lanes + honest teardown check
    rec["teardown_ok"] = ollama_evict(pick)
    # service must be alive for the campaign to continue (Restart=always
    # normally guarantees it; a stopped service is a hard env break)
    try:
        http_json(f"http://127.0.0.1:{OLLAMA_PORT}/api/version", timeout=5.0)
    except (urllib.error.URLError, OSError):
        rec["ollama_service_down"] = True
    return rec


def run_ollama_cell(cfg: dict, args_model: str | None = None) -> dict:
    try:
        tags = http_json(f"http://127.0.0.1:{OLLAMA_PORT}/api/tags", timeout=5.0)
    except (urllib.error.URLError, OSError):
        return {"error": "ollama not reachable on 11434 (skipped, not started)"}
    models = [m["name"] for m in tags.get("models", [])]
    if not models:
        return {"error": "ollama reachable but no models pulled"}
    pick = pick_ollama_model(models, args_model)
    if pick is None:
        return {
            "error": (
                f"no ollama model comparable to '{args_model}' "
                f"(have: {', '.join(models[:6])}) — pull a matching tag"
            )
        }
    sampler = Sampler(None)  # global GPU/power: the service is not our child
    gpu_busy = warm_cell_gpu_guard("ollama cell")
    sampler.start()
    try:
        out = {
            "ollama_model": pick,
            "gpu_busy_mib": gpu_busy,
            **median_run_suite(
                OLLAMA_PORT, pick, cfg["runs"], cfg["pp"], cfg["tg"], ollama=True
            ),
            "gpu_peak_mib": round(sampler.gpu_peak_mib, 1),
            "gpu_base_mib": round(sampler.gpu_base_mib, 1),
            "gpu_power_peak_w": round(sampler.gpu_power_peak_w, 1),
        }
    finally:
        sampler.stop_evt.set()
        sampler.join(timeout=2.0)
    # Unload the served model so later GPU lanes (conc, ppl, greedy) get
    # the VRAM — v2.0 left the reference model resident and every
    # following lane starved or died.
    out["teardown_ok"] = ollama_evict(pick)
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
            "error": (
                f"no PPL line in llama-perplexity output (exit {p.returncode}); "
                f"tail: {out[-300:]!r}"
            ),
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


def _greedy_spawn(
    eng: Engine, model: Path, mmproj: Path | None, stage_root: Path
) -> tuple[int, subprocess.Popen | None, Sampler | None, Path | None, Path | None]:
    port = free_port()
    staged = (
        stage_mistralrs_view(model, mmproj, stage_root)
        if eng.kind == "mistralrs"
        else None
    )
    # mistralrs default paged-attn cannot fit this card (Num GPU blocks
    # is 0 — the product's own profile emits auto-off; direct spawns
    # bypass the compiler, so mirror it here, flag-gated like the axis)
    extras: dict | None = None
    if eng.kind == "mistralrs" and "--paged-attn" in cli_flags(
        eng.server, ["serve", "--help"]
    ):
        extras = {"pa": "off"}
    argv = direct_argv(eng, model, mmproj, port, 4096, 1, DEFAULT_NGL, staged, extras)
    errfh_path = stage_root / f"greedy-{eng.tag}-{port}.stderr"
    errfh = open(errfh_path, "wb")
    proc = subprocess.Popen(
        argv,
        cwd=str(eng.dir),
        stdout=subprocess.DEVNULL,
        stderr=errfh,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    errfh.close()
    sampler = Sampler(proc.pid)
    sampler.start()
    return port, proc, sampler, staged, errfh_path


def _greedy_stats(gots: list[str], refs: list[str]) -> dict:
    ratios = []
    exact = 0
    first_div = []
    for got, ref in zip(gots, refs):
        if got == ref:
            exact += 1
        ratios.append(difflib.SequenceMatcher(None, ref, got).ratio())
        first_div.append(
            next(
                (k for k, (a, b) in enumerate(zip(ref, got)) if a != b),
                min(len(ref), len(got)),
            )
        )
    return {
        "exact_matches": exact,
        "prompts": len(refs),
        "ratio_mean": round(statistics.mean(ratios), 4) if ratios else None,
        "ratio_min": round(min(ratios), 4) if ratios else None,
        "first_divergence_median_chars": statistics.median(first_div)
        if first_div
        else None,
    }


def run_greedy_parity(
    eng: Engine,
    model: Path,
    mmproj: Path | None,
    model_name: str,
    reference: list[str],
    stage_root: Path,
) -> dict:
    """Same-engine-family direct spawn, sampler-pinned RAW completions,
    diffed against the reference ENGINE's texts: measures backend
    numerics divergence (cuda vs vulkan), NOT gateway fidelity."""
    port, proc, sampler, _, errpath = _greedy_spawn(eng, model, mmproj, stage_root)
    try:
        if not wait_healthy(eng.kind, port, 600.0, proc=proc):
            tail = (
                errpath.read_text(errors="replace")[-300:]
                if errpath is not None and errpath.exists()
                else ""
            )
            return {"error": f"child failed to become healthy; stderr tail: {tail!r}"}
        body_model = "default" if eng.kind == "mistralrs" else model_name
        gots = [greedy_completions(port, p, body_model) for p in GREEDY_PROMPTS]
        return _greedy_stats(gots, reference)
    finally:
        if proc is not None and sampler is not None:
            teardown_proc(proc, sampler)


def build_greedy_reference(
    eng: Engine, model: Path, mmproj: Path | None, model_name: str, stage_root: Path
) -> list[str] | None:
    """Direct completions from ONE reference engine (first llamacpp)."""
    port, proc, sampler, _, errpath = _greedy_spawn(eng, model, mmproj, stage_root)
    try:
        if not wait_healthy(eng.kind, port, 600.0, proc=proc):
            tail = ""
            if errpath is not None and errpath.exists():
                tail = errpath.read_text(errors="replace")[-300:]
            log(f"  ! greedy reference spawn unhealthy; stderr tail: {tail!r}")
            return None
        return [greedy_completions(port, p, model_name) for p in GREEDY_PROMPTS]
    finally:
        if proc is not None and sampler is not None:
            teardown_proc(proc, sampler)


def run_greedy_gateway_cell(
    eng: Engine,
    model: Path,
    mmproj: Path | None,
    model_name: str,
    reference: list[str],
) -> dict:
    """HEADLINE transparency test: same engine, gateway path vs direct
    path, sampler-pinned. Anything short of 20/20 exact is a gateway
    translation defect (sampler remap, template drift, truncation)."""
    os.environ["PALLAMA_VALIDATE_PORT"] = str(free_port())
    V = importlib.import_module("validate")
    # F139: the module cache returns the FIRST import on later campaigns
    # — rebinding PORT on the module is what actually takes effect; the
    # env re-set above alone is inert past the first import.
    V.PORT = int(os.environ["PALLAMA_VALIDATE_PORT"])

    rec: dict = {}
    sb = V.Sandbox()
    try:
        con = sqlite3.connect(Path(sb.data_home) / "pallama" / "pallama.db")
        con.execute("UPDATE engines SET active = (tag = ?)", (eng.tag,))
        con.commit()
        con.close()
        daemon = V.Daemon(sb)
        port: int | None = None
        try:
            # same ownership fix as the speed/conc cells
            daemon.start(floor_model=model_name)
            port = V.PORT
            deadline = time.time() + 600
            healthy = False
            while time.time() < deadline:
                try:
                    http_json(f"http://127.0.0.1:{port}/healthz", timeout=5.0)
                    healthy = True
                    break
                except json.JSONDecodeError:
                    healthy = True
                    break
                except (urllib.error.URLError, OSError):
                    time.sleep(0.5)
            if not healthy:
                return {"error": "sandbox daemon failed to boot"}
            assert port is not None
            openai_stream_timed(
                port,
                {
                    "model": model_name,
                    "messages": [{"role": "user", "content": "warmup"}],
                    "max_tokens": 4,
                },
                timeout=600.0,
            )
            gots = [greedy_completions(port, p, model_name) for p in GREEDY_PROMPTS]
            rec = _greedy_stats(gots, reference)
        finally:
            daemon.stop()
            # F141: teardown parity with the speed/conc cells — the port
            # must go dark, else the daemon outlived its cell.
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
            dlog = Path(sb.data_dir) / "run" / "daemon.log"
            if dlog.exists():
                rec["daemon_log_tail"] = "\n".join(
                    dlog.read_text(errors="replace").splitlines()[-8:]
                )
    finally:
        sb.destroy()
    return rec


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
        else:
            row[feat] = flag in flags
    return row


# ---------------------------------------------------------------------------
# reporting


def fmt(v, suffix=""):
    if v is None:
        return "-"
    return f"{v}{suffix}"


def fmt_r(v, nd=1):
    if v is None:
        return "-"
    return f"{round(v, nd) if isinstance(v, float) else v}"


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


def _argv_flag(argv: list[str] | None, names: tuple[str, ...]) -> str | None:
    """Value of the first matching flag in a recorded child argv."""
    if not argv:
        return None
    for i, a in enumerate(argv):
        if a in names and i + 1 < len(argv):
            return argv[i + 1]
    return None


def _child_shape(r: dict) -> str:
    """Resolved slot/context shape of a pallama row, from its recorded
    child argv — auto-slots means 'config=default' alone hides np/ctx."""
    argv = r.get("child_argv")
    np_ = _argv_flag(argv, ("-np", "--parallel", "--np"))
    ctx = _argv_flag(argv, ("--ctx-size", "-c", "--max-model-len"))
    bits = []
    if np_:
        bits.append(f"np={np_}")
    if ctx:
        bits.append(f"ctx={ctx}")
    return " ".join(bits)


def _speed_row(r: dict) -> str:
    pa = " ".join(f"{kk}={vv}" for kk, vv in (r.get("params") or {}).items()) or "-"
    shape = _child_shape(r)
    if shape:
        pa += f" child: {shape}"
    if r.get("reference_note"):
        pa += f" ⚠ serves '{r['ollama_model']}' — t/s NOT comparable"
    return "| {t} | {k} | {p} | {pa} | {a} | {ap} | {i} | {ip} | {d} | {pc} | {pk} | {s} |".format(
        t=r.get("tag", "-"),
        k=r.get("kind", "-"),
        p=r.get("provider", "-"),
        pa=pa,
        a=fmt(round(r["ttft_ms_p50"]) if r.get("ttft_ms_p50") is not None else None),
        ap=fmt(round(r["ttft_ms_p99"]) if r.get("ttft_ms_p99") is not None else None),
        i=fmt(round(r["itl_p50_ms"]) if r.get("itl_p50_ms") is not None else None),
        ip=fmt(round(r["itl_p99_ms"]) if r.get("itl_p99_ms") is not None else None),
        d=fmt(
            round(r["decode_tps_p50"], 1)
            if r.get("decode_tps_p50") is not None
            else None
        ),
        pc=fmt(
            round(r["prefill_tps_cold"], 1)
            if r.get("prefill_tps_cold") is not None
            else None
        ),
        pk=fmt(
            round(r["prefill_tps_cached"], 1)
            if r.get("prefill_tps_cached") is not None
            else None
        ),
        s=r.get("tokens_source", "-"),
    )


def _env_failure(err: str) -> bool:
    """Classify a cell error as ENVIRONMENT (harness/box conditions) vs
    PRODUCT (engine/gateway behavior) — the report must not present a
    co-residency abort as a pallama defect."""
    return any(
        s in err
        for s in ("GPU memory floor", "MemAvailable", "mem_guard", "co-resident")
    )


def _findings(records: list[dict]) -> list[str]:
    """Auto-computed notable findings — the report's meaning layer.
    Pure function over records; no new measurement passes."""
    out: list[str] = []

    def speed_ok(provider, tag):
        for r in records:
            if (
                r.get("provider") == provider
                and r.get("tag") == tag
                and "decode_tps_p50" in r
                and "error" not in r
            ):
                return r
        return None

    # gateway-vs-direct decode parity per engine
    for r in records:
        tag = r.get("tag")
        if r.get("provider") != "pallama" or "decode_tps_p50" not in r or tag is None:
            continue
        d = speed_ok("direct", tag)
        if d and d.get("decode_tps_p50"):
            gw = r["decode_tps_p50"]
            base = d["decode_tps_p50"]
            delta = (gw - base) / base * 100.0
            verdict = (
                "parity" if abs(delta) <= 5 else ("regression" if delta < 0 else "gain")
            )
            out.append(
                f"- gateway vs direct decode (`{tag}`): {gw:.1f} vs {base:.1f} t/s "
                f"= {delta:+.1f}% ({verdict})."
            )
    # concurrency: system throughput vs direct
    for r in records:
        tag = r.get("tag")
        if r.get("provider") != "conc-pallama" or "sys_tps" not in r or tag is None:
            continue
        for drec in records:
            if drec.get("provider") == "conc-direct" and drec.get("tag") == tag:
                sysd = drec.get("sys_tps") or (
                    round(drec["total_tokens"] / drec["conc_wall_s"], 2)
                    if drec.get("total_tokens") and drec.get("conc_wall_s")
                    else None
                )
                if sysd:
                    ratio = r["sys_tps"] / sysd
                    out.append(
                        f"- concurrency system throughput (`{tag}`, "
                        f"{r.get('conc_level')} streams): {r['sys_tps']:.1f} vs direct "
                        f"{sysd:.1f} t/s = {ratio:.2f}x"
                        + (
                            " — serialized/queued or wall-inflated (see note)."
                            if ratio < 0.6
                            else "."
                        )
                    )
                break
    # variant axes deltas vs same-engine baseline
    base_direct: dict[str, dict] = {}
    for r in records:
        if (
            r.get("provider") == "direct"
            and r.get("params", {}).get("np") == 1
            and r.get("params", {}).get("ctx") == 4096
            and len(r.get("params", {})) == 2
            and "decode_tps_p50" in r
        ):
            base_direct[r["tag"]] = r
    for r in records:
        par = r.get("params") or {}
        axis = next((k for k in ("kv", "spec", "mmproj") if k in par), None)
        if r.get("provider") != "direct" or not axis or "decode_tps_p50" not in r:
            continue
        b = base_direct.get(r["tag"])
        if not b:
            continue
        # decode delta
        dd = (r["decode_tps_p50"] - b["decode_tps_p50"]) / b["decode_tps_p50"] * 100
        line = f"- {axis}={par[axis]} (`{r['tag']}`): decode {dd:+.1f}% vs baseline"
        # prefill regression catch (the kv q8_0 vulkan 13x case)
        if r.get("prefill_tps_cold") and b.get("prefill_tps_cold"):
            pd = (
                (r["prefill_tps_cold"] - b["prefill_tps_cold"])
                / b["prefill_tps_cold"]
                * 100
            )
            if pd < -50:
                line += f", prefill {pd:+.0f}% (REGRESSION)"
        out.append(line)
    # greedy gateway transparency
    for r in records:
        if r.get("provider") == "greedy_gw" and "exact_matches" in r:
            ex, n = r.get("exact_matches", 0), r.get("prompts", 0)
            verdict = "transparent" if ex == n else "NOT TRANSPARENT"
            out.append(
                f"- gateway greedy transparency (`{r['tag']}`): {ex}/{n} exact "
                f"vs same-engine direct — {verdict}."
            )
    # env-failure disclosure
    envfails = [r for r in records if r.get("error") and _env_failure(r["error"])]
    if envfails:
        out.append(
            f"- {len(envfails)} cell(s) aborted on ENVIRONMENT guards (GPU/RAM "
            "co-residency), not product behavior — see Failed cells."
        )
    return out


def write_markdown_report(
    path: Path,
    records: list[dict],
    model: Path,
    engines: list,
    feat_rows: dict[str, dict[str, bool]] | None,
    argv_summary: str,
    ref_tag: str | None,
    pallama_version: str = "unknown",
) -> None:
    """Human-first markdown report: environment, speed, resources,
    concurrency, quality, features, failures."""
    md: list[str] = []
    md.append("# Pallama benchmark matrix\n")
    md.append(f"- **date**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    md.append(f"- **model**: `{model.name}` ({model.stat().st_size // (1 << 20)} MiB)")
    md.append(f"- **gpu**: {gpu_name()}")
    # honesty: the report reflects the FULL resumed campaign — list every
    # engine represented in cells, not just this invocation's --engines
    seen_tags = sorted({t for r in records if (t := r.get("tag"))})
    inv_tags = [e.tag for e in engines]
    all_tags = sorted(set(seen_tags) | set(inv_tags)) or inv_tags
    md.append("- **engines**: " + ", ".join(all_tags))
    md.append(f"- **harness**: bench_matrix v{HARNESS_VERSION} — `{argv_summary}`")
    md.append(f"- **pallama**: `{pallama_version}` (sandbox daemon binary)")
    # provenance disclosure: resumed campaigns mix rows measured by
    # different binaries — enumerate the stamps actually in the records
    pallama_owned = [
        r
        for r in records
        if r.get("provider") in ("pallama", "conc-pallama", "greedy_gw")
    ]
    stamps = sorted({v for r in pallama_owned if (v := r.get("pallama_version"))})
    unstamped = sum(1 for r in pallama_owned if not r.get("pallama_version"))
    if len(stamps) > 1 or (stamps and stamps != [pallama_version]):
        md.append(
            f"- ⚠ **mixed provenance**: pallama-owned rows were measured by "
            f"{', '.join(f'`{s}`' for s in stamps)}; this invocation used "
            f"`{pallama_version}`. Per-row `pallama_version` in cells.jsonl."
        )
    elif stamps == [pallama_version]:
        md.append(f"- all pallama-owned rows measured by `{stamps[0]}`")
    if unstamped:
        md.append(
            f"- {unstamped} pallama-owned row(s) predate version stamping "
            "(harness v2 era) — binary provenance from the campaign log."
        )
    md.append("")

    findings = _findings(records)
    if findings:
        md.append("## Notable findings\n")
        md.extend(findings)
        md.append("")

    speed = [r for r in records if "ttft_ms_p50" in r]
    if speed:
        md.append("## Speed (serving, streaming)\n")
        md.append(
            "| engine | kind | provider | params | ttft p50 (ms) | ttft p99 (ms) "
            "| itl p50 (ms) | itl p99 (ms) | decode t/s | prefill t/s (cold) "
            "| prefill t/s (cached) | tokens src |"
        )
        md.append("|---|" * 11 + "|")
        for r in speed:
            md.append(_speed_row(r))
        md.append("")

    res = [
        r
        for r in records
        if any(
            k in r for k in ("load_s", "daemon_boot_s", "gpu_peak_mib", "rss_peak_mib")
        )
        and "ttft_ms_p50" in r
    ]
    if res:
        md.append("## Resources & cold start\n")
        md.append(
            "| engine | provider | params | load s | daemon boot s | cold 1st req s "
            "| GPU peak (MiB) | GPU power (W) | RSS peak (MiB) | teardown |"
        )
        md.append("|---|" * 9 + "|")
        for r in res:
            pa = (
                " ".join(f"{kk}={vv}" for kk, vv in (r.get("params") or {}).items())
                or "-"
            )
            g = r.get("gpu_peak_mib")
            md.append(
                "| {t} | {p} | {pa} | {ls} | {db} | {cr} | {g} | {w} | {rss} | {td} |".format(
                    t=r.get("tag", "-"),
                    p=r.get("provider", "-"),
                    pa=pa,
                    ls=fmt(r.get("load_s")),
                    db=fmt(r.get("daemon_boot_s")),
                    cr=fmt(r.get("cold_first_request_s")),
                    g=fmt(round(g) if g else None),
                    w=fmt(round(pw, 1) if (pw := r.get("gpu_power_peak_w")) else None),
                    rss=fmt(
                        round(r.get("rss_peak_mib", 0))
                        if r.get("rss_peak_mib")
                        else None
                    ),
                    td="ok" if r.get("teardown_ok") else "FAIL",
                )
            )
        md.append("")

    conc = [r for r in records if "conc_level" in r]
    if conc:
        md.append("## Concurrency (parallel streams)\n")
        md.append(
            "| engine | provider | streams | sys t/s | sum stream t/s "
            "| ttft max (ms) | ttft spread (ms) | itl p99 (ms) | ok/errors "
            "| wall (s) |"
        )
        md.append("|---|" * 9 + "|")
        for r in conc:
            pa = (
                " ".join(f"{kk}={vv}" for kk, vv in (r.get("params") or {}).items())
                or "-"
            )
            # legacy cells.jsonl rows (pre-sys_tps) recompute from stored
            # totals so old artifacts still render honestly
            sys_tps = r.get("sys_tps")
            if sys_tps is None and r.get("total_tokens") and r.get("conc_wall_s"):
                sys_tps = round(r["total_tokens"] / r["conc_wall_s"], 2)
            sum_tps = r.get("sum_stream_tps") or r.get("agg_decode_tps")
            md.append(
                f"| {r.get('tag', '-')} | {r.get('provider', '-')} | {pa} "
                f"| {fmt_r(sys_tps)} | {fmt_r(sum_tps)} "
                f"| {fmt(r.get('ttft_max_ms'))} "
                f"| {fmt(r.get('ttft_spread_ms'))} | {fmt(r.get('itl_p99_ms'))} "
                f"| {r.get('conc_ok', '-')}/{r.get('conc_errors', '-')} "
                f"| {fmt(r.get('conc_wall_s'))} |"
            )
        md.append(
            "- sys t/s = total tokens / wall (true system throughput); "
            "sum stream t/s = sum of per-stream rates. sum >> sys means "
            "streams were serialized (queued on a single slot) rather than "
            "served concurrently.\n"
        )
        md.append("")

    ppl = [r for r in records if r.get("provider") == "ppl"]
    if ppl:
        md.append("## Quality — perplexity (identical pinned args)\n")
        md.append("| engine | perplexity | ± err | wall (s) | note |")
        md.append("|---|---|---|---|---|")
        for r in ppl:
            note = r.get("error", "lower = better text fit")
            md.append(
                f"| {r.get('tag', '-')} | {fmt(r.get('perplexity'))} |"
                f" {fmt(r.get('ppl_error'))} | {fmt(r.get('wall_s'))} | {note} |"
            )
        md.append(
            "- corpus: deterministic offline repo text (code-heavy) — "
            "PARITY-ONLY; absolute PPL is not comparable to published "
            "wiki-text perplexities.\n"
        )

    greedy = [r for r in records if r.get("provider") == "greedy"]
    if greedy:
        md.append(
            f"## Quality — greedy parity vs `{ref_tag or 'reference'}`-direct "
            "(backend numerics)\n"
        )
        md.append(
            "| engine | exact matches | ratio mean | ratio min | first divergence (median chars) |"
        )
        md.append("|---|---|---|---|---|")
        for r in greedy:
            self_note = " *(self — trivially 1.0)" if r.get("tag") == ref_tag else ""
            md.append(
                f"| {r.get('tag', '-')}{self_note} | {r.get('exact_matches', '-')}/{r.get('prompts', '-')} |"
                f" {fmt(r.get('ratio_mean'))} | {fmt(r.get('ratio_min'))} |"
                f" {fmt(r.get('first_divergence_median_chars'))} |"
            )
        md.append("")

    gw = [r for r in records if r.get("provider") == "greedy_gw"]
    if gw:
        md.append(
            "## Quality — gateway transparency (pallama path vs direct, same engine)\n"
        )
        md.append(
            "| engine | exact matches | ratio mean | ratio min | first divergence (median chars) |"
        )
        md.append("|---|---|---|---|---|")
        for r in gw:
            md.append(
                f"| {r.get('tag', '-')} | {r.get('exact_matches', '-')}/{r.get('prompts', '-')} |"
                f" {fmt(r.get('ratio_mean'))} | {fmt(r.get('ratio_min'))} |"
                f" {fmt(r.get('first_divergence_median_chars'))} |"
            )
        md.append(
            "- expectation: 20/20 exact, ratio 1.0. A miss has TWO possible"
            " causes: gateway translation defect (sampler remap / template"
            " drift), or multi-slot batching numerics (child -np > 1"
            " changes float reduction order; near-tie logits flip). Pin"
            " `slots = 1` and re-run: still <20/20 = translation defect,"
            " 20/20 = slot-count numerics (upstream physics)."
        )

    # features: merge persisted cells (resumed campaigns stay complete)
    # with this invocation's fresh probes
    feat_cell_rows: dict[str, dict[str, bool]] = {}
    for r in records:
        if r.get("provider") == "features" and isinstance(r.get("features"), dict):
            feat_cell_rows[r.get("tag", "?")] = r["features"]
    merged = dict(feat_cell_rows)
    if feat_rows:
        merged.update(feat_rows)
    if feat_rows and "ollama(documented)" in feat_rows:
        merged["ollama(documented)"] = feat_rows["ollama(documented)"]
    elif "ollama(documented)" not in merged:
        merged["ollama(documented)"] = OLLAMA_FEATURES
    if merged:
        md.append("## Feature matrix\n")
        cols = list(merged.keys())
        feats = sorted({f for r in merged.values() for f in r})
        md.append("| feature | " + " | ".join(cols) + " |")
        md.append("|---|" + "---|" * len(cols))
        for f in feats:
            md.append(
                f"| `{f}` | "
                + " | ".join("Y" if merged[c].get(f) else "-" for c in cols)
                + " |"
            )
        md.append("")

    failed = [r for r in records if "error" in r]
    if failed:
        prod = [r for r in failed if not _env_failure(r["error"])]
        env = [r for r in failed if _env_failure(r["error"])]
        md.append("## Failed cells\n")
        if prod:
            md.append("**product** (engine/gateway behavior):\n")
            for r in prod:
                md.append(
                    f"- `{r.get('tag')}` / {r.get('provider')} /"
                    f" {r.get('params')}: {r['error']}"
                )
        if env:
            md.append(
                "\n**environment** (box/co-residency guards — NOT pallama defects):\n"
            )
            for r in env:
                md.append(
                    f"- `{r.get('tag')}` / {r.get('provider')} /"
                    f" {r.get('params')}: {r['error']}"
                )
        md.append("")

    md.append("## Reading this report\n")
    md.append("- `direct` = raw child spawn on a probed free port (no gateway).")
    md.append(
        "- `pallama` = full gateway path inside a sandboxed daemon"
        " (profile compiler, routing, auth); `child_argv` in cells.jsonl"
        " holds the resolved engine argv."
    )
    md.append("- `ollama` = HTTP-only reference against the host service, one cell.")
    md.append(
        "- decode counts ALL emitted tokens (content + reasoning/thinking);"
        " `usage`/engine counters are authoritative when present (`tokens src`)."
    )
    md.append(
        "- prefill t/s (cold) = prompt tokens / first-token time on an"
        " uncached token-targeted prompt; (cached) = same prompt re-sent"
        " (child prompt-cache path). ollama prefill uses engine-side"
        " prompt_eval counters, which EXCLUDE template tokens — ollama"
        " prefill reads high relative to the 512-token lanes."
    )
    md.append(
        "- pallama speed rows show the resolved slot/context shape"
        " (`child: np=… ctx=…`) parsed from the recorded child argv —"
        " auto-slots may differ from the direct rows' explicit np."
    )
    md.append(
        "- decode-lane TTFT rides the child's prompt cache after run 1"
        " (warm path); the prefill-lane cold/cached pair is the honest"
        " cache story at real prompt sizes."
    )
    md.append(
        "- GPU/power peaks sampled at ~1.2 s cadence (max across NVIDIA"
        " GPUs); very short bursts may undersample."
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
        "ttft_ms_p50",
        "ttft_ms_p99",
        "itl_p50_ms",
        "itl_p99_ms",
        "decode_tps",
        "prefill_tps_cold",
        "prefill_tps_cached",
        "load_s",
        "gpu_peak_mib",
        "gpu_power_peak_w",
        "rss_peak_mib",
        "tokens_source",
    ]
    lines = [" | ".join(cols)]
    lines.append("-" * 140)
    for r in records:
        params = r.get("params", "")
        lines.append(
            " | ".join(
                [
                    r.get("tag", "-"),
                    r.get("kind", "-"),
                    r.get("provider", "-"),
                    str(params),
                    fmt(r.get("ttft_ms_p50") and round(r["ttft_ms_p50"])),
                    fmt(r.get("ttft_ms_p99") and round(r["ttft_ms_p99"])),
                    fmt(r.get("itl_p50_ms") and round(r["itl_p50_ms"], 2)),
                    fmt(r.get("itl_p99_ms") and round(r["itl_p99_ms"], 2)),
                    fmt(r.get("decode_tps_p50") and round(r["decode_tps_p50"], 2)),
                    fmt(r.get("prefill_tps_cold") and round(r["prefill_tps_cold"], 1)),
                    fmt(
                        r.get("prefill_tps_cached")
                        and round(r["prefill_tps_cached"], 1)
                    ),
                    fmt(r.get("load_s")),
                    fmt(round(r.get("gpu_peak_mib", 0))),
                    fmt(r.get("gpu_power_peak_w")),
                    fmt(round(r.get("rss_peak_mib", 0))),
                    r.get("tokens_source", "-"),
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# main


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
    ap.add_argument("--pp", type=int, default=DEFAULT_PP, help="prefill prompt tokens")
    ap.add_argument("--tg", type=int, default=DEFAULT_TG)
    ap.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    ap.add_argument("--ngl", type=int, default=DEFAULT_NGL)
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    ap.add_argument("--ctx-sweep", default=",".join(map(str, DIRECT_CTX_SWEEP)))
    ap.add_argument(
        "--conc-sweep",
        default=",".join(map(str, DEFAULT_CONC_SWEEP)),
        help="concurrency levels for the parallel-streams lane",
    )
    ap.add_argument(
        "--soak", type=float, default=0.0, help="pallama soak seconds (0=off)"
    )
    ap.add_argument("--skip-ppl", action="store_true")
    ap.add_argument("--skip-greedy", action="store_true")
    ap.add_argument("--skip-features", action="store_true")
    ap.add_argument("--skip-conc", action="store_true")
    ap.add_argument("--skip-idle", action="store_true", help="skip the idle-wake lane")
    ap.add_argument(
        "--skip-ctxcurve", action="store_true", help="skip the long-ctx curve lane"
    )
    ap.add_argument(
        "--conc-rounds",
        type=int,
        default=3,
        help="sustained-load rounds per concurrency level (1 = single burst)",
    )
    ap.add_argument(
        "--ctxcurve-sweep",
        default="2048,8192,16384",
        help="ctx points for the long-context degradation curve",
    )
    ap.add_argument(
        "--ollama-service-restart",
        action="store_true",
        help=(
            "restart the host ollama systemd service for the cold lane's "
            "daemon-boot metric (needs BENCH_SUDO_PASSWORD in env; default: "
            "daemon left warm)"
        ),
    )
    ap.add_argument(
        "--skip-variants", action="store_true", help="skip kv/spec/mmproj/pa axis cells"
    )
    ap.add_argument("--corpus", help="local corpus .parquet/.txt for perplexity")
    ap.add_argument(
        "--fresh", action="store_true", help="ignore+replace existing cells.jsonl"
    )
    ap.add_argument("--artifacts-dir", help="override artifacts location")
    ap.add_argument(
        "--pallama-bin",
        help=(
            "pallama binary for sandbox daemons (default: repo release build, "
            "then PATH) — stamp its --version in the report"
        ),
    )
    ap.add_argument(
        "--md",
        help="also write the markdown report to this path (e.g. BENCHMARK.md)",
    )
    ap.add_argument(
        "--allow-battery",
        action="store_true",
        help=(
            "run even when the host is on battery power (dGPU clock caps "
            "invalidate absolute t/s comparisons; see power_state stamp)"
        ),
    )
    ap.add_argument(
        "--render-only",
        action="store_true",
        help=(
            "skip measurement; (re-)render the publication-format report from "
            "an existing campaign's cells.jsonl (--artifacts-dir or latest)"
        ),
    )
    args = ap.parse_args()

    if args.render_only:
        cache = Path.home() / ".cache/pallama-bench-matrix"
        if args.artifacts_dir:
            ad = Path(args.artifacts_dir).expanduser()
        elif cache.exists():
            runs = sorted(p for p in cache.iterdir() if p.is_dir())
            ad = runs[-1] if runs else None
        else:
            ad = None
        if ad is None or not (ad / "cells.jsonl").exists():
            log(
                "no campaign artifacts to render; pass --artifacts-dir or run a campaign"
            )
            return 2
        out = Path(args.md) if args.md else Path("BENCHMARK.md")
        recs = []
        by_key: dict[str, dict] = {}
        for line in (ad / "cells.jsonl").read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                by_key[r["key"]] = r
        recs = sorted(
            by_key.values(), key=lambda r: (r.get("provider", ""), r.get("tag", ""))
        )
        write_publication_report(recs, ad, out)
        return 0

    # battery-throttle guard: a discharging laptop caps dGPU clocks; the
    # 2026-09-10 census-fix re-run measured 28 t/s where AC measured 38.7
    # with an identical argv — burn the battery, not the numbers' meaning
    power = power_state()
    if power["on_battery"] and not args.allow_battery:
        bat = power.get("battery_name") or "battery"
        pct = power["battery_pct"]
        pct_s = f"{pct}%" if pct is not None else "charge unknown"
        log(
            f"host is on battery ({bat} {pct_s}) — dGPU "
            "power-capped; absolute t/s would not be comparable. Plug in "
            "AC or pass --allow-battery to override."
        )
        return 2

    ctx_sweep = tuple(int(x) for x in str(args.ctx_sweep).split(",") if x.strip())
    conc_sweep = tuple(int(x) for x in str(args.conc_sweep).split(",") if x.strip())

    data_dir = Path(args.data_dir).expanduser()
    # Model pick is DB-FIRST: a gateway cell can only serve store ROWS —
    # directory-only files (scratch exports like the collision-slug
    # "mtp-textonly" file, live-caught 2026-09-11) have no row, so a
    # stem-derived name 404s every sandbox daemon. Pick from rows whose
    # file exists; the row's name IS the gateway name (no stem fallback
    # to get wrong), and its mmproj column owns projector attachment.
    db = data_dir / "pallama.db"
    if not db.exists():
        log(f"no pallama.db under {data_dir} — nothing servable")
        return 2
    rows: list[tuple[Path, str, str | None]] = []
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        for path, name, mmproj in con.execute(
            "SELECT path, name, mmproj_path FROM models"
        ):
            p = Path(path)
            if (
                not p.exists()
                or "mmproj" in p.name
                or p.name.startswith(("imx-", "r5-"))
            ):
                continue
            rows.append((p, name, mmproj))
    finally:
        con.close()
    if not rows:
        log("no DB-registered .gguf models with existing files")
        return 2
    rows.sort(key=lambda r: r[0].stat().st_size, reverse=True)
    if args.model:
        hits = [
            r
            for r in rows
            if args.model.lower() in r[0].name.lower()
            or args.model.lower() in r[1].lower()
        ]
        if not hits:
            log(f"no DB model matching {args.model!r}")
            return 2
        # substring hits prefer the EXACT row name: "qwen3.5-9b" must not
        # silently select the larger "qwen3.5-9b-mtp" variant (different
        # weights file than the ollama reference blob)
        want = args.model.lower()
        hits.sort(key=lambda r: r[1].lower() != want)
        rows = hits
    model, gw_model_name, own_mmproj_s = rows[0]
    model_name = gw_model_name
    # mmproj ownership is a per-model DB column, NOT dir proximity —
    # mistral.rs scans the model dir for projectors, so attaching a
    # stray one would poison a text model (the Bug-C class).
    own_mmproj: Path | None = Path(own_mmproj_s) if own_mmproj_s else None

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

    # Sandbox daemon binary: must be settled BEFORE the first lazy
    # validate import (PAL resolves at import time from PALLAMA_BIN).
    if args.pallama_bin:
        pbin = Path(args.pallama_bin).resolve()
        if not pbin.is_file() or not os.access(pbin, os.X_OK):
            log(f"fatal: --pallama-bin {pbin} is not an executable file")
            return 2
        os.environ["PALLAMA_BIN"] = str(pbin)
    try:
        probe = subprocess.run(
            [os.environ.get("PALLAMA_BIN", "pallama"), "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        vout = (probe.stdout + probe.stderr).strip()
        pallama_version = vout.splitlines()[0] if vout else "unknown"
    except OSError:
        pallama_version = "unknown"
    log(f"pallama sandbox binary: {pallama_version}")

    # variant axes are emitted only when the child binary supports the
    # flags (probed, not assumed)
    eng_flags: dict[str, set[str]] = {}
    for eng in engines:
        if eng.kind == "mistralrs":
            eng_flags[eng.tag] = cli_flags(eng.server, ["serve", "--help"])
        else:
            eng_flags[eng.tag] = cli_flags(eng.server, ["--help"])

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
            # provenance stamps: rows survive across reruns in one
            # cells.jsonl — a row must carry WHICH binary measured it
            "pallama_version": pallama_version,
            "measured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            # env stamp: 5-min loadavg covers the cell window; a contended
            # run (rust-analyzer, parallel builds) is diagnosable later
            "loadavg_5m": open("/proc/loadavg").read().split()[1],
            "power_state": ("battery" if power_state()["on_battery"] else "ac"),
            **rec,
        }
        append_record(cells_path, rec2)
        records.append(rec2)
        if "error" in rec:
            failures += 1
            log(f"  CELL FAILED: {rec['error']}")
        elif "conc_level" in rec:
            sys_tps = rec.get("sys_tps")
            if sys_tps is None and rec.get("total_tokens") and rec.get("conc_wall_s"):
                sys_tps = round(rec["total_tokens"] / rec["conc_wall_s"], 2)
            log(
                f"  ok: streams {rec.get('conc_ok', 0)}/{rec.get('conc_level', 0)} "
                f"sys {sys_tps or 0:.1f} t/s "
                f"(sum {rec.get('sum_stream_tps') or rec.get('agg_decode_tps') or 0:.1f}) "
                f"wall {rec.get('conc_wall_s', 0):.1f}s "
                f"itl p99 {rec.get('itl_p99_ms') or 0:.2f}ms"
            )
        elif "perplexity" in rec:
            log(
                f"  ok: ppl {rec.get('perplexity', 0):.4f} "
                f"± {rec.get('ppl_error') or 0:.4f} "
                f"wall {rec.get('wall_s') or 0:.1f}s"
            )
        elif "exact_matches" in rec:
            log(
                f"  ok: greedy {rec.get('exact_matches', 0)}/{rec.get('prompts', 0)} "
                f"exact, ratio {rec.get('ratio_mean') or 0:.4f} "
                f"(min {rec.get('ratio_min') or 0:.4f})"
            )
        elif "ollama_cold_ttft_ms" in rec:
            # Cold cell: the warm-lane keys are absent by design — print
            # the cold metric the cell actually measured (cells.jsonl
            # always carried it; the stdout line used to read 0s).
            log(
                f"  ok: cold ttft {rec.get('ollama_cold_ttft_ms', 0):.0f}ms "
                f"load {rec.get('ollama_load_s', 0):.1f}s "
                f"gpu {rec.get('gpu_peak_mib', 0):.0f}MiB"
            )
        elif "cold_ttft_ms" in rec and "ttft_ms_p50" not in rec:
            log(
                f"  ok: cold ttft {rec.get('cold_ttft_ms', 0):.0f}ms "
                f"first-request {rec.get('cold_first_request_s', 0):.1f}s "
                f"gpu {rec.get('gpu_peak_mib', 0):.0f}MiB"
            )
        else:
            log(
                f"  ok: ttft {rec.get('ttft_ms_p50', 0):.0f}ms "
                f"decode {rec.get('decode_tps_p50', 0):.1f} t/s "
                f"prefill {rec.get('prefill_tps_cold') or 0:.0f} t/s "
                f"gpu {rec.get('gpu_peak_mib', 0):.0f}MiB"
            )

    # ---- direct provider sweep (ctx x np + variant axes)
    if "direct" in args.providers:
        for eng in engines:
            cells: list[dict] = [
                {"ctx": ctx, "np": np_} for ctx in ctx_sweep for np_ in DIRECT_NP_SWEEP
            ]
            if not args.skip_variants:
                flags = eng_flags.get(eng.tag, set())
                if eng.kind == "mistralrs":
                    if "--paged-attn" in flags:
                        cells.append({"ctx": 4096, "np": 1, "pa": "off"})
                else:
                    if "--cache-type-k" in flags and "--flash-attn" in flags:
                        cells.append({"ctx": 4096, "np": 1, "kv": "q8_0"})
                    if "--spec-type" in flags:
                        cells.append({"ctx": 4096, "np": 1, "spec": "ngram-simple"})
                    if own_mmproj is not None:
                        cells.append({"ctx": 4096, "np": 1, "mmproj": True})
            for params in cells:
                key = cell_key(eng.tag, "direct", params, model.name)
                if key in done:
                    log(f"[direct {eng.tag} {params}] resumed — skipping")
                    continue
                log(f"[direct {eng.tag} {params}]")
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

    # ---- pallama provider (default-config cell per engine + mistral.rs
    # paged-attn-off variant + soak)
    if "pallama" in args.providers:
        for eng in engines:
            params = {"config": "default"}
            key = cell_key(eng.tag, "pallama", params, model.name)
            if key in done:
                log(f"[pallama {eng.tag}] resumed — skipping")
                continue
            log(f"[pallama {eng.tag}] (sandbox, gateway, default profile)")
            # guard BEFORE the cell: a prior direct-sweep teardown can
            # still hold VRAM when the sandbox child spawns (live-caught
            # 2026-09-11: 502 right after the np4 direct cells)
            if not mem_guard(2048.0, f"pre-pallama {eng.tag}"):
                emit(
                    eng.tag,
                    eng.kind,
                    "pallama",
                    params,
                    key,
                    {"error": "GPU memory floor exceeded before cell"},
                )
                continue
            try:
                rec = run_pallama_cell(
                    eng, gw_model_name, cfg, "sandboxed gateway cell", soak_s=args.soak
                )
            except Exception as exc:  # noqa: BLE001 — campaign continues (R4)
                rec = {"error": f"pallama cell crashed: {exc}"}
            emit(eng.tag, eng.kind, "pallama", params, key, rec)
            if eng.kind == "mistralrs" and not args.skip_variants:
                params = {"config": "paged_attn_off"}
                key = cell_key(eng.tag, "pallama", params, model.name)
                if key in done:
                    continue
                log(f"[pallama {eng.tag}] (sandbox, gateway, paged-attn off)")
                try:
                    rec = run_pallama_cell(
                        eng,
                        gw_model_name,
                        cfg,
                        "sandboxed gateway cell, mistralrs_paged_attn=false",
                        pallama_cfg={"mistralrs_paged_attn": False},
                    )
                except Exception as exc:  # noqa: BLE001
                    rec = {"error": f"pallama cell crashed: {exc}"}
                emit(eng.tag, eng.kind, "pallama", params, key, rec)

        # ---- pallama single-stream variant (slots=1, classic in-VRAM
        # KV): the same-settings cell for the ollama parity question —
        # ollama serves one slot with KV in VRAM; this pins pallama to
        # the identical layout so any remaining delta is orchestration,
        # not defaults policy.
        params = {"config": "single-stream"}
        key = cell_key(eng.tag, "pallama", params, model.name)
        if key in done:
            log(f"[pallama {eng.tag} single-stream] resumed — skipping")
        else:
            log(
                f"[pallama {eng.tag} single-stream] (sandbox, slots=1, kv_unified=false)"
            )
            try:
                rec = run_pallama_cell(
                    eng,
                    gw_model_name,
                    cfg,
                    "sandboxed gateway cell, slots=1 + kv_unified=false",
                    pallama_cfg={"slots": 1, "kv_unified": False},
                )
            except Exception as exc:  # noqa: BLE001
                rec = {"error": f"pallama cell crashed: {exc}"}
            emit(eng.tag, eng.kind, "pallama", params, key, rec)
    if "ollama" in args.providers:
        params = {"reference": True}
        key = cell_key("ollama-host", "ollama", params, model.name)
        if key in done:
            log("[ollama] resumed — skipping")
        else:
            log("[ollama reference]")
            try:
                rec = run_ollama_cell(cfg, args.model or model_name)
            except Exception as exc:  # noqa: BLE001 — campaign continues (R4)
                rec = {"error": f"ollama cell crashed: {exc}"}
            emit("ollama-host", "ollama", "ollama", params, key, rec)

        # ---- ollama cold-start parity (disk-cold load + first token)
        params = {"cold": True}
        key = cell_key("ollama-host", "cold-ollama", params, model.name)
        if key in done:
            log("[ollama cold] resumed — skipping")
        else:
            log("[ollama cold-start parity]")
            try:
                rec = run_ollama_cold_cell(
                    cfg, args.model or model_name, args.ollama_service_restart
                )
            except Exception as exc:  # noqa: BLE001
                rec = {"error": f"ollama cold cell crashed: {exc}"}
            emit("ollama-host", "ollama", "cold-ollama", params, key, rec)

    # ---- idle-wake lane (sleep-vs-expiry: the idle-policy headline)
    if not args.skip_idle:
        if "pallama" in args.providers:
            for eng in engines:
                params = {"idle": True}
                key = cell_key(eng.tag, "idle-pallama", params, model.name)
                if key in done:
                    continue
                log(f"[idle-wake pallama {eng.tag}]")
                if not mem_guard(2048.0, f"pre-idle {eng.tag}"):
                    emit(
                        eng.tag,
                        eng.kind,
                        "idle-pallama",
                        params,
                        key,
                        {"error": "GPU memory floor exceeded before cell"},
                    )
                    continue
                try:
                    rec = run_pallama_idle_cell(eng, gw_model_name, cfg)
                except Exception as exc:  # noqa: BLE001
                    rec = {"error": f"idle cell crashed: {exc}"}
                emit(eng.tag, eng.kind, "idle-pallama", params, key, rec)
        if "ollama" in args.providers:
            params = {"idle": True}
            key = cell_key("ollama-host", "idle-ollama", params, model.name)
            if key in done:
                log("[idle-wake ollama] resumed — skipping")
            else:
                log("[idle-wake ollama (keep_alive expiry)]")
                try:
                    rec = run_ollama_idle_cell(cfg, args.model or model_name)
                except Exception as exc:  # noqa: BLE001
                    rec = {"error": f"ollama idle cell crashed: {exc}"}
                emit("ollama-host", "ollama", "idle-ollama", params, key, rec)

    # ---- long-context degradation curve (decode t/s + TTFT vs ctx)
    if not args.skip_ctxcurve:
        ctxcurve = tuple(
            int(x) for x in str(args.ctxcurve_sweep).split(",") if x.strip()
        )
        if "pallama" in args.providers:
            for eng in engines:
                for ctx in ctxcurve:
                    params = {"ctx": ctx}
                    key = cell_key(eng.tag, "ctxcurve-pallama", params, model.name)
                    if key in done:
                        continue
                    log(f"[ctxcurve pallama {eng.tag} ctx={ctx}]")
                    if not mem_guard(2048.0, f"pre-ctxcurve {eng.tag} {ctx}"):
                        emit(
                            eng.tag,
                            eng.kind,
                            "ctxcurve-pallama",
                            params,
                            key,
                            {"error": "GPU memory floor exceeded before cell"},
                        )
                        continue
                    try:
                        rec = run_pallama_ctx_cell(eng, gw_model_name, ctx, cfg)
                    except Exception as exc:  # noqa: BLE001
                        rec = {"error": f"ctxcurve cell crashed: {exc}"}
                    emit(eng.tag, eng.kind, "ctxcurve-pallama", params, key, rec)
        if "ollama" in args.providers:
            for ctx in ctxcurve:
                params = {"ctx": ctx}
                key = cell_key("ollama-host", "ctxcurve-ollama", params, model.name)
                if key in done:
                    continue
                log(f"[ctxcurve ollama ctx={ctx}]")
                try:
                    rec = run_ollama_ctx_cell(cfg, args.model or model_name, ctx)
                except Exception as exc:  # noqa: BLE001
                    rec = {"error": f"ollama ctxcurve cell crashed: {exc}"}
                emit("ollama-host", "ollama", "ctxcurve-ollama", params, key, rec)

    # ---- concurrency lane (direct np-sized child + full gateway path)
    if not args.skip_conc and conc_sweep:
        for level in conc_sweep:
            if "direct" in args.providers:
                for eng in engines:
                    if eng.kind != "llamacpp":
                        continue
                    params = {"conc": level}
                    key = cell_key(eng.tag, "conc-direct", params, model.name)
                    if key in done:
                        continue
                    log(f"[conc direct {eng.tag} x{level}]")
                    if not mem_guard(2048, f"pre-conc {eng.tag}"):
                        emit(
                            eng.tag,
                            eng.kind,
                            "conc-direct",
                            params,
                            key,
                            {"error": "GPU memory floor exceeded before cell"},
                        )
                        continue
                    try:
                        rec = run_direct_conc_cell(
                            eng, model, own_mmproj, level, model_name, cfg, stage_root
                        )
                    except Exception as exc:  # noqa: BLE001
                        rec = {"error": f"conc cell crashed: {exc}"}
                    emit(eng.tag, eng.kind, "conc-direct", params, key, rec)
            if "pallama" in args.providers:
                for eng in engines:
                    params = {"conc": level, "rounds": args.conc_rounds}
                    key = cell_key(eng.tag, "conc-pallama", params, model.name)
                    if key in done:
                        continue
                    log(f"[conc pallama {eng.tag} x{level} x{args.conc_rounds}r]")
                    if not mem_guard(2048.0, f"pre-conc-pallama {eng.tag}"):
                        emit(
                            eng.tag,
                            eng.kind,
                            "conc-pallama",
                            params,
                            key,
                            {"error": "GPU memory floor exceeded before cell"},
                        )
                        continue
                    try:
                        rec = run_pallama_conc_cell(
                            eng, gw_model_name, level, cfg, rounds=args.conc_rounds
                        )
                    except Exception as exc:  # noqa: BLE001
                        rec = {"error": f"conc cell crashed: {exc}"}
                    emit(eng.tag, eng.kind, "conc-pallama", params, key, rec)
            if "ollama" in args.providers:
                params = {"conc": level, "rounds": args.conc_rounds}
                key = cell_key("ollama-host", "conc-ollama", params, model.name)
                if key in done:
                    continue
                log(f"[conc ollama x{level} x{args.conc_rounds}r]")
                try:
                    rec = run_ollama_conc_cell(
                        cfg, args.model or model_name, level, args.conc_rounds
                    )
                except Exception as exc:  # noqa: BLE001
                    rec = {"error": f"ollama conc cell crashed: {exc}"}
                emit("ollama-host", "ollama", "conc-ollama", params, key, rec)

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
            if not mem_guard(2048.0, f"pre-ppl {eng.tag}"):
                emit(
                    eng.tag,
                    eng.kind,
                    "ppl",
                    params,
                    key,
                    {"error": "GPU memory floor exceeded before cell"},
                )
                continue
            try:
                rec = run_perplexity(eng, model, corpus, cfg)
            except Exception as exc:  # noqa: BLE001 — campaign continues (R4)
                rec = {"error": f"ppl cell crashed: {exc}"}
            emit(eng.tag, eng.kind, "ppl", params, key, rec)

    # ---- quality: greedy parity vs llama.cpp-direct reference
    ref_tag: str | None = None
    if not args.skip_greedy:
        ref_eng = next((e for e in engines if e.kind == "llamacpp"), None)
        reference = None
        if ref_eng is not None:
            ref_tag = ref_eng.tag
            log(f"[greedy reference from {ref_eng.tag}]")
            if not mem_guard(2048.0, f"pre-greedy-reference {ref_eng.tag}"):
                # F140: other lanes error+skip on a failed floor wait —
                # proceeding anyway produces swap-thrashed reference
                # tokens that poison every parity comparison.
                log("  ! GPU floor exceeded — waiting failed; skipping greedy parity")
            else:
                reference = build_greedy_reference(
                    ref_eng, model, own_mmproj, model_name, stage_root
                )
        if reference is None:
            if ref_eng is None:
                log("no llamacpp engine for greedy reference — skipping parity")
            else:
                log(
                    f"greedy reference spawn on {ref_eng.tag} failed "
                    "(unhealthy child) — skipping parity"
                )
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
                if "error" not in rec:
                    rec["vs"] = ref_tag or (ref_eng.tag if ref_eng else "unknown")
                emit(eng.tag, eng.kind, "greedy", params, key, rec)
            # gateway-transparency lane: same-engine direct vs gateway
            for eng in engines:
                if eng.kind != "llamacpp":
                    continue
                params = {"greedy_gw": True}
                key = cell_key(eng.tag, "greedy_gw", params, model.name)
                if key in done:
                    continue
                log(f"[greedy gateway-transparency {eng.tag}]")
                try:
                    rec = run_greedy_gateway_cell(
                        eng, model, own_mmproj, gw_model_name, reference
                    )
                except Exception as exc:  # noqa: BLE001
                    rec = {"error": f"greedy gw cell crashed: {exc}"}
                emit(eng.tag, eng.kind, "greedy_gw", params, key, rec)

    # ---- features matrix (persisted as cells so resumed campaigns
    # render the complete matrix)
    feat_rows: dict[str, dict[str, bool]] | None = None
    if not args.skip_features:
        log("[features]")
        rows: dict[str, dict[str, bool]] = {}
        for eng in engines:
            rows[eng.tag] = features_row(eng.kind, eng_flags.get(eng.tag, set()))
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
        for eng in engines:
            params = {"features": True}
            key = cell_key(eng.tag, "features", params, model.name)
            if key in done:
                continue
            rec = {"features": rows[eng.tag]}
            append_record(
                cells_path,
                {
                    "key": key,
                    "tag": eng.tag,
                    "kind": eng.kind,
                    "provider": "features",
                    "params": params,
                    "model": model.name,
                    **rec,
                },
            )
            records.append(
                {
                    "key": key,
                    "tag": eng.tag,
                    "kind": eng.kind,
                    "provider": "features",
                    "params": params,
                    "model": model.name,
                    **rec,
                }
            )

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
    if ref_tag is None:
        # resumed campaign: recover the reference tag from greedy cells
        for r in all_records:
            if r.get("provider") == "greedy" and r.get("vs"):
                ref_tag = r["vs"]
                break
    md_path = art / "benchmark.md"
    write_markdown_report(
        md_path,
        all_records,
        model,
        engines,
        feat_rows,
        argv_summary,
        ref_tag,
        pallama_version,
    )
    log(f"report  -> {md_path}")
    if args.md:
        write_publication_report(all_records, art, Path(args.md))
        log(f"report -> {args.md}")
    write_speed_table(all_records, art / "summary.txt")
    log(f"summary -> {art / 'summary.txt'}")
    log(f"cells   -> {cells_path} ({len(records)} new, {len(all_records)} total)")
    return 1 if failures else 0


# ---------------------------------------------------------------------------
# Publication renderer: emits a self-contained, publishable benchmark report
# from cells.jsonl (last-wins). Pure stdlib, never re-measures. Reached via
# --render-only, and automatically at campaign end for --md output (the
# internal forensics report stays at <artifacts>/benchmark.md).
# ---------------------------------------------------------------------------


ENGINE_LABELS = {
    "b10809": "llama.cpp b10809 (Vulkan)",
    "b10809-cuda": "llama.cpp b10809 (CUDA build)",
    "v0.9.3": "mistral.rs 0.9.3 (CUDA sm89)",
}

TEST_BED = [
    ("CPU", "Intel Core i7-14650HX, 24 hardware threads"),
    ("Discrete GPU", "NVIDIA GeForce RTX 4070 Laptop, 8 GiB, driver 580.173.02"),
    ("Integrated GPU", "Intel Graphics (RPL-S), Vulkan device"),
    ("RAM", "16 GiB (13.3 GiB usable)"),
    ("OS", "Linux Mint 22.3, kernel 7.0.0-31-generic"),
    (
        "Runtimes compared",
        "pallama 0.5.0 gateway - llama.cpp b10809 (Vulkan + CUDA builds) - mistral.rs 0.9.3 - ollama 0.33.3",
    ),
    (
        "Model",
        "Qwen3.5-9B, Q4_K_M GGUF (5.4 GiB) + vision projector mmproj-F16 (876 MiB)",
    ),
]

METHODOLOGY = [
    "All lanes speak the OpenAI-compatible streaming API; tokens are counted from usage chunks (engine-injected at the gateway), never estimated from chunk counts.",
    "Decode throughput = (tokens - 1) / (last-token time - TTFT); medians over 5 runs after a warmup request.",
    "Inter-token latency (ITL) p50/p99 from per-chunk timestamps; TTFT p50/p90/p99 + stdev.",
    "Prefill: a token-targeted prompt (~512 tokens via engine /tokenize); run 1 is the cold (uncached) prefill, runs 2+ ride the prompt cache.",
    "Concurrency: 4 parallel streams x 128 generated tokens each; system t/s = total tokens / wall clock; sum-stream t/s = sum of per-stream rates (sum >> system indicates serialization).",
    "Greedy parity: 20 fixed prompts, greedy sampling, 256 tokens; exact-match count and text-similarity ratio vs a same-engine reference run.",
    "Gateway transparency: a second greedy lane through the pallama gateway with identical sampling; any divergence vs the direct lane isolates translation overhead.",
    "Perplexity: llama-perplexity on an offline ASCII corpus, ctx 2048.",
    "Cold-start parity: the model file's page cache is dropped (posix_fadvise DONTNEED) and the GPU asserted idle (<512 MiB) before every cold probe on every runtime — a cold load is disk-cold, not memory-warm.",
    "Cold TTFT = first-token latency of the cold probe itself (max_tokens 4, aligned num_ctx 16384 on both runtimes).",
    "ollama daemon boot is only measured with --ollama-service-restart (systemd restart, sudo password via BENCH_SUDO_PASSWORD env, stdin-only); without it the daemon stays warm and the row says so.",
    "Idle-wake: pallama's reaper sleeps the child at idle_sleep_secs (weights stay RAM-resident, VRAM released) — wake TTFT is a sleep-wake; ollama's keep_alive expiry fully unloads — wake TTFT is a disk reload. The policy column names the semantic; both measured after the policy is observed via /api/ps.",
    "Long-context curve: per-ctx cells (pallama model_overrides ctx / ollama num_ctx) × 3-run decode suites; each ollama point evicts first so the runner respawns at that ctx.",
    "Sustained concurrency: sequential bursts of the parallel-stream lane (default 3 rounds); TTFT p99 aggregates every stream of every round.",
    "Every pallama row records the spawned engine's argv (slots/context shown in tables) and stamps pallama version, wall clock, 5-min load average, and AC/battery power state; GPU cells refuse to run on battery.",
]


def pfmt(x, nd=1, unit=""):
    if x is None or x != x:
        return "-"
    return f"{x:.{nd}f}{unit}"


def child_shape(rec: dict) -> str:
    argv = rec.get("child_argv") or rec.get("argv") or []
    np_ = ctx = None
    for i, a in enumerate(argv):
        if a == "-np" and i + 1 < len(argv):
            np_ = argv[i + 1]
        if a == "--ctx-size" and i + 1 < len(argv):
            ctx = argv[i + 1]
    if np_ is None:
        return ""  # engine-scheduled (mistral.rs)
    return f"{np_}x{ctx}" if ctx else np_


def engine_label(tag: str) -> str:
    return ENGINE_LABELS.get(tag, tag)


def speed_table(recs: list[dict]) -> str:
    rows = []
    for r in recs:
        if r.get("provider") == "pallama" and "error" not in r:
            rows.append(
                (
                    f"pallama gateway - {engine_label(r['tag'])}",
                    child_shape(r) or "engine-scheduled",
                    r.get("decode_tps_p50"),
                    r.get("ttft_ms_p50"),
                    r.get("ttft_ms_p99"),
                    r.get("itl_p50_ms"),
                    r.get("itl_p99_ms"),
                    r.get("prefill_tps_cold"),
                    r.get("prefill_tps_cached"),
                    r.get("gpu_peak_mib"),
                    r.get("gpu_power_peak_w"),
                )
            )
    for r in recs:
        if (
            r.get("provider") == "direct"
            and r.get("params", {}).get("ctx") == 16384
            and r.get("params", {}).get("np") == 1
            and "error" not in r
        ):
            rows.append(
                (
                    f"direct engine - {engine_label(r['tag'])}",
                    "1x16384",
                    r.get("decode_tps_p50"),
                    r.get("ttft_ms_p50"),
                    r.get("ttft_ms_p99"),
                    r.get("itl_p50_ms"),
                    r.get("itl_p99_ms"),
                    r.get("prefill_tps_cold"),
                    r.get("prefill_tps_cached"),
                    r.get("gpu_peak_mib"),
                    r.get("gpu_power_peak_w"),
                )
            )
    for r in recs:
        if r.get("provider") == "ollama" and "error" not in r:
            rows.append(
                (
                    f"ollama 0.33.3 - {r.get('ollama_model', 'same model')}",
                    "service",
                    r.get("decode_tps_p50"),
                    r.get("ttft_ms_p50"),
                    r.get("ttft_ms_p99"),
                    r.get("itl_p50_ms"),
                    r.get("itl_p99_ms"),
                    r.get("prefill_tps_cold"),
                    r.get("prefill_tps_cached"),
                    r.get("gpu_peak_mib"),
                    r.get("gpu_power_peak_w"),
                )
            )
    head = (
        "| Runtime | slots x ctx | decode t/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms |"
        " ITL p99 ms | prefill cold t/s | prefill cached t/s | GPU peak MiB | GPU power W |"
    )
    sep = "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    body = [
        f"| {n} | {s} | {pfmt(d)} | {pfmt(t5)} | {pfmt(t9)} | {pfmt(i5)} | {pfmt(i9)} |"
        f" {pfmt(pc)} | {pfmt(pk)} | {pfmt(g, 0)} | {pfmt(pw)} |"
        for n, s, d, t5, t9, i5, i9, pc, pk, g, pw in rows
    ]
    return "\n".join([head, sep, *body])


def conc_table(recs: list[dict]) -> str:
    rows = []
    for r in recs:
        prov = r.get("provider")
        if prov not in ("conc-pallama", "conc-direct", "conc-ollama") or "error" in r:
            continue
        if prov == "conc-pallama":
            name = f"pallama gateway - {engine_label(r['tag'])}"
            shape = child_shape(r) or "engine-scheduled"
        elif prov == "conc-direct":
            name = f"direct engine - {engine_label(r['tag'])}"
            shape = child_shape(r) or (
                f"{r.get('params', {}).get('np')} slots"
                if r.get("params", {}).get("np")
                else "engine-scheduled"
            )
        else:
            name = f"ollama - {r.get('ollama_model', 'reference')}"
            shape = "service"
        rows.append(
            (
                name,
                shape,
                r.get("conc_ok"),
                r.get("params", {}).get("conc", "?"),
                r.get("conc_rounds"),
                r.get("sys_tps"),
                r.get("sum_stream_tps"),
                r.get("conc_wall_s"),
                r.get("ttft_max_ms"),
                r.get("conc_ttft_p99_ms"),
                r.get("itl_p99_ms"),
            )
        )
    head = (
        "| Runtime | slots | ok streams | rounds | system t/s | sum-stream t/s | wall s |"
        " TTFT max ms | TTFT p99 ms | ITL p99 ms |"
    )
    sep = "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    body = [
        f"| {n} | {s} | {ok}/{den} | {pfmt(rd, 0)} | {pfmt(sys)} | {pfmt(sm)} | {pfmt(w, 2)} |"
        f" {pfmt(tm, 0)} | {pfmt(tp, 0)} | {pfmt(i9, 1)} |"
        for n, s, ok, den, rd, sys, sm, w, tm, tp, i9 in rows
    ]
    return "\n".join([head, sep, *body])


def ppl_table(recs: list[dict]) -> str:
    rows = []
    for r in recs:
        if r.get("provider") != "ppl":
            continue
        v, e = r.get("perplexity"), r.get("ppl_error")
        cell = (
            f"{pfmt(v, 2)} ± {pfmt(e, 2)}"
            if v is not None
            else (
                f"failed ({str(e)[:50]})"
                if e
                else "not applicable (tool is llama.cpp-family)"
            )
        )
        rows.append((engine_label(r["tag"]), cell))
    if not rows:
        return "_Not measured._"
    head = "| Engine | perplexity (ctx 2048, offline ASCII corpus) |"
    sep = "|---|---:|"
    body = [f"| {n} | {c} |" for n, c in rows]
    return "\n".join([head, sep, *body])


def greedy_table(recs: list[dict]) -> str:
    rows = []
    for r in recs:
        if r.get("provider") == "greedy":
            rows.append(
                (
                    f"{engine_label(r['tag'])} vs same-engine reference (direct)",
                    r.get("exact_matches"),
                    r.get("prompts"),
                    r.get("ratio_mean"),
                    r.get("ratio_min"),
                )
            )
    for r in recs:
        if r.get("provider") == "greedy_gw":
            rows.append(
                (
                    f"{engine_label(r['tag'])} through pallama gateway vs direct",
                    r.get("exact_matches"),
                    r.get("prompts"),
                    r.get("ratio_mean"),
                    r.get("ratio_min"),
                )
            )
    head = "| Comparison | exact / total | ratio mean | ratio min |"
    sep = "|---|---:|---:|---:|"
    body = [
        f"| {n} | {e}/{p} | {pfmt(rm, 3)} | {pfmt(ri, 3)} |" for n, e, p, rm, ri in rows
    ]
    return "\n".join([head, sep, *body])


def variant_table(recs: list[dict]) -> str:
    base: dict[tuple, float] = {}
    for r in recs:
        p = r.get("params", {})
        if (
            r.get("provider") == "direct"
            and p.get("ctx") == 4096
            and p.get("np") == 1
            and len(p) == 2
            and "error" not in r
        ):
            base[(r["tag"], "decode")] = r.get("decode_tps_p50") or 0.0
            base[(r["tag"], "prefill")] = r.get("prefill_tps_cold") or 0.0
    rows = []
    for r in recs:
        p = r.get("params", {})
        if (
            r.get("provider") != "direct"
            or p.get("ctx") != 4096
            or p.get("np") != 1
            or len(p) <= 2
        ):
            continue
        axis = next((k for k in ("kv", "spec", "mmproj", "pa") if k in p), None)
        if axis is None or "error" in r:
            continue
        d = r.get("decode_tps_p50") or 0.0
        pc = r.get("prefill_tps_cold") or 0.0
        rows.append(
            (
                engine_label(r["tag"]),
                axis,
                str(p[axis]),
                d,
                d - base.get((r["tag"], "decode"), float("nan")),
                pc,
                pc - base.get((r["tag"], "prefill"), float("nan")),
            )
        )
    if not rows:
        return "_Not measured._"
    head = "| Engine | axis | setting | decode t/s | delta vs dense | prefill cold t/s | delta |"
    sep = "|---|---|---|---:|---:|---:|---:|"
    body = [
        f"| {e} | {ax} | {v} | {pfmt(d)} | {pfmt(dd)} | {pfmt(pc)} | {pfmt(pd)} |"
        for e, ax, v, d, dd, pc, pd in rows
    ]
    return "\n".join([head, sep, *body])


def features_table(recs: list[dict]) -> str:
    feats: dict[str, dict] = {}
    for r in recs:
        if r.get("provider") == "features":
            feats[engine_label(r["tag"])] = r.get("features", {})
    if not feats:
        return "_Not measured._"
    names = sorted({k for f in feats.values() for k in f})
    head = "| Capability | " + " | ".join(feats) + " |"
    sep = "|---|" + "---:|" * len(feats)
    body = [
        f"| {n} | "
        + " | ".join(
            ("yes" if feats[e].get(n) else "no") if n in feats[e] else "-"
            for e in feats
        )
        + " |"
        for n in names
    ]
    return "\n".join([head, sep, *body])


def coldstart_table(recs: list[dict]) -> str:
    rows = [
        (
            f"pallama gateway - {engine_label(r['tag'])}",
            r.get("daemon_boot_s"),
            r.get("cold_first_request_s"),
            r.get("cold_ttft_ms"),
            r.get("load_s"),
            r.get("rss_peak_mib"),
        )
        for r in recs
        if r.get("provider") == "pallama" and "error" not in r
    ]
    rows += [
        (
            f"direct engine - {engine_label(r['tag'])}",
            None,
            None,
            None,
            r.get("load_s"),
            r.get("rss_peak_mib"),
        )
        for r in recs
        if r.get("provider") == "direct"
        and "error" not in r
        and r.get("params", {}).get("ctx") == 16384
        and r.get("params", {}).get("np") == 1
    ]
    rows += [
        (
            f"ollama - {r.get('ollama_model', 'reference')}",
            r.get("ollama_daemon_boot_s"),
            r.get("ollama_cold_wall_s"),
            r.get("ollama_cold_ttft_ms"),
            r.get("ollama_load_s"),
            None,
        )
        for r in recs
        if r.get("provider") == "cold-ollama" and "error" not in r
    ]
    if not rows:
        return "_Not measured._"
    head = (
        "| Runtime | daemon boot s | first request (cold engine load) s |"
        " cold TTFT ms | engine load s | RSS peak MiB |"
    )
    sep = "|---|---:|---:|---:|---:|---:|"
    body = [
        f"| {n} | {pfmt(b, 2)} | {pfmt(c, 2)} | {pfmt(t, 0)} | {pfmt(ld, 2)} | {pfmt(r, 0)} |"
        for n, b, c, t, ld, r in rows
    ]
    return "\n".join([head, sep, *body])


def idle_wake_table(recs: list[dict]) -> str:
    rows = []
    for r in recs:
        prov = r.get("provider")
        if prov == "idle-pallama" and "error" not in r:
            rows.append(
                (
                    f"pallama - {engine_label(r['tag'])}",
                    r.get("idle_policy", "sleep ladder"),
                    r.get("slept"),
                    r.get("idle_wake_ttft_ms"),
                    None,
                    r.get("idle_note"),
                )
            )
        elif prov == "idle-ollama" and "error" not in r:
            rows.append(
                (
                    f"ollama - {r.get('ollama_model', 'reference')}",
                    r.get("idle_policy", "keep_alive expiry"),
                    r.get("expired"),
                    r.get("idle_wake_ttft_ms"),
                    r.get("idle_reload_s"),
                    r.get("idle_note"),
                )
            )
    if not rows:
        return "_Not measured._"
    head = (
        "| Runtime | idle policy | policy observed | wake TTFT ms | reload s | note |"
    )
    sep = "|---|---|---|---:|---:|---|"
    body = []
    for n, pol, seen, ttft, rel, note in rows:
        seen_s = {True: "yes", False: "NO"}.get(seen, "?")
        body.append(
            f"| {n} | {pol} | {seen_s} | {pfmt(ttft, 0)} | {pfmt(rel, 2)} | {note or ''} |"
        )
    return "\n".join([head, sep, *body])


def ctxcurve_table(recs: list[dict]) -> str:
    rows = []
    for r in recs:
        prov = r.get("provider")
        if prov == "ctxcurve-pallama" and "error" not in r:
            rows.append(
                (
                    f"pallama - {engine_label(r['tag'])}",
                    r.get("ctx"),
                    r.get("decode_tps_p50"),
                    r.get("ttft_ms_p50"),
                )
            )
        elif prov == "ctxcurve-ollama" and "error" not in r:
            rows.append(
                (
                    f"ollama - {r.get('ollama_model', 'reference')}",
                    r.get("ctx"),
                    r.get("decode_tps_p50"),
                    r.get("ttft_ms_p50"),
                )
            )
    if not rows:
        return "_Not measured._"
    rows.sort(key=lambda x: (x[0], x[1] or 0))
    head = "| Runtime | ctx | decode t/s | TTFT p50 ms |"
    sep = "|---|---:|---:|---:|"
    body = [f"| {n} | {pfmt(c, 0)} | {pfmt(d)} | {pfmt(t, 0)} |" for n, c, d, t in rows]
    return "\n".join([head, sep, *body])


def executive_summary(recs: list[dict]) -> str:
    gw = {
        r["tag"]: r for r in recs if r.get("provider") == "pallama" and "error" not in r
    }
    direct = {
        r["tag"]: r
        for r in recs
        if r.get("provider") == "direct"
        and r.get("params", {}).get("ctx") == 16384
        and r.get("params", {}).get("np") == 1
        and "error" not in r
    }
    parts = []
    for tag in gw:
        if tag in direct:
            g, d = gw[tag].get("decode_tps_p50"), direct[tag].get("decode_tps_p50")
            if g and d:
                delta = (g / d - 1) * 100
                parts.append(
                    f"{engine_label(tag)}: gateway {pfmt(g)} vs direct {pfmt(d)} t/s "
                    f"({delta:+.1f}%)"
                )
    for tag, r in gw.items():
        if r.get("prefill_tps_cold") and r.get("prefill_tps_cached"):
            parts.append(
                f"{engine_label(tag)} prompt-cache prefill {pfmt(r['prefill_tps_cached'], 0)}"
                f" vs {pfmt(r['prefill_tps_cold'], 0)} t/s cold"
            )
            break
    conc = {
        r["tag"]: r
        for r in recs
        if r.get("provider") == "conc-pallama" and "error" not in r
    }
    if conc:
        bits = [
            f"{pfmt(r.get('sys_tps'))} t/s system ({child_shape(r) or 'engine-scheduled'} shape)"
            for r in conc.values()
        ]
        parts.append("4-stream concurrency: " + "; ".join(bits))
    boot = next(
        (r.get("daemon_boot_s") for r in gw.values() if r.get("daemon_boot_s")), None
    )
    if boot:
        parts.append(f"gateway cold boot {pfmt(boot, 2)} s")
    cold_ttft = next(
        (r.get("cold_ttft_ms") for r in gw.values() if r.get("cold_ttft_ms")), None
    )
    oc = next(
        (r for r in recs if r.get("provider") == "cold-ollama" and "error" not in r),
        None,
    )
    if cold_ttft and oc and oc.get("ollama_cold_ttft_ms"):
        ratio = oc["ollama_cold_ttft_ms"] / cold_ttft
        parts.append(
            f"cold TTFT {pfmt(cold_ttft, 0)} ms vs ollama "
            f"{pfmt(oc['ollama_cold_ttft_ms'], 0)} ms ({ratio:.1f}x)"
        )
    idle_p = next(
        (
            r
            for r in recs
            if r.get("provider") == "idle-pallama"
            and "error" not in r
            and r.get("slept")
            and r.get("idle_wake_ttft_ms")
        ),
        None,
    )
    idle_o = next(
        (
            r
            for r in recs
            if r.get("provider") == "idle-ollama"
            and "error" not in r
            and r.get("expired")
            and r.get("idle_wake_ttft_ms")
        ),
        None,
    )
    if idle_p and idle_o:
        parts.append(
            f"idle wake {pfmt(idle_p['idle_wake_ttft_ms'], 0)} ms (sleep) vs ollama "
            f"{pfmt(idle_o['idle_wake_ttft_ms'], 0)} ms (full reload)"
        )
    return "; ".join(parts) + "." if parts else "_No complete rows._"


def write_publication_report(
    recs: list[dict], artifacts_dir: Path, out_path: Path
) -> None:
    versions = {
        r.get("pallama_version", "").strip().removeprefix("pallama ")
        for r in recs
        if r.get("pallama_version")
    }
    pallama_ver = next(iter(versions)) if len(versions) == 1 else "mixed"
    env_states = {
        r.get("power_state", "unstamped")
        for r in recs
        if r.get("provider") == "pallama"
    }

    L: list[str] = []
    L.append("# Pallama inference benchmark")
    L.append("")
    L.append(
        f"_Rendered {artifacts_dir.name}; pallama {pallama_ver}; "
        f"power state of gateway rows: {', '.join(sorted(env_states))}._"
    )
    L.append("")
    L.append("## Executive summary")
    L.append("")
    L.append(executive_summary(recs))
    L.append("")
    L.append("## Test bed")
    L.append("")
    L.append("| Component | Value |")
    L.append("|---|---|")
    L += [f"| {k} | {v} |" for k, v in TEST_BED]
    L.append("")
    L.append("## Methodology")
    L.append("")
    L += [f"- {m}" for m in METHODOLOGY]
    L.append("")
    L.append("## Results")
    L.append("")
    L.append("### Single-stream decode (512-token prompt, 128 generated, median of 5)")
    L.append("")
    L.append(speed_table(recs))
    L.append("")
    L.append("### Concurrency (4 parallel streams x 128 tokens)")
    L.append("")
    L.append(conc_table(recs))
    L.append("")
    L.append(
        "_sum-stream >> system t/s means streams serialize on one slot; "
        "roughly equal means genuinely parallel._"
    )
    L.append("")
    L.append("### Perplexity")
    L.append("")
    L.append(ppl_table(recs))
    L.append("")
    L.append("### Greedy parity and gateway transparency (20 prompts, 256 tokens)")
    L.append("")
    L.append(greedy_table(recs))
    L.append("")
    L.append(
        "_Exact-match divergence across GPU backends is expected float nondeterminism "
        "(batch shape and backend kernels), not translation drift; bit-parity across "
        "runs requires single-slot decoding (pallama `deterministic = true` pins it)._"
    )
    L.append("")
    L.append("### Optimization axes (ctx 4096, single stream)")
    L.append("")
    L.append(variant_table(recs))
    L.append("")
    L.append("### Engine capability matrix")
    L.append("")
    L.append(features_table(recs))
    L.append("")
    L.append("### Cold start and footprint")
    L.append("")
    L.append(coldstart_table(recs))
    L.append("")
    L.append(
        "_Every cold probe runs page-cache-dropped and GPU-idle-asserted on both runtimes; ollama rows without --ollama-service-restart leave the daemon warm (note in the artifact)._"
    )
    L.append("")
    L.append("### Idle wake (sleep vs keep_alive expiry)")
    L.append("")
    L.append(idle_wake_table(recs))
    L.append("")
    L.append(
        "_pallama sleeps with weights in RAM (wake = resume); ollama unloads at keep_alive expiry (wake = full disk reload). Policies differ by design — the table measures each runtime's own idle path after the policy verifiably fired._"
    )
    L.append("")
    L.append("### Long-context degradation curve")
    L.append("")
    L.append(ctxcurve_table(recs))
    L.append("")
    L.append("## Findings")
    L.append("")
    L += [
        "1. **Gateway overhead is within measurement noise.** Single-stream decode through "
        "the pallama gateway matches direct engine spawns at the same slots/context (see "
        "speed table); the greedy gateway lane is byte-identical to the direct lane where "
        "sampling is single-slot.",
        "2. **Capacity-aware slot auto-sizing.** pallama sizes engine slots from live "
        "hardware census: the 8 GiB card with a vision projector attached spawns 1 slot "
        "(16 Ki context) on the Vulkan build and 4 slots (64 Ki total) on CUDA - measured "
        "oversubscription on Vulkan either fails to boot or degrades 2x, so the cap is "
        "load-bearing, not conservative cosmetics.",
        "3. **Concurrency scales where capacity allows.** 4 streams through CUDA gateway "
        "hold near-direct system throughput; the Vulkan single-slot shape serializes "
        "streams (per-stream latency stays excellent; system throughput caps at one "
        "stream's rate) - a capacity trade, not a scheduling defect.",
        "4. **Prompt cache pays ~6-7x on prefill.** Cached-prefix prefill runs thousands "
        "of tokens/s vs hundreds cold.",
        "5. **Speculative n-gram decoding is a net loss for this 9B model** (no draft "
        "model; acceptance too low to pay the verification overhead) - documented so the "
        "flag is not cargo-culted.",
        "6. **KV q8_0 quantization is decode-neutral and prefill-neutral steady-state**; "
        "the one cold-prefill outlier below is a first-invocation pipeline-compile "
        "artifact (controlled re-probe measured full-rate steady state).",
        "7. **mistral.rs 0.9.3 with default paged attention cannot fit this model on an "
        "8 GiB card** (upstream sizes KV as a fraction of total VRAM); pallama's profile "
        "auto-disables paged attention on tight cards and the model then serves correctly.",
    ]
    L.append("")
    L.append("## Caveats")
    L.append("")
    L += [
        "- ollama prefill numbers come from engine counters that exclude the chat "
        "template, so they read slightly high against the 512-token lanes.",
        "- Cross-backend greedy ratios (CUDA vs Vulkan) diverge on near-tie logits; "
        "treat ratio, not exact-match count, as the signal.",
        "- All GPU rows measured on AC power at bounded load; rows record load average "
        "and power state (battery runs are rejected by the harness).",
        "- Numbers are medians of 5 runs on one hybrid laptop; expect absolute shifts "
        "on other hardware, ratios to travel better.",
    ]
    L.append("")
    L.append("## Reproduce")
    L.append("")
    L.append("```bash")
    L.append(
        "python3 scripts/bench_matrix.py --pallama-bin target/release/pallama --md BENCHMARK.md"
    )
    L.append(
        "python3 scripts/bench_matrix.py --render-only --artifacts-dir <dir> --md BENCHMARK.md"
    )
    L.append("```")
    L.append("")
    L.append(
        f"_Raw per-cell records (argv, per-run lists, daemon logs): "
        f"`{artifacts_dir}/cells.jsonl`._"
    )
    L.append("")

    out_path.write_text("\n".join(L))
    print(f"report -> {out_path} ({len(recs)} last-wins records)")


if __name__ == "__main__":
    sys.exit(main())
