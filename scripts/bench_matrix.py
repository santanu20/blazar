#!/usr/bin/env python3
"""Professional cross-engine, cross-provider benchmark matrix for Blazar.

Sweeps every installed engine (llama.cpp builds AND mistral.rs) across
server providers (direct child spawn, blazar gateway, ollama reference),
plus the media lanes (sdcpp image/video, piper TTS, whisper speech)
through the sandboxed gateway, measuring:

  speed      TTFT p50/p90/p99, inter-token latency p50/p99, decode t/s,
             TRUE prefill t/s (prompt tokens / first-token time) with
             cache-hit variants, token counts from `usage` where the
             server provides it (chunks only as fallback)
  resources  peak RSS, peak GPU memory, peak GPU power, load time
             (spawn->healthy), teardown-verified VRAM return
  serving    greedy-parity text quality vs the SAME-engine direct
             reference (backend numerics) AND gateway-transparency
             parity (blazar path vs direct path, same engine),
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
  - blazar cells record the resolved child argv + a real GPU/RSS
    sampler (v1 showed GPU 0)
  - greedy reference is per-engine; the gateway lane is the headline
    transparency test

Cell model: (engine_tag, provider, params) -> one record in cells.jsonl
(append + resume; a cell key hashes engine/provider/params/model and the
harness version so stale records never masquerade as current).

Exit codes: 0 = all cells ok, 1 = some cells failed (recorded, campaign
continued), 2 = environment abort (no model / no engines / corpus fetch
failure).

Never touches the user's daemon, config, or ollama service: blazar
cells run inside the validate.py Sandbox (hardlinked engines + DB
backup), direct cells spawn our own PIDs on probed free ports, the
ollama cell is HTTP-only against an already-running service.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import ctypes
import difflib
import hashlib
import http.client
import importlib
import io
import itertools
import json
import math
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
import traceback
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Optional quality-lane dependency: the image lane stamps perceptual
# metrics (contrast / entropy / color diversity) when Pillow is importable
# and degrades to an honest "skipped" note otherwise — the harness itself
# stays stdlib-only.
try:
    from PIL import Image as PILImage
    from PIL import ImageStat as PILImageStat
except ImportError:  # pragma: no cover - exercised only on PIL-less hosts
    PILImage = None
    PILImageStat = None

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))


HARNESS_VERSION = 2
ARTIFACTS_ROOT = Path.home() / ".cache" / "blazar-bench-matrix"

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

# Media-lane defaults. The image lane sweeps diffusion steps at a fixed
# 512x512; the video lane sweeps Wan-aligned frame counts at 320x320
# (RAM-guarded: the wan child stages ~5.6 GiB host RAM for weights, so
# the axis stops at 33 frames where scratch stays flat ~3.0 GiB VRAM).
# Gate probe rides the video family: one expected-400 monster (duration
# 60s -> 960 aligned frames, far past any 8 GiB budget) + one legit 5f
# pass to price the check itself.
MEDIA_IMAGE_STEPS = (4,)
MEDIA_IMAGE_SIZE = "512x512"
MEDIA_VIDEO_FRAMES = (5, 13, 33)
MEDIA_VIDEO_SIZE = "320x320"
MEDIA_VIDEO_STEPS = 8
MEDIA_TTS_CHARS = 840
MEDIA_TTS_CONC_STREAMS = 4
MEDIA_TTS_CONC_CHARS = 400
MEDIA_GATE_MONSTER = {"duration": 60, "size": "512x512", "steps": 8}
MEDIA_MEM_FLOOR_MIB = 4000.0  # image-family floor (child RSS ~3-4G host)
MEDIA_VIDEO_MEM_FLOOR_MIB = 6000.0  # wan offload-to-cpu child stages ~5.6G host
DEFAULT_MEDIA_RUNS = 3

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


def portable_path(text: str) -> str:
    """Rewrite the invoking user's home dir to `~` so receipts and
    reports stay machine-independent — the repo ships them as
    evidence, and a hard-coded /home/<user> path pins the artifact to
    one box. Applied at write time only; resume keys are
    (tag, provider, params) tuples and never carry paths."""
    home = str(Path.home())
    if home not in ("", "/") and home in text:
        return text.replace(home, "~")
    return text


def append_record(path: Path, record: dict) -> None:
    with path.open("a") as fh:
        fh.write(portable_path(json.dumps(record, sort_keys=True)) + "\n")


# ---------------------------------------------------------------------------
# engine discovery (DB kinds + binaries)


@dataclass
class Engine:
    tag: str
    kind: str  # llamacpp | mistralrs | sdcpp | whisper
    dir: Path
    server: Path | None  # llama-server / mistralrs / sd-server binary
    bench: Path | None = None
    perplexity: Path | None = None


# Kinds load_engines deliberately drops, with the reason the coverage
# table prints — the artifact must answer "was every engine benched?"
ENGINE_EXCLUSIONS = {
    "sglang": (
        "needs an HF safetensors model; this box serves GGUF only and "
        "8 GiB VRAM cannot host sglang beside the media children"
    )
}


# Tool-call quality lane: single-turn scenarios scored deterministically
# (temp 0). A small toolset keeps function SELECTION non-trivial while
# parsing stays trivial; each scenario pins expected_fn + required args so
# selection and schema validity score independently. The control scenario
# must NOT trigger a call — it measures the false-positive rate, the
# number selection accuracy alone can be gamed with.
TOOL_BENCH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a math expression",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_flights",
            "description": "Search flights between two cities on a date",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string"},
                    "destination": {"type": "string"},
                    "date": {"type": "string"},
                },
                "required": ["origin", "destination"],
            },
        },
    },
]

TOOL_BENCH_SCENARIOS = [
    {
        "name": "weather-tokyo",
        "prompt": "What is the current weather in Tokyo? Use the tool.",
        "expected_fn": "get_weather",
        "required": ["city"],
    },
    {
        "name": "math-product",
        "prompt": "Compute 17 * 23 using the calculator tool.",
        "expected_fn": "calculate",
        "required": ["expression"],
    },
    {
        "name": "flights-berlin-seoul",
        "prompt": "Find flights from Berlin to Seoul on March 3rd using the tool.",
        "expected_fn": "search_flights",
        "required": ["origin", "destination"],
    },
    {
        "name": "weather-london",
        "prompt": "Is it raining in London right now? Check with the tool.",
        "expected_fn": "get_weather",
        "required": ["city"],
    },
    {
        "name": "math-distractor",
        "prompt": "How many hours are in 3.5 days? Use the calculator tool.",
        "expected_fn": "calculate",
        "required": ["expression"],
    },
    {
        "name": "control-no-tool",
        "prompt": "Say the word hello and nothing else.",
        "expected_fn": None,
        "required": [],
    },
]


def load_engines(data_dir: Path) -> list[Engine]:
    db = data_dir / "blazar.db"
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
            # real engine — a blazar cell for it would flip NO row
            # active and the sandbox daemon exits "no engine installed"
            continue
        kind = kinds[tag]
        if kind == "sdcpp":
            # Media lane: the gateway spawns sd-server on demand; the
            # harness only needs the tag for activation + stamps.
            out.append(Engine(tag, kind, edir, None, None, None))
            continue
        if kind == "whisper":
            out.append(Engine(tag, kind, edir, None, None, None))
            continue
        if kind not in ("llamacpp", "mistralrs"):
            # Text-bench lanes only. sglang text cells run through the
            # same gateway provider once a safetensors model fits the
            # card; on this 8 GiB box they cannot coexist with the media
            # children, so the kind stays discovered-but-not-swept.
            continue
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
    sandbox daemon's engine). pid may also be assigned LATE (blazar
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
    cached would hand blazar a warmer cold start than the ollama lane
    (which has no projector at all)."""
    db = Path(sb.data_home) / "blazar" / "blazar.db"
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


# ---------------------------------------------------------------------------
# media-lane plumbing: byte-format ground truth + first-byte timing +
# sandbox data staging. Everything here is stdlib; parsers are derived
# from the container specs (PNG/IHDR, EBML/Matroska, RIFF/WAV) and
# cross-checked against recorded engine artifacts before trusting them
# for a receipt (a parser bug would silently corrupt every frame count
# in the table — the failure mode this section exists to prevent).


def png_dims(data: bytes) -> tuple[int, int] | None:
    """PNG width/height from the IHDR chunk (fixed offset: signature
    + length + type precede it in every conformant encoder)."""
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    if data[12:16] != b"IHDR":
        return None
    w = int.from_bytes(data[16:20], "big")
    h = int.from_bytes(data[20:24], "big")
    return (w, h)


def _png_quality_metrics(png_bytes: bytes) -> dict[str, float] | None:
    """PIL-gated perceptual stamps for a generated image: rms contrast
    (luma stddev), luma entropy in bits (detail proxy), mean luma, and
    unique colors on a 256x256 downsample (palette-diversity proxy).
    Returns None when PIL is absent or the bytes will not decode — the
    caller stamps an honest 'skipped' note instead of a fake number."""
    if PILImage is None or PILImageStat is None:
        return None
    try:
        with PILImage.open(io.BytesIO(png_bytes)) as im:
            rgb = im.convert("RGB")
            luma = rgb.convert("L")
            stat = PILImageStat.Stat(luma)
            small = rgb.resize((256, 256))
            colors = small.getcolors(maxcolors=65536) or []
            return {
                "rms_contrast": round(stat.stddev[0], 3),
                "mean_luma": round(stat.mean[0], 3),
                "entropy_bits": round(luma.entropy(), 3),
                "unique_colors_256": float(len(colors)),
            }
    except Exception:
        return None


def _ebml_vint(data: bytes, pos: int) -> tuple[int, int] | None:
    """EBML variable-length integer -> (value, next_pos)."""
    if pos >= len(data):
        return None
    first = data[pos]
    if first == 0:
        return None  # 8-byte vints never occur in webm sizes we walk
    length = 1
    mask = 0x80
    while not (first & mask):
        mask >>= 1
        length += 1
    if pos + length > len(data):
        return None
    value = first & (mask - 1)
    for i in range(1, length):
        value = (value << 8) | data[pos + i]
    return value, pos + length


def _ebml_id(data: bytes, pos: int) -> tuple[int, int] | None:
    """EBML element ID -> (id, next_pos). IDs keep the marker bits (unlike sizes)."""
    if pos >= len(data):
        return None
    first = data[pos]
    if first == 0:
        return None
    length = 1
    mask = 0x80
    while not (first & mask):
        mask >>= 1
        length += 1
    if pos + length > len(data):
        return None
    value = first
    for i in range(1, length):
        value = (value << 8) | data[pos + i]
    return value, pos + length


def webm_frame_count(data: bytes) -> tuple[int | None, str | None]:
    """Count presented video frames in a Matroska byte stream.

    Descends Segment (0x18538067) -> Cluster (0x1F43B675) -> SimpleBlock,
    honoring lacing (one SimpleBlock can lace up to 128 frames; block count
    alone under-reports). Returns (frames, parser_note) - a note instead of a
    silent wrong number.
    """
    CONTAINERS = (0x18538067, 0x1F43B675)  # Segment, Cluster
    frames = 0
    blocks = 0
    note = None

    def walk(pos: int, end: int, depth: int) -> None:
        nonlocal frames, blocks, note
        while pos < end and note is None:
            vid = _ebml_id(data, pos)
            if vid is None:
                return
            element_id, body_pos = vid
            size = _ebml_vint(data, body_pos)
            if size is None:
                return
            payload_len, payload_start = size
            payload_end = min(payload_start + payload_len, end)
            if payload_end <= pos:
                note = "zero-size element (malformed?)"
                return
            if element_id in CONTAINERS and depth < 8:
                walk(payload_start, payload_end, depth + 1)
            elif element_id == 0xA3:  # SimpleBlock
                blocks += 1
                p = payload_start
                _tc = _ebml_vint(data, p)  # timecode (signed vint, usually 2 bytes)
                if _tc is None or _tc[1] >= payload_end:
                    note = "truncated block header"
                    return
                p = _tc[1]
                flags = data[p]
                p += 1
                lacing = (flags >> 1) & 0x3
                if lacing == 0:
                    frames += 1
                else:
                    lace = _ebml_vint(data, p)
                    if lace is None:
                        note = "truncated lace header"
                        return
                    frames += 1 + lace[0]
            pos = payload_end

    walk(0, len(data), 0)
    if frames == 0 and note is None:
        return None, f"no SimpleBlocks found ({blocks} blocks)"
    if frames == 0:
        return None, note
    return frames, note


def wav_layout(data: bytes) -> dict | None:
    """RIFF/WAV header walk -> rate/channels/bits/data bytes. Returns
    None when the bytes are not a parseable RIFF (honest null)."""
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return None
    pos = 12
    rate = channels = bits = None
    data_bytes = None
    while pos + 8 <= len(data):
        cid = data[pos : pos + 4]
        clen = int.from_bytes(data[pos + 4 : pos + 8], "little")
        body = data[pos + 8 : pos + 8 + clen]
        if cid == b"fmt " and len(body) >= 16:
            channels = int.from_bytes(body[2:4], "little")
            rate = int.from_bytes(body[4:8], "little")
            bits = int.from_bytes(body[14:16], "little")
        elif cid == b"data":
            data_bytes = clen
        pos += 8 + clen + (clen & 1)
    if rate is None or data_bytes is None:
        return None
    return {
        "rate": rate,
        "channels": channels,
        "bits": bits,
        "data_bytes": data_bytes,
        "audio_s": data_bytes / float(rate * (channels or 1) * ((bits or 16) // 8)),
    }


def http_timed(
    port: int,
    path: str,
    payload: bytes | dict | None,
    timeout: float = 300.0,
    headers: dict[str, str] | None = None,
) -> dict:
    """POST with first-BODY-byte timing via http.client (urllib hides
    chunk boundaries, and TTFB for streaming lanes means first audio
    byte, not response headers).

    Returns status, ttfb_ms, total_s, byte counts and the body. A
    non-2xx status is a RESULT, not an exception — the gate lane times
    400 rejections, and callers assert expected statuses.
    """
    import http.client

    if isinstance(payload, dict):
        payload = json.dumps(payload).encode()
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    t0 = time.perf_counter()
    ttfb = None
    body = b""
    status = None
    resp_headers: dict[str, str] = {}
    try:
        conn.request("POST", path, body=payload, headers=hdrs)
        resp = conn.getresponse()
        status = resp.status
        resp_headers = {k.lower(): v for k, v in resp.getheaders()}
        first = True
        while True:
            chunk = resp.read(65536)
            if not chunk:
                break
            if first:
                ttfb = time.perf_counter()
                first = False
            body += chunk
        total = time.perf_counter()
        return {
            "status": status,
            "ttfb_ms": round((ttfb - t0) * 1000.0, 1) if ttfb else None,
            "total_s": round(total - t0, 3),
            "bytes": len(body),
            "headers": resp_headers,
            "body": body,
        }
    finally:
        conn.close()


def _hardlink_tree(src: Path, dst: Path) -> None:
    """Sandbox-safe staging: hardlink copy (same filesystem), so a
    sandboxed process can unlink its view without touching the real
    store — mirrors validate.py's engines handling."""
    shutil.copytree(src, dst, symlinks=True, copy_function=os.link, dirs_exist_ok=True)


@contextlib.contextmanager
def media_family(
    label: str,
    activate_kinds: tuple[str, ...],
    stage_voices: bool = False,
    stage_whisper_models: bool = False,
):
    """Context manager: sandboxed daemon ready for media requests.

    Mirrors run_blazar_cell's isolation (unique port, BLAZAR_VALIDATE
    reaper protection, teardown-dark verify) but for media families the
    child spawns lazily on the first lane request — the COLD number is
    the spawn+first-media wall, labeled cold_request_s per lane.

    Deliberately NOT wait_gpu_idle-gated: media boxes routinely carry a
    warm child from a previous family (or the user's daemon); the lane
    stamps gpu_busy at entry and the receipt carries the truth.
    """
    os.environ["BLAZAR_VALIDATE_PORT"] = str(free_port())
    V = importlib.import_module("validate")
    V.PORT = int(os.environ["BLAZAR_VALIDATE_PORT"])
    sb = V.Sandbox()
    port = V.PORT
    daemon = None
    try:
        con = sqlite3.connect(Path(sb.data_home) / "blazar" / "blazar.db")
        marks = ",".join("?" * len(activate_kinds)) or "NULL"
        con.execute(
            f"UPDATE engines SET active = (kind IN ({marks}))",
            activate_kinds,
        )
        con.commit()
        con.close()
        real_data = Path.home() / ".local/share/blazar"
        if stage_voices:
            _hardlink_tree(real_data / "voices", Path(sb.data_dir) / "voices")
            # the piper engine binary lives outside engines/ — stage it too,
            # the gateway 404s with "piper is not installed" without it
            _hardlink_tree(real_data / "piper", Path(sb.data_dir) / "piper")
        if stage_whisper_models:
            src = real_data / "whisper" / "models"
            dst = Path(sb.data_dir) / "whisper" / "models"
            dst.parent.mkdir(parents=True, exist_ok=True)
            _hardlink_tree(src, dst)
        daemon = V.Daemon(sb)
        daemon.start(cfg={"port": port})
        deadline = time.time() + 120
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
            raise RuntimeError(f"media family '{label}' daemon failed to boot")
        yield {"port": port, "sb": sb}
    finally:
        log_tail = None
        if daemon is not None:
            daemon.stop()
            dlog = Path(sb.data_dir) / "run" / "daemon.log"
            if dlog.exists():
                log_tail = "\n".join(
                    dlog.read_text(errors="replace").splitlines()[-12:]
                )
        sb.destroy()
        if log_tail:
            Path(Path.home() / ".cache/blazar-bench-matrix").mkdir(
                parents=True, exist_ok=True
            )
        # log tail surfaces via the lane record, not stdout spam
        media_family.last_log_tail = log_tail  # type: ignore[attr-defined]


def _media_env_stamps() -> dict:
    """Per-cell honesty stamps: GPU census + RAM head + loadavg. Media
    lanes run with warm children by design; the receipt says so."""
    stamps = {
        "gpu_busy_mib": round(gpu_used_mib(), 0),
        "loadavg_5m": Path("/proc/loadavg").read_text().split()[1],
    }
    try:
        meminfo = {
            parts[0].rstrip(":"): int(parts[1])
            for parts in (
                line.split()[:2]
                for line in Path("/proc/meminfo").read_text().splitlines()
            )
        }
        stamps["ram_avail_mib"] = round(meminfo.get("MemAvailable", 0) / 1024, 0)
    except (OSError, ValueError):
        stamps["ram_avail_mib"] = None
    return stamps


def _media_child_stamps() -> dict:
    """Attach argv of a live sandbox engine child, when one exists."""
    pid = find_sandbox_engine_pid()
    if pid is None:
        return {}
    return {"child_pid": pid, "child_argv": read_proc_argv(pid)}


def run_media_image_cell(eng: Engine, model_id: str, cfg: dict) -> dict:
    """Image lane: sync generations, steps axis at fixed size. Cold row
    = spawn + first image (the user-felt wait); warm rows follow."""
    rec: dict = {"lane": "image"}
    with media_family(f"image-{eng.tag}", ("sdcpp",)) as fam:
        sampler = Sampler(None)
        sampler.start()
        port = fam["port"]
        try:
            rec["gpu_busy_entry_mib"] = round(gpu_used_mib(), 0)
            body = {
                "model": model_id,
                "prompt": "a lighthouse on a cliff at dusk, oil painting",
                "size": MEDIA_IMAGE_SIZE,
                "steps": MEDIA_IMAGE_STEPS[0],
            }
            t0 = time.perf_counter()
            cold = http_timed(port, "/v1/images/generations", body, timeout=600.0)
            rec["cold_request_s"] = round(time.perf_counter() - t0, 2)
            rec["cold_status"] = cold["status"]
            rec.update(_media_child_stamps())
            runs: list[dict] = []
            saved_steps: set[int] = set()
            steps_axis = list(MEDIA_IMAGE_STEPS)
            for steps in steps_axis:
                for _run in range(cfg["runs"]):
                    got = http_timed(
                        port,
                        "/v1/images/generations",
                        {**body, "steps": steps},
                        timeout=600.0,
                    )
                    run_rec = {
                        "steps": steps,
                        "status": got["status"],
                        "total_s": got["total_s"],
                    }
                    if got["status"] == 200:
                        try:
                            doc = json.loads(got["body"])
                            img = base64.b64decode(doc["data"][0]["b64_json"])
                            run_rec["png_bytes"] = len(img)
                            dims = png_dims(img)
                            if dims is None:
                                run_rec["png_parse_note"] = "not a PNG body"
                            else:
                                run_rec["dims"] = f"{dims[0]}x{dims[1]}"
                            quality = _png_quality_metrics(img)
                            if quality is not None:
                                run_rec["quality"] = quality
                            elif PILImage is None:
                                rec.setdefault("quality_note", "skipped: PIL absent")
                            # one audit artifact per steps point so the
                            # perceptual stamps stay reproducible offline
                            if (
                                dims is not None
                                and cfg.get("art_dir")
                                and steps not in saved_steps
                            ):
                                art = Path(cfg["art_dir"])
                                fname = (
                                    f"image-{model_id.replace('/', '_')}"
                                    f"-steps{steps}.png"
                                )
                                (art / fname).write_bytes(img)
                                saved_steps.add(steps)
                        except (ValueError, KeyError, IndexError) as exc:
                            run_rec["png_parse_note"] = f"decode failed: {exc}"
                    runs.append(run_rec)
            rec["runs"] = runs
            ok = [r for r in runs if r.get("status") == 200 and "dims" in r]
            if ok:
                ts = [r["total_s"] for r in ok]
                rec["total_s_median"] = round(statistics.median(ts), 2)
                rec["total_s_min"] = round(min(ts), 2)
                rec["total_s_max"] = round(max(ts), 2)
                rec["dims_seen"] = sorted({r["dims"] for r in ok})
                rec["cold_status"] = cold["status"]
                qkeys = [
                    "rms_contrast",
                    "mean_luma",
                    "entropy_bits",
                    "unique_colors_256",
                ]
                medians: dict[str, float] = {}
                for qk in qkeys:
                    vals = [
                        r["quality"][qk]
                        for r in ok
                        if isinstance(r.get("quality"), dict)
                    ]
                    if vals:
                        medians[qk] = round(statistics.median(vals), 3)
                if medians:
                    rec["quality_medians"] = medians
        finally:
            rec["gpu_peak_mib"] = round(sampler.gpu_peak_mib, 1)
            rec["rss_peak_mib"] = round(sampler.rss_peak_mib, 1)
            sampler.stop_evt.set()
            rec["daemon_log_tail"] = getattr(media_family, "last_log_tail", None)
    return rec


def run_media_video_cell(eng: Engine, model_id: str, cfg: dict) -> dict:
    """Video lane: frames axis at 320x320 (RAM-guarded). Ground truth =
    container block count vs the aligned request (Wan floors to a 4k+1
    temporal grid); the gate probe rides this family after the runs."""
    rec: dict = {"lane": "video"}
    with media_family(f"video-{eng.tag}", ("sdcpp",)) as fam:
        sampler = Sampler(None)
        sampler.start()
        port = fam["port"]
        try:
            rec["gpu_busy_entry_mib"] = round(gpu_used_mib(), 0)
            rec.update(_media_env_stamps())
            body_base = {
                "model": model_id,
                "prompt": "waves rolling onto a rocky shore at sunset",
                "size": MEDIA_VIDEO_SIZE,
                "steps": MEDIA_VIDEO_STEPS,
            }
            first_frames = MEDIA_VIDEO_FRAMES[0]
            t0 = time.perf_counter()
            cold = http_timed(
                port,
                "/v1/videos/generations",
                {**body_base, "frames": first_frames},
                timeout=600.0,
            )
            rec["cold_request_s"] = round(time.perf_counter() - t0, 2)
            rec["cold_status"] = cold["status"]
            rec.update(_media_child_stamps())
            per_frames: list[dict] = []
            for frames in MEDIA_VIDEO_FRAMES:
                if not mem_guard(MEDIA_VIDEO_MEM_FLOOR_MIB, f"video {frames}f"):
                    per_frames.append({"frames": frames, "skipped": "RAM floor"})
                    continue
                runs = []
                for _run in range(cfg["runs"]):
                    got = http_timed(
                        port,
                        "/v1/videos/generations",
                        {**body_base, "frames": frames},
                        timeout=600.0,
                    )
                    run_rec = {"status": got["status"], "total_s": got["total_s"]}
                    if got["status"] == 200:
                        try:
                            doc = json.loads(got["body"])
                            item = doc["data"][0]
                            run_rec["reported_frame_count"] = item.get("frame_count")
                            webm = base64.b64decode(item["b64_json"])
                            mux_frames, note = webm_frame_count(webm)
                            run_rec["mux_frames"] = mux_frames
                            run_rec["mux_note"] = note
                            run_rec["webm_bytes"] = len(webm)
                            aligned = 4 * ((frames - 1) // 4) + 1
                            run_rec["frames_requested_aligned"] = aligned
                            if mux_frames is not None and mux_frames != aligned:
                                run_rec["frame_mismatch"] = (
                                    f"container {mux_frames} != aligned {aligned}"
                                )
                        except (ValueError, KeyError, IndexError) as exc:
                            run_rec["mux_note"] = f"decode failed: {exc}"
                    runs.append(run_rec)
                ok = [r for r in runs if r.get("status") == 200 and "total_s" in r]
                agg = {"frames": frames, "runs": runs}
                if ok:
                    ts = [r["total_s"] for r in ok]
                    agg["total_s_median"] = round(statistics.median(ts), 2)
                    agg["total_s_min"] = round(min(ts), 2)
                    agg["total_s_max"] = round(max(ts), 2)
                per_frames.append(agg)
            rec["per_frames"] = per_frames

            # gate probe: expected-400 monster x3 + one legit pass to
            # price the check itself (the pass's total_s rides the same
            # warm child as the axis runs, so the delta vs the 5f row
            # IS the gate overhead).
            gate_runs = []
            for _ in range(3):
                got = http_timed(
                    port,
                    "/v1/videos/generations",
                    {**body_base, **MEDIA_GATE_MONSTER},
                    timeout=60.0,
                )
                gate_runs.append({"status": got["status"], "total_s": got["total_s"]})
                if got["status"] != 400:
                    rec["gate_note"] = (
                        f"expected 400 from monster, got {got['status']} "
                        "(gate bypassed or daemon predates it)"
                    )
            rec["gate_reject_s_list"] = [g["total_s"] for g in gate_runs]
            rec["gate_reject_s_median"] = (
                round(statistics.median([g["total_s"] for g in gate_runs]), 3)
                if gate_runs
                else None
            )
        finally:
            rec["gpu_peak_mib"] = round(sampler.gpu_peak_mib, 1)
            rec["rss_peak_mib"] = round(sampler.rss_peak_mib, 1)
            sampler.stop_evt.set()
            rec["daemon_log_tail"] = getattr(media_family, "last_log_tail", None)
    return rec


def _tts_input_text(n_chars: int) -> str:
    """Deterministic ~n-char input built from the prefill bank (steady
    content across campaigns — no per-run prose drift in RTF)."""
    out = []
    total = 0
    bank = itertools.cycle(PREFILL_BANK)
    while total < n_chars:
        s = next(bank)
        out.append(s)
        total += len(s) + 1
    return " ".join(out)


def run_media_tts_cell(cfg: dict) -> dict:
    """TTS lane: buffered WAV vs streamed PCM on the same input.

    WAV: TTFB == total (buffered by design). PCM: first-body-byte is
    the first synthesized chunk — the interactive-audio number. RTF =
    synthesis wall / audio seconds for both formats.
    """
    voice = "en_US-amy-medium"
    rec: dict = {"lane": "tts", "voice": voice, "input_chars": MEDIA_TTS_CHARS}
    with media_family("tts", ("sdcpp",), stage_voices=True) as fam:
        sampler = Sampler(None)
        sampler.start()
        port = fam["port"]
        try:
            rec.update(_media_env_stamps())
            text = _tts_input_text(MEDIA_TTS_CHARS)
            base_body = {"model": voice, "input": text}

            wav_runs = []
            wav_body = b""
            for _ in range(cfg["runs"]):
                got = http_timed(
                    port,
                    "/v1/audio/speech",
                    {**base_body, "response_format": "wav"},
                    timeout=300.0,
                )
                wav_body = got["body"]
                wav_runs.append(
                    {
                        "status": got["status"],
                        "total_s": got["total_s"],
                        "bytes": got["bytes"],
                    }
                )
            rec["wav_runs"] = wav_runs
            lay = wav_layout(wav_body) if wav_body else None
            if lay:
                rec["audio_s"] = round(lay["audio_s"], 2)
                rec["wav_rate"] = lay["rate"]
                rec["wav_channels"] = lay["channels"]
                rec["wav_bits"] = lay["bits"]

            pcm_runs = []
            for _ in range(cfg["runs"]):
                got = http_timed(
                    port,
                    "/v1/audio/speech",
                    {**base_body, "response_format": "pcm"},
                    timeout=300.0,
                )
                pcm_runs.append(
                    {
                        "status": got["status"],
                        "ttfb_ms": got["ttfb_ms"],
                        "total_s": got["total_s"],
                        "bytes": got["bytes"],
                        "pcm_format_header": got["headers"].get("x-blazar-pcm-format"),
                    }
                )
            rec["pcm_runs"] = pcm_runs

            ok_wav = [r for r in wav_runs if r["status"] == 200]
            ok_pcm = [r for r in pcm_runs if r["status"] == 200]
            if ok_wav:
                ts = [r["total_s"] for r in ok_wav]
                rec["wav_total_s_median"] = round(statistics.median(ts), 3)
                if rec.get("audio_s"):
                    rec["wav_rtf"] = round(statistics.median(ts) / rec["audio_s"], 4)
            if ok_pcm:
                tt = [r["ttfb_ms"] for r in ok_pcm if r["ttfb_ms"]]
                ts = [r["total_s"] for r in ok_pcm]
                if tt:
                    rec["pcm_ttfb_ms_median"] = round(statistics.median(tt), 1)
                rec["pcm_total_s_median"] = round(statistics.median(ts), 3)
                if rec.get("audio_s"):
                    rec["pcm_rtf"] = round(statistics.median(ts) / rec["audio_s"], 4)
            if rec.get("wav_total_s_median") and rec.get("pcm_ttfb_ms_median"):
                rec["ttfb_speedup_x"] = round(
                    rec["wav_total_s_median"] * 1000.0 / rec["pcm_ttfb_ms_median"],
                    2,
                )
        finally:
            sampler.stop_evt.set()
            rec["daemon_log_tail"] = getattr(media_family, "last_log_tail", None)
    return rec


def run_media_tts_concurrency_cell(cfg: dict) -> dict:
    """TTS concurrency probe: N parallel streamed-PCM requests through
    one sandboxed gateway (scalability receipt). Efficiency = sum of
    per-stream totals / wall clock: ~1 means the lane serializes, -> N
    means perfectly parallel. Fails loudly when any stream errors or
    truncates; identical input across streams must yield identical
    byte counts (deterministic piper + limiter) or the row says so."""
    voice = "en_US-amy-medium"
    n_streams = int(cfg.get("conc_streams", MEDIA_TTS_CONC_STREAMS))
    rec: dict = {
        "lane": "tts-concurrency",
        "voice": voice,
        "streams": n_streams,
        "input_chars": MEDIA_TTS_CONC_CHARS,
    }
    with media_family("tts-conc", ("sdcpp",), stage_voices=True) as fam:
        sampler = Sampler(None)
        sampler.start()
        port = fam["port"]
        try:
            rec.update(_media_env_stamps())
            body = {
                "model": voice,
                "input": _tts_input_text(MEDIA_TTS_CONC_CHARS),
                "response_format": "pcm",
            }
            results: list[dict] = [{} for _ in range(n_streams)]

            def _one_stream(i: int) -> None:
                got = http_timed(port, "/v1/audio/speech", body, timeout=300.0)
                results[i] = {
                    "status": got["status"],
                    "ttfb_ms": got["ttfb_ms"],
                    "total_s": got["total_s"],
                    "bytes": got["bytes"],
                }

            threads = [
                threading.Thread(target=_one_stream, args=(i,))
                for i in range(n_streams)
            ]
            t0 = time.perf_counter()
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            wall_s = time.perf_counter() - t0
            rec["wall_s"] = round(wall_s, 3)
            rec["streams_detail"] = results
            ok = [r for r in results if r.get("status") == 200 and r.get("bytes")]
            if len(ok) != n_streams:
                rec["error"] = f"only {len(ok)}/{n_streams} streams completed"
            else:
                per = [r["total_s"] for r in ok]
                rec["per_stream_total_s_median"] = round(statistics.median(per), 3)
                rec["efficiency_sum_over_wall"] = round(sum(per) / wall_s, 2)
                tt = [r["ttfb_ms"] for r in ok if r["ttfb_ms"]]
                if tt:
                    rec["ttfb_ms_median"] = round(statistics.median(tt), 1)
                    rec["ttfb_ms_max"] = round(max(tt), 1)
                uniq_bytes = sorted({r["bytes"] for r in ok})
                rec["bytes_uniform"] = len(uniq_bytes) == 1
                rec["bytes_seen"] = uniq_bytes[:3]
        finally:
            rec["gpu_peak_mib"] = round(sampler.gpu_peak_mib, 1)
            rec["rss_peak_mib"] = round(sampler.rss_peak_mib, 1)
            sampler.stop_evt.set()
            rec["daemon_log_tail"] = getattr(media_family, "last_log_tail", None)
    return rec


def _multipart_body(
    fields: dict[str, str],
    file_field: str,
    filename: str,
    file_bytes: bytes,
    content_type: str,
) -> tuple[bytes, str]:
    """Minimal stdlib multipart encoder (whisper transcriptions: file +
    model field). Returns (body, content_type with boundary)."""
    boundary = f"blazarbench{int(time.time() * 1000)}"
    parts = []
    for name, value in fields.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"'
            f"\r\n\r\n{value}\r\n".encode()
        )
    parts.append(
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; '
            f'filename="{filename}"\r\nContent-Type: {content_type}\r\n\r\n'
        ).encode()
    )
    parts.append(file_bytes + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def run_media_whisper_cell(eng: Engine, cfg: dict) -> dict:
    """Speech lane: transcribe a WAV the TTS lane just synthesized (the
    family stages voices for exactly this) — no fixture files, the input
    provably comes from the piper voice under test."""
    rec: dict = {"lane": "whisper"}
    # whisper alone cannot boot the daemon (audio lane is not a serving
    # engine by design) — sdcpp rides along as the serving row, lazily idle
    with media_family(
        f"whisper-{eng.tag}",
        ("sdcpp", "whisper"),
        stage_voices=True,
        stage_whisper_models=True,
    ) as fam:
        sampler = Sampler(None)
        sampler.start()
        port = fam["port"]
        try:
            rec.update(_media_env_stamps())
            text = _tts_input_text(400)
            src = http_timed(
                port,
                "/v1/audio/speech",
                {"model": "en_US-amy-medium", "input": text},
                timeout=300.0,
            )
            if src["status"] != 200:
                return {
                    **rec,
                    "error": f"tts input synthesis failed: {src['status']}",
                }
            wav = src["body"]
            lay = wav_layout(wav)
            rec["input_audio_s"] = round(lay["audio_s"], 2) if lay else None
            rec["input_bytes"] = len(wav)

            # cold run = spawn + transcribe (multipart)
            body, ctype = _multipart_body(
                {"model": "whisper-1"}, "file", "input.wav", wav, "audio/wav"
            )
            t0 = time.perf_counter()
            got = http_timed_raw(port, "/v1/audio/transcriptions", body, 600.0, ctype)
            rec["cold_request_s"] = round(time.perf_counter() - t0, 2)
            rec["cold_status"] = got["status"]
            rec.update(_media_child_stamps())
            runs = []
            for _ in range(cfg["runs"]):
                body, ctype = _multipart_body(
                    {"model": "whisper-1"}, "file", "input.wav", wav, "audio/wav"
                )
                g = http_timed_raw(port, "/v1/audio/transcriptions", body, 600.0, ctype)
                run_rec = {"status": g["status"], "total_s": g["total_s"]}
                if g["status"] == 200:
                    try:
                        doc = json.loads(g["body"])
                        run_rec["text_chars"] = len(doc.get("text", ""))
                    except ValueError:
                        run_rec["text_parse_note"] = "non-JSON body"
                runs.append(run_rec)
            rec["runs"] = runs
            ok = [r for r in runs if r["status"] == 200 and "total_s" in r]
            if ok:
                ts = [r["total_s"] for r in ok]
                rec["total_s_median"] = round(statistics.median(ts), 3)
                if rec.get("input_audio_s"):
                    rec["rtf"] = round(statistics.median(ts) / rec["input_audio_s"], 4)
        finally:
            rec["gpu_peak_mib"] = round(sampler.gpu_peak_mib, 1)
            rec["rss_peak_mib"] = round(sampler.rss_peak_mib, 1)
            sampler.stop_evt.set()
            rec["daemon_log_tail"] = getattr(media_family, "last_log_tail", None)
    return rec


def http_timed_raw(
    port: int, path: str, payload: bytes, timeout: float, content_type: str
) -> dict:
    """http_timed for pre-encoded non-JSON bodies (multipart)."""
    return http_timed(
        port, path, payload, timeout, headers={"Content-Type": content_type}
    )


def openai_stream_timed(port: int, body: dict, timeout: float = 300.0) -> dict:
    """POST /v1/chat/completions (stream) -> timing metrics.

    Token counts prefer the final `usage` (server-authoritative; the
    blazar gateway injects usage into /v1 streams) with per-chunk
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
    tokens = tokens_usage or len(stamps)
    src = "usage" if tokens_usage else "chunks"
    itls = [(b - a) * 1000 for a, b in itertools.pairwise(stamps)]
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
    itls = [(b - a) * 1000 for a, b in itertools.pairwise(stamps)]
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
    model for the ollama-compatible translation layer) — the blazar
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
        return None  # blazar lanes tokenize server-side already
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
            # lane at 371 real tokens vs the blazar lane's 527 at the
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
    blazar path (WFQ/slot leases/predictive reject) and llama-server
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
            except Exception as exc:
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
    Blazar serving lane does."""
    d = stage_root / re.sub(r"[^A-Za-z0-9_.-]", "_", model_path.stem)[:80]
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    os.symlink(model_path, d / model_path.name)
    if mmproj is not None and mmproj.exists():
        os.symlink(mmproj, d / mmproj.name)
    # absolute: direct cells spawn the child with cwd=<engine dir>, so a
    # relative stage path only resolves when the harness happens to run
    # from the repo root - mistral.rs would reject the model with
    # "does not exist or is not a file" from any other cwd.
    return (d / model_path.name).resolve()


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
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            os.killpg(proc.pid, signal.SIGKILL)
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
            with contextlib.suppress(OSError):
                tail = errlog.read_text(errors="replace")[-400:].replace("\n", " | ")
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
        # the blazar gateway rewrites at the proxy — we do it here.
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
# blazar provider (validate.py Sandbox; per-engine active flip)


def find_sandbox_engine_pid() -> int | None:
    """Locate the engine child the sandbox daemon spawned.

    The sandbox DB carries REAL engine paths (the daemon resolves the
    binary from its manifest), so a path-prefix scan misses it. The
    daemon is spawned with BLAZAR_VALIDATE=1 and the engine child
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
        if b"BLAZAR_VALIDATE=1" in environ:
            return int(pid_s)
    return None


def read_proc_argv(pid: int) -> list[str]:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            return [a.decode("utf-8", "replace") for a in fh.read().split(b"\0") if a]
    except OSError:
        return []


def run_blazar_cell(
    eng: Engine,
    model_name: str,
    cfg: dict,
    skip_ollama_note: str,
    blazar_cfg: dict | None = None,
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
    # BLAZAR_VALIDATE_PORT once at import time — a shared fixed port is
    # exactly how orphaned sandbox daemons hijacked campaigns (leak class
    # fixed in validate.py; this makes collisions structurally impossible)
    os.environ["BLAZAR_VALIDATE_PORT"] = str(free_port())
    V = importlib.import_module("validate")
    # F139: the module cache returns the FIRST import on later campaigns
    # — rebinding PORT on the module is what actually takes effect; the
    # env re-set above alone is inert past the first import.
    V.PORT = int(os.environ["BLAZAR_VALIDATE_PORT"])

    rec: dict = {}
    sb = V.Sandbox()
    sampler = Sampler(None)
    sampler.start()
    try:
        con = sqlite3.connect(Path(sb.data_home) / "blazar" / "blazar.db")
        con.execute("UPDATE engines SET active = (tag = ?)", (eng.tag,))
        con.commit()
        con.close()
        daemon = V.Daemon(sb)
        rec["gpu_busy_mib"] = warm_cell_gpu_guard("blazar cell")
        t_boot0 = time.perf_counter()
        port: int | None = None
        try:
            # start() INSIDE the stop()-owning try: a post-Popen raise
            # (healthz timeout, liveness guard) used to leak a live
            # daemon whose sandbox got destroyed under it (2026-09-10).
            daemon.start(
                cfg={"port": V.PORT, **(blazar_cfg or {})}, floor_model=model_name
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
            except Exception as cold_exc:
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


def probe_tools_once(port: int, model_body: str, prompt: str) -> dict:
    """One scenario through /v1/chat/completions with an OpenAI tools array.

    Streams the response (SSE) and accumulates tool_call deltas by index —
    argument JSON arrives in fragments across chunks. Returns per-scenario
    outcome: http status, accumulated calls, ttft, finish_reason, content.
    HTTP failure is an OUTCOME (returned), not an exception — the caller
    decides whether it is a cell error or a zero score.
    """
    body = json.dumps(
        {
            "model": model_body,
            "stream": True,
            "temperature": 0,
            "max_tokens": 192,
            "messages": [{"role": "user", "content": prompt}],
            "tools": TOOL_BENCH_TOOLS,
        }
    ).encode()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
    t0 = time.perf_counter()
    ttft_ms: float | None = None
    calls: dict[int, dict] = {}
    content_parts: list[str] = []
    finish_reason = None
    status = None
    try:
        conn.request(
            "POST",
            "/v1/chat/completions",
            body=body,
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        status = resp.status
        if status != 200:
            return {
                "status": status,
                "error_body": resp.read(4096).decode("utf-8", "replace"),
            }
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            if ttft_ms is None:
                ttft_ms = (time.perf_counter() - t0) * 1000.0
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choice = (chunk.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
            if delta.get("content"):
                content_parts.append(delta["content"])
            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index", 0)
                slot = calls.setdefault(idx, {"name": None, "args": ""})
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["name"] = fn["name"]
                slot["args"] += fn.get("arguments") or ""
        return {
            "status": 200,
            "ttft_ms": ttft_ms,
            "finish_reason": finish_reason,
            "calls": [calls[i] for i in sorted(calls)],
            "content": "".join(content_parts),
        }
    finally:
        conn.close()


def score_tools_scenario(scenario: dict, outcome: dict) -> dict:
    """Score one scenario outcome against its expectation.

    Kept separate from the probe so the selftest can score fabricated
    outcomes without a server. A scenario whose expected_fn is None is the
    control: ANY tool call is a false positive.
    """
    if outcome.get("status") != 200:
        # transport/HTTP failure: the cell machinery reports it; never
        # silently score a broken request as model failure
        return {"transport_error": outcome.get("error_body") or outcome.get("status")}
    emitted = [c for c in outcome.get("calls", []) if c.get("name")]
    wellformed = bool(emitted) and all(_args_parse(c["args"]) for c in emitted)
    expected_fn = scenario.get("expected_fn")
    if expected_fn is None:
        return {
            "wellformed": wellformed,
            "selection": None,
            "args_valid": None,
            "false_positive": bool(emitted),
            "ttft_ms": outcome.get("ttft_ms"),
        }
    required = scenario.get("required_args", [])
    selected_objs = []
    for c in emitted:
        if c["name"] != expected_fn:
            continue
        try:
            selected_objs.append(json.loads(c["args"]))
        except (json.JSONDecodeError, TypeError):
            continue
    args_valid = (
        any(set(required) <= set(obj) for obj in selected_objs)
        if selected_objs
        else False
    )
    return {
        "wellformed": wellformed,
        "selection": bool(selected_objs),
        "args_valid": args_valid,
        "false_positive": False,
        "ttft_ms": outcome.get("ttft_ms"),
    }


def _args_parse(args_str: str):
    try:
        json.loads(args_str)
        return True
    except (json.JSONDecodeError, TypeError):
        return False


def _child_np(pid: int) -> int | None:
    """Current -np of a spawned engine child, from /proc."""
    try:
        with contextlib.suppress(Exception):
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            for i, tok in enumerate(cmd):
                if tok == b"-np" and i + 1 < len(cmd):
                    return int(cmd[i + 1])
    except Exception:
        pass
    return None


def run_reshape_cell(eng: Engine, body_model: str, duration_s: float = 300.0) -> dict:
    """Sustained-concurrency reshape lane: hold C=8 streams long enough for
    the 6-tick adoption streak + graceful drain, and prove the child actually
    reshapes (no dropped streams, capacity gain measured before/after)."""
    port = free_port()
    os.environ["BLAZAR_VALIDATE_PORT"] = str(port)
    V = importlib.import_module("validate")
    V.PORT = port
    sb = V.Sandbox()
    con = sqlite3.connect(f"file:{Path(sb.data_dir) / 'blazar.db'}?mode=rw", uri=True)
    try:
        con.execute("UPDATE engines SET active = 0")
        con.execute("UPDATE engines SET active = 1 WHERE tag = ?", (eng.tag,))
        con.commit()
    finally:
        con.close()
    daemon = V.Daemon(sb)
    daemon.start({"port": port}, floor_model=body_model)
    stop_at = time.monotonic() + 600.0
    base = f"http://127.0.0.1:{port}"
    while time.monotonic() < stop_at:
        try:
            with urllib.request.urlopen(f"{base}/healthz", timeout=2) as resp:
                if resp.status == 200:
                    break
        except (json.JSONDecodeError, urllib.error.URLError, OSError):
            time.sleep(0.5)
    started = time.monotonic()
    deadline = started + duration_s
    lock = threading.Lock()
    results: list[dict] = []
    timeline: list[dict] = []
    failed = [0]
    fail_tally: dict[tuple, int] = {}

    def worker() -> None:
        while time.monotonic() < deadline:
            out = probe_tools_once(port, body_model, "Count from 1 to 40 slowly.")
            if out.get("status") not in (None, 200):
                with lock:
                    failed[0] += 1
                    key = (out.get("status"), str(out.get("error_body", ""))[:100])
                    fail_tally[key] = fail_tally.get(key, 0) + 1
                continue
            rec = {"t_rel": time.monotonic() - started, "ttft_ms": out.get("ttft_ms")}
            with lock:
                results.append(rec)

    def poller() -> None:
        while time.monotonic() < deadline:
            np_now, inflight = None, None
            with contextlib.suppress(Exception):
                with urllib.request.urlopen(f"{base}/api/ps", timeout=2) as resp:
                    ps = json.loads(resp.read())
                    # gateway /api/ps shape: {"instances": [ {...row with pid} ]}
                    procs = ps.get("instances") or ps.get("models") or []
                    if procs and procs[0].get("pid"):
                        np_now = _child_np(int(procs[0]["pid"]))
                inflight = ps.get("in_flight") or ps.get("inflight")
            with lock:
                timeline.append(
                    {
                        "t_rel": time.monotonic() - started,
                        "np": np_now,
                        "in_flight": inflight,
                    }
                )
            time.sleep(2.0)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    threads.append(threading.Thread(target=poller))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    def _pctl(vals: list[float], q: float) -> float | None:
        if not vals:
            return None
        vals = sorted(vals)
        return vals[min(len(vals) - 1, int(q * len(vals)))]

    if not results:
        daemon.stop()
        with contextlib.suppress(Exception):
            Path(daemon.log_path).read_text(errors="replace")
        sb.destroy()
        return {
            "error": (
                f"all {failed[0]} request(s) failed against the sandbox gateway; "
                "see daemon_tail"
            )
        }

    nps = [t["np"] for t in timeline if t["np"]]
    reshape_at = None
    first_np = nps[0] if nps else None
    for t in timeline:
        if t["np"] and first_np and t["np"] > first_np:
            reshape_at = t["t_rel"]
            break
    before = [r for r in results if reshape_at is None or r["t_rel"] < reshape_at]
    after = [r for r in results if reshape_at is not None and r["t_rel"] >= reshape_at]
    window = max((r["t_rel"] for r in results), default=duration_s)
    rec = {
        "reshape_observed": reshape_at is not None,
        "slots_from": first_np,
        "slots_to": max(nps) if nps else None,
        "time_to_reshape_s": round(reshape_at, 1) if reshape_at is not None else None,
        "requests_before": len(before),
        "requests_after": len(after),
        "ttft_p50_before_ms": _pctl(
            [r["ttft_ms"] for r in before if r["ttft_ms"]], 0.5
        ),
        "ttft_p50_after_ms": _pctl([r["ttft_ms"] for r in after if r["ttft_ms"]], 0.5),
        "sys_tps_before": round(len(before) * 96 / max(r["t_rel"] for r in before), 1)
        if before
        else None,
        "sys_tps_after": round(
            len(after)
            * 96
            / max(window - min((r["t_rel"] for r in after), default=0), 1e-9),
            1,
        )
        if after
        else None,
        "timeline_samples": len(timeline),
        "requests_failed": failed[0],
        "fail_tally_top": [
            {"status": k[0], "body": k[1], "n": v}
            for k, v in sorted(fail_tally.items(), key=lambda kv: -kv[1])[:5]
        ]
        if fail_tally
        else [],
        "wall_s": round(window, 1),
    }
    daemon.stop()
    with contextlib.suppress(Exception):
        tail = Path(daemon.log_path).read_text(errors="replace").splitlines()[-200:]
        rec["daemon_tail"] = tail
    sb.destroy()
    return rec


def run_tools_cell(eng: Engine, model_name: str) -> dict:
    """Tool-call quality lane: single-turn selection + schema adherence.

    Mirrors the run_blazar_cell skeleton (sandbox → active engine → daemon
    → healthz → probe → teardown) minus the cold-start instrumentation:
    this lane measures QUALITY, not latency of load.
    """
    os.environ["BLAZAR_VALIDATE_PORT"] = str(free_port())
    V = importlib.import_module("validate")
    V.PORT = int(os.environ["BLAZAR_VALIDATE_PORT"])
    rec: dict = {}
    sb = V.Sandbox()
    try:
        con = sqlite3.connect(Path(sb.data_home) / "blazar" / "blazar.db")
        con.execute("UPDATE engines SET active = (tag = ?)", (eng.tag,))
        con.commit()
        con.close()
        daemon = V.Daemon(sb)
        try:
            daemon.start(cfg={"port": V.PORT}, floor_model=model_name)
            deadline = time.time() + 600
            healthy = False
            while time.time() < deadline:
                try:
                    http_json(f"http://127.0.0.1:{V.PORT}/healthz", timeout=5.0)
                    healthy = True
                    break
                except json.JSONDecodeError:
                    healthy = True
                    break
                except (urllib.error.URLError, OSError):
                    time.sleep(0.5)
            if not healthy:
                return {"error": "sandbox daemon failed to boot"}
            # registry name for every engine: the gateway resolves DB
            # names at routing ('default' is engine-side only and 404s
            # through the proxy — proven live on the v0.9.3 tools cell)
            body_model = model_name
            per_scenario: list[dict] = []
            for sc in TOOL_BENCH_SCENARIOS:
                outcome = probe_tools_once(V.PORT, body_model, sc["prompt"])
                scored = score_tools_scenario(sc, outcome)
                per_scenario.append(
                    {
                        "name": sc["name"],
                        "expected_fn": sc.get("expected_fn"),
                        **{k: v for k, v in scored.items() if k != "transport_error"},
                        **(
                            {"transport_error": scored["transport_error"]}
                            if "transport_error" in scored
                            else {}
                        ),
                    }
                )
            tool_scenarios = [s for s in per_scenario if s.get("expected_fn")]
            control_scenarios = [s for s in per_scenario if not s.get("expected_fn")]
            ttfts = sorted(
                s["ttft_ms"] for s in per_scenario if s.get("ttft_ms") is not None
            )
            rec.update(
                {
                    "tools_scenarios": len(per_scenario),
                    "tools_wellformed": sum(
                        1 for s in tool_scenarios if s.get("wellformed")
                    ),
                    "tools_selection": sum(
                        1 for s in tool_scenarios if s.get("selection")
                    ),
                    "tools_args_valid": sum(
                        1 for s in tool_scenarios if s.get("args_valid")
                    ),
                    "tools_false_positive": sum(
                        1 for s in control_scenarios if s.get("false_positive")
                    ),
                    "tools_ttft_p50_ms": (ttfts[len(ttfts) // 2] if ttfts else None),
                    "tools_detail": per_scenario,
                }
            )
            if all("transport_error" in s for s in per_scenario):
                rec["error"] = (
                    "tools lane: every scenario failed transport — "
                    + str(per_scenario[0].get("transport_error"))[:160]
                )
            return rec
        finally:
            daemon.stop()
            # validate.py has no tail helper: read the daemon log the same
            # way its own autopsy code does (log_path direct read)
            with contextlib.suppress(OSError):
                with open(daemon.log_path, "rb") as f:
                    lines = f.read()[-4000:].decode("utf-8", "replace").splitlines()
                if lines:
                    rec.setdefault("daemon_tail", lines[-12:])
    finally:
        sb.destroy()


def run_blazar_conc_cell(
    eng: Engine, model_name: str, level: int, cfg: dict, rounds: int = 1
) -> dict:
    """Concurrency lane through the full gateway path (admission,
    queueing, slot leases — Blazar's scheduling surface)."""
    os.environ["BLAZAR_VALIDATE_PORT"] = str(free_port())
    V = importlib.import_module("validate")
    # F139: the module cache returns the FIRST import on later campaigns
    # — rebinding PORT on the module is what actually takes effect; the
    # env re-set above alone is inert past the first import.
    V.PORT = int(os.environ["BLAZAR_VALIDATE_PORT"])

    rec: dict = {}
    sb = V.Sandbox()
    sampler = Sampler(None)
    sampler.start()
    try:
        con = sqlite3.connect(Path(sb.data_home) / "blazar" / "blazar.db")
        con.execute("UPDATE engines SET active = (tag = ?)", (eng.tag,))
        con.commit()
        con.close()
        daemon = V.Daemon(sb)
        port: int | None = None
        try:
            # same ownership fix as run_blazar_cell: start() must be
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
# idle-wake lane (sleep-vs-expiry semantics: blazar keeps weights in RAM
# and wakes cheap; ollama's keep_alive expiry unloads and pays a reload)


def run_blazar_idle_cell(eng: Engine, model_name: str, cfg: dict) -> dict:
    """Warm the model, let the reaper ladder sleep it (idle_sleep_secs),
    then measure the wake TTFT — blazar's structural idle advantage."""
    os.environ["BLAZAR_VALIDATE_PORT"] = str(free_port())
    V = importlib.import_module("validate")
    V.PORT = int(os.environ["BLAZAR_VALIDATE_PORT"])

    idle_sleep = 15
    rec: dict[str, Any] = {
        "idle_policy": f"sleep at {idle_sleep}s (weights stay RAM-resident)",
    }
    sb = V.Sandbox()
    sampler = Sampler(None)
    sampler.start()
    try:
        con = sqlite3.connect(Path(sb.data_home) / "blazar" / "blazar.db")
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
            # sleep detect: /api/ps blazar_state flips to Sleeping (the
            # child sleeps itself; weights stay RAM, VRAM released);
            # VRAM drop as fallback signal. Timeout must cover the idle
            # window + the 10s reaper tick + margin.
            slept = False
            deadline = time.time() + idle_sleep + 10 + 60
            while time.time() < deadline:
                try:
                    ps = http_json(f"http://127.0.0.1:{port}/api/ps", timeout=5.0)
                    states = [
                        str(r.get("blazar_state", "")).lower()
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
    complete reload — measure that TTFT against blazar's sleep-wake."""
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


def run_blazar_ctx_cell(eng: Engine, model_name: str, ctx: int, cfg: dict) -> dict:
    """One ctx point on the curve: sandbox daemon with the per-model ctx
    override, 3-run decode suite. The profile compiler resolves ctx into
    the child argv (recorded) so the exact allocation is in the artifact."""
    os.environ["BLAZAR_VALIDATE_PORT"] = str(free_port())
    V = importlib.import_module("validate")
    V.PORT = int(os.environ["BLAZAR_VALIDATE_PORT"])

    rec: dict[str, Any] = {"ctx": ctx}
    sb = V.Sandbox()
    sampler = Sampler(None)
    sampler.start()
    try:
        con = sqlite3.connect(Path(sb.data_home) / "blazar" / "blazar.db")
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
    → first token. Mirrors the blazar cold probe exactly (fadvise'd
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
        # num_ctx 16384 = the blazar gateway cell's resolved ctx for the
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
        str(model.resolve()),
        "-f",
        # llama-perplexity runs with cwd=eng.dir: a repo-relative corpus
        # path is unresolvable there (same class as the mistral.rs
        # staging bug) — always hand the child an absolute path
        str(corpus.resolve()),
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
    errfh = open(errfh_path, "wb")  # noqa: SIM115 — stderr sink handed to Popen, lives with the child
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
    for got, ref in zip(gots, refs, strict=True):
        if got == ref:
            exact += 1
        ratios.append(difflib.SequenceMatcher(None, ref, got).ratio())
        first_div.append(
            next(
                (k for k, (a, b) in enumerate(zip(ref, got, strict=False)) if a != b),
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
    os.environ["BLAZAR_VALIDATE_PORT"] = str(free_port())
    V = importlib.import_module("validate")
    # F139: the module cache returns the FIRST import on later campaigns
    # — rebinding PORT on the module is what actually takes effect; the
    # env re-set above alone is inert past the first import.
    V.PORT = int(os.environ["BLAZAR_VALIDATE_PORT"])

    rec: dict = {}
    sb = V.Sandbox()
    try:
        con = sqlite3.connect(Path(sb.data_home) / "blazar" / "blazar.db")
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


def cli_flags(binary: Path | None, sub: list[str]) -> set[str]:
    if binary is None:
        return set()
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
    """Resolved slot/context shape of a blazar row, from its recorded
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
    co-residency abort as a blazar defect."""
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
        if r.get("provider") != "blazar" or "decode_tps_p50" not in r or tag is None:
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
        if r.get("provider") != "conc-blazar" or "sys_tps" not in r or tag is None:
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
    blazar_version: str = "unknown",
) -> None:
    """Human-first markdown report: environment, speed, resources,
    concurrency, quality, features, failures."""
    md: list[str] = []
    md.append("# Blazar benchmark matrix\n")
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
    md.append(f"- **blazar**: `{blazar_version}` (sandbox daemon binary)")
    # provenance disclosure: resumed campaigns mix rows measured by
    # different binaries — enumerate the stamps actually in the records
    blazar_owned = [
        r
        for r in records
        if r.get("provider") in ("blazar", "conc-blazar", "greedy_gw")
    ]
    stamps = sorted({v for r in blazar_owned if (v := r.get("blazar_version"))})
    unstamped = sum(1 for r in blazar_owned if not r.get("blazar_version"))
    if len(stamps) > 1 or (stamps and stamps != [blazar_version]):
        md.append(
            f"- ⚠ **mixed provenance**: blazar-owned rows were measured by "
            f"{', '.join(f'`{s}`' for s in stamps)}; this invocation used "
            f"`{blazar_version}`. Per-row `blazar_version` in cells.jsonl."
        )
    elif stamps == [blazar_version]:
        md.append(f"- all blazar-owned rows measured by `{stamps[0]}`")
    if unstamped:
        md.append(
            f"- {unstamped} blazar-owned row(s) predate version stamping "
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
            "## Quality — gateway transparency (blazar path vs direct, same engine)\n"
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
                "\n**environment** (box/co-residency guards — NOT blazar defects):\n"
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
        "- `blazar` = full gateway path inside a sandboxed daemon"
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
        "- blazar speed rows show the resolved slot/context shape"
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
    ap.add_argument("--data-dir", default=str(Path.home() / ".local/share/blazar"))
    ap.add_argument("--model", help="model name substring (default: largest .gguf)")
    ap.add_argument("--engines", nargs="*", help="engine tags (default: all)")
    ap.add_argument(
        "--providers",
        nargs="*",
        default=["direct", "blazar", "ollama"],
        choices=["direct", "blazar", "ollama"],
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
        "--soak", type=float, default=0.0, help="blazar soak seconds (0=off)"
    )
    ap.add_argument("--skip-ppl", action="store_true")
    ap.add_argument("--skip-greedy", action="store_true")
    ap.add_argument(
        "--skip-tools",
        action="store_true",
        help="skip the tool-call quality lane (single-turn selection + schema)",
    )
    ap.add_argument(
        "--skip-reshape",
        action="store_true",
        help="skip the no-lag adaptive-reshape lane (sustained C=8)",
    )
    ap.add_argument(
        "--reshape-seconds",
        type=int,
        default=300,
        help="sustained-load duration for the reshape lane",
    )
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
    ap.add_argument(
        "--skip-media",
        action="store_true",
        help="skip the media lanes (image/video/gate/tts/whisper)",
    )
    ap.add_argument(
        "--media-runs",
        type=int,
        default=DEFAULT_MEDIA_RUNS,
        help="runs per media lane point (median + min/max reported)",
    )
    ap.add_argument(
        "--media-image-model",
        help="image-lane model id (default: auto-resolve from store by name)",
    )
    ap.add_argument(
        "--media-video-model",
        help="video-lane model id (default: auto-resolve from store by name)",
    )
    ap.add_argument(
        "--media-only",
        action="store_true",
        help=(
            "run only the media lanes (image/video/gate/tts/whisper): clears "
            "the provider sweep and sets every text-lane skip flag — the "
            "shorthand for a media-focused session (still needs a text model "
            "row in the store for harness bookkeeping)"
        ),
    )
    ap.add_argument("--corpus", help="local corpus .parquet/.txt for perplexity")
    ap.add_argument(
        "--fresh", action="store_true", help="ignore+replace existing cells.jsonl"
    )
    ap.add_argument("--artifacts-dir", help="override artifacts location")
    ap.add_argument(
        "--blazar-bin",
        help=(
            "blazar binary for sandbox daemons (default: repo release build, "
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

    # --media-only: one flag instead of the eight a media-focused session
    # would otherwise repeat. Applied post-parse so --skip-media stays
    # orthogonal and the combination is caught loudly instead of running
    # an empty campaign that "succeeds".
    if args.media_only:
        if args.skip_media:
            log("--media-only and --skip-media together: nothing would run")
            return 2
        args.providers = []
        args.skip_ppl = True
        args.skip_greedy = True
        args.skip_features = True
        args.skip_conc = True
        args.skip_idle = True
        args.skip_ctxcurve = True
        args.skip_variants = True

    if args.render_only:
        cache = Path.home() / ".cache/blazar-bench-matrix"
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
    db = data_dir / "blazar.db"
    if not db.exists():
        log(f"no blazar.db under {data_dir} — nothing servable")
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
    # text-capable kinds only — sdcpp/whisper rows exist for the media
    # lanes and must never be fed to the model-serving speed lanes
    text_engines = [e for e in engines if e.kind in ("llamacpp", "mistralrs", "sglang")]
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
    # validate import (PAL resolves at import time from BLAZAR_BIN).
    if args.blazar_bin:
        pbin = Path(args.blazar_bin).resolve()
        if not pbin.is_file() or not os.access(pbin, os.X_OK):
            log(f"fatal: --blazar-bin {pbin} is not an executable file")
            return 2
        os.environ["BLAZAR_BIN"] = str(pbin)
    try:
        probe = subprocess.run(
            [os.environ.get("BLAZAR_BIN", "blazar"), "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        vout = (probe.stdout + probe.stderr).strip()
        blazar_version = vout.splitlines()[0] if vout else "unknown"
    except OSError:
        blazar_version = "unknown"
    log(f"blazar sandbox binary: {blazar_version}")

    # variant axes are emitted only when the child binary supports the
    # flags (probed, not assumed)
    eng_flags: dict[str, set[str]] = {}
    for eng in engines:
        if eng.server is None:
            # media kinds (sdcpp/whisper): no introspectable server
            # binary on the harness path — the gateway owns the spawn
            eng_flags[eng.tag] = set()
        elif eng.kind == "mistralrs":
            eng_flags[eng.tag] = cli_flags(eng.server, ["serve", "--help"])
        else:
            eng_flags[eng.tag] = cli_flags(eng.server, ["--help"])

    failures = 0
    records: list[dict] = []

    def emit(tag, kind, provider, params, key, rec, model_name_override=None):
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
            # media lanes bench their own models (image/video/voice);
            # the text model name would be a lie on those rows
            "model": model_name_override or model.name,
            # provenance stamps: rows survive across reruns in one
            # cells.jsonl — a row must carry WHICH binary measured it
            "blazar_version": blazar_version,
            "measured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            # env stamp: 5-min loadavg covers the cell window; a contended
            # run (rust-analyzer, parallel builds) is diagnosable later
            "loadavg_5m": Path("/proc/loadavg").read_text().split()[1],
            "power_state": ("battery" if power_state()["on_battery"] else "ac"),
            **rec,
        }
        append_record(cells_path, rec2)
        records.append(rec2)
        if provider == "inventory":
            log(
                f"  ok: engine inventory stamped "
                f"({len(rec.get('engines', []))} benchable, "
                f"{len(rec.get('excluded', {}))} excluded)"
            )
        elif "error" in rec:
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

    # ---- engine inventory stamp: the coverage audit trail. Every store
    # engine appears in the artifact either as cells or as an exclusion
    # reason, so "did we bench everything?" is answerable from the receipt.
    if "engine-inventory" not in done:
        db = data_dir / "blazar.db"
        store_rows: list[tuple[str, str]] = []
        if db.exists():
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            try:
                store_rows = list(con.execute("SELECT tag, kind FROM engines"))
            finally:
                con.close()
        benchable = {e.tag for e in engines}
        inv_engines = [
            {"tag": tag, "kind": kind or "llamacpp"}
            for tag, kind in store_rows
            if tag in benchable
        ]
        excluded = {
            tag: ENGINE_EXCLUSIONS.get(
                kind or "llamacpp", f"kind '{kind}' has no bench lane in this harness"
            )
            for tag, kind in store_rows
            if tag not in benchable
        }
        emit(
            "inventory",
            "inventory",
            "inventory",
            {},
            "engine-inventory",
            {"engines": inv_engines, "excluded": excluded},
        )

    # ---- direct provider sweep (ctx x np + variant axes)
    if "direct" in args.providers:
        for eng in text_engines:
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
                except Exception as exc:
                    rec = {"error": f"direct cell crashed: {exc}"}
                emit(eng.tag, eng.kind, "direct", params, key, rec)

    # ---- blazar provider (default-config cell per engine + mistral.rs
    # paged-attn-off variant + soak)
    if "blazar" in args.providers:
        for eng in text_engines:
            params = {"config": "default"}
            key = cell_key(eng.tag, "blazar", params, model.name)
            if key in done:
                log(f"[blazar {eng.tag}] resumed — skipping")
                continue
            log(f"[blazar {eng.tag}] (sandbox, gateway, default profile)")
            # guard BEFORE the cell: a prior direct-sweep teardown can
            # still hold VRAM when the sandbox child spawns (live-caught
            # 2026-09-11: 502 right after the np4 direct cells)
            if not mem_guard(2048.0, f"pre-blazar {eng.tag}"):
                emit(
                    eng.tag,
                    eng.kind,
                    "blazar",
                    params,
                    key,
                    {"error": "GPU memory floor exceeded before cell"},
                )
                continue
            try:
                rec = run_blazar_cell(
                    eng, gw_model_name, cfg, "sandboxed gateway cell", soak_s=args.soak
                )
            except Exception as exc:
                rec = {"error": f"blazar cell crashed: {exc}"}
            emit(eng.tag, eng.kind, "blazar", params, key, rec)
            if eng.kind == "mistralrs" and not args.skip_variants:
                params = {"config": "paged_attn_off"}
                key = cell_key(eng.tag, "blazar", params, model.name)
                if key in done:
                    continue
                log(f"[blazar {eng.tag}] (sandbox, gateway, paged-attn off)")
                try:
                    rec = run_blazar_cell(
                        eng,
                        gw_model_name,
                        cfg,
                        "sandboxed gateway cell, mistralrs_paged_attn=false",
                        blazar_cfg={"mistralrs_paged_attn": False},
                    )
                except Exception as exc:
                    rec = {"error": f"blazar cell crashed: {exc}"}
                emit(eng.tag, eng.kind, "blazar", params, key, rec)

            # ---- blazar single-stream variant (slots=1, classic in-VRAM
            # KV): the same-settings cell for the ollama parity question —
            # ollama serves one slot with KV in VRAM; this pins blazar to
            # the identical layout so any remaining delta is orchestration,
            # not defaults policy. (Inside the engine loop on purpose: it
            # is per-engine and reads eng — an earlier revision left it
            # outside, running once on the leftover loop variable.)
            params = {"config": "single-stream"}
            key = cell_key(eng.tag, "blazar", params, model.name)
            if key in done:
                log(f"[blazar {eng.tag} single-stream] resumed — skipping")
            else:
                log(
                    f"[blazar {eng.tag} single-stream] (sandbox, slots=1, kv_unified=false)"
                )
                try:
                    rec = run_blazar_cell(
                        eng,
                        gw_model_name,
                        cfg,
                        "sandboxed gateway cell, slots=1 + kv_unified=false",
                        blazar_cfg={"slots": 1, "kv_unified": False},
                    )
                except Exception as exc:
                    rec = {"error": f"blazar cell crashed: {exc}"}
                emit(eng.tag, eng.kind, "blazar", params, key, rec)

        params = {"reference": True}
        key = cell_key("ollama-host", "ollama", params, model.name)
        if key in done:
            log("[ollama] resumed — skipping")
        else:
            log("[ollama reference]")
            try:
                rec = run_ollama_cell(cfg, args.model or model_name)
            except Exception as exc:
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
            except Exception as exc:
                rec = {"error": f"ollama cold cell crashed: {exc}"}
            emit("ollama-host", "ollama", "cold-ollama", params, key, rec)

    # ---- media lanes (image / video+gate / tts / whisper): sandboxed
    # gateway families; each lane resolves its own model from the store
    # and stamps GPU/RAM census at entry (warm children by design).
    if not args.skip_media:
        sdcpp_eng = next((e for e in engines if e.kind == "sdcpp"), None)
        whisper_eng = next((e for e in engines if e.kind == "whisper"), None)
        media_cfg = {"runs": args.media_runs, "art_dir": str(art)}

        def store_models() -> dict[str, list]:
            db = Path(args.data_dir).expanduser() / "blazar.db"
            if not db.exists():
                return {}
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            try:
                rows = {
                    n: (json.loads(c) if c and c != "null" else [])
                    for n, c in con.execute("SELECT name, components FROM models")
                }
            finally:
                con.close()
            return rows

        store = store_models()

        def resolve_media_model(prefer: str) -> str | None:
            hits = [n for n in store if prefer in n.lower()]
            return min(hits) if hits else None

        if sdcpp_eng is not None:
            image_model = (
                args.media_image_model
                or resolve_media_model("image")
                or resolve_media_model("flux")
                or resolve_media_model("stable-diffusion")
            )
            video_model = args.media_video_model or resolve_media_model("wan")
            if image_model:
                params = {
                    "size": MEDIA_IMAGE_SIZE,
                    "steps": list(MEDIA_IMAGE_STEPS),
                    "runs": media_cfg["runs"],
                }
                key = cell_key(sdcpp_eng.tag, "media-image", params, image_model)
                if key in done:
                    log("[media image] resumed — skipping")
                else:
                    log(
                        f"[media image] {image_model} {MEDIA_IMAGE_SIZE} steps={MEDIA_IMAGE_STEPS}"
                    )
                    try:
                        rec = run_media_image_cell(sdcpp_eng, image_model, media_cfg)
                    except Exception as exc:
                        rec = {"error": f"media image cell crashed: {exc}"}
                    emit(
                        sdcpp_eng.tag,
                        "sdcpp",
                        "media-image",
                        params,
                        key,
                        rec,
                        model_name_override=image_model,
                    )
            else:
                log("[media image] no diffusion model resolved in store — skipping")
            if video_model:
                if not mem_guard(MEDIA_VIDEO_MEM_FLOOR_MIB, "pre-media-video"):
                    log("[media video] RAM floor exceeded — skipping family")
                else:
                    params = {
                        "size": MEDIA_VIDEO_SIZE,
                        "frames": list(MEDIA_VIDEO_FRAMES),
                        "steps": MEDIA_VIDEO_STEPS,
                        "runs": media_cfg["runs"],
                        "gate": dict(MEDIA_GATE_MONSTER),
                    }
                    key = cell_key(sdcpp_eng.tag, "media-video", params, video_model)
                    if key in done:
                        log("[media video] resumed — skipping")
                    else:
                        log(
                            f"[media video] {video_model} {MEDIA_VIDEO_SIZE} "
                            f"frames={MEDIA_VIDEO_FRAMES} + gate probe"
                        )
                        try:
                            rec = run_media_video_cell(
                                sdcpp_eng, video_model, media_cfg
                            )
                        except Exception as exc:
                            rec = {"error": f"media video cell crashed: {exc}"}
                        emit(
                            sdcpp_eng.tag,
                            "sdcpp",
                            "media-video",
                            params,
                            key,
                            rec,
                            model_name_override=video_model,
                        )
            else:
                log("[media video] no video model resolved in store — skipping")
        else:
            log("[media] no sdcpp engine installed — image/video lanes skipped")

        voices_dir = Path(args.data_dir).expanduser() / "voices"
        if voices_dir.is_dir() and any(voices_dir.iterdir()):
            params = {
                "chars": MEDIA_TTS_CHARS,
                "formats": ["wav", "pcm"],
                "runs": media_cfg["runs"],
            }
            key = cell_key("piper", "media-tts", params, "en_US-amy-medium")
            if key in done:
                log("[media tts] resumed — skipping")
            else:
                log(f"[media tts] en_US-amy-medium {MEDIA_TTS_CHARS} chars wav+pcm")
                try:
                    rec = run_media_tts_cell(media_cfg)
                except Exception as exc:
                    rec = {"error": f"media tts cell crashed: {exc}"}
                emit(
                    "piper",
                    "piper",
                    "media-tts",
                    params,
                    key,
                    rec,
                    model_name_override="en_US-amy-medium",
                )

            conc_params = {
                "streams": MEDIA_TTS_CONC_STREAMS,
                "chars": MEDIA_TTS_CONC_CHARS,
                "format": "pcm",
            }
            conc_key = cell_key(
                "piper", "media-tts-conc", conc_params, "en_US-amy-medium"
            )
            if conc_key in done:
                log("[media tts-conc] resumed — skipping")
            else:
                log(f"[media tts-conc] {MEDIA_TTS_CONC_STREAMS} parallel pcm streams")
                try:
                    rec_c = run_media_tts_concurrency_cell(media_cfg)
                except Exception as exc:
                    rec_c = {"error": f"media tts-conc cell crashed: {exc}"}
                emit(
                    "piper",
                    "piper",
                    "media-tts-conc",
                    conc_params,
                    conc_key,
                    rec_c,
                    model_name_override="en_US-amy-medium",
                )
        else:
            log("[media tts] no voices pulled — skipping")

        whisper_models = Path(args.data_dir).expanduser() / "whisper" / "models"
        if whisper_eng is not None and any(whisper_models.glob("ggml-*.bin")):
            params = {"runs": media_cfg["runs"], "input": "piper-wav"}
            key = cell_key(whisper_eng.tag, "media-whisper", params, "ggml-base")
            if key in done:
                log("[media whisper] resumed — skipping")
            else:
                log("[media whisper] transcribe piper-synthesized wav")
                try:
                    rec = run_media_whisper_cell(whisper_eng, media_cfg)
                except Exception as exc:
                    rec = {"error": f"media whisper cell crashed: {exc}"}
                emit(
                    whisper_eng.tag,
                    "whisper",
                    "media-whisper",
                    params,
                    key,
                    rec,
                    model_name_override="ggml-base",
                )
        else:
            log("[media whisper] no whisper engine+model pair — skipping")

    # ---- idle-wake lane (sleep-vs-expiry: the idle-policy headline)
    if not args.skip_idle:
        if "blazar" in args.providers:
            for eng in text_engines:
                params = {"idle": True}
                key = cell_key(eng.tag, "idle-blazar", params, model.name)
                if key in done:
                    continue
                log(f"[idle-wake blazar {eng.tag}]")
                if not mem_guard(2048.0, f"pre-idle {eng.tag}"):
                    emit(
                        eng.tag,
                        eng.kind,
                        "idle-blazar",
                        params,
                        key,
                        {"error": "GPU memory floor exceeded before cell"},
                    )
                    continue
                try:
                    rec = run_blazar_idle_cell(eng, gw_model_name, cfg)
                except Exception as exc:
                    rec = {"error": f"idle cell crashed: {exc}"}
                emit(eng.tag, eng.kind, "idle-blazar", params, key, rec)
        if "ollama" in args.providers:
            params = {"idle": True}
            key = cell_key("ollama-host", "idle-ollama", params, model.name)
            if key in done:
                log("[idle-wake ollama] resumed — skipping")
            else:
                log("[idle-wake ollama (keep_alive expiry)]")
                try:
                    rec = run_ollama_idle_cell(cfg, args.model or model_name)
                except Exception as exc:
                    rec = {"error": f"ollama idle cell crashed: {exc}"}
                emit("ollama-host", "ollama", "idle-ollama", params, key, rec)

    # ---- long-context degradation curve (decode t/s + TTFT vs ctx)
    if not args.skip_ctxcurve:
        ctxcurve = tuple(
            int(x) for x in str(args.ctxcurve_sweep).split(",") if x.strip()
        )
        if "blazar" in args.providers:
            for eng in text_engines:
                for ctx in ctxcurve:
                    params = {"ctx": ctx}
                    key = cell_key(eng.tag, "ctxcurve-blazar", params, model.name)
                    if key in done:
                        continue
                    log(f"[ctxcurve blazar {eng.tag} ctx={ctx}]")
                    if not mem_guard(2048.0, f"pre-ctxcurve {eng.tag} {ctx}"):
                        emit(
                            eng.tag,
                            eng.kind,
                            "ctxcurve-blazar",
                            params,
                            key,
                            {"error": "GPU memory floor exceeded before cell"},
                        )
                        continue
                    try:
                        rec = run_blazar_ctx_cell(eng, gw_model_name, ctx, cfg)
                    except Exception as exc:
                        rec = {"error": f"ctxcurve cell crashed: {exc}"}
                    emit(eng.tag, eng.kind, "ctxcurve-blazar", params, key, rec)
        if "ollama" in args.providers:
            for ctx in ctxcurve:
                params = {"ctx": ctx}
                key = cell_key("ollama-host", "ctxcurve-ollama", params, model.name)
                if key in done:
                    continue
                log(f"[ctxcurve ollama ctx={ctx}]")
                try:
                    rec = run_ollama_ctx_cell(cfg, args.model or model_name, ctx)
                except Exception as exc:
                    rec = {"error": f"ollama ctxcurve cell crashed: {exc}"}
                emit("ollama-host", "ollama", "ctxcurve-ollama", params, key, rec)

    # ---- concurrency lane (direct np-sized child + full gateway path)
    if not args.skip_conc and conc_sweep:
        for level in conc_sweep:
            if "direct" in args.providers:
                for eng in text_engines:
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
                    except Exception as exc:
                        rec = {"error": f"conc cell crashed: {exc}"}
                    emit(eng.tag, eng.kind, "conc-direct", params, key, rec)
            if "blazar" in args.providers:
                for eng in text_engines:
                    params = {"conc": level, "rounds": args.conc_rounds}
                    key = cell_key(eng.tag, "conc-blazar", params, model.name)
                    if key in done:
                        continue
                    log(f"[conc blazar {eng.tag} x{level} x{args.conc_rounds}r]")
                    if not mem_guard(2048.0, f"pre-conc-blazar {eng.tag}"):
                        emit(
                            eng.tag,
                            eng.kind,
                            "conc-blazar",
                            params,
                            key,
                            {"error": "GPU memory floor exceeded before cell"},
                        )
                        continue
                    try:
                        rec = run_blazar_conc_cell(
                            eng, gw_model_name, level, cfg, rounds=args.conc_rounds
                        )
                    except Exception as exc:
                        rec = {"error": f"conc cell crashed: {exc}"}
                    emit(eng.tag, eng.kind, "conc-blazar", params, key, rec)
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
                except Exception as exc:
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
        for eng in text_engines:
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
            except Exception as exc:
                rec = {"error": f"ppl cell crashed: {exc}"}
            emit(eng.tag, eng.kind, "ppl", params, key, rec)

    # ---- quality: tool-call selection + schema (single-turn, temp 0)
    # ---- no-lag reshape lane: sustained C=8 proves graceful-drain adoption
    if not args.skip_reshape and "blazar" in args.providers:
        for eng in text_engines:
            key = cell_key(eng.tag, "reshape", {"reshape": True}, model.name)
            if key in done:
                log(f"[resumed] reshape {eng.tag}")
                continue
            if not mem_guard(2048.0, f"pre-reshape {eng.tag}"):
                emit(
                    eng.tag,
                    eng.kind,
                    "reshape",
                    {"reshape": True},
                    key,
                    {"error": "GPU floor refused the reshape cell"},
                )
                continue
            log(
                f"  reshape lane: {eng.tag} sustained C=8 ({int(args.reshape_seconds)}s)"
            )
            try:
                rec = run_reshape_cell(eng, gw_model_name, float(args.reshape_seconds))
            except Exception as exc:  # receipt, not silence
                rec = {
                    "error": f"reshape cell crashed: {exc}",
                    "traceback": traceback.format_exc().splitlines()[-6:],
                }
            emit(eng.tag, eng.kind, "reshape", {"reshape": True}, key, rec)

    if not args.skip_tools and "blazar" in args.providers:
        for eng in text_engines:
            key = cell_key(eng.tag, "tools", {"tools": True}, model.name)
            if key in done:
                log(f"[tools {eng.tag}] resumed — skipping")
                continue
            if not mem_guard(2048.0, f"pre-tools {eng.tag}"):
                emit(
                    eng.tag,
                    eng.kind,
                    "tools",
                    {"tools": True},
                    key,
                    {"error": "mem_guard: GPU too busy for tools lane"},
                )
                continue
            log(f"[tools {eng.tag}]")
            try:
                # registry name, not the file stem: the gateway resolves
                # body models against DB names (file stem = 404)
                rec = run_tools_cell(eng, gw_model_name)
            except Exception as exc:  # cell crash is a recorded receipt
                rec = {"error": f"tools cell crashed: {exc}"}
            emit(eng.tag, eng.kind, "tools", {"tools": True}, key, rec)
        if "ollama" in args.providers:
            # direct OpenAI-compat probe against the ollama daemon: no
            # gateway in the path, the reference is ollama's own tools
            # handling on the same model family
            key = cell_key("ollama", "tools-ollama", {"tools": True}, model.name)
            if key in done:
                log("[tools ollama] resumed — skipping")
            else:
                log("[tools ollama]")
                per: list[dict] = []
                for sc in TOOL_BENCH_SCENARIOS:
                    outcome = probe_tools_once(11434, "qwen3.5:9b", sc["prompt"])
                    per.append(score_tools_scenario(sc, outcome))
                if per and all(p.get("transport_error") for p in per):
                    rec: dict = {"error": "ollama unreachable for tools lane"}
                else:
                    tts = sorted(
                        p["ttft_ms"] for p in per if p.get("ttft_ms") is not None
                    )
                    rec = {
                        "tools_scenarios": len(per),
                        "tools_wellformed": sum(1 for p in per if p.get("wellformed")),
                        "tools_selection": sum(
                            1 for p in per if p.get("selection") is True
                        ),
                        "tools_args_valid": sum(
                            1 for p in per if p.get("args_valid") is True
                        ),
                        "tools_false_positive": any(
                            p.get("false_positive") for p in per
                        ),
                        "tools_ttft_p50_ms": (tts[len(tts) // 2] if tts else None),
                        "tools_detail": per,
                    }
                emit(
                    "ollama-host",
                    "ollama",
                    "tools-ollama",
                    {"tools": True},
                    key,
                    rec,
                )

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
            for eng in text_engines:
                params = {"greedy": True}
                key = cell_key(eng.tag, "greedy", params, model.name)
                if key in done:
                    continue
                log(f"[greedy parity {eng.tag}]")
                try:
                    rec = run_greedy_parity(
                        eng, model, own_mmproj, model_name, reference, stage_root
                    )
                except Exception as exc:
                    rec = {"error": f"greedy cell crashed: {exc}"}
                if "error" not in rec:
                    rec["vs"] = ref_tag or (ref_eng.tag if ref_eng else "unknown")
                emit(eng.tag, eng.kind, "greedy", params, key, rec)
            # gateway-transparency lane: same-engine direct vs gateway
            for eng in text_engines:
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
                except Exception as exc:
                    rec = {"error": f"greedy gw cell crashed: {exc}"}
                emit(eng.tag, eng.kind, "greedy_gw", params, key, rec)

    # ---- features matrix (persisted as cells so resumed campaigns
    # render the complete matrix)
    feat_rows: dict[str, dict[str, bool]] | None = None
    if not args.skip_features:
        log("[features]")
        rows: dict[str, dict[str, bool]] = {}
        for eng in text_engines:
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
        for eng in text_engines:
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

    argv_summary = portable_path(" ".join(sys.argv[1:]))
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
        blazar_version,
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
    "b11070-cuda": "llama.cpp b11070 (CUDA build)",
    "master-890-74988b2": "stable-diffusion.cpp master-890 (Vulkan)",
    "b5130": "whisper.cpp b5130",
    "piper": "piper (gateway TTS lane)",
}

# Hardware rows describe the measuring host (this repo's reference box);
# the volatile rows (runtimes, model) are DERIVED from the cells at render
# time so a re-render can never publish a stale context next to fresh
# numbers.
TEST_BED = [
    ("CPU", "Intel Core i7-14650HX, 24 hardware threads"),
    ("Discrete GPU", "NVIDIA GeForce RTX 4070 Laptop, 8 GiB, driver 580.173.02"),
    ("Integrated GPU", "Intel Graphics (RPL-S), Vulkan device"),
    ("RAM", "16 GiB (13.3 GiB usable)"),
    ("OS", "Linux Mint 22.3, kernel 7.0.0-31-generic"),
]


def derived_test_bed_rows(recs: list[dict], blazar_ver: str) -> list[tuple[str, str]]:
    """Runtimes and model come from the cells themselves, never hardcoded."""
    engines = sorted({engine_label(r.get("tag", "?")) for r in recs if r.get("tag")})
    models = sorted({r.get("model", "?") for r in recs if r.get("model")})
    return [
        ("Runtimes compared", f"blazar {blazar_ver} gateway - " + " - ".join(engines)),
        ("Model", ", ".join(models)),
    ]


METHODOLOGY = [
    "All lanes speak the OpenAI-compatible streaming API; tokens are counted from usage chunks (engine-injected at the gateway), never estimated from chunk counts.",
    "Decode throughput = (tokens - 1) / (last-token time - TTFT); medians over 5 runs after a warmup request.",
    "Inter-token latency (ITL) p50/p99 from per-chunk timestamps; TTFT p50/p90/p99 + stdev.",
    "Prefill: a token-targeted prompt (~512 tokens via engine /tokenize); run 1 is the cold (uncached) prefill, runs 2+ ride the prompt cache.",
    "Concurrency: N parallel streams x 128 generated tokens each (levels via --conc-sweep, per-row `C` column); system t/s = total tokens / wall clock; sum-stream t/s = sum of per-stream rates (sum >> system indicates serialization).",
    "Greedy parity: 20 fixed prompts, greedy sampling, 256 tokens; exact-match count and text-similarity ratio vs a same-engine reference run.",
    "Tool calls: 6 single-turn scenarios (3-tool set: weather/calculate/flights), temperature 0, max 192 tokens (call JSON must complete); scored on well-formed calls, correct function selection, valid JSON arguments with required keys, and a no-tool control for false positives; stream deltas accumulated per OpenAI spec.",
    "Gateway transparency: a second greedy lane through the blazar gateway with identical sampling; any divergence vs the direct lane isolates translation overhead.",
    "Perplexity: llama-perplexity on an offline ASCII corpus, ctx 2048.",
    "Cold-start parity: the model file's page cache is dropped (posix_fadvise DONTNEED) and the GPU asserted idle (<512 MiB) before every cold probe on every runtime — a cold load is disk-cold, not memory-warm.",
    "Cold TTFT = first-token latency of the cold probe itself (max_tokens 4, aligned num_ctx 16384 on both runtimes).",
    "ollama daemon boot is only measured with --ollama-service-restart (systemd restart, sudo password via BENCH_SUDO_PASSWORD env, stdin-only); without it the daemon stays warm and the row says so.",
    "Idle-wake: blazar's reaper sleeps the child at idle_sleep_secs (weights stay RAM-resident, VRAM released) — wake TTFT is a sleep-wake; ollama's keep_alive expiry fully unloads — wake TTFT is a disk reload. The policy column names the semantic; both measured after the policy is observed via /api/ps.",
    "Long-context curve: per-ctx cells (blazar model_overrides ctx / ollama num_ctx) x 3-run decode suites; each ollama point evicts first so the runner respawns at that ctx.",
    "Sustained concurrency: sequential bursts of the parallel-stream lane (default 3 rounds); TTFT p99 aggregates every stream of every round.",
    "Every blazar row records the spawned engine's argv (slots/context shown in tables) and stamps blazar version, wall clock, 5-min load average, and AC/battery power state; GPU cells refuse to run on battery.",
    "Media lanes run in isolated sandbox daemons (same protocol as text blazar cells); the engine child spawns lazily, so each family's first request is the COLD number (spawn + weights + first artifact), labeled cold_request_s.",
    "Media TTFB = time to first BODY byte (first audio sample for streamed PCM, not response headers); buffered WAV TTFB equals its total by construction and the table says so.",
    "Video frame counts are container ground truth: the response webm is parsed for lacing-aware SimpleBlock counts and asserted against the Wan 4k+1 temporal grid (a mismatch is recorded loudly, never averaged away).",
    "The video scratch-gate probe sends one expected-rejected monster (duration 60s -> 960 aligned frames) and times the 400; the legit 5-frame pass rides the same warm child, so axis-row vs probe deltas price the gate itself.",
    "Media cells stamp GPU-busy and RAM-available at entry instead of asserting an idle GPU: a warm child from the previous family is the normal media workflow, and the receipt carries the occupancy rather than hiding it.",
    "TTS RTF = synthesis wall / audio seconds, audio duration parsed from the RIFF data-chunk length (not estimated from characters); whisper transcribes a WAV synthesized by the same campaign's piper voice, so the input is reproducible from the receipt.",
    "Image quality stamps are PIL-gated luma-domain metrics (rms contrast = luma stddev, entropy in bits, unique colors on a 256x256 downsample); when PIL is absent the row carries an honest 'skipped' note instead of a fake number, and one audit PNG per steps point is saved beside the cells for offline re-measurement.",
    "TTS concurrency probe: N parallel streamed-PCM requests through one sandboxed gateway; wall clock vs sum of per-stream totals yields an efficiency ratio (sum/wall ~ 1 means serialized, -> N means perfectly parallel), and the probe fails loudly if any stream errors or truncates.",
    "Adaptive reshape: sustained 8 concurrent streams (adoption needs a 60 s saturation streak plus a graceful drain); the child engine -np is polled from /proc every 2 s to prove the reshape landed; throughput and TTFT p50 are compared before vs after the slot transition; a dropped request anywhere fails the lane.",
]


def pfmt(x, nd=1, unit=""):
    if x is None or (isinstance(x, float) and math.isnan(x)):
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
        if r.get("provider") == "blazar" and "error" not in r:
            rows.append(
                (
                    speed_row_name(r, f"blazar gateway - {engine_label(r['tag'])}"),
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


def speed_row_name(rec: dict, base: str) -> str:
    # distinct spawn configs (single-stream pin, PA-off variant, ...) must
    # not render as unlabeled near-duplicate rows
    cfg = rec.get("params", {}).get("config")
    return f"{base} ({cfg})" if cfg and cfg != "default" else base


def conc_table(recs: list[dict]) -> str:
    rows = []
    for r in recs:
        prov = r.get("provider")
        if prov not in ("conc-blazar", "conc-direct", "conc-ollama") or "error" in r:
            continue
        if prov == "conc-blazar":
            name = f"blazar gateway - {engine_label(r['tag'])}"
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


def reshape_table(recs: list[dict]) -> str:
    rows = [r for r in recs if r.get("provider") == "reshape" and "error" not in r]
    if not rows:
        return "_Not measured._"
    out = [
        "| Runtime | reshape | slots | time to reshape s | req before/after | TTFT p50 before→after ms | sys t/s before→after | failed |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(rows, key=lambda r: r.get("tag") or ""):
        name = f"blazar gateway - {engine_label(r.get('tag', ''))}"
        reshaped = "yes" if r.get("reshape_observed") else "NO"
        slots = f"{r.get('slots_from') or '-'}→{r.get('slots_to') or '-'}"
        ttr = r.get("time_to_reshape_s")
        ttft_b, ttft_a = r.get("ttft_p50_before_ms"), r.get("ttft_p50_after_ms")
        stps_b, stps_a = r.get("sys_tps_before"), r.get("sys_tps_after")

        def fmt(v):
            if v is None:
                return "-"
            return f"{v:.0f}" if isinstance(v, (int, float)) else str(v)

        out.append(
            f"| {name} | {reshaped} | {slots} | {fmt(ttr)} "
            f"| {r.get('requests_before', 0)}/{r.get('requests_after', 0)} "
            f"| {fmt(ttft_b)}→{fmt(ttft_a)} | {fmt(stps_b)}→{fmt(stps_a)} "
            f"| {r.get('requests_failed', 0)} |"
        )
    return "\n".join(out)


def tools_table(recs: list[dict]) -> str:
    """Single-turn tool-call quality: selection, schema, control false-rate."""
    rows = [
        r
        for r in recs
        if r.get("provider") in ("tools", "tools-ollama") and "error" not in r
    ]
    if not rows:
        return "_Not measured._"
    out = [
        "| Runtime | scenarios | well-formed | selection | args valid | control FP | TTFT p50 ms |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for r in sorted(rows, key=lambda r: (r.get("provider") or "", r.get("tag") or "")):
        if r.get("provider") == "tools-ollama":
            name = "ollama - qwen3.5:9b"
        else:
            name = f"blazar gateway - {engine_label(r.get('tag') or '')}"
        n = r.get("tools_scenarios") or 0
        scored = max(n - 1, 0)  # control scenario is not a selection case
        ttft = r.get("tools_ttft_p50_ms")
        out.append(
            f"| {name} | {n} "
            f"| {r.get('tools_wellformed', 0)}/{n} "
            f"| {r.get('tools_selection', 0)}/{scored} "
            f"| {r.get('tools_args_valid', 0)}/{scored} "
            f"| {'yes' if r.get('tools_false_positive') else 'no'} "
            f"| {pfmt(ttft, 0) if ttft is not None else '-'} |"
        )
    return "\n".join(out)


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
                    f"{engine_label(r['tag'])} through blazar gateway vs direct",
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
            f"blazar gateway - {engine_label(r['tag'])}",
            r.get("daemon_boot_s"),
            r.get("cold_first_request_s"),
            r.get("cold_ttft_ms"),
            r.get("load_s"),
            r.get("rss_peak_mib"),
        )
        for r in recs
        if r.get("provider") == "blazar" and "error" not in r
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
        if prov == "idle-blazar" and "error" not in r:
            rows.append(
                (
                    f"blazar - {engine_label(r['tag'])}",
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
        if prov == "ctxcurve-blazar" and "error" not in r:
            rows.append(
                (
                    f"blazar - {engine_label(r['tag'])}",
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
        r["tag"]: r for r in recs if r.get("provider") == "blazar" and "error" not in r
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
    for tag, gw_row in gw.items():
        if tag in direct:
            g, d = gw_row.get("decode_tps_p50"), direct[tag].get("decode_tps_p50")
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
    conc = [
        r
        for r in recs
        if r.get("provider") == "conc-blazar"
        and "error" not in r
        and r.get("sys_tps")
        and r.get("conc_level")
    ]
    if conc:
        # per engine: a multi-engine sweep must not interleave levels
        # from different runtimes into one ladder
        by_tag: dict[str, list[dict]] = {}
        for r in conc:
            by_tag.setdefault(r.get("tag") or "?", []).append(r)
        sweep_bits = []
        for tag in sorted(by_tag):
            rows_ = sorted(by_tag[tag], key=lambda r: r["conc_level"])
            levels = "/".join(str(r["conc_level"]) for r in rows_)
            ladder = "; ".join(
                f"C{r['conc_level']}: {pfmt(r['sys_tps'])} t/s system"
                f" ({child_shape(r) or 'engine-scheduled'})"
                for r in rows_
            )
            sweep_bits.append(f"{engine_label(tag)} sweep C={levels}: {ladder}")
        parts.append(" | ".join(sweep_bits))
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
            if r.get("provider") == "idle-blazar"
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
    if parts:
        return "; ".join(parts) + "."
    # Media-only campaigns have no text rows to rank — say that instead
    # of a bare "no complete rows" that reads like a failed campaign.
    media = [
        r
        for r in recs
        if (r.get("provider") or "").startswith("media-") and "error" not in r
    ]
    if media:
        return (
            f"_No complete text rows — {len(media)} media measurement(s) "
            "in the sections below._"
        )
    return "_No complete rows._"


def media_table(recs: list[dict]) -> str:
    """One row per measured media lane point; blank cells where a lane has no value."""
    rows = []
    for r in recs:
        prov = r.get("provider") or ""
        if not prov.startswith("media-") or "error" in r:
            continue
        model = r.get("model") or ""
        if prov == "media-image":
            p = r.get("params") or {}
            gt = "x".join(str(d) for d in (r.get("dims_seen") or ["?"])) + " PNG"
            qm = r.get("quality_medians") or {}
            if qm:
                gt += (
                    f", entropy {pfmt(qm.get('entropy_bits'), 1)} bits, "
                    f"contrast {pfmt(qm.get('rms_contrast'), 1)}"
                )
            elif r.get("quality_note"):
                gt += f" ({r['quality_note']})"
            rows.append(
                (
                    f"image - {model} ({p.get('size')}, steps={p.get('steps')})",
                    r.get("cold_request_s"),
                    r.get("total_s_median"),
                    r.get("total_s_min"),
                    r.get("total_s_max"),
                    gt,
                    "",
                    "",
                )
            )
        elif prov == "media-video":
            p = r.get("params") or {}
            for pt in r.get("per_frames") or []:
                runs = pt.get("runs") or []
                mism = [
                    run
                    for run in runs
                    if run.get("mux_frames") != run.get("reported_frame_count")
                    or run.get("mux_frames") != run.get("frames_requested_aligned")
                ]
                gt = (
                    f"{pt.get('frames')}f: mux==reported==aligned"
                    if not mism
                    else f"MISMATCH on {len(mism)}/{len(runs)} runs"
                )
                rows.append(
                    (
                        f"video - {model} ({p.get('size')}, steps={p.get('steps')})",
                        r.get("cold_request_s")
                        if pt is (r.get("per_frames") or [None])[0]
                        else None,
                        pt.get("total_s_median"),
                        pt.get("total_s_min"),
                        pt.get("total_s_max"),
                        gt,
                        pfmt(r.get("gate_reject_s_median"), 3)
                        if pt is (r.get("per_frames") or [None])[0]
                        and r.get("gate_reject_s_median") is not None
                        else "",
                        "",
                    )
                )
        elif prov == "media-tts":
            ground = (
                f"RTF wav {pfmt(r.get('wav_rtf'), 3)} / pcm {pfmt(r.get('pcm_rtf'), 3)} "
                f"({pfmt(r.get('audio_s'), 0)}s audio)"
            )
            rows.append(
                (
                    f"tts - {model} ({r.get('input_chars')} chars, wav+pcm)",
                    None,
                    r.get("wav_total_s_median"),
                    None,
                    r.get("pcm_total_s_median"),
                    ground,
                    "",
                    pfmt(r.get("ttfb_speedup_x"), 2) + "x"
                    if r.get("ttfb_speedup_x") is not None
                    else "",
                )
            )
        elif prov == "media-tts-conc":
            n = r.get("streams") or "?"
            ground = (
                f"efficiency {pfmt(r.get('efficiency_sum_over_wall'), 2)} "
                f"of {n} streams"
                if r.get("efficiency_sum_over_wall") is not None
                else f"uniform bytes: {r.get('bytes_uniform')}"
            )
            rows.append(
                (
                    f"tts-conc - {model} ({r.get('input_chars')} chars x{n} pcm)",
                    None,
                    r.get("per_stream_total_s_median"),
                    r.get("wall_s"),
                    None,
                    ground,
                    "",
                    pfmt(r.get("ttfb_ms_max"), 0) + "ms max TTFB"
                    if r.get("ttfb_ms_max") is not None
                    else "",
                )
            )
        elif prov == "media-whisper":
            rows.append(
                (
                    f"whisper - {model} (transcribes piper wav)",
                    r.get("cold_request_s"),
                    r.get("total_s_median"),
                    r.get("total_s_min"),
                    r.get("total_s_max"),
                    f"RTF {pfmt(r.get('rtf'), 3)}",
                    "",
                    "",
                )
            )
    if not rows:
        return "_Not measured._"
    head = "| Lane | cold s | median s | min s | max s | ground truth | gate reject s | TTFB speedup |"
    sep = "|---|---:|---:|---:|---:|---|---:|---:|"
    body = [
        f"| {n} | {pfmt(c, 2)} | {pfmt(m, 2)} | {pfmt(lo, 2)} | {pfmt(hi, 2)} | {g} | {ga} | {sp} |"
        for n, c, m, lo, hi, g, ga, sp in rows
    ]
    return "\n".join([head, sep, *body])


def campaign_scoped(table: str, lane: str, campaign: str) -> str:
    """Publication layer: bare empty markers become explicit campaign scoping.

    An empty section next to confident findings is exactly how the
    receipt/conclusion desync happened; naming the campaign that did NOT
    measure the lane keeps the artifact honest. Catches both the explicit
    markers and tables rendered as header+separator with zero body rows.
    """
    t = table.strip()
    rows = [
        line
        for line in t.splitlines()
        if line.startswith("|") and not line.startswith("|---")
    ]
    if t in ("_No complete rows._", "_Not measured._") or len(rows) < 2:
        return f"_Not measured in this campaign ({campaign}); {lane} lane not run._"
    return table


def text_findings(recs: list[dict]) -> list[tuple[str | None, str]]:
    """Text-lane findings as (backed_render | None, carried_text) pairs.

    A finding whose backing lane has cells in THIS campaign renders from a
    template with live numbers (never static prose); otherwise the original
    narrative moves to the carried-over section with its provenance note.
    """
    ok = [r for r in recs if "error" not in r]
    out: list[tuple[str | None, str]] = []

    # F1 - gateway overhead: paired gateway/direct decode + greedy parity.
    # Pair on (tag, ctx, np): a gateway row is only comparable to a direct
    # baseline of the IDENTICAL engine shape - a multi-slot gateway row must
    # never masquerade as the single-slot pair. Gateway rows run the default
    # profile, so params carries no ctx/np - the shape is read from the
    # recorded child argv (-c/--ctx-size, -np) instead, which both providers
    # stamp.
    def _shape(r):
        ctx = r.get("params", {}).get("ctx")
        np_ = r.get("params", {}).get("np")
        argv = r.get("child_argv") or []
        if ctx is None or np_ is None:
            for i, a in enumerate(argv):
                if a in ("-c", "--ctx-size") and i + 1 < len(argv):
                    with contextlib.suppress(ValueError):
                        ctx = int(argv[i + 1])
                elif a == "-np" and i + 1 < len(argv):
                    with contextlib.suppress(ValueError):
                        np_ = int(argv[i + 1])
        return (r["tag"], ctx, np_)

    gw = {_shape(r): r for r in ok if r.get("provider") == "blazar"}
    direct = {_shape(r): r for r in ok if r.get("provider") == "direct"}
    gw_greedy = next((r for r in ok if r.get("provider") == "greedy_gw"), None)
    backed = None
    pair = next(
        (
            k
            for k in gw
            if k in direct
            and gw[k].get("decode_tps_p50")
            and direct[k].get("decode_tps_p50")
        ),
        None,
    )
    if pair:
        g, d = gw[pair]["decode_tps_p50"], direct[pair]["decode_tps_p50"]
        delta = (g / d - 1) * 100
        verdict = (
            "within measurement noise"
            if abs(delta) <= 5.0
            else f"gateway overhead {delta:+.1f}%"
        )
        parity = ""
        if gw_greedy and gw_greedy.get("prompts"):
            parity = (
                f", greedy parity through the gateway {gw_greedy.get('exact_matches')}"
                f"/{gw_greedy['prompts']} exact"
            )
        backed = (
            f"**Gateway overhead vs direct spawn: {verdict}.** {engine_label(pair[0])} "
            f"decode {pfmt(g)} t/s through the gateway vs {pfmt(d)} t/s direct "
            f"({delta:+.1f}%){parity}."
        )
    out.append(
        (
            backed,
            "**Gateway overhead is within measurement noise.** Single-stream decode through "
            "the blazar gateway matches direct engine spawns at the same slots/context (see "
            "speed table); the greedy gateway lane is byte-identical to the direct lane where "
            "sampling is single-slot.",
        )
    )

    # F2 - capacity-aware slot auto-sizing: distinct observed child shapes.
    shapes = {
        child_shape(r)
        for r in ok
        if r.get("provider") in ("blazar", "conc-blazar") and child_shape(r)
    }
    backed = None
    if len(shapes) >= 2:
        backed = (
            "**Capacity-aware slot auto-sizing observed in argv.** Distinct engine shapes "
            f"this campaign: {', '.join(f'{s}' for s in sorted(shapes))} - slots follow the "
            "live hardware census, each row's child_argv carries the receipt."
        )
    out.append(
        (
            backed,
            "**Capacity-aware slot auto-sizing.** blazar sizes engine slots from live "
            "hardware census: the 8 GiB card with a vision projector attached spawns 1 slot "
            "(16 Ki context) on the Vulkan build and 4 slots (64 Ki total) on CUDA - measured "
            "oversubscription on Vulkan either fails to boot or degrades 2x, so the cap is "
            "load-bearing, not conservative cosmetics.",
        )
    )

    # F3 - concurrency scaling: per-level system throughput + efficiency.
    # Per engine: two gateway engines in one sweep must not merge into a
    # single levels list (the numbers belong to different runtimes).
    conc_by_tag: dict[str, list[dict]] = {}
    for r in ok:
        if (
            r.get("provider") == "conc-blazar"
            and r.get("sys_tps")
            and r.get("conc_level")
        ):
            conc_by_tag.setdefault(r.get("tag") or "?", []).append(r)
    backed = None
    if conc_by_tag:
        bits = []
        for tag in sorted(conc_by_tag):
            rows_ = sorted(conc_by_tag[tag], key=lambda r: r["conc_level"])
            levels = "/".join(str(r["conc_level"]) for r in rows_)
            peak = max(rows_, key=lambda r: r["sys_tps"])
            eff_note = ""
            base1 = next((r for r in rows_ if r["conc_level"] == 1), None)
            if base1 and peak["conc_level"] > 1:
                eff = peak["sys_tps"] / (peak["conc_level"] * base1["sys_tps"])
                eff_note = f", {pfmt(eff * 100, 0)}% of ideal at C={peak['conc_level']}"
            bits.append(
                f"{engine_label(tag)} (C={levels}): peak {pfmt(peak['sys_tps'])} t/s "
                f"at C={peak['conc_level']}{eff_note}"
            )
        backed = (
            "**Concurrency scaling per engine.** "
            + "; ".join(bits)
            + "; serialization behavior per level in the frontier table below."
        )
    out.append(
        (
            backed,
            "**Concurrency scales where capacity allows.** 4 streams through CUDA gateway "
            "hold near-direct system throughput; the Vulkan single-slot shape serializes "
            "streams (per-stream latency stays excellent; system throughput caps at one "
            "stream's rate) - a capacity trade, not a scheduling defect.",
        )
    )

    # F4 - prompt cache prefill ratio.
    r4 = next(
        (r for r in ok if r.get("prefill_tps_cold") and r.get("prefill_tps_cached")),
        None,
    )
    backed = None
    if r4:
        ratio = r4["prefill_tps_cached"] / r4["prefill_tps_cold"]
        backed = (
            f"**Prompt cache pays {pfmt(ratio, 1)}x on prefill** "
            f"({pfmt(r4['prefill_tps_cached'], 0)} cached vs {pfmt(r4['prefill_tps_cold'], 0)} "
            "t/s cold)."
        )
    out.append(
        (
            backed,
            "**Prompt cache pays ~6-7x on prefill.** Cached-prefix prefill runs thousands "
            "of tokens/s vs hundreds cold.",
        )
    )

    # F5/F6 - optimization axes (spec decoding, KV quantization).
    base: dict[tuple, float] = {}
    for r in ok:
        p = r.get("params", {})
        if (
            r.get("provider") == "direct"
            and p.get("ctx") == 4096
            and p.get("np") == 1
            and len(p) == 2
        ):
            base[(r["tag"], "decode")] = r.get("decode_tps_p50") or 0.0
    variants = [
        r
        for r in ok
        if r.get("provider") == "direct"
        and r.get("params", {}).get("ctx") == 4096
        and r.get("params", {}).get("np") == 1
        and len(r.get("params", {})) > 2
        and any(k in r.get("params", {}) for k in ("kv", "spec"))
    ]
    spec = next((r for r in variants if "spec" in r.get("params", {})), None)
    backed = None
    if spec and (spec["tag"], "decode") in base:
        d = spec.get("decode_tps_p50") or 0.0
        dd = d - base[(spec["tag"], "decode")]
        verdict = (
            "a net loss"
            if dd < 0
            else (
                "neutral"
                if abs(dd / max(base[(spec["tag"], "decode")], 1e-9)) <= 0.05
                else "a net gain"
            )
        )
        backed = (
            f"**Speculative n-gram decoding is {verdict} for this model** "
            f"(decode {pfmt(d)} t/s, {dd:+.1f} vs dense baseline) - measured, not assumed."
        )
    out.append(
        (
            backed,
            "**Speculative n-gram decoding is a net loss for this 9B model** (no draft "
            "model; acceptance too low to pay the verification overhead) - documented so "
            "the flag is not cargo-culted.",
        )
    )
    kv = next((r for r in variants if "kv" in r.get("params", {})), None)
    backed = None
    if kv and (kv["tag"], "decode") in base:
        d = kv.get("decode_tps_p50") or 0.0
        dd = d - base[(kv["tag"], "decode")]
        verdict = (
            "decode-neutral"
            if abs(dd / max(base[(kv["tag"], "decode")], 1e-9)) <= 0.05
            else f"{dd:+.1f} t/s vs dense"
        )
        backed = (
            f"**KV quantization ({kv.get('params', {}).get('kv')}) is {verdict}** "
            f"(decode {pfmt(d)} t/s vs {pfmt(base[(kv['tag'], 'decode')])} dense)."
        )
    out.append(
        (
            backed,
            "**KV q8_0 quantization is decode-neutral and prefill-neutral steady-state**; "
            "the one cold-prefill outlier below is a first-invocation pipeline-compile "
            "artifact (controlled re-probe measured full-rate steady state).",
        )
    )

    # F7 - mistral.rs paged-attention fit on tight cards.
    mistral = next((r for r in ok if "mistral" in engine_label(r.get("tag", ""))), None)
    backed = None
    if mistral:
        pa_refused = [
            r
            for r in recs
            if "mistral" in engine_label(r.get("tag", ""))
            and "Num GPU blocks is 0" in str(r.get("error", ""))
        ]
        refused_note = ""
        if pa_refused:
            refused_note = (
                f" default paged attention cannot fit this card "
                f"({len(pa_refused)} direct cell(s) refused at load: "
                "'Num GPU blocks is 0');"
            )
        backed = (
            f"**{engine_label(mistral['tag'])} serves this model through blazar's profile** "
            f"({refused_note} blazar auto-disables PA on tight cards and the "
            "model then serves; the row's argv is the receipt)."
        )
    out.append(
        (
            backed,
            "**mistral.rs 0.9.3 with default paged attention cannot fit this model on an "
            "8 GiB card** (upstream sizes KV as a fraction of total VRAM); blazar's profile "
            "auto-disables paged attention on tight cards and the model then serves correctly.",
        )
    )

    # F12 - tool-call selection + schema quality (single-turn, temp 0).
    tools_rows = [
        r
        for r in recs
        if r.get("provider") in ("tools", "tools-ollama") and "error" not in r
    ]
    backed = None
    if tools_rows:
        bits = []
        for r in sorted(
            tools_rows, key=lambda r: (r.get("provider") or "", r.get("tag") or "")
        ):
            if r.get("provider") == "tools-ollama":
                label = "ollama"
            else:
                label = engine_label(r.get("tag") or "")
            n = r.get("tools_scenarios") or 0
            scored = max(n - 1, 0)
            sel = r.get("tools_selection", 0)
            args_v = r.get("tools_args_valid", 0)
            fp = "control clean" if not r.get("tools_false_positive") else "control FP"
            bits.append(
                f"{label}: selection {sel}/{scored}, args {args_v}/{scored} ({fp})"
            )
        backed = (
            "**Tool-call quality (single-turn, temp 0).** "
            + "; ".join(bits)
            + "; per-scenario detail in cells.jsonl."
        )
    out.append(
        (
            backed,
            "**Gateway passes OpenAI tools verbatim** (tools-aware validation, no "
            "schema rewriting); tool-call quality is the engine's own - selection and "
            "argument schema are scored per scenario with a no-tool control for false "
            "positives.",
        )
    )

    # F13 - adaptive reshape under sustained load (the no-lag proof).
    resh = [r for r in ok if r.get("provider") == "reshape"]
    backed = None
    if resh:
        bits = []
        for r in sorted(resh, key=lambda r: r.get("tag") or ""):
            label = engine_label(r.get("tag", ""))
            if r.get("reshape_observed"):
                if r.get("requests_failed", 0) == 0:
                    bits.append(
                        f"{label}: reshaped {r.get('slots_from')}->{r.get('slots_to')} slots "
                        f"after {r.get('time_to_reshape_s')} s under sustained C=8, "
                        f"TTFT p50 {r.get('ttft_p50_before_ms'):.0f}->{r.get('ttft_p50_after_ms'):.0f} ms, "
                        f"0 dropped requests"
                    )
                else:
                    bits.append(
                        f"{label}: reshaped but {r.get('requests_failed')} request(s) dropped"
                    )
            else:
                bits.append(f"{label}: no reshape observed in the lane window")
        backed = (
            "**Adaptive reshape lands under sustained load.** "
            + "; ".join(bits)
            + " (graceful-drain: adoption waits for in-flight streams, never kills one)."
        )
    out.append(
        (
            backed,
            "Gateway adaptive slots reshape under sustained concurrency without dropping streams.",
        )
    )
    return out


def media_findings(recs: list[dict]) -> list[str]:
    """Media findings computed from cells (gating identical to the table)."""
    out: list[str] = []
    for r in recs:
        prov = r.get("provider")
        if not (prov or "").startswith("media-"):
            continue
        if prov == "media-tts" and r.get("ttfb_speedup_x"):
            ttfb_s = (r.get("pcm_ttfb_ms_median") or 0) / 1000.0
            out.append(
                f"**Streamed PCM cuts time-to-first-audio {pfmt(r.get('ttfb_speedup_x'), 2)}x "
                f"vs buffered WAV** (piper lane, first audio {pfmt(ttfb_s, 2)}s vs "
                f"{pfmt(r.get('wav_total_s_median'), 2)}s full synthesis) - total wall time "
                "is slightly higher (per-chunk synthesis), the win is interactivity."
            )
        elif prov == "media-video" and r.get("gate_reject_s_median") is not None:
            out.append(
                f"**Video VRAM gate rejects an over-budget request in "
                f"{pfmt(r.get('gate_reject_s_median') * 1000, 0)} ms** with the full estimate "
                "math and override levers in the error body - instead of an opaque child "
                "abort minutes later."
            )
        elif prov == "media-tts-conc" and r.get("error"):
            out.append(
                f"**TTS concurrency probe FAILED: {r['error']}** - the gateway did not "
                f"sustain {r.get('streams')} parallel PCM streams; needs investigation."
            )
        elif prov == "media-tts-conc" and r.get("efficiency_sum_over_wall") is not None:
            eff = r["efficiency_sum_over_wall"]
            n = r.get("streams") or 0
            verdict = (
                "perfectly parallel"
                if eff >= 0.75 * n
                else (
                    "partially parallel"
                    if eff > 1.25
                    else "serialized (single synth lane)"
                )
            )
            out.append(
                f"**{n} parallel PCM streams through one gateway: {verdict}** (efficiency "
                f"{pfmt(eff, 2)} = sum of per-stream totals / {pfmt(r.get('wall_s'), 2)}s "
                f"wall, max TTFB {pfmt(r.get('ttfb_ms_max'), 0)} ms"
                + (
                    ", byte-identical outputs across streams"
                    if r.get("bytes_uniform")
                    else ", NON-uniform stream outputs - flagged"
                )
                + ") - the scalability receipt for the TTS lane."
            )
        elif prov == "media-image" and r.get("quality_medians"):
            qm = r["quality_medians"]
            out.append(
                f"**Image quality stamps (PIL, luma domain): entropy "
                f"{pfmt(qm.get('entropy_bits'), 2)} bits, rms contrast "
                f"{pfmt(qm.get('rms_contrast'), 1)}, {pfmt(qm.get('unique_colors_256'), 0)} "
                f"unique colors @256x256** on {r.get('model')} - perceptual baseline for "
                "cross-run comparisons; audit PNG saved beside the cells."
            )
        elif prov == "media-video" and any(
            (run.get("mux_frames") != run.get("frames_requested_aligned"))
            for pt in (r.get("per_frames") or [])
            for run in (pt.get("runs") or [])
        ):
            bad = [
                (
                    pt.get("frames"),
                    run.get("mux_frames"),
                    run.get("frames_requested_aligned"),
                )
                for pt in (r.get("per_frames") or [])
                for run in (pt.get("runs") or [])
                if run.get("mux_frames") != run.get("frames_requested_aligned")
            ]
            out.append(
                f"**Frame-count mismatch on the {r.get('model')} lane**: container vs "
                f"aligned-request disagreements {bad} - flagged loudly, needs upstream "
                "investigation."
            )
    return out


def conc_frontier(recs: list[dict]) -> tuple[str, list[str]]:
    """Throughput/latency frontier across concurrency levels, per runtime.

    Returns (markdown table, verdict lines). Efficiency vs C=1 is
    sys(C) / (C x sys(1)) - 100% is perfectly parallel scaling. A saturation
    verdict needs at least 3 ascending levels (R5: fewer = honest 'insufficient
    levels', never an extrapolated claim).
    """
    groups: dict[tuple[str, str], list[dict]] = {}
    names: dict[tuple[str, str], str] = {}
    for r in recs:
        prov = r.get("provider")
        if prov not in ("conc-blazar", "conc-direct", "conc-ollama") or "error" in r:
            continue
        if prov == "conc-ollama":
            gkey = (prov, "")
            name = f"ollama - {r.get('ollama_model', 'reference')}"
        else:
            # one group per engine: two gateway engines must never merge
            # into one row set (the label would lie about whose numbers)
            gkey = (prov, r.get("tag") or "?")
            prefix = "blazar gateway" if prov == "conc-blazar" else "direct engine"
            name = f"{prefix} - {engine_label(r['tag'])}"
        names[gkey] = name
        groups.setdefault(gkey, []).append(r)
    if not groups:
        return "_Not measured._", []
    rows = []
    verdicts = []
    for gkey in sorted(groups):
        g = sorted(groups[gkey], key=lambda r: r.get("conc_level") or 0)
        base1 = next(
            (
                r.get("sys_tps")
                for r in g
                if r.get("conc_level") == 1 and r.get("sys_tps")
            ),
            None,
        )
        for r in g:
            lvl, sys_ = r.get("conc_level"), r.get("sys_tps")
            eff = None
            if base1 and sys_ and lvl and lvl >= 1:
                eff = sys_ / (lvl * base1)
            rows.append(
                (
                    names[gkey],
                    lvl,
                    r.get("conc_ok"),
                    sys_,
                    r.get("sum_stream_tps"),
                    eff,
                    r.get("conc_ttft_p99_ms"),
                    r.get("itl_p99_ms"),
                )
            )
        lvls = [r.get("conc_level") for r in g if r.get("conc_level")]
        tps = [r.get("sys_tps") for r in g if r.get("sys_tps")]
        if len(lvls) >= 3 and len(tps) == len(lvls):
            peak_i = max(range(len(tps)), key=lambda i: tps[i])
            plateau_i = None
            for i in range(1, len(tps)):
                if tps[i - 1] > 0 and (tps[i] - tps[i - 1]) / tps[i - 1] < 0.10:
                    plateau_i = i
                    break
            if plateau_i is not None:
                verdicts.append(
                    f"{names[gkey]}: throughput plateaus at C={lvls[plateau_i]} "
                    "(<10% per-level gain), "
                    f"peak {pfmt(tps[peak_i])} t/s at C={lvls[peak_i]}."
                )
            else:
                verdicts.append(
                    f"{names[gkey]}: still gaining at C={lvls[-1]} "
                    f"({pfmt(tps[0])} -> {pfmt(tps[-1])} t/s) - saturation not reached "
                    "within the sweep."
                )
        else:
            verdicts.append(
                f"{names[gkey]}: {len(lvls)} level(s) measured - insufficient levels "
                "for a saturation verdict."
            )
    head = (
        "| Runtime | C | ok streams | system t/s | sum-stream t/s | eff vs C=1 |"
        " TTFT p99 ms | ITL p99 ms |"
    )
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|"
    body = [
        f"| {n} | {pfmt(lvl, 0)} | {pfmt(ok, 0)} | {pfmt(s_)} | {pfmt(sm)} |"
        f" {pfmt(e * 100, 0) + '%' if e is not None else '-'} | {pfmt(t9, 0)} | {pfmt(i9, 1)} |"
        for n, lvl, ok, s_, sm, e, t9, i9 in rows
    ]
    return "\n".join([head, sep, *body]), verdicts


def engine_coverage(recs: list[dict]) -> list[str]:
    """Coverage audit: every store engine either has cells in this campaign
    or an exclusion reason — the artifact answers 'did we bench everything?'"""
    inv = next(
        (r for r in recs if r.get("provider") == "inventory" and "error" not in r),
        None,
    )
    if inv is None:
        return [
            "_Engine inventory not stamped (campaign predates coverage "
            "stamping); coverage cannot be audited for this artifact._"
        ]
    per_tag: dict[str, dict[str, int]] = {}
    for r in recs:
        if r.get("provider") == "inventory":
            continue
        cell = per_tag.setdefault(r.get("tag") or "?", {"ok": 0, "err": 0})
        cell["err" if "error" in r else "ok"] += 1
    lane_of = {
        "llamacpp": "text (direct + gateway)",
        "mistralrs": "text (direct + gateway)",
        "sdcpp": "media",
        "whisper": "media",
    }
    out = [
        "| Engine | Kind | Lane | ok cells | err cells | Status |",
        "|---|---|---|---:|---:|---|",
    ]
    for eng in inv.get("engines", []):
        tag, kind = eng["tag"], eng["kind"]
        c = per_tag.get(tag, {"ok": 0, "err": 0})
        if c["ok"]:
            status = "benchmarked"
        elif c["err"]:
            status = "attempted, all cells errored (see cells.jsonl)"
        else:
            status = "no cells in this campaign"
        out.append(
            f"| {tag} | {kind} | {lane_of.get(kind, '—')} "
            f"| {c['ok']} | {c['err']} | {status} |"
        )
    for tag, reason in sorted(inv.get("excluded", {}).items()):
        out.append(f"| {tag} | — | — | 0 | 0 | excluded: {reason} |")
    return out


def update_campaign_index(
    artifacts_dir: Path, recs: list[dict], blazar_ver: str
) -> None:
    """Idempotent bench-artifacts/INDEX.md row per campaign (replace-by-name)."""
    index = artifacts_dir.parent / "INDEX.md"
    name = artifacts_dir.name
    date = name.split("-", 1)[0] if name[:1].isdigit() else "?"
    measured = [r for r in recs if r.get("provider") != "inventory"]
    lanes = ",".join(
        sorted({(r.get("provider") or "?").split("-", 1)[0] for r in measured})
    )
    row = f"| {name} | {date} | {lanes} | {len(measured)} | {blazar_ver} |"
    header = [
        "# Benchmark campaign index",
        "",
        "| Campaign | Date | Lanes | Cells | blazar |",
        "|---|---|---|---:|---|",
    ]
    body: list[str] = []
    if index.exists():
        for line in index.read_text().splitlines():
            if line.startswith("| ") and not line.startswith("| Campaign"):
                body.append(line)
    body = [r for r in body if not r.startswith(f"| {name} |")]
    body.append(row)
    index.write_text("\n".join(header + sorted(body)) + "\n")
    print(f"campaign index -> {index} ({len(body)} campaign(s))")


def write_publication_report(
    recs: list[dict], artifacts_dir: Path, out_path: Path
) -> None:
    versions = {
        r.get("blazar_version", "").strip().removeprefix("blazar ")
        for r in recs
        if r.get("blazar_version")
    }
    blazar_ver = next(iter(versions)) if len(versions) == 1 else "mixed"
    env_states = {
        r.get("power_state", "unstamped") for r in recs if r.get("provider") == "blazar"
    }

    L: list[str] = []
    L.append("# Blazar inference benchmark")
    L.append("")
    # Byline slot is built from what exists: media-only campaigns have no
    # gateway power states, and an empty slot rendered as dangling
    # punctuation ("power state of gateway rows: .").
    byline = f"blazar {blazar_ver}"
    if env_states:
        byline += f"; power state of gateway rows: {', '.join(sorted(env_states))}"
    L.append(f"_Rendered {artifacts_dir.name}; {byline}._")
    L.append("")
    L.append("## Executive summary")
    L.append("")
    L.append(executive_summary(recs))
    L.append("")
    lane_counts: dict[str, int] = {}
    for r in recs:
        if r.get("provider") == "inventory":
            continue
        lane_counts[r.get("provider") or "?"] = (
            lane_counts.get(r.get("provider") or "?", 0) + 1
        )
    if lane_counts:
        L.append("## Measured in this campaign")
        L.append("")
        L += [f"- {prov}: {n} cell(s)" for prov, n in sorted(lane_counts.items())]
        L.append("")
    L.append("## Engine coverage")
    L.append("")
    L += engine_coverage(recs)
    L.append("")
    L.append("## Test bed")
    L.append("")
    L.append("| Component | Value |")
    L.append("|---|---|")
    L += [
        f"| {k} | {v} |" for k, v in TEST_BED + derived_test_bed_rows(recs, blazar_ver)
    ]
    L.append("")
    L.append("## Methodology")
    L.append("")
    L += [f"- {m}" for m in METHODOLOGY]
    L.append("")
    L.append("## Results")
    L.append("")
    L.append("### Single-stream decode (512-token prompt, 128 generated, median of 5)")
    L.append("")
    L.append(
        campaign_scoped(speed_table(recs), "single-stream speed", artifacts_dir.name)
    )
    L.append("")
    conc_levels = sorted(
        {
            r.get("params", {}).get("conc")
            for r in recs
            if r.get("provider", "").startswith("conc-") and "error" not in r
        }
    )
    conc_hdr = "x".join(str(c) for c in conc_levels) if conc_levels else "N"
    L.append(f"### Concurrency ({conc_hdr} parallel streams x 128 tokens)")
    L.append("")
    L.append(campaign_scoped(conc_table(recs), "concurrency", artifacts_dir.name))
    L.append("")
    L.append(
        "_sum-stream >> system t/s means streams serialize on one slot; "
        "roughly equal means genuinely parallel._"
    )
    L.append("")
    frontier_tbl, frontier_verdicts = conc_frontier(recs)
    L.append("### Concurrency frontier (system t/s and tail latency vs level)")
    L.append("")
    L.append(campaign_scoped(frontier_tbl, "concurrency frontier", artifacts_dir.name))
    L.append("")
    if frontier_verdicts:
        L += [f"- {v}" for v in frontier_verdicts]
        L.append("")
    L.append("### Adaptive reshape under sustained load (no-lag proof)")
    L.append("")
    L.append(
        campaign_scoped(reshape_table(recs), "adaptive reshape", artifacts_dir.name)
    )
    L.append("")
    L.append("### Perplexity")
    L.append("")
    L.append(campaign_scoped(ppl_table(recs), "perplexity", artifacts_dir.name))
    L.append("")
    L.append("### Greedy parity and gateway transparency (20 prompts, 256 tokens)")
    L.append("")
    L.append(campaign_scoped(greedy_table(recs), "greedy parity", artifacts_dir.name))
    L.append("")
    L.append(
        "_Exact-match divergence across GPU backends is expected float nondeterminism "
        "(batch shape and backend kernels), not translation drift; bit-parity across "
        "runs requires single-slot decoding (blazar `deterministic = true` pins it)._"
    )
    L.append("")
    L.append("### Tool calls (single-turn selection + schema quality)")
    L.append("")
    L.append(campaign_scoped(tools_table(recs), "tool calls", artifacts_dir.name))
    L.append("")
    L.append("")
    L.append("### Optimization axes (ctx 4096, single stream)")
    L.append("")
    L.append(
        campaign_scoped(variant_table(recs), "optimization axes", artifacts_dir.name)
    )
    L.append("")
    L.append("### Engine capability matrix")
    L.append("")
    L.append(
        campaign_scoped(features_table(recs), "capability matrix", artifacts_dir.name)
    )
    L.append("")
    L.append("### Cold start and footprint")
    L.append("")
    L.append(campaign_scoped(coldstart_table(recs), "cold start", artifacts_dir.name))
    L.append("")
    L.append(
        "_Every cold probe runs page-cache-dropped and GPU-idle-asserted on both runtimes; ollama rows without --ollama-service-restart leave the daemon warm (note in the artifact)._"
    )
    L.append("")
    L.append("### Idle wake (sleep vs keep_alive expiry)")
    L.append("")
    L.append(campaign_scoped(idle_wake_table(recs), "idle wake", artifacts_dir.name))
    L.append("")
    L.append(
        "_blazar sleeps with weights in RAM (wake = resume); ollama unloads at keep_alive expiry (wake = full disk reload). Policies differ by design — the table measures each runtime's own idle path after the policy verifiably fired._"
    )
    L.append("")
    L.append("### Long-context degradation curve")
    L.append("")
    L.append(
        campaign_scoped(ctxcurve_table(recs), "long-context curve", artifacts_dir.name)
    )
    L.append("")
    L.append("### Media lanes (image / video / TTS / whisper)")
    L.append("")
    L.append(campaign_scoped(media_table(recs), "media", artifacts_dir.name))
    L.append("")
    L.append(
        "_Media cells run through the same sandboxed gateway as text lanes but do not assert "
        "GPU-idle: a warm engine child is the normal serving shape, so each row stamps "
        "gpu_busy_mib / ram_avail_mib / loadavg instead. 3 runs (not 5) — media variance is "
        "dominated by the model, not the scheduler. Video frame counts are read from the EBML "
        "container (lacing-aware), never from an API field; the VRAM gate probe times how fast "
        "an over-budget request is rejected with a teaching error._"
    )
    L.append("")
    backed_findings: list[str] = []
    carried_findings: list[str] = []
    for backed_render, carried_text in text_findings(recs):
        if backed_render:
            backed_findings.append(backed_render)
        else:
            carried_findings.append(carried_text)
    backed_findings += media_findings(recs)
    L.append("## Findings (this campaign)")
    L.append("")
    if backed_findings:
        for i, f in enumerate(backed_findings, 1):
            L.append(f"{i}. {f}")
    else:
        L.append(
            "_No complete findings from this campaign's cells; "
            "see carried-over findings below._"
        )
    L.append("")
    if carried_findings:
        L.append("## Carried-over findings (no receipt in this campaign)")
        L.append("")
        L.append(
            "_Established in earlier campaigns whose receipts live in their "
            "bench-artifacts/ directories; this campaign did not measure these lanes._"
        )
        L.append("")
        for i, f in enumerate(carried_findings, 1):
            L.append(f"{i}. {f}")
        L.append("")
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
        "python3 scripts/bench_matrix.py --blazar-bin target/release/blazar --md BENCHMARK.md"
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

    update_campaign_index(artifacts_dir, recs, blazar_ver)
    out_path.write_text("\n".join(L))
    print(f"report -> {out_path} ({len(recs)} last-wins records)")


if __name__ == "__main__":
    sys.exit(main())
