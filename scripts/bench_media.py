#!/usr/bin/env python3
"""Media-lane benchmark harness for a running blazar gateway.

Drives the real gateway endpoints end to end (no mocks) at concurrency
ladders and emits reproducibility receipts per the project benchmarking
contract: raw samples preserved, JSONL + CSV + a markdown report, and a
reproducibility header (timestamp, versions, hardware, configuration).

Lanes:
  embeddings  text (bert child), image (candle CLIP), content-array (VLM)
  speech      piper lane and engine /tts lane, RTF + chars/s
  image       diffusion render wall-time + VRAM watermark
  video       video render wall-time + VRAM watermark

Direct-engine baselines (--direct where applicable) spawn the engine
binary standalone on a private port so gateway overhead can be isolated;
the gateway side of the comparison is always the same lane through the
gateway. Candle CLIP and piper have no separate server to spawn - the
candle lane is in-process and piper is a local synthesizer - so those
lanes have no direct baseline by construction.

Everything is written straight to the output directory (default
./bench-artifacts/<date>-media-lanes under the current working
directory) so a timeout never loses completed lanes.

Stdlib only; python3.10+.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import csv
import json
import math
import os
import sqlite3
import statistics
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone

DEFAULT_GATEWAY = "http://127.0.0.1:11599"
LANES_OUT = datetime.now(timezone.utc).strftime("%Y%m%d") + "-media-lanes"


# ---------------------------------------------------------------- helpers


def http_json(
    url: str,
    payload: dict | None = None,
    timeout: int = 600,
    headers: dict | None = None,
) -> tuple[int, dict | bytes, float, bytes | None]:
    """POST json (or GET when payload is None). Returns (status, parsed-json-or-raw, wall_s, raw)."""
    data = json.dumps(payload).encode() if payload is not None else None
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(
        url, data=data, headers=hdrs, method="POST" if data else "GET"
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            wall = time.perf_counter() - t0
            ct = resp.headers.get("Content-Type", "")
            if "json" in ct:
                return resp.status, json.loads(raw), wall, raw
            return resp.status, raw, wall, raw
    except urllib.error.HTTPError as e:
        raw = e.read()
        wall = time.perf_counter() - t0
        try:
            return e.code, json.loads(raw), wall, raw
        except Exception:
            return e.code, raw, wall, raw


def pct(samples: list[float], p: float) -> float:
    if not samples:
        return float("nan")
    s = sorted(samples)
    k = min(len(s) - 1, max(0, round((p / 100.0) * (len(s) - 1))))
    return s[k]


def wav_seconds(raw: bytes) -> float | None:
    """Duration of a RIFF/WAVE payload (16-bit PCM assumed for data-size math)."""
    if len(raw) < 44 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        return None
    # walk chunks for 'fmt ' and 'data'
    pos, fmt, datalen = 12, None, None
    while pos + 8 <= len(raw):
        cid = raw[pos : pos + 4]
        clen = struct.unpack("<I", raw[pos + 4 : pos + 8])[0]
        if cid == b"fmt ":
            fmt = raw[pos + 8 : pos + 8 + clen]
        elif cid == b"data":
            datalen = clen
            break
        pos += 8 + clen + (clen & 1)
    if not fmt or datalen is None or len(fmt) < 16:
        return None
    rate = struct.unpack("<I", fmt[4:8])[0]
    channels = struct.unpack("<H", fmt[2:4])[0]
    bits = struct.unpack("<H", fmt[14:16])[0]
    if rate <= 0 or channels <= 0 or bits <= 0:
        return None
    return datalen / (rate * channels * (bits // 8))


def l2_norm(vec: list[float]) -> float:
    return math.fsum(v * v for v in vec) ** 0.5


def tiny_png(w: int = 64, h: int = 64) -> bytes:
    """Deterministic small PNG for image-input lanes (stdlib only)."""
    rows = b""
    for y in range(h):
        rows += b"\x00" + bytes([(y * 4) % 256, 90, 180] * w)

    def chunk(tag: bytes, data: bytes) -> bytes:
        c = tag + data
        return (
            struct.pack(">I", len(data))
            + c
            + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


# ------------------------------------------------------------- VRAM probe


class VramSampler(threading.Thread):
    """Samples nvidia-smi memory.used; watermark = max observed."""

    def __init__(self, interval: float = 0.5):
        super().__init__(daemon=True)
        self.interval = interval
        self.samples: list[int] = []
        self.watermark_mib = 0
        self._halt = threading.Event()

    def run(self) -> None:
        while not self._halt.is_set():
            try:
                out = subprocess.run(
                    check=False,
                    args=[
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                ).stdout.strip()
                first = out.splitlines()[0].strip() if out else ""
                if first.isdigit():
                    mib = int(first)
                    self.samples.append(mib)
                    self.watermark_mib = max(self.watermark_mib, mib)
            except Exception:
                pass
            self._halt.wait(self.interval)

    def stop(self) -> dict:
        self._halt.set()
        self.join(timeout=3)
        return {
            "watermark_mib": self.watermark_mib,
            "samples": len(self.samples),
            "baseline_mib": self.samples[0] if self.samples else None,
        }


# ------------------------------------------------------- engine discovery


def find_binary(
    engines_root: str, tag_substr: str, names: list[str]
) -> tuple[str, str]:
    """Locate an engine binary under engines/<tag>/<dir>/<name>; returns (dir, path)."""
    for tag in sorted(os.listdir(engines_root)):
        if tag_substr not in tag:
            continue
        base = os.path.join(engines_root, tag)
        for dirpath, _, files in os.walk(base):
            for n in names:
                if n in files:
                    return dirpath, os.path.join(dirpath, n)
    raise FileNotFoundError(
        f"no engine matching '{tag_substr}' with binaries {names} under {engines_root}"
    )


@dataclass
class Child:
    proc: subprocess.Popen
    log_path: str

    def kill(self) -> None:
        try:
            self.proc.terminate()
            self.proc.wait(timeout=10)
        except Exception:
            with contextlib.suppress(Exception):
                self.proc.kill()


def spawn_child(
    argv: list[str],
    env_extra: dict[str, str],
    port: int,
    log_dir: str,
    health_path: str = "/health",
    timeout: float = 180.0,
) -> Child:
    """Spawn a standalone engine child (setsid so tool-shell exits never orphan-kill it)."""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"child-{port}.log")
    env = dict(os.environ)
    env.update(env_extra)
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(
            argv, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            log.close()
            raise RuntimeError(f"child exited rc={proc.returncode}; see {log_path}")
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}{health_path}", timeout=2
            ) as r:
                if r.status == 200:
                    return Child(proc, log_path)
        except Exception:
            time.sleep(2)
    proc.terminate()
    log.close()
    raise RuntimeError(f"child did not become healthy on port {port}; see {log_path}")


# --------------------------------------------------------- store accessors


def store_row(store_path: str, name: str) -> dict | None:
    if not os.path.exists(store_path):
        return None
    db = sqlite3.connect(f"file:{store_path}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        row = db.execute("SELECT * FROM models WHERE name = ?", (name,)).fetchone()
        return dict(row) if row else None
    finally:
        db.close()


def active_engine_tag(store_path: str) -> str | None:
    if not os.path.exists(store_path):
        return None
    db = sqlite3.connect(f"file:{store_path}?mode=ro", uri=True)
    try:
        row = db.execute(
            "SELECT tag FROM engines WHERE active = 1 AND kind = 'llamacpp'"
        ).fetchone()
        return row[0] if row else None
    finally:
        db.close()


# ------------------------------------------------------------- receipts


def environment_block(gateway: str, store_path: str, note: str) -> dict:
    env = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gateway": gateway,
        "note": note,
        "kernel": subprocess.run(
            ["uname", "-sr"], capture_output=True, text=True, check=False
        ).stdout.strip(),
    }
    try:
        _st, ver, _, _ = http_json(f"{gateway}/api/version", timeout=10)
        env["gateway_version"] = str(ver)[:200]
    except Exception:
        env["gateway_version"] = "unreachable"
    tag = active_engine_tag(store_path)
    env["active_llamacpp_engine"] = tag or "unknown"
    try:
        q = subprocess.run(
            check=False,
            args=[
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
        ).stdout.strip()
        env["gpu"] = q.splitlines()[0] if q else "none"
    except Exception:
        env["gpu"] = "none"
    return env


class Receipts:
    """Append-only JSONL + per-lane CSV, written immediately (never staged in /tmp)."""

    def __init__(self, out_dir: str, env: dict):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.jsonl_path = os.path.join(out_dir, "receipts.jsonl")
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"type": "environment", **env}) + "\n")

    def lane(self, record: dict, csv_columns: list[str], rows: list[dict]) -> None:
        record = {"type": "lane", **record}
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        csv_path = os.path.join(
            self.out_dir, f"{record['lane']}-{record.get('mode', 'gateway')}.csv"
        )
        new = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=csv_columns)
            if new:
                w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in csv_columns})


# ------------------------------------------------------------------ lanes


def run_ladder(fn, conc: int, reps: int) -> dict:
    """Run fn() as (conc x reps) requests; returns aggregate metrics."""
    walls: list[float] = []
    errors = 0
    lock = threading.Lock()
    t0 = time.perf_counter()

    def one(_):
        nonlocal errors
        try:
            w = fn()
            with lock:
                walls.append(w)
        except Exception:
            with lock:
                errors += 1

    with ThreadPoolExecutor(max_workers=conc) as ex:
        list(ex.map(one, range(reps)))
    total = time.perf_counter() - t0
    ok = len(walls)
    return {
        "conc": conc,
        "reps": reps,
        "ok": ok,
        "errors": errors,
        "wall_total_s": round(total, 3),
        "req_per_s": round(ok / total, 3) if total > 0 else 0.0,
        "p50_ms": round(pct(walls, 50) * 1000, 1) if walls else None,
        "p95_ms": round(pct(walls, 95) * 1000, 1) if walls else None,
        "p99_ms": round(pct(walls, 99) * 1000, 1) if walls else None,
        "mean_ms": round(statistics.fmean(walls) * 1000, 1) if walls else None,
        "raw_walls_s": [round(w, 4) for w in walls],
    }


EMB_TEXTS = [
    "the lighthouse keeper climbs the spiral staircase at dawn",
    "a quiet harbor town nested between two steep green hills",
    "the engine hums steadily while the cargo ship departs",
]

EMB_CSV = [
    "preset",
    "conc",
    "ok",
    "errors",
    "req_per_s",
    "p50_ms",
    "p95_ms",
    "p99_ms",
    "mean_ms",
    "dim",
    "l2",
    "wall_total_s",
]


def lane_embeddings(args, rc: Receipts, env: dict) -> None:
    png_b64 = base64.b64encode(tiny_png()).decode()
    presets = {
        "text-bert": {
            "model": args.model or "bge-small-en",
            "body": {"input": EMB_TEXTS},
        },
        "clip-image": {
            "model": args.model or "clip-vit-base-patch32",
            "body": {
                "input": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + png_b64},
                    }
                ]
            },
        },
        "vlm-content": {
            "model": args.model or "qwen3vl-8b-instruct",
            "body": {
                "input": [
                    {
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "data:image/png;base64," + png_b64
                                },
                            }
                        ]
                    }
                ]
            },
        },
    }
    if args.preset not in presets:
        sys.exit(f"unknown preset {args.preset}; one of {sorted(presets)}")
    p = presets[args.preset]

    def one() -> float:
        st, body, wall, _ = http_json(
            f"{args.gateway}/v1/embeddings",
            {"model": p["model"], **p["body"]},
            timeout=args.timeout,
        )
        if st != 200 or not isinstance(body, dict) or not body.get("data"):
            raise RuntimeError(f"status {st}: {str(body)[:120]}")
        return wall

    # warmup: boots the child/loads weights outside the measured window
    st, body, _, _ = http_json(
        f"{args.gateway}/v1/embeddings",
        {"model": p["model"], **p["body"]},
        timeout=args.timeout,
    )
    if st != 200:
        sys.exit(f"warmup failed ({st}): {str(body)[:200]}")
    dim = len(body["data"][0]["embedding"])
    l2 = round(l2_norm(body["data"][0]["embedding"]), 4)

    rows = []
    for c in [int(x) for x in args.conc.split(",")]:
        m = run_ladder(one, c, args.reps)
        m.update({"preset": args.preset, "dim": dim, "l2": l2})
        rows.append(m)
        print(
            f"  {args.preset} C={c}: {m['req_per_s']} req/s p50={m['p50_ms']}ms errors={m['errors']}"
        )
    rc.lane(
        {
            "lane": f"embeddings-{args.preset}",
            "mode": "gateway",
            "model": p["model"],
            "env": env,
            "rows": rows,
        },
        EMB_CSV,
        rows,
    )


def lane_embeddings_direct(args, rc: Receipts, env: dict) -> None:
    """Direct llama-server baseline for the bert text lane (single model, own port)."""
    row = store_row(args.store, args.model or "bge-small-en")
    if not row:
        sys.exit("store row not found for direct bert baseline")
    eng_dir, bin_path = find_binary(args.engines_root, "b120", ["llama-server"])
    port = args.direct_port
    child = spawn_child(
        [
            bin_path,
            "-m",
            row["path"],
            "--port",
            str(port),
            "--host",
            "127.0.0.1",
            "-ngl",
            "0",
            "-c",
            "2048",
            "--embeddings",
            "--pooling",
            "last",
        ],
        {"LD_LIBRARY_PATH": eng_dir},
        port,
        args.log_dir,
    )

    def one() -> float:
        st, body, wall, _ = http_json(
            f"http://127.0.0.1:{port}/v1/embeddings",
            {"input": EMB_TEXTS},
            timeout=args.timeout,
        )
        if st != 200 or not isinstance(body, dict) or not body.get("data"):
            raise RuntimeError(f"status {st}")
        return wall

    try:
        wst, wbody, _, _ = http_json(
            f"http://127.0.0.1:{port}/v1/embeddings",
            {"input": EMB_TEXTS},
            timeout=args.timeout,
        )
        dim = (
            len(wbody["data"][0]["embedding"])
            if wst == 200 and isinstance(wbody, dict) and wbody.get("data")
            else None
        )
        rows = []
        for c in [int(x) for x in args.conc.split(",")]:
            m = run_ladder(one, c, args.reps)
            m.update(
                {
                    "preset": "text-bert-direct",
                    "dim": dim,
                    "l2": None,
                }
            )
            rows.append(m)
            print(f"  bert-direct C={c}: {m['req_per_s']} req/s p50={m['p50_ms']}ms")
        rc.lane(
            {
                "lane": "embeddings-text-bert",
                "mode": "direct",
                "model": row["name"],
                "engine": bin_path,
                "env": env,
                "rows": rows,
            },
            EMB_CSV,
            rows,
        )
    finally:
        child.kill()


SPEECH_CSV = [
    "lane_mode",
    "conc",
    "ok",
    "errors",
    "req_per_s",
    "p50_ms",
    "p95_ms",
    "mean_ms",
    "audio_s_mean",
    "rtf_mean",
    "chars_per_s",
    "wall_total_s",
]

SPEECH_TEXT = (
    "The harbor lights guide every ship safely home through the narrow channel."
)


def speech_one(gateway: str, model: str, text: str, timeout: int):
    st, raw, wall, _ = http_json(
        f"{gateway}/v1/audio/speech",
        {"model": model, "input": text, "response_format": "wav"},
        timeout=timeout,
    )
    if st != 200 or not isinstance(raw, (bytes, bytearray)) or raw[:4] != b"RIFF":
        raise RuntimeError(f"status {st}: {str(raw)[:120]}")
    return wall, wav_seconds(bytes(raw)) or 0.0


def lane_speech(args, rc: Receipts, env: dict) -> None:
    if args.lane_speech == "piper":
        model, mode = args.model or "en_US-amy-medium", "gateway-piper"
        no_direct = (
            "piper is a local in-process synthesizer; no separate server to baseline"
        )
    else:
        row = store_row(args.store, args.model or "pocket-tts")
        if not row:
            sys.exit("engine speech lane needs a store row (--model); none found")
        model, mode = row["name"], "gateway-engine"
        no_direct = None
    text = SPEECH_TEXT

    def one() -> float:
        wall, _ = speech_one(args.gateway, model, text, args.timeout)
        return wall

    speech_one(args.gateway, model, text, args.timeout)  # warmup
    rows = []
    for c in [int(x) for x in args.conc.split(",")]:
        m = run_ladder(one, c, args.reps)
        # one timed sample for audio-duration-derived metrics
        _, audio_s = speech_one(args.gateway, model, text, args.timeout)
        m.update(
            {
                "lane_mode": mode,
                "audio_s_mean": round(audio_s, 3),
                "rtf_mean": round(audio_s / m["mean_ms"] * 1000, 4)
                if m["mean_ms"]
                else None,
                "chars_per_s": round(len(text) / (m["mean_ms"] / 1000), 1)
                if m["mean_ms"]
                else None,
            }
        )
        rows.append(m)
        print(f"  {mode} C={c}: {m['req_per_s']} req/s rtf={m['rtf_mean']}")
    rc.lane(
        {
            "lane": "speech",
            "mode": mode,
            "model": model,
            "env": env,
            "rows": rows,
            "no_direct_note": no_direct,
        },
        SPEECH_CSV,
        rows,
    )


def lane_speech_direct(args, rc: Receipts, env: dict) -> None:
    row = store_row(args.store, args.model or "pocket-tts")
    if not row:
        sys.exit("store row not found for direct engine-tts baseline")
    eng_dir, bin_path = find_binary(args.engines_root, "b120", ["llama-server"])
    port = args.direct_port
    argv = [
        bin_path,
        "-m",
        row["path"],
        "--port",
        str(port),
        "--host",
        "127.0.0.1",
        "-ngl",
        "0",
        "-c",
        "4096",
    ]
    if row.get("mmproj_path"):
        argv += ["--mmproj", row["mmproj_path"]]
    child = spawn_child(argv, {"LD_LIBRARY_PATH": eng_dir}, port, args.log_dir)
    text = SPEECH_TEXT

    def one() -> float:
        st, raw, wall, _ = http_json(
            f"http://127.0.0.1:{port}/tts", {"input": text}, timeout=args.timeout
        )
        if st != 200 or not isinstance(raw, (bytes, bytearray)) or raw[:4] != b"RIFF":
            raise RuntimeError(f"status {st}")
        return wall

    try:
        http_json(f"http://127.0.0.1:{port}/tts", {"input": text}, timeout=args.timeout)
        rows = []
        for c in [int(x) for x in args.conc.split(",")]:
            m = run_ladder(one, c, args.reps)
            _st2, raw2, _, _ = http_json(
                f"http://127.0.0.1:{port}/tts", {"input": text}, timeout=args.timeout
            )
            audio_s = (
                wav_seconds(bytes(raw2))
                if isinstance(raw2, (bytes, bytearray))
                else 0.0
            ) or 0.0
            m.update(
                {
                    "lane_mode": "direct-engine",
                    "audio_s_mean": round(audio_s, 3),
                    "rtf_mean": round(audio_s / m["mean_ms"] * 1000, 4)
                    if m["mean_ms"]
                    else None,
                    "chars_per_s": round(len(text) / (m["mean_ms"] / 1000), 1)
                    if m["mean_ms"]
                    else None,
                }
            )
            rows.append(m)
            print(f"  engine-direct C={c}: {m['req_per_s']} req/s rtf={m['rtf_mean']}")
        rc.lane(
            {
                "lane": "speech",
                "mode": "direct-engine",
                "model": row["name"],
                "engine": bin_path,
                "env": env,
                "rows": rows,
            },
            SPEECH_CSV,
            rows,
        )
    finally:
        child.kill()


IMAGE_CSV = ["mode", "rep", "wall_s", "bytes", "valid_png", "vram_watermark_mib"]


def _sd_argv(row: dict, engines_root: str, port: int) -> tuple[list[str], str]:
    """Build a direct sd-server argv from the store row: the row IS the contract."""
    _, bin_path = find_binary(
        engines_root, "master-1", ["sd-server", "sdcpp-server"]
    )
    argv = [
        bin_path,
        "--listen-ip",
        "127.0.0.1",
        "--listen-port",
        str(port),
        "--diffusion-model",
        row["path"],
    ]
    for comp in json.loads(row.get("components") or "[]"):
        argv += [str(comp.get("flag")), str(comp.get("path"))]
    return argv, bin_path


def lane_image(args, rc: Receipts, env: dict) -> None:
    body = {
        "model": args.model,
        "prompt": args.prompt,
        "size": args.size,
        "steps": args.steps,
        "seed": args.seed,
    }
    sampler = VramSampler()
    sampler.start()
    http_json(
        f"{args.gateway}/v1/images/generations", body, timeout=args.timeout
    )  # warmup boot
    rows = []
    for rep in range(args.reps):
        st, resp, wall, _raw = http_json(
            f"{args.gateway}/v1/images/generations", body, timeout=args.timeout
        )
        b64 = ""
        if st == 200 and isinstance(resp, dict):
            with contextlib.suppress(Exception):
                b64 = resp["data"][0].get("b64_json", "")
        png = base64.b64decode(b64) if b64 else b""
        valid = png[:8] == b"\x89PNG\r\n\x1a\n" and len(png) > 1000
        rows.append(
            {
                "mode": "gateway",
                "rep": rep,
                "wall_s": round(wall, 2),
                "bytes": len(png),
                "valid_png": valid,
            }
        )
        print(f"  image rep{rep}: {wall:.1f}s {len(png)}B valid={valid}")
    vram = sampler.stop()
    for r in rows:
        r["vram_watermark_mib"] = vram["watermark_mib"]
    rc.lane(
        {
            "lane": "image",
            "mode": "gateway",
            "model": args.model,
            "body": body,
            "env": env,
            "vram": vram,
            "rows": rows,
        },
        IMAGE_CSV,
        rows,
    )


def lane_image_direct(args, rc: Receipts, env: dict) -> None:
    row = store_row(args.store, args.model)
    if not row:
        sys.exit("store row not found for direct image baseline")
    argv, bin_path = _sd_argv(row, args.engines_root, args.direct_port)
    env_extra = {
        "LD_LIBRARY_PATH": os.path.dirname(bin_path),
        "XDG_CACHE_HOME": args.log_dir,
    }
    child = spawn_child(
        argv,
        env_extra,
        args.direct_port,
        args.log_dir,
        # sd-server has no /health route; /v1/models answers once listening.
        health_path="/v1/models",
        # cold load of the full DiT+conditioner+VAE set (11+ GiB via Vulkan,
        # params auto-fitted to disk) measured at ~7 min on the receipt box.
        timeout=900,
    )
    base = f"http://127.0.0.1:{args.direct_port}"
    body = {
        "prompt": args.prompt,
        "size": args.size,
        "steps": args.steps,
        "seed": args.seed,
        "response_format": "b64_json",
    }
    sampler = VramSampler()
    sampler.start()

    def direct_call(url: str, payload: dict, timeout: float, retries: int = 3):
        # sd-server occasionally drops a fresh connection mid-render (the
        # render itself finishes server-side). Retry after letting the
        # in-flight orphan render drain; walls below count only the attempt
        # that answered, and the receipt row records the retry count.
        last: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                st, resp, wall, raw = http_json(url, payload, timeout=int(timeout))
                return st, resp, wall, raw, attempt - 1
            except (ConnectionError, TimeoutError) as e:
                last = e
                print(
                    f"  direct attempt {attempt} dropped: {type(e).__name__}; draining + retry"
                )
                time.sleep(45)
        if last is None:
            raise RuntimeError("direct call made no attempts")
        raise last

    try:
        direct_call(f"{base}/v1/images/generations", body, timeout=args.timeout)
        rows = []
        for rep in range(args.reps):
            _st, resp, wall, _raw, dropped = direct_call(
                f"{base}/v1/images/generations", body, timeout=args.timeout
            )
            b64 = resp["data"][0].get("b64_json", "") if isinstance(resp, dict) else ""
            png = base64.b64decode(b64) if b64 else b""
            valid = png[:8] == b"\x89PNG\r\n\x1a\n" and len(png) > 1000
            rows.append(
                {
                    "mode": "direct",
                    "rep": rep,
                    "wall_s": round(wall, 2),
                    "bytes": len(png),
                    "valid_png": valid,
                    "dropped_retries": dropped,
                }
            )
            print(
                f"  image-direct rep{rep}: {wall:.1f}s {len(png)}B valid={valid} dropped={dropped}"
            )
        vram = sampler.stop()
        for r in rows:
            r["vram_watermark_mib"] = vram["watermark_mib"]
        rc.lane(
            {
                "lane": "image",
                "mode": "direct",
                "model": args.model,
                "body": body,
                "engine": bin_path,
                "env": env,
                "vram": vram,
                "rows": rows,
            },
            IMAGE_CSV,
            rows,
        )
    finally:
        child.kill()


VIDEO_CSV = ["mode", "rep", "wall_s", "bytes", "valid_webm"]


def lane_video(args, rc: Receipts, env: dict) -> None:
    body = {
        "model": args.model,
        "prompt": args.prompt,
        "size": args.size,
        "seconds": args.seconds,
    }
    sampler = VramSampler()
    sampler.start()
    http_json(f"{args.gateway}/v1/videos/generations", body, timeout=args.timeout)
    rows = []
    for rep in range(args.reps):
        st, resp, wall, _raw = http_json(
            f"{args.gateway}/v1/videos/generations", body, timeout=args.timeout
        )
        b64 = ""
        if st == 200 and isinstance(resp, dict):
            with contextlib.suppress(Exception):
                b64 = resp["data"][0].get("b64_json", "")
        webm = base64.b64decode(b64) if b64 else b""
        valid = webm[:4] == b"\x1a\x45\xdf\xa3" and len(webm) > 1000
        rows.append(
            {
                "mode": "gateway",
                "rep": rep,
                "wall_s": round(wall, 2),
                "bytes": len(webm),
                "valid_webm": valid,
            }
        )
        print(f"  video rep{rep}: {wall:.1f}s {len(webm)}B valid={valid}")
    vram = sampler.stop()
    rc.lane(
        {
            "lane": "video",
            "mode": "gateway",
            "model": args.model,
            "body": body,
            "env": env,
            "vram": vram,
            "rows": rows,
        },
        VIDEO_CSV,
        rows,
    )


# -------------------------------------------------------------------- cli


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--gateway", default=os.environ.get("BLAZAR_GATEWAY", DEFAULT_GATEWAY)
    )
    ap.add_argument("--out", default=os.path.join("bench-artifacts", LANES_OUT))
    ap.add_argument(
        "--store",
        default=os.path.join(
            os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share")),
            "blazar",
            "blazar.db",
        ),
    )
    ap.add_argument(
        "--engines-root",
        default=os.path.join(
            os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share")),
            "blazar",
            "engines",
        ),
    )
    ap.add_argument(
        "--log-dir", default=os.path.join(os.path.expanduser("~/.cache"), "bench-media")
    )
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--direct-port", type=int, default=18270)
    sub = ap.add_subparsers(dest="lane", required=True)

    emb = sub.add_parser(
        "embeddings", help="text-bert | clip-image | vlm-content presets"
    )
    emb.add_argument("--preset", default="text-bert")
    emb.add_argument("--model", default=None)
    emb.add_argument("--conc", default="1,2,4,8")
    emb.add_argument("--reps", type=int, default=16)
    emb.add_argument(
        "--direct",
        action="store_true",
        help="direct llama-server baseline (text-bert only)",
    )

    sp = sub.add_parser("speech", help="piper | engine lanes")
    sp.add_argument(
        "--lane-speech",
        dest="lane_speech",
        default="piper",
        choices=["piper", "engine"],
    )
    sp.add_argument("--model", default=None)
    sp.add_argument("--conc", default="1,2,4")
    sp.add_argument("--reps", type=int, default=8)
    sp.add_argument(
        "--direct",
        action="store_true",
        help="direct llama-server /tts baseline (engine lane only)",
    )

    im = sub.add_parser("image", help="diffusion render wall-time + VRAM")
    im.add_argument("--model", required=True)
    im.add_argument("--prompt", default="a red lighthouse on a rocky cliff at dusk")
    im.add_argument("--size", default="512x512")
    im.add_argument("--steps", type=int, default=20)
    im.add_argument("--seed", type=int, default=42)
    im.add_argument("--reps", type=int, default=3)
    im.add_argument("--direct", action="store_true", help="direct sd-server baseline")

    vd = sub.add_parser("video", help="video render wall-time + VRAM")
    vd.add_argument("--model", required=True)
    vd.add_argument("--prompt", default="waves rolling onto a pebble beach at sunrise")
    vd.add_argument("--size", default="320x320")
    vd.add_argument("--seconds", type=int, default=1)
    vd.add_argument("--reps", type=int, default=2)

    args = ap.parse_args()
    env = environment_block(args.gateway, args.store, sys.argv[2:].__str__())
    rc = Receipts(args.out, env)
    print(f"bench_media -> {args.out} (gateway {args.gateway})")

    if args.lane == "embeddings":
        lane_embeddings(args, rc, env)
        if args.direct:
            lane_embeddings_direct(args, rc, env)
    elif args.lane == "speech":
        lane_speech(args, rc, env)
        if args.direct:
            lane_speech_direct(args, rc, env)
    elif args.lane == "image":
        # --direct is exclusive: two 11.6 GiB diffusion children co-resident
        # (gateway's + the direct one) starved host memory and silently killed
        # renders on the bench box; run each side as its own invocation.
        if not args.direct:
            lane_image(args, rc, env)
        else:
            lane_image_direct(args, rc, env)
    elif args.lane == "video":
        lane_video(args, rc, env)

    print(f"receipts: {rc.jsonl_path}")


if __name__ == "__main__":
    main()
