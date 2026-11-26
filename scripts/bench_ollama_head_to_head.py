#!/usr/bin/env python3
"""M1 head-to-head: matched campaign, Blazar gateway vs ollama 0.40.17.

Repo harness (scripts/): run from anywhere - the repo root, binary and
artifact dir resolve relative to this file. Examples:
  python3 scripts/bench_ollama_head_to_head.py --lane pilot
  python3 scripts/bench_ollama_head_to_head.py --lane warm
  python3 scripts/bench_ollama_head_to_head.py --lane receipt

Roadmap M1 (docs/research/2026-11-23-sota-improvement-feasibility-roi.md):
re-run the matched campaign - same model, quant, sampling and cache state -
against ollama 0.40.17, and add a non-stream per-request overhead lane in
microseconds (the figure the gateway market publishes). Matched pair:
Qwen2.5-0.5B-Instruct Q4_K_M on both sides
(ollama "qwen2.5:0.5b", Blazar row "qwen2.5-0.5b-instruct").

Protocol: OpenAI /v1/chat/completions on both products (identical client
code), temperature 0 + seed 7 (recorded; temp-0 makes the seed inert),
max_tokens 128 for generation lanes / 16 for the overhead lane. Warm state
is primary; cold is one-shot per product. The ollama service is treated
READ-ONLY: /api/version, /api/tags, /api/show, /api/ps probes plus normal
generation POSTs, and the request-scoped keep_alive=0 knob (an API-level
unload - no restart, no store change) for the cold arm.

Lanes (run one at a time; arms strictly sequential, ollama first):
  pilot     env evidence + 2 requests per product + dialect/cold probes
  warm      ladder C=1,2,4,8 x N=32 per level, stream + non-stream, both
  cold      one-shot first-request walls per product
  overhead  Blazar gateway vs MATCHED direct child (gateway child argv
            minus the auth-file flag, state dirs redirected), C=1
            non-stream, N=200, max_tokens=16 - pure gateway cost in us
  quality   20-prompt temp-0 battery, token-level agreement
  receipt   assemble summary JSON + CSV + SVG charts + markdown from raw

Every request lands in <out-dir>/<lane>.jsonl (raw, failures included,
never cherry-picked). The receipt lane only re-renders, never re-measures.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

# ----------------------------------------------------------------- runtime
# Must resolve before validate import: the harness reads BLAZAR_BIN and
# picks its port at module import time.
REPO = Path(
    os.environ.get("BLAZAR_REPO", Path(__file__).resolve().parent.parent)
).resolve()
BLAZAR_BIN = Path(
    os.environ.get("BLAZAR_BIN", str(REPO / "target" / "release" / "blazar"))
).resolve()
os.environ["BLAZAR_BIN"] = str(BLAZAR_BIN)
os.environ.setdefault("BLAZAR_VALIDATE_MEM_FLOOR_MIB", "2048")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

sys.path.insert(0, str(REPO / "scripts"))
import validate as V  # noqa: E402
import bench_saturation as BS  # noqa: E402
import bench_charts as BC  # noqa: E402

OLLAMA_BASE = os.environ.get("M1_OLLAMA_BASE", "http://127.0.0.1:11434")
OLLAMA_MODEL = "qwen2.5:0.5b"
MODEL_FULL = "qwen2.5-0.5b-instruct"  # full row name; the short prefix is ambiguous
BLZ_BASE = f"http://127.0.0.1:{V.PORT}"

SEED = 7
GEN_MAX_TOKENS = 128
OVERHEAD_MAX_TOKENS = 16
REQ_TIMEOUT = 120.0
LADDER = [1, 2, 4, 8]
N_PER_LEVEL = 32
OVERHEAD_N = 200
DIRECT_PORT = int(os.environ.get("M1_DIRECT_PORT", "8298"))
RAW_DIR = Path(
    os.environ.get("M1_OUT_DIR", os.path.join(tempfile.gettempdir(), "blazar-m1-raw"))
)
ARTIFACT_DIR = REPO / "bench-artifacts" / "2026-11-23-sota-roadmap"

QUALITY_BATTERY = [
    "What is the capital of France? Answer with the city name only.",
    "Count from one to ten, one number per line.",
    "Name the four seasons, comma-separated.",
    "What is 17 + 25? Answer with the number only.",
    "Write a Python function that returns the square of a number, in one line.",
    "Translate to German: The weather is beautiful today.",
    "List three primary colors, comma-separated.",
    "What planet is closest to the sun? Answer with the name only.",
    "Reverse the letters of the word 'stream' and print the result only.",
    "What is 9 times 6? Answer with the number only.",
    "Write a haiku about rivers.",
    "Name the largest ocean on Earth, one word.",
    "Give the chemical symbol for gold, one word.",
    "Summarize in one sentence: A cat sat on a mat and fell asleep.",
    "What color do you get mixing blue and yellow paint? One word.",
    "Write the alphabet's first five letters, no spaces.",
    "How many days are in a leap year? Answer with the number only.",
    "Translate to Spanish: Good morning, my friend.",
    "Write a bash one-liner that lists files in the current directory.",
    "What is the boiling point of water in Celsius at sea level? Number only.",
]


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def prompt_digest(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]


# ------------------------------------------------------------------ probes
def gpu_snapshot() -> dict | None:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode != 0:
            return None
        util, used, total = (
            x.strip() for x in out.stdout.strip().splitlines()[0].split(",")
        )
        return {
            "util_pct": int(util),
            "vram_used_mib": int(used),
            "vram_total_mib": int(total),
        }
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def ollama_json(path: str, body: dict | None = None, timeout: float = 30.0) -> dict:
    if body is None:
        req = urllib.request.Request(OLLAMA_BASE + path)
    else:
        req = urllib.request.Request(
            OLLAMA_BASE + path,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


# ----------------------------------------------------------------- clients
def chat_nonstream(
    base: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    extra: dict | None = None,
) -> dict:
    """One non-stream chat completion. Returns wall_ms, usage and text;
    raises nothing - failures return ok=False + error text."""
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "seed": SEED,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if extra:
        body.update(extra)
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=REQ_TIMEOUT) as r:
            payload = json.loads(r.read())
        wall_ms = (time.monotonic() - t0) * 1000.0
        choice = (payload.get("choices") or [{}])[0]
        usage = payload.get("usage") or {}
        text = (choice.get("message") or {}).get("content") or ""
        return {
            "ok": True,
            "wall_ms": round(wall_ms, 3),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "text": text,
            "finish_reason": choice.get("finish_reason"),
        }
    except Exception as e:  # recorded, never raised
        return {
            "ok": False,
            "wall_ms": (time.monotonic() - t0) * 1000.0,
            "error": f"{type(e).__name__}: {e}",
        }


def chat_stream(base: str, model: str, messages: list[dict], max_tokens: int) -> dict:
    """One streaming chat completion. TTFT = time to the first chunk that
    carries non-empty content (not the role-only first event some stacks
    emit); t_last_ms = time of the final data chunk; tokens prefer the
    authoritative usage field, else the content-delta count (llama.cpp and
    ollama both stream one token per non-empty delta)."""
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "seed": SEED,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    first = None
    last = None
    deltas = 0
    usage_completion = 0
    text_parts: list[str] = []
    try:
        with urllib.request.urlopen(req, timeout=REQ_TIMEOUT) as r:
            for raw in r:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    last = time.monotonic()
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                last = time.monotonic()
                if chunk.get("choices"):
                    delta = chunk["choices"][0].get("delta") or {}
                    content = delta.get("content")
                    if content:
                        if first is None:
                            first = last
                        deltas += 1
                        text_parts.append(content)
                usage = chunk.get("usage") or {}
                if usage.get("completion_tokens"):
                    usage_completion = usage["completion_tokens"]
        if first is None:
            return {
                "ok": False,
                "wall_ms": (time.monotonic() - t0) * 1000.0,
                "error": "stream produced no content tokens",
            }
        return {
            "ok": True,
            "wall_ms": round((time.monotonic() - t0) * 1000.0, 3),
            "ttft_ms": round((first - t0) * 1000.0, 3),
            "t_last_ms": round((last - t0) * 1000.0, 3) if last else None,
            "completion_tokens": usage_completion if usage_completion > 0 else deltas,
            "text": "".join(text_parts),
        }
    except Exception as e:
        return {
            "ok": False,
            "wall_ms": (time.monotonic() - t0) * 1000.0,
            "error": f"{type(e).__name__}: {e}",
        }


# ------------------------------------------------------------ raw records
class Jsonl:
    def __init__(self, lane: str) -> None:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        self.path = RAW_DIR / f"{lane}.jsonl"
        self._fh = open(self.path, "a", encoding="utf-8")
        self._lock = threading.Lock()

    def write(self, rec: dict) -> None:
        rec.setdefault("ts", now_iso())
        with self._lock:
            self._fh.write(json.dumps(rec, sort_keys=True) + "\n")
            self._fh.flush()

    def close(self) -> None:
        self._fh.close()


# ------------------------------------------------------------ perf prompts
def level_prompt(level: int, idx: int) -> str:
    """Deterministic unique prompt per (level, request index). md5-seeded
    digits (NOT Python hash(), which is process-salted) so re-runs and
    cross-product arms share the same workload shape while no request
    re-hits a warm prefix from an earlier one."""
    material = f"m1-{level}-{idx}"
    digest = hashlib.md5(material.encode()).hexdigest()
    digits = " ".join(
        str((int(digest[i : i + 4], 16) * (i + 7)) % 9973) for i in range(0, 24, 4)
    )
    return (
        f"Repeat these numbers back verbatim, then count down from 50 in "
        f"words, entry {material}: {digits}. Produce at least {GEN_MAX_TOKENS} tokens."
    )


def drive_level(
    base: str,
    model: str,
    product: str,
    mode: str,
    c: int,
    n: int,
    log: Jsonl,
    extra_body: dict | None = None,
) -> None:
    """One ladder level: n requests total, c in flight (threads). GPU is
    sampled before/after the level and once mid-level - contention is
    recorded on the level record, never used to discard requests."""
    gpu_before = gpu_snapshot()
    results: list[dict] = []
    lock = threading.Lock()
    next_idx = [0]

    def worker() -> None:
        while True:
            with lock:
                idx = next_idx[0]
                next_idx[0] += 1
                if idx >= n:
                    return
            prompt = level_prompt(c, idx)
            messages = [{"role": "user", "content": prompt}]
            rec: dict = {
                "type": "request",
                "product": product,
                "mode": mode,
                "c": c,
                "idx": idx,
                "prompt_sha": prompt_digest(prompt),
                "max_tokens": GEN_MAX_TOKENS,
            }
            if mode == "stream":
                out = chat_stream(base, model, messages, GEN_MAX_TOKENS)
            else:
                out = chat_nonstream(base, model, messages, GEN_MAX_TOKENS, extra_body)
            rec.update(out)
            with lock:
                results.append(rec)
            log.write(rec)

    threads = [threading.Thread(target=worker) for _ in range(c)]
    t0 = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    level_secs = time.monotonic() - t0
    gpu_after = gpu_snapshot()
    log.write(
        {
            "type": "level",
            "product": product,
            "mode": mode,
            "c": c,
            "n": n,
            "level_secs": round(level_secs, 3),
            "gpu_before": gpu_before,
            "gpu_after": gpu_after,
            "ok_count": sum(1 for r in results if r.get("ok")),
            "err_count": sum(1 for r in results if not r.get("ok")),
        }
    )
    print(
        f"  [{product} {mode} C={c}] {sum(1 for r in results if r.get('ok'))}/{n} ok, "
        f"level {level_secs:.1f}s, gpu {gpu_before and gpu_before['util_pct']}%->{gpu_after and gpu_after['util_pct']}%"
    )


# ------------------------------------------------------------ daemon arms
def _child_pid_of(pid: int) -> int | None:
    """Direct child pid of `pid` on Linux (/proc/<pid>/task/*/children).
    Used only to replicate the engine child's environment for the direct
    leg; None falls back to inheriting this process's environment."""
    try:
        pids: list[int] = []
        for f in Path(f"/proc/{pid}/task").glob("*/children"):
            pids.extend(int(x) for x in f.read_text().split())
        return pids[0] if pids else None
    except OSError:
        return None


class BlazarLane:
    """Owns the sandbox + daemon lifecycle for the blazar arm so every
    lane gets a warm-or-cold daemon by explicit choice, and teardown is
    guaranteed (H19: every start has a stop on every exit path)."""

    def __init__(self) -> None:
        self.sb = V.Sandbox()
        self.daemon = V.Daemon(self.sb)

    def start(self, cold: bool = False) -> None:
        if not cold and self.daemon.proc is not None:
            return
        cfg = {"port": V.PORT, "model_load_timeout_secs": 600}
        self.daemon.start(cfg=cfg, floor_model=MODEL_FULL)

    def warm(self, max_tokens: int = 8) -> dict:
        return chat_nonstream(
            BLZ_BASE, MODEL_FULL, [{"role": "user", "content": "ping"}], max_tokens
        )

    def stop(self) -> None:
        try:
            self.daemon.stop()
        except Exception as e:  # teardown must never mask the result path
            print(f"  [daemon stop: {type(e).__name__}: {e}]", file=sys.stderr)


# ------------------------------------------------------------------- lanes
def lane_pilot() -> int:
    log = Jsonl("pilot")
    env: dict = {"type": "env", "lane": "pilot", "ts": now_iso()}
    try:
        ver = ollama_json("/api/version")
        env["ollama_version"] = ver.get("version")
        tags = ollama_json("/api/tags")
        env["ollama_models"] = sorted(
            m["name"] for m in tags.get("models", []) if m.get("name")
        )
        show = ollama_json("/api/show", {"model": OLLAMA_MODEL})
        info = show.get("model_info") or {}
        env["ollama_model_info"] = {
            k: v
            for k, v in info.items()
            if k
            in (
                "general.architecture",
                "general.file_type",
                "general.parameter_count",
                "qwen2.context_length",
                "general.license",
            )
        }
        env["ollama_quant"] = show.get("details", {}).get("quantization_level")
    except Exception as e:
        env["ollama_probe_error"] = f"{type(e).__name__}: {e}"
    env["ollama_base"] = OLLAMA_BASE
    env["ollama_model"] = OLLAMA_MODEL

    env["blazar_bin"] = str(BLAZAR_BIN)
    env["blazar_bin_size"] = BLAZAR_BIN.stat().st_size if BLAZAR_BIN.exists() else None
    env["blazar_bin_sha256"] = (
        hashlib.sha256(BLAZAR_BIN.read_bytes()).hexdigest()
        if BLAZAR_BIN.exists()
        else None
    )
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=10,
        )
        env["worktree_head"] = head.stdout.strip()
        env["worktree_dirty"] = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=REPO,
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout.strip()
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        env["git_error"] = str(e)
    env["gpu"] = gpu_snapshot()
    env["ram_mib"] = {
        "total": int(
            os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1048576
        ),
        "available": int(
            open("/proc/meminfo").read().split("MemAvailable:")[1].split()[0]
        )
        / 1024,
    }
    env["blazar_model"] = MODEL_FULL
    env["port"] = V.PORT
    env["direct_port"] = DIRECT_PORT

    # Dialect probe: does the ollama OpenAI endpoint take our payload?
    dialect = chat_nonstream(
        OLLAMA_BASE, OLLAMA_MODEL, [{"role": "user", "content": "Say 'ok'."}], 8
    )
    env["ollama_openai_dialect_ok"] = dialect.get("ok")
    log.write(env)
    with open(RAW_DIR / "m1_env.json", "w") as f:
        json.dump(env, f, indent=1, sort_keys=True)

    # Two measured requests per product per mode (pilot, not campaign).
    print("  pilot ollama nonstream/stream ...")
    log.write(
        {
            "type": "request",
            "product": "ollama",
            "mode": "nonstream",
            "pilot": True,
            **chat_nonstream(
                OLLAMA_BASE, OLLAMA_MODEL, [{"role": "user", "content": "Pilot A."}], 32
            ),
        }
    )
    log.write(
        {
            "type": "request",
            "product": "ollama",
            "mode": "stream",
            "pilot": True,
            **chat_stream(
                OLLAMA_BASE, OLLAMA_MODEL, [{"role": "user", "content": "Pilot B."}], 32
            ),
        }
    )

    lane = BlazarLane()
    try:
        print("  pilot blazar sandbox daemon ...")
        lane.start()
        out = lane.warm()
        log.write(
            {
                "type": "request",
                "product": "blazar",
                "mode": "nonstream",
                "pilot": True,
                **out,
            }
        )
        log.write(
            {
                "type": "request",
                "product": "blazar",
                "mode": "stream",
                "pilot": True,
                **chat_stream(
                    BLZ_BASE, MODEL_FULL, [{"role": "user", "content": "Pilot B."}], 32
                ),
            }
        )
        # CLI evidence (engine rows) read against the sandbox XDG homes.
        sb_env = {
            **os.environ,
            "XDG_CONFIG_HOME": lane.sb.config_home,
            "XDG_DATA_HOME": lane.sb.data_home,
        }
        env2 = {"type": "blazar_evidence"}
        try:
            out = subprocess.run(
                [str(BLAZAR_BIN), "engine", "list"],
                capture_output=True,
                text=True,
                timeout=60,
                env=sb_env,
            )
            env2["engine_list_rc"] = out.returncode
            env2["engine_list_head"] = out.stdout[:400]
        except OSError as e:
            env2["engine_list_error"] = str(e)
        log.write(env2)
    finally:
        lane.stop()
    log.close()
    return 0 if dialect.get("ok") else 3


def lane_warm() -> int:
    log = Jsonl("warm")
    print("warm lane: ollama first (blazar engine down during it)")
    for mode in ("stream", "nonstream"):
        for c in LADDER:
            drive_level(OLLAMA_BASE, OLLAMA_MODEL, "ollama", mode, c, N_PER_LEVEL, log)
    print("warm lane: blazar")
    lane = BlazarLane()
    try:
        lane.start()
        lane.warm()
        for mode in ("stream", "nonstream"):
            for c in LADDER:
                drive_level(BLZ_BASE, MODEL_FULL, "blazar", mode, c, N_PER_LEVEL, log)
    finally:
        lane.stop()
    log.close()
    return 0


def lane_cold() -> int:
    log = Jsonl("cold")
    print("cold lane: ollama (request-scoped keep_alive=0 unload, service untouched)")
    try:
        ps0 = ollama_json("/api/ps")
        log.write(
            {
                "type": "ollama_ps",
                "phase": "before",
                "models": [m.get("name") for m in ps0.get("models", [])],
            }
        )
        if any(
            m.get("name", "").startswith("qwen2.5:0.5b") for m in ps0.get("models", [])
        ):
            # API-level unload vehicle: tiny request that releases the model
            # afterwards. No service restart, no store change.
            ollama_json(
                "/api/chat",
                {
                    "model": OLLAMA_MODEL,
                    "messages": [{"role": "user", "content": "un"}],
                    "stream": False,
                    "keep_alive": 0,
                },
                timeout=120.0,
            )
            time.sleep(2.0)
            ps1 = ollama_json("/api/ps")
            log.write(
                {
                    "type": "ollama_ps",
                    "phase": "after_unload",
                    "models": [m.get("name") for m in ps1.get("models", [])],
                }
            )
        t0 = time.monotonic()
        out = chat_nonstream(
            OLLAMA_BASE,
            OLLAMA_MODEL,
            [{"role": "user", "content": "Cold load probe: reply with one word."}],
            32,
        )
        out["wall_ms"] = (
            round((time.monotonic() - t0) * 1000.0, 3)
            if not out.get("ok")
            else out["wall_ms"]
        )
        log.write({"type": "cold_request", "product": "ollama", **out})
        ps2 = ollama_json("/api/ps")
        log.write(
            {
                "type": "ollama_ps",
                "phase": "after_request",
                "models": [m.get("name") for m in ps2.get("models", [])],
            }
        )
    except Exception as e:
        log.write(
            {
                "type": "cold_error",
                "product": "ollama",
                "error": f"{type(e).__name__}: {e}",
            }
        )

    print("cold lane: blazar (fresh daemon, first request pays the lazy spawn)")
    lane = BlazarLane()
    try:
        lane.start(cold=True)
        t0 = time.monotonic()
        out = chat_nonstream(
            BLZ_BASE,
            MODEL_FULL,
            [{"role": "user", "content": "Cold load probe: reply with one word."}],
            32,
        )
        if not out.get("ok"):
            out["wall_ms"] = round((time.monotonic() - t0) * 1000.0, 3)
        log.write({"type": "cold_request", "product": "blazar", **out})
    finally:
        lane.stop()
    log.close()
    return 0


def lane_overhead() -> int:
    log = Jsonl("overhead")
    print(
        "overhead lane: blazar gateway vs matched direct child (C=1, N=%d)" % OVERHEAD_N
    )
    lane = BlazarLane()
    direct_proc = None
    direct_log = None
    try:
        lane.start()
        lane.warm()
        gw_pid = lane.daemon.proc.pid
        # bench_saturation's /proc walk picks the daemon's serving child
        # (llama-server argv carrying a -m/--model flag) - probe argvs
        # run the census and exit, so they never qualify.
        argv = BS.child_argv_from_proc(gw_pid)
        if not argv or "llama-server" not in Path(argv[0]).name:
            log.write(
                {"type": "overhead_error", "error": "gateway child argv not found"}
            )
            return 4
        child = _child_pid_of(gw_pid)
        log.write({"type": "gateway_child", "argv": argv, "child_pid": child})

        state_dir = BS.make_direct_state_dir(argv)
        dargv = BS.matched_direct_argv(argv, DIRECT_PORT, state_dir)
        log.write({"type": "direct_child", "argv": dargv, "state_dir": str(state_dir)})
        env = V.child_environ(child) if child else dict(os.environ)

        # Gateway leg first, with its engine child up and warm.
        prompt = "Overhead probe: reply with one short sentence."
        messages = [{"role": "user", "content": prompt}]
        log.write({"type": "gpu", "phase": "before", "snap": gpu_snapshot()})
        for i in range(OVERHEAD_N):
            if i == OVERHEAD_N // 2:
                log.write({"type": "gpu", "phase": "mid", "snap": gpu_snapshot()})
            log.write(
                {
                    "type": "request",
                    "leg": "gateway",
                    "idx": i,
                    "mode": "nonstream",
                    **chat_nonstream(
                        BLZ_BASE, MODEL_FULL, messages, OVERHEAD_MAX_TOKENS
                    ),
                }
            )
        log.write({"type": "gpu", "phase": "between", "snap": gpu_snapshot()})

        # Stop the gateway BEFORE spawning the direct child: on a
        # memory-tight box two full engine contexts (plus any co-tenant)
        # cannot both hold VRAM, and the C=1 latency comparison needs no
        # co-residency - each leg runs alone, same binary, same argv shape.
        lane.stop()
        time.sleep(2.0)

        direct_log = open(RAW_DIR / "direct_child.log", "w")
        direct_proc = subprocess.Popen(
            dargv, stdout=direct_log, stderr=subprocess.STDOUT, env=env, cwd=str(REPO)
        )
        # The direct model handle: matched_direct_argv swaps --alias to
        # sat-direct, so the OpenAI route reaches it by that name.
        direct_model = "sat-direct"
        BS.wait_healthy(f"http://127.0.0.1:{DIRECT_PORT}", deadline_s=90.0)
        # One warm request: the first request on a fresh child pays slot
        # initialization; the overhead figure is steady-state.
        chat_nonstream(
            f"http://127.0.0.1:{DIRECT_PORT}",
            direct_model,
            messages,
            OVERHEAD_MAX_TOKENS,
        )
        for i in range(OVERHEAD_N):
            if i == OVERHEAD_N // 2:
                log.write({"type": "gpu", "phase": "mid", "snap": gpu_snapshot()})
            log.write(
                {
                    "type": "request",
                    "leg": "direct",
                    "idx": i,
                    "mode": "nonstream",
                    **chat_nonstream(
                        f"http://127.0.0.1:{DIRECT_PORT}",
                        direct_model,
                        messages,
                        OVERHEAD_MAX_TOKENS,
                    ),
                }
            )
        log.write({"type": "gpu", "phase": "after", "snap": gpu_snapshot()})
        print(
            "  overhead legs done (gateway, then direct alone, %d requests each)"
            % OVERHEAD_N
        )
    finally:
        if direct_proc is not None:
            direct_proc.terminate()
            try:
                direct_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                direct_proc.kill()
                direct_proc.wait(timeout=10)
        if direct_log is not None:
            direct_log.close()
        lane.stop()
    log.close()
    return 0


def lane_quality() -> int:
    log = Jsonl("quality")
    print(
        "quality lane: %d-prompt temp-0 battery, both products" % len(QUALITY_BATTERY)
    )
    print("  ollama ...")
    for i, prompt in enumerate(QUALITY_BATTERY):
        log.write(
            {
                "type": "quality",
                "product": "ollama",
                "idx": i,
                "prompt_sha": prompt_digest(prompt),
                "prompt": prompt,
                **chat_nonstream(
                    OLLAMA_BASE,
                    OLLAMA_MODEL,
                    [{"role": "user", "content": prompt}],
                    GEN_MAX_TOKENS,
                ),
            }
        )
    print("  blazar ...")
    lane = BlazarLane()
    try:
        lane.start()
        lane.warm()
        for i, prompt in enumerate(QUALITY_BATTERY):
            log.write(
                {
                    "type": "quality",
                    "product": "blazar",
                    "idx": i,
                    "prompt_sha": prompt_digest(prompt),
                    "prompt": prompt,
                    **chat_nonstream(
                        BLZ_BASE,
                        MODEL_FULL,
                        [{"role": "user", "content": prompt}],
                        GEN_MAX_TOKENS,
                    ),
                }
            )
    finally:
        lane.stop()
    log.close()
    return 0


# ------------------------------------------------------------------ receipt
def pct(values: list[float], p: float) -> float | None:
    return BS.pct(values, p)


def load_lane(lane: str) -> list[dict]:
    path = RAW_DIR / f"{lane}.jsonl"
    if not path.exists():
        return []
    recs = []
    for line in path.read_text().splitlines():
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return recs


def summarize_warm(recs: list[dict]) -> list[dict]:
    rows = []
    for product in ("ollama", "blazar"):
        for mode in ("stream", "nonstream"):
            for c in LADDER:
                reqs = [
                    r
                    for r in recs
                    if r.get("type") == "request"
                    and r.get("product") == product
                    and r.get("mode") == mode
                    and r.get("c") == c
                ]
                ok = [r for r in reqs if r.get("ok")]
                errs = len(reqs) - len(ok)
                row = {
                    "product": product,
                    "mode": mode,
                    "c": c,
                    "n": len(reqs),
                    "ok": len(ok),
                    "errors": errs,
                }
                if ok:
                    walls = [r["wall_ms"] for r in ok]
                    row["wall_p50_ms"] = round(pct(walls, 50), 2)
                    row["wall_p99_ms"] = round(pct(walls, 99), 2)
                    toks = [r.get("completion_tokens") or 0 for r in ok]
                    row["mean_completion_tokens"] = round(sum(toks) / len(toks), 1)
                    if mode == "stream":
                        ttfts = [
                            r["ttft_ms"] for r in ok if r.get("ttft_ms") is not None
                        ]
                        decode_tps = [
                            r["completion_tokens"]
                            / ((r["t_last_ms"] - r["ttft_ms"]) / 1000.0)
                            for r in ok
                            if r.get("t_last_ms")
                            and r.get("ttft_ms") is not None
                            and r["t_last_ms"] > r["ttft_ms"]
                            and r.get("completion_tokens")
                        ]
                        if ttfts:
                            row["ttft_p50_ms"] = round(pct(ttfts, 50), 2)
                            row["ttft_p95_ms"] = round(pct(ttfts, 95), 2)
                            row["ttft_p99_ms"] = round(pct(ttfts, 99), 2)
                        if decode_tps:
                            row["decode_tps_p50"] = round(pct(decode_tps, 50), 1)
                    level_secs = sum(
                        r.get("level_secs", 0)
                        for r in recs
                        if r.get("type") == "level"
                        and r.get("product") == product
                        and r.get("mode") == mode
                        and r.get("c") == c
                    )
                    row["aggregate_tps"] = (
                        round(sum(toks) / level_secs, 1) if level_secs else None
                    )
                rows.append(row)
    return rows


def summarize_overhead(recs: list[dict]) -> dict:
    gw = {
        r["idx"]: r
        for r in recs
        if r.get("type") == "request" and r.get("leg") == "gateway"
    }
    dr = {
        r["idx"]: r
        for r in recs
        if r.get("type") == "request" and r.get("leg") == "direct"
    }
    pairs = []
    for i in sorted(set(gw) & set(dr)):
        if gw[i].get("ok") and dr[i].get("ok"):
            pairs.append((gw[i]["wall_ms"], dr[i]["wall_ms"], i))
    if not pairs:
        return {"pairs": 0}
    deltas_us = [(g - d) * 1000.0 for g, d, _ in pairs]
    return {
        "pairs": len(pairs),
        "gateway_wall_p50_ms": round(pct([g for g, _, _ in pairs], 50), 3),
        "direct_wall_p50_ms": round(pct([d for _, d, _ in pairs], 50), 3),
        "gateway_wall_p95_ms": round(pct([g for g, _, _ in pairs], 95), 3),
        "direct_wall_p95_ms": round(pct([d for _, d, _ in pairs], 95), 3),
        "gateway_wall_p99_ms": round(pct([g for g, _, _ in pairs], 99), 3),
        "direct_wall_p99_ms": round(pct([d for _, d, _ in pairs], 99), 3),
        "delta_us_p50": round(pct(deltas_us, 50), 1),
        "delta_us_p95": round(pct(deltas_us, 95), 1),
        "delta_us_p99": round(pct(deltas_us, 99), 1),
        "gateway_over_direct_ratio_p50": round(
            pct([g for g, _, _ in pairs], 50) / pct([d for _, d, _ in pairs], 50), 3
        ),
    }


def summarize_quality(recs: list[dict]) -> dict:
    by_idx: dict[int, dict[str, str]] = {}
    for r in recs:
        if r.get("type") == "quality" and r.get("ok"):
            by_idx.setdefault(r["idx"], {})[r["product"]] = r.get("text") or ""
    pairs = {i: v for i, v in by_idx.items() if "ollama" in v and "blazar" in v}
    exact = sum(1 for v in pairs.values() if v["ollama"].strip() == v["blazar"].strip())
    tok_agree = []
    for v in pairs.values():
        a, b = v["ollama"].split(), v["blazar"].split()
        if not a and not b:
            tok_agree.append(1.0)
            continue
        common = sum(1 for x, y in zip(a, b) if x == y)
        tok_agree.append(common / max(len(a), len(b)))
    return {
        "prompts": len(pairs),
        "exact_match": exact,
        "exact_match_pct": round(100.0 * exact / len(pairs), 1) if pairs else None,
        "token_agreement_pct": round(100.0 * sum(tok_agree) / len(tok_agree), 1)
        if tok_agree
        else None,
        "samples": [
            {
                "idx": i,
                "prompt": QUALITY_BATTERY[i],
                "ollama": v["ollama"],
                "blazar": v["blazar"],
            }
            for i, v in sorted(pairs.items())[:2]
        ],
    }


def lane_receipt() -> int:
    print("receipt lane: assembling from raw (no re-measurement)")
    warm = load_lane("warm")
    cold = load_lane("cold")
    overhead = load_lane("overhead")
    quality = load_lane("quality")
    pilot = load_lane("pilot")
    env = {}
    env_path = RAW_DIR / "m1_env.json"
    if env_path.exists():
        env = json.loads(env_path.read_text())

    warm_rows = summarize_warm(warm)
    overhead_sum = summarize_overhead(overhead)
    quality_sum = summarize_quality(quality)
    cold_rows = [
        {
            k: v
            for k, v in r.items()
            if k in ("product", "ok", "wall_ms", "error", "type")
        }
        for r in cold
        if r.get("type") == "cold_request"
    ]

    summary = {
        "campaign": "m1-ollama-head-to-head",
        "ts": now_iso(),
        "env": env,
        "matched_pair": {
            "model": "Qwen2.5-0.5B-Instruct",
            "quant": "Q4_K_M",
            "ollama_model": OLLAMA_MODEL,
            "blazar_model": MODEL_FULL,
            "ollama_size_bytes": 397821319,
        },
        "protocol": {
            "wire": "OpenAI /v1/chat/completions both sides",
            "sampling": {
                "temperature": 0,
                "seed": SEED,
                "seed_note": "inert at temp-0, sent for symmetry",
                "max_tokens": GEN_MAX_TOKENS,
            },
            "ladder": LADDER,
            "n_per_level": N_PER_LEVEL,
            "overhead_n": OVERHEAD_N,
            "overhead_max_tokens": OVERHEAD_MAX_TOKENS,
            "cache_state": "warm primary (1 warmup + measured); cold one-shot",
        },
        "warm": warm_rows,
        "cold": cold_rows,
        "overhead_us": overhead_sum,
        "quality": quality_sum,
        "raw_counts": {
            lane: len(load_lane(lane))
            for lane in ("pilot", "warm", "cold", "overhead", "quality")
        },
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = ARTIFACT_DIR / "m1-ollama-head-to-head-receipt.json"
    out_json.write_text(json.dumps(summary, indent=1, sort_keys=True))

    # CSV: one row per warm (product, mode, c) cell.
    cols = [
        "product",
        "mode",
        "c",
        "n",
        "ok",
        "errors",
        "wall_p50_ms",
        "wall_p99_ms",
        "ttft_p50_ms",
        "ttft_p95_ms",
        "ttft_p99_ms",
        "decode_tps_p50",
        "aggregate_tps",
        "mean_completion_tokens",
    ]
    with open(ARTIFACT_DIR / "m1-warm-ladder.csv", "w", newline="") as f:
        f.write(",".join(cols) + "\n")
        for row in warm_rows:
            f.write(
                ",".join(
                    str(row.get(c) if row.get(c) is not None else "") for c in cols
                )
                + "\n"
            )

    # Charts (deterministic house SVG).
    charts_dir = ARTIFACT_DIR / "m1-charts"
    charts_dir.mkdir(exist_ok=True)
    o_color = BC.FAMILY_COLOR.get("ollama", "#009E73")
    b_color = BC.FAMILY_COLOR.get("blazar", "#0072B2")

    def stream_metric(metric: str, title: str, unit: str) -> None:
        series = []
        for product, color in (("ollama", o_color), ("blazar", b_color)):
            pts = [
                (row["c"], row[metric])
                for row in warm_rows
                if row["mode"] == "stream" and row.get(metric) is not None
            ]
            if pts:
                series.append((product, sorted(pts), color))
        if series:
            (charts_dir / f"m1-stream-{metric}.svg").write_text(
                BC.line_svg(
                    title,
                    "warm ladder, C=1..8, N=32 per level",
                    series,
                    "concurrency",
                    unit,
                    footnote="m1 head-to-head, temp-0, matched model",
                )
            )

    stream_metric("ttft_p50_ms", "TTFT p50 vs concurrency (stream)", "ms")
    stream_metric(
        "aggregate_tps", "aggregate throughput vs concurrency (stream)", "tok/s"
    )

    if overhead_sum.get("pairs"):
        rows = [
            (
                "gateway - direct p50",
                overhead_sum["delta_us_p50"],
                overhead_sum["delta_us_p50"],
                overhead_sum["delta_us_p50"],
                b_color,
            ),
            (
                "gateway - direct p95",
                overhead_sum["delta_us_p95"],
                overhead_sum["delta_us_p95"],
                overhead_sum["delta_us_p95"],
                b_color,
            ),
            (
                "gateway - direct p99",
                overhead_sum["delta_us_p99"],
                overhead_sum["delta_us_p99"],
                overhead_sum["delta_us_p99"],
                b_color,
            ),
        ]
        (charts_dir / "m1-overhead-us.svg").write_text(
            BC.hbar_svg(
                "Blazar per-request gateway overhead (C=1, non-stream, max_tokens=16)",
                "wall-latency delta vs matched direct child, N=%d"
                % overhead_sum["pairs"],
                rows,
                "us",
                footnote="positive = gateway costs that many microseconds",
            )
        )

    # Markdown receipt.
    md: list[str] = []
    md.append("# M1 - Blazar vs ollama 0.40.17 head-to-head receipt\n")
    md.append(
        "Matched campaign per roadmap M1: same model (Qwen2.5-0.5B-Instruct Q4_K_M), "
        "same prompts, temperature 0, seed 7, warm cache state, plus the non-stream "
        "per-request gateway overhead lane in microseconds.\n"
    )
    md.append("## Environment\n")
    md.append("| field | value |")
    md.append("|---|---|")
    for k in (
        "ollama_version",
        "ollama_quant",
        "blazar_bin_size",
        "worktree_head",
        "port",
        "direct_port",
    ):
        v = env.get(k)
        md.append(f"| {k} | {v} |")
    if env.get("gpu"):
        md.append(f"| gpu | {env['gpu']} |")
    md.append(
        "\nMatched pair: ollama `{}` ({:,} bytes) vs blazar `{}`.\n".format(
            OLLAMA_MODEL,
            env.get("matched", {}).get("ollama_size_bytes", 397821319),
            MODEL_FULL,
        )
    )
    md.append("\n## Warm ladder (stream)\n")
    md.append(
        "| product | C | ok/err | TTFT p50 ms | TTFT p95 ms | decode tok/s p50 | aggregate tok/s |"
    )
    md.append("|---|---|---|---|---|---|---|")
    for row in warm_rows:
        if row["mode"] != "stream":
            continue
        md.append(
            "| {} | {} | {}/{} | {} | {} | {} | {} |".format(
                row["product"],
                row["c"],
                row["ok"],
                row["errors"],
                row.get("ttft_p50_ms", "-"),
                row.get("ttft_p95_ms", "-"),
                row.get("decode_tps_p50", "-"),
                row.get("aggregate_tps", "-"),
            )
        )
    md.append("\n## Warm ladder (non-stream)\n")
    md.append("| product | C | ok/err | wall p50 ms | wall p99 ms | aggregate tok/s |")
    md.append("|---|---|---|---|---|---|")
    for row in warm_rows:
        if row["mode"] != "nonstream":
            continue
        md.append(
            "| {} | {} | {}/{} | {} | {} | {} |".format(
                row["product"],
                row["c"],
                row["ok"],
                row["errors"],
                row.get("wall_p50_ms", "-"),
                row.get("wall_p99_ms", "-"),
                row.get("aggregate_tps", "-"),
            )
        )
    md.append("\n## Cold start (one-shot)\n")
    for row in cold_rows:
        md.append(
            "- {}: ok={} wall_ms={}".format(
                row.get("product"), row.get("ok"), row.get("wall_ms")
            )
        )
    md.append("\n## Per-request gateway overhead (the published figure)\n")
    md.append(
        "Blazar gateway vs matched direct child (gateway engine argv minus the auth-file "
        "flag, state dirs redirected, same binary/model/ctx): C=1, non-stream, "
        "max_tokens=16, N={} paired requests.\n".format(overhead_sum.get("pairs", 0))
    )
    md.append(
        "```json\n" + json.dumps(overhead_sum, indent=1, sort_keys=True) + "\n```\n"
    )
    md.append("\n## Quality (temp-0, 20-prompt battery)\n")
    md.append(
        "```json\n"
        + json.dumps({k: v for k, v in quality_sum.items() if k != "samples"}, indent=1)
        + "\n```\n"
    )
    if quality_sum.get("samples"):
        md.append("\n### Spot samples (first 2 prompts)\n")
        for s in quality_sum["samples"]:
            md.append("**Prompt {}:** {}\n".format(s["idx"], s["prompt"]))
            md.append("- ollama: `{}`\n".format(s["ollama"][:200].replace("\n", " ")))
            md.append("- blazar: `{}`\n".format(s["blazar"][:200].replace("\n", " ")))
    md.append("\n## Caveats\n")
    md.append(
        "- Engine builds differ by product reality: ollama 0.40.17 bundles its own "
        "llama.cpp fork; blazar serves via its llama.cpp b12728-cuda lane. The "
        "overhead lane isolates the gateway hop by comparing blazar-gateway vs "
        "blazar-direct on the SAME engine build.\n"
    )
    md.append(
        "- ollama was treated read-only (probes + generation + request-scoped "
        "keep_alive=0 for the cold unload; no restart, no store change).\n"
    )
    md.append(
        "- Co-tenancy: another blazar session (GPU port 36829) was live on this host "
        "during parts of the run; per-level GPU snapshots are in warm.jsonl and any "
        "contended levels are identifiable there - no request was discarded.\n"
    )
    md.append(
        "- Charts: m1-charts/m1-stream-ttft_p50_ms.svg, m1-charts/m1-stream-aggregate_tps.svg, "
        "m1-charts/m1-overhead-us.svg.\n"
    )
    md.append("\n## Reproduction\n")
    md.append("The campaign runs through the repository validation harness "
              "(scripts/validate.py Sandbox/Daemon) and the matched-argv machinery of "
              "scripts/bench_saturation.py against the release binary; charts render via "
              "scripts/bench_charts.py.\n")
    md.append("```bash\n")
    md.append("python3 m1_head_to_head.py --lane pilot   # env evidence + probes\n")
    md.append("python3 m1_head_to_head.py --lane warm\n")
    md.append("python3 m1_head_to_head.py --lane cold\n")
    md.append("python3 m1_head_to_head.py --lane overhead\n")
    md3 = "python3 m1_head_to_head.py --lane quality && python3 m1_head_to_head.py --lane receipt\n"
    md.append(md3)
    md.append("```\n")
    md.append("Raw JSONL (every request incl. failures): `{}`.\n".format(str(RAW_DIR)))

    (ARTIFACT_DIR / "m1-ollama-head-to-head-receipt.md").write_text(
        "\n".join(part.rstrip("\n") for part in md) + "\n"
    )
    print("receipt ->", ARTIFACT_DIR / "m1-ollama-head-to-head-receipt.md")
    print("summary ->", out_json)
    print("csv     ->", ARTIFACT_DIR / "m1-warm-ladder.csv")
    print("charts  ->", ARTIFACT_DIR / "m1-charts/")
    return 0


LANES = {
    "pilot": lane_pilot,
    "warm": lane_warm,
    "cold": lane_cold,
    "overhead": lane_overhead,
    "quality": lane_quality,
    "receipt": lane_receipt,
}


def main() -> int:
    p = argparse.ArgumentParser(description="M1 head-to-head campaign harness")
    p.add_argument("--lane", required=True, choices=sorted(LANES))
    args = p.parse_args()
    t0 = time.monotonic()
    rc = LANES[args.lane]()
    print(f"lane {args.lane} rc={rc} in {time.monotonic() - t0:.1f}s")
    return rc


if __name__ == "__main__":
    sys.exit(main())
