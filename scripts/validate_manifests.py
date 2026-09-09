"""Single source of truth for Pallama real-integration coverage manifests.

Everything validate.py enforces for 100% command + config coverage is
declared here, table-driven. Adding a CLI subcommand or a Config knob
WITHOUT extending this file makes the completeness gates fail loudly
(bidirectional set comparisons), so drift is impossible to miss.

Field inventories verified against crates/pallama-core/src/config.rs:
  Config          134 fields (13 Option, 4 containers: keys/remotes/engine_env/model_overrides)
  ModelOverride    20 fields (all Option)
  SamplerDefaults  18 fields (all Option, skip_serializing_if none)
  ApiKey            7 fields    Remote  3 fields
Command set verified against `pallama --help` (38 subcommands + help).

Tier semantics (honest evidence classes):
  argv       knob value reaches child llama-server argv (child_argv assert)
  behavior   observable daemon/gateway behavior (HTTP probe / environ / nice / phase)
  existing   already covered by a pre-existing validate.py phase (cov row exists or added)
  tune       exercised via `pallama tune` lane
  roundtrip  set -> config list echo + full-manifest daemon boot (serde deny_unknown_fields proof)
  boundary   honest boundary with documented reason (needs 2nd box / RAM / unsupported transport)
"""

# ---------------------------------------------------------------------------
# COMMANDS manifest: every CLI leaf path that must have >=1 real (no-mock)
# check. attrs: daemon=needs running daemon, model=needs loaded model,
# heavy=big disk/net/RAM lane, net=network required, fast=runs in FAST mode
# (fast=False -> honest boundary row in FAST runs).
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
    "import.hardlink": (False, False, False, False, True),
    "import.copy": (False, False, False, False, True),
    "mmproj.happy": (False, False, True, True, False),
    "mmproj.refusal": (False, False, False, False, True),
    "rm": (False, False, False, False, True),
    "list": (False, False, False, False, True),
    "ls": (False, False, False, False, True),
    "show": (False, False, False, False, True),
    "ps": (True, False, False, False, True),
    "ps.reset": (True, False, False, False, True),
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
# TOPLEVEL_KNOBS manifest: all 134 Config fields.
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
        "binds 127.0.0.1:11499 (every daemon phase)",
    ),
    (
        "port",
        False,
        False,
        "behavior",
        None,
        "binds 127.0.0.1:11499 (every daemon phase)",
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
    ("engine_pin", False, False, "existing", None, "engine mgmt: doctor + engine list"),
    ("spec", False, False, "existing", None, "phase_config B: --spec-type ngram"),
    ("cache_reuse", False, False, "argv", "G1", "--cache-reuse 128"),
    ("keys", False, True, "behavior", None, "keys lifecycle + phase_auth"),
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
# MODEL_OVERRIDE manifest: all 20 ModelOverride fields + 18 SamplerDefaults
# leaves. Evidence = overlay round-trip (gate d) + argv/wave lanes noted.
# ---------------------------------------------------------------------------

MODEL_OVERRIDE_FIELDS = [
    ("ctx", "argv: --ctx-size (phase_config C override-wins)"),
    ("slots", "argv: -np"),
    ("spec", "argv: --spec-type"),
    ("loras", "wave: lora attach"),
    ("extra_args", "argv: passthrough tokens"),
    ("cache_type", "argv: --cache-type-k/v"),
    ("kv_unified", "roundtrip echo"),
    ("ctx_extend", "argv: rope flags"),
    ("cpu_moe_n", "argv: --n-cpu-moe"),
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


if __name__ == "__main__":
    print(
        f"COMMANDS: {len(COMMANDS)} leaf paths, {len(TOPLEVEL_COMMANDS)} top-level (+help)"
    )
    print(
        f"TOPLEVEL_KNOBS: {len(TOPLEVEL_KNOBS)} (options={len(OPTION_KNOBS)}, containers={len(CONTAINER_KNOBS)}, fresh-visible={len(FRESH_VISIBLE_KNOBS)})"
    )
    print(
        f"MODEL_OVERRIDE_FIELDS: {len(MODEL_OVERRIDE_FIELDS)}  SAMPLER_FIELDS: {len(SAMPLER_FIELDS)}"
    )
    tiers = {}
    for k in TOPLEVEL_KNOBS:
        tiers[k["tier"]] = tiers.get(k["tier"], 0) + 1
    print(f"tiers: {dict(sorted(tiers.items()))}")
