//! Profile compiler: deterministic llama-server argv from (model GGUF meta,
//! hardware, config+overlay, engine flag set). Rules 1–12 from the plan,
//! applied in order. Every emitted flag is validated against the installed
//! engine's manifest; a needed-but-missing flag is a hard error naming it.
//!
//! Near-pure: no time, no randomness — table-testable. The one I/O
//! touch is read-only `fs::metadata` on the optional mmproj projector
//! (sizing must see the real file). The supervisor
//! picks the endpoint (free port / socket) and passes it in.

use std::collections::BTreeSet;

use crate::config::{Config, MmprojPolicy, ModelOverride, DEFAULT_CACHE_RAM_MB};
use crate::config::{MistralrsTuning, SglangTuning};
use crate::gguf::GgufMeta;
use crate::hardware::GpuInfo;
use crate::hardware::Hardware;
use crate::hfmeta::{HfMeta, ModelMeta};

/// Where the child will listen (supervisor picks port/sock first; the
/// argv must name it, so it is an input, not an output).
#[derive(Debug, Clone, PartialEq)]
pub enum Endpoint {
    Tcp { host: String, port: u16 },
    Unix { socket: String },
}

#[derive(Debug, Clone)]
pub struct ProfileInput<'a> {
    pub model_name: &'a str,
    /// Instance key for per-instance paths (sessions/, speccache/):
    /// equals `model_name` except for replica instances (`model#N`),
    /// where each replica owns its own cache files. Never affects
    /// `--alias` or overlay lookups (those stay model-level).
    pub instance_key: &'a str,
    /// First shard path (upstream mmap-rejoins the rest).
    pub model_path: &'a str,
    pub model_bytes: u64,
    /// Model metadata for the storage lane the row lives in: GGUF single
    /// files carry `GgufMeta` (llamacpp/mistralrs engines), safetensors
    /// dirs carry `HfMeta` (sglang engine). The compiler branches on the
    /// enum; the llamacpp grammar path requires the Gguf variant and
    /// errors with a teaching message otherwise.
    pub meta: ModelMeta<'a>,
    pub hardware: &'a Hardware,
    pub config: &'a Config,
    pub overlay: &'a ModelOverride,
    /// (path, scale) pairs from the loras table.
    pub loras: &'a [(String, f64)],
    /// Local path of the pulled draft model when spec=auto resolved one.
    pub draft_path: Option<&'a str>,
    /// Parsed GGUF header of that draft model, when available — lets the
    /// planner charge the draft's device-side KV alongside the dense
    /// model's. `None` = header unreadable (spawn still proceeds; the
    /// charge degrades to dense-only with a daemon-side warn).
    pub draft_gguf: Option<&'a GgufMeta>,
    /// Multimodal projector pulled alongside the model (vision/audio-in).
    /// Emitted as `-mm` when the engine supports it; the store's
    /// `mmproj_path` feeds this (rule 19).
    pub mmproj_path: Option<&'a str>,
    /// Caller-mandated projector attach that overrides the mmproj policy
    /// (Attach/Skip/Lazy). Set by `ensure_vision`'s `@vision` respawn so
    /// a Lazy-spawned text-only instance comes back WITH the projector,
    /// and by the router preset (router serves every model incl VL).
    /// `extra_args -mm` still wins over this (explicit user argv).
    pub mmproj_force: bool,
    pub engine_tag: &'a str,
    /// Capability manifest flag set of the ACTIVE engine.
    pub supported_flags: &'a BTreeSet<String>,
    /// `--spec-type` values the ACTIVE engine advertises (manifest help
    /// parsing). Gates `spec = "mtp"` (draft-mtp) with a fail-fast error
    /// naming the engine update path — an unadvertised value would die in
    /// the child with an opaque argv error.
    pub spec_types: &'a [String],
    /// Dialect selector: the engines.kind column owns this truth (the
    /// manifest carries no kind). `MistralRs` compiles a minimal profile —
    /// the child CLI grammar is foreign and `MistralRsEngine` argv
    /// translator owns it.
    pub engine_kind: crate::engine_kind::EngineKind,
    pub endpoint: Endpoint,
    /// Pallama data dir (base for speccache/ + sessions/ paths).
    pub data_dir: &'a str,
    /// Measured prefix-cache hit rate hint (0.0..=1.0) from the live
    /// daemon, when available. Drives the adaptive `--cache-ram` clamp:
    /// prefix-heavy traffic earns a bigger cache budget, cache-cold
    /// traffic releases RAM back. None = static 30% clamp.
    pub cache_hit_rate: Option<f64>,
    /// Sum of OTHER live instances' weights (MiB) at spawn time. The
    /// mlock auto-policy charges it: a dying engine's pinned pages are
    /// not released until its teardown completes, so overlapping
    /// replacements must not re-pin the same RAM share (churn live-
    /// repro'd: 2 × 5.4 GiB mlock on 13.6 GiB RAM → spawn failures).
    pub resident_ram_mib: u64,
    /// Auto-picked GPU id (e.g. "Vulkan1") from `--list-devices` free
    /// memory at spawn time. Only consulted when neither overlay nor
    /// config set `devices` — manual selection always wins. The same
    /// pick scopes `hardware` to that single GPU so every downstream
    /// VRAM estimate (ngl ladder, KV, cache-ram) sizes against the card
    /// the child will actually land on.
    pub device_hint: Option<&'a str>,
    /// FULL engine device census BEFORE single-card scoping: the build
    /// class of the engine binary (e.g. a vulkan build enumerates the
    /// integrated GPU alongside discrete cards) is a property of the
    /// CENSUS, not of the card the spawn lands on — `hardware.gpus` may
    /// be scoped to just the picked card, so build-class detection
    /// (vulkan-class slot capping in `auto_slots_capacity`) reads THIS.
    /// Capacity math never does: it stays on `hardware`.
    pub engine_census: Vec<GpuInfo>,
    /// Discrete GPUs NOT taken by the auto-pick (or by manual `devices`),
    /// best-free-first. Drives placement RECOMMENDATIONS only (spec-draft
    /// and mmproj offload hints) — never a silent placement decision.
    /// Empty on single-GPU boxes and when every sibling is integrated.
    pub sibling_devices: Vec<String>,
    /// Supervisor-planned `--tensor-split` ratio string (e.g. "2,1"),
    /// proportional to each discrete card's MEASURED free VRAM. Emitted
    /// only as a last resort: weights+KV exceed the best single card but
    /// fit the discrete cards combined, and every manual pin
    /// (`tensor_split` / `devices` / non-default `main_gpu`) is unset.
    /// `None` = no auto split planned.
    pub auto_tensor_split: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Profile {
    pub argv: Vec<String>,
    /// Non-fatal notes (skipped rules, unverifiable clamps). Surfaced via
    /// tracing and `pallama show`.
    pub warnings: Vec<String>,
    /// The ctx actually compiled in (post-clamp).
    pub ctx: u32,
    /// Resolved GPU-offload label for `ps`: "full" | "cpu" | "auto"
    /// ("auto" = tight fit left to the engine's layer juggling; "partial"
    /// = weights exceed VRAM, engine splits CPU+GPU).
    pub gpu: &'static str,
    /// Estimated f16-equivalent KV-cache bytes at the compiled ctx, after
    /// w KV quantization the profile chose (`q8_0` halves, `q4_0`
    /// quarters).
    /// Feeds the co-residency planner (A15); None when GGUF geometry is
    /// missing.
    pub kv_est_bytes: Option<u64>,
    /// `Some((per_slot_ctx, slots))` when auto-fit divided the default ctx
    /// to earn parallel slots (see `AUTOFIT_CTX_FLOOR`): the emitted
    /// `--ctx-size` total buys `-np slots` at `per_slot_ctx` each.
    /// `Profile.ctx` already reports `per_slot_ctx`; this field exists so
    /// telemetry can distinguish "ctx 4096 because user pinned" from
    /// "ctx 4096 because auto-fit traded depth for concurrency".
    pub ctx_autofit: Option<(u32, u32)>,
}

/// Tuning knobs the bench grid may override; None = use heuristic value.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TuningOverrides {
    pub ctx: Option<u32>,
    pub kv_quant: Option<bool>,
    pub threads: Option<u32>,
    /// Flash attention on/off (default heuristic: auto).
    pub fa: Option<bool>,
    /// Physical batch size (-b) for prompt processing.
    pub batch: Option<u32>,
    /// Physical ubatch size (-ub): micro-batch ceiling for prefill.
    /// Not a tune --search axis (argmax objective is tg; ubatch moves pp).
    pub ubatch: Option<u32>,
}

/// Compile the launch argv (plan rules D.1–12).
///
/// One function per the plan's ordered rule list: rules 1–12 append in
/// sequence and each needs the same locals; splitting further would
/// shuffle state through six one-use helpers.
#[allow(clippy::too_many_lines)]
pub fn compile(input: &ProfileInput<'_>, tuning: &TuningOverrides) -> Result<Profile, String> {
    // Dialect fork: mistral.rs has no llama-server grammar (no --jinja,
    // --ctx-size, -np slots...). Its argv is assembled by
    // MistralRsEngine::build_argv; the profile only resolves what that
    // translator consumes — ctx and slot count (mined back out of argv
    // as `-np N`). Skipping here avoids the manifest flag gate, which
    // would hard-error on every llama-server-only flag.
    if input.engine_kind == crate::engine_kind::EngineKind::MistralRs {
        return compile_mistralrs(input, tuning);
    }
    // Dialect fork: sglang speaks `launch_server` grammar, sizes KV from a
    // VRAM fraction (not ctx math), and reads safetensors dirs — none of
    // the llama-server rules below apply. SglangEngine::build_argv owns
    // the connection quintet (model-path/host/port/api-key/served-name);
    // this profile owns the numbers (ctx ladder, VRAM tiers, concurrency).
    if input.engine_kind == crate::engine_kind::EngineKind::Sglang {
        return compile_sglang(input, tuning);
    }
    // The llama-server grammar below is GGUF-only: every rule reads GGUF
    // tensor metadata. A safetensors row on a GGUF engine is a routing
    // mistake — teach instead of crashing deep in a rule.
    let gguf = match input.meta {
        ModelMeta::Gguf(g) => g,
        ModelMeta::Hf(h) => {
            return Err(format!(
                "model {} is a safetensors directory (arch {}) — llamacpp/mistralrs engines \
                 consume GGUF files; serve it on the sglang engine: `pallama engine install \
                 --kind sglang`, `pallama engine use <sglang-tag>` (see `pallama engine list`), \
                 then restart the daemon",
                input.model_name, h.architecture
            ));
        }
    };
    let mut argv: Vec<String> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();
    let overlay = input.overlay;
    let config = input.config;

    // Draft-row freshness (mirrors the 11y mcp-config existence check):
    // a stale store row — file deleted or moved after the pull — fails
    // HERE with a re-pull instruction instead of five seconds into a
    // child boot with an opaque engine error.
    if let Some(draft) = input.draft_path {
        if !std::path::Path::new(draft).is_file() {
            return Err(format!(
                "draft model file for {} is missing at {draft} — the store row is \
                 stale; pull the draft model again to refresh it",
                input.model_name
            ));
        }
    }

    // --- 1. model + endpoint + alias
    argv.push("-m".into());
    argv.push(input.model_path.to_string());
    match &input.endpoint {
        Endpoint::Tcp { host, port } => {
            argv.push("--host".into());
            argv.push(host.clone());
            argv.push("--port".into());
            argv.push(port.to_string());
        }
        Endpoint::Unix { socket } => {
            argv.push("--host".into());
            argv.push(socket.clone());
        }
    }
    argv.push("--alias".into());
    argv.push(input.model_name.to_string());

    // --- 2. templating/metrics/attention + ctx
    let base_ctx = tuning
        .ctx
        .unwrap_or_else(|| resolve_ctx(input, overlay, &mut warnings));
    // --- 2a. slots BEFORE the ctx push: auto slots scale the TOTAL ctx
    // (upstream divides --ctx-size across -np slots), so the emitted ctx
    // and every downstream capacity rule (offload resolver, KV ladder, KV
    // estimate) must see the scaled total. Each slot keeps `base_ctx` —
    // Profile.ctx reports the per-slot value so prompt preflight bounds a
    // single request correctly.
    let vram_bytes = capacity_bytes(input.hardware);
    // default_ctx is a CEILING auto-fit may divide; tuning (bench) and
    // overlay ctx are hard pins — never divided.
    let ctx_pinned = tuning.ctx.is_some() || overlay.ctx.is_some();
    // --- 2a-bis. effective --cache-ram budget, computed ONCE and BEFORE
    // the slot resolver: the pinned-ctx slot walk-down inside
    // resolve_slots must judge pool shapes against the EXACT budget the
    // 2b fit below sees — a second, drifting copy would walk to a
    // different np than 2b then hosts. Otherwise: the adaptive clamp
    // (A16) may pull the budget below the model's own weights, which
    // `--cache-ram` caps the model WEIGHTS mmap plus the prompt cache —
    // it does NOT move the KV pool off the GPU (live-verified b10948:
    // `llama_kv_cache: CUDA0 KV buffer` with `--kv-unified` on; the
    // upstream flag only shares ONE buffer across sequences, and
    // `--cache-ram` is the mmap cache cap per PR #16391). A sub-weights
    // budget is still nonsense — the upstream live fitter then
    // CPU-splits layers to honor it — live-measured on a 9B/8
    // GiB-VRAM/13.6 GiB box: --cache-ram 4102 < weights 5417 decoded at
    // 15.6 t/s while a budget covering the weights decoded at 39.9 t/s,
    // same argv otherwise. Floor the budget at weights + headroom when
    // unified is on and the box's RAM can actually host it (60% sanity
    // guard); otherwise keep the clamp and escalate.
    let cache_ram_budget: Option<u64> = if let Some(v) =
        cache_ram_from_extra_args(input.overlay.extra_args.as_deref().unwrap_or_default())
    {
        // Explicit `extra_args --cache-ram` wins over the config
        // knob (same doctrine as `-mm`): deliberate argv owns the
        // budget — no clamp, no floor, rule 12 emits nothing (the
        // flag lands exactly once, via extra_args itself).
        Some(v)
    } else if config.cache_ram_mb > 0 {
        let requested = u64::try_from(config.cache_ram_mb).unwrap_or(u64::MAX);
        let clamped = effective_cache_ram_mib(input).map_or(requested, |cap| requested.min(cap));
        if clamped < requested {
            // The shipped default meeting the adaptive cap is auto-tuning
            // doing its designed job — journal only. A user-pinned budget
            // being lowered is a real divergence and stays a warning.
            let user_pinned = config.cache_ram_mb != DEFAULT_CACHE_RAM_MB;
            let line = format!(
                "cache_ram_mb {} clamped to {} ({}% of {} MiB RAM, hit-rate {}); override via model_overrides extra_args --cache-ram or set 0 = unlimited",
                config.cache_ram_mb,
                clamped,
                cache_ram_pct(input.cache_hit_rate),
                input.hardware.total_ram_mib,
                input
                    .cache_hit_rate
                    .map_or_else(|| "n/a".to_string(), |h| format!("{h:.2}"))
            );
            if user_pinned {
                warnings.push(line);
            } else {
                tracing::info!(model = input.model_name, "profile: {line}");
            }
        }
        if kv_unified_emitted(input) && input.model_bytes > 0 {
            let weights_mib = input.model_bytes / (1024 * 1024);
            // Floor FROM the budget's true job: it must host the weights
            // mmap (x20/17 integer ÷0.85 headroom). The KV pool is
            // device-side and deliberately NOT part of this budget.
            let floor_mib = (weights_mib + 64) * 20 / 17;
            let ram_guard = input.hardware.total_ram_mib * 60 / 100;
            if clamped < floor_mib && floor_mib <= ram_guard {
                warnings.push(format!(
                    "cache-ram budget {clamped} MiB < weights {weights_mib} MiB (the budget \
                     caps the weights mmap) — floored to {floor_mib} MiB; a sub-weights budget \
                     makes the engine fitter CPU-split layers (measured 15.6 vs 39.9 t/s on \
                     a 9B)"
                ));
                Some(floor_mib)
            } else if clamped < floor_mib {
                warnings.push(format!(
                    "cache-ram budget {clamped} MiB cannot cover weights {weights_mib} MiB \
                     and the {floor_mib} MiB floor exceeds the 60% RAM guard ({ram_guard} MiB) \
                     — set kv_unified = false, quant down (pallama fit), or raise the budget; \
                     the fitter may CPU-split layers"
                ));
                Some(clamped)
            } else {
                Some(clamped)
            }
        } else {
            Some(clamped)
        }
    } else {
        None
    };
    let mut rs = resolve_slots(
        input,
        overlay,
        base_ctx,
        vram_bytes,
        ctx_pinned,
        &mut warnings,
    );
    // --- 2b. KV pool must fit the GPU. Live-verified child physics
    // (b10948, manual exec): the KV cache allocates DEVICE-side on
    // BOTH lanes — `--kv-unified` shares ONE buffer across sequences
    // (its real saving: no per-slot duplication) and `--cache-ram` is
    // the weights-mmap cap, not a KV relocation. Judge the unified
    // pool against VRAM exactly like the classic path; the only
    // unified-specific part is the shared-buffer geometry (KV scales
    // with TOTAL ctx across slots either way). Shrink ctx (never
    // below AUTOFIT_CTX_FLOOR) to fit; pinned ctx is never touched —
    // verdict/refuse instead. f16 bytes are an upper bound (quantized
    // KV only shrinks it) — conservative by design.
    if kv_unified_emitted(input) {
        if let Some(f16) = kv_f16_bytes(input, rs.total_ctx).filter(|kv| *kv > 0) {
            // A PIN the f16 pool cannot host may still be hostable at
            // the quant the spawn will actually run: an explicit
            // cache_type override, or the ladder demotion the same
            // tight card triggers anyway. The refuse teaching names
            // this exact lever — it must not be a dead end.
            let pinned_quant = if tuning.kv_quant == Some(true) {
                Some("q8_0")
            } else {
                let explicit = config.effective_cache_type(input.model_name);
                if explicit.is_empty() {
                    let mut scratch: Vec<String> = Vec::new();
                    kv_quant_ladder(input, vram_bytes, rs.total_ctx, &mut scratch)
                } else {
                    match explicit {
                        "f32" | "f16" | "bf16" => None,
                        t => Some(match t {
                            "q8_0" => "q8_0",
                            "q4_0" => "q4_0",
                            "q4_1" => "q4_1",
                            "q5_0" => "q5_0",
                            "q5_1" => "q5_1",
                            _ => "f16",
                        }),
                    }
                }
            };
            let kv = match pinned_quant {
                Some("q8_0") => f16 / 2,
                Some("q4_0") => f16 / 4,
                Some("q4_1") => f16 * 9 / 20,
                Some("q5_0") => f16 * 11 / 32,
                Some("q5_1") => f16 * 3 / 8,
                _ => f16,
            };
            let mib = |b: u64| b / (1024 * 1024);
            let demand = input.model_bytes.saturating_add(kv);
            let vram85 = vram_bytes / 100 * 85;
            if ctx_pinned {
                // Pinned ctx is sovereign — never shrunk. The SHARED
                // verdict (same fn the gateway preflights with) refuses
                // a pin the device cannot host; warn-yet-proceed here is
                // how a doomed pin 502-looped through spawn retries
                // while the child died at context creation every time.
                if let UnifiedCtxVerdict::Refuse(msg) =
                    unified_ctx_verdict(input.model_bytes, kv, vram_bytes, rs.total_ctx)
                {
                    return Err(msg);
                }
                if demand > vram85 {
                    // Middle zone (85%..100%): physically hostable, so
                    // serve it — but gpu-layers stays on the engine's
                    // live fitter, which may CPU-split the last layers.
                    // Honest degraded serve, not a refuse.
                    warnings.push(format!(
                        "pinned ctx {} fits the {} MiB VRAM only above the 85% share \
                         (demand {} MiB) — gpu-layers left to the engine fitter; it may \
                         CPU-split layers for the last stretch",
                        rs.total_ctx,
                        mib(vram_bytes),
                        mib(demand)
                    ));
                }
            } else if demand > vram_bytes {
                let per_ctx = kv / u64::from(rs.total_ctx);
                let fit =
                    (vram_bytes / 100 * 85).saturating_sub(input.model_bytes) / per_ctx.max(1);
                // 256-token multiple keeps upstream-friendly sizes.
                let new_ctx = u32::try_from(fit).unwrap_or(u32::MAX) & !255;
                if new_ctx >= AUTOFIT_CTX_FLOOR && new_ctx < rs.total_ctx {
                    let per_slot = (new_ctx / rs.slots).max(1);
                    warnings.push(format!(
                        "unified KV pool fit: ctx {} -> {} (f16 KV {} MiB + weights {} MiB \
                             vs {} MiB VRAM)",
                        rs.total_ctx,
                        new_ctx,
                        mib(kv),
                        mib(input.model_bytes),
                        mib(vram_bytes)
                    ));
                    // Same (per_slot, slots) pair resolve_slots uses, so
                    // the SlotsCtxAutoFit event stays truthful.
                    rs.autofit = Some((per_slot, rs.slots));
                    rs.total_ctx = new_ctx;
                    rs.per_slot_ctx = per_slot;
                } else {
                    warnings.push(format!(
                        "unified KV pool cannot fit the {} MiB VRAM even at the ctx floor \
                             {AUTOFIT_CTX_FLOOR}: spawn will likely fail — pull a smaller quant \
                             (pallama fit) or set cache_type = \"q8_0\"",
                        mib(vram_bytes)
                    ));
                }
            }
        }
    }
    let slots = rs.slots;
    let ctx = rs.total_ctx;

    let fa = match tuning.fa {
        Some(true) => "on",
        Some(false) => "off",
        None => "auto",
    };
    argv.extend([
        "--jinja".into(),
        "--metrics".into(),
        "--flash-attn".into(),
        fa.into(),
    ]);
    argv.push("--ctx-size".into());
    argv.push(ctx.to_string());

    // --- 3. threads
    let threads = tuning
        .threads
        .unwrap_or_else(|| input.hardware.physical_cores.max(1));
    argv.push("--threads".into());
    argv.push(threads.to_string());
    // Batch plumbing: bench-adopted tuning wins over the config knob,
    // which itself wins over the engine default (0 = unset).
    if let Some(b) = tuning.batch {
        argv.push("-b".into());
        argv.push(b.to_string());
    } else if config.batch_size > 0 {
        argv.push("--batch-size".into());
        argv.push(config.batch_size.to_string());
    }
    if let Some(ub) = tuning.ubatch {
        argv.push("--ubatch-size".into());
        argv.push(ub.to_string());
    } else if config.ubatch_size > 0 {
        argv.push("--ubatch-size".into());
        argv.push(config.ubatch_size.to_string());
    }
    if config.threads_batch > 0 {
        argv.push("--threads-batch".into());
        argv.push(config.threads_batch.to_string());
    }

    // --- 4. gpu layers: resolve "auto" ourselves when the answer is
    // unambiguous so `ps` can show the split and warn on CPU fallback
    // (ollama's #1 complaint: GPU present, model silently on CPU). A
    // comfortable full fit pins 999; no GPU pins 0 (labeled); the tight
    // middle keeps `auto` — the engine's fine-grained layer juggling beats
    // our estimate there, and guessing wrong OOMs loads.
    argv.push("--gpu-layers".into());
    let (gpu_layers, gpu_label) = resolve_gpu_offload(input, ctx, vram_bytes, &mut warnings);
    argv.push(gpu_layers.into());

    // --- 5. prefix-cache chunk reuse. Upstream disables cache_reuse
    // whenever a multimodal projector is attached ("cache_reuse is not
    // supported by multimodal") — emitting it on VL spawns is dead argv
    // that misleads profile readers, so skip and say why.
    if config.cache_reuse > 0 {
        if input.mmproj_path.is_some() {
            warnings.push(
                "cache_reuse skipped: upstream disables it with a multimodal projector attached (mmproj); prefix reuse is unavailable on VL spawns".into(),
            );
        } else {
            argv.push("--cache-reuse".into());
            argv.push(config.cache_reuse.to_string());
        }
    }

    // --- 6. KV cache quantization. Bench-adopted tuning wins, then an
    // explicit config/overlay type ("f16"-class = force off), then the
    // capacity ladder: none -> q8_0 (KV/2) -> q4_0 (KV/4).
    let kv_type: Option<String> = if let Some(on) = tuning.kv_quant {
        on.then(|| "q8_0".to_string())
    } else {
        let explicit = config.effective_cache_type(input.model_name);
        if explicit.is_empty() {
            kv_quant_ladder(input, vram_bytes, ctx, &mut warnings).map(str::to_string)
        } else {
            match explicit {
                "f32" | "f16" | "bf16" => None,
                t => Some(t.to_string()),
            }
        }
    };
    if let Some(t) = kv_type.clone() {
        argv.extend([
            "--cache-type-k".into(),
            t.clone(),
            "--cache-type-v".into(),
            t,
        ]);
    }

    // --- 7. cpu-moe when the model cannot fit VRAM but RAM can host it
    let ram_bytes = Hardware::bytes(input.hardware.total_ram_mib);
    if gguf.expert_count.unwrap_or(0) > 0
        && input.model_bytes > vram_bytes
        && ram_bytes >= input.model_bytes * 11 / 10
    {
        argv.push("--cpu-moe".into());
    }

    // --- 8. child-native sleep when a GPU is present
    if input.hardware.has_gpu() {
        argv.push("--sleep-idle-seconds".into());
        argv.push(config.idle_sleep_secs.to_string());
    }

    // --- 9. slots (resolved at rule 2a: `slots = 0` pallama-auto sizes
    // concurrency from capacity and scales the total ctx; explicit pins
    // pass through). The emitted value is always >= 1 — the auto path
    // subsumes the old upstream `-1` passthrough.
    argv.push("-np".into());
    argv.push(slots.to_string());

    // --- 10. rpc + loras
    // C6: per-model override replaces the global list (empty = inherit).
    let rpc_servers = config.effective_rpc_servers(input.model_name);
    if !rpc_servers.is_empty() {
        argv.push("--rpc".into());
        argv.push(rpc_servers.to_string());
    }
    for (path, scale) in input.loras {
        if (scale - 1.0).abs() < f64::EPSILON {
            argv.push("--lora".into());
            argv.push(path.clone());
        } else {
            argv.push("--lora-scaled".into());
            argv.push(format!("{path}:{scale}"));
        }
    }

    // --- 10b. explicit device selection (multi-GPU boxes where auto-pick
    // lands wrong). Per-model `devices` overlay replaces the global list.
    // Manifest-gated: an old engine without --device is a named error,
    // not a silent drop.
    let mut devices = overlay
        .devices
        .clone()
        .unwrap_or_else(|| config.effective_devices(input.model_name).to_vec());
    if devices.is_empty() {
        if let Some(hint) = input.device_hint {
            devices.push(hint.to_string());
        }
    }
    if !devices.is_empty() {
        if !input.supported_flags.contains("--device") {
            return Err(format!(
                "devices set but engine {} lacks --device; run: pallama engine update",
                input.engine_tag
            ));
        }
        for dev in &devices {
            argv.push("--device".into());
            argv.push(dev.clone());
        }
    }

    // --- 10d. multi-GPU split plumbing: preferred GPU, split strategy
    // and per-GPU tensor ratios. Manifest-gated warn-skip so an engine
    // update never turns a set knob into a hard failure.
    if config.main_gpu >= 0 {
        if input.supported_flags.contains("--main-gpu") {
            argv.push("--main-gpu".into());
            argv.push(config.main_gpu.to_string());
        } else {
            warnings.push(format!(
                "main_gpu set but engine {} lacks --main-gpu; run: pallama engine update",
                input.engine_tag
            ));
        }
    }
    if !config.split_mode.is_empty() {
        if input.supported_flags.contains("--split-mode") {
            argv.push("--split-mode".into());
            argv.push(config.split_mode.clone());
        } else {
            warnings.push(format!(
                "split_mode set but engine {} lacks --split-mode; run: pallama engine update",
                input.engine_tag
            ));
        }
    }
    if !config.tensor_split.is_empty() {
        if input.supported_flags.contains("--tensor-split") {
            argv.push("--tensor-split".into());
            argv.push(config.tensor_split.clone());
        } else {
            warnings.push(format!(
                "tensor_split set but engine {} lacks --tensor-split; run: pallama engine update",
                input.engine_tag
            ));
        }
    } else if let Some(ratios) = &input.auto_tensor_split {
        // Supervisor-planned last-resort split: weights+KV exceed the best
        // single card's MEASURED free VRAM but fit the discrete cards
        // combined. Manual `tensor_split` above always wins (pins
        // authoritative); this branch is the unset-pin auto path.
        if input.supported_flags.contains("--tensor-split") {
            argv.push("--tensor-split".into());
            argv.push(ratios.clone());
            warnings.push(format!(
                "auto tensor-split {ratios}: weights+KV exceed the best single card's free \
                 VRAM but fit the discrete cards combined — layer split buys capacity, \
                 not speed (inter-card bandwidth taxes every token); a smaller quant \
                 (pallama fit) or kv quantization may serve faster. Pin `tensor_split` \
                 to silence"
            ));
        } else {
            warnings.push(format!(
                "auto tensor-split planned but engine {} lacks --tensor-split; run: pallama engine update",
                input.engine_tag
            ));
        }
    }

    // --- 10c. embedding-class models (GGUF carries {arch}.pooling_type,
    // e.g. nomic-bert): enable the embeddings endpoint so /v1/embeddings
    // and /api/embeddings work without manual flags. Generative models
    // gain nothing from --embeddings and keep it off.
    // Late-chunking override (R1) wins over GGUF metadata: pooling MUST be
    // `none` so the gateway receives the per-token embedding matrix.
    if input.overlay.late_chunking == Some(true) {
        if input.supported_flags.contains("--embeddings")
            && input.supported_flags.contains("--pooling")
        {
            argv.push("--embeddings".into());
            argv.push("--pooling".into());
            argv.push("none".into());
            // Embedding tasks cannot split across micro-batches (upstream
            // `!slot.can_split()` rejects input > n_ubatch), so a late-
            // chunking doc must fit ONE ubatch. Default to the upstream
            // embedding-preset scale (embeddinggemma etc. use 2048) — a
            // full-ctx ubatch OOMs the compute buffer on tight GPUs —
            // while never shrinking an explicit config/tuning value. The
            // gateway token guard enforces the same number.
            let late_ubatch = if config.ubatch_size > 0 {
                config.ubatch_size
            } else {
                LATE_CHUNK_UBATCH_DEFAULT.min(ctx)
            };
            ensure_batch_flag(&mut argv, "--ubatch-size", late_ubatch);
            ensure_batch_flag(&mut argv, "--batch-size", late_ubatch);
            floor_batch_flag(&mut argv, "-b", late_ubatch);
        } else {
            warnings.push(format!(
                "late_chunking = true but engine {} lacks --embeddings/--pooling; run: pallama engine update",
                input.engine_tag
            ));
        }
    } else if let Some(pooling) = gguf.pooling_type {
        if input.supported_flags.contains("--embeddings") {
            argv.push("--embeddings".into());
            let mode = match pooling {
                1 => "mean",
                2 => "cls",
                _ => "last",
            };
            if input.supported_flags.contains("--pooling") {
                argv.push("--pooling".into());
                argv.push(mode.into());
            }
        } else {
            warnings.push(format!(
                "embedding model detected but engine {} lacks --embeddings; run: pallama engine update",
                input.engine_tag
            ));
        }
    }

    // --- 11. speculative decoding
    let spec_mode = overlay.spec.as_deref().unwrap_or(config.spec.as_str());
    if spec_mode == "auto" {
        // Embedded MTP head wins over a catalog draft pair: it drafts from
        // the target's own trained weights (no separate model to pull) —
        // the exact lane ollama auto-enables for MTP-bearing GGUFs. Still
        // opt-in: fires only because the user set spec = "auto"
        // (spec = "off" never reaches this branch).
        let embedded_mtp = if let Some(n_layers) = gguf.mtp_layers {
            if input.supported_flags.contains("--spec-type")
                && input.spec_types.iter().any(|t| t == "draft-mtp")
            {
                argv.push("--spec-type".into());
                argv.push("draft-mtp".into());
                // Draft steps are bounded by the trained head count — more
                // is pure verification overhead. Cap at 2 until benched.
                let n_max = n_layers.min(2);
                if input.supported_flags.contains("--spec-draft-n-max") {
                    argv.push("--spec-draft-n-max".into());
                    argv.push(n_max.to_string());
                }
                if input
                    .supported_flags
                    .contains("--spec-draft-backend-sampling")
                {
                    argv.push("--spec-draft-backend-sampling".into());
                }
                warnings.push(format!(
                    "spec auto: MTP head ({n_layers} layer(s)) baked into the GGUF — \
                     draft-mtp enabled (n-max {n_max}); set spec = \"off\" to disable, \
                     bench before adopting at scale"
                ));
                true
            } else {
                warnings.push(format!(
                    "spec auto: GGUF carries an MTP head but engine {} lacks \
                     draft-mtp — falling back to the catalog draft-pair path (if \
                     one exists); run: pallama engine update",
                    input.engine_tag
                ));
                false
            }
        } else {
            false
        };
        if !embedded_mtp {
            push_spec_args(input, &mut argv, &mut warnings);
        }
    } else if is_ngram_spec(spec_mode) {
        // Self-drafting n-gram speculation: no draft model to pull; drafts
        // from the context's own n-grams ("ngram" is pallama shorthand for
        // the upstream "ngram-simple"; the typed variants map verbatim).
        // Bench before adopting at scale — verification overhead can
        // regress non-repetitive workloads.
        argv.push("--spec-type".into());
        argv.push(if spec_mode == "ngram" {
            "ngram-simple".into()
        } else {
            spec_mode.into()
        });
    } else if spec_mode == "mtp" {
        // Multi-token-prediction head baked into the GGUF itself (no
        // separate draft model). The head must ship IN the weights — e.g.
        // the Unsloth Qwen3.5-9B GGUF carries none — so this stays strictly
        // opt-in, like every other spec mode (never default-on: ngram
        // precedent showed unmeasured speculation can net-lose).
        if !input.supported_flags.contains("--spec-type")
            || !input.spec_types.iter().any(|t| t == "draft-mtp")
        {
            return Err(format!(
                "spec = \"mtp\" needs an engine advertising draft-mtp; \
                 engine {} does not — run: pallama engine update",
                input.engine_tag
            ));
        }
        if gguf.mtp_layers.is_none() {
            warnings.push(
                "spec = \"mtp\" but this GGUF carries no MTP head (no \
                 n_predict_layers metadata) — the engine will likely fail to \
                 create an MTP context; pull an MTP-bearing build \
                 (e.g. unsloth ...-MTP-GGUF)"
                    .into(),
            );
        }
        argv.push("--spec-type".into());
        argv.push("draft-mtp".into());
        // Same cap as the auto lane: draft steps are bounded by the
        // trained head count, and the upstream n-max default (3) makes a
        // 1-2 layer head pay pure verification overhead.
        if let Some(n_layers) = gguf.mtp_layers {
            let n_max = n_layers.min(2);
            if input.supported_flags.contains("--spec-draft-n-max") {
                argv.push("--spec-draft-n-max".into());
                argv.push(n_max.to_string());
            }
        }
    } else if let Some(spec_val) = match spec_mode {
        "eagle3" => Some("draft-eagle3"),
        "dflash" => Some("draft-dflash"),
        "dspark" => Some("draft-dspark"),
        _ => None,
    } {
        // Trained / block-diffusion draft lanes: a SEPARATE draft model
        // (speculator GGUF or dflash/dspark sibling) paired with the
        // target — unlike MTP there is a real second file to resolve and
        // place (upstream asserts ctx_dft for all three). Force-modes for
        // users who benched them; auto keeps its own precedence.
        if !input.supported_flags.contains("--spec-type")
            || !input.spec_types.iter().any(|t| t == spec_val)
        {
            return Err(format!(
                "spec = {spec_mode:?} needs an engine advertising {spec_val}; \
                 engine {} does not — run: pallama engine update",
                input.engine_tag
            ));
        }
        match crate::catalog::spec_pair_for_typed(input.model_name, spec_val) {
            Some(pair) => {
                if let Some(draft) = input.draft_path {
                    argv.push("--spec-type".into());
                    argv.push(spec_val.into());
                    argv.push("--spec-draft-model".into());
                    argv.push(draft.to_string());
                } else {
                    // Same always-hard-error discipline as the auto pair:
                    // never boot dense when the user asked for a draft lane.
                    return Err(format!(
                        "spec={spec_mode} for {} but the draft model is not pulled; run: pallama pull {}",
                        input.model_name, pair.draft_repo
                    ));
                }
            }
            None => {
                return Err(format!(
                    "spec = {spec_mode:?} has no {spec_val} draft pair for {} in the catalog; \
                     running dense instead means un-training your intent — pick another \
                     spec mode (\"auto\" pairs automatically, \"mtp\" uses a baked-in head)",
                    input.model_name
                ));
            }
        }
    }

    // --- 11b. spec-draft placement (only when a draft model is actually
    // resolved: spec = "auto"/"eagle3"/"dflash"/"dspark" + pulled pair).
    // Pin the draft to P-cores / a spare GPU / fewer threads so it stops
    // stealing from the target.
    if matches!(spec_mode, "auto" | "eagle3" | "dflash" | "dspark") && input.draft_path.is_some() {
        if !config.spec_draft_device.is_empty() {
            push_gated(
                input,
                &mut argv,
                &mut warnings,
                "spec_draft_device",
                "--spec-draft-device",
                std::slice::from_ref(&config.spec_draft_device),
            );
        } else if let Some(spare) = input.sibling_devices.first() {
            // Teaching, never silent placement: offloading a draft is a
            // measurable win only on some models — name the spare card
            // and let the user opt in.
            warnings.push(format!(
                "spec draft shares the target's GPU; spare discrete card {spare} \
                 is idle — try spec_draft_device = \"{spare}\" (bench both ways)"
            ));
        }
        if config.spec_draft_cpu_strict {
            push_gated(
                input,
                &mut argv,
                &mut warnings,
                "spec_draft_cpu_strict",
                "--spec-draft-cpu-strict",
                &["1".to_string()],
            );
        }
        if !config.spec_draft_cpu_range.is_empty() {
            push_gated(
                input,
                &mut argv,
                &mut warnings,
                "spec_draft_cpu_range",
                "--spec-draft-cpu-range",
                std::slice::from_ref(&config.spec_draft_cpu_range),
            );
        }
        if !config.spec_draft_ngl.is_empty() {
            push_gated(
                input,
                &mut argv,
                &mut warnings,
                "spec_draft_ngl",
                "--spec-draft-ngl",
                std::slice::from_ref(&config.spec_draft_ngl),
            );
        }
        if config.spec_draft_threads > 0 {
            push_gated(
                input,
                &mut argv,
                &mut warnings,
                "spec_draft_threads",
                "--spec-draft-threads",
                &[config.spec_draft_threads.to_string()],
            );
        }
        if let Some(p) = config.spec_draft_p_min {
            push_gated(
                input,
                &mut argv,
                &mut warnings,
                "spec_draft_p_min",
                "--spec-draft-p-min",
                &[format_trimmed(p)],
            );
        }
        if let Some(p) = config.spec_draft_p_split {
            push_gated(
                input,
                &mut argv,
                &mut warnings,
                "spec_draft_p_split",
                "--spec-draft-p-split",
                &[format_trimmed(p)],
            );
        }
        if config.spec_draft_poll.is_some() || config.spec_draft_poll_batch.is_some() {
            // Poll level governs both phases unless batch is explicit.
            if let Some(p) = config.spec_draft_poll {
                push_gated(
                    input,
                    &mut argv,
                    &mut warnings,
                    "spec_draft_poll",
                    "--spec-draft-poll",
                    &[p.to_string()],
                );
            }
            if let Some(pb) = config.spec_draft_poll_batch {
                push_gated(
                    input,
                    &mut argv,
                    &mut warnings,
                    "spec_draft_poll_batch",
                    "--spec-draft-poll-batch",
                    &[if pb { "1" } else { "0" }.to_string()],
                );
            }
        }
        if config.spec_draft_prio != 0 {
            push_gated(
                input,
                &mut argv,
                &mut warnings,
                "spec_draft_prio",
                "--spec-draft-prio",
                &[config.spec_draft_prio.to_string()],
            );
        }
        if config.spec_draft_prio_batch != 0 {
            push_gated(
                input,
                &mut argv,
                &mut warnings,
                "spec_draft_prio_batch",
                "--spec-draft-prio-batch",
                &[config.spec_draft_prio_batch.to_string()],
            );
        }
        if config.spec_draft_cpu_strict_batch {
            push_gated(
                input,
                &mut argv,
                &mut warnings,
                "spec_draft_cpu_strict_batch",
                "--spec-draft-cpu-strict-batch",
                &["1".to_string()],
            );
        }
        if config.spec_draft_threads_batch > 0 {
            push_gated(
                input,
                &mut argv,
                &mut warnings,
                "spec_draft_threads_batch",
                "--spec-draft-threads-batch",
                &[config.spec_draft_threads_batch.to_string()],
            );
        }
        if !config.spec_draft_type_k.is_empty() {
            push_gated(
                input,
                &mut argv,
                &mut warnings,
                "spec_draft_type_k",
                "--spec-draft-type-k",
                std::slice::from_ref(&config.spec_draft_type_k),
            );
        }
        if !config.spec_draft_type_v.is_empty() {
            push_gated(
                input,
                &mut argv,
                &mut warnings,
                "spec_draft_type_v",
                "--spec-draft-type-v",
                std::slice::from_ref(&config.spec_draft_type_v),
            );
        }
        for ot in &config.spec_draft_override_tensor {
            push_gated(
                input,
                &mut argv,
                &mut warnings,
                "spec_draft_override_tensor",
                "--spec-draft-override-tensor",
                std::slice::from_ref(ot),
            );
        }
        if config.spec_draft_n_cpu_moe > 0 {
            push_gated(
                input,
                &mut argv,
                &mut warnings,
                "spec_draft_n_cpu_moe",
                "--spec-draft-n-cpu-moe",
                &[config.spec_draft_n_cpu_moe.to_string()],
            );
        }
        if config.spec_draft_cpu_moe {
            push_gated(
                input,
                &mut argv,
                &mut warnings,
                "spec_draft_cpu_moe",
                "--spec-draft-cpu-moe",
                &[],
            );
        }
        if !config.spec_draft_backend_sampling {
            push_gated(
                input,
                &mut argv,
                &mut warnings,
                "spec_draft_backend_sampling = false",
                "--no-spec-draft-backend-sampling",
                &[],
            );
        }
        if config.adaptive_decay > 0 {
            push_gated(
                input,
                &mut argv,
                &mut warnings,
                "adaptive_decay",
                "--adaptive-decay",
                &[config.adaptive_decay.to_string()],
            );
        }
        if config.adaptive_target > 0.0 {
            push_gated(
                input,
                &mut argv,
                &mut warnings,
                "adaptive_target",
                "--adaptive-target",
                &[format_trimmed(config.adaptive_target)],
            );
        }
    }

    // --- 11c. n-gram tuning. Upstream b10833 REMOVED the generic
    // --spec-ngram-* forms — the typed --spec-ngram-<family>-* flags
    // are the live surface. Warn-skip class: tuning is an enhancement, an
    // older engine still serves with engine defaults. The generic
    // size-m/size-n/min-hits knobs feed the simple, map-k and map-k4v
    // families (identical shape); ngram-mod has its own n-min/n-max/
    // n-match knobs; ngram-cache is parameterless.
    if is_ngram_spec(spec_mode) {
        let family: [(&str, u32); 3] = match spec_mode {
            "ngram" => [
                ("--spec-ngram-simple-size-m", config.ngram_size_m),
                ("--spec-ngram-simple-size-n", config.ngram_size_n),
                ("--spec-ngram-simple-min-hits", config.ngram_min_hits),
            ],
            "ngram-map-k" => [
                ("--spec-ngram-map-k-size-m", config.ngram_size_m),
                ("--spec-ngram-map-k-size-n", config.ngram_size_n),
                ("--spec-ngram-map-k-min-hits", config.ngram_min_hits),
            ],
            "ngram-map-k4v" => [
                ("--spec-ngram-map-k4v-size-m", config.ngram_size_m),
                ("--spec-ngram-map-k4v-size-n", config.ngram_size_n),
                ("--spec-ngram-map-k4v-min-hits", config.ngram_min_hits),
            ],
            "ngram-mod" => [
                ("--spec-ngram-mod-n-match", config.ngram_mod_n_match),
                ("--spec-ngram-mod-n-max", config.ngram_mod_n_max),
                ("--spec-ngram-mod-n-min", config.ngram_mod_n_min),
            ],
            _ => [("", 0), ("", 0), ("", 0)], // ngram-cache: parameterless
        };
        for (flag, value) in family {
            if flag.is_empty() || value == 0 {
                continue;
            }
            if input.supported_flags.contains(flag) {
                argv.push(flag.into());
                argv.push(value.to_string());
            } else {
                warnings.push(format!(
                    "ngram tuning skipped: engine {} lacks {flag} (engine update recommended)",
                    input.engine_tag
                ));
            }
        }
    }

    // --- 11z. on-demand tensor loading (`--lazy-mode`): big-tensor
    // mmap reads from disk instead of keeping them resident — the
    // big-MoE RAM-relief lever. Emitted ONLY on deviation: "auto" is
    // already the engine default (tensors > 4 GiB), so default config
    // stays argv-clean; a non-default mode on an engine without the
    // flag is a teaching warning (never a silent no-op of user intent).
    let lazy_mode = overlay
        .lazy_mode
        .as_deref()
        .unwrap_or(config.lazy_mode.as_str());
    if lazy_mode != "auto" {
        if input.supported_flags.contains("--lazy-mode") {
            argv.push("--lazy-mode".into());
            argv.push(lazy_mode.into());
        } else {
            warnings.push(format!(
                "lazy_mode = {lazy_mode:?} skipped: engine {} lacks --lazy-mode \
                 (engine update recommended)",
                input.engine_tag
            ));
        }
    }

    // --- 11y. experimental upstream agent-tooling passthrough
    // (`--tools` / `--tools-runtime` / `--mcp-servers-config|-json`).
    // Opt-in only — never defaulted. The engine limits CORS to localhost
    // when w of these is set; a flag-less engine degrades to a teaching
    // warning (user intent stays visible). The MCP config PATH is
    // existence-checked here: a typo should fail at profile-compile,
    // not after a 5 s boot (mirrors the draft-model discipline).
    for (flag, value) in [
        ("--tools", config.server_tools.as_deref()),
        ("--tools-runtime", config.server_tools_runtime.as_deref()),
        ("--mcp-servers-config", config.mcp_servers_config.as_deref()),
        ("--mcp-servers-json", config.mcp_servers_json.as_deref()),
    ] {
        let Some(value) = value else { continue };
        if flag == "--mcp-servers-config" && !std::path::Path::new(value).is_file() {
            return Err(format!(
                "mcp_servers_config = {value:?} does not exist — fix the path \
                 (engine would refuse the MCP definitions at boot)"
            ));
        }
        if input.supported_flags.contains(flag) {
            argv.push(flag.into());
            argv.push(value.into());
        } else {
            warnings.push(format!(
                "{flag} skipped: engine {} lacks the flag (engine update recommended)",
                input.engine_tag
            ));
        }
    }

    // --- 12. prompt-cache budget + vision projector
    if config.cache_ram_mb > 0
        // Explicit extra_args --cache-ram owns the flag (parsed in
        // 2a-bis as the budget model) — emitting here too would land
        // the flag twice with different values.
        && cache_ram_from_extra_args(input.overlay.extra_args.as_deref().unwrap_or_default()).is_none()
    {
        // 2a-bis computed the effective budget (adaptive clamp +
        // unified weights floor, or the 2b pinned raise) once, with
        // its warnings; emit it here.
        let budget = i64::try_from(
            cache_ram_budget
                .unwrap_or_else(|| u64::try_from(config.cache_ram_mb).unwrap_or(u64::MAX)),
        )
        .unwrap_or(i64::MAX);
        argv.push("--cache-ram".into());
        argv.push(budget.to_string());
    }

    // --- 12b. KV buffer layout (upstream unified cache: the cheap radix —
    // one shared buffer + K-shift lets a sequence reuse another's prefix
    // via --cache-reuse). Manifest-gated; unsupported knobs warn loudly
    // only when the user explicitly set them.
    let kvu = input.config.effective_kv_unified(input.model_name);
    if let Some(on) = kvu {
        let flag = if on {
            "--kv-unified"
        } else {
            "--no-kv-unified"
        };
        if input.supported_flags.contains(flag) {
            argv.push(flag.into());
        } else {
            warnings.push(format!(
                "kv_unified = {on} skipped: engine lacks {flag} (engine update recommended)"
            ));
        }
    } else if kv_unified_emitted(input) {
        // Default ON: the unified buffer is upstream's mainline KV path
        // (their own default when slots are auto); K-shift prefix reuse
        // (our --cache-reuse 256) rides it even at -np 1. Opt out with
        // kv_unified = false.
        argv.push("--kv-unified".into());
    }
    if input.config.kv_unified_per_slot > 0 {
        if input.supported_flags.contains("--kv-unified-per-slot") {
            argv.push("--kv-unified-per-slot".into());
            argv.push(input.config.kv_unified_per_slot.to_string());
        } else {
            warnings.push("kv_unified_per_slot skipped: engine lacks --kv-unified-per-slot".into());
        }
    }
    if input.config.swa_full {
        if input.supported_flags.contains("--swa-full") {
            argv.push("--swa-full".into());
        } else {
            warnings.push("swa_full skipped: engine lacks --swa-full".into());
        }
    }
    if input.config.ctx_checkpoints > 0 {
        if input.supported_flags.contains("--ctx-checkpoints") {
            argv.push("--ctx-checkpoints".into());
            argv.push(input.config.ctx_checkpoints.to_string());
        } else {
            warnings.push("ctx_checkpoints skipped: engine lacks --ctx-checkpoints".into());
        }
    }
    if input.config.no_kv_offload {
        if input.supported_flags.contains("--no-kv-offload") {
            argv.push("--no-kv-offload".into());
        } else {
            warnings.push("no_kv_offload skipped: engine lacks --no-kv-offload".into());
        }
    }
    if !input.config.load_mode.is_empty() {
        if input.supported_flags.contains("--load-mode") {
            argv.push("--load-mode".into());
            argv.push(input.config.load_mode.clone());
        } else {
            warnings.push(format!(
                "load_mode = {:?} skipped: engine lacks --load-mode",
                input.config.load_mode
            ));
        }
    } else if input.supported_flags.contains("--load-mode") {
        // Auto policy (elim sweep 2026-09-12, 9B q4, 3 reps, 4070
        // laptop): --load-mode mlock front-loads page-in during load and
        // measured ~0.7s faster to first token than lazy mmap faults.
        // Gate on a RAM share that leaves room for a second model;
        // explicit load_mode always wins. Low-RLIMIT_MEMLOCK boxes
        // degrade to a benign upstream warning + plain mmap.
        let weights_mib = input.model_bytes / (1024 * 1024);
        let ram_mib = input.hardware.total_ram_mib;
        let pinned_mib = weights_mib + input.resident_ram_mib;
        if ram_mib > 0 && pinned_mib * 100 <= ram_mib * 40 {
            argv.push("--load-mode".into());
            argv.push("mlock".into());
            // Auto policy rationale: journal only — the decision note is
            // not a user-facing warning.
            tracing::info!(
                model = input.model_name,
                "profile: load-mode mlock auto — weights {weights_mib} MiB (+ {} MiB resident) \
                 <= 40% of {ram_mib} MiB RAM; eager page-in measured ~0.7s faster to first \
                 token; set load_mode = \"mmap\" to opt out",
                input.resident_ram_mib
            );
        }
    }

    // --- 13. latency/affinity/reasoning/vision passthrough
    // (config-validated; every flag below is manifest-gated with a
    // teaching warn-skip — an older engine serves with its defaults
    // instead of failing to boot). Semantics verified against upstream
    // arg.cpp b10816: --cpu-range pins child threads to a "lo-hi" CPU
    // set (P/E hybrid boxes: pin to P-cores); --poll 1..100 busy-polls
    // waiting for work (CPU for TTFT); --reasoning-format selects
    // thought-tag extraction in responses.
    if !config.cpu_range.is_empty() {
        push_gated(
            input,
            &mut argv,
            &mut warnings,
            "cpu_range",
            "--cpu-range",
            std::slice::from_ref(&config.cpu_range),
        );
    }
    if config.poll > 0 {
        push_gated(
            input,
            &mut argv,
            &mut warnings,
            "poll",
            "--poll",
            &[config.poll.to_string()],
        );
    }
    if !config.reasoning_format.is_empty() {
        push_gated(
            input,
            &mut argv,
            &mut warnings,
            "reasoning_format",
            "--reasoning-format",
            std::slice::from_ref(&config.reasoning_format),
        );
    }

    // --- 13b. reasoning control (server-side thinking budget/effort).
    // The overlay parameter is authoritative when set (mirrors resolve_ctx
    // semantics); otherwise the config's effective value (which itself
    // honors `model_overrides` tables).
    {
        let budget = overlay
            .reasoning_budget
            .unwrap_or_else(|| config.effective_reasoning_budget(input.model_name));
        if budget != -1 {
            push_gated(
                input,
                &mut argv,
                &mut warnings,
                "reasoning_budget",
                "--reasoning-budget",
                &[budget.to_string()],
            );
        }
        if !config.reasoning_budget_message.is_empty() {
            push_gated(
                input,
                &mut argv,
                &mut warnings,
                "reasoning_budget_message",
                "--reasoning-budget-message",
                std::slice::from_ref(&config.reasoning_budget_message),
            );
        }
        let effort = overlay
            .reasoning_effort
            .as_deref()
            .filter(|e| !e.is_empty())
            .unwrap_or_else(|| config.effective_reasoning_effort(input.model_name));
        if !effort.is_empty() {
            push_gated(
                input,
                &mut argv,
                &mut warnings,
                "reasoning_effort",
                "--reasoning-effort",
                &[effort.to_string()],
            );
        }
        // Server-side reasoning switch — authoritative for templates
        // that ignore the `thinking`/`enable_thinking` request vars.
        let reasoning = overlay
            .reasoning
            .as_deref()
            .filter(|r| !r.is_empty())
            .unwrap_or_else(|| config.effective_reasoning(input.model_name));
        if !reasoning.is_empty() {
            push_gated(
                input,
                &mut argv,
                &mut warnings,
                "reasoning",
                "--reasoning",
                &[reasoning.to_string()],
            );
        }
        if let Some(preserve) = config.reasoning_preserve {
            if preserve {
                push_gated(
                    input,
                    &mut argv,
                    &mut warnings,
                    "reasoning_preserve",
                    "--reasoning-preserve",
                    &[],
                );
            } else {
                push_gated(
                    input,
                    &mut argv,
                    &mut warnings,
                    "reasoning_preserve",
                    "--no-reasoning-preserve",
                    &[],
                );
            }
        }
    }

    // --- 13c. scheduling extras (server-level).
    if config.cpu_strict {
        push_gated(
            input,
            &mut argv,
            &mut warnings,
            "cpu_strict",
            "--cpu-strict",
            &["1".to_string()],
        );
    }
    if config.prio != 0 {
        push_gated(
            input,
            &mut argv,
            &mut warnings,
            "prio",
            "--prio",
            &[config.prio.to_string()],
        );
    }
    if config.prio_batch != 0 {
        push_gated(
            input,
            &mut argv,
            &mut warnings,
            "prio_batch",
            "--prio-batch",
            &[config.prio_batch.to_string()],
        );
    }
    if let Some(pb) = config.poll_batch {
        push_gated(
            input,
            &mut argv,
            &mut warnings,
            "poll_batch",
            "--poll-batch",
            &[if pb { "1" } else { "0" }.to_string()],
        );
    }
    if config.threads_http > 0 {
        push_gated(
            input,
            &mut argv,
            &mut warnings,
            "threads_http",
            "--threads-http",
            &[config.threads_http.to_string()],
        );
    }

    // --- 12b-bis. child HTTP server behavior knobs (llama-server
    // lane; all flag-gated with the usual engine-update teaching).
    if let Some(secs) = config.sse_ping_interval {
        push_gated(
            input,
            &mut argv,
            &mut warnings,
            "sse_ping_interval",
            "--sse-ping-interval",
            &[secs.to_string()],
        );
    }
    if let Some(secs) = config.server_timeout_secs {
        push_gated(
            input,
            &mut argv,
            &mut warnings,
            "server_timeout_secs",
            "--timeout",
            &[secs.to_string()],
        );
    }
    if let Some(kwargs) = &config.chat_template_kwargs {
        push_gated(
            input,
            &mut argv,
            &mut warnings,
            "chat_template_kwargs",
            "--chat-template-kwargs",
            std::slice::from_ref(kwargs),
        );
    }
    match config.cont_batching {
        Some(true) => push_gated(
            input,
            &mut argv,
            &mut warnings,
            "cont_batching",
            "--cont-batching",
            &[],
        ),
        Some(false) => push_gated(
            input,
            &mut argv,
            &mut warnings,
            "cont_batching = false",
            "--no-cont-batching",
            &[],
        ),
        None => {}
    }
    if config.reuse_port {
        push_gated(
            input,
            &mut argv,
            &mut warnings,
            "reuse_port",
            "--reuse-port",
            &[],
        );
    }
    if config.lora_init_without_apply {
        push_gated(
            input,
            &mut argv,
            &mut warnings,
            "lora_init_without_apply",
            "--lora-init-without-apply",
            &[],
        );
    }

    // --- 12c. vision / multimodal tuning + embeddings normalization.
    if config.image_max_tokens > 0 {
        push_gated(
            input,
            &mut argv,
            &mut warnings,
            "image_max_tokens",
            "--image-max-tokens",
            &[config.image_max_tokens.to_string()],
        );
    }
    if config.image_min_tokens > 0 {
        push_gated(
            input,
            &mut argv,
            &mut warnings,
            "image_min_tokens",
            "--image-min-tokens",
            &[config.image_min_tokens.to_string()],
        );
    }
    if config.mtmd_batch_max_tokens > 0 {
        push_gated(
            input,
            &mut argv,
            &mut warnings,
            "mtmd_batch_max_tokens",
            "--mtmd-batch-max-tokens",
            &[config.mtmd_batch_max_tokens.to_string()],
        );
    }
    if !config.mmproj_offload {
        push_gated(
            input,
            &mut argv,
            &mut warnings,
            "mmproj_offload = false",
            "--no-mmproj-offload",
            &[],
        );
    }
    if !config.mmproj_auto {
        push_gated(
            input,
            &mut argv,
            &mut warnings,
            "mmproj_auto = false",
            "--no-mmproj-auto",
            &[],
        );
    }
    if !config.mmproj_device.is_empty() {
        argv.push("--mmproj-device".into());
        argv.push(config.mmproj_device.clone());
    } else if input.mmproj_path.is_some() {
        if let Some(spare) = input.sibling_devices.first() {
            // Vision prefill bursts steal compute cycles from the target
            // card; an idle discrete sibling absorbs them for free. Opt-in
            // (teaching warning) because projector placement is workload-
            // dependent.
            warnings.push(format!(
                "mmproj shares the target's GPU; spare discrete card {spare} \
                 is idle — try mmproj_device = \"{spare}\""
            ));
        }
    }
    if config.embd_normalize > 0 {
        argv.push("--embd-normalize".into());
        argv.push(config.embd_normalize.to_string());
    }
    // --- 19. multimodal projector: the pulled mmproj wires vision/
    // audio-in into the engine. Explicit `extra_args` -mm wins; a pulled
    // projector on an engine without the flag warns (model still loads,
    // vision off) — never silently, never fatally. Policy resolution
    // (`attach` | `skip` | `lazy`, default `lazy`): `skip` spawns
    // TEXT-ONLY with vision failing loudly per request; `lazy` also
    // spawns text-only but the gateway respawns WITH the projector on
    // the first vision request (KV bank carries the conversation) —
    // everyone gets the measured cold win (875 MiB projector ≈ +3.9 s
    // cold TTFT + 1126 MiB VRAM on the 9B VL row) without losing
    // vision. The supervisor reads the same policy via
    // `mmproj_policy_effective` to know a spawn is projector-less.
    let explicit_mm = overlay
        .extra_args
        .as_deref()
        .and_then(find_mmproj_arg)
        .map(str::to_string)
        .or_else(|| {
            config
                .overlay_for(input.model_name)
                .extra_args
                .as_deref()
                .and_then(find_mmproj_arg)
                .map(str::to_string)
        });
    let mm_policy = mmproj_policy_effective(config, input.model_name, overlay);
    if let Some(mmproj) = explicit_mm {
        argv.push("-mm".into());
        argv.push(mmproj);
    } else if input.mmproj_force {
        // caller-mandated attach (@vision respawn, router preset): policy
        // machinery stays out of the way — the caller already decided
        if let Some(mmproj) = input.mmproj_path {
            argv.push("-mm".into());
            argv.push(mmproj.to_string());
        }
    } else if mm_policy == crate::config::MmprojPolicy::Skip {
        if let Some(mmproj) = input.mmproj_path {
            warnings.push(format!(
                "mmproj suppressed: model_overrides.{}.mmproj = skip — \
                 text-only spawn, projector {} skipped (cold boot saves the \
                 projector read; multimodal requests will fail loudly)",
                input.model_name, mmproj
            ));
        }
    } else if mm_policy == crate::config::MmprojPolicy::Lazy {
        if let Some(mmproj) = input.mmproj_path {
            warnings.push(format!(
                "mmproj lazy: {} spawns text-only (projector {} held back — \
                 faster cold start, less VRAM); the first vision request \
                 triggers a projector respawn that carries the conversation \
                 via the KV bank",
                input.model_name, mmproj
            ));
        }
    } else if let Some(mmproj) = input.mmproj_path {
        if input.supported_flags.contains("--mmproj") || input.supported_flags.contains("-mm") {
            argv.push("-mm".into());
            argv.push(mmproj.to_string());
        } else {
            // F108: warnings is the teaching channel (ps/show/tracing) —
            // eprintln! bypassed it and vanished under daemons.
            warnings.push(format!(
                "mmproj skipped: model {} has a multimodal projector but \
                 engine {} lacks -mm/--mmproj; vision disabled (engine update)",
                input.model_name, input.engine_tag
            ));
        }
    }

    // --- 14. persistent n-gram speculative cache: the lookup table
    // survives restarts, so speculation is warm from the first request
    // after a respawn (default-on convenience: warn-skip on old engines).
    // Every n-gram family benefits — ngram-cache literally consumes the
    // lookup-cache paths upstream (common.h ngram_cache struct).
    if is_ngram_spec(spec_mode) && config.spec_cache {
        if input.supported_flags.contains("--lookup-cache-dynamic") {
            argv.push("--lookup-cache-dynamic".into());
            argv.push(format!(
                "{}/speccache/{}.lcache",
                input.data_dir,
                path_safe(input.instance_key)
            ));
        } else {
            warnings.push(
                "spec_cache skipped: engine lacks --lookup-cache-dynamic (engine update recommended)".into(),
            );
        }
    }

    // --- 15. session checkpoints (`pallama session save/restore`).
    // Default-on convenience: warn-skip on engines without the flag.
    // Per-model subdirectory: upstream appends the bare filename, so the
    // model qualifier must live in the directory, not the name.
    if config.sessions {
        if input.supported_flags.contains("--slot-save-path") {
            argv.push("--slot-save-path".into());
            argv.push(format!(
                "{}/sessions/{}/",
                input.data_dir,
                path_safe(input.instance_key)
            ));
        } else {
            warnings.push(
                "sessions skipped: engine lacks --slot-save-path (engine update recommended)"
                    .into(),
            );
        }
    }

    // --- 16. YaRN context extension (explicitly configured; emit-then-gate
    // so an ancient engine names the missing flag instead of silently
    // running at base ctx).
    let ctx_extend = config.effective_ctx_extend(input.model_name);
    if ctx_extend > 1.0 {
        argv.extend([
            "--rope-scaling".into(),
            "yarn".into(),
            "--rope-scale".into(),
            format_trimmed(ctx_extend),
        ]);
        warnings.push(format!(
            "ctx_extend {ctx_extend}: YaRN stretches beyond the trained window; long-context quality may degrade"
        ));
    }

    // --- 16b. YaRN fine-tuning (each knob independent; 0/empty = default).
    if config.yarn_orig_ctx > 0 {
        argv.push("--yarn-orig-ctx".into());
        argv.push(config.yarn_orig_ctx.to_string());
    }
    if !config.yarn_ext_factor.is_sign_negative() {
        argv.push("--yarn-ext-factor".into());
        argv.push(format_trimmed(config.yarn_ext_factor));
    }
    if config.yarn_attn_factor > 0.0 {
        argv.push("--yarn-attn-factor".into());
        argv.push(format_trimmed(config.yarn_attn_factor));
    }
    if config.yarn_beta_fast > 0.0 {
        argv.push("--yarn-beta-fast".into());
        argv.push(format_trimmed(config.yarn_beta_fast));
    }
    if config.yarn_beta_slow > 0.0 {
        argv.push("--yarn-beta-slow".into());
        argv.push(format_trimmed(config.yarn_beta_slow));
    }

    // --- 17. fine-grained MoE expert offload (count beats the boolean
    // rule-7 heuristic when the user knows their split).
    let cpu_moe_n = config.effective_cpu_moe_n(input.model_name);
    if cpu_moe_n > 0 {
        argv.push("--n-cpu-moe".into());
        argv.push(cpu_moe_n.to_string());
    }
    let cpu_ffn_n = config.effective_cpu_ffn_n(input.model_name);
    if cpu_ffn_n > 0 {
        argv.push("--n-cpu-ffn".into());
        argv.push(cpu_ffn_n.to_string());
    }

    // --- 18. per-tensor device overrides (expert patterns to CPU etc.)
    for ot in config.effective_override_tensor(input.model_name) {
        argv.push("--override-tensor".into());
        argv.push(ot.clone());
    }

    // --- 19. agent mode: built-in tools + MCP CORS proxy. Opt-in only —
    // tools include exec_shell_command; the warning keeps it visible.
    if config.agent {
        argv.push("--agent".into());
        warnings.push(
            "agent mode ON: child exposes built-in tools (incl. exec_shell_command) and the MCP CORS proxy".into(),
        );
    }

    // --- 20. slot prompt affinity (`-sps`): how closely a request's prompt
    // must match a slot's cached prompt to reuse it. Upstream default is
    // 0.1 (enabled); 0 here emits nothing = upstream default.
    if config.slot_prompt_similarity > 0.0 {
        argv.push("--slot-prompt-similarity".into());
        argv.push(format!("{}", config.slot_prompt_similarity));
    }

    // --- 21. engine behavior toggles + power-user escapes. Defaults mirror
    // upstream so unset knobs emit nothing; the knobs exist to disable or
    // to reach surfaces config otherwise can't name.
    if !overlay
        .warmup
        .unwrap_or_else(|| config.effective_warmup(input.model_name))
    {
        argv.push("--no-warmup".into());
    }
    if !config.repack {
        argv.push("--no-repack".into());
    }
    if !config.cache_idle_slots {
        argv.push("--no-cache-idle-slots".into());
    }
    if let Some(path) = &config.lookup_cache_static {
        argv.push("--lookup-cache-static".into());
        argv.push(path.clone());
    }
    if let Some(path) = &config.lookup_cache_dynamic {
        argv.push("--lookup-cache-dynamic".into());
        argv.push(path.clone());
    }
    if config.no_host {
        argv.push("--no-host".into());
    }
    if let Some(op) = config.op_offload {
        argv.push(if op {
            "--op-offload".into()
        } else {
            "--no-op-offload".into()
        });
    }
    if config.keep_tokens != 0 {
        argv.push("--keep".into());
        argv.push(config.keep_tokens.to_string());
    }
    for kv in &config.override_kv {
        argv.push("--override-kv".into());
        argv.push(kv.clone());
    }
    for cv in &config.control_vectors {
        argv.push("--control-vector".into());
        argv.push(cv.clone());
    }
    for cv in &config.control_vectors_scaled {
        argv.push("--control-vector-scaled".into());
        argv.push(cv.clone());
    }
    if !config.control_vector_layer_range.is_empty() {
        argv.push("--control-vector-layer-range".into());
        argv.push(config.control_vector_layer_range.clone());
    }

    // --- 21b. remaining engine escapes (numa, tensor validation,
    // context-shift opt-out, default sampler chain, video-in lane).
    if !config.numa.is_empty() {
        argv.push("--numa".into());
        argv.push(config.numa.clone());
    }
    if config.check_tensors {
        argv.push("--check-tensors".into());
    }
    if config.context_shift {
        argv.push("--context-shift".into());
    }
    if !config.samplers.is_empty() {
        argv.push("--samplers".into());
        argv.push(config.samplers.clone());
    }
    if !config.video_ffmpeg_dir.is_empty() {
        argv.push("--video-ffmpeg-dir".into());
        argv.push(config.video_ffmpeg_dir.clone());
    }
    if config.video_fps > 0.0 {
        argv.push("--video-fps".into());
        argv.push(format_trimmed(config.video_fps));
    }
    if config.video_timestamp_interval > 0.0 {
        argv.push("--video-timestamp-interval".into());
        argv.push(format_trimmed(config.video_timestamp_interval));
    }

    // --- 21c. per-model chat-template override and sampling defaults.
    // Both are warn-skip when this engine predates the flag; `extra_args`
    // remains the full escape hatch and is appended after (wins upstream).
    if let Some(tpl) = overlay.chat_template.as_deref().filter(|t| !t.is_empty()) {
        if input.supported_flags.contains("--chat-template") {
            argv.push("--chat-template".into());
            argv.push(tpl.to_string());
        } else {
            warnings.push("chat_template skipped: engine lacks --chat-template".into());
        }
    }
    if let Some(f) = overlay
        .chat_template_file
        .as_deref()
        .filter(|f| !f.is_empty())
    {
        if input.supported_flags.contains("--chat-template-file") {
            argv.push("--chat-template-file".into());
            argv.push(f.to_string());
        } else {
            warnings.push("chat_template_file skipped: engine lacks --chat-template-file".into());
        }
    }
    // Infill token-order toggle: Suffix/Prefix/Middle for coder models
    // that were trained on that order (CodeGemma family). Warn-skip on
    // engines without the flag; `extra_args` remains the escape hatch.
    if overlay.spm_infill == Some(true) {
        if input.supported_flags.contains("--spm-infill") {
            argv.push("--spm-infill".into());
        } else {
            warnings.push("spm_infill skipped: engine lacks --spm-infill".into());
        }
    }
    if let Some(sd) = &overlay.sampler_defaults {
        // (flag, value) pairs in upstream argv order; None fields emit
        // nothing so per-request body params keep upstream defaults.
        let pairs: Vec<(&str, String)> = [
            sd.temperature.map(|v| ("--temp", format_trimmed(v))),
            sd.top_k.map(|v| ("--top-k", v.to_string())),
            sd.top_p.map(|v| ("--top-p", format_trimmed(v))),
            sd.min_p.map(|v| ("--min-p", format_trimmed(v))),
            sd.top_n_sigma.map(|v| ("--top-n-sigma", format_trimmed(v))),
            sd.typical_p.map(|v| ("--typical-p", format_trimmed(v))),
            sd.repeat_penalty
                .map(|v| ("--repeat-penalty", format_trimmed(v))),
            sd.repeat_last_n.map(|v| ("--repeat-last-n", v.to_string())),
            sd.presence_penalty
                .map(|v| ("--presence-penalty", format_trimmed(v))),
            sd.frequency_penalty
                .map(|v| ("--frequency-penalty", format_trimmed(v))),
            sd.dry_multiplier
                .map(|v| ("--dry-multiplier", format_trimmed(v))),
            sd.dry_base.map(|v| ("--dry-base", format_trimmed(v))),
            sd.dry_allowed_length
                .map(|v| ("--dry-allowed-length", v.to_string())),
            sd.dry_penalty_last_n
                .map(|v| ("--dry-penalty-last-n", v.to_string())),
            sd.xtc_probability
                .map(|v| ("--xtc-probability", format_trimmed(v))),
            sd.xtc_threshold
                .map(|v| ("--xtc-threshold", format_trimmed(v))),
            sd.mirostat.map(|v| ("--mirostat", v.to_string())),
            sd.seed.map(|v| ("--seed", v.to_string())),
        ]
        .into_iter()
        .flatten()
        .collect();
        for (flag, value) in pairs {
            if input.supported_flags.contains(flag) {
                argv.push(flag.into());
                argv.push(value);
            } else {
                warnings.push(format!(
                    "sampler_defaults {flag} skipped: engine lacks {flag}"
                ));
            }
        }
    }

    // --- overlay extra args (validated like everything else)
    if let Some(extra) = &overlay.extra_args {
        argv.extend(extra.iter().cloned());
    }

    // --- manifest gate: every emitted long flag must exist on this engine
    let mut missing: Vec<String> = Vec::new();
    for a in &argv {
        if a.starts_with("--") {
            let name = a.split('=').next().unwrap_or(a);
            if !input.supported_flags.contains(name) && !missing.iter().any(|m| m == name) {
                missing.push(name.to_string());
            }
        }
    }
    if !missing.is_empty() {
        return Err(format!(
            "engine {} does not support: {}; try `pallama engine update` or `pallama engine use <tag>`",
            input.engine_tag,
            missing.join(", ")
        ));
    }

    // KV estimate for the co-residency planner. Unified layout: the buffer
    // lives in the --cache-ram (system RAM) budget, so VRAM only sees the
    // measured working-set floor — charging full f16 here made the planner
    // read unified spawns as 2-4x their real VRAM demand and downgrade or
    // evict on phantom pressure (KV quantization shrinks the RAM buffer,
    // not this floor). Classic path: f16 bytes scaled by the quantization
    // grade actually emitted above. Hybrid-linear models with a provable
    // recurrent/full split already get the true (fractional) KV from
    // `kv_f16_bytes`; warn exactly when the split is NOT provable, so an
    // inflated estimate is never a surprise (R8).
    if gguf.attention_class() == crate::gguf::AttentionClass::HybridLinear
        && !gguf.recurrent_split_provable()
    {
        warnings.push(format!(
            "arch {} is hybrid-linear but the GGUF lacks recurrent_layers/full_attention_interval \
             metadata — KV estimate counts all layers (upper bound); a newer GGUF conversion may \
             emit the layer map",
            gguf.architecture
        ));
    }
    // Draft KV: the spec pair allocates its own device-side KV at the
    // same compiled ctx; charge it (f16, conservative) so the
    // co-residency planner stops under-counting spec pairs. Device
    // truth: dense and draft KV both live on the card on EITHER lane
    // (`--kv-unified` shares buffers, it does not relocate them).
    let draft_kv = input
        .draft_gguf
        .and_then(|g| kv_f16_bytes_meta(g, rs.total_ctx))
        .unwrap_or(0);
    let kv_est_bytes = kv_f16_bytes(input, ctx).map(|f16| {
        let dense = match kv_type.as_deref() {
            Some("q8_0") => f16 / 2,
            Some("q4_0") => f16 / 4,
            Some("q4_1") => f16 * 9 / 20,
            Some("q5_0") => f16 * 11 / 32,
            Some("q5_1") => f16 * 3 / 8,
            _ => f16,
        };
        dense + draft_kv
    });

    Ok(Profile {
        argv,
        warnings,
        // per-slot ctx: with auto slots the argv --ctx-size is the scaled
        // TOTAL, but a single request lives in ONE slot — preflight must
        // bound prompts by what a slot can hold, not the total
        ctx: rs.per_slot_ctx,
        gpu: gpu_label,
        kv_est_bytes,
        // Some((per_slot_ctx, slots)) when auto-fit divided the default
        // ctx to earn parallel slots; telemetry (`ps`, events) uses this
        // to surface the trade instead of silently shrinking ctx.
        ctx_autofit: rs.autofit,
    })
}

/// Late-chunking ubatch default: upstream embedding-preset scale
/// (embeddinggemma sets `n_batch` = `n_ubatch` = 2048). One embedding doc
/// must fit a single micro-batch; a full-ctx ubatch OOMs tight GPUs.
pub const LATE_CHUNK_UBATCH_DEFAULT: u32 = 2048;

/// VRAM the unified KV cache still touches on TOP of its f16 pool when
/// `--kv-unified` is on: the resident slice + paging working set
/// (device-backed pool — see `estimate_kv_vram_charge`). Measured as
/// part of the ~548 MiB non-weights overhead of a fully-offloaded 16k
/// spawn (CUDA context + compute buffers included) on an 8 GiB card.
/// Public: the supervisor's bytes admission charges it as the standing
/// working-set slice (see `admission_floor_bytes`).
pub const KV_UNIFIED_VRAM_FLOOR_BYTES: u64 = 512 * 1024 * 1024;

/// Fixed spawn overhead (CUDA/Vulkan context, graphs, compute buffers)
/// charged as the standing overhead when the planner floor-charges
/// instead of using a percentage margin (bytes admission, the unified
/// ctx verdict's refuse threshold, spec-draft attach gate). Measured
/// ~548 MiB total non-weights overhead on a fully-offloaded b10809
/// spawn; 700 MiB adds margin for bigger compute buffers (long ctx,
/// mmproj bursts).
const UNIFIED_SPAWN_OVERHEAD_BYTES: u64 = 700 * 1024 * 1024;

/// Measured-floor admission charge for a candidate spawn: weights +
/// projector + the unified KV working-set floor + the fixed spawn
/// overhead — the same standing charges the unified gpu-layers pin
/// trusts, exposed so the supervisor's bytes admission charges incoming
/// models identically (one decision source). Weights-only admission
/// oversubscribed an 8 GiB card on 2026-09-11: a 9B VL model settled at
/// 7302 MiB measured (5417 weights + 875 mmproj) and the 0.5B sibling
/// (~977 MiB actual, 437 weights) joined it into an `NVRM NO_MEMORY`
/// storm that SIGABRT-looped every reload at the clip loader.
#[must_use]
pub fn admission_floor_bytes(model_bytes: u64, mmproj_bytes: u64) -> u64 {
    model_bytes
        .saturating_add(mmproj_bytes)
        .saturating_add(KV_UNIFIED_VRAM_FLOOR_BYTES)
        .saturating_add(UNIFIED_SPAWN_OVERHEAD_BYTES)
}

/// VRAM that ctx-scaled compute buffers cost per token of TOTAL ctx on
/// vulkan-class builds (mixed integrated+discrete census) when a
/// projector is attached — the one combination whose allocation footprint
/// pallama cannot see ahead of spawn. Measured on an 8 GiB card
/// (Qwen3.5-9B `Q4_K_M` + mmproj, `--kv-unified`): total ctx 16384 boots
/// clean (load 3.5s, 88 t/s system), 32768 boots degraded (load 27s,
/// 47 t/s), 65536 OOMs or crawls at 96% commitment — while the same
/// shapes without the projector, and CUDA builds with it, stay healthy.
/// Bounds the true per-token cost to (21, 43) KiB; 24 KiB sits just
/// above the np2 failure edge. See `auto_slots_capacity`.
const UNIFIED_COMPUTE_PER_TOKEN_BYTES: u64 = 24 * 1024;

/// Raise an existing `<flag> <value>` argv pair to at least `floor`
/// (late-chunking correctness: one doc must fit one ubatch). No-op when
/// the flag is absent or already large enough.
fn floor_batch_flag(argv: &mut [String], flag: &str, floor: u32) {
    if let Some(pos) = argv.iter().position(|a| a == flag) {
        if let Some(v) = argv.get(pos + 1).and_then(|s| s.parse::<u32>().ok()) {
            if v < floor {
                argv[pos + 1] = floor.to_string();
            }
        }
    }
}

/// `floor_batch_flag`, but appends `<flag> <floor>` when the flag is
/// absent — embedding docs need the knob present at all.
fn ensure_batch_flag(argv: &mut Vec<String>, flag: &str, floor: u32) {
    if argv.iter().any(|a| a == flag) {
        floor_batch_flag(argv, flag, floor);
    } else {
        argv.push(flag.to_string());
        argv.push(floor.to_string());
    }
}

/// Mistral.rs tuning flags that take a value. Single source of truth:
/// `compile_mistralrs` emits them and the runtime argv translator
/// forwards them — the lockstep unit test diffs a full-knob compile
/// against these lists so neither side can drift alone.
pub const MISTRALRS_TUNING_VALUE_FLAGS: &[&str] = &[
    "--max-batch-size",
    "--max-prefill-chunk-tokens",
    "--max-decode-steps-before-prefill",
    "--prefix-cache-n",
    "--pa-block-size",
    "--pa-cache-type",
    "--pa-context-len",
    "--lora",
    "--lora-max-rank",
    "--lora-max-adapters",
    "--lora-max-bytes",
    "--mtp-model",
    "--mtp-n-predict",
    "--mtp-draft-sampling",
    "--encoder-cache-memory-mb",
    "--max-num-images",
    "--max-image-length",
    "--device-layers",
];

/// Valueless (`store_true`) mistral.rs tuning flags. Same lockstep
/// contract as [`MISTRALRS_TUNING_VALUE_FLAGS`].
pub const MISTRALRS_TUNING_BOOL_FLAGS: &[&str] = &[
    "--mtp",
    "--disable-metrics",
    "--disable-access-log",
    "--enable-lora",
];

/// Effective `ctx`: overlay > config default, clamped to the model's
/// `context_length` when known. Missing metadata warns, never guesses.
/// Minimal mistral.rs profile: ctx via the same overlay/config/train-cap
/// clamp as llama-server (so `pallama ps` and KV planning share one
/// truth), slots as `-np N` for the argv translator to mine. KV estimate
/// is None: mistral.rs sizes its paged KV from a VRAM fraction, not from
/// ctx math, so an f16 estimate here would be a lie.
#[allow(clippy::too_many_lines)] // one flat flag inventory, not logic
#[allow(clippy::cast_precision_loss)] // MiB-scale integers: 52-bit f64 mantissa is exact here
/// f16-element KV bytes for a full context: `layers × kv_heads ×
/// head_dim × 2 (K+V) × ctx × elem`. Shared by the sglang fit ladder
/// and the mistral.rs pa-fraction derive so both engines size KV from
/// ONE formula. `None` = checkpoint lacks complete KV geometry.
fn hf_kv_bytes(kv: &crate::hfmeta::KvGeom, ctx: u64, elem: u64) -> Option<u64> {
    Some(
        kv.layers?
            .saturating_mul(kv.kv_heads?)
            .saturating_mul(kv.head_dim?)
            .saturating_mul(2)
            .saturating_mul(ctx)
            .saturating_mul(elem),
    )
}

/// Derive `--pa-memory-fraction` from what the spawn actually NEEDS,
/// not from what the card happens to have. Two candidates, min wins:
///
/// * geometry — weights + ctx-sized f16 KV + runtime floor, from the
///   checkpoint's own KV geometry. Upstream's 0.90-of-TOTAL default
///   made a 0.5B spawn hold ~7 GiB of an 8 GiB card (live-proven:
///   6984 MiB resident for a 942 MiB model) and starve every
///   co-resident lane pallama routes beside it — sizing KV to serving
///   intent (ctx × slots geometry) is the fix, mirroring the sglang
///   ladder. Absent geometry (incomplete config.json) degrades
///   silently to the co-tenant guard below.
/// * free-VRAM budget — when a GPU co-tenant shrinks the pool
///   upstream's fraction would silently truncate into empty responses
///   (live-proven: instant ~60ms empties on long prompts, real replies
///   on short ones — the worst failure mode, serving while broken).
///
/// No-op when the operator pinned `mistralrs_pa_memory_fraction`,
/// the engine lacks the flag, or (no geometry AND single-tenant —
/// nothing to correct). `forced_on`: `mistralrs_paged_attn = true` was
/// pinned — never emit a conflicting `--paged-attn off`, warn instead.
#[allow(clippy::cast_precision_loss)] // MiB-scale integers: 52-bit f64 mantissa is exact here
fn derive_pa_fraction(
    input: &ProfileInput<'_>,
    argv: &mut Vec<String>,
    warnings: &mut Vec<String>,
    forced_on: bool,
) {
    const COTENANT_SLACK_MIB: u64 = 256;
    const HEADROOM_MIB: i64 = 512;
    const MIN_KV_MIB: i64 = 256;
    const MISTRALRS_RUNTIME_FLOOR_MIB: i64 = 512;
    const MIB: u64 = 1024 * 1024;
    if input.config.mistralrs_pa_memory_fraction.is_some() {
        return;
    }
    if !input.supported_flags.contains("--pa-memory-fraction") {
        return;
    }
    let total = input.hardware.total_vram_mib();
    if total == 0 {
        return;
    }
    let free = input.hardware.free_vram_mib();
    let mmproj = input
        .mmproj_path
        .and_then(|p| std::fs::metadata(p).ok())
        .map_or(0, |m| m.len());
    // Probe values are MiB from real hardware/files — always inside i64.
    let resident_mib: i64 =
        i64::try_from(input.model_bytes.saturating_add(mmproj) / MIB).expect("MiB fits i64");

    // Candidate 1: geometry — only for safetensors dirs (HfMeta). The
    // GGUF-on-mistralrs lane keeps the co-tenant guard only; GGUF KV
    // sizing lives in the llamacpp arch tables.
    let ctx = u64::from(resolve_ctx(input, input.overlay, &mut Vec::new()));
    let geometry_mib: Option<i64> = match input.meta {
        crate::hfmeta::ModelMeta::Hf(hf) => {
            hf_kv_bytes(&hf.kv, ctx, 2).map(|b| i64::try_from(b / MIB).expect("MiB fits i64"))
        }
        crate::hfmeta::ModelMeta::Gguf(_) => None,
    };
    let geometry_frac = geometry_mib
        .map(|kv| (resident_mib + kv + MISTRALRS_RUNTIME_FLOOR_MIB) as f64 / total as f64);

    // Candidate 2: free-VRAM budget under a co-tenant.
    let cotenant = free > 0 && free + COTENANT_SLACK_MIB < total;
    let budget_mib: Option<i64> = cotenant
        .then_some(i64::try_from(free).expect("MiB fits i64") - resident_mib - HEADROOM_MIB);

    let mut emit = |frac: f64, warn: String| {
        let frac = frac.clamp(0.05, 0.90);
        argv.push("--pa-memory-fraction".into());
        argv.push(format!("{frac:.2}"));
        warnings.push(warn);
    };

    match (geometry_frac, budget_mib) {
        // Both apply: the need must fit the free pool — min wins.
        (Some(g), Some(b)) if b >= MIN_KV_MIB => {
            emit(
                g.min(b as f64 / total as f64),
                format!(
                    "pa-memory-fraction {g:.2}→min(geometry, free) applied: a GPU co-tenant holds \
                 part of the card (total {total} MiB, free {free} MiB, weights+projector \
                 {resident_mib} MiB); pin mistralrs_pa_memory_fraction to override"
                ),
            );
        }
        // Geometry alone (single tenant): the idle-GPU grab fix — KV
        // sized to ctx×geometry, the rest of the card stays free for
        // co-resident lanes.
        (Some(g), None) => {
            emit(g, format!(
                "pa-memory-fraction {g:.2} sized from model geometry (weights+projector \
                 {resident_mib} MiB + f16 KV {} MiB at ctx {ctx} + floor {MISTRALRS_RUNTIME_FLOOR_MIB} MiB, \
                 of {total} MiB total) — upstream's 0.90-of-total default would grab the \
                 whole card and starve co-resident lanes; pin mistralrs_pa_memory_fraction \
                 to override",
                geometry_mib.unwrap_or(0)
            ));
        }
        // Co-tenant guard only (no geometry): the historical behavior.
        (None, Some(b)) if b >= MIN_KV_MIB => {
            let frac = b as f64 / total as f64;
            emit(
                frac,
                format!(
                    "pa-memory-fraction {frac:.2} derived from free VRAM: a GPU co-tenant holds \
                 most of the card (total {total} MiB, free {free} MiB, weights+projector \
                 {resident_mib} MiB) — upstream's 0.90-of-total default would silently truncate \
                 the KV pool into empty responses; pin mistralrs_pa_memory_fraction to override"
                ),
            );
        }
        // No viable paged pool next to the tenant (geometry can't
        // rescue it: the need includes weights, which already exceed
        // free) — classic ctx-sized KV either serves or fails loudly.
        (_, Some(_)) if !forced_on => {
            argv.push("--paged-attn".into());
            argv.push("off".into());
            warnings.push(format!(
                "paged-attn off: the GPU co-tenant leaves no paged-KV room (total {total} MiB, \
                 free {free} MiB, weights+projector {resident_mib} MiB) — classic ctx-sized KV \
                 either serves or fails loudly at load; free the GPU or reroute the model"
            ));
        }
        (_, Some(_)) => {
            warnings.push(format!(
                "GPU co-tenant leaves ~{} MiB for paged KV (total {total} MiB, free {free} MiB, \
                 weights+projector {resident_mib} MiB) — expect a loud load failure; set \
                 mistralrs_paged_attn = false for classic KV or free the GPU",
                budget_mib.map_or(0, |b| b.max(0))
            ));
        }
        // No geometry, single tenant: nothing to correct.
        (None, None) => {}
    }
}

#[allow(clippy::too_many_lines)] // one flat flag inventory, not logic
fn compile_mistralrs(
    input: &ProfileInput<'_>,
    tuning: &TuningOverrides,
) -> Result<Profile, String> {
    let ctx = tuning
        .ctx
        .unwrap_or_else(|| resolve_ctx(input, input.overlay, &mut Vec::new()));
    let mut slots = input.overlay.slots.unwrap_or(input.config.slots);
    let mut argv: Vec<String> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();
    if input
        .overlay
        .deterministic
        .unwrap_or(input.config.deterministic)
    {
        if slots > 1 {
            warnings.push(format!(
                "deterministic = true pins slots = 1 — explicit slots = {slots} ignored: \
                 multi-slot batches perturb logits in near-tie positions and break \
                 token-for-token greedy reproducibility; drop one of the two knobs"
            ));
        } else if slots == 0 {
            warnings.push(
                "deterministic = true: slots pinned to 1 — concurrent streams queue \
                 instead of batching; unset the pin to recover multi-slot throughput"
                    .to_string(),
            );
        }
        slots = 1;
    }
    if slots != 0 {
        argv.push("-np".into());
        argv.push(slots.to_string());
    }
    // Paged-KV budget: upstream's 0.90 default hard-fails at load
    // ("Num GPU blocks is 0", live-proven with a 9B vision model on an
    // 8 GB card) — an explicit fraction reclaims the balance for KV.
    if let Some(frac) = input.config.mistralrs_pa_memory_fraction {
        if input.supported_flags.contains("--pa-memory-fraction") {
            argv.push("--pa-memory-fraction".into());
            argv.push(format!("{frac}"));
        } else {
            warnings.push(
                "config mistralrs_pa_memory_fraction set but this engine lacks \
                 --pa-memory-fraction; flag skipped (pallama engine install updates it)"
                    .into(),
            );
        }
    }
    // Attention backend: classic KV (sized by ctx) instead of paged KV
    // (sized by VRAM fraction) — the only fit for big vision models on
    // 8 GB cards, live-proven with qwen3.5-9b + projector.
    //
    // Co-tenancy: upstream's --pa-memory-fraction is a fraction of
    // TOTAL VRAM, and its 0.90 default silently truncates the KV pool
    // when another GPU tenant (a desktop ollama, a second model server)
    // already holds most of the card — the engine then serves
    // instant-empty responses instead of failing (live-proven: 0.5B
    // BF16 on a 8 GiB card with 6.3 GiB resident elsewhere = ~60ms
    // empty replies on long tool prompts, real replies on short ones).
    // When a co-tenant is detected we therefore DERIVE a fraction from
    // what is actually free, mirroring the sglang fit ladder:
    // derive → classic-KV fallback → refuse with numbers.
    match input.config.mistralrs_paged_attn {
        Some(pa) => {
            if input.supported_flags.contains("--paged-attn") {
                argv.push("--paged-attn".into());
                argv.push(if pa { "on".into() } else { "off".into() });
                if pa {
                    derive_pa_fraction(input, &mut argv, &mut warnings, true);
                }
            } else {
                warnings.push(
                    "config mistralrs_paged_attn set but this engine lacks \
                     --paged-attn; flag skipped (pallama engine install updates it)"
                        .into(),
                );
            }
        }
        None => {
            // Auto fallback: upstream budgets paged KV as a fraction of
            // TOTAL VRAM; once weights(+projector) consume most of the
            // card the block pool computes to zero and the load hard-fails
            // ("Num GPU blocks is 0", live-proven with a 9B vision model
            // on an 8 GiB card). Classic ctx-sized KV serves there.
            if input.supported_flags.contains("--paged-attn") {
                let vram_bytes = capacity_bytes(input.hardware);
                let mmproj = input
                    .mmproj_path
                    .and_then(|p| std::fs::metadata(p).ok())
                    .map_or(0, |m| m.len());
                let resident = input.model_bytes.saturating_add(mmproj);
                if vram_bytes > 0 && resident > vram_bytes * 75 / 100 {
                    argv.push("--paged-attn".into());
                    argv.push("off".into());
                    warnings.push(
                        "paged-attn auto-off: weights+projector occupy >75% of VRAM and \
                         upstream's paged-KV block pool computes to zero (load would fail with \
                         'Num GPU blocks is 0'); classic ctx-sized KV serves instead — set \
                         mistralrs_paged_attn = true to override"
                            .into(),
                    );
                } else {
                    // PA stays on (weights fit): under a co-tenant the
                    // upstream 0.90-of-total fraction still points at
                    // absent memory — derive from free instead.
                    derive_pa_fraction(input, &mut argv, &mut warnings, false);
                }
            }
        }
    }
    // mistral.rs tuning struct (global `[mistralrs]` or per-model
    // `model_overrides.<name>.mistralrs`, override-replaces). All
    // emissions flag-gated with the usual engine-update teaching.
    let tun = MistralrsTuning::effective(input.overlay.mistralrs.as_ref(), &input.config.mistralrs);

    // LoRA adapter lane: mistral.rs `--lora ALIAS=SOURCE` accepts
    // repeated pairs (alias --lora-modules), GGUF and safetensors
    // alike — unlike the sglang lane there is no format split. The
    // llamacpp `scale` field has no ALIAS=SOURCE form (the upstream
    // JSON form is undocumented in --help); warn instead of guessing.
    for (path, scale) in input.loras {
        let alias = std::path::Path::new(path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("lora");
        push_gated(
            input,
            &mut argv,
            &mut warnings,
            "loras",
            "--lora",
            &[format!("{alias}={path}")],
        );
        if (*scale - 1.0).abs() > f64::EPSILON {
            warnings.push(format!(
                "lora scale {scale} ignored on the mistralrs engine — its ALIAS=SOURCE \
                 form has no scale; the adapter serves at its trained strength"
            ));
        }
    }

    // Scheduler family.
    if let Some(n) = tun.max_batch_size {
        push_gated(
            input,
            &mut argv,
            &mut warnings,
            "mistralrs.max_batch_size",
            "--max-batch-size",
            &[n.to_string()],
        );
    }
    if let Some(n) = tun.max_prefill_chunk_tokens {
        push_gated(
            input,
            &mut argv,
            &mut warnings,
            "mistralrs.max_prefill_chunk_tokens",
            "--max-prefill-chunk-tokens",
            &[n.to_string()],
        );
    }
    if let Some(n) = tun.max_decode_steps_before_prefill {
        push_gated(
            input,
            &mut argv,
            &mut warnings,
            "mistralrs.max_decode_steps_before_prefill",
            "--max-decode-steps-before-prefill",
            &[n.to_string()],
        );
    }
    if let Some(n) = tun.prefix_cache_n {
        push_gated(
            input,
            &mut argv,
            &mut warnings,
            "mistralrs.prefix_cache_n",
            "--prefix-cache-n",
            &[n.to_string()],
        );
    }

    // Paged-attention detail family (the two legacy globals above cover
    // the fraction and the on/off switch).
    if let Some(n) = tun.pa_block_size {
        push_gated(
            input,
            &mut argv,
            &mut warnings,
            "mistralrs.pa_block_size",
            "--pa-block-size",
            &[n.to_string()],
        );
    }
    if let Some(t) = &tun.pa_cache_type {
        push_gated(
            input,
            &mut argv,
            &mut warnings,
            "mistralrs.pa_cache_type",
            "--pa-cache-type",
            std::slice::from_ref(t),
        );
    }
    if let Some(n) = tun.pa_context_len {
        push_gated(
            input,
            &mut argv,
            &mut warnings,
            "mistralrs.pa_context_len",
            "--pa-context-len",
            &[n.to_string()],
        );
    }

    // LoRA family: --enable-lora (dynamic serving switch) leads; the
    // runtime limits only take effect with it or a preloaded adapter —
    // upstream rejects bare limits with a hard error at boot.
    let lora_serving = tun.enable_lora == Some(true) || !input.loras.is_empty();
    if !lora_serving
        && (tun.lora_max_rank.is_some()
            || tun.lora_max_adapters.is_some()
            || tun.lora_max_bytes.is_some())
    {
        warnings.push(
            "mistralrs.lora_max_rank/lora_max_adapters/lora_max_bytes set but no LoRA serving \
             (mistralrs.enable_lora absent and no adapters preloaded); upstream rejects bare \
             LoRA runtime limits — set mistralrs.enable_lora = true or attach adapters"
                .to_string(),
        );
    }
    if tun.enable_lora == Some(true) {
        push_gated(
            input,
            &mut argv,
            &mut warnings,
            "mistralrs.enable_lora",
            "--enable-lora",
            &[],
        );
    }
    if lora_serving {
        if let Some(n) = tun.lora_max_rank {
            push_gated(
                input,
                &mut argv,
                &mut warnings,
                "mistralrs.lora_max_rank",
                "--lora-max-rank",
                &[n.to_string()],
            );
        }
        if let Some(n) = tun.lora_max_adapters {
            push_gated(
                input,
                &mut argv,
                &mut warnings,
                "mistralrs.lora_max_adapters",
                "--lora-max-adapters",
                &[n.to_string()],
            );
        }
        if let Some(n) = tun.lora_max_bytes {
            push_gated(
                input,
                &mut argv,
                &mut warnings,
                "mistralrs.lora_max_bytes",
                "--lora-max-bytes",
                &[n.to_string()],
            );
        }
    }

    // MTP speculative family (--mtp first so the head is enabled before
    // its parameters land).
    if tun.mtp == Some(true) {
        push_gated(
            input,
            &mut argv,
            &mut warnings,
            "mistralrs.mtp",
            "--mtp",
            &[],
        );
    }
    if let Some(m) = &tun.mtp_model {
        push_gated(
            input,
            &mut argv,
            &mut warnings,
            "mistralrs.mtp_model",
            "--mtp-model",
            std::slice::from_ref(m),
        );
    }
    if let Some(n) = tun.mtp_n_predict {
        push_gated(
            input,
            &mut argv,
            &mut warnings,
            "mistralrs.mtp_n_predict",
            "--mtp-n-predict",
            &[n.to_string()],
        );
    }
    if let Some(s) = &tun.mtp_draft_sampling {
        push_gated(
            input,
            &mut argv,
            &mut warnings,
            "mistralrs.mtp_draft_sampling",
            "--mtp-draft-sampling",
            std::slice::from_ref(s),
        );
    }

    // Vision capacity family.
    if let Some(mb) = tun.encoder_cache_memory_mb {
        push_gated(
            input,
            &mut argv,
            &mut warnings,
            "mistralrs.encoder_cache_memory_mb",
            "--encoder-cache-memory-mb",
            &[mb.to_string()],
        );
    }
    if let Some(n) = tun.max_num_images {
        push_gated(
            input,
            &mut argv,
            &mut warnings,
            "mistralrs.max_num_images",
            "--max-num-images",
            &[n.to_string()],
        );
    }
    if let Some(n) = tun.max_image_length {
        push_gated(
            input,
            &mut argv,
            &mut warnings,
            "mistralrs.max_image_length",
            "--max-image-length",
            &[n.to_string()],
        );
    }

    // Observability switches (child's Prometheus recorder and HTTP
    // access log are on by default upstream).
    if tun.disable_metrics == Some(true) {
        push_gated(
            input,
            &mut argv,
            &mut warnings,
            "mistralrs.disable_metrics",
            "--disable-metrics",
            &[],
        );
    }
    if tun.disable_access_log == Some(true) {
        push_gated(
            input,
            &mut argv,
            &mut warnings,
            "mistralrs.disable_access_log",
            "--disable-access-log",
            &[],
        );
    }

    // Layer mapping pin: emit only with a teaching nudge on single-GPU
    // boxes (mirrors the sglang parallelism pins).
    if let Some(layers) = &tun.device_layers {
        if input.hardware.gpus.len() < 2 {
            warnings.push(
                "mistralrs.device_layers set but this box has a single GPU — a manual \
                 layer split only makes sense multi-GPU; drop the pin for automatic \
                 mapping"
                    .to_string(),
            );
        }
        push_gated(
            input,
            &mut argv,
            &mut warnings,
            "mistralrs.device_layers",
            "--device-layers",
            std::slice::from_ref(layers),
        );
    }

    // extra_args: strict manifest-gated passthrough — the same contract
    // as the sglang lane. The probed manifest IS the dialect's grammar:
    // unknown flags hard-fail with it (a silent warn-drop once measured
    // nothing for a whole ISQ A/B run), and pallama-owned flags refuse
    // rather than double-set host/port/model/scheduling pins.
    if let Some(extra) = &input.overlay.extra_args {
        const RESERVED: &[&str] = &[
            "--host",
            "--port",
            "-m",
            "-f",
            "--mmproj",
            "--max-model-len",
            "-np",
            "--max-seqs",
            "--max-num-batched-tokens",
            "--paged-attn",
            "--pa-memory-fraction",
            "--no-ui",
        ];
        for tok in extra {
            if !tok.starts_with('-') {
                continue; // value token riding its preceding flag
            }
            if RESERVED.contains(&tok.as_str()) {
                return Err(format!(
                    "extra_args {tok} is reserved — pallama owns it on the mistralrs \
                     engine (host/port/model/scheduling pins); remove it from the \
                     override"
                ));
            }
            if !input.supported_flags.contains(tok.as_str()) {
                return Err(format!(
                    "extra_args {tok} is not in this mistralrs engine's probed manifest \
                     (pallama engine list) — drop it, or pallama engine update refreshes \
                     the probe"
                ));
            }
        }
        argv.extend(extra.iter().cloned());
    }
    Ok(Profile {
        argv,
        warnings,
        ctx,
        gpu: "auto",
        kv_est_bytes: None,
        ctx_autofit: None,
    })
}

/// sglang runtime overhead the server needs beyond weights+KV: CUDA
/// graphs, activation workspace, NCCL/comm buffers, allocator
/// fragmentation. Deliberately coarse — the ladder only needs it to stop
/// mem-fraction from promising the last gibibyte to tensors (live OOM
/// class upstream warns about in their own memory guide).
const SGLANG_RUNTIME_FLOOR_BYTES: u64 = 1024 * 1024 * 1024;

/// Share of measured free VRAM the ladder is allowed to promise to
/// weights+KV (torch reserves the rest for activations/workspace).
const SGLANG_VRAM_USABLE_PCT: u64 = 92;

/// Hard clamp band for emitted `--mem-fraction-static` — below 0.20 the
/// server starves weights; above 0.90 it OOMs at graph capture (upstream
/// default 0.88 lives at the top of the band for a reason).
const SGLANG_MEM_FRACTION_MIN: f32 = 0.20;
const SGLANG_MEM_FRACTION_MAX: f32 = 0.90;

/// Flags `SglangEngine::build_argv` owns. An `extra_args` entry naming
/// one of these is a hand-written attempt to fight the connection
/// quintet, the VRAM ladder, or the loras lane — hard error with the
/// reason, never a silent duplicate flag (last-one-wins argv semantics
/// would let a user accidentally unpublish the API key or move the
/// child off loopback).
const SGLANG_RESERVED_FLAGS: &[&str] = &[
    "--model-path",
    "--host",
    "--port",
    "--api-key",
    "--served-model-name",
    "--context-length",
    "--mem-fraction-static",
    "--cpu-offload-gb",
    "--enable-lora",
    "--lora-paths",
];

/// Flag-gated argv push for tuning knobs: knobs may target a newer sglang
/// than the installed engine, so an unadvertised flag warns-and-skips
/// (engine updates revive it) instead of erroring the whole spawn.
fn push_tuned(
    argv: &mut Vec<String>,
    flags: &std::collections::BTreeSet<String>,
    warning_prefix: &str,
    flag: &str,
    value: &str,
    warnings: &mut Vec<String>,
) {
    if flags.contains(flag) {
        argv.push(flag.to_string());
        // Empty value = boolean flag (store_true class): the flag alone,
        // never a phantom "" positional (argparse would reject it).
        if !value.is_empty() {
            argv.push(value.to_string());
        }
    } else {
        warnings.push(format!(
            "{warning_prefix} set but this sglang engine lacks {flag}; \
             skipped (pallama engine install updates it)"
        ));
    }
}

/// List-shaped sibling of `push_tuned` for nargs flags (`--cuda-graph-bs
/// 1 2 4`): the flag once, then every value as its own argv token —
/// argparse consumes tokens until the next `--` flag.
fn push_tuned_list(
    argv: &mut Vec<String>,
    flags: &std::collections::BTreeSet<String>,
    warning_prefix: &str,
    flag: &str,
    values: &[String],
    warnings: &mut Vec<String>,
) {
    if flags.contains(flag) {
        argv.push(flag.to_string());
        argv.extend(values.iter().cloned());
    } else {
        warnings.push(format!(
            "{warning_prefix} set but this sglang engine lacks {flag}; \
             skipped (pallama engine install updates it)"
        ));
    }
}

/// sglang profile: `launch_server` grammar, VRAM-fraction KV sizing, and
/// a fit ladder the mistralrs path never needed because that engine
/// juggles layers itself. The connection quintet (model-path/host/port/
/// api-key/served-model-name) is prepended by `SglangEngine::build_argv`
/// — this argv is pure tuning + fit output.
///
/// Fit ladder (user-facing low-VRAM contract, "Tier A..D"):
/// - A: weights + f16 KV inside usable VRAM -> full GPU.
/// - B: f16 KV overflows, fp8 KV fits -> `--kv-cache-dtype fp8_e5m2` +
///   tight-tail knobs (small cuda-graph bs, smaller chunked prefill).
/// - C: weights alone overflow -> derived `--cpu-offload-gb` for the
///   overflow (host RAM permitting) + fp8 KV + tight knobs.
/// - D: even offload can't bridge it (or host RAM is too small) -> hard
///   refusal with the numbers, BEFORE w child spawns. An OOM
///   crash-loop at spawn is a planner bug, not an operational state.
///
/// No GPU at all → `--device cpu` lane with a loud warning.
#[allow(clippy::too_many_lines)]
fn compile_sglang(input: &ProfileInput<'_>, tuning: &TuningOverrides) -> Result<Profile, String> {
    let mut argv: Vec<String> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();
    let tun = SglangTuning::effective(input.overlay.sglang.as_ref(), &input.config.sglang);

    let hf = match input.meta {
        ModelMeta::Hf(h) => h,
        ModelMeta::Gguf(g) => {
            return Err(format!(
                "model {} is a GGUF file (arch {}) — the sglang engine consumes HF \
                 safetensors directories; run it on the llamacpp engine (default) or \
                 pull a safetensors checkpoint (e.g. Qwen/Qwen2.5-0.5B-Instruct)",
                input.model_name, g.architecture
            ));
        }
    };

    // --- ctx: overlay > config default, clamped to the training ceiling
    // (max_position_embeddings). YaRN ctx_extend is a llama-server knob;
    // sglang 0.5.19 has no equivalent flag, so an active extend gets a
    // warning instead of a silently ignored number.
    let requested = tuning
        .ctx
        .unwrap_or_else(|| input.overlay.ctx.unwrap_or(input.config.default_ctx));
    let ctx = match hf.ctx_train {
        Some(train) if u64::from(requested) > train => {
            warnings.push(format!(
                "ctx {requested} clamped to model max_position_embeddings {train} \
                 (config.json metadata)"
            ));
            u32::try_from(train).unwrap_or(u32::MAX)
        }
        _ => requested,
    };
    let extend = input.config.effective_ctx_extend(input.model_name);
    if extend > 1.0 {
        warnings.push(format!(
            "ctx_extend {extend} ignored: sglang 0.5.19 has no YaRN context-extension \
             flag; ctx stays at the trained window"
        ));
    }
    push_tuned(
        &mut argv,
        input.supported_flags,
        "model ctx",
        "--context-length",
        &ctx.to_string(),
        &mut warnings,
    );

    // --- concurrency: slots -> --max-running-requests (sglang's batching
    // parallelism; 0/None keeps upstream auto). Deterministic pin forces
    // single-stream exactly like the mistralrs profile, and additionally
    // engages sglang's native deterministic-inference flag when the
    // installed engine advertises it (batch-invariant kernels on top of
    // the serialization pin — additive, never a replacement: the slots=1
    // pin is what actually guarantees token-for-token greedy
    // reproducibility).
    let mut slots = input.overlay.slots.unwrap_or(input.config.slots);
    if input
        .overlay
        .deterministic
        .unwrap_or(input.config.deterministic)
    {
        if slots > 1 {
            warnings.push(format!(
                "deterministic = true pins max-running-requests = 1 — explicit \
                 slots = {slots} ignored: continuous batching perturbs logits in \
                 near-tie positions and breaks token-for-token greedy \
                 reproducibility; drop one of the two knobs"
            ));
        }
        slots = 1;
        push_tuned(
            &mut argv,
            input.supported_flags,
            "deterministic",
            "--enable-deterministic-inference",
            "",
            &mut warnings,
        );
    }
    if slots != 0 {
        push_tuned(
            &mut argv,
            input.supported_flags,
            "model slots",
            "--max-running-requests",
            &slots.to_string(),
            &mut warnings,
        );
    }

    // --- loras: sglang serves PEFT/safetensors adapters via --enable-lora
    // + --lora-paths name=path. llama.cpp-class .gguf/.bin adapters are a
    // different format entirely — hard error with the lane teaching
    // instead of a child crash mid-boot. The llama.cpp `scale` dial has no
    // sglang equivalent (upstream applies the adapter at its trained
    // strength): warn, never silently drop the pair.
    if !input.loras.is_empty() {
        let mut lora_tokens: Vec<String> = Vec::new();
        let mut scale_warned = false;
        for (path, scale) in input.loras {
            let p = std::path::Path::new(path);
            let is_llamacpp_adapter = p
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("gguf") || e.eq_ignore_ascii_case("bin"));
            if is_llamacpp_adapter {
                return Err(format!(
                    "lora {path} is a .gguf/.bin adapter — the sglang engine loads \
                     PEFT/safetensors adapters only (adapter_model.safetensors \
                     dirs); run it on the llamacpp engine or pull a PEFT variant"
                ));
            }
            let name = p.file_stem().and_then(|s| s.to_str()).unwrap_or("adapter");
            lora_tokens.push(format!("{name}={path}"));
            if (*scale - 1.0).abs() > f64::EPSILON && !scale_warned {
                warnings.push(
                    "lora scale is llama.cpp vocabulary and is ignored on sglang — \
                     adapters apply at their trained strength"
                        .into(),
                );
                scale_warned = true;
            }
        }
        push_tuned(
            &mut argv,
            input.supported_flags,
            "loras",
            "--enable-lora",
            "",
            &mut warnings,
        );
        push_tuned_list(
            &mut argv,
            input.supported_flags,
            "loras",
            "--lora-paths",
            &lora_tokens,
            &mut warnings,
        );
    }

    // --- speculative pair: sglang rows resolve `spec = "eagle3"`-class
    // drafts as safetensors dirs; the manifest carries no spec_types for
    // sglang (no --spec-type flag upstream), so the pair gates on flags.
    if let Some(draft) = input.draft_path {
        if input.supported_flags.contains("--speculative-algorithm")
            && input
                .supported_flags
                .contains("--speculative-draft-model-path")
        {
            argv.push("--speculative-algorithm".into());
            argv.push("EAGLE3".into());
            argv.push("--speculative-draft-model-path".into());
            argv.push(draft.to_string());
        } else {
            warnings.push(format!(
                "spec draft {draft} resolved but this sglang engine lacks the \
                 speculative pair; spawning dense"
            ));
        }
    }

    // --- llama-vocabulary knobs that do NOT translate: cache_type is
    // q8_0/q4_0 grammar; sglang wants fp8_e5m2/fp8_e4m3/bf16 via
    // sglang.kv_cache_dtype. Teach, never silently remap.
    if input
        .overlay
        .cache_type
        .as_deref()
        .is_some_and(|c| !c.is_empty())
    {
        warnings.push(
            "cache_type is llama-server vocabulary and is ignored on sglang — set \
             models.<name>.sglang.kv_cache_dtype (auto, bf16, fp8_e5m2, fp8_e4m3) \
             instead"
                .into(),
        );
    }

    // --- the fit ladder
    let kv_dtype_effective = tun
        .kv_cache_dtype
        .clone()
        .unwrap_or_else(|| "auto".to_string());
    let kv_bytes_at = |elem: u64| -> Option<u64> {
        // layers * kv_heads * head_dim * 2 (K and V) * ctx * elem bytes
        // — shared with the mistral.rs pa-fraction derive (one formula).
        hf_kv_bytes(&hf.kv, u64::from(ctx), elem)
    };

    let gpu_label: &'static str;
    let kv_est_bytes: Option<u64>;
    if input.hardware.has_gpu() {
        let vram = capacity_bytes(input.hardware);
        let budget = vram
            .saturating_mul(SGLANG_VRAM_USABLE_PCT)
            .saturating_div(100)
            .saturating_sub(SGLANG_RUNTIME_FLOOR_BYTES);
        let weights = input.model_bytes;
        let kv16 = kv_bytes_at(2);
        let kv8 = kv_bytes_at(1);
        // An explicit cpu_offload_gb pin is honored in EVERY tier (the
        // user asked for offload; the ladder only derives it when
        // unpinned) and feeds the mem-fraction below.
        let mut offload_total_gb = 0.0f32;
        if let Some(pin) = tun.cpu_offload_gb {
            push_tuned(
                &mut argv,
                input.supported_flags,
                "sglang.cpu_offload_gb",
                "--cpu-offload-gb",
                &format!("{pin}"),
                &mut warnings,
            );
            offload_total_gb = pin;
        }
        match kv16 {
            None => {
                warnings.push(
                    "fit ladder skipped: config.json lacks KV geometry \
                     (num_hidden_layers / num_key_value_heads / head_dim); \
                     sglang defaults apply and may OOM on tight cards — a \
                     rebuilt/complete checkpoint dir fixes this"
                        .into(),
                );
                gpu_label = "auto";
                kv_est_bytes = None;
            }
            Some(kv16_v) => {
                let kv8_v = kv8.unwrap_or(kv16_v / 2);
                if weights.saturating_add(kv16_v) <= budget {
                    // Tier A: full GPU, f16 KV.
                    gpu_label = "full";
                    kv_est_bytes = Some(kv16_v);
                    if tun.kv_cache_dtype.is_none() && input.overlay.cache_type.is_none() {
                        // f16 KV is the default; nothing to emit.
                    }
                } else if weights.saturating_add(kv8_v) <= budget {
                    // Tier B: fp8 KV fits.
                    gpu_label = "full";
                    kv_est_bytes = Some(kv8_v);
                    if tun.kv_cache_dtype.is_none() {
                        push_tuned(
                            &mut argv,
                            input.supported_flags,
                            "fit ladder",
                            "--kv-cache-dtype",
                            "fp8_e5m2",
                            &mut warnings,
                        );
                        warnings.push(
                            "fit ladder Tier B: f16 KV would overflow the card — \
                             KV cache pinned to fp8_e5m2 (halves KV VRAM; slight \
                             precision cost on long-tail logits). Override with \
                             models.<name>.sglang.kv_cache_dtype"
                                .into(),
                        );
                    }
                    sglang_tight_fit_knobs(
                        &mut argv,
                        input.supported_flags,
                        slots,
                        &tun,
                        &mut warnings,
                    );
                } else {
                    // Tier C: offload the weights overflow to host RAM.
                    let overflow = weights.saturating_add(kv8_v).saturating_sub(budget);
                    let offload_gb = ceil_gb(overflow);
                    let host_budget = input
                        .hardware
                        .total_ram_mib
                        .saturating_mul(4)
                        .saturating_div(10);
                    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
                    let offload_bytes = (f64::from(offload_gb) * 1e9).round() as u64;
                    if offload_gb > 0.0 && offload_bytes <= Hardware::bytes(host_budget) {
                        gpu_label = "partial";
                        kv_est_bytes = Some(kv8_v);
                        let emitted = if let Some(pin) = tun.cpu_offload_gb {
                            pin
                        } else {
                            warnings.push(format!(
                                "fit ladder Tier C: weights {:.2} GiB + KV {:.2} GiB \
                                 exceed the {:.2} GiB usable VRAM — offloading \
                                 {offload_gb} GB of weights to host RAM (decode \
                                 speed drops roughly with the offloaded share); \
                                 override with models.<name>.sglang.cpu_offload_gb",
                                bytes_gib(weights),
                                bytes_gib(kv8_v),
                                bytes_gib(budget),
                            ));
                            offload_gb
                        };
                        offload_total_gb = emitted;
                        if tun.kv_cache_dtype.is_none() {
                            push_tuned(
                                &mut argv,
                                input.supported_flags,
                                "fit ladder",
                                "--kv-cache-dtype",
                                "fp8_e5m2",
                                &mut warnings,
                            );
                        }
                        push_tuned(
                            &mut argv,
                            input.supported_flags,
                            "fit ladder",
                            "--cpu-offload-gb",
                            &format!("{emitted}"),
                            &mut warnings,
                        );
                        sglang_tight_fit_knobs(
                            &mut argv,
                            input.supported_flags,
                            slots,
                            &tun,
                            &mut warnings,
                        );
                    } else {
                        // Tier D: refusal — spawning would only OOM-loop.
                        return Err(format!(
                            "model {} does not fit this machine: weights {:.2} GiB + \
                             fp8 KV {:.2} GiB vs {:.2} GiB usable VRAM (needs \
                             {:.1} GB host offload, host RAM budget {:.2} GiB). \
                             Use a smaller checkpoint, lower ctx ({ctx}), or \
                             slots = 1",
                            input.model_name,
                            bytes_gib(weights),
                            bytes_gib(kv8_v),
                            bytes_gib(budget),
                            offload_gb,
                            bytes_gib(Hardware::bytes(host_budget)),
                        ));
                    }
                }

                // mem-fraction-static: the anti-OOM heart. Derived from
                // what actually stays device-side (weights minus the
                // EMITTED offload — pin or ladder-derived — plus KV);
                // explicit fraction pins win outright.
                #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
                let gpu_static =
                    weights.saturating_sub((f64::from(offload_total_gb) * 1e9).round() as u64);
                let static_demand = gpu_static.saturating_add(kv_est_bytes.unwrap_or(0));
                let frac = tun.mem_fraction_static.unwrap_or_else(|| {
                    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
                    let raw = (static_demand as f64 / vram.max(1) as f64) as f32;
                    raw.clamp(SGLANG_MEM_FRACTION_MIN, SGLANG_MEM_FRACTION_MAX)
                });
                if vram > 0 {
                    push_tuned(
                        &mut argv,
                        input.supported_flags,
                        "fit ladder",
                        "--mem-fraction-static",
                        &format!("{frac:.3}"),
                        &mut warnings,
                    );
                }
            }
        }
    } else {
        gpu_label = "cpu";
        kv_est_bytes = kv_bytes_at(HfMeta::kv_elem_bytes(&kv_dtype_effective));
        push_tuned(
            &mut argv,
            input.supported_flags,
            "cpu lane",
            "--device",
            "cpu",
            &mut warnings,
        );
        warnings.push(
            "no GPU detected: sglang runs on --device cpu (torch-native attention); \
             expect server-class decode speeds, not GPU throughput"
                .into(),
        );
    }

    // --- HiCache (KV tiering to host RAM), opt-in.
    if tun.hicache_enable == Some(true) {
        push_tuned(
            &mut argv,
            input.supported_flags,
            "sglang.hicache_enable",
            "--enable-hierarchical-cache",
            "",
            &mut warnings,
        );
        if let Some(r) = tun.hicache_ratio {
            push_tuned(
                &mut argv,
                input.supported_flags,
                "sglang.hicache_ratio",
                "--hicache-ratio",
                &format!("{r}"),
                &mut warnings,
            );
        }
        if let Some(s) = tun.hicache_size {
            push_tuned(
                &mut argv,
                input.supported_flags,
                "sglang.hicache_size",
                "--hicache-size",
                &format!("{s}"),
                &mut warnings,
            );
        }
    }

    // --- scalar tuning passthrough (flag-gated, warn-skip).
    // Portability defaults: flashinfer (sglang's pick on CUDA) JIT-compiles
    // kernels with the system nvcc and dies whenever it does not match the
    // bundled CUDA wheels (e.g. nvcc 12.0 vs cu13 torch). Triton attention
    // + pytorch sampling need no external toolchain, so they are the
    // out-of-the-box lane; a tuning pin overrides either.
    if tun.attention_backend.is_none() {
        push_tuned(
            &mut argv,
            input.supported_flags,
            "sglang defaults",
            "--attention-backend",
            "triton",
            &mut warnings,
        );
        // Static portability rationale, not a per-instance health fact:
        // journal only (once per spawn), never the per-session REPL
        // [profile] surface — there the line is noise the user cannot act
        // on whenever the local toolchain cannot host flashinfer anyway.
        tracing::info!(
            model = input.model_name,
            "sglang: attention defaults to triton for portability (flashinfer JIT requires a \
             matching nvcc; the sglang venv bundles one on the child PATH); override with \
             models.<name>.sglang.attention_backend"
        );
    }
    if tun.sampling_backend.is_none() {
        push_tuned(
            &mut argv,
            input.supported_flags,
            "sglang defaults",
            "--sampling-backend",
            "pytorch",
            &mut warnings,
        );
    }
    let mut tun_str = |flag: &str, v: &Option<String>, warnings: &mut Vec<String>| {
        if let Some(v) = v {
            push_tuned(
                &mut argv,
                input.supported_flags,
                "sglang tuning",
                flag,
                v,
                warnings,
            );
        }
    };
    tun_str("--attention-backend", &tun.attention_backend, &mut warnings);
    tun_str("--tool-call-parser", &tun.tool_call_parser, &mut warnings);
    tun_str("--reasoning-parser", &tun.reasoning_parser, &mut warnings);
    tun_str("--tokenizer-path", &tun.tokenizer_path, &mut warnings);
    tun_str("--dtype", &tun.dtype, &mut warnings);
    tun_str("--quantization", &tun.quantization, &mut warnings);
    tun_str("--kv-cache-dtype", &tun.kv_cache_dtype, &mut warnings);
    tun_str("--schedule-policy", &tun.schedule_policy, &mut warnings);
    if let Some(v) = tun.page_size {
        push_tuned(
            &mut argv,
            input.supported_flags,
            "sglang.page_size",
            "--page-size",
            &v.to_string(),
            &mut warnings,
        );
    }
    if let Some(v) = tun.schedule_conservativeness {
        push_tuned(
            &mut argv,
            input.supported_flags,
            "sglang.schedule_conservativeness",
            "--schedule-conservativeness",
            &format!("{v}"),
            &mut warnings,
        );
    }
    if let Some(v) = tun.chunked_prefill_size {
        push_tuned(
            &mut argv,
            input.supported_flags,
            "sglang.chunked_prefill_size",
            "--chunked-prefill-size",
            &v.to_string(),
            &mut warnings,
        );
    }
    if let Some(v) = tun.max_prefill_tokens {
        push_tuned(
            &mut argv,
            input.supported_flags,
            "sglang.max_prefill_tokens",
            "--max-prefill-tokens",
            &v.to_string(),
            &mut warnings,
        );
    }
    if let Some(v) = tun.stream_interval {
        push_tuned(
            &mut argv,
            input.supported_flags,
            "sglang.stream_interval",
            "--stream-interval",
            &v.to_string(),
            &mut warnings,
        );
    }
    if let Some(v) = tun.random_seed {
        push_tuned(
            &mut argv,
            input.supported_flags,
            "sglang.random_seed",
            "--random-seed",
            &v.to_string(),
            &mut warnings,
        );
    }
    if let Some(v) = tun.cuda_graph_max_bs {
        push_tuned(
            &mut argv,
            input.supported_flags,
            "sglang.cuda_graph_max_bs",
            "--cuda-graph-max-bs",
            &v.to_string(),
            &mut warnings,
        );
    }
    if tun.metrics == Some(true) {
        push_tuned(
            &mut argv,
            input.supported_flags,
            "sglang.metrics",
            "--enable-metrics",
            "",
            &mut warnings,
        );
    }
    if tun.skip_warmup == Some(true) {
        push_tuned(
            &mut argv,
            input.supported_flags,
            "sglang.skip_warmup",
            "--skip-server-warmup",
            "",
            &mut warnings,
        );
    }
    if tun.torch_compile == Some(true) {
        push_tuned(
            &mut argv,
            input.supported_flags,
            "sglang.torch_compile",
            "--enable-torch-compile",
            "",
            &mut warnings,
        );
    }

    // --- tokenizer / detokenizer throughput family
    if let Some(v) = &tun.tokenizer_mode {
        push_tuned(
            &mut argv,
            input.supported_flags,
            "sglang.tokenizer_mode",
            "--tokenizer-mode",
            v,
            &mut warnings,
        );
    }
    if let Some(v) = &tun.tokenizer_backend {
        push_tuned(
            &mut argv,
            input.supported_flags,
            "sglang.tokenizer_backend",
            "--tokenizer-backend",
            v,
            &mut warnings,
        );
    }
    if let Some(v) = tun.tokenizer_worker_num {
        push_tuned(
            &mut argv,
            input.supported_flags,
            "sglang.tokenizer_worker_num",
            "--tokenizer-worker-num",
            &v.to_string(),
            &mut warnings,
        );
    }
    if let Some(v) = tun.detokenizer_worker_num {
        push_tuned(
            &mut argv,
            input.supported_flags,
            "sglang.detokenizer_worker_num",
            "--detokenizer-worker-num",
            &v.to_string(),
            &mut warnings,
        );
    }
    if tun.dynamic_batch_tokenizer == Some(true) {
        push_tuned(
            &mut argv,
            input.supported_flags,
            "sglang.dynamic_batch_tokenizer",
            "--enable-dynamic-batch-tokenizer",
            "",
            &mut warnings,
        );
    }
    if let Some(v) = tun.dynamic_batch_tokenizer_batch_size {
        push_tuned(
            &mut argv,
            input.supported_flags,
            "sglang.dynamic_batch_tokenizer_batch_size",
            "--dynamic-batch-tokenizer-batch-size",
            &v.to_string(),
            &mut warnings,
        );
    }
    if let Some(v) = tun.dynamic_batch_tokenizer_batch_timeout {
        push_tuned(
            &mut argv,
            input.supported_flags,
            "sglang.dynamic_batch_tokenizer_batch_timeout",
            "--dynamic-batch-tokenizer-batch-timeout",
            &format!("{v}"),
            &mut warnings,
        );
    }

    // --- structured output + radix cache policy family
    if let Some(v) = &tun.grammar_backend {
        push_tuned(
            &mut argv,
            input.supported_flags,
            "sglang.grammar_backend",
            "--grammar-backend",
            v,
            &mut warnings,
        );
    }
    if let Some(v) = &tun.radix_eviction_policy {
        push_tuned(
            &mut argv,
            input.supported_flags,
            "sglang.radix_eviction_policy",
            "--radix-eviction-policy",
            v,
            &mut warnings,
        );
    }
    if tun.session_radix_cache == Some(true) {
        push_tuned(
            &mut argv,
            input.supported_flags,
            "sglang.session_radix_cache",
            "--enable-session-radix-cache",
            "",
            &mut warnings,
        );
    }
    if tun.mixed_chunk == Some(true) {
        push_tuned(
            &mut argv,
            input.supported_flags,
            "sglang.mixed_chunk",
            "--enable-mixed-chunk",
            "",
            &mut warnings,
        );
    }

    // --- idle / lifecycle hygiene family
    if tun.sleep_on_idle == Some(true) {
        push_tuned(
            &mut argv,
            input.supported_flags,
            "sglang.sleep_on_idle",
            "--sleep-on-idle",
            "",
            &mut warnings,
        );
    }
    if tun.memory_saver == Some(true) {
        push_tuned(
            &mut argv,
            input.supported_flags,
            "sglang.memory_saver",
            "--enable-memory-saver",
            "",
            &mut warnings,
        );
    }
    if let Some(v) = tun.watchdog_timeout {
        push_tuned(
            &mut argv,
            input.supported_flags,
            "sglang.watchdog_timeout",
            "--watchdog-timeout",
            &format!("{v}"),
            &mut warnings,
        );
    }

    // --- scheduler / observability family
    if tun.cache_report == Some(true) {
        push_tuned(
            &mut argv,
            input.supported_flags,
            "sglang.cache_report",
            "--enable-cache-report",
            "",
            &mut warnings,
        );
    }
    if let Some(v) = tun.batch_notify_size {
        push_tuned(
            &mut argv,
            input.supported_flags,
            "sglang.batch_notify_size",
            "--batch-notify-size",
            &v.to_string(),
            &mut warnings,
        );
    }
    if let Some(v) = tun.scheduler_recv_interval {
        push_tuned(
            &mut argv,
            input.supported_flags,
            "sglang.scheduler_recv_interval",
            "--scheduler-recv-interval",
            &v.to_string(),
            &mut warnings,
        );
    }

    // --- explicit cuda-graph capture list + KV token cap
    if let Some(list) = &tun.cuda_graph_bs {
        let tokens: Vec<String> = list.iter().map(ToString::to_string).collect();
        push_tuned_list(
            &mut argv,
            input.supported_flags,
            "sglang.cuda_graph_bs",
            "--cuda-graph-bs",
            &tokens,
            &mut warnings,
        );
    }
    // Prefill graph backend: the 42-shape breakable prefill capture has
    // deadlocked at 0% on hybrid-GPU laptops (live-proven with an AWQ
    // marlin model); `disabled` skips it while decode graphs stay on.
    if let Some(backend) = &tun.cuda_graph_backend_prefill {
        push_tuned(
            &mut argv,
            input.supported_flags,
            "sglang.cuda_graph_backend_prefill",
            "--cuda-graph-backend-prefill",
            backend,
            &mut warnings,
        );
    }
    if let Some(v) = tun.max_total_tokens {
        push_tuned(
            &mut argv,
            input.supported_flags,
            "sglang.max_total_tokens",
            "--max-total-tokens",
            &v.to_string(),
            &mut warnings,
        );
    }

    // --- parallel sizes: only meaningful above 1; a >1 pin on a
    // single-GPU machine is a spawn-time death (sglang shards across
    // tp/dp/pp/ep ranks), so warn at compile time with the census.
    let parallel_pins = [
        ("tp_size", &tun.tp_size, "--tp-size"),
        ("dp_size", &tun.dp_size, "--dp-size"),
        ("pp_size", &tun.pp_size, "--pp-size"),
        ("ep_size", &tun.ep_size, "--ep-size"),
    ];
    let multi_pin = parallel_pins
        .iter()
        .any(|(_, v, _)| v.is_some_and(|n| n > 1));
    for (field, v, flag) in parallel_pins {
        if let Some(n) = v {
            if *n > 1 {
                push_tuned(
                    &mut argv,
                    input.supported_flags,
                    &format!("sglang.{field}"),
                    flag,
                    &n.to_string(),
                    &mut warnings,
                );
            } else {
                warnings.push(format!(
                    "sglang.{field} = {n} is the upstream default; nothing emitted"
                ));
            }
        }
    }
    if multi_pin && input.hardware.gpus.len() < 2 {
        warnings.push(format!(
            "sglang parallel sizes > 1 on a {} GPU machine — sglang shards the \
             model across ranks and will fail or CPU-shard at spawn; this pin \
             only makes sense multi-GPU",
            input.hardware.gpus.len()
        ));
    }

    // --- lora capacity knobs (adapters themselves ride the loras lane)
    if let Some(v) = tun.max_lora_rank {
        push_tuned(
            &mut argv,
            input.supported_flags,
            "sglang.max_lora_rank",
            "--max-lora-rank",
            &v.to_string(),
            &mut warnings,
        );
    }
    if let Some(v) = &tun.lora_backend {
        push_tuned(
            &mut argv,
            input.supported_flags,
            "sglang.lora_backend",
            "--lora-backend",
            v,
            &mut warnings,
        );
    }

    // --- extra_args: strict. Reserved flags (connection quintet + ladder
    // outputs) are a hard error — a duplicate would silently shadow the
    // supervisor-owned values (loopback bind, API key, VRAM budget).
    if let Some(extra) = &input.overlay.extra_args {
        for a in extra {
            if a.starts_with("--") {
                let name = a.split('=').next().unwrap_or(a);
                if SGLANG_RESERVED_FLAGS.contains(&name) {
                    return Err(format!(
                        "extra_args {name} is reserved on sglang: the supervisor \
                         owns the connection flags and the VRAM ladder owns {name}; \
                         use the config knobs (ctx, slots, sglang.*) instead"
                    ));
                }
                if !input.supported_flags.contains(name) {
                    return Err(format!(
                        "engine {} does not support {name}; try `pallama engine \
                         install --kind sglang` for a newer sglang",
                        input.engine_tag
                    ));
                }
            }
        }
        argv.extend(extra.iter().cloned());
    }

    Ok(Profile {
        argv,
        warnings,
        ctx,
        gpu: gpu_label,
        kv_est_bytes,
        ctx_autofit: None,
    })
}

/// Tight-fit tail knobs for ladder Tiers B/C: shrink cuda-graph capture
/// (upstream default bs 256 allocates graphs per batch size — pure waste
/// at slots <= 4 on an 8 GB card) and chunked prefill (default 8192-token
/// activation peaks). Explicit tuning pins win.
fn sglang_tight_fit_knobs(
    argv: &mut Vec<String>,
    flags: &std::collections::BTreeSet<String>,
    slots: u32,
    tun: &SglangTuning,
    warnings: &mut Vec<String>,
) {
    // Explicit pins win: either a max-bs scalar or an explicit capture
    // list suppresses the ladder's derived value.
    if tun.cuda_graph_max_bs.is_none() && tun.cuda_graph_bs.is_none() {
        let bs = if slots == 0 { 4 } else { slots.min(4) };
        push_tuned(
            argv,
            flags,
            "fit ladder",
            "--cuda-graph-max-bs",
            &bs.to_string(),
            warnings,
        );
    }
    if tun.chunked_prefill_size.is_none() {
        push_tuned(
            argv,
            flags,
            "fit ladder",
            "--chunked-prefill-size",
            "2048",
            warnings,
        );
    }
}

/// Ceil bytes to whole GB (f32 flag grammar upstream).
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn ceil_gb(bytes: u64) -> f32 {
    (bytes as f64 / 1e9).ceil().max(0.0) as f32
}

fn bytes_gib(bytes: u64) -> f64 {
    f64::from(u32::try_from(bytes / (1024 * 1024)).unwrap_or(u32::MAX)) / 1024.0
}

fn resolve_ctx(
    input: &ProfileInput<'_>,
    overlay: &ModelOverride,
    warnings: &mut Vec<String>,
) -> u32 {
    let requested = overlay.ctx.unwrap_or(input.config.default_ctx);
    // F109: when YaRN ctx extension is active the usable window is the
    // TRAINED length stretched by ctx_extend — clamping to the bare
    // trained length defeated the feature (extend 2.0 on a 64k model
    // could never run past 65536 no matter what ctx asked for).
    let extend = input.config.effective_ctx_extend(input.model_name);
    if let Some(train_raw) = input.meta.context_length() {
        let train = u32::try_from(train_raw).unwrap_or(u32::MAX);
        let ceiling = if extend > 1.0 {
            // train <= u32::MAX and extend is clamped > 1.0 by config
            // validation, and the min() keeps the product inside u32
            // range before the cast — no sign or truncation hazard.
            #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
            let stretched = (f64::from(train) * extend).floor().min(f64::from(u32::MAX)) as u64;
            u32::try_from(stretched).unwrap_or(u32::MAX)
        } else {
            train
        };
        if requested > ceiling {
            if extend > 1.0 {
                warnings.push(format!(
                    "ctx {requested} clamped to yarn-stretched window {ceiling} (trained {train} x ctx_extend {extend})"
                ));
            } else {
                warnings.push(format!(
                    "ctx {requested} clamped to model context_length {train} (GGUF metadata)"
                ));
            }
            ceiling
        } else {
            requested
        }
    } else {
        warnings.push(
            "ctx not clamped: GGUF lacks context_length (rule skipped, not estimated)".into(),
        );
        requested
    }
}
/// Rule 6 ladder: KV cache quantization by capacity math, not stacked
/// thresholds. Estimated f16 KV is `2*blocks*kv_heads*head_dim*ctx*2`
/// bytes; `q8_0` halves it, `q4_0` quarters it. The cheapest grade that keeps
/// weights+KV within 0.9x VRAM wins. Missing GGUF fields skip with a named
/// warning — never guessed.
fn kv_quant_ladder(
    input: &ProfileInput<'_>,
    vram_bytes: u64,
    ctx: u32,
    warnings: &mut Vec<String>,
) -> Option<&'static str> {
    // The multimodal projector is GPU-resident too — capacity math that
    // ignores it OOMs at load on vision models (live incident).
    let mmproj_bytes = input
        .mmproj_path
        .and_then(|p| std::fs::metadata(p).ok())
        .map_or(0, |m| m.len());
    let resident = input.model_bytes.saturating_add(mmproj_bytes);
    let gpu_resident = input.hardware.has_gpu() && resident <= vram_bytes;
    if !gpu_resident {
        return None;
    }
    let Some(kv) = kv_f16_bytes(input, ctx) else {
        warnings.push("KV-quant rule skipped: GGUF lacks layer geometry".into());
        return None;
    };
    let budget = vram_bytes / 10 * 9;
    if resident.saturating_add(kv) <= budget {
        None
    } else if resident.saturating_add(kv / 2) <= budget {
        Some("q8_0")
    } else {
        if resident.saturating_add(kv / 4) > budget {
            warnings.push(
                "KV q4_0 engaged but weights+KV still exceed 90% VRAM; --fit will shrink ctx or the engine may OOM — consider a smaller quant".into(),
            );
        }
        Some("q4_0")
    }
}

/// f16 KV-cache bytes at `ctx` for this model, when the GGUF carries
/// enough geometry (shared by the KV ladder and the offload resolver).
/// Formula lives on `GgufMeta::kv_f16_bytes` (MLA/SWA-aware) so every
/// caller — ladder, offload resolver, gateway preflight — shares one math.
#[must_use]
fn kv_f16_bytes(input: &ProfileInput<'_>, ctx: u32) -> Option<u64> {
    match input.meta {
        ModelMeta::Gguf(g) => g.kv_f16_bytes(u64::from(ctx)),
        ModelMeta::Hf(_) => None,
    }
}

/// Geometry-only f16 KV bytes for a given header (shared by the dense
/// estimate and the draft-model planner charge).
fn kv_f16_bytes_meta(gguf: &GgufMeta, ctx: u32) -> Option<u64> {
    gguf.kv_f16_bytes(u64::from(ctx))
}

/// Public KV estimate for callers outside the compiler (the supervisor's
/// co-residency planner). Same math as the internal ladder input.
#[must_use]
pub fn estimate_kv_f16(input: &ProfileInput<'_>, ctx: Option<u32>) -> Option<u64> {
    let ctx = ctx.unwrap_or_else(|| input.overlay.ctx.unwrap_or(input.config.default_ctx));
    kv_f16_bytes(input, ctx)
}

/// VRAM the KV cache will actually occupy at spawn. Device truth
/// (live-verified b10948): the KV cache allocates DEVICE-side on BOTH
/// lanes — `--kv-unified` shares one buffer across sequences but never
/// moves it to system RAM — so the charge is the full f16 estimate
/// either way. Lane-independent by design: unified spawns must not be
/// read as 2-4x cheaper than they really are, or co-residency planning
/// downgrades/evicts on phantom headroom.
#[must_use]
pub fn estimate_kv_vram_charge(input: &ProfileInput<'_>, ctx: Option<u32>) -> Option<u64> {
    estimate_kv_f16(input, ctx)
}

/// A16 RAM-share for the prompt-cache budget, from the live hit-rate hint
/// (None = static 30%): prefix-heavy traffic earns 40%, cache-cold
/// releases to 20%.
fn cache_ram_pct(hit: Option<f64>) -> u32 {
    match hit {
        Some(h) if h > 0.5 => 40,
        Some(h) if h < 0.1 => 20,
        _ => 30,
    }
}

/// The `--cache-ram` budget rule 12 will actually emit, in MiB: the
/// configured request clamped to the A16 RAM-share cap. None when the
/// knob requests unlimited (0) or the box RAM is unknown — callers apply
/// their own fallback share. Single source of truth so slot sizing and
/// the emitted flag can never disagree.
fn effective_cache_ram_mib(input: &ProfileInput<'_>) -> Option<u64> {
    if input.config.cache_ram_mb <= 0 || input.hardware.total_ram_mib == 0 {
        return None;
    }
    let cap = input.hardware.total_ram_mib * u64::from(cache_ram_pct(input.cache_hit_rate)) / 100;
    let requested = u64::try_from(input.config.cache_ram_mb).unwrap_or(u64::MAX);
    Some(requested.min(cap))
}

/// Verdict for a PINNED unified-KV ctx against the GPU: the 2b fit
/// constraint as ONE decision shared by profile compilation and the
/// gateway `num_ctx` preflight. Both sites must refuse the same shapes
/// — a split brain here is how a doomed pin evicts a healthy instance
/// and 502-loops the requester (live-repro'd 2026-09-13: 21x
/// spawn-retry of a child that could never create its context).
#[must_use]
pub enum UnifiedCtxVerdict {
    /// Weights + f16 KV fit the device (the 85%..100% band is still
    /// `Fit` — the offload resolver answers it with `auto`, letting
    /// the engine fitter split layers; honest degraded serve).
    Fit,
    /// The device cannot host the pin even at 100%: refuse, with
    /// teaching text.
    Refuse(String),
}

/// The shared 2b decision, judged against DEVICE VRAM — the unified KV
/// pool is device-backed (live-verified b10948: `CUDA0 KV buffer` with
/// `--kv-unified` on; the flag shares one buffer across sequences, it
/// does not move the pool to system RAM). `model_bytes`/`kv_bytes` are
/// bytes (f16 KV at the pinned TOTAL ctx), `total_vram_bytes` the
/// capacity the spawn would contend for.
pub fn unified_ctx_verdict(
    model_bytes: u64,
    kv_bytes: u64,
    total_vram_bytes: u64,
    ctx: u32,
) -> UnifiedCtxVerdict {
    let mib = |b: u64| b / (1024 * 1024);
    let weights_mib = mib(model_bytes);
    let kv_mib = mib(kv_bytes);
    let vram_mib = mib(total_vram_bytes);
    let demand_mib = weights_mib + kv_mib;
    if demand_mib
        .saturating_mul(1024 * 1024)
        .saturating_add(UNIFIED_SPAWN_OVERHEAD_BYTES)
        <= total_vram_bytes
    {
        return UnifiedCtxVerdict::Fit;
    }
    UnifiedCtxVerdict::Refuse(format!(
        "pinned num_ctx {ctx} cannot fit the GPU: weights {weights_mib} MiB + KV {kv_mib} \
         MiB = {demand_mib} MiB demand (+{} MiB spawn overhead) exceeds the {vram_mib} MiB VRAM \
         — lower num_ctx, cache_type = \"q8_0\" (halves KV), or a smaller quant (pallama fit)",
        UNIFIED_SPAWN_OVERHEAD_BYTES / (1024 * 1024)
    ))
}

/// The `--cache-ram <MiB>` value spelled in `extra_args`, if w
/// (`--cache-ram=X` accepted). Explicit argv wins over the config knob
/// (the same doctrine as `-mm`): the user owns the budget, the A16
/// clamp and the 2a-bis floor do not override deliberate argv, and
/// rule 12 skips its own emission so the flag lands exactly once.
#[must_use]
pub fn cache_ram_from_extra_args(extra: &[String]) -> Option<u64> {
    let mut it = extra.iter().peekable();
    while let Some(a) = it.next() {
        if a == "--cache-ram" {
            if let Some(v) = it.peek().and_then(|s| s.parse::<u64>().ok()) {
                return Some(v);
            }
        } else if let Some(v) = a.strip_prefix("--cache-ram=") {
            if let Ok(v) = v.parse::<u64>() {
                return Some(v);
            }
        }
    }
    None
}

/// Outcome of slot/ctx resolution: `slots` is the `-np` to emit,
/// `total_ctx` the `--ctx-size` (upstream divides it across slots),
/// `per_slot_ctx` what ONE slot holds (feeds `Profile.ctx` so prompt
/// preflight bounds a single request). `autofit` is Some when auto-fit
/// divided the resolved ctx to earn parallel slots (see
/// `AUTOFIT_CTX_FLOOR`).
struct ResolvedSlots {
    slots: u32,
    total_ctx: u32,
    per_slot_ctx: u32,
    autofit: Option<(u32, u32)>,
}

/// Per-slot ctx floor for auto-fit: ollama's own default context (and its
/// VRAM-tier auto-fit never goes below it). Halving past 4096 would trade
/// usable prompt room for slots the workload may not have.
const AUTOFIT_CTX_FLOOR: u32 = 4096;

/// Resolve the slot count and the TOTAL ctx to emit. `slots = 0` (pallama
/// auto, the default) sizes concurrency from capacity: upstream divides
/// `--ctx-size` across `-np` slots, so N slots cost N x per-slot KV (RAM
/// under `--kv-unified`, VRAM on the classic path) and no extra weights —
/// scaling the total keeps each slot at the resolved per-slot ctx while
/// concurrent streams batch on the GPU instead of queueing (measured
/// ~2.5x system throughput at 4 slots for +138 MiB VRAM). Explicit pins —
/// overlay slots (which the supervisor also uses for LC4-adopted values),
/// `slots = 1`, manual N — pass through untouched; manual N keeps the
/// historical slicing semantics (total = resolved ctx, per-slot = ctx/N).
///
/// When capacity caps auto at ONE slot and the ctx came from the default
/// (not pinned by tuning/overlay), auto-fit re-spends the same total-ctx
/// budget as shallower parallel slots (4x4096 costs the same capacity as
/// 1x16384) — ollama's throughput-first default. Pinned ctx (per-request,
/// tuning bench, or overlay) is a hard pin: never divided.
/// Pinned-pool walk (unified lane): the capacity vram-slots axis is
/// CAP-blind under `--kv-unified`, so it can admit an np whose whole pool
/// (np × `base_ctx` of device-backed KV on top of the weights) the 2b
/// verdict refuses — the spawn would die at context creation and retry
/// per request. Walk down to the largest count the verdict actually
/// hosts and say so. Explicit slot pins never reach here (an unfit pin
/// must error loudly at 2b, not be silently reshaped); the classic lane's
/// capacity axis already subtracts the weights, so it keeps its own math.
fn walk_pinned_np(
    input: &ProfileInput<'_>,
    np: u32,
    base_ctx: u32,
    vram_bytes: u64,
    warnings: &mut Vec<String>,
) -> u32 {
    let pool_hosts = |n: u32| match kv_f16_bytes(input, n * base_ctx) {
        Some(kv) => !matches!(
            unified_ctx_verdict(input.model_bytes, kv, vram_bytes, base_ctx),
            UnifiedCtxVerdict::Refuse(_)
        ),
        None => false, // unknown KV charge — never guess that it fits
    };
    let full = np;
    let mut np = np;
    while np > 1 && !pool_hosts(np) {
        np -= 1;
    }
    if np < full {
        warnings.push(format!(
            "pinned ctx {base_ctx}: auto slots np {full} pool ({} ctx) exceeds the GPU \
             VRAM budget — walked down to np {np} (pool {}); pin slots = {full} to force \
             the conflict to error instead",
            full * base_ctx,
            np * base_ctx
        ));
    }
    np
}

fn resolve_slots(
    input: &ProfileInput<'_>,
    overlay: &ModelOverride,
    base_ctx: u32,
    vram_bytes: u64,
    ctx_pinned: bool,
    warnings: &mut Vec<String>,
) -> ResolvedSlots {
    let slots = overlay.slots.unwrap_or(input.config.slots);
    if overlay.deterministic.unwrap_or(input.config.deterministic) {
        if slots > 1 {
            warnings.push(format!(
                "deterministic = true pins slots = 1 — explicit slots = {slots} ignored: \
                 multi-slot batches perturb logits in near-tie positions and break \
                 token-for-token greedy reproducibility; drop one of the two knobs"
            ));
        } else if slots == 0 {
            warnings.push(
                "deterministic = true: slots pinned to 1 — concurrent streams queue \
                 instead of batching; unset the pin to recover multi-slot throughput"
                    .to_string(),
            );
        }
        return ResolvedSlots {
            slots: 1,
            total_ctx: base_ctx,
            per_slot_ctx: base_ctx,
            autofit: None,
        };
    }
    if slots != 0 {
        return ResolvedSlots {
            slots,
            total_ctx: base_ctx,
            per_slot_ctx: base_ctx / slots.max(1),
            autofit: None,
        };
    }
    // --- auto: probe capacity at the resolved per-slot ctx first
    let mut np = auto_slots_capacity(input, base_ctx, vram_bytes);
    if ctx_pinned && np >= 2 && kv_unified_emitted(input) {
        np = walk_pinned_np(input, np, base_ctx, vram_bytes, warnings);
    }
    if vulkan_mmproj_guard(input) {
        warn_vulkan_slot_cap(input, base_ctx, vram_bytes, warnings);
    }
    if np >= 2 {
        // Auto-slot decision rationale: journal only — a machine decision
        // with a tuning tip is not a user-facing warning.
        tracing::info!(
            model = input.model_name,
            "profile: slots auto -np {np} with total ctx {} (per-slot {base_ctx}) — concurrent \
             streams batch on the GPU instead of queueing; pin slots = 1 for single-client \
             full-speed",
            np * base_ctx
        );
        return ResolvedSlots {
            slots: np,
            total_ctx: np * base_ctx,
            per_slot_ctx: base_ctx,
            autofit: None,
        };
    }
    // --- auto-fit: single slot at the full resolved ctx. If that ctx is
    // only the config default (not a pin), re-spend its capacity budget
    // as N shallower slots — the total-ctx charge the capacity guards
    // scale on is UNCHANGED (4x4096 == 1x16384 under both the unified
    // per-token compute guard and classic KV math), so this adds zero
    // boot risk while un-serializing concurrent streams.
    if !ctx_pinned && base_ctx > AUTOFIT_CTX_FLOOR {
        let mut best: Option<(u32, u32)> = None;
        let mut per_slot = base_ctx / 2;
        while per_slot >= AUTOFIT_CTX_FLOOR {
            let cand = auto_slots_capacity(input, per_slot, vram_bytes);
            if cand >= 2 {
                // iterate order is largest-per-slot first, so keeping the
                // FIRST shape on np ties tie-breaks to the larger ctx
                if best.is_none_or(|(bn, _)| cand > bn) {
                    best = Some((cand, per_slot));
                }
            }
            per_slot /= 2;
        }
        if let Some((np, per_slot)) = best {
            // Auto-fit decision rationale: journal only — same
            // classification as the -np auto note above.
            tracing::info!(
                model = input.model_name,
                "profile: slots auto-fit default ctx {base_ctx} fits only 1 concurrent slot — \
                 re-spent the same capacity budget as -np {np} x {per_slot} ctx (total {}, \
                 within the capacity guards — floor-rounding may grant a little extra \
                 headroom; prompts longer than {per_slot} tokens trigger the num_ctx \
                 restart-once path at a wider ctx); pin ctx or slots to disable",
                np * per_slot
            );
            return ResolvedSlots {
                slots: np,
                total_ctx: np * per_slot,
                per_slot_ctx: per_slot,
                autofit: Some((per_slot, np)),
            };
        }
    }
    ResolvedSlots {
        slots: 1,
        total_ctx: base_ctx,
        per_slot_ctx: base_ctx,
        autofit: None,
    }
}

/// True when the conservative vulkan-class + projector guard applies (see
/// `auto_slots_capacity`): mixed integrated+discrete census on a unified
/// build with a projector attached.
fn vulkan_mmproj_guard(input: &ProfileInput<'_>) -> bool {
    let mmproj_bytes = mmproj_size(input);
    input.engine_census.iter().any(GpuInfo::is_integrated)
        && input.engine_census.iter().any(|g| !g.is_integrated())
        && mmproj_bytes > 0
}

fn mmproj_size(input: &ProfileInput<'_>) -> u64 {
    input
        .mmproj_path
        .and_then(|p| std::fs::metadata(p).ok())
        .map_or(0, |m| m.len())
}

/// The capacity warning previously emitted inline by the auto-slot
/// resolver; kept for the non-autofit path so `ps` readers still see WHY
/// concurrency is capped on vulkan-class projector spawns.
fn warn_vulkan_slot_cap(
    input: &ProfileInput<'_>,
    per_slot_ctx: u32,
    vram_bytes: u64,
    warnings: &mut Vec<String>,
) {
    const AUTO_SLOTS_CAP: u32 = 4;
    let resident = input.model_bytes.saturating_add(mmproj_size(input));
    let Some(kv_per_slot) = kv_f16_bytes(input, per_slot_ctx) else {
        return;
    };
    let headroom = vram_bytes
        .saturating_sub(resident)
        .saturating_sub(kv_per_slot);
    let cap = (headroom / (u64::from(per_slot_ctx) * UNIFIED_COMPUTE_PER_TOKEN_BYTES))
        .min(u64::from(AUTO_SLOTS_CAP))
        .max(1);
    if cap < u64::from(AUTO_SLOTS_CAP) {
        warnings.push(format!(
            "slots auto capped at {cap}: vulkan-class build (mixed integrated+discrete \
             census) with a projector attached overcommits VRAM as total ctx scales — \
             measured degraded boot at np2/32k total ctx on an 8 GiB card; use a \
             projector-free model or explicit slots to override"
        ));
    }
}

/// Capacity-aware slot count for `slots = 0`, pure (no warnings — the
/// auto-fit loop probes it repeatedly). Clamps, in order: the measured
/// 4-slot sweet spot; the model's trained context (the scaled total
/// cannot exceed it); the `--cache-ram` mmap soft cap (prompt-cache
/// working set, NOT the KV pool — the pool is device-backed on both
/// lanes); and the 85% VRAM envelope the offload resolver uses (the KV
/// charge is VRAM-resident everywhere). Unknown KV geometry means
/// unknown cost: stay single-slot, never guess.
fn auto_slots_capacity(input: &ProfileInput<'_>, base_ctx: u32, vram_bytes: u64) -> u32 {
    // Capacity-aware ceiling. Cards with 8 GiB+ take 8 (np-sweep on an
    // 8 GiB RTX 4070: slots 8 = 440.7 tok/s combined on a 4-stream burst
    // vs 394 at the old cap of 4, identical per-request walls = true
    // parallelism, not queueing); smaller cards keep the conservative 4
    // where the KV budget, not the scheduler, stays the binding cap.
    let auto_slots_cap: u32 = if vram_bytes >= Hardware::bytes(8192) {
        8
    } else {
        4
    };
    if base_ctx == 0 {
        return 1;
    }
    let train_slots = match input.meta.context_length() {
        Some(train) => u32::try_from(train / u64::from(base_ctx))
            .unwrap_or(1)
            .max(1),
        None => 1,
    };
    let Some(kv_per_slot) = kv_f16_bytes(input, base_ctx) else {
        return 1;
    };
    if kv_per_slot == 0 {
        return 1;
    }
    let ram_budget_bytes =
        Hardware::bytes(effective_cache_ram_mib(input).unwrap_or(input.hardware.total_ram_mib / 3));
    let clamp_to_cap = |slots: u64| {
        auto_slots_cap
            .min(u32::try_from(slots).unwrap_or(auto_slots_cap))
            .max(1)
    };
    let ram_slots = clamp_to_cap(ram_budget_bytes / kv_per_slot);
    let resident = input.model_bytes.saturating_add(mmproj_size(input));
    let vram_slots = if vulkan_mmproj_guard(input) {
        // Mixed integrated+discrete census with a projector attached:
        // each slot costs its f16 KV PLUS the measured per-token compute
        // charge (see UNIFIED_COMPUTE_PER_TOKEN_BYTES) — the vulkan
        // build allocates fat ctx-scaled compute buffers that the KV
        // term alone never sees (measured degraded boot at np2/32k on
        // an 8 GiB card).
        let per_slot_bytes = kv_per_slot + u64::from(base_ctx) * UNIFIED_COMPUTE_PER_TOKEN_BYTES;
        let headroom = vram_bytes.saturating_sub(resident);
        clamp_to_cap(headroom / per_slot_bytes)
    } else {
        // Device truth: the KV pool is VRAM-resident on BOTH lanes, so
        // slot capacity pays the same 85% envelope everywhere.
        let headroom = (vram_bytes / 100 * 85).saturating_sub(resident);
        clamp_to_cap(headroom / kv_per_slot)
    };
    auto_slots_cap
        .min(train_slots)
        .min(ram_slots)
        .min(vram_slots)
        .max(1)
}

/// Resolve `--gpu-layers` deterministically when the answer is obvious.
/// Returns (flag-value, ps-label).
///
/// - no GPU -> "0" / "cpu" (expected on CPU boxes: label, no warning)
/// - weights(+mmproj) and f16 KV comfortably fit VRAM (<= 85%) -> "999" /
///   "full" — same placement the engine's auto picks, now *known*
/// - weights alone exceed VRAM but a GPU exists -> "auto" / "partial" +
///   warning (engine splits layers across CPU+GPU; tok/s drops)
/// - everything else (tight fits) -> "auto" / "auto": the engine's
///   fine-grained estimate beats ours and a wrong pin OOMs the load
///
/// Spawn-time capacity for every VRAM-budgeted profile rule: FREE VRAM
/// when the probe reports w, else the card total. Sizing against totals
/// pins `-ngl 999` on boxes whose VRAM a neighbour already consumed at
/// spawn time — the engine's `--fit` refuses to shrink an EXPLICIT ngl
/// ("already set by user, abort", live-proven) and the load OOMs instead
/// of degrading to the `auto` band. Free-based math keeps the offload
/// pin, KV ladder, and slot capacity honest under contention; idle boxes
/// report free ≈ total so nothing changes there. Zero/absent probe
/// readings fail open to the total (a stale number beats refusing to
/// spawn).
fn capacity_bytes(hw: &Hardware) -> u64 {
    let free_mib = hw.free_vram_mib();
    if free_mib > 0 {
        Hardware::bytes(free_mib)
    } else {
        Hardware::bytes(hw.total_vram_mib())
    }
}

/// True when the speculative draft will actually ride this spawn:
/// path known, file present, and the picked card fits dense + draft +
/// KV floor + spawn overhead. Rule 4's gpu-layers unpin and rule 7's
/// capacity gate must AGREE — a declined draft (running dense) must not
/// needlessly hand gpu-layers to the live fitter (live-repro'd: 9B with
/// an MTP pair too big for the card ran dense but unpinned).
fn spec_draft_will_attach(input: &ProfileInput<'_>) -> bool {
    let Some(draft) = input.draft_path else {
        return false;
    };
    let draft_bytes = std::fs::metadata(draft).map_or(0, |m| m.len());
    if draft_bytes == 0 {
        return false;
    }
    let card_free_bytes: u64 = input
        .hardware
        .gpus
        .iter()
        .map(|g| g.free_mib.saturating_mul(1024 * 1024))
        .sum();
    input
        .model_bytes
        .saturating_add(draft_bytes)
        .saturating_add(KV_UNIFIED_VRAM_FLOOR_BYTES)
        .saturating_add(UNIFIED_SPAWN_OVERHEAD_BYTES)
        <= card_free_bytes
}

fn resolve_gpu_offload(
    input: &ProfileInput<'_>,
    ctx: u32,
    vram_bytes: u64,
    warnings: &mut Vec<String>,
) -> (&'static str, &'static str) {
    if !input.hardware.has_gpu() {
        return ("0", "cpu");
    }
    let mmproj = input
        .mmproj_path
        .and_then(|p| std::fs::metadata(p).ok())
        .map_or(0, |m| m.len());
    let resident = input.model_bytes.saturating_add(mmproj);
    let kv = kv_f16_bytes(input, ctx).unwrap_or(0);
    if resident > vram_bytes {
        warnings.push(format!(
            "weights {} MiB exceed VRAM {} MiB — engine will split layers across CPU+GPU; expect lower tok/s or a smaller quant",
            resident / (1 << 20),
            vram_bytes / (1 << 20)
        ));
        return ("auto", "partial");
    }
    // One placement model for both lanes (device truth): the KV pool is
    // device-backed under `--kv-unified` too (shared buffer, not host
    // memory), so the classic 85% envelope is THE gate. The 85%..100%
    // band falls through to "auto" — the engine's live fitter places
    // the last layers, possibly CPU-splitting them (honest degraded
    // serve), and a hard pin there would abort the fitter ("already
    // set by user to 999") into a cudaMalloc OOM.
    if resident.saturating_add(kv) <= vram_bytes / 100 * 85 {
        if spec_draft_will_attach(input) {
            warnings.push(
                "speculative draft adds device-side weights+KV beyond the planner charge; \
                 --gpu-layers left to the engine live fitter (common_fit_params)"
                    .to_string(),
            );
            return ("auto", "auto");
        }
        return ("999", "full");
    }
    ("auto", "auto")
}

/// True when the compiled argv will carry the POSITIVE `--kv-unified`
/// flag: the engine manifest supports it and the user did not opt out
/// (explicit `kv_unified = false` — including via overlay — means the KV
/// buffer IS VRAM-resident, so capacity math must charge it in full).
/// Single source of truth shared by rule 12b's default-on branch and the
/// gpu-offload resolver so the two can never disagree.
fn kv_unified_emitted(input: &ProfileInput<'_>) -> bool {
    kv_unified_for(input.config, input.model_name, input.supported_flags)
}

/// Same decision as [`kv_unified_emitted`], callable outside profile
/// compilation: the per-request `num_ctx` preflight (gateway) and the
/// `coreside` planner need to know whether the spawn will carry
/// `--kv-unified` (shared single KV buffer) for argv-shape decisions.
/// Capacity math itself is lane-independent now — the KV pool is
/// device-backed either way. Callers without a compiled `ProfileInput`
/// supply the active engine's manifest flag set directly.
#[must_use]
pub fn kv_unified_for(
    config: &Config,
    model_name: &str,
    supported_flags: &BTreeSet<String>,
) -> bool {
    config.effective_kv_unified(model_name) != Some(false)
        && supported_flags.contains("--kv-unified")
}

/// True for every self-drafting n-gram spec mode ("ngram" is pallama
/// shorthand for upstream "ngram-simple").
fn is_ngram_spec(mode: &str) -> bool {
    matches!(
        mode,
        "ngram" | "ngram-map-k" | "ngram-map-k4v" | "ngram-mod" | "ngram-cache"
    )
}

/// Manifest-gated passthrough for config knobs: emit `flag` + `values`
/// only when the active engine advertises the flag; otherwise degrade
/// to a teaching warning (an older engine still serves with its own
/// defaults) instead of dying on an unknown flag at child boot.
fn push_gated(
    input: &ProfileInput<'_>,
    argv: &mut Vec<String>,
    warnings: &mut Vec<String>,
    key: &str,
    flag: &str,
    values: &[String],
) {
    if !input.supported_flags.contains(flag) {
        warnings.push(format!(
            "{key} skipped: engine {} lacks {flag}; run: pallama engine update",
            input.engine_tag
        ));
        return;
    }
    argv.push(flag.to_string());
    argv.extend(values.iter().cloned());
}

/// Rule 11: spec=auto draft pairing, opportunistic by design — an
/// unpulled or unsupported pair degrades to dense with a teaching
/// warning; manifest-gated emission when the draft is live.
fn push_spec_args(input: &ProfileInput<'_>, argv: &mut Vec<String>, warnings: &mut Vec<String>) {
    if let Some(pair) = crate::catalog::spec_pair_for(input.model_name) {
        if let Some(draft) = input.draft_path {
            // Self-draft guard: the draft row's registry name can
            // prefix-collide with its own main model (qwen3.5-9b-mtp
            // matches the qwen3.5-9b pair) — drafting from the main
            // weights would loop the same file through both roles.
            if draft == input.model_path {
                warnings.push(format!(
                    "spec=auto: catalog draft for {} resolved to the main model \
                         file itself; running dense",
                    input.model_name
                ));
                return;
            }
            if !input.supported_flags.contains("--spec-type")
                || !input
                    .spec_types
                    .iter()
                    .any(|t| t == pair.spec_type.as_str())
            {
                // Old engine + resolved pair: degrade to dense with a
                // teaching warning, never a fatal child boot.
                warnings.push(format!(
                    "spec=auto: catalog draft pair ({}) for {} is pulled but \
                         engine {} lacks it; running dense — run: pallama engine update",
                    pair.spec_type, input.model_name, input.engine_tag
                ));
                return;
            }
            // Capacity gate: the draft rides the SAME card as the main
            // model, and the pre-spawn census's free MiB predates the
            // main load — so the draft must fit alongside model + KV
            // floor + spawn overhead. Without this an 8 GiB card
            // (main 5.4 GiB + MTP draft 5.9 GiB) boot-OOMs instead of
            // serving dense (live-measured: draft-on-CPU is 2x slower,
            // so a partial-fit spawn is never the fallback).
            let draft_bytes = std::fs::metadata(draft).map_or(0, |m| m.len());
            let card_free_bytes: u64 = input
                .hardware
                .gpus
                .iter()
                .map(|g| g.free_mib.saturating_mul(1024 * 1024))
                .sum();
            let needed = input
                .model_bytes
                .saturating_add(draft_bytes)
                .saturating_add(KV_UNIFIED_VRAM_FLOOR_BYTES)
                .saturating_add(UNIFIED_SPAWN_OVERHEAD_BYTES);
            // Same predicate rule 4 consults (spec_draft_will_attach)
            // — keep the local math only for the warning numbers.
            if draft_bytes == 0 || needed > card_free_bytes {
                warnings.push(format!(
                    "spec=auto: draft {} ({} MiB) does not fit the picked card \
                         alongside {} (model {} MiB + KV floor + spawn overhead vs \
                         {} MiB free); running dense — speculation engages \
                         automatically on a card that fits both",
                    pair.spec_type,
                    draft_bytes / (1024 * 1024),
                    input.model_name,
                    input.model_bytes / (1024 * 1024),
                    card_free_bytes / (1024 * 1024)
                ));
                return;
            }
            argv.push("--spec-type".into());
            argv.push(pair.spec_type.clone());
            argv.push("--spec-draft-model".into());
            argv.push(draft.to_string());
            if input.supported_flags.contains("--spec-draft-n-max") {
                argv.push("--spec-draft-n-max".into());
                argv.push("3".into());
            }
        } else {
            // Opportunistic auto: an unpulled catalog draft degrades
            // to dense with a teaching warning — auto must never
            // refuse a spawn (hard errors belong to the explicit
            // typed modes, where the user asked for THAT drafter).
            // (`--spec-draft-hf` auto-download exists in b10840+ but
            // resolves to an empty path and the child exits fatally,
            // verified live 2026-09-07 — revisit if upstream fixes
            // draft-side HF resolution.)
            warnings.push(format!(
                "spec=auto: draft pair {} for {} is not pulled; running dense — run: \
                     pallama pull {} to enable speculation",
                pair.spec_type, input.model_name, pair.draft_repo
            ));
        }
    } else {
        tracing::info!(
            model = input.model_name,
            "profile: spec=auto found no draft pair for {} in the catalog; running dense",
            input.model_name
        );
    }
}

/// Router-preset INI generation. Upstream router mode (llama-server with
/// `--models-preset`, no `-m`) reads one INI: a `[*]` section of
/// server-global options plus one `[<model>]` section per model, keys =
/// long-flag names without dashes, booleans as bare `true`. Pallama
/// compiles each model's normal profile (same rules, same manifest gate)
/// and translates: model-scoped flags land in the model section, the rest
/// in the global section. Host/port/alias are reserved (the router
/// assigns them per model child).
///
/// Keys that live WITH a model child (verified against upstream arg.cpp:
/// every flag below is read per-context when a model loads).
const ROUTER_MODEL_KEYS: &[&str] = &[
    "model",
    "ctx-size",
    "threads",
    "threads-batch",
    "gpu-layers",
    "flash-attn",
    "cache-reuse",
    "cache-type-k",
    "cache-type-v",
    "cpu-moe",
    "n-cpu-moe",
    "n-cpu-ffn",
    "override-tensor",
    "rope-scaling",
    "rope-scale",
    "batch-size",
    "ubatch-size",
    "parallel",
    "rpc",
    "lora",
    "lora-scaled",
    "spec-type",
    "spec-draft-model",
    "spec-draft-n-max",
    "lookup-cache-dynamic",
    "reasoning-format",
    "mmproj",
    // Wire-wave model-scoped keys (arg.cpp families: draft/spec/mtmd/
    // reasoning/context all live with the model context).
    "spec-draft-cpu-range",
    "spec-draft-cpu-strict",
    "spec-draft-device",
    "spec-draft-ngl",
    "spec-draft-threads",
    "spec-draft-p-min",
    "spec-draft-p-split",
    "spec-draft-poll",
    "spec-draft-prio",
    "spec-ngram-simple-size-m",
    "spec-ngram-simple-size-n",
    "spec-ngram-simple-min-hits",
    "spec-ngram-map-k-size-m",
    "spec-ngram-map-k-size-n",
    "spec-ngram-map-k-min-hits",
    "spec-ngram-map-k4v-size-m",
    "spec-ngram-map-k4v-size-n",
    "spec-ngram-map-k4v-min-hits",
    "spec-ngram-mod-n-match",
    "spec-ngram-mod-n-max",
    "spec-ngram-mod-n-min",
    "spm-infill",
    "reasoning-budget",
    "reasoning-budget-message",
    "reasoning-effort",
    "reasoning-preserve",
    "image-max-tokens",
    "image-min-tokens",
    "mtmd-batch-max-tokens",
    "mmproj-device",
    // F110: per-context knobs the normal profile emits but the router
    // INI silently dropped — device placement, chat templating, sampler
    // defaults, sampler chain, warmup and mmproj offload all live with
    // the model context in upstream arg.cpp.
    "device",
    "chat-template",
    "chat-template-file",
    "samplers",
    "no-warmup",
    "no-mmproj-offload",
    "temp",
    "top-k",
    "top-p",
    "min-p",
    "top-n-sigma",
    "typical-p",
    "repeat-penalty",
    "repeat-last-n",
    "presence-penalty",
    "frequency-penalty",
    "dry-multiplier",
    "dry-base",
    "dry-allowed-length",
    "dry-penalty-last-n",
    "xtc-probability",
    "xtc-threshold",
    "mirostat",
    "seed",
    "embd-normalize",
    "yarn-orig-ctx",
    "yarn-ext-factor",
    "yarn-attn-factor",
    "yarn-beta-fast",
    "yarn-beta-slow",
    "keep",
    "override-kv",
    "control-vector",
    "control-vector-scaled",
    "control-vector-layer-range",
    // Per-model paths (each model's profile bakes its own dir):
    "slot-save-path",
];

/// Reserved keys the router assigns itself; never emitted.
const ROUTER_RESERVED_KEYS: &[&str] = &["host", "port", "alias"];

/// Serialize one `flag value` pair as an INI line (`flag = value`, dashes
/// stripped from the flag name).
fn ini_line(flag: &str, value: &str) -> String {
    format!("{} = {}", flag.trim_start_matches('-'), value)
}

/// Generate the router preset INI from compiled per-model profiles.
/// `models` = (model name, compiled profile argv) pairs; `global` = one
/// compiled profile whose model-scoped flags are ignored (only its
/// server-level flags seed the `[*]` section).
#[must_use]
pub fn generate_router_preset(models: &[(String, Vec<String>)], global: &[String]) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();

    // Global section: server-level knobs only.
    out.push_str("[*]\n");
    let mut seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    let mut i = 0;
    while i < global.len() {
        let flag = global[i].as_str();
        let bare_g = flag.trim_start_matches('-');
        if flag.starts_with("--")
            && !ROUTER_MODEL_KEYS.contains(&bare_g)
            && !ROUTER_RESERVED_KEYS.contains(&bare_g)
        {
            let value = global.get(i + 1).filter(|v| !v.starts_with("--"));
            let key = bare_g.to_string();
            if seen.insert(key.clone()) {
                match value {
                    Some(v) => {
                        out.push_str(&ini_line(&key, v));
                        out.push('\n');
                        i += 2;
                        continue;
                    }
                    None => {
                        let _ = writeln!(out, "{key} = true");
                    }
                }
            }
        }
        i += 1;
    }

    // Per-model sections.
    for (name, argv) in models {
        out.push('\n');
        let _ = writeln!(out, "[{name}]");
        let mut i = 0;
        while i < argv.len() {
            let flag = argv[i].as_str();
            let bare = flag.trim_start_matches('-');
            let is_long = flag.starts_with("--");
            // -m/-np/-b/-mm short flags translate to their long names.
            let (key, short_value) = match flag {
                "-m" => ("model", true),
                "-np" => ("parallel", true),
                "-b" => ("batch-size", true),
                "-mm" => ("mmproj", true),
                _ => (bare, false),
            };
            if (is_long || short_value) && ROUTER_MODEL_KEYS.contains(&key) {
                let value = argv
                    .get(i + 1)
                    .filter(|v| !v.starts_with("--") || short_value);
                if let Some(v) = value {
                    out.push_str(&ini_line(key, v));
                    out.push('\n');
                    i += 2;
                    continue;
                }
                if is_long {
                    let _ = writeln!(out, "{key} = true");
                }
            }
            i += 1;
        }
    }
    out
}

/// Filesystem-safe form of a model name for cache files (HF repo names
/// may contain `/` and `:`). Injective up to a 64-bit hash collision:
/// a discriminator (FNV-1a of the ORIGINAL name) is appended so `a/b`
/// and `a_b` cannot collide into one speccache/session file (F112).
/// FNV-1a is inline because `DefaultHasher` is not stable across
/// runs/versions — these names must survive restarts.
#[must_use]
pub fn path_safe(name: &str) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for b in name.as_bytes() {
        hash ^= u64::from(*b);
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    let sanitized: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    format!("{sanitized}-{hash:016x}")
}

/// Rust's f64 Display already prints the minimal form (2.0 -> "2",
/// 1.5 -> "1.5"), which is exactly the flag format upstream parses.
fn format_trimmed(v: f64) -> String {
    format!("{v}")
}

/// Extract the value following a `-mm`/`--mmproj` token in `extra_args` so
/// the projector rides along when users configure it per-model.
fn find_mmproj_arg(extra: &[String]) -> Option<&str> {
    extra
        .iter()
        .position(|a| a == "-mm" || a == "--mmproj")
        .and_then(|i| extra.get(i + 1))
        .map(String::as_str)
}

/// Dual-resolved projector policy: per-model override > global knob >
/// Lazy default. Shared by rule 19 (argv emission) and the supervisor's
/// vision-respawn check so both sides always agree on whether a running
/// instance can serve images.
#[must_use]
pub fn mmproj_policy_effective(
    config: &Config,
    model: &str,
    overlay: &ModelOverride,
) -> MmprojPolicy {
    MmprojPolicy::effective(
        overlay.mmproj.or_else(|| config.overlay_for(model).mmproj),
        config.mmproj_policy,
    )
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use std::sync::LazyLock;
    const MIB: u64 = 1024 * 1024;
    use super::*;
    use crate::config::SamplerDefaults;
    use crate::gguf::GgufMeta;
    use crate::hardware::GpuInfo;

    #[allow(clippy::too_many_lines)] // one flat flag inventory, not logic
    fn full_flags() -> BTreeSet<String> {
        [
            "-m",
            "--host",
            "--port",
            "--alias",
            "--jinja",
            "--device",
            "--lookup-cache-static",
            "--lookup-cache-dynamic",
            "--metrics",
            "--flash-attn",
            "--ctx-size",
            "--threads",
            "--gpu-layers",
            "--cache-reuse",
            "--cache-idle-slots",
            "--no-cache-idle-slots",
            "--kv-unified",
            "--no-kv-unified",
            "--kv-unified-per-slot",
            "--swa-full",
            "--ctx-checkpoints",
            "--no-kv-offload",
            // "--load-mode" deliberately ABSENT: the shared fixture set
            // keeps argv-pinning tests stable while the mlock auto
            // policy exists; dedicated load-mode tests opt in.
            "--cache-type-k",
            "--cache-type-v",
            "--cpu-moe",
            "--sleep-idle-seconds",
            "-np",
            "--rpc",
            "--lora",
            "--lora-scaled",
            "--spec-type",
            "--spec-draft-model",
            "--spec-draft-n-max",
            "--spec-draft-backend-sampling",
            "--lazy-mode",
            "--tools",
            "--tools-runtime",
            "--mcp-servers-config",
            "--mcp-servers-json",
            "--cache-ram",
            "-mm",
            "--mmproj",
            "--ubatch-size",
            "--batch-size",
            "--threads-batch",
            "--main-gpu",
            "--split-mode",
            "--tensor-split",
            "--cpu-range",
            "--poll",
            "--reasoning-format",
            "--slot-save-path",
            "--rope-scaling",
            "--rope-scale",
            "--n-cpu-moe",
            "--n-cpu-ffn",
            "--override-tensor",
            "--agent",
            "--slot-prompt-similarity",
            "--chat-template",
            "--chat-template-file",
            "--temp",
            "--top-k",
            "--top-p",
            "--min-p",
            "--top-n-sigma",
            "--typical-p",
            "--repeat-penalty",
            "--repeat-last-n",
            "--presence-penalty",
            "--frequency-penalty",
            "--dry-multiplier",
            "--dry-base",
            "--dry-allowed-length",
            "--dry-penalty-last-n",
            "--xtc-probability",
            "--xtc-threshold",
            "--mirostat",
            "--seed",
            "--no-warmup",
            "--samplers",
            "--embeddings",
            "--pooling",
            // Modern-engine superset (b10896): the spec-draft placement
            // family, reasoning control, scheduling extras, and
            // multimodal knobs. Kept in the canonical fixture so every
            // manifest gate has a fully-flagged engine to test against
            // (the wire-everything wave folded in here).
            "--spec-draft-cpu-range",
            "--spec-draft-cpu-strict",
            "--spec-draft-device",
            "--spec-draft-ngl",
            "--spec-draft-threads",
            "--spec-draft-p-min",
            "--spec-draft-p-split",
            "--spec-draft-poll",
            "--spec-draft-poll-batch",
            "--spec-draft-prio",
            "--spec-draft-prio-batch",
            "--spec-draft-cpu-strict-batch",
            "--spec-draft-threads-batch",
            "--spec-draft-type-k",
            "--spec-draft-type-v",
            "--spec-draft-override-tensor",
            "--spec-draft-n-cpu-moe",
            "--spec-draft-cpu-moe",
            "--no-spec-draft-backend-sampling",
            "--adaptive-decay",
            "--adaptive-target",
            "--reasoning-budget",
            "--reasoning-budget-message",
            "--reasoning",
            "--reasoning-effort",
            "--reasoning-preserve",
            "--no-reasoning-preserve",
            "--image-max-tokens",
            "--image-min-tokens",
            "--mtmd-batch-max-tokens",
            "--no-mmproj-offload",
            "--no-mmproj-auto",
            "--mmproj-device",
            "--embd-normalize",
            "--cpu-strict",
            "--prio",
            "--prio-batch",
            "--poll-batch",
            "--threads-http",
            "--spec-ngram-simple-size-m",
            "--spec-ngram-simple-size-n",
            "--spec-ngram-simple-min-hits",
            "--spec-ngram-map-k-size-m",
            "--spec-ngram-map-k-size-n",
            "--spec-ngram-map-k-min-hits",
            "--spec-ngram-map-k4v-size-m",
            "--spec-ngram-map-k4v-size-n",
            "--spec-ngram-map-k4v-min-hits",
            "--spec-ngram-mod-n-match",
            "--spec-ngram-mod-n-max",
            "--spec-ngram-mod-n-min",
            "--spm-infill",
            "--yarn-orig-ctx",
            "--yarn-ext-factor",
            "--yarn-attn-factor",
            "--yarn-beta-fast",
            "--yarn-beta-slow",
            "--no-repack",
            "--no-host",
            "--op-offload",
            "--no-op-offload",
            "--keep",
            "--override-kv",
            "--control-vector",
            "--control-vector-scaled",
            "--control-vector-layer-range",
        ]
        .iter()
        .map(|f| (*f).to_string())
        .collect()
    }
    fn gpu_hw(vram_mib: u64, ram_mib: u64, cores: u32) -> Hardware {
        Hardware {
            physical_cores: cores,
            total_ram_mib: ram_mib,
            gpus: vec![GpuInfo {
                name: "RTX".into(),
                description: "NVIDIA CUDA".into(),
                total_mib: vram_mib,
                free_mib: vram_mib,
            }],
        }
    }

    fn meta() -> GgufMeta {
        GgufMeta {
            architecture: "qwen3".into(),
            name: Some("x".into()),
            block_count: Some(28),
            context_length: Some(40_960),
            expert_count: None,
            head_count: Some(16),
            head_count_kv: Some(8),
            embedding_length: Some(1024),
            head_dim: Some(64),
            key_length: None,
            value_length: None,
            sliding_window: None,
            sliding_window_per_layer: None,
            full_attention_interval: None,
            recurrent_layers: None,
            quantized_by: None,
            general_version: None,
            pooling_type: None,
            chat_template: None,
            mtp_layers: None,
            num_loops: None,
        }
    }

    fn meta_with_mtp(layers: u64) -> GgufMeta {
        GgufMeta {
            mtp_layers: Some(layers),
            ..meta()
        }
    }

    /// Materialize a real (empty) draft file so compile's stale-row
    /// existence check sees a live path — the fixture must not lie
    /// about freshness w more than the store may.
    fn draft_file(tag: &str) -> String {
        let p =
            std::env::temp_dir().join(format!("pallama-draft-{tag}-{}.gguf", std::process::id()));
        std::fs::write(&p, b"gguf").unwrap();
        p.to_string_lossy().into_owned()
    }

    fn input<'a>(
        gguf: &'a GgufMeta,
        hw: &'a Hardware,
        cfg: &'a Config,
        flags: &'a BTreeSet<String>,
    ) -> ProfileInput<'a> {
        input_with_spec(gguf, hw, cfg, flags, &[])
    }

    fn input_with_spec<'a>(
        gguf: &'a GgufMeta,
        hw: &'a Hardware,
        cfg: &'a Config,
        flags: &'a BTreeSet<String>,
        spec_types: &'a [String],
    ) -> ProfileInput<'a> {
        ProfileInput {
            engine_kind: crate::engine_kind::EngineKind::default(),
            model_name: "qwen3-8b",
            instance_key: "qwen3-8b",
            model_path: "/models/qwen3-8b.gguf",
            model_bytes: 5_000 * MIB,
            meta: ModelMeta::Gguf(gguf),
            hardware: hw,
            config: cfg,
            overlay: &DEFAULT_OVERLAY,
            loras: &[],
            draft_path: None,
            draft_gguf: None,
            mmproj_path: None,
            mmproj_force: false,
            engine_tag: "b-test",
            supported_flags: flags,
            spec_types,
            endpoint: Endpoint::Tcp {
                host: "127.0.0.1".into(),
                port: 12345,
            },
            data_dir: "/tmp/pallama-test-data",
            cache_hit_rate: None,
            resident_ram_mib: 0,
            device_hint: None,
            engine_census: hw.gpus.clone(),
            sibling_devices: Vec::new(),
            auto_tensor_split: None,
        }
    }

    static MTP_SPEC_TYPES: LazyLock<Vec<String>> = LazyLock::new(|| vec!["draft-mtp".to_string()]);

    static EAGLE3_SPEC_TYPES: LazyLock<Vec<String>> =
        LazyLock::new(|| vec!["draft-eagle3".to_string()]);

    static DFLASH_SPEC_TYPES: LazyLock<Vec<String>> =
        LazyLock::new(|| vec!["draft-dflash".to_string()]);

    /// Same as `input_with_spec` but with a controllable model name — the
    /// default "qwen3-8b" matches the catalog spec-pair prefix, which is
    /// wrong for tests that need the no-pair path.
    fn input_named_with_spec<'a>(
        model_name: &'a str,
        gguf: &'a GgufMeta,
        hw: &'a Hardware,
        cfg: &'a Config,
        flags: &'a BTreeSet<String>,
        spec_types: &'a [String],
    ) -> ProfileInput<'a> {
        let mut i = input_with_spec(gguf, hw, cfg, flags, spec_types);
        i.model_name = model_name;
        i.instance_key = model_name;
        i
    }

    static ALL_FLAGS: LazyLock<BTreeSet<String>> = LazyLock::new(full_flags);
    static DEFAULT_OVERLAY: ModelOverride = ModelOverride {
        cpu_ffn_n: None,
        ctx: None,
        slots: None,
        spec: None,
        lazy_mode: None,
        loras: None,
        extra_args: None,
        cache_type: None,
        kv_unified: None,
        ctx_extend: None,
        cpu_moe_n: None,
        override_tensor: None,
        devices: None,
        mistralrs: None,
        engine: None,
        warmup: None,
        reasoning_budget: None,
        reasoning_effort: None,
        reasoning: None,
        replicas: None,
        pin: None,
        chat_template: None,
        chat_template_file: None,
        sampler_defaults: None,
        spm_infill: None,
        late_chunking: None,
        rpc_servers: None,
        deterministic: None,
        mmproj: None,
        sglang: None,
    };

    #[test]
    fn unit__late_chunking__forces_embeddings_pooling_none_over_gguf() {
        let cfg = Config::default();
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        // GGUF says pooling mean (embedding-class model); the R1 override
        // must win with `none` so the gateway gets a per-token matrix.
        let mut g = g;
        g.pooling_type = Some(1);
        let late = ModelOverride {
            late_chunking: Some(true),
            ..ModelOverride::default()
        };
        let mut inp = input(&g, &hw, &cfg, &ALL_FLAGS);
        inp.overlay = &late;
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(p
            .argv
            .windows(3)
            .any(|w| w[0] == "--embeddings" && w[1] == "--pooling" && w[2] == "none"));
        // no duplicate pooling flag from the GGUF branch
        assert_eq!(p.argv.iter().filter(|a| *a == "--pooling").count(), 1);
        // embedding docs must fit one ubatch: default floors at the
        // upstream preset scale (2048), explicit config.ubatch_size wins
        assert!(p.argv.windows(2).any(
            |w| w[0] == "--ubatch-size" && w[1].parse::<u32>() == Ok(LATE_CHUNK_UBATCH_DEFAULT)
        ));
        // late off + GGUF pooling -> the metadata branch still wins
        let mut g2 = meta();
        g2.pooling_type = Some(1);
        let p2 = compile(
            &input(&g2, &hw, &cfg, &ALL_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(p2
            .argv
            .windows(2)
            .any(|w| w[0] == "--pooling" && w[1] == "mean"));
    }

    #[test]
    fn unit__profile_base_rules__emitted_in_order() {
        let cfg = Config {
            spec: "off".into(), // purpose-scoped: base rules, not the spec lane
            ..Config::default()
        };
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        let p = compile(
            &input(&g, &hw, &cfg, &ALL_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        // Rule 1: model, endpoint, alias
        assert_eq!(p.argv[0], "-m");
        assert_eq!(p.argv[1], "/models/qwen3-8b.gguf");
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--host" && w[1] == "127.0.0.1"));
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--port" && w[1] == "12345"));
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--alias" && w[1] == "qwen3-8b"));
        // Rule 2: jinja + metrics + flash-attn auto + ctx — default slots=0
        // auto earns np=2 (train 40960 / base 16384), scaling the total.
        assert!(p.argv.contains(&"--jinja".to_string()));
        assert!(p.argv.contains(&"--metrics".to_string()));
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--flash-attn" && w[1] == "auto"));
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--ctx-size" && w[1] == "32768"));
        // Rule 3: threads = physical cores
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--threads" && w[1] == "8"));
        // Rule 4: comfortable full fit (5 GB + 469 MB KV <= 85% of 12 GB)
        // -> pinned 999, labeled "full" in ps
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--gpu-layers" && w[1] == "999"));
        assert_eq!(p.gpu, "full");
        // Rule 5: cache-reuse defaults OFF (native slot cache covers
        // identical prefixes; --cache-reuse measured +0.6s cold)
        assert!(!p.argv.contains(&"--cache-reuse".to_string()));
        // Rule 6: KV at the scaled total = 2*28*8*64*32768*2 = 938MB;
        // +5GB < 0.9*12GB -> NO kv quant
        assert!(!p.argv.contains(&"--cache-type-k".to_string()));
        // Rule 8: sleep (GPU present)
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--sleep-idle-seconds" && w[1] == "300"));
        // Rule 9: default slots=0 = pallama auto -> np 2 on this fixture.
        assert!(p.argv.windows(2).any(|w| w[0] == "-np" && w[1] == "2"));
        // Auto-slot rationale is journal-only now.
        assert!(!p.warnings.iter().any(|w| w.contains("slots auto")));
        // Rule 12: cache-ram default 8192
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--cache-ram" && w[1] == "8192"));
        // Rule 15: sessions dir default-on when the engine supports it
        assert!(p.argv.windows(2).any(|w| w[0] == "--slot-save-path"));
        assert_eq!(p.ctx, 16384, "Profile.ctx reports the per-slot ctx");
        // A default battery rides NO warnings: auto-decision rationale
        // lives in the journal, warnings are for user-input divergence.
        assert_eq!(
            p.warnings.len(),
            0,
            "unexpected extra warnings: {:?}",
            p.warnings
        );
    }

    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn unit__kv_layout__explicit_and_auto_unified() {
        let hw = gpu_hw(24_000, 64_000, 8);
        let mut cfg = Config::default();
        cfg.kv_unified = Some(true);
        let p = compile(
            &input(&GgufMeta::default(), &hw, &cfg, &ALL_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(p.argv.contains(&"--kv-unified".to_string()));
        assert!(!p.argv.contains(&"--no-kv-unified".to_string()));

        let mut cfg = Config::default();
        cfg.kv_unified = Some(false);
        let p = compile(
            &input(&GgufMeta::default(), &hw, &cfg, &ALL_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(p.argv.contains(&"--no-kv-unified".to_string()));

        // Auto: unified is the DEFAULT whenever the engine supports it
        // (single- and multi-slot alike — K-shift reuse rides it).
        let cfg = Config::default();
        let p = compile(
            &input(&GgufMeta::default(), &hw, &cfg, &ALL_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(p.argv.contains(&"--kv-unified".to_string()));
        // Unsupported engine: default path emits nothing and adds no
        // warning about it (only EXPLICIT knobs warn).
        let mut no_kvu = full_flags();
        no_kvu.remove("--kv-unified");
        let p = compile(
            &input(&GgufMeta::default(), &hw, &Config::default(), &no_kvu),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(!p.argv.contains(&"--kv-unified".to_string()));
        assert!(!p.warnings.iter().any(|w| w.contains("kv_unified")));
    }

    #[test]
    fn unit__gpu_offload__unified_tight_fit_leaves_fitter_band() {
        // Classic-attention meta: kv @16384 = 8*(64+64)*2*28*16384 = 896 MiB.
        // VRAM 6400 MiB: 5000+896 = 5896 > 85% (5440) — device truth: the
        // unified pool is VRAM-resident too, so this tight fit goes to the
        // engine's live fitter band (may CPU-split the last layers)
        // instead of the old floor-accounted 999 pin. That pin was the
        // storm class: it disabled the fitter ("already set by user to
        // 999, abort") while the true device demand OOM'd.
        let hw = gpu_hw(6_400, 64_000, 8);
        let p = compile(
            &input(&meta(), &hw, &Config::default(), &ALL_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--gpu-layers" && w[1] == "auto"));
        assert_eq!(p.gpu, "auto");
        assert!(p.argv.contains(&"--kv-unified".to_string()));
        assert!(!p
            .warnings
            .iter()
            .any(|w| w.contains("unified-KV accounting")));
    }

    #[test]
    fn unit__gpu_offload__engine_without_unified_keeps_auto() {
        // Same tight fit, engine manifest lacks --kv-unified: full charge,
        // auto band, and no floor accounting note (nothing was discounted).
        let hw = gpu_hw(6_400, 64_000, 8);
        let mut no_kvu = full_flags();
        no_kvu.remove("--kv-unified");
        let p = compile(
            &input(&meta(), &hw, &Config::default(), &no_kvu),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--gpu-layers" && w[1] == "auto"));
        assert_eq!(p.gpu, "auto");
        assert!(!p.warnings.iter().any(|w| w.contains("unified-KV")));
    }

    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn unit__gpu_offload__explicit_kv_unified_false_charges_full_kv() {
        // Opt-out: --no-kv-unified means the KV buffer IS VRAM-resident,
        // so the pin math must charge all 896 MiB (auto band on 6700 MiB).
        let hw = gpu_hw(6_400, 64_000, 8);
        let mut cfg = Config::default();
        cfg.kv_unified = Some(false);
        let p = compile(
            &input(&meta(), &hw, &cfg, &ALL_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(p.argv.contains(&"--no-kv-unified".to_string()));
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--gpu-layers" && w[1] == "auto"));
        assert!(!p.warnings.iter().any(|w| w.contains("unified-KV")));
    }

    #[test]
    fn unit__rule5__cache_reuse_skipped_with_mmproj() {
        // Upstream disables cache_reuse with a multimodal projector: the
        // flag must not ride VL argv, and the skip must be said aloud.
        let hw = gpu_hw(24_000, 64_000, 8);
        // cache_reuse 256 explicit: rule 5 contract (default is 0 now).
        let cfg = Config {
            cache_reuse: 256,
            ..Config::default()
        };
        let m = meta();
        let mut i = input(&m, &hw, &cfg, &ALL_FLAGS);
        i.mmproj_path = Some("/nonexistent/mmproj-F16.gguf");
        let p = compile(&i, &TuningOverrides::default()).unwrap();
        assert!(!p.argv.contains(&"--cache-reuse".to_string()));
        assert!(
            p.warnings.iter().any(|w| w.contains("multimodal")),
            "skip must warn: {:?}",
            p.warnings
        );
    }

    #[test]
    fn unit__rule19__mmproj_suppress_knob_and_extra_args_precedence() {
        // mmproj=skip spawns text-only (saves the projector cold read +
        // VL init, measured ~875 MiB file / 1126 MiB VRAM / ~3.9 s cold
        // TTFT on the 9B VL row) and must say so; mmproj=lazy (the
        // default) also spawns text-only but promises the @vision
        // respawn; an explicit extra_args -mm still wins (rule 19's
        // first branch) without a double warning; mmproj_force
        // (@vision respawn / router preset) attaches regardless.
        let hw = gpu_hw(24_000, 64_000, 8);
        let g = meta();
        let mmp = "/nonexistent/mmproj-F16.gguf";
        let ovr = |mo: ModelOverride, cfg: Config| Config {
            model_overrides: std::collections::BTreeMap::from([("qwen3-8b".into(), mo)]),
            ..cfg
        };
        let base = Config::default();

        // 1. Skip: drop -mm + suppress warning
        let skip = ovr(
            ModelOverride {
                mmproj: Some(MmprojPolicy::Skip),
                ..ModelOverride::default()
            },
            base.clone(),
        );
        let mut i = input(&g, &hw, &skip, &ALL_FLAGS);
        i.mmproj_path = Some(mmp);
        let p = compile(&i, &TuningOverrides::default()).unwrap();
        assert!(
            !p.argv.iter().any(|a| a == "-mm"),
            "skip knob must drop -mm: {:?}",
            p.argv
        );
        assert!(
            p.warnings.iter().any(|w| w.contains("mmproj suppressed")),
            "skip must warn: {:?}",
            p.warnings
        );

        // 2. Lazy (default, no override, no global): drop -mm + lazy warning
        let mut i2 = input(&g, &hw, &base, &ALL_FLAGS);
        i2.mmproj_path = Some(mmp);
        let p2 = compile(&i2, &TuningOverrides::default()).unwrap();
        assert!(
            !p2.argv.iter().any(|a| a == "-mm"),
            "lazy default must spawn text-only: {:?}",
            p2.argv
        );
        assert!(
            p2.warnings.iter().any(|w| w.contains("mmproj lazy")),
            "lazy must warn: {:?}",
            p2.warnings
        );

        // 3. extra_args -mm beats the lazy default — attach, no lazy warning
        let explicit = ovr(
            ModelOverride {
                extra_args: Some(vec!["-mm".into(), mmp.into()]),
                ..ModelOverride::default()
            },
            base.clone(),
        );
        let mut i3 = input(&g, &hw, &explicit, &ALL_FLAGS);
        i3.mmproj_path = Some(mmp);
        let p3 = compile(&i3, &TuningOverrides::default()).unwrap();
        assert!(p3.argv.iter().any(|a| a == "-mm"), "extra_args wins");
        assert!(
            !p3.warnings.iter().any(|w| w.contains("mmproj lazy")),
            "explicit -mm must not double-warn: {:?}",
            p3.warnings
        );

        // 4. mmproj_force (@vision respawn / router preset) attaches
        //    even under the lazy default
        let mut i4 = input(&g, &hw, &base, &ALL_FLAGS);
        i4.mmproj_path = Some(mmp);
        i4.mmproj_force = true;
        let p4 = compile(&i4, &TuningOverrides::default()).unwrap();
        assert!(
            p4.argv.iter().any(|a| a == "-mm"),
            "mmproj_force must attach: {:?}",
            p4.argv
        );
        assert!(
            !p4.warnings.iter().any(|w| w.contains("mmproj lazy")),
            "forced attach is not lazy: {:?}",
            p4.warnings
        );

        // 5. From<bool> spelling: true = Attach, false = Skip
        assert_eq!(MmprojPolicy::from(true), MmprojPolicy::Attach);
        assert_eq!(MmprojPolicy::from(false), MmprojPolicy::Skip);
    }

    #[test]
    fn unit__policy__mmproj_effective_precedence_overlay_global_lazy() {
        // overlay > global > Lazy default, resolved per-model
        let mo = ModelOverride {
            mmproj: Some(MmprojPolicy::Skip),
            ..ModelOverride::default()
        };
        let cfg = Config {
            model_overrides: std::collections::BTreeMap::from([("m".into(), mo)]),
            mmproj_policy: Some(MmprojPolicy::Attach),
            ..Config::default()
        };
        assert_eq!(
            mmproj_policy_effective(&cfg, "m", &cfg.overlay_for("m")),
            MmprojPolicy::Skip,
            "overlay wins"
        );
        assert_eq!(
            mmproj_policy_effective(&cfg, "other", &cfg.overlay_for("other")),
            MmprojPolicy::Attach,
            "global serves models without an overlay"
        );
        let bare = Config::default();
        assert_eq!(
            mmproj_policy_effective(&bare, "other", &bare.overlay_for("other")),
            MmprojPolicy::Lazy,
            "None everywhere = Lazy default"
        );
    }

    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn unit__load_mode_auto__mlock_when_weights_fit_ram_share() {
        // Elim-sweep contract (2026-09-12): eager mlock page-in measured
        // ~0.7s faster to first token than lazy mmap faults, so the
        // auto policy pins when weights fit a 40% RAM share.
        let mut flags = full_flags();
        flags.insert("--load-mode".to_string());
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        // Fixture default model_bytes = 5000 MiB = 15.6% of 32 GiB.
        let p = compile(
            &input(&g, &hw, &Config::default(), &flags),
            &TuningOverrides::default(),
        )
        .unwrap();
        let i = p.argv.iter().position(|a| a == "--load-mode").unwrap();
        assert_eq!(p.argv[i + 1], "mlock");
        // Decision rationale lives in the journal, never the warning vec.
        assert!(
            !p.warnings
                .iter()
                .any(|w| w.contains("load-mode mlock auto")),
            "{:?}",
            p.warnings
        );

        // Over the 40% share: no flag, no warning (plain mmap default).
        let cfg_big = Config::default();
        let mut big = input(&g, &hw, &cfg_big, &flags);
        big.model_bytes = 20_000 * MIB;
        let p = compile(&big, &TuningOverrides::default()).unwrap();
        assert!(!p.argv.contains(&"--load-mode".to_string()));
        assert!(
            !p.warnings.iter().any(|w| w.contains("load-mode")),
            "{:?}",
            p.warnings
        );

        // Resident siblings charge the same share: a churn replacement
        // whose dying predecessor's pins overlap must NOT re-pin.
        let cfg_res = Config::default();
        let mut res = input(&g, &hw, &cfg_res, &flags);
        res.resident_ram_mib = 9_000;
        let p = compile(&res, &TuningOverrides::default()).unwrap();
        assert!(
            !p.argv.contains(&"--load-mode".to_string()),
            "resident charge must suppress mlock: {:?}",
            p.argv
        );

        // Explicit load_mode always wins over the auto policy.
        let mut cfg = Config::default();
        cfg.load_mode = "mmap".into();
        let p = compile(&input(&g, &hw, &cfg, &flags), &TuningOverrides::default()).unwrap();
        let i = p.argv.iter().position(|a| a == "--load-mode").unwrap();
        assert_eq!(p.argv[i + 1], "mmap");
    }

    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn unit__kv_layout__aux_flags_and_gating() {
        let hw = gpu_hw(24_000, 64_000, 8);
        let mut cfg = Config::default();
        cfg.kv_unified_per_slot = 4096;
        cfg.swa_full = true;
        cfg.ctx_checkpoints = 8;
        cfg.no_kv_offload = true;
        cfg.load_mode = "mlock".into();
        let mut with_load_mode = full_flags();
        with_load_mode.insert("--load-mode".to_string());
        let p = compile(
            &input(&GgufMeta::default(), &hw, &cfg, &with_load_mode),
            &TuningOverrides::default(),
        )
        .unwrap();
        for pair in [
            ("--kv-unified-per-slot", "4096"),
            ("--swa-full", ""),
            ("--ctx-checkpoints", "8"),
            ("--no-kv-offload", ""),
            ("--load-mode", "mlock"),
        ] {
            if pair.1.is_empty() {
                assert!(p.argv.contains(&pair.0.to_string()), "missing {}", pair.0);
            } else {
                let i = p.argv.iter().position(|a| a == pair.0).unwrap();
                assert_eq!(p.argv[i + 1], pair.1);
            }
        }

        // Engine without the flags: explicit knobs warn, nothing emitted.
        let mut flags = full_flags();
        flags.remove("--kv-unified");
        flags.remove("--swa-full");
        flags.remove("--load-mode");
        let mut cfg = Config::default();
        cfg.kv_unified = Some(true);
        cfg.swa_full = true;
        cfg.load_mode = "mlock".into();
        let p = compile(
            &input(&GgufMeta::default(), &hw, &cfg, &flags),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(!p.argv.contains(&"--kv-unified".to_string()));
        assert!(p.warnings.iter().any(|w| w.contains("--kv-unified")));
        assert!(p.warnings.iter().any(|w| w.contains("--swa-full")));
        assert!(p.warnings.iter().any(|w| w.contains("--load-mode")));
    }

    #[test]
    fn unit__cache_ram_clamped_to_30pct_of_ram_on_small_boxes() {
        // Live case: 13 GiB laptop, default 8192 -> cap 4007 (30% of 13359).
        // Unclamped, the child RSS plateaus at 8.3 GiB and the box swap-thrashes.
        // The budget caps the weights mmap, so the 2a-bis floor lifts
        // 4007 -> 5957 ((5000 weights + 64 slack) over the 0.85 compute
        // share, inside the 60% guard): a sub-weights budget makes the
        // engine fitter CPU-split layers (measured 15.6 vs 39.9 t/s on a 9B).
        let cfg = Config::default();
        let hw = gpu_hw(12_000, 13_359, 8);
        let g = meta();
        let p = compile(
            &input(&g, &hw, &cfg, &ALL_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--cache-ram" && w[1] == "5957"));
        // Default budget meeting the adaptive cap is silent auto-tuning.
        assert!(p.warnings.iter().any(|w| w.contains("floored to 5957 MiB")));
        assert!(
            !p.warnings
                .iter()
                .any(|w| w.contains("cache_ram_mb 8192 clamped")),
            "{:?}",
            p.warnings
        );

        // A user-pinned budget getting clamped stays a warning.
        let cfg_pinned = Config {
            cache_ram_mb: 16384,
            ..Config::default()
        };
        let p2 = compile(
            &input(&g, &hw, &cfg_pinned, &ALL_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(p2
            .warnings
            .iter()
            .any(|w| w.contains("cache_ram_mb 16384 clamped to 4007")));
    }

    #[test]
    fn unit__cache_ram_zero_disables_flag_entirely() {
        let cfg = Config {
            cache_ram_mb: 0,
            ..Config::default()
        };
        let hw = gpu_hw(12_000, 13_359, 8);
        let g = meta();
        let p = compile(
            &input(&g, &hw, &cfg, &ALL_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(!p.argv.contains(&"--cache-ram".to_string()));
    }

    #[test]
    fn unit__latency_affinity_knobs__emitted_when_set_absent_by_default() {
        let g = meta();
        let hw = gpu_hw(12_000, 32_000, 8);
        let p = compile(
            &input(&g, &hw, &Config::default(), &ALL_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(!p.argv.contains(&"--cpu-range".to_string()));
        assert!(!p.argv.contains(&"--poll".to_string()));
        assert!(!p.argv.contains(&"--reasoning-format".to_string()));

        let cfg = Config {
            cpu_range: "0-15".into(),
            poll: 50,
            reasoning_format: "deepseek".into(),
            ..Config::default()
        };
        let p = compile(
            &input(&g, &hw, &cfg, &ALL_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--cpu-range" && w[1] == "0-15"));
        assert!(p.argv.windows(2).any(|w| w[0] == "--poll" && w[1] == "50"));
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--reasoning-format" && w[1] == "deepseek"));
    }

    #[test]
    fn unit__spec_ngram__emits_self_drafting_no_draft_model() {
        let cfg = Config {
            spec: "ngram".into(),
            slots: 1, // purpose-scoped: spec flags, not slot sizing
            ..Config::default()
        };
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        let p = compile(
            &input(&g, &hw, &cfg, &ALL_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--spec-type" && w[1] == "ngram-simple"));
        assert!(!p.argv.contains(&"--spec-draft-model".to_string()));
        // Rule 14: persistent lookup cache rides along with ngram mode.
        // Expected path derives via path_safe (FNV-suffixed since the
        // collision-proofing change) — never hardcode it.
        let lcache = format!(
            "/tmp/pallama-test-data/speccache/{}.lcache",
            path_safe("qwen3-8b")
        );
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--lookup-cache-dynamic" && w[1] == lcache));
        assert!(p.warnings.is_empty(), "{:?}", p.warnings);
    }

    #[test]
    fn unit__spec_mtp__emits_draft_mtp_when_advertised() {
        let cfg = Config {
            spec: "mtp".into(),
            slots: 1, // purpose-scoped: spec flags, not slot sizing
            ..Config::default()
        };
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        let p = compile(
            &input_with_spec(&g, &hw, &cfg, &ALL_FLAGS, &MTP_SPEC_TYPES),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--spec-type" && w[1] == "draft-mtp"));
        // The MTP head ships inside the GGUF — no draft model to name.
        assert!(!p.argv.contains(&"--spec-draft-model".to_string()));
        // n-gram-only lookup cache must not ride along (rule 14 gate).
        assert!(!p.argv.contains(&"--lookup-cache-dynamic".to_string()));
        // Headless GGUF under manual mtp: teaching warning fires.
        assert!(
            p.warnings.iter().any(|w| w.contains("no MTP head")),
            "{:?}",
            p.warnings
        );
    }

    #[test]
    fn unit__spec_mtp__engine_lacking_draft_mtp__hard_error() {
        let cfg = Config {
            spec: "mtp".into(),
            slots: 1,
            ..Config::default()
        };
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        // Engine advertises --spec-type but not draft-mtp among its values.
        let err = compile(
            &input_with_spec(&g, &hw, &cfg, &ALL_FLAGS, &[]),
            &TuningOverrides::default(),
        )
        .unwrap_err();
        assert!(
            err.contains("draft-mtp") && err.contains("pallama engine update"),
            "{err}"
        );
    }

    #[test]
    fn unit__spec_eagle3__emits_pair_and_draft_when_advertised() {
        let cfg = Config {
            spec: "eagle3".into(),
            slots: 1,
            ..Config::default()
        };
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        let mut inp =
            input_named_with_spec("qwen3-8b", &g, &hw, &cfg, &ALL_FLAGS, &EAGLE3_SPEC_TYPES);
        let draft = draft_file("eagle3");
        inp.draft_path = Some(&draft);
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--spec-type" && w[1] == "draft-eagle3"));
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--spec-draft-model" && w[1] == draft.as_str()));
        // Spec pair: gpu-layers is deliberately NOT pinned so the engine
        // live fitter owns the split (draft weights+KV are unplanned).
        assert!(
            !p.argv
                .windows(2)
                .any(|w| w[0] == "--gpu-layers" && w[1] == "999"),
            "spec pair must not pin gpu-layers: {:?}",
            p.argv
        );
        assert!(
            p.warnings
                .iter()
                .any(|w| w.contains("left to the engine live fitter")),
            "unpin warning required: {:?}",
            p.warnings
        );
    }

    #[test]
    fn unit__spec_draft__gpu_layers_unpinned_both_kv_modes() {
        // Classic-KV branch (kv_unified off) must unpin too: draft device
        // allocations are unplanned in BOTH accounting modes.
        let cfg = Config {
            spec: "eagle3".into(),
            slots: 1,
            kv_unified: Some(false),
            ..Config::default()
        };
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        let mut inp =
            input_named_with_spec("qwen3-8b", &g, &hw, &cfg, &ALL_FLAGS, &EAGLE3_SPEC_TYPES);
        let draft = draft_file("eagle3-classic");
        inp.draft_path = Some(&draft);
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(
            !p.argv
                .windows(2)
                .any(|w| w[0] == "--gpu-layers" && w[1] == "999"),
            "classic-KV spec pair must not pin gpu-layers: {:?}",
            p.argv
        );
        assert!(
            p.warnings
                .iter()
                .any(|w| w.contains("left to the engine live fitter")),
            "unpin warning required: {:?}",
            p.warnings
        );
    }

    #[test]
    fn unit__spec_eagle3__engine_lacking__hard_error() {
        let cfg = Config {
            spec: "eagle3".into(),
            slots: 1,
            ..Config::default()
        };
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        let err = compile(
            &input_named_with_spec("qwen3-8b", &g, &hw, &cfg, &ALL_FLAGS, &[]),
            &TuningOverrides::default(),
        )
        .unwrap_err();
        assert!(
            err.contains("draft-eagle3") && err.contains("pallama engine update"),
            "{err}"
        );
    }

    #[test]
    fn unit__spec_eagle3__unpulled_draft__hard_error() {
        let cfg = Config {
            spec: "eagle3".into(),
            slots: 1,
            ..Config::default()
        };
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        let err = compile(
            &input_named_with_spec("qwen3-8b", &g, &hw, &cfg, &ALL_FLAGS, &EAGLE3_SPEC_TYPES),
            &TuningOverrides::default(),
        )
        .unwrap_err();
        assert!(
            err.contains("pallama pull")
                && err.contains("williamliao/Qwen3-8B-EAGLE3-Speculator-GGUF"),
            "{err}"
        );
    }

    #[test]
    fn unit__spec_eagle3__no_pair__hard_error() {
        let cfg = Config {
            spec: "eagle3".into(),
            slots: 1,
            ..Config::default()
        };
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        let err = compile(
            &input_named_with_spec("llama3.2-3b", &g, &hw, &cfg, &ALL_FLAGS, &EAGLE3_SPEC_TYPES),
            &TuningOverrides::default(),
        )
        .unwrap_err();
        assert!(
            err.contains("no draft-eagle3 draft pair") && err.contains("llama3.2-3b"),
            "{err}"
        );
    }

    #[test]
    fn unit__spec_dflash__engine_lacking__hard_error() {
        let cfg = Config {
            spec: "dflash".into(),
            slots: 1,
            ..Config::default()
        };
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        let err = compile(
            &input_named_with_spec("qwen3-8b", &g, &hw, &cfg, &ALL_FLAGS, &EAGLE3_SPEC_TYPES),
            &TuningOverrides::default(),
        )
        .unwrap_err();
        assert!(
            err.contains("draft-dflash") && err.contains("pallama engine update"),
            "{err}"
        );
    }

    #[test]
    fn unit__spec_dspark__advertised_but_no_catalog_pair__hard_error() {
        let cfg = Config {
            spec: "dspark".into(),
            slots: 1,
            ..Config::default()
        };
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        let mut err = compile(
            &input_named_with_spec("qwen3-8b", &g, &hw, &cfg, &ALL_FLAGS, &DFLASH_SPEC_TYPES),
            &TuningOverrides::default(),
        )
        .unwrap_err();
        // dflash type advertised but dspark asked -> engine gate fires…
        assert!(err.contains("draft-dspark"), "{err}");
        // …and a dspark-advertising engine with no catalog pair for a
        // model outside the pair prefix -> pair gate.
        err = compile(
            &input_named_with_spec(
                "llama3.2-3b",
                &g,
                &hw,
                &cfg,
                &ALL_FLAGS,
                ["draft-dspark".to_string()].as_slice(),
            ),
            &TuningOverrides::default(),
        )
        .unwrap_err();
        assert!(err.contains("no draft-dspark draft pair"), "{err}");
    }

    #[test]
    fn unit__spec_auto__embedded_mtp_emits_draft_mtp_trio() {
        let cfg = Config {
            spec: "auto".into(),
            slots: 1,
            ..Config::default()
        };
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta_with_mtp(1);
        let p = compile(
            &input_with_spec(&g, &hw, &cfg, &ALL_FLAGS, &MTP_SPEC_TYPES),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--spec-type" && w[1] == "draft-mtp"));
        // n-max capped by the trained head count.
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--spec-draft-n-max" && w[1] == "1"));
        assert!(p
            .argv
            .contains(&"--spec-draft-backend-sampling".to_string()));
        // No external draft model — the head ships in the weights.
        assert!(!p.argv.contains(&"--spec-draft-model".to_string()));
        assert!(
            p.warnings.iter().any(|w| w.contains("MTP head")),
            "{:?}",
            p.warnings
        );
    }

    #[test]
    fn unit__spec_auto__mtp_n_max_capped_at_two_for_chained_heads() {
        let cfg = Config {
            spec: "auto".into(),
            slots: 1,
            ..Config::default()
        };
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta_with_mtp(3); // chain-heads arch with 3 trained steps
        let p = compile(
            &input_with_spec(&g, &hw, &cfg, &ALL_FLAGS, &MTP_SPEC_TYPES),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--spec-draft-n-max" && w[1] == "2"));
    }

    #[test]
    fn unit__spec_auto__mtp_engine_lacks_draft_mtp__warn_and_no_mtp_args() {
        let cfg = Config {
            spec: "auto".into(),
            slots: 1,
            ..Config::default()
        };
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta_with_mtp(1);
        let p = compile(
            &input_named_with_spec("llama3.2-3b", &g, &hw, &cfg, &ALL_FLAGS, &[]),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(!p.argv.contains(&"--spec-type".to_string()));
        assert!(
            p.warnings
                .iter()
                .any(|w| w.contains("lacks") && w.contains("pallama engine update")),
            "{:?}",
            p.warnings
        );
    }

    #[test]
    fn unit__spec_auto__no_mtp_gguf__catalog_path_unchanged() {
        let cfg = Config {
            spec: "auto".into(),
            slots: 1,
            ..Config::default()
        };
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta(); // no MTP head
        let p = compile(
            &input_named_with_spec("llama3.2-3b", &g, &hw, &cfg, &ALL_FLAGS, &MTP_SPEC_TYPES),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(!p
            .argv
            .windows(2)
            .any(|w| w[0] == "--spec-type" && w[1] == "draft-mtp"));
        // qwen3-8b has no catalog pair either: dense, no spec args at all.
        assert!(!p.argv.contains(&"--spec-type".to_string()));
    }

    #[test]
    fn unit__spec_auto_pair__engine_lacking_spec_type__warn_and_dense() {
        // R2-1: an old engine (no --spec-type) + resolved catalog pair
        // must degrade to dense with a teaching warning, not push spec
        // argv the child rejects at boot.
        let cfg = Config {
            spec: "auto".into(),
            slots: 1,
            ..Config::default()
        };
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        let mut flags = ALL_FLAGS.clone();
        flags.remove("--spec-type");
        flags.remove("--spec-draft-model");
        flags.remove("--spec-draft-n-max");
        let mut inp = input_named_with_spec("qwen3-8b", &g, &hw, &cfg, &flags, &[]);
        let draft = draft_file("auto-pair-old-engine");
        inp.draft_path = Some(&draft);
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(
            !p.argv.iter().any(|a| a == "--spec-type"),
            "no spec argv on a flagless engine: {:?}",
            p.argv
        );
        assert!(
            p.warnings
                .iter()
                .any(|w| w.contains("lacks it") && w.contains("pallama engine update")),
            "teaching warning required: {:?}",
            p.warnings
        );
    }

    #[test]
    fn unit__spec_mtp__manual_caps_n_max_at_head_layers() {
        // R2-4: manual spec="mtp" must cap --spec-draft-n-max at the
        // trained head count (min 2), matching the auto lane — the
        // upstream default of 3 would make a 1-2 layer head pay pure
        // verification overhead.
        let cfg = Config {
            spec: "mtp".into(),
            slots: 1,
            ..Config::default()
        };
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta_with_mtp(3);
        let p = compile(
            &input_with_spec(&g, &hw, &cfg, &ALL_FLAGS, &MTP_SPEC_TYPES),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--spec-draft-n-max" && w[1] == "2"));
    }

    #[test]
    fn unit__compile__stale_draft_row__hard_error() {
        // R2-6: a draft_path whose file is gone (stale store row) fails
        // at profile-compile with a re-pull instruction, not at child
        // boot with an opaque engine error.
        let cfg = Config {
            spec: "auto".into(),
            slots: 1,
            ..Config::default()
        };
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        let mut inp = input_named_with_spec("qwen3-8b", &g, &hw, &cfg, &ALL_FLAGS, &[]);
        let missing =
            std::env::temp_dir().join(format!("pallama-draft-gone-{}.gguf", std::process::id()));
        let missing = missing.to_string_lossy().into_owned();
        inp.draft_path = Some(&missing);
        let err = compile(&inp, &TuningOverrides::default()).unwrap_err();
        assert!(err.contains("stale") && err.contains("pull"), "{err}");
    }

    #[test]
    fn unit__section13__config_knobs_manifest_gated() {
        // R2-7 representative for the whole 13/13b/13c/12c family: a
        // configured knob on an engine without the flag warn-skips
        // (engine defaults serve on) instead of boot-failing the child.
        let cfg = Config {
            reasoning_budget: 4096,
            prio: 2,
            threads_http: 4,
            ..Config::default()
        };
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        let flags: BTreeSet<String> = ALL_FLAGS
            .iter()
            .filter(|f| {
                !f.starts_with("--reasoning-budget") && *f != "--prio" && *f != "--threads-http"
            })
            .cloned()
            .collect();
        let p = compile(&input(&g, &hw, &cfg, &flags), &TuningOverrides::default()).unwrap();
        assert!(!p.argv.iter().any(|a| a == "--reasoning-budget"));
        assert!(!p.argv.iter().any(|a| a == "--prio"));
        assert!(!p.argv.iter().any(|a| a == "--threads-http"));
        for key in ["reasoning_budget", "prio", "threads_http"] {
            assert!(
                p.warnings
                    .iter()
                    .any(|w| w.starts_with(&format!("{key} skipped"))),
                "{key} teach-skip missing: {:?}",
                p.warnings
            );
        }
        // Same config on a fully-flagged engine: all three emitted.
        let p2 = compile(
            &input(&g, &hw, &cfg, &ALL_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(p2
            .argv
            .windows(2)
            .any(|w| w[0] == "--reasoning-budget" && w[1] == "4096"));
        assert!(p2.argv.windows(2).any(|w| w[0] == "--prio" && w[1] == "2"));
        assert!(p2
            .argv
            .windows(2)
            .any(|w| w[0] == "--threads-http" && w[1] == "4"));
    }

    #[test]
    fn unit__spec_off__never_emits_spec_type_even_with_mtp_gguf() {
        let cfg = Config {
            spec: "off".into(),
            slots: 1,
            ..Config::default()
        };
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta(); // no MTP head
        let p = compile(
            &input_named_with_spec("llama3.2-3b", &g, &hw, &cfg, &ALL_FLAGS, &MTP_SPEC_TYPES),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(!p
            .argv
            .windows(2)
            .any(|w| w[0] == "--spec-type" && w[1] == "draft-mtp"));
        // llama3.2-3b has no catalog pair either: dense, no spec args at all.
        assert!(!p.argv.contains(&"--spec-type".to_string()));
    }

    #[test]
    fn unit__ubatch_override__emitted_only_when_set() {
        let g = meta();
        let hw = gpu_hw(12_000, 32_000, 8);
        let t = TuningOverrides {
            ubatch: Some(2048),
            ..TuningOverrides::default()
        };
        let p = compile(&input(&g, &hw, &Config::default(), &ALL_FLAGS), &t).unwrap();
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--ubatch-size" && w[1] == "2048"));
        let p2 = compile(
            &input(&g, &hw, &Config::default(), &ALL_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(!p2.argv.contains(&"--ubatch-size".to_string()));
    }

    #[test]
    fn unit__batch_plumbing__config_knobs_and_tuning_precedence() {
        let g = meta();
        let hw = gpu_hw(12_000, 32_000, 8);
        let cfg = Config {
            batch_size: 4096,
            ubatch_size: 1024,
            threads_batch: 4,
            ..Config::default()
        };
        let p = compile(
            &input(&g, &hw, &cfg, &ALL_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--batch-size" && w[1] == "4096"));
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--ubatch-size" && w[1] == "1024"));
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--threads-batch" && w[1] == "4"));
        // Bench-adopted tuning wins over the config knob (short -b form).
        let t = TuningOverrides {
            batch: Some(2048),
            ubatch: Some(512),
            ..TuningOverrides::default()
        };
        let p2 = compile(&input(&g, &hw, &cfg, &ALL_FLAGS), &t).unwrap();
        assert!(p2.argv.windows(2).any(|w| w[0] == "-b" && w[1] == "2048"));
        assert!(!p2.argv.contains(&"--batch-size".to_string()));
        assert!(p2
            .argv
            .windows(2)
            .any(|w| w[0] == "--ubatch-size" && w[1] == "512"));
        // Defaults: nothing emitted.
        let p3 = compile(
            &input(&g, &hw, &Config::default(), &ALL_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(!p3.argv.contains(&"--batch-size".to_string()));
        assert!(!p3.argv.contains(&"--threads-batch".to_string()));
    }

    #[test]
    fn unit__multi_gpu_split__emitted_and_engine_gated() {
        let g = meta();
        let hw = gpu_hw(12_000, 32_000, 8);
        let cfg = Config {
            main_gpu: 1,
            split_mode: "row".into(),
            tensor_split: "3,1".into(),
            ..Config::default()
        };
        let p = compile(
            &input(&g, &hw, &cfg, &ALL_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--main-gpu" && w[1] == "1"));
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--split-mode" && w[1] == "row"));
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--tensor-split" && w[1] == "3,1"));
        // Old engine lacking the trio: warn-skip, never a hard failure.
        let sparse: BTreeSet<String> = ALL_FLAGS
            .iter()
            .filter(|f| !matches!(f.as_str(), "--main-gpu" | "--split-mode" | "--tensor-split"))
            .cloned()
            .collect();
        let p2 = compile(&input(&g, &hw, &cfg, &sparse), &TuningOverrides::default()).unwrap();
        assert!(!p2.argv.contains(&"--main-gpu".to_string()));
        assert!(!p2.argv.contains(&"--split-mode".to_string()));
        assert!(!p2.argv.contains(&"--tensor-split".to_string()));
        assert_eq!(
            p2.warnings
                .iter()
                .filter(|w| w.contains("pallama engine update"))
                .count(),
            3
        );
    }

    #[test]
    fn unit__ctx_clamped_to_train_context() {
        let cfg = Config {
            default_ctx: 131_072, // > ctx_train 40960
            ..Config::default()
        };
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        let p = compile(
            &input(&g, &hw, &cfg, &ALL_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        // Train clamp bounds the TOTAL; the classic VRAM axis (device
        // truth) then re-spends it as 8 shallow slots (train_slots 1 at
        // the full clamp, so the re-spend branch wins; the auto ceiling
        // on this 8 GiB+ card is 8): 8x5120.
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--ctx-size" && w[1] == "40960"));
        assert!(p.argv.windows(2).any(|w| w == ["-np", "8"]));
        assert_eq!(p.ctx, 5_120);
        assert_eq!(p.ctx_autofit, Some((5_120, 8)));
        assert!(p.warnings.iter().any(|w| w.contains("clamped")));
    }

    #[test]
    fn unit__ctx_extend_yarn__stretches_clamp_ceiling() {
        // F109 regression: ctx_extend 2.0 on a 40960-trained model lifts
        // the clamp ceiling to 81920 — the trained-length clamp must not
        // defeat YaRN.
        let cfg = Config {
            default_ctx: 131_072,
            ctx_extend: 2.0,
            ..Config::default()
        };
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        let p = compile(
            &input(&g, &hw, &cfg, &ALL_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        // YaRN lifts the CEILING (81920); the capacity re-spend then
        // lands on 8x5120 (train_slots 2 caps the 20480 probe; the
        // 5120 probe earns the full auto ceiling of 8 on this 8 GiB+
        // card) — f16 KV 2240 MiB + weights 5000 <= the 12000 MiB card,
        // no shrink needed.
        assert_eq!(p.ctx, 5_120);
        assert_eq!(p.ctx_autofit, Some((5_120, 8)));
        assert!(p
            .warnings
            .iter()
            .any(|w| w.contains("yarn-stretched window 81920")));
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--ctx-size" && w[1] == "40960"));
    }

    #[test]
    fn unit__unified_kv_pool_fit__shrinks_ctx_to_cache_ram_budget() {
        // Device truth (live b10948): the KV pool is VRAM-resident on
        // BOTH lanes, so an autofit ctx whose pool overflows the card
        // must shrink against the 85% VRAM envelope — at the quant the
        // spawn will actually run. Geometry head_dim 64 -> 57344
        // B/ctx-token: kv(131072) = 7168 MiB f16; the tight 6000 MiB
        // card drives the ladder to q4_0 (1792 MiB pool, 14336
        // B/token), so the shrink fits (5100 - 4800) MiB / 14336 B =
        // 21951 -> floored to a 256 multiple = 21760.
        let cfg = Config {
            default_ctx: 131_072,
            ..Config::default()
        };
        let hw = gpu_hw(6_000, 13_674, 8);
        let g = GgufMeta {
            context_length: Some(131_072),
            ..meta()
        };
        let mut inp = input(&g, &hw, &cfg, &ALL_FLAGS);
        inp.model_bytes = 4_800 * MIB;
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(p.argv.windows(2).any(|w| w == ["--ctx-size", "21760"]));
        assert!(p.argv.windows(2).any(|w| w == ["--cache-type-k", "q8_0"]));
        assert!(p.warnings.iter().any(|w| w.contains("unified KV pool fit")));
        // (per_slot, slots) pair stays truthful for SlotsCtxAutoFit.
        let total: u64 = p
            .argv
            .windows(2)
            .find(|w| w[0] == "--ctx-size")
            .and_then(|w| w[1].parse::<u32>().ok())
            .map(u64::from)
            .expect("--ctx-size present");
        assert_eq!(total, 21_760);
        let (per_slot, slots) = p.ctx_autofit.expect("autofit marker set");
        assert_eq!(u64::from(per_slot) * u64::from(slots), total);
    }

    #[test]
    fn unit__unified_kv_pool_fit__pinned_ctx_refuses_when_guard_blocks() {
        // Pinned ctx is sovereign (never shrunk) but a pin the DEVICE
        // cannot host — even at the ladder's max q4_0 demotion — is a
        // REFUSAL, not a warning: warn-yet-proceed here is how the
        // num_ctx storm 502-looped (child died at context creation on
        // every spawn retry). Shape: weights 5000 + KV 7168 f16 / 1792
        // q4_0 = 6792 MiB demand + 700 overhead > 6000 MiB VRAM.
        let cfg = Config {
            default_ctx: 131_072,
            ..Config::default()
        };
        let hw = gpu_hw(6_000, 13_674, 8);
        let g = GgufMeta {
            context_length: Some(131_072),
            ..meta()
        };
        let mut inp = input(&g, &hw, &cfg, &ALL_FLAGS);
        inp.model_bytes = 5_000 * MIB;
        let err = compile(
            &inp,
            &TuningOverrides {
                ctx: Some(131_072), // hard pin, never divided
                ..Default::default()
            },
        )
        .unwrap_err();
        assert!(err.contains("pinned num_ctx 131072 cannot fit"), "{err}");
        assert!(err.contains("6792"), "names the demand: {err}");
        assert!(err.contains("6000"), "names the VRAM: {err}");
        assert!(err.contains("q8_0"), "teaches the KV-halving lever: {err}");
    }

    #[test]
    fn unit__pinned_pool_walk__auto_slots_walk_to_hostable_np() {
        // The capacity vram-slots axis is CAP-blind under --kv-unified: it
        // admits np4 at a pinned 8192 while the DEVICE verdict refuses
        // every pool above np1 (weights 379 + KV@8192 469 + 700 overhead
        // = 1548 <= 1600 VRAM fits ONLY at np1; np2 pool 16384 = 2017
        // refuses). Auto walks down and says so; the pin itself stays.
        let cfg = Config {
            default_ctx: 8_192,
            ..Config::default()
        };
        let hw = gpu_hw(1_600, 13_674, 8);
        let g = meta();
        let mut inp = input(&g, &hw, &cfg, &ALL_FLAGS);
        inp.model_bytes = 379 * MIB;
        let p = compile(
            &inp,
            &TuningOverrides {
                ctx: Some(8_192),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(
            p.warnings.iter().any(|w| w.contains("walked down to np 1")),
            "walk warning present: {:?}",
            p.warnings
        );
        assert!(p.argv.windows(2).any(|w| w == ["--ctx-size", "8192"]));
        assert!(p.argv.windows(2).any(|w| w == ["-np", "1"]));
    }

    #[test]
    fn unit__pinned_pool_walk__all_np_refuse_errors_at_np1() {
        // Nothing hostable at ANY count (weights 2000 + KV 469 + 700 =
        // 3169 > 1600 even at np1): the walk lands at 1 and the 2b
        // verdict teaches the refusal — never a warn-yet-proceed spawn.
        let cfg = Config {
            default_ctx: 8_192,
            ..Config::default()
        };
        let hw = gpu_hw(1_600, 13_674, 8);
        let g = meta();
        let mut inp = input(&g, &hw, &cfg, &ALL_FLAGS);
        inp.model_bytes = 2_000 * MIB;
        let err = compile(
            &inp,
            &TuningOverrides {
                ctx: Some(8_192),
                ..Default::default()
            },
        )
        .unwrap_err();
        assert!(err.contains("pinned num_ctx 8192 cannot fit"), "{err}");
        assert!(err.contains("q8_0"), "teaches the KV lever: {err}");
    }

    #[test]
    fn unit__pinned_pool_walk__explicit_slots_never_walk() {
        // An explicit slots pin is sovereign: the walk must NOT reshape
        // it. Explicit -np divides --ctx-size across slots upstream (the
        // pool total STAYS the pin — argv keeps ctx 8192 + -np 2), so a
        // hostable pin serves at the requested concurrency with no walk
        // warning; an unhostable pin errors at 2b (covered by the
        // refuses-when-guard-blocks pin).
        let cfg = Config {
            default_ctx: 8_192,
            ..Config::default()
        };
        let hw = gpu_hw(1_600, 13_674, 8);
        let g = meta();
        let mut inp = input(&g, &hw, &cfg, &ALL_FLAGS);
        inp.model_bytes = 379 * MIB;
        let ov = ModelOverride {
            slots: Some(2),
            ..Default::default()
        };
        inp.overlay = &ov;
        let p = compile(
            &inp,
            &TuningOverrides {
                ctx: Some(8_192),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(p.argv.windows(2).any(|w| w == ["--ctx-size", "8192"]));
        assert!(p.argv.windows(2).any(|w| w == ["-np", "2"]));
        assert!(
            !p.warnings.iter().any(|w| w.contains("walked down")),
            "explicit pins never walk: {:?}",
            p.warnings
        );
    }

    #[test]
    fn unit__unified_kv_pool_fit__pinned_middle_zone_serves_degraded() {
        // Middle zone (85%..100% of VRAM): physically hostable, so the
        // pin is SERVED — verdict Fit — but gpu-layers goes to the
        // engine fitter (may CPU-split the last layers) with an honest
        // warning. Shape: weights 2700 + f16 KV 1792 = 4492 MiB; VRAM
        // 5200 (85% share = 4420): demand fits the card (4492 + 700
        // overhead <= 5200) but not the 85% share.
        let cfg = Config {
            default_ctx: 32_768,
            ..Config::default()
        };
        let hw = gpu_hw(5_200, 13_674, 8);
        let g = meta(); // trained 40960 ≥ the pin; 57344 B/ctx-token
        let mut inp = input(&g, &hw, &cfg, &ALL_FLAGS);
        inp.model_bytes = 2_700 * MIB;
        let p = compile(
            &inp,
            &TuningOverrides {
                ctx: Some(32_768),
                ..Default::default()
            },
        )
        .unwrap();
        // The pin is honored untouched…
        assert!(p.argv.windows(2).any(|w| w == ["--ctx-size", "32768"]));
        assert_eq!(p.ctx_autofit, None);
        // …and the degraded serve is surfaced, not hidden.
        assert!(p.warnings.iter().any(
            |w| w.contains("fits the 5200 MiB VRAM only above the 85% share (demand 4492 MiB)")
        ));
    }

    #[test]
    fn unit__unified_ctx_verdict__pure_helper_boundary() {
        use super::{unified_ctx_verdict, UnifiedCtxVerdict};
        let weights = 2_455 * MIB;
        let kv = 1_792 * MIB;
        // Weights + KV + spawn overhead fit the device as-is
        // (2455 + 1792 + 700 = 4947 <= 5200).
        assert!(matches!(
            unified_ctx_verdict(weights, kv, 5_200 * MIB, 32_768),
            UnifiedCtxVerdict::Fit
        ));
        // Exact boundary: demand + overhead == VRAM is still hostable.
        assert!(matches!(
            unified_ctx_verdict(weights, kv, 4_947 * MIB, 32_768),
            UnifiedCtxVerdict::Fit
        ));
        // One MiB less → physically impossible → refuse with teaching.
        let v = unified_ctx_verdict(weights, kv, 4_946 * MIB, 32_768);
        let UnifiedCtxVerdict::Refuse(msg) = v else {
            panic!("must refuse below the demand+overhead line");
        };
        assert!(
            msg.contains("q8_0") && msg.contains("smaller quant"),
            "{msg}"
        );
        // A fat-KV pin over a small card refuses too.
        assert!(matches!(
            unified_ctx_verdict(weights, kv * 4, 5_200 * MIB, 131_072),
            UnifiedCtxVerdict::Refuse(_)
        ));
    }

    #[test]
    fn unit__extra_args_cache_ram__explicit_wins_and_lands_once() {
        // Explicit extra_args --cache-ram owns the budget (same doctrine
        // as `-mm`): rule 12 must not double-emit the flag and the A16
        // clamp must not touch it (the explicit branch wins before the
        // clamp is even computed).
        let cfg = Config {
            default_ctx: 32_768,
            cache_ram_mb: 8192,
            ..Config::default()
        };
        let hw = gpu_hw(20_000, 32_000, 8);
        let g = meta();
        let mut inp = input(&g, &hw, &cfg, &ALL_FLAGS);
        let explicit = ModelOverride {
            extra_args: Some(vec!["--cache-ram".into(), "12000".into()]),
            ..Default::default()
        };
        inp.overlay = &explicit;
        inp.model_bytes = 6_000 * MIB;
        let p = compile(
            &inp,
            &TuningOverrides {
                ctx: Some(32_768),
                ..Default::default()
            },
        )
        .unwrap();
        let emissions = p
            .argv
            .windows(2)
            .filter(|w| w[0] == "--cache-ram")
            .collect::<Vec<_>>();
        assert_eq!(emissions.len(), 1, "flag lands exactly once");
        assert_eq!(emissions[0], ["--cache-ram", "12000"]);
        // The explicit branch wins before the A16 clamp — no clamp
        // warning for the explicit value.
        assert!(!p.warnings.iter().any(|w| w.contains("clamped")));
    }

    #[test]
    fn unit__unified_kv_pool_fit__tiny_model_argv_unchanged() {
        // Regression guard: pools that already fit must keep resolve_slots'
        // own plan untouched (slots-auto np2, total ctx 32768) and get no
        // new warning from the unified-KV fit rule.
        let cfg = Config::default();
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        let mut inp = input(&g, &hw, &cfg, &ALL_FLAGS);
        inp.model_bytes = 500 * MIB;
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(p.argv.windows(2).any(|w| w == ["--ctx-size", "32768"]));
        assert!(!p
            .warnings
            .iter()
            .any(|w| w.contains("unified KV pool") || w.contains("pinned ctx")));
    }

    #[test]
    fn unit__ctx_extend_yarn__still_clamps_past_stretched() {
        let cfg = Config {
            default_ctx: 131_072,
            ctx_extend: 1.5, // 40960 -> 61440
            ..Config::default()
        };
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        let p = compile(
            &input(&g, &hw, &cfg, &ALL_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        // Stretched ceiling 61440; auto-fit divides to 5x7680 (38400
        // total) — the raised auto ceiling admits the fifth slot on
        // this 8 GiB+ card.
        assert_eq!(p.ctx, 7_680);
        assert_eq!(p.ctx_autofit, Some((7_680, 5)));
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--ctx-size" && w[1] == "38400"));
    }

    #[test]
    fn unit__ctx_train_missing__warning_not_guess() {
        let mut g = meta();
        g.context_length = None;
        let cfg = Config::default();
        let hw = gpu_hw(12_000, 32_000, 8);
        let p = compile(
            &input(&g, &hw, &cfg, &ALL_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--ctx-size" && w[1] == "16384"));
        assert!(p
            .warnings
            .iter()
            .any(|w| w.contains("lacks context_length")));
    }

    #[test]
    fn unit__kv_quant__q8_when_halving_fits() {
        // 5 GiB model + 896 MiB f16 KV (2*28*8*64*16384*2 bytes) on
        // 6.1 GiB VRAM: budget 5.36 GiB. f16 total 5.75 GiB > budget ->
        // engage; KV/2 total 5.32 GiB <= budget -> q8_0 (the cheapest grade
        // that fits; the old fixed threshold picked q8_0 even when it did
        // not fit).
        let cfg = Config {
            slots: 1,           // purpose-scoped: ladder grades, not slot sizing
            spec: "off".into(), // purpose-scoped: not the auto spec lane
            ..Config::default()
        };
        let hw = gpu_hw(6_100, 32_000, 8);
        let g = meta();
        let p = compile(
            &input(&g, &hw, &cfg, &ALL_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--cache-type-k" && w[1] == "q8_0"));
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--cache-type-v" && w[1] == "q8_0"));
        assert!(p.warnings.is_empty(), "{:?}", p.warnings);
    }

    #[test]
    fn unit__kv_quant__q4_when_only_quarter_fits() {
        // Same model on 5.5 GiB VRAM: budget 4.95 GiB. f16 5.469 and
        // q8_0 5.234 both overflow; KV/4 = 5.117 GiB still overflows ->
        // q4_0 engaged WITH the still-overflow warning.
        let cfg = Config::default();
        let hw = gpu_hw(5_500, 32_000, 8);
        let g = meta();
        let p = compile(
            &input(&g, &hw, &cfg, &ALL_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--cache-type-k" && w[1] == "q4_0"));
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--cache-type-v" && w[1] == "q4_0"));
        assert!(p
            .warnings
            .iter()
            .any(|w| w.contains("still exceed 90% VRAM")));
    }

    #[test]
    fn unit__kv_quant__explicit_config_beats_ladder() {
        let cfg = Config {
            cache_type: "q5_0".into(),
            ..Config::default()
        };
        let hw = gpu_hw(24_000, 32_000, 8); // plenty of VRAM: ladder says none
        let g = meta();
        let p = compile(
            &input(&g, &hw, &cfg, &ALL_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--cache-type-k" && w[1] == "q5_0"));
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--cache-type-v" && w[1] == "q5_0"));
    }

    #[test]
    fn unit__kv_quant__explicit_f16_forces_off() {
        let cfg = Config {
            cache_type: "f16".into(),
            ..Config::default()
        };
        let hw = gpu_hw(5_500, 32_000, 8); // ladder would engage q4_0
        let g = meta();
        let p = compile(
            &input(&g, &hw, &cfg, &ALL_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(!p.argv.contains(&"--cache-type-k".to_string()));
    }

    #[test]
    fn unit__ctx_extend__yarn_emitted_with_warning() {
        let cfg = Config {
            ctx_extend: 2.0,
            ..Config::default()
        };
        let hw = gpu_hw(24_000, 32_000, 8);
        let g = meta();
        let p = compile(
            &input(&g, &hw, &cfg, &ALL_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--rope-scaling" && w[1] == "yarn"));
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--rope-scale" && w[1] == "2"));
        assert!(p.warnings.iter().any(|w| w.contains("YaRN")));
    }

    #[test]
    fn unit__moe_and_tensor_overrides__emitted() {
        let cfg = Config {
            cpu_moe_n: 12,
            override_tensor: vec![".ffn_.*_exps.=CPU".into()],
            ..Config::default()
        };
        let hw = gpu_hw(24_000, 32_000, 8);
        let g = meta();
        let p = compile(
            &input(&g, &hw, &cfg, &ALL_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--n-cpu-moe" && w[1] == "12"));
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--override-tensor" && w[1] == ".ffn_.*_exps.=CPU"));
    }

    #[test]
    fn unit__cpu_ffn_n__emitted_and_overlay_wins() {
        let cfg = Config {
            cpu_ffn_n: 3,
            ..Config::default()
        };
        let hw = gpu_hw(24_000, 32_000, 8);
        let g = meta();
        let p = compile(
            &input(&g, &hw, &cfg, &ALL_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--n-cpu-ffn" && w[1] == "3"));
        // Default stays argv-silent.
        let p0 = compile(
            &input(&g, &hw, &Config::default(), &ALL_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(!p0.argv.contains(&"--n-cpu-ffn".to_string()));
        // Overlay Some(0) is an explicit off — F114 discipline.
        let cfg_off = Config {
            cpu_ffn_n: 3,
            model_overrides: std::collections::BTreeMap::from([(
                "qwen3-8b".into(),
                ModelOverride {
                    cpu_ffn_n: Some(0),
                    ..ModelOverride::default()
                },
            )]),
            ..Config::default()
        };
        let p_off = compile(
            &input(&g, &hw, &cfg_off, &ALL_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(!p_off.argv.contains(&"--n-cpu-ffn".to_string()));
    }

    #[test]
    fn unit__agent__opt_in_with_loud_warning() {
        let cfg = Config::default();
        let hw = gpu_hw(24_000, 32_000, 8);
        let g = meta();
        let p = compile(
            &input(&g, &hw, &cfg, &ALL_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(!p.argv.contains(&"--agent".to_string()));
        let cfg2 = Config {
            agent: true,
            ..Config::default()
        };
        let p2 = compile(
            &input(&g, &hw, &cfg2, &ALL_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(p2.argv.contains(&"--agent".to_string()));
        assert!(p2.warnings.iter().any(|w| w.contains("exec_shell_command")));
    }

    #[test]
    fn unit__spec_cache_and_sessions__disabled_by_config() {
        let cfg = Config {
            spec: "ngram".into(),
            spec_cache: false,
            sessions: false,
            slots: 1, // purpose-scoped: opt-out flags, not slot sizing
            ..Config::default()
        };
        let hw = gpu_hw(24_000, 32_000, 8);
        let g = meta();
        let p = compile(
            &input(&g, &hw, &cfg, &ALL_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(!p.argv.contains(&"--lookup-cache-dynamic".to_string()));
        assert!(!p.argv.contains(&"--slot-save-path".to_string()));
        assert!(p.warnings.is_empty(), "{:?}", p.warnings);
    }

    #[test]
    fn unit__router_preset__sections_keys_and_bools() {
        let hw = gpu_hw(24_000, 32_000, 8);
        let g = meta();
        let cfg = Config {
            slots: 1, // purpose-scoped: INI shape, not slot sizing
            ..Config::default()
        };
        let p = compile(
            &input(&g, &hw, &cfg, &ALL_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        // second model with an overlay knob (F111: the overlay is now
        // actually APPLIED — the fixture used to build it and discard)
        let o = ModelOverride {
            ctx: Some(8192),
            warmup: Some(false),
            sampler_defaults: Some(SamplerDefaults {
                temperature: Some(0.7),
                ..SamplerDefaults::default()
            }),
            ..ModelOverride::default()
        };
        let cfg2 = Config {
            slots: 1,
            ..Config::default()
        };
        let mut ip2 = input(&g, &hw, &cfg2, &ALL_FLAGS);
        ip2.overlay = &o;
        let p2 = compile(&ip2, &TuningOverrides::default()).unwrap();
        let ini = generate_router_preset(
            &[
                ("m1".to_string(), p.argv.clone()),
                ("m2".to_string(), p2.argv.clone()),
            ],
            &p.argv,
        );
        // global section: server-level only
        assert!(ini.starts_with("[*]\n"));
        assert!(ini.contains("jinja = true"));
        assert!(ini.contains("metrics = true"));
        assert!(ini.contains("sleep-idle-seconds = 300"));
        assert!(ini.contains("cache-ram = 8192"));
        assert!(ini.contains("slot-save-path = "));
        // reserved/global-excluded keys never appear
        assert!(!ini.contains("host ="));
        assert!(!ini.contains("port ="));
        assert!(!ini.contains("alias ="));
        // per-model sections carry model-scoped flags
        assert!(ini.contains("[m1]\n"));
        assert!(ini.contains("[m2]\n"));
        assert!(ini.contains("model = /models/qwen3-8b.gguf"));
        assert!(ini.contains("gpu-layers = 999"));
        assert!(ini.contains("parallel = 1"));
        assert!(!ini.contains("cache-reuse"));
        // F111: m2's overlay actually applied — its ctx + sampler +
        // warmup knobs land in the m2 section.
        let m2 = ini.split("[m2]\n").nth(1).unwrap_or_default();
        assert!(m2.contains("ctx-size = 8192"), "{m2}");
        assert!(m2.contains("temp = 0.7"), "{m2}");
        // F110: per-context knobs no longer dropped by the INI.
        assert!(m2.contains("no-warmup = true"), "{m2}");
        // and they stay OUT of the global section
        let global = ini.split("\n\n").next().unwrap_or_default();
        assert!(!global.contains("no-warmup"), "{global}");
        assert!(!global.contains("ctx-size"), "{global}");
    }

    #[test]
    fn unit__slot_prompt_similarity__emitted_only_when_set() {
        let hw = gpu_hw(24_000, 32_000, 8);
        let g = meta();
        let cfg = Config::default();
        let p = compile(
            &input(&g, &hw, &cfg, &ALL_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(!p.argv.contains(&"--slot-prompt-similarity".to_string()));
        let cfg2 = Config {
            slot_prompt_similarity: 0.3,
            ..Config::default()
        };
        let p2 = compile(
            &input(&g, &hw, &cfg2, &ALL_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(p2
            .argv
            .windows(2)
            .any(|w| w[0] == "--slot-prompt-similarity" && w[1] == "0.3"));
    }

    #[test]
    fn unit__new_rules__warn_skip_on_old_engine() {
        // Engine without the default-on conveniences: named warnings, no
        // emission, no hard error.
        let mut flags = ALL_FLAGS.clone();
        flags.remove("--slot-save-path");
        flags.remove("--lookup-cache-dynamic");
        let cfg = Config {
            spec: "ngram".into(),
            ..Config::default()
        };
        let hw = gpu_hw(24_000, 32_000, 8);
        let g = meta();
        let p = compile(&input(&g, &hw, &cfg, &flags), &TuningOverrides::default()).unwrap();
        assert!(!p.argv.contains(&"--slot-save-path".to_string()));
        assert!(!p.argv.contains(&"--lookup-cache-dynamic".to_string()));
        assert!(p.warnings.iter().any(|w| w.contains("--slot-save-path")));
        assert!(p
            .warnings
            .iter()
            .any(|w| w.contains("--lookup-cache-dynamic")));
    }

    #[test]
    fn unit__kv_quant__head_count_kv_missing_falls_back_to_head_count() {
        // Upstream llama.cpp defaults head_count_kv = head_count when the
        // GGUF omits it (no-GQA models like qwen2.5-0.5b) — same rule here.
        let mut g = meta();
        g.head_count_kv = None; // falls back to head_count 16
        let cfg = Config::default();
        let hw = gpu_hw(5_500, 32_000, 8);
        let p = compile(
            &input(&g, &hw, &cfg, &ALL_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        // KV with 16 heads vs 8 is bigger; 5GiB model + KV on 5.4GiB VRAM
        // crosses the 0.9x threshold either way -> q8_0 engaged, no warning.
        assert!(p.argv.contains(&"--cache-type-k".to_string()));
        assert!(!p.warnings.iter().any(|w| w.contains("head_count_kv")));
    }

    #[test]
    fn unit__cpu_moe__when_model_exceeds_vram_and_ram_fits() {
        let mut g = meta();
        g.expert_count = Some(128);
        let cfg = Config::default();
        let hw = gpu_hw(2_000, 64_000, 8); // model 5GiB > 2GiB VRAM; RAM 64GiB >= 1.1x
        let p = compile(
            &input(&g, &hw, &cfg, &ALL_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(p.argv.contains(&"--cpu-moe".to_string()));
        // RAM too small -> not emitted
        let hw_small = gpu_hw(2_000, 5_000, 8);
        let p2 = compile(
            &input(&g, &hw_small, &cfg, &ALL_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(!p2.argv.contains(&"--cpu-moe".to_string()));
    }

    #[test]
    fn unit__no_gpu__gpu_layers_zero_labeled_cpu() {
        let cfg = Config::default();
        let hw = Hardware {
            physical_cores: 8,
            total_ram_mib: 32_000,
            gpus: vec![],
        };
        let g = meta();
        let p = compile(
            &input(&g, &hw, &cfg, &ALL_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(!p.argv.contains(&"--sleep-idle-seconds".to_string()));
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--gpu-layers" && w[1] == "0"));
        assert_eq!(p.gpu, "cpu");
    }

    #[test]
    fn unit__capacity__spawn_free_vram_gates_the_full_pin() {
        // Contended card: 12 GiB total but a neighbour holds all but
        // 900 MiB at spawn time. The unified-KV projection (weights +
        // 512 MiB KV floor + spawn overhead) cannot fit 900 MiB, so the
        // resolver must decline the `-ngl 999` pin and hand the decision
        // to the engine's dynamic `auto` band — the live-proven OOM
        // shape ("n_gpu_layers already set by user to 999, abort").
        let cfg = Config::default();
        let mut hw = gpu_hw(12_000, 32_000, 8);
        hw.gpus[0].free_mib = 900;
        let g = meta();
        let p = compile(
            &input(&g, &hw, &cfg, &ALL_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(
            p.argv
                .windows(2)
                .any(|w| w[0] == "--gpu-layers" && w[1] == "auto"),
            "contended free VRAM must defer to the engine auto band: {:?}",
            p.argv
        );
        assert!(!p
            .argv
            .windows(2)
            .any(|w| w[0] == "--gpu-layers" && w[1] == "999"));

        // Same box idle (free == total): the comfortable full fit keeps
        // the deterministic 999 pin (ps labeling + no estimator drift).
        let hw_idle = gpu_hw(12_000, 32_000, 8);
        let p2 = compile(
            &input(&g, &hw_idle, &cfg, &ALL_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(p2
            .argv
            .windows(2)
            .any(|w| w[0] == "--gpu-layers" && w[1] == "999"));
    }

    #[test]
    fn unit__gpu_partial__weights_exceed_vram_warns_and_stays_auto() {
        let cfg = Config::default();
        let hw = gpu_hw(2_000, 32_000, 8); // 5 GiB model vs 2 GiB VRAM
        let g = meta();
        let p = compile(
            &input(&g, &hw, &cfg, &ALL_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--gpu-layers" && w[1] == "auto"));
        assert_eq!(p.gpu, "partial");
        assert!(p.warnings.iter().any(|w| w.contains("exceed VRAM")));
    }

    #[test]
    fn unit__rpc_and_loras() {
        let cfg = Config {
            rpc_servers: "box1:50052,box2:50052".into(),
            ..Config::default()
        };
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        let mut inp = input(&g, &hw, &cfg, &ALL_FLAGS);
        let loras = vec![
            ("/loras/a.bin".to_string(), 1.0),
            ("/loras/b.bin".to_string(), 0.5),
        ];
        inp.loras = &loras;
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--rpc" && w[1] == "box1:50052,box2:50052"));
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--lora" && w[1] == "/loras/a.bin"));
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--lora-scaled" && w[1] == "/loras/b.bin:0.5"));
    }

    #[test]
    fn unit__rpc__per_model_override_replaces_global_and_empty_inherits() {
        // C6: a non-empty per-model rpc_servers replaces the global list;
        // an empty/absent overlay inherits it untouched.
        let cfg = Config {
            rpc_servers: "box1:50052,box2:50052".into(),
            model_overrides: std::collections::BTreeMap::from([(
                "qwen3-8b".to_string(),
                ModelOverride {
                    rpc_servers: Some("gpu3:50052".into()),
                    ..ModelOverride::default()
                },
            )]),
            ..Config::default()
        };
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        let inp = input(&g, &hw, &cfg, &ALL_FLAGS);
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--rpc" && w[1] == "gpu3:50052"));
        assert!(!p
            .argv
            .windows(2)
            .any(|w| w[0] == "--rpc" && w[1].contains("box1")));

        // Overlay present but empty -> global inherited.
        let cfg = Config {
            rpc_servers: "box1:50052".into(),
            model_overrides: std::collections::BTreeMap::from([(
                "qwen3-8b".to_string(),
                ModelOverride {
                    rpc_servers: Some("   ".into()),
                    ..ModelOverride::default()
                },
            )]),
            ..Config::default()
        };
        let inp = input(&g, &hw, &cfg, &ALL_FLAGS);
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--rpc" && w[1] == "box1:50052"));
    }

    #[test]
    fn unit__unix_socket_endpoint() {
        let cfg = Config::default();
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        let mut inp = input(&g, &hw, &cfg, &ALL_FLAGS);
        inp.endpoint = Endpoint::Unix {
            socket: "/run/pallama/m.sock".into(),
        };
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--host" && w[1] == "/run/pallama/m.sock"));
        assert!(!p.argv.contains(&"--port".to_string()));
    }

    #[test]
    fn unit__spec_auto__draft_present_and_missing() {
        static DRAFT_SIMPLE_SPEC_TYPES: LazyLock<Vec<String>> =
            LazyLock::new(|| vec!["draft-simple".to_string()]);
        let cfg = Config {
            spec: "auto".into(),
            ..Config::default()
        };
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        let mut inp = input_with_spec(&g, &hw, &cfg, &ALL_FLAGS, &DRAFT_SIMPLE_SPEC_TYPES);
        inp.model_name = "qwen3-8b"; // has catalog draft pair
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(p.warnings.iter().any(
            |w| w.contains("not pulled") && w.contains("pallama pull ggml-org/Qwen3-0.6B-GGUF")
        ));
        assert!(!p.argv.contains(&"--spec-type".to_string()));

        // Draft pulled -> flags emitted.
        let draft = draft_file("auto-pair");
        inp.draft_path = Some(&draft);
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--spec-type" && w[1] == "draft-simple"));
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--spec-draft-model" && w[1] == draft.as_str()));
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--spec-draft-n-max" && w[1] == "3"));
    }

    #[test]
    fn unit__cache_idle_slots__opt_out_only() {
        // Default (true, upstream default): nothing emitted. The knob
        // exists to disable.
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        let p = compile(
            &input(&g, &hw, &Config::default(), &ALL_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(!p.argv.contains(&"--no-cache-idle-slots".to_string()));

        let cfg = Config {
            cache_idle_slots: false,
            ..Config::default()
        };
        let p = compile(
            &input(&g, &hw, &cfg, &ALL_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(p.argv.contains(&"--no-cache-idle-slots".to_string()));
    }

    #[test]
    fn unit__device_hint__auto_pick_and_manual_wins() {
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        // Auto pick lands on the hinted card.
        let cfg0 = Config::default();
        let mut i = input(&g, &hw, &cfg0, &ALL_FLAGS);
        i.device_hint = Some("Vulkan1");
        let p = compile(&i, &TuningOverrides::default()).unwrap();
        let idx = p.argv.iter().position(|a| a == "--device").unwrap();
        assert_eq!(p.argv[idx + 1], "Vulkan1");

        // Manual `devices` always beats the hint.
        let cfg = Config {
            devices: vec!["ManualGPU".to_string()],
            ..Config::default()
        };
        let mut i = input(&g, &hw, &cfg, &ALL_FLAGS);
        i.device_hint = Some("Vulkan1");
        let p = compile(&i, &TuningOverrides::default()).unwrap();
        assert!(p.argv.windows(2).any(|w| w == ["--device", "ManualGPU"]));
        assert!(!p.argv.windows(2).any(|w| w == ["--device", "Vulkan1"]));
    }

    #[test]
    fn unit__slots__overlay_shadows_global() {
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        let ov = ModelOverride {
            slots: Some(3),
            ..Default::default()
        };
        let cfg0 = Config::default();
        let mut i = input(&g, &hw, &cfg0, &ALL_FLAGS);
        i.overlay = &ov;
        let p = compile(&i, &TuningOverrides::default()).unwrap();
        assert!(p.argv.windows(2).any(|w| w == ["-np", "3"]));
        // No hint, no overlay, default slots=0 (pallama auto): capacity
        // decides — this fixture (train 40960, base 16384, unified KV,
        // comfortable RAM/VRAM) earns np=2 with the total ctx scaled.
        let p = compile(
            &input(&g, &hw, &Config::default(), &ALL_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(p.argv.windows(2).any(|w| w == ["-np", "2"]));
        assert!(
            p.argv.windows(2).any(|w| w == ["--ctx-size", "32768"]),
            "total ctx scales with auto slots: {:?}",
            p.argv
        );
        assert_eq!(p.ctx, 16_384, "Profile.ctx stays per-slot");
        // Auto-slot rationale is journal-only now.
        assert!(!p.warnings.iter().any(|w| w.contains("slots auto")));
    }

    #[test]
    fn unit__auto_slots__matrix_train_ram_vram_and_pins() {
        let g = meta(); // kv/slot @16384 = 896 MiB, train 40960
        let cfg = Config::default();
        let cfg_tight = Config {
            cache_ram_mb: 1024,
            ..Config::default()
        };
        let cfg_pin = Config {
            slots: 3,
            ..Config::default()
        };
        let g_default = GgufMeta::default();
        let ov1 = ModelOverride {
            slots: Some(1),
            ..Default::default()
        };
        // Train cap: unlimited-ish budgets still clamp np to train/base.
        let hw = gpu_hw(24_000, 64_000, 8);
        let p = compile(
            &input(&g, &hw, &cfg, &ALL_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(p.argv.windows(2).any(|w| w == ["-np", "2"]));
        assert!(p.argv.windows(2).any(|w| w == ["--ctx-size", "32768"]));

        // RAM clamp: 1 GiB effective cache-ram < 2 x 896 MiB KV -> single
        // at the FULL ctx, but the default ctx is divisible — auto-fit
        // re-spends the same RAM budget as 4 x 4096 (224 MiB KV/slot).
        let p2 = compile(
            &input(&g, &hw, &cfg_tight, &ALL_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(
            p2.argv.windows(2).any(|w| w == ["-np", "4"]),
            "{:?}",
            p2.argv
        );
        assert!(p2.argv.windows(2).any(|w| w == ["--ctx-size", "16384"]));
        assert_eq!(
            p2.ctx, 4_096,
            "Profile.ctx reports the autofit per-slot ctx"
        );
        assert_eq!(p2.ctx_autofit, Some((4_096, 4)));
        // Auto-fit rationale is journal-only now.
        assert!(!p2.warnings.iter().any(|w| w.contains("slots auto")));

        // Classic (non-unified) VRAM guard: 85% headroom admits no second
        // slot's KV -> single, ctx unscaled.
        let mut classic = full_flags();
        classic.remove("--kv-unified");
        let hw_small = gpu_hw(6_400, 64_000, 8);
        let p3 = compile(
            &input(&g, &hw_small, &cfg, &classic),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(p3.argv.windows(2).any(|w| w == ["-np", "1"]));

        // Unknown KV geometry: never guess a multi-slot cost.
        let p4 = compile(
            &input(&g_default, &hw, &cfg, &ALL_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(p4.argv.windows(2).any(|w| w == ["-np", "1"]));

        // Explicit manual pin: pass-through, ctx NOT scaled (upstream
        // slices the resolved ctx across the pinned slots).
        let p5 = compile(
            &input(&g, &hw, &cfg_pin, &ALL_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(p5.argv.windows(2).any(|w| w == ["-np", "3"]));
        assert!(p5.argv.windows(2).any(|w| w == ["--ctx-size", "16384"]));

        // Overlay pin shadows the global auto entirely.
        let mut inp6 = input(&g, &hw, &cfg, &ALL_FLAGS);
        inp6.overlay = &ov1;
        let p6 = compile(&inp6, &TuningOverrides::default()).unwrap();
        assert!(p6.argv.windows(2).any(|w| w == ["-np", "1"]));
        assert!(!p6.warnings.iter().any(|w| w.contains("slots auto")));
    }

    #[test]
    fn unit__auto_slots__eight_cap_on_big_cards_only() {
        // 8 GiB+ card with train/RAM/VRAM budgets that all admit 8: the
        // ceiling itself binds (np-sweep evidence — true parallelism at
        // slots 8 on an 8 GiB RTX 4070, 440.7 vs 394 tok/s at cap 4).
        // Classic (non-unified) flags keep this on the direct
        // auto_slots_capacity path; a small model keeps the small
        // card's budget math away from the resident-weight floor.
        let g = GgufMeta {
            context_length: Some(131_072),
            ..meta()
        };
        let cfg = Config::default();
        let mut classic = full_flags();
        classic.remove("--kv-unified");
        let hw = gpu_hw(24_000, 64_000, 8);
        let inp = ProfileInput {
            model_bytes: 500 * MIB,
            ..input(&g, &hw, &cfg, &classic).clone()
        };
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(p.argv.windows(2).any(|w| w == ["-np", "8"]), "{:?}", p.argv);

        // Same budgets on a sub-8-GiB card keep the conservative 4.
        let hw_small = gpu_hw(6_400, 64_000, 8);
        let inp2 = ProfileInput {
            model_bytes: 500 * MIB,
            ..input(&g, &hw_small, &cfg, &classic).clone()
        };
        let p2 = compile(&inp2, &TuningOverrides::default()).unwrap();
        assert!(
            p2.argv.windows(2).any(|w| w == ["-np", "4"]),
            "{:?}",
            p2.argv
        );
    }

    #[test]
    fn unit__deterministic__pins_slots_one_llama_lane() {
        let g = meta();
        let hw = gpu_hw(24_000, 64_000, 8); // auto would pick np 2 (train cap)
        let cfg = Config {
            deterministic: true,
            ..Config::default()
        };
        let p = compile(
            &input(&g, &hw, &cfg, &ALL_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(p.argv.windows(2).any(|w| w == ["-np", "1"]));
        assert!(p.argv.windows(2).any(|w| w == ["--ctx-size", "16384"]));
        assert!(p
            .warnings
            .iter()
            .any(|w| w.contains("deterministic = true: slots pinned")));

        // Explicit slots > 1 loses to the pin, loudly.
        let cfg_pin = Config {
            deterministic: true,
            slots: 3,
            ..Config::default()
        };
        let p2 = compile(
            &input(&g, &hw, &cfg_pin, &ALL_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(p2.argv.windows(2).any(|w| w == ["-np", "1"]));
        assert!(p2
            .warnings
            .iter()
            .any(|w| w.contains("explicit slots = 3 ignored")));

        // Overlay false un-pins a global true.
        let ov_off = ModelOverride {
            deterministic: Some(false),
            ..Default::default()
        };
        let mut inp3 = input(&g, &hw, &cfg, &ALL_FLAGS);
        inp3.overlay = &ov_off;
        let p3 = compile(&inp3, &TuningOverrides::default()).unwrap();
        assert!(
            p3.argv.windows(2).any(|w| w == ["-np", "2"]),
            "{:?}",
            p3.argv
        );

        // slots = 1 + deterministic: already deterministic-shaped, silent.
        let cfg1 = Config {
            deterministic: true,
            slots: 1,
            ..Config::default()
        };
        let p4 = compile(
            &input(&g, &hw, &cfg1, &ALL_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(p4.argv.windows(2).any(|w| w == ["-np", "1"]));
        assert!(!p4.warnings.iter().any(|w| w.contains("deterministic")));
    }

    #[test]
    fn unit__deterministic__mistralrs_lane_pins_np() {
        let g = meta();
        let mut flags = BTreeSet::new();
        flags.insert("--paged-attn".to_string());
        let cfg = Config {
            deterministic: true,
            ..Config::default()
        };
        let hw = gpu_hw(24_000, 32_000, 8);
        let mut inp = input(&g, &hw, &cfg, &flags);
        inp.engine_kind = crate::engine_kind::EngineKind::MistralRs;
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(p.argv.windows(2).any(|w| w == ["-np", "1"]));
        assert!(p
            .warnings
            .iter()
            .any(|w| w.contains("deterministic = true")));
    }

    /// Hardware with a co-tenant GPU: total 8 GiB, only `free` still
    /// available (the live bug shape: ollama holding 6.3 GiB).
    fn cotenant_hw(total_mib: u64, free_mib: u64, ram_mib: u64) -> Hardware {
        let mut hw = gpu_hw(total_mib, ram_mib, 8);
        hw.gpus[0].free_mib = free_mib;
        hw
    }

    fn mistralrs_pa_flags() -> BTreeSet<String> {
        ["--paged-attn", "--pa-memory-fraction"]
            .into_iter()
            .map(String::from)
            .collect()
    }

    /// mistral.rs enables paged attention BY DEFAULT and its stock
    /// `--pa-memory-fraction` is 0.90 of TOTAL VRAM — with a co-tenant
    /// on the GPU that pool points at memory the engine cannot have,
    /// silently truncating KV into ~60 ms empty responses (live-proven:
    /// 0.5B BF16 + 6.3 GiB tenant on an 8 GiB card). Under co-tenancy
    /// the compiler must derive a fraction from what is actually FREE.
    #[test]
    fn unit__mistralrs_pa_fraction__derived_under_cotenancy() {
        let g = meta();
        let flags = mistralrs_pa_flags();
        let cfg = Config::default();
        // The live incident: total 8192, free 1843, model 942 MiB →
        // budget 1843-942-512 = 389 MiB → 389/8192 = 0.047 → floor 0.05.
        let hw = cotenant_hw(8_192, 1_843, 16_000);
        let mut inp = input(&g, &hw, &cfg, &flags);
        inp.engine_kind = crate::engine_kind::EngineKind::MistralRs;
        inp.model_bytes = 942 * MIB;
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(
            p.argv
                .windows(2)
                .any(|w| w == ["--pa-memory-fraction", "0.05"]),
            "co-tenant spawn must derive a floor-clamped fraction, got {:?}",
            p.argv
        );
        assert!(p.warnings.iter().any(|w| w.contains("co-tenant")));
    }

    /// Healthy co-tenant budget derives a proportional fraction (not the
    /// floor): free 6144 − resident 942 − headroom 512 = 4690/8192 ≈ 0.57.
    #[test]
    fn unit__mistralrs_pa_fraction__healthy_budget_derives_proportionally() {
        let g = meta();
        let flags = mistralrs_pa_flags();
        let cfg = Config::default();
        let hw = cotenant_hw(8_192, 6_144, 16_000);
        let mut inp = input(&g, &hw, &cfg, &flags);
        inp.engine_kind = crate::engine_kind::EngineKind::MistralRs;
        inp.model_bytes = 942 * MIB;
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(
            p.argv
                .windows(2)
                .any(|w| w == ["--pa-memory-fraction", "0.57"]),
            "expected derived 0.57, got {:?}",
            p.argv
        );
    }

    /// mistralrs pa-fraction input over an HF (safetensors dir) meta —
    /// the geometry path these pins exercise.
    fn mistralrs_hf_input<'a>(
        hf: &'a crate::hfmeta::HfMeta,
        hw: &'a Hardware,
        cfg: &'a Config,
        flags: &'a BTreeSet<String>,
        weights: u64,
        overlay: &'a ModelOverride,
    ) -> ProfileInput<'a> {
        ProfileInput {
            engine_kind: crate::engine_kind::EngineKind::MistralRs,
            model_name: "qwen2.5-0.5b-instruct",
            instance_key: "qwen2.5-0.5b-instruct",
            model_path: "/models/qwen2.5-0.5b-instruct.d",
            model_bytes: weights,
            meta: ModelMeta::Hf(hf),
            hardware: hw,
            config: cfg,
            overlay,
            loras: &[],
            draft_path: None,
            draft_gguf: None,
            mmproj_path: None,
            mmproj_force: false,
            engine_tag: "v0.9.3",
            supported_flags: flags,
            spec_types: &[],
            endpoint: Endpoint::Tcp {
                host: "127.0.0.1".into(),
                port: 12345,
            },
            data_dir: "/tmp/pallama-test-data",
            cache_hit_rate: None,
            resident_ram_mib: 0,
            device_hint: None,
            engine_census: hw.gpus.clone(),
            sibling_devices: Vec::new(),
            auto_tensor_split: None,
        }
    }

    /// THE live incident (user's JIT run): idle 8 GiB card, 0.5B BF16
    /// dir (942 MiB) — upstream's 0.90-of-total default made the child
    /// hold 6984 MiB and starve the next routed lane. With KV geometry
    /// the compiler must size the fraction from need: weights 942 +
    /// f16 KV 896 (28*2*128*2*32768*2 B) + floor 512 = 2350 of 8192 =
    /// 0.29 — the other 71% of the card stays free for co-residents.
    #[test]
    fn unit__mistralrs_pa_fraction__geometry_sizes_idle_card() {
        let hf = hf_meta();
        let flags = mistralrs_pa_flags();
        let cfg = Config::default();
        let hw = cotenant_hw(8_192, 8_192, 16_000); // idle: free == total
        let overlay = ModelOverride {
            ctx: Some(32_768), // deterministic KV: 896 MiB
            ..DEFAULT_OVERLAY.clone()
        };
        let inp = mistralrs_hf_input(&hf, &hw, &cfg, &flags, 942 * MIB, &overlay);
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(
            p.argv
                .windows(2)
                .any(|w| w == ["--pa-memory-fraction", "0.29"]),
            "idle card must get geometry-sized 0.29, got {:?}",
            p.argv
        );
        assert!(p
            .warnings
            .iter()
            .any(|w| w.contains("sized from model geometry")));
    }

    /// Geometry caps under a co-tenant too: free 6144 gives a budget
    /// fraction of 0.57, but geometry only NEEDS 0.29 — min wins, so
    /// the tenant keeps its memory and the spawn stops there.
    #[test]
    fn unit__mistralrs_pa_fraction__geometry_and_cotenant_take_the_min() {
        let hf = hf_meta();
        let flags = mistralrs_pa_flags();
        let cfg = Config::default();
        let hw = cotenant_hw(8_192, 6_144, 16_000);
        let overlay = ModelOverride {
            ctx: Some(32_768),
            ..DEFAULT_OVERLAY.clone()
        };
        let inp = mistralrs_hf_input(&hf, &hw, &cfg, &flags, 942 * MIB, &overlay);
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(
            p.argv
                .windows(2)
                .any(|w| w == ["--pa-memory-fraction", "0.29"]),
            "min(geometry 0.29, budget 0.57) must emit 0.29, got {:?}",
            p.argv
        );
    }

    /// No geometry (GGUF file on the mistralrs lane) + idle card: the
    /// historical no-op stands — nothing to correct without geometry.
    #[test]
    fn unit__mistralrs_pa_fraction__gguf_idle_card_stays_noop() {
        let g = meta();
        let flags = mistralrs_pa_flags();
        let cfg = Config::default();
        let hw = cotenant_hw(8_192, 8_192, 16_000);
        let mut inp = input(&g, &hw, &cfg, &flags);
        inp.engine_kind = crate::engine_kind::EngineKind::MistralRs;
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(
            !p.argv.iter().any(|a| a == "--pa-memory-fraction"),
            "GGUF without geometry must stay no-op, got {:?}",
            p.argv
        );
    }

    /// When even a minimal KV pool does not fit next to the tenant, the
    /// auto lane falls back to classic KV (--paged-attn off) with a loud
    /// numbers warning instead of serving silently-truncated context.
    #[test]
    fn unit__mistralrs_pa_fraction__nothing_fits_auto_falls_back_to_classic_kv() {
        let g = meta();
        let flags = mistralrs_pa_flags();
        let cfg = Config::default();
        // resident 800 = 50% of free 1550 (auto-off 75% heuristic does
        // NOT fire) but budget 1550-800-512 = 238 < 256 → classic KV.
        let hw = cotenant_hw(8_192, 1_550, 16_000);
        let mut inp = input(&g, &hw, &cfg, &flags);
        inp.engine_kind = crate::engine_kind::EngineKind::MistralRs;
        inp.model_bytes = 800 * MIB;
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(
            p.argv.windows(2).any(|w| w == ["--paged-attn", "off"]),
            "tight-but-under-heuristic spawn must fall back to classic KV, got {:?}",
            p.argv
        );
        assert!(!p.argv.windows(2).any(|w| w[0] == "--pa-memory-fraction"));
        assert!(p
            .warnings
            .iter()
            .any(|w| w.contains("classic ctx-sized KV")));
    }

    /// An explicit `mistralrs_paged_attn = true` pin is respected even
    /// when nothing fits: warn with numbers, but never emit a conflicting
    /// `--paged-attn off` after the user forced it on.
    #[test]
    fn unit__mistralrs_pa_fraction__forced_on_never_conflicts() {
        let g = meta();
        let flags = mistralrs_pa_flags();
        let cfg = Config {
            mistralrs_paged_attn: Some(true),
            ..Config::default()
        };
        let hw = cotenant_hw(8_192, 1_550, 16_000);
        let mut inp = input(&g, &hw, &cfg, &flags);
        inp.engine_kind = crate::engine_kind::EngineKind::MistralRs;
        inp.model_bytes = 800 * MIB;
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(p.argv.windows(2).any(|w| w == ["--paged-attn", "on"]));
        assert!(!p.argv.windows(2).any(|w| w == ["--paged-attn", "off"]));
        assert!(!p.argv.windows(2).any(|w| w[0] == "--pa-memory-fraction"));
        assert!(p
            .warnings
            .iter()
            .any(|w| w.contains("expect a loud load failure")));
    }

    /// Single-tenant GPUs (free ≈ total) are exactly today's argv: the
    /// engine's own defaults are already correct there — no derivation,
    /// no warning, no behavior change.
    #[test]
    fn unit__mistralrs_pa_fraction__single_tenant_unchanged() {
        let g = meta();
        let flags = mistralrs_pa_flags();
        let cfg = Config::default();
        let hw = gpu_hw(24_000, 32_000, 8);
        let mut inp = input(&g, &hw, &cfg, &flags);
        inp.engine_kind = crate::engine_kind::EngineKind::MistralRs;
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(!p.argv.windows(2).any(|w| w[0] == "--pa-memory-fraction"));
        assert!(!p.argv.windows(2).any(|w| w == ["--paged-attn", "off"]));
        assert!(!p.warnings.iter().any(|w| w.contains("co-tenant")));
    }

    /// An explicit `mistralrs_pa_memory_fraction` pin always wins — the
    /// derivation never second-guesses an operator.
    #[test]
    fn unit__mistralrs_pa_fraction__explicit_pin_wins() {
        let g = meta();
        let flags = mistralrs_pa_flags();
        let cfg = Config {
            mistralrs_pa_memory_fraction: Some(0.40),
            ..Config::default()
        };
        let hw = cotenant_hw(8_192, 1_843, 16_000);
        let mut inp = input(&g, &hw, &cfg, &flags);
        inp.engine_kind = crate::engine_kind::EngineKind::MistralRs;
        inp.model_bytes = 942 * MIB;
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(
            p.argv
                .windows(2)
                .any(|w| w == ["--pa-memory-fraction", "0.4"]),
            "explicit pin must emit verbatim, got {:?}",
            p.argv
        );
        assert!(!p.warnings.iter().any(|w| w.contains("co-tenant")));
    }

    fn llama_server_knobs_flags() -> BTreeSet<String> {
        [
            "--sse-ping-interval",
            "--timeout",
            "--chat-template-kwargs",
            "--no-cont-batching",
            "--reuse-port",
            "--lora-init-without-apply",
        ]
        .into_iter()
        .map(String::from)
        .collect()
    }

    fn llama_server_cfg() -> Config {
        Config {
            sse_ping_interval: Some(-1),
            server_timeout_secs: Some(600),
            chat_template_kwargs: Some(r#"{"enable_thinking": false}"#.into()),
            cont_batching: Some(false),
            reuse_port: true,
            lora_init_without_apply: true,
            ..Config::default()
        }
    }

    #[test]
    fn unit__llama_server_knobs__emitted_with_negation() {
        let g = meta();
        let cfg = llama_server_cfg();
        let mut flags = full_flags();
        flags.extend(llama_server_knobs_flags());
        let hw = gpu_hw(24_000, 32_000, 8);
        let p = compile(&input(&g, &hw, &cfg, &flags), &TuningOverrides::default()).unwrap();
        let a = &p.argv;
        assert!(a.windows(2).any(|w| w == ["--sse-ping-interval", "-1"]));
        assert!(a.windows(2).any(|w| w == ["--timeout", "600"]));
        assert!(a
            .windows(2)
            .any(|w| w == ["--chat-template-kwargs", r#"{"enable_thinking": false}"#]));
        // Some(false) is the NEGATION spelling, never the positive flag.
        assert!(a.iter().any(|t| t == "--no-cont-batching"));
        assert!(!a.iter().any(|t| t == "--cont-batching"));
        assert!(a.iter().any(|t| t == "--reuse-port"));
        assert!(a.iter().any(|t| t == "--lora-init-without-apply"));
        for key in [
            "sse_ping_interval",
            "server_timeout_secs",
            "chat_template_kwargs",
            "cont_batching",
            "reuse_port",
            "lora_init_without_apply",
        ] {
            assert!(
                !p.warnings.iter().any(|w| w.contains(key)),
                "{key} must not warn on a full-flag engine: {:?}",
                p.warnings
            );
        }
    }

    #[test]
    fn unit__llama_server_knobs__warn_skip_on_old_engine() {
        let g = meta();
        let cfg = llama_server_cfg();
        // full_flags() deliberately lacks the six new flags (house rule:
        // shared fixture stays minimal) — exactly the old-engine shape.
        let flags = full_flags();
        let hw = gpu_hw(24_000, 32_000, 8);
        let p = compile(&input(&g, &hw, &cfg, &flags), &TuningOverrides::default()).unwrap();
        for flag in [
            "--sse-ping-interval",
            "--timeout",
            "--chat-template-kwargs",
            "--no-cont-batching",
            "--reuse-port",
            "--lora-init-without-apply",
        ] {
            assert!(!p.argv.iter().any(|t| t == flag), "{flag} must be gated");
        }
        for key in [
            "sse_ping_interval",
            "server_timeout_secs",
            "chat_template_kwargs",
            "cont_batching",
            "reuse_port",
            "lora_init_without_apply",
        ] {
            assert!(
                p.warnings.iter().any(|w| w.contains(key)),
                "{key} warn-skip missing: {:?}",
                p.warnings
            );
        }
    }

    fn mistralrs_knobs_flags() -> BTreeSet<String> {
        [
            "--lora",
            "--enable-lora",
            "--max-batch-size",
            "--max-prefill-chunk-tokens",
            "--max-decode-steps-before-prefill",
            "--prefix-cache-n",
            "--pa-block-size",
            "--pa-cache-type",
            "--pa-context-len",
            "--lora-max-rank",
            "--lora-max-adapters",
            "--lora-max-bytes",
            "--mtp",
            "--mtp-model",
            "--mtp-n-predict",
            "--mtp-draft-sampling",
            "--encoder-cache-memory-mb",
            "--max-num-images",
            "--max-image-length",
            "--disable-metrics",
            "--disable-access-log",
            "--device-layers",
        ]
        .into_iter()
        .map(String::from)
        .collect()
    }

    fn full_mistralrs_tuning() -> MistralrsTuning {
        MistralrsTuning {
            enable_lora: Some(true),
            max_batch_size: Some(4),
            max_prefill_chunk_tokens: Some(1024),
            max_decode_steps_before_prefill: Some(16),
            prefix_cache_n: Some(0),
            pa_block_size: Some(64),
            pa_cache_type: Some("bf16".into()),
            pa_context_len: Some(4096),
            lora_max_rank: Some(64),
            lora_max_adapters: Some(2),
            lora_max_bytes: Some(1_073_741_824),
            mtp: Some(true),
            mtp_model: Some("mtp-draft".into()),
            mtp_n_predict: Some(3),
            mtp_draft_sampling: Some("greedy".into()),
            encoder_cache_memory_mb: Some(512),
            max_num_images: Some(2),
            max_image_length: Some(1024),
            disable_metrics: Some(true),
            disable_access_log: Some(true),
            device_layers: Some("0:12".into()),
        }
    }

    #[test]
    fn unit__mistralrs_knobs__emit_across_all_families() {
        let g = meta();
        let cfg = Config {
            mistralrs: full_mistralrs_tuning(),
            ..Config::default()
        };
        let loras = [("loras/style.gguf".to_string(), 2.0)];
        let hw = gpu_hw(24_000, 32_000, 8);
        let flags = mistralrs_knobs_flags();
        let mut inp = input(&g, &hw, &cfg, &flags);
        inp.engine_kind = crate::engine_kind::EngineKind::MistralRs;
        inp.loras = &loras;
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        let a = &p.argv;
        for pair in [
            ("--max-batch-size", "4"),
            ("--max-prefill-chunk-tokens", "1024"),
            ("--max-decode-steps-before-prefill", "16"),
            ("--prefix-cache-n", "0"),
            ("--pa-block-size", "64"),
            ("--pa-cache-type", "bf16"),
            ("--pa-context-len", "4096"),
            ("--lora-max-rank", "64"),
            ("--lora-max-adapters", "2"),
            ("--lora-max-bytes", "1073741824"),
            ("--mtp-model", "mtp-draft"),
            ("--mtp-n-predict", "3"),
            ("--mtp-draft-sampling", "greedy"),
            ("--encoder-cache-memory-mb", "512"),
            ("--max-num-images", "2"),
            ("--max-image-length", "1024"),
            ("--device-layers", "0:12"),
            ("--lora", "style=loras/style.gguf"),
        ] {
            assert!(
                a.windows(2).any(|w| w == [pair.0, pair.1]),
                "missing {pair:?}"
            );
        }
        for bare in [
            "--mtp",
            "--disable-metrics",
            "--disable-access-log",
            "--enable-lora",
        ] {
            assert!(a.iter().any(|t| t == bare), "missing {bare}");
        }
        // --mtp is the family switch: it must precede its dependents.
        let mtp = a.iter().position(|t| t == "--mtp").unwrap();
        let mtp_model = a.iter().position(|t| t == "--mtp-model").unwrap();
        assert!(mtp < mtp_model, "--mtp must lead its dependents");
        // Single-GPU box: device_layers still emits, with the teaching.
        assert!(p
            .warnings
            .iter()
            .any(|w| w.contains("single GPU") && w.contains("device_layers")));
        // GGUF adapters are first-class on mistral.rs; scale warns.
        assert!(p
            .warnings
            .iter()
            .any(|w| w.contains("ignored on the mistralrs engine")));
    }

    /// `extra_args` on the mistralrs lane: manifest-gated strict
    /// passthrough (the sglang contract) — known flags ride the argv,
    /// unknown flags fail with the manifest pointer, pallama-owned
    /// flags refuse instead of double-setting.
    #[test]
    fn unit__mistralrs_extra_args__strict_passthrough_and_refusals() {
        let g = meta();
        let hw = gpu_hw(24_000, 32_000, 8);
        let cfg = Config::default();
        let mut flags = mistralrs_knobs_flags();
        for f in ["--isq", "--host", "--port", "-m", "--no-ui"] {
            flags.insert(f.to_string());
        }
        let base = input(&g, &hw, &cfg, &flags);

        let isq = crate::config::ModelOverride {
            extra_args: Some(vec!["--isq".into(), "q4k".into()]),
            ..crate::config::ModelOverride::default()
        };
        let p = compile(
            &ProfileInput {
                engine_kind: crate::engine_kind::EngineKind::MistralRs,
                model_bytes: 900 * MIB,
                overlay: &isq,
                ..base.clone()
            },
            &TuningOverrides::default(),
        )
        .expect("known extra_args compile");
        assert!(
            p.argv.windows(2).any(|w| w == ["--isq", "q4k"]),
            "isq pair must ride the argv"
        );

        let bad = crate::config::ModelOverride {
            extra_args: Some(vec!["--ctx-size".into(), "4096".into()]),
            ..crate::config::ModelOverride::default()
        };
        let err = compile(
            &ProfileInput {
                engine_kind: crate::engine_kind::EngineKind::MistralRs,
                model_bytes: 900 * MIB,
                overlay: &bad,
                ..base.clone()
            },
            &TuningOverrides::default(),
        )
        .expect_err("llama-dialect flag must fail loudly");
        assert!(
            err.contains("not in this mistralrs engine's probed manifest"),
            "teaching must point at the manifest: {err}"
        );

        let res = crate::config::ModelOverride {
            extra_args: Some(vec!["--host".into(), "0.0.0.0".into()]),
            ..crate::config::ModelOverride::default()
        };
        let err = compile(
            &ProfileInput {
                engine_kind: crate::engine_kind::EngineKind::MistralRs,
                model_bytes: 900 * MIB,
                overlay: &res,
                ..base.clone()
            },
            &TuningOverrides::default(),
        )
        .expect_err("reserved flag must refuse");
        assert!(
            err.contains("reserved") && err.contains("pallama owns it"),
            "reserved refusal must name the owner: {err}"
        );
    }

    #[test]
    fn unit__mistralrs_tuning_flags__lockstep_with_compile_emission() {
        // The argv translator forwards exactly the flags in
        // MISTRALRS_TUNING_{VALUE,BOOL}_FLAGS. Diff a full-knob compile
        // (plus a LoRA adapter, whose --lora token also rides the
        // passthrough) against a default compile: every flag the knobs
        // add must be listed (else the child argv silently drops it),
        // and every listed flag must be emitted (else the list is stale).
        let g = meta();
        let hw = gpu_hw(24_000, 32_000, 8);
        let flags = mistralrs_knobs_flags();
        let loras = [("loras/style.gguf".to_string(), 2.0)];
        let cfg_full = Config {
            mistralrs: full_mistralrs_tuning(),
            ..Config::default()
        };
        let mut inp_full = input(&g, &hw, &cfg_full, &flags);
        inp_full.engine_kind = crate::engine_kind::EngineKind::MistralRs;
        inp_full.loras = &loras;
        let argv_full = compile(&inp_full, &TuningOverrides::default())
            .unwrap()
            .argv;
        let cfg_base = Config::default();
        let mut inp_base = input(&g, &hw, &cfg_base, &flags);
        inp_base.engine_kind = crate::engine_kind::EngineKind::MistralRs;
        let argv_base = compile(&inp_base, &TuningOverrides::default())
            .unwrap()
            .argv;

        let emitted_by_knobs: std::collections::BTreeSet<&str> = argv_full
            .iter()
            .filter(|t| t.starts_with("--") && !argv_base.contains(*t))
            .map(String::as_str)
            .collect();
        let listed: std::collections::BTreeSet<&str> = MISTRALRS_TUNING_VALUE_FLAGS
            .iter()
            .chain(MISTRALRS_TUNING_BOOL_FLAGS.iter())
            .copied()
            .collect();
        for t in &emitted_by_knobs {
            assert!(
                listed.contains(t),
                "compile emits {t} but the translator flag lists lack it — the child argv would drop it"
            );
        }
        for f in &listed {
            assert!(
                argv_full.iter().any(|t| t == f),
                "translator lists {f} but no knob or adapter lane emits it — stale entry"
            );
        }
        assert!(
            !emitted_by_knobs.is_empty(),
            "diff produced nothing — the fixture stopped exercising the knobs"
        );
    }

    #[test]
    fn unit__mistralrs_knobs__lora_caps_require_serving() {
        // Bare LoRA runtime limits (no enable_lora, no preloaded adapters)
        // are rejected by upstream at boot — warn and suppress them.
        let g = meta();
        let cfg = Config {
            mistralrs: MistralrsTuning {
                lora_max_rank: Some(64),
                lora_max_adapters: Some(2),
                ..MistralrsTuning::default()
            },
            ..Config::default()
        };
        let hw = gpu_hw(24_000, 32_000, 8);
        let flags = mistralrs_knobs_flags();
        let mut inp = input(&g, &hw, &cfg, &flags);
        inp.engine_kind = crate::engine_kind::EngineKind::MistralRs;
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(
            !p.argv.iter().any(|t| t.starts_with("--lora-max")),
            "{:?}",
            p.argv
        );
        assert!(!p.argv.contains(&"--enable-lora".to_string()));
        assert!(
            p.warnings.iter().any(|w| w.contains("no LoRA serving")),
            "{:?}",
            p.warnings
        );
    }

    #[test]
    fn unit__mistralrs_knobs__warn_skip_on_old_engine() {
        let g = meta();
        let cfg = Config {
            mistralrs: MistralrsTuning {
                max_batch_size: Some(4),
                mtp: Some(true),
                mtp_model: Some("mtp-draft".into()),
                device_layers: Some("0:12".into()),
                ..MistralrsTuning::default()
            },
            ..Config::default()
        };
        let empty: BTreeSet<String> = BTreeSet::new();
        let hw = gpu_hw(24_000, 32_000, 8);
        let mut inp = input(&g, &hw, &cfg, &empty);
        inp.engine_kind = crate::engine_kind::EngineKind::MistralRs;
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        for flag in [
            "--max-batch-size",
            "--mtp",
            "--mtp-model",
            "--device-layers",
        ] {
            assert!(!p.argv.iter().any(|t| t == flag), "{flag} must be gated");
        }
        for key in [
            "mistralrs.max_batch_size",
            "mistralrs.mtp",
            "mistralrs.mtp_model",
            "mistralrs.device_layers",
        ] {
            assert!(
                p.warnings.iter().any(|w| w.contains(key)),
                "{key} warn-skip missing: {:?}",
                p.warnings
            );
        }
    }

    #[test]
    fn unit__auto_slots__vulkan_class_with_projector_caps_conservatively() {
        let g = meta();
        let cfg = Config::default();
        // Mixed census — integrated iGPU alongside the discrete card —
        // is the vulkan-class signature (a CUDA census lists CUDA
        // devices only). 5900 MiB vram: headroom 688 MiB admits exactly
        // one 16384-ctx slot at the conservative per-token charge.
        let vulkan_hw = Hardware {
            physical_cores: 8,
            total_ram_mib: 64_000,
            gpus: vec![
                GpuInfo {
                    name: "iGPU".into(),
                    description: "Intel(R) Graphics (RPL-S)".into(),
                    total_mib: 10_256,
                    free_mib: 10_256,
                },
                GpuInfo {
                    name: "RTX".into(),
                    description: "NVIDIA CUDA".into(),
                    total_mib: 5_900,
                    free_mib: 5_900,
                },
            ],
        };
        let mmproj = std::env::temp_dir().join("pallama-test-mmproj.gguf");
        std::fs::write(&mmproj, vec![0u8; 1024]).unwrap();
        let mut inp = input(&g, &vulkan_hw, &cfg, &ALL_FLAGS);
        inp.mmproj_path = Some(mmproj.to_str().unwrap());
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        // Device truth: the vulkan axis (per-slot KV + measured
        // per-token compute) refuses the full 16384 slot, but the
        // autofit re-spend at 4096 honestly earns np2 on the card.
        assert!(p.argv.windows(2).any(|w| w == ["-np", "2"]), "{:?}", p.argv);
        assert!(p.argv.windows(2).any(|w| w == ["--ctx-size", "8192"]));
        assert!(
            p.warnings
                .iter()
                .any(|w| w.contains("slots auto capped at 1") && w.contains("vulkan-class")),
            "{:?}",
            p.warnings
        );

        // CUDA-class census with the same projector: the classic 85%
        // axis gates it — 5000 weights + 2x448 KV do not fit 5900 MiB
        // at the share, and no shallower re-spend pays either.
        let hw = gpu_hw(5_900, 64_000, 8);
        let mut inp2 = input(&g, &hw, &cfg, &ALL_FLAGS);
        inp2.mmproj_path = Some(mmproj.to_str().unwrap());
        let p2 = compile(&inp2, &TuningOverrides::default()).unwrap();
        assert!(p2.argv.windows(2).any(|w| w == ["-np", "1"]));
        assert!(!p2.warnings.iter().any(|w| w.contains("capped")));

        // Vulkan-class census WITHOUT a projector: classic axis only —
        // same single-slot truth, no conservative-cap warning.
        let p3 = compile(
            &input(&g, &vulkan_hw, &cfg, &ALL_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(p3.argv.windows(2).any(|w| w == ["-np", "1"]));
        assert!(!p3.warnings.iter().any(|w| w.contains("capped")));
    }

    #[test]
    fn unit__auto_slots__scoped_hardware_full_census_still_caps() {
        // Product shape (supervisor spawn path): the auto GPU pick
        // scopes `hardware` to the PICKED card only, while the engine
        // census stays mixed — the vulkan build enumerated the iGPU
        // alongside the discrete card. Build-class detection must read
        // the census, or the cap silently dies at spawn time
        // (measured live: scoped census spawned np4/65536 and rode the
        // VRAM cliff that the mixed-census cap was built to avoid).
        let g = meta();
        let cfg = Config::default();
        let mixed_census = vec![
            GpuInfo {
                name: "iGPU".into(),
                description: "Intel(R) Graphics (RPL-S)".into(),
                total_mib: 10_256,
                free_mib: 10_256,
            },
            GpuInfo {
                name: "RTX".into(),
                description: "NVIDIA GeForce RTX 4070 Laptop GPU".into(),
                total_mib: 5_900,
                free_mib: 5_900,
            },
        ];
        let scoped_hw = Hardware {
            physical_cores: 8,
            total_ram_mib: 64_000,
            gpus: vec![mixed_census[1].clone()],
        };
        let mmproj = std::env::temp_dir().join("pallama-test-mmproj-scoped.gguf");
        std::fs::write(&mmproj, vec![0u8; 1024]).unwrap();
        let mut inp = input(&g, &scoped_hw, &cfg, &ALL_FLAGS);
        inp.mmproj_path = Some(mmproj.to_str().unwrap());
        inp.engine_census = mixed_census;
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        // The census (not the scoped hardware) drives the vulkan-class
        // detection: conservative full-slot cap + honest shallow
        // re-spend, exactly like the unscoped mixed-box case.
        assert!(
            p.argv.windows(2).any(|w| w == ["-np", "2"]),
            "scoped hardware must not hide the vulkan-class build: {:?}",
            p.argv
        );
        assert!(p.warnings.iter().any(|w| w.contains("vulkan-class")));
    }

    #[test]
    fn unit__auto_slots__autofit_divides_default_ctx_not_pins() {
        let g = meta(); // kv/slot @16384 = 896 MiB, train 40960
        let hw = gpu_hw(24_000, 64_000, 8);
        // RAM-clamped card (1 GiB effective cache-ram): single slot at
        // full ctx, but the halved ctx earns capacity — 4x4096 wins.
        let cfg_tight = Config {
            cache_ram_mb: 1024,
            ..Config::default()
        };
        let p = compile(
            &input(&g, &hw, &cfg_tight, &ALL_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert_eq!(p.ctx_autofit, Some((4_096, 4)));
        assert!(p.argv.windows(2).any(|w| w == ["-np", "4"]));
        assert!(p.argv.windows(2).any(|w| w == ["--ctx-size", "16384"]));
        assert_eq!(p.ctx, 4_096);
        // Auto-fit rationale is journal-only now.
        assert!(!p.warnings.iter().any(|w| w.contains("slots auto")));

        // Tuning pin (bench grid): hard pin, never divided.
        let p2 = compile(
            &input(&g, &hw, &cfg_tight, &ALL_FLAGS),
            &TuningOverrides {
                ctx: Some(16_384),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(p2.ctx_autofit, None);
        assert!(p2.argv.windows(2).any(|w| w == ["-np", "1"]));
        assert!(p2.argv.windows(2).any(|w| w == ["--ctx-size", "16384"]));

        // Overlay pin: same hard-pin semantics.
        let ov = ModelOverride {
            ctx: Some(16_384),
            ..Default::default()
        };
        let mut inp3 = input(&g, &hw, &cfg_tight, &ALL_FLAGS);
        inp3.overlay = &ov;
        let p3 = compile(&inp3, &TuningOverrides::default()).unwrap();
        assert_eq!(p3.ctx_autofit, None);
        assert!(p3.argv.windows(2).any(|w| w == ["-np", "1"]));

        // Default ctx already at the floor: nothing to divide.
        let cfg_floor = Config {
            default_ctx: 4_096,
            ..Config::default()
        };
        let p4 = compile(
            &input(&g, &hw, &cfg_floor, &ALL_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert_eq!(p4.ctx_autofit, None);
        // 8 GiB+ card: the raised auto ceiling (8) now binds where the
        // old cap of 4 used to.
        assert!(p4.argv.windows(2).any(|w| w == ["-np", "8"]));

        // Comfortable card with a shallow default: train cap at the base
        // ctx, no autofit trade.
        let cfg_shallow = Config {
            default_ctx: 8_192,
            ..Config::default()
        };
        let p5 = compile(
            &input(&g, &hw, &cfg_shallow, &ALL_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert_eq!(p5.ctx_autofit, None);
        assert!(p5.argv.windows(2).any(|w| w == ["-np", "5"]));
        assert!(p5.argv.windows(2).any(|w| w == ["--ctx-size", "40960"]));
        assert_eq!(p5.ctx, 8_192);
    }

    #[test]
    fn unit__kv_est__unified_lane_charges_full_device_pool() {
        // Device truth: the unified pool is VRAM-resident (live-verified
        // b10948), so the planner charge is the full f16 estimate —
        // and the quant ladder applies to it like the classic lane
        // (q8_0 halves the buffer the child actually allocates).
        let hw = gpu_hw(24_000, 64_000, 8);
        let g = meta();
        let cfg = Config::default();
        let inp = input(&g, &hw, &cfg, &ALL_FLAGS);
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        let total: u64 = p
            .argv
            .windows(2)
            .find(|w| w[0] == "--ctx-size")
            .and_then(|w| w[1].parse::<u32>().ok())
            .map(u64::from)
            .expect("--ctx-size present");
        assert_eq!(p.kv_est_bytes, g.kv_f16_bytes(total));
        let t = TuningOverrides {
            kv_quant: Some(true),
            ..Default::default()
        };
        let p2 = compile(&inp, &t).unwrap();
        assert_eq!(
            p2.kv_est_bytes,
            g.kv_f16_bytes(total).map(|f16| f16 / 2),
            "quant ladder halves the device pool estimate"
        );
    }

    #[test]
    fn unit__kv_est__draft_kv_charged_unified() {
        // Spec pair on unified KV: BOTH pools are device-backed (the
        // flag shares buffers, it does not relocate them) — the planner
        // charge is dense f16 + draft f16 at the compiled total ctx.
        let hw = gpu_hw(24_000, 64_000, 8);
        let g = meta();
        let draft = GgufMeta {
            block_count: Some(2),
            ..meta()
        };
        let cfg = Config {
            spec: "eagle3".into(),
            slots: 1,
            ..Config::default()
        };
        let mut inp =
            input_named_with_spec("qwen3-8b", &g, &hw, &cfg, &ALL_FLAGS, &EAGLE3_SPEC_TYPES);
        let path = draft_file("kv-charge");
        inp.draft_path = Some(&path);
        inp.draft_gguf = Some(&draft);
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        let expect = g.kv_f16_bytes(16_384).unwrap() + draft.kv_f16_bytes(16_384).unwrap();
        assert_eq!(p.kv_est_bytes, Some(expect));
    }

    #[test]
    fn unit__kv_est__draft_kv_charged_classic() {
        // Classic KV: dense quantized estimate PLUS the draft's f16 KV.
        // Roomy VRAM keeps the quant ladder off, so dense stays f16.
        let hw = gpu_hw(24_000, 64_000, 8);
        let g = meta();
        let draft = GgufMeta {
            block_count: Some(2),
            ..meta()
        };
        let cfg = Config {
            spec: "eagle3".into(),
            slots: 1,
            kv_unified: Some(false),
            ..Config::default()
        };
        let mut inp =
            input_named_with_spec("qwen3-8b", &g, &hw, &cfg, &ALL_FLAGS, &EAGLE3_SPEC_TYPES);
        let path = draft_file("kv-charge");
        inp.draft_path = Some(&path);
        inp.draft_gguf = Some(&draft);
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        let expect = g.kv_f16_bytes(16_384).unwrap() + draft.kv_f16_bytes(16_384).unwrap();
        assert_eq!(p.kv_est_bytes, Some(expect));
    }

    #[test]
    fn unit__kv_unified_for__truth_table_for_offline_callers() {
        // The gateway num_ctx preflight and `coreside` call this WITHOUT a
        // ProfileInput: same decision as compilation, from (config, model,
        // manifest flags) alone.
        let cfg = Config::default();
        let mut flags = full_flags();
        assert!(kv_unified_for(&cfg, "qwen3-8b", &flags));
        flags.remove("--kv-unified");
        assert!(!kv_unified_for(&cfg, "qwen3-8b", &flags), "engine gate");
        let opt_out = Config {
            kv_unified: Some(false),
            ..Config::default()
        };
        let flags = full_flags();
        assert!(
            !kv_unified_for(&opt_out, "qwen3-8b", &flags),
            "user pin wins"
        );
    }

    #[test]
    fn unit__admission_floor_bytes__charges_standing_overheads() {
        // Same standing charges the unified gpu-layers pin trusts — the
        // 2026-09-11 crash: weights-only admission let a 0.5B sibling
        // (floor 1712 MiB) join a 7302 MiB measured 9B VL resident on an
        // 8 GiB card. Saturating: a u64-weights row must not wrap.
        let mib = |m: u64| m * 1024 * 1024;
        assert_eq!(
            admission_floor_bytes(mib(500), 0),
            mib(500 + 512 + 700),
            "weights + KV floor + spawn overhead"
        );
        assert_eq!(
            admission_floor_bytes(mib(5417), mib(875)),
            mib(5417 + 875 + 512 + 700),
            "projector bytes join the charge"
        );
        assert_eq!(
            admission_floor_bytes(u64::MAX, mib(875)),
            u64::MAX,
            "saturates instead of wrapping"
        );
    }

    #[test]
    fn unit__estimate_kv_vram_charge__floor_vs_full() {
        // Device truth: the KV pool is VRAM-resident on BOTH lanes
        // (live-verified b10948 — `CUDA0 KV buffer` with --kv-unified),
        // so the charge is the full f16 estimate regardless of the
        // unified flag. Geometry: 57344 B/token x 16384 = 896 MiB.
        let hw = gpu_hw(24_000, 64_000, 8);
        let g = meta();
        let cfg = Config::default();
        let unified = input(&g, &hw, &cfg, &ALL_FLAGS);
        assert_eq!(
            estimate_kv_vram_charge(&unified, Some(16_384)).map(|b| b / (1024 * 1024)),
            Some(896),
            "unified path charges the full device-backed pool"
        );
        let mut classic_flags = full_flags();
        classic_flags.remove("--kv-unified");
        let classic = input(&g, &hw, &cfg, &classic_flags);
        assert_eq!(
            estimate_kv_vram_charge(&classic, Some(16_384)).map(|b| b / (1024 * 1024)),
            Some(896),
            "classic path keeps the full f16 charge"
        );
    }

    #[test]
    fn unit__mistralrs_pa__auto_off_on_tight_card() {
        let g = meta();
        let mut flags = BTreeSet::new();
        flags.insert("--paged-attn".to_string());
        let cfg = Config::default();
        let hw = gpu_hw(6_500, 32_000, 8); // 5000 MiB weights > 75% of 6500
        let mut tight = input(&g, &hw, &cfg, &flags);
        tight.engine_kind = crate::engine_kind::EngineKind::MistralRs;
        let p = compile(&tight, &TuningOverrides::default()).unwrap();
        assert!(p.argv.windows(2).any(|w| w == ["--paged-attn", "off"]));
        assert!(p.warnings.iter().any(|w| w.contains("paged-attn auto-off")));

        // Comfortable card: the default paged path stands, no flag, no warn.
        let hw_big = gpu_hw(24_000, 32_000, 8);
        let mut comfy = input(&g, &hw_big, &cfg, &flags);
        comfy.engine_kind = crate::engine_kind::EngineKind::MistralRs;
        let p2 = compile(&comfy, &TuningOverrides::default()).unwrap();
        assert!(!p2.argv.iter().any(|a| a == "--paged-attn"));
        assert!(!p2.warnings.iter().any(|w| w.contains("paged-attn")));
    }

    #[test]
    fn unit__lookup_cache__emits_both_paths() {
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        let cfg = Config {
            lookup_cache_static: Some("/tmp/lc-static.bin".to_string()),
            lookup_cache_dynamic: Some("/tmp/lc-dynamic.bin".to_string()),
            ..Config::default()
        };
        let p = compile(
            &input(&g, &hw, &cfg, &ALL_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(p
            .argv
            .windows(2)
            .any(|w| w == ["--lookup-cache-static", "/tmp/lc-static.bin"]));
        assert!(p
            .argv
            .windows(2)
            .any(|w| w == ["--lookup-cache-dynamic", "/tmp/lc-dynamic.bin"]));
    }

    #[test]
    fn unit__spec_auto__draft_missing__opportunistic_dense_with_warning() {
        // Live evidence (b10840): `--spec-draft-hf` resolves the repo to an
        // empty draft path and the child exits fatally — so auto (now the
        // default) must degrade to dense with a pull hint instead of
        // refusing the spawn; hard errors belong to the explicit modes.
        let cfg = Config {
            spec: "auto".into(),
            ..Config::default()
        };
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        let mut flags = ALL_FLAGS.clone();
        flags.insert("--spec-draft-hf".to_string());
        let mut inp = input(&g, &hw, &cfg, &flags);
        inp.model_name = "qwen3-8b"; // catalog pair, draft NOT pulled
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(p.warnings.iter().any(|w| w.contains("not pulled")
            && w.contains("pallama pull ggml-org/Qwen3-0.6B-GGUF:Q4_0")));
        assert!(!p.argv.contains(&"--spec-type".to_string()));
    }

    #[test]
    fn unit__spec_auto__no_pair_for_model__dense_without_user_noise() {
        let cfg = Config {
            spec: "auto".into(),
            ..Config::default()
        };
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        let mut inp = input(&g, &hw, &cfg, &ALL_FLAGS);
        inp.model_name = "gemma3-4b"; // no catalog pair
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(!p.argv.contains(&"--spec-type".to_string()));
        // Auto-mode fallback to dense is the system working as designed —
        // rationale lives in the journal, not the warning vec.
        assert!(!p.warnings.iter().any(|w| w.contains("draft pair")));
    }

    #[test]
    fn unit__missing_flag_on_engine__hard_error_names_flag() {
        let mut flags = full_flags();
        flags.remove("--sleep-idle-seconds");
        let cfg = Config::default();
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        let err = compile(&input(&g, &hw, &cfg, &flags), &TuningOverrides::default()).unwrap_err();
        assert!(err.contains("--sleep-idle-seconds"), "{err}");
        assert!(err.contains("engine use"), "{err}");
    }

    #[test]
    fn unit__overlay_extra_args__appended_and_validated() {
        let cfg = Config::default();
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        let mut flags = full_flags();
        flags.insert("--tensor-split".to_string());
        let mut inp = input(&g, &hw, &cfg, &flags);
        let overlay = ModelOverride {
            extra_args: Some(vec!["--tensor-split".into(), "3,1".into()]),
            ..Default::default()
        };
        inp.overlay = &overlay;
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--tensor-split" && w[1] == "3,1"));

        // Unknown extra flag -> error.
        let bad = ModelOverride {
            extra_args: Some(vec!["--no-such-flag".into()]),
            ..Default::default()
        };
        inp.overlay = &bad;
        let err = compile(&inp, &TuningOverrides::default()).unwrap_err();
        assert!(err.contains("--no-such-flag"), "{err}");
    }

    #[test]
    fn unit__overlay_chat_template__emitted_and_engine_gated() {
        let cfg = Config::default();
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        let mut inp = input(&g, &hw, &cfg, &ALL_FLAGS);
        let o_tpl = ModelOverride {
            chat_template: Some("{{ custom }}".into()),
            ..Default::default()
        };
        inp.overlay = &o_tpl;
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--chat-template" && w[1] == "{{ custom }}"));

        // Engine without the flag: warn-skip, argv clean, compile still OK.
        let sparse: BTreeSet<String> = ALL_FLAGS
            .iter()
            .filter(|f| !f.starts_with("--chat-template"))
            .cloned()
            .collect();
        let mut inp2 = input(&g, &hw, &cfg, &sparse);
        let o_file = ModelOverride {
            chat_template_file: Some("/x.tpl".into()),
            ..Default::default()
        };
        inp2.overlay = &o_file;
        let p2 = compile(&inp2, &TuningOverrides::default()).unwrap();
        assert!(
            !p2.argv.iter().any(|a| a.starts_with("--chat-template")),
            "must not emit on a flagless engine"
        );
        assert!(
            p2.warnings.iter().any(|w| w.contains("chat_template")),
            "warn: {:?}",
            p2.warnings
        );
    }

    #[test]
    fn unit__overlay_sampler_defaults__argv_flags_and_gating() {
        let cfg = Config {
            slots: 1,           // purpose-scoped: sampler flags, not slot sizing
            spec: "off".into(), // purpose-scoped: not the auto spec lane
            ..Config::default()
        };
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        let sd = SamplerDefaults {
            temperature: Some(0.7),
            top_k: Some(40),
            min_p: Some(0.05),
            repeat_penalty: Some(1.1),
            mirostat: Some(2),
            seed: Some(-1),
            ..Default::default()
        };
        let mut inp = input(&g, &hw, &cfg, &ALL_FLAGS);
        let o_sd = ModelOverride {
            sampler_defaults: Some(sd),
            ..Default::default()
        };
        inp.overlay = &o_sd;
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        for (flag, val) in [
            ("--temp", "0.7"),
            ("--top-k", "40"),
            ("--min-p", "0.05"),
            ("--repeat-penalty", "1.1"),
            ("--mirostat", "2"),
            ("--seed", "-1"),
        ] {
            assert!(
                p.argv.windows(2).any(|w| w[0] == flag && w[1] == val),
                "missing {flag} {val} in {:?}",
                p.argv
            );
        }
        assert!(p.warnings.is_empty(), "{:?}", p.warnings);

        // Flagless engine: every sampler flag warn-skips.
        let sampler_flags = [
            "--temp",
            "--top-k",
            "--top-p",
            "--min-p",
            "--top-n-sigma",
            "--typical-p",
            "--repeat-penalty",
            "--repeat-last-n",
            "--presence-penalty",
            "--frequency-penalty",
            "--dry-multiplier",
            "--dry-base",
            "--dry-allowed-length",
            "--dry-penalty-last-n",
            "--xtc-probability",
            "--xtc-threshold",
            "--mirostat",
            "--seed",
        ];
        let sparse: BTreeSet<String> = ALL_FLAGS
            .iter()
            .filter(|f| !sampler_flags.contains(&f.as_str()))
            .cloned()
            .collect();
        let mut inp2 = input(&g, &hw, &cfg, &sparse);
        let o_sd2 = ModelOverride {
            sampler_defaults: Some(SamplerDefaults {
                temperature: Some(0.7),
                ..Default::default()
            }),
            ..Default::default()
        };
        inp2.overlay = &o_sd2;
        let p2 = compile(&inp2, &TuningOverrides::default()).unwrap();
        assert!(!p2.argv.contains(&"--temp".to_string()));
        assert!(
            p2.warnings.iter().any(|w| w.contains("--temp")),
            "{:?}",
            p2.warnings
        );
    }

    #[test]
    fn unit__tuning_overrides__fa_and_batch_emitted() {
        let cfg = Config::default();
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        let t = TuningOverrides {
            fa: Some(false),
            batch: Some(1024),
            ..Default::default()
        };
        let p = compile(&input(&g, &hw, &cfg, &ALL_FLAGS), &t).unwrap();
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--flash-attn" && w[1] == "off"));
        assert!(p.argv.windows(2).any(|w| w[0] == "-b" && w[1] == "1024"));
    }

    #[test]
    fn unit__tuning_overrides__win_over_heuristics() {
        let cfg = Config {
            slots: 1, // purpose-scoped: tuning precedence, not slot sizing
            ..Config::default()
        };
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        let t = TuningOverrides {
            ctx: Some(8192),
            threads: Some(6),
            kv_quant: Some(true),
            ..Default::default()
        };
        let p = compile(&input(&g, &hw, &cfg, &ALL_FLAGS), &t).unwrap();
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--ctx-size" && w[1] == "8192"));
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--threads" && w[1] == "6"));
        assert!(p.argv.contains(&"--cache-type-k".to_string()));
        assert_eq!(p.ctx, 8192);
    }

    #[test]
    fn unit__mmproj_from_extra_args__emitted_as_dash_mm() {
        let cfg = Config::default();
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        let mut inp = input(&g, &hw, &cfg, &ALL_FLAGS);
        let overlay = ModelOverride {
            extra_args: Some(vec!["--mmproj".into(), "/models/mmproj.gguf".into()]),
            ..Default::default()
        };
        inp.overlay = &overlay;
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "-mm" && w[1] == "/models/mmproj.gguf"));
        // The raw --mmproj passthrough also remains (harmless duplicate of
        // intent; llama-server takes the last one).
    }

    #[test]
    fn unit__mmproj_from_store__emitted_when_engine_supports() {
        // Rule 19: a pulled projector reaches the engine when the policy
        // asks for it. Default is Lazy (text-only spawn + @vision respawn
        // on first image); Attach — explicit or the @vision/router
        // mmproj_force — emits -mm from the store row (the gap that
        // once left downloaded projectors dead on disk).
        let cfg = Config::default();
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();

        // Lazy default: text-only spawn, no -mm, lazy teaching warning
        let mut inp = input(&g, &hw, &cfg, &ALL_FLAGS);
        inp.mmproj_path = Some("/models/mmproj-F16.gguf");
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(
            !p.argv.iter().any(|a| a == "-mm"),
            "lazy default spawns text-only: {:?}",
            p.argv
        );

        // Attach policy: pulled projector emitted
        let attach_cfg = Config {
            mmproj_policy: Some(MmprojPolicy::Attach),
            ..Config::default()
        };
        let mut inp_a = input(&g, &hw, &attach_cfg, &ALL_FLAGS);
        inp_a.mmproj_path = Some("/models/mmproj-F16.gguf");
        let p_a = compile(&inp_a, &TuningOverrides::default()).unwrap();
        assert!(
            p_a.argv
                .windows(2)
                .any(|w| w[0] == "-mm" && w[1] == "/models/mmproj-F16.gguf"),
            "attach must wire the pulled projector: {:?}",
            p_a.argv
        );

        // mmproj_force (@vision respawn / router preset): attached even
        // under the lazy default
        let mut inp_f = input(&g, &hw, &cfg, &ALL_FLAGS);
        inp_f.mmproj_path = Some("/models/mmproj-F16.gguf");
        inp_f.mmproj_force = true;
        let p_f = compile(&inp_f, &TuningOverrides::default()).unwrap();
        assert!(
            p_f.argv
                .windows(2)
                .any(|w| w[0] == "-mm" && w[1] == "/models/mmproj-F16.gguf"),
            "forced attach must wire the pulled projector: {:?}",
            p_f.argv
        );
    }

    #[test]
    fn unit__mmproj_store__extra_args_override_wins() {
        let cfg = Config::default();
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        let mut inp = input(&g, &hw, &cfg, &ALL_FLAGS);
        inp.mmproj_path = Some("/models/pulled.gguf");
        let overlay = ModelOverride {
            extra_args: Some(vec!["--mmproj".into(), "/models/custom.gguf".into()]),
            ..Default::default()
        };
        inp.overlay = &overlay;
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "-mm" && w[1] == "/models/custom.gguf"));
        assert!(!p
            .argv
            .windows(2)
            .any(|w| w[0] == "-mm" && w[1] == "/models/pulled.gguf"));
    }

    #[test]
    fn unit__mmproj_store__unsupported_engine_skips_with_warning() {
        let cfg = Config::default();
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        let mut flags = ALL_FLAGS.clone();
        flags.retain(|f| f != "--mmproj" && f != "-mm");
        let mut inp = input(&g, &hw, &cfg, &flags);
        inp.mmproj_path = Some("/models/mmproj-F16.gguf");
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(
            !p.argv.iter().any(|a| a == "-mm"),
            "no -mm without manifest support"
        );
    }

    // ---- wire-everything wave pins -------------------------------------

    fn wire_flags() -> BTreeSet<String> {
        // The wire-everything wave folded its modern-engine additions
        // into the canonical full_flags fixture; this alias keeps the
        // intent-named entry point for the wire battery.
        full_flags()
    }

    static WIRE_FLAGS: LazyLock<BTreeSet<String>> = LazyLock::new(wire_flags);

    #[test]
    fn unit__wire__defaults_emit_nothing_new() {
        // Golden: an all-default config compiles to the SAME argv shape
        // as before the wire wave (no new flags appear).
        let cfg = Config::default();
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        let inp = input(&g, &hw, &cfg, &WIRE_FLAGS);
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        for f in [
            "--reasoning-budget",
            "--spec-draft-ngl",
            "--no-warmup",
            "--no-repack",
            "--no-cache-idle-slots",
            "--yarn-orig-ctx",
            "--override-kv",
            "--control-vector",
            "--prio",
            "--image-max-tokens",
            "--lazy-mode",
            "--tools",
            "--tools-runtime",
            "--mcp-servers-config",
            "--mcp-servers-json",
        ] {
            assert!(!p.argv.iter().any(|a| a == f), "{f} must stay unset");
        }
    }

    #[test]
    fn unit__lazy_mode__emitted_only_on_deviation() {
        let cfg = Config {
            lazy_mode: "on".into(),
            slots: 1,           // purpose-scoped: lazy-mode, not slot sizing
            spec: "off".into(), // purpose-scoped: not the auto spec lane
            ..Default::default()
        };
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        let p = compile(
            &input(&g, &hw, &cfg, &ALL_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--lazy-mode" && w[1] == "on"));
        // Overlay wins over the global pin.
        let over = ModelOverride {
            lazy_mode: Some("off".into()),
            ..DEFAULT_OVERLAY.clone()
        };
        let mut inp2 = input(&g, &hw, &cfg, &ALL_FLAGS);
        inp2.overlay = &over;
        let p2 = compile(&inp2, &TuningOverrides::default()).unwrap();
        assert!(
            p2.argv
                .windows(2)
                .any(|w| w[0] == "--lazy-mode" && w[1] == "off"),
            "{:?}",
            p2.argv
        );
        assert!(
            p.warnings.is_empty() && p2.warnings.is_empty(),
            "p: {:?} p2: {:?}",
            p.warnings,
            p2.warnings
        );
    }

    #[test]
    fn unit__lazy_mode__engine_lacking__teaching_warning() {
        let cfg = Config {
            lazy_mode: "on".into(),
            slots: 1, // purpose-scoped: lazy-mode, not slot sizing
            ..Default::default()
        };
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        // Flags set WITHOUT --lazy-mode: build a trimmed copy of ALL_FLAGS.
        let flags: BTreeSet<String> = ALL_FLAGS
            .iter()
            .filter(|f| f.as_str() != "--lazy-mode")
            .cloned()
            .collect();
        let p = compile(&input(&g, &hw, &cfg, &flags), &TuningOverrides::default()).unwrap();
        assert!(!p.argv.contains(&"--lazy-mode".to_string()));
        assert!(
            p.warnings
                .iter()
                .any(|w| w.contains("lazy_mode") && w.contains("engine update")),
            "{:?}",
            p.warnings
        );
    }

    #[test]
    fn unit__server_tools__emits_quartet_when_set() {
        let mcp = tempfile::NamedTempFile::new().unwrap();
        let cfg = Config {
            server_tools: Some("grep_search,read_file".into()),
            server_tools_runtime: Some("ssh:gpu-box".into()),
            mcp_servers_config: Some(mcp.path().to_string_lossy().into_owned()),
            mcp_servers_json: Some(r#"{"mcpServers":{"fs":{}}}"#.into()),
            slots: 1,           // purpose-scoped: tool passthrough, not slot sizing
            spec: "off".into(), // purpose-scoped: not the auto spec lane
            ..Default::default()
        };
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        let p = compile(
            &input(&g, &hw, &cfg, &ALL_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        for (flag, val) in [
            ("--tools", "grep_search,read_file"),
            ("--tools-runtime", "ssh:gpu-box"),
            ("--mcp-servers-config", mcp.path().to_str().unwrap()),
            ("--mcp-servers-json", r#"{"mcpServers":{"fs":{}}}"#),
        ] {
            assert!(
                p.argv.windows(2).any(|w| w[0] == flag && w[1] == val),
                "{flag} missing: {:?}",
                p.argv
            );
        }
        assert!(p.warnings.is_empty(), "{:?}", p.warnings);
    }

    #[test]
    fn unit__server_tools__missing_mcp_config__hard_error() {
        let cfg = Config {
            mcp_servers_config: Some("/nope/does-not-exist.json".into()),
            slots: 1,
            ..Default::default()
        };
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        let err = compile(
            &input(&g, &hw, &cfg, &ALL_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap_err();
        assert!(
            err.contains("does not exist") && err.contains("fix the path"),
            "{err}"
        );
    }

    #[test]
    fn unit__server_tools__engine_lacking_flags__teaching_warning() {
        let mcp = tempfile::NamedTempFile::new().unwrap();
        let cfg = Config {
            server_tools: Some("all".into()),
            server_tools_runtime: Some("ssh:gpu-box".into()),
            mcp_servers_config: Some(mcp.path().to_string_lossy().into_owned()),
            mcp_servers_json: Some(r#"{"mcpServers":{"fs":{}}}"#.into()),
            slots: 1,
            ..Default::default()
        };
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        let flags: BTreeSet<String> = ALL_FLAGS
            .iter()
            .filter(|f| {
                !matches!(
                    f.as_str(),
                    "--tools" | "--tools-runtime" | "--mcp-servers-config" | "--mcp-servers-json"
                )
            })
            .cloned()
            .collect();
        let p = compile(&input(&g, &hw, &cfg, &flags), &TuningOverrides::default()).unwrap();
        assert!(!p.argv.iter().any(|a| a.starts_with("--tools")));
        assert!(!p.argv.iter().any(|a| a.starts_with("--mcp")));
        for flag in [
            "--tools",
            "--tools-runtime",
            "--mcp-servers-config",
            "--mcp-servers-json",
        ] {
            assert!(
                p.warnings
                    .iter()
                    .any(|w| w.contains(flag) && w.contains("lacks the flag")),
                "{flag} warning missing: {:?}",
                p.warnings
            );
        }
    }

    #[test]
    fn unit__wire__reasoning_knobs_and_overlay() {
        let cfg = Config {
            reasoning_budget: 256,
            reasoning_effort: "low".into(),
            reasoning_preserve: Some(false),
            reasoning: "on".into(),
            ..Default::default()
        };
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        let inp = input(&g, &hw, &cfg, &WIRE_FLAGS);
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--reasoning-budget" && w[1] == "256"));
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--reasoning-effort" && w[1] == "low"));
        assert!(p.argv.iter().any(|a| a == "--no-reasoning-preserve"));
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--reasoning" && w[1] == "on"));
        // Default (empty) never emits the flag — child keeps its own
        // auto-detect default, argv byte-identical to pre-knob.
        let p_default = compile(
            &input(&g, &hw, &Config::default(), &WIRE_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(
            !p_default.argv.iter().any(|a| a == "--reasoning"),
            "default config must not pass --reasoning"
        );
        // Per-model overlay wins over the global budget and the switch.
        let overlay = ModelOverride {
            reasoning_budget: Some(64),
            reasoning: Some("off".into()),
            ..DEFAULT_OVERLAY.clone()
        };
        let mut inp = input(&g, &hw, &cfg, &WIRE_FLAGS);
        inp.overlay = &overlay;
        let p2 = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(
            p2.argv
                .windows(2)
                .any(|w| w[0] == "--reasoning-budget" && w[1] == "64"),
            "overlay budget must win"
        );
        assert!(
            p2.argv
                .windows(2)
                .any(|w| w[0] == "--reasoning" && w[1] == "off"),
            "overlay reasoning switch must win"
        );
    }

    #[test]
    fn unit__wire__ngram_typed_flags_and_warn_skip() {
        let cfg = Config {
            spec: "ngram".into(),
            ngram_size_m: 32,
            ngram_min_hits: 2,
            ..Default::default()
        };
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        // Full engine: typed flags emitted.
        let inp = input(&g, &hw, &cfg, &WIRE_FLAGS);
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--spec-ngram-simple-size-m" && w[1] == "32"));
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--spec-ngram-simple-min-hits" && w[1] == "2"));
        // Old engine without the typed flags: warn-skip, spec still on.
        let mut old_flags = WIRE_FLAGS.clone();
        for f in [
            "--spec-ngram-simple-size-m",
            "--spec-ngram-simple-size-n",
            "--spec-ngram-simple-min-hits",
        ] {
            old_flags.remove(f);
        }
        let inp = input(&g, &hw, &cfg, &old_flags);
        let p2 = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(p2
            .argv
            .windows(2)
            .any(|w| w[0] == "--spec-type" && w[1] == "ngram-simple"));
        assert!(p2
            .warnings
            .iter()
            .any(|w| w.contains("--spec-ngram-simple-size-m")));
    }

    #[test]
    fn unit__spec_ngram_typed__family_dispatch() {
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        // map-k: generic size knobs feed the map-k flag family.
        let cfg = Config {
            spec: "ngram-map-k".into(),
            ngram_size_m: 64,
            ngram_size_n: 16,
            ngram_min_hits: 2,
            ..Default::default()
        };
        let p = compile(
            &input(&g, &hw, &cfg, &WIRE_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--spec-type" && w[1] == "ngram-map-k"));
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--spec-ngram-map-k-size-m" && w[1] == "64"));
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--spec-ngram-map-k-size-n" && w[1] == "16"));
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--spec-ngram-map-k-min-hits" && w[1] == "2"));
        assert!(!p.argv.iter().any(|a| a.starts_with("--spec-ngram-simple")));
        // mod: its own knob family, upstream defaults 24/64/48.
        let cfg = Config {
            spec: "ngram-mod".into(),
            ngram_mod_n_match: 32,
            ngram_mod_n_max: 96,
            ngram_mod_n_min: 12,
            ..Default::default()
        };
        let p = compile(
            &input(&g, &hw, &cfg, &WIRE_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--spec-type" && w[1] == "ngram-mod"));
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--spec-ngram-mod-n-match" && w[1] == "32"));
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--spec-ngram-mod-n-max" && w[1] == "96"));
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--spec-ngram-mod-n-min" && w[1] == "12"));
        // cache: parameterless — only --spec-type, but the rule-14 lookup
        // cache still rides when spec_cache is on.
        let cfg = Config {
            spec: "ngram-cache".into(),
            ..Default::default()
        };
        let p = compile(
            &input(&g, &hw, &cfg, &WIRE_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--spec-type" && w[1] == "ngram-cache"));
        assert!(!p.argv.iter().any(|a| a.starts_with("--spec-ngram-")));
        assert!(p.argv.windows(2).any(|w| w[0] == "--lookup-cache-dynamic"));
    }

    #[test]
    fn unit__overlay_spm_infill__emitted_and_engine_gated() {
        let cfg = Config {
            slots: 1,           // purpose-scoped: spm flag gating, not slot sizing
            spec: "off".into(), // purpose-scoped: not the auto spec lane
            ..Config::default()
        };
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        let o = ModelOverride {
            spm_infill: Some(true),
            ..Default::default()
        };
        let mut inp = input(&g, &hw, &cfg, &WIRE_FLAGS);
        inp.overlay = &o;
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(p.argv.contains(&"--spm-infill".to_string()));
        // Old engine: warn-skip, argv clean, compile still OK.
        let sparse: BTreeSet<String> = WIRE_FLAGS
            .iter()
            .filter(|f| !f.starts_with("--spm-infill"))
            .cloned()
            .collect();
        let mut inp2 = input(&g, &hw, &cfg, &sparse);
        inp2.overlay = &o;
        let p2 = compile(&inp2, &TuningOverrides::default()).unwrap();
        assert!(!p2.argv.contains(&"--spm-infill".to_string()));
        assert!(p2.warnings.iter().any(|w| w.contains("--spm-infill")));
        // Some(false) = explicitly off: nothing emitted, no warning.
        let o_off = ModelOverride {
            spm_infill: Some(false),
            ..Default::default()
        };
        let mut inp3 = input(&g, &hw, &cfg, &WIRE_FLAGS);
        inp3.overlay = &o_off;
        let p3 = compile(&inp3, &TuningOverrides::default()).unwrap();
        assert!(!p3.argv.contains(&"--spm-infill".to_string()));
        assert!(p3.warnings.is_empty(), "{:?}", p3.warnings);
    }

    #[test]
    fn unit__wire__sched_vision_yarn_and_negations() {
        let cfg = Config {
            cpu_strict: true,
            prio: 2,
            poll_batch: Some(false),
            image_max_tokens: 1024,
            mmproj_offload: false,
            yarn_orig_ctx: 4096,
            warmup: false,
            keep_tokens: 64,
            ..Default::default()
        };
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        let inp = input(&g, &hw, &cfg, &WIRE_FLAGS);
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--cpu-strict" && w[1] == "1"));
        assert!(p.argv.windows(2).any(|w| w[0] == "--prio" && w[1] == "2"));
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--poll-batch" && w[1] == "0"));
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--image-max-tokens" && w[1] == "1024"));
        assert!(p.argv.iter().any(|a| a == "--no-mmproj-offload"));
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--yarn-orig-ctx" && w[1] == "4096"));
        assert!(p.argv.iter().any(|a| a == "--no-warmup"));
        assert!(p.argv.windows(2).any(|w| w[0] == "--keep" && w[1] == "64"));
    }

    #[test]
    fn unit__wire__override_kv_and_control_vectors_repeatable() {
        let cfg = Config {
            override_kv: vec![
                "tokenizer.ggml.add_bos_token=bool:false".into(),
                "qwen35.rope.dimension_sections=int:4".into(),
            ],
            control_vectors: vec!["/cv/steer.gguf".into()],
            control_vectors_scaled: vec!["/cv/soft.gguf:0.5".into()],
            control_vector_layer_range: "0-10".into(),
            ..Default::default()
        };
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        let inp = input(&g, &hw, &cfg, &WIRE_FLAGS);
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert_eq!(
            p.argv.iter().filter(|a| *a == "--override-kv").count(),
            2,
            "both kv overrides emitted"
        );
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--control-vector" && w[1] == "/cv/steer.gguf"));
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--control-vector-scaled" && w[1] == "/cv/soft.gguf:0.5"));
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--control-vector-layer-range" && w[1] == "0-10"));
    }

    #[test]
    fn unit__wire__per_model_devices_replace_global() {
        let cfg = Config {
            devices: vec!["Vulkan1".into()],
            ..Default::default()
        };
        let overlay = ModelOverride {
            devices: Some(vec!["CUDA0".into()]),
            ..DEFAULT_OVERLAY.clone()
        };
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        let mut flags = WIRE_FLAGS.clone();
        flags.insert("--device".to_string());
        let mut inp = input(&g, &hw, &cfg, &flags);
        inp.overlay = &overlay;
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--device" && w[1] == "CUDA0"));
        assert!(!p
            .argv
            .windows(2)
            .any(|w| w[0] == "--device" && w[1] == "Vulkan1"));
    }

    #[test]
    fn unit__wire__tensor_preset_expands() {
        let cfg = Config {
            tensor_preset: "moe-cpu-offload".into(),
            ..Default::default()
        };
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        let inp = input(&g, &hw, &cfg, &WIRE_FLAGS);
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--override-tensor" && w[1] == "exps=CPU"));
    }

    #[test]
    fn unit__wire__cache_hint_adapts_ram_cap() {
        // 16384 (not the shipped 8192 default): a user-pinned budget, so
        // the adaptive clamp is a divergence worth a warning — the tier
        // numbers below stay visible through that warning.
        let cfg = Config {
            cache_ram_mb: 16384,
            ..Default::default()
        };
        let hw = gpu_hw(12_000, 16_384, 8); // 16 GiB RAM
        let g = meta();
        // Hot prefix traffic: 40% cap = 6553.
        let mut inp = input(&g, &hw, &cfg, &WIRE_FLAGS);
        inp.cache_hit_rate = Some(0.8);
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--cache-ram" && w[1] == "6553"));
        // Cold: 20% cap = 3276, but the 2a-bis weights floor (5000 + 64
        // over 0.85 = 5957) lifts every below-floor cap to 5957 — a
        // sub-weights budget CPU-splits layers at the fitter.
        let mut inp = input(&g, &hw, &cfg, &WIRE_FLAGS);
        inp.cache_hit_rate = Some(0.01);
        let p2 = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(p2
            .argv
            .windows(2)
            .any(|w| w[0] == "--cache-ram" && w[1] == "5957"));
        // The tier itself stays visible in the clamp warning.
        assert!(p2.warnings.iter().any(|w| w.contains("clamped to 3276")));
        // None: static 30% = 4915 — also below the floor.
        let inp = input(&g, &hw, &cfg, &WIRE_FLAGS);
        let p3 = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(p3
            .argv
            .windows(2)
            .any(|w| w[0] == "--cache-ram" && w[1] == "5957"));
    }

    #[test]
    fn unit__wire__kv_est_bytes_quantized() {
        // Classic KV layout (kv_unified off): the f16-equivalent estimate
        // and its quantized grades are the test's subject. The unified
        // floor has its own test below.
        let cfg = Config {
            slots: 1,
            kv_unified: Some(false),
            ..Config::default()
        };
        let hw = gpu_hw(120_000, 64_000, 8); // huge VRAM: no ladder quant
        let g = meta();
        let inp = input(&g, &hw, &cfg, &WIRE_FLAGS);
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        let f16 = p.kv_est_bytes.expect("geometry present");
        // 2*28*8*64*ctx*2 with ctx=16384:
        assert_eq!(f16, 2 * 28 * 8 * 64 * 16_384 * 2);
        // Forced q8_0 halves it.
        let t = TuningOverrides {
            kv_quant: Some(true),
            ..Default::default()
        };
        let p2 = compile(&inp, &t).unwrap();
        assert_eq!(p2.kv_est_bytes, Some(f16 / 2));
    }

    #[test]
    fn unit__wire__hybrid_linear_without_metadata_warns_and_counts_all_layers() {
        let cfg = Config {
            slots: 1,
            kv_unified: Some(false),
            ..Config::default()
        };
        let hw = gpu_hw(120_000, 64_000, 8); // huge VRAM: no ladder quant
        let mut g = meta();
        g.architecture = "kimi-k3".into();
        // No recurrent_layers array and no full_attention_interval: the
        // split is unprovable, so the estimate must stay a full-layer upper
        // bound and the profile must say so.
        let inp = input(&g, &hw, &cfg, &WIRE_FLAGS);
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        let full = p.kv_est_bytes.expect("geometry present");
        assert_eq!(full, 2 * 28 * 8 * 64 * 16_384 * 2);
        assert!(
            p.warnings.iter().any(|w| w.contains("hybrid-linear")),
            "expected hybrid-linear upper-bound warning, got {:?}",
            p.warnings
        );
    }

    #[test]
    fn unit__wire__hybrid_linear_interval_fractional_kv_no_warning() {
        let cfg = Config {
            slots: 1,
            kv_unified: Some(false),
            ..Config::default()
        };
        let hw = gpu_hw(120_000, 64_000, 8); // huge VRAM: no ladder quant
        let mut g = meta();
        // qwen35-class: interval 4 over 28 blocks -> 7 full-attention
        // layers; the other 21 are recurrent (gated delta net) and carry no
        // per-token KV. The split is provable, so no warning and the KV
        // estimate is the fractional truth downstream consumers see.
        g.architecture = "qwen35".into();
        g.full_attention_interval = Some(4);
        let inp = input(&g, &hw, &cfg, &WIRE_FLAGS);
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert_eq!(
            p.kv_est_bytes,
            Some(2 * 7 * 8 * 64 * 16_384 * 2),
            "expected 7/28 of the full-attention KV"
        );
        assert!(
            !p.warnings.iter().any(|w| w.contains("hybrid-linear")),
            "provable split must not warn, got {:?}",
            p.warnings
        );
        // Downstream ladder math consumes the fractional number: a forced
        // q8_0 grade halves the quarter estimate, not a full-layer one.
        let t = TuningOverrides {
            kv_quant: Some(true),
            ..Default::default()
        };
        let p2 = compile(&inp, &t).unwrap();
        assert_eq!(p2.kv_est_bytes, Some(2 * 7 * 8 * 64 * 16_384 * 2 / 2));
    }

    #[test]
    fn unit__compile__mistralrs_dialect_minimal_profile() {
        // The manifest flag gate must NOT run for mistral.rs engines:
        // the llama-server dialect is foreign and --jinja/--ctx-size
        // style flags would hard-error against a foreign flag set (the
        // live 500 that birthed this fork). The profile is ctx plus
        // `-np N` for the argv translator to mine — nothing else.
        // Explicit slots=1 pins the emission; slots=0 (the new default)
        // emits nothing and lets the translator's own default apply.
        let cfg = Config {
            slots: 1,
            ..Config::default()
        };
        let hw = gpu_hw(0, 32_000, 8);
        let g = meta();
        let empty: BTreeSet<String> = BTreeSet::new();
        let mut inp = input(&g, &hw, &cfg, &empty);
        inp.engine_kind = crate::engine_kind::EngineKind::MistralRs;
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert_eq!(p.argv, vec!["-np".to_string(), "1".to_string()]);
        assert!(p.warnings.is_empty());
        assert_eq!(p.kv_est_bytes, None);
        // Default config (slots=0) also emits nothing on this dialect —
        // pallama's llama-lane auto sizing has no mistral.rs equivalent
        // (upstream sizes its own sequences).
        let cfg_def = Config::default();
        let mut inp_def = input(&g, &hw, &cfg_def, &empty);
        inp_def.engine_kind = crate::engine_kind::EngineKind::MistralRs;
        let p_def = compile(&inp_def, &TuningOverrides::default()).unwrap();
        assert!(p_def.argv.is_empty(), "{:?}", p_def.argv);
        assert!(p_def.warnings.is_empty(), "{:?}", p_def.warnings);
        // pa-memory-fraction rides the dialect argv when the binary has
        // the flag; a missing flag teaches instead of silently dropping.
        let mut flags = BTreeSet::new();
        flags.insert("--pa-memory-fraction".to_string());
        let cfg_frac = Config {
            mistralrs_pa_memory_fraction: Some(0.35),
            ..Default::default()
        };
        let mut inp_flagged = input(&g, &hw, &cfg_frac, &flags);
        inp_flagged.engine_kind = crate::engine_kind::EngineKind::MistralRs;
        let pf = compile(&inp_flagged, &TuningOverrides::default()).unwrap();
        let i = pf
            .argv
            .iter()
            .position(|a| a == "--pa-memory-fraction")
            .expect("fraction flag emitted");
        assert_eq!(pf.argv[i + 1], "0.35");
        let mut inp_unflagged = input(&g, &hw, &cfg_frac, &empty);
        inp_unflagged.engine_kind = crate::engine_kind::EngineKind::MistralRs;
        let pn = compile(&inp_unflagged, &TuningOverrides::default()).unwrap();
        assert!(
            !pn.argv.iter().any(|a| a == "--pa-memory-fraction"),
            "unsupported flag must not emit"
        );
        assert_eq!(pn.warnings.len(), 1);
        // paged-attn off: the only fit for big vision models on 8 GB
        // cards (live-proven) — same gate shape as the fraction knob.
        let cfg_pa = Config {
            mistralrs_paged_attn: Some(false),
            ..Default::default()
        };
        let mut flags_pa = BTreeSet::new();
        flags_pa.insert("--paged-attn".to_string());
        let mut inppa = input(&g, &hw, &cfg_pa, &flags_pa);
        inppa.engine_kind = crate::engine_kind::EngineKind::MistralRs;
        let ppa = compile(&inppa, &TuningOverrides::default()).unwrap();
        let ipa = ppa
            .argv
            .iter()
            .position(|a| a == "--paged-attn")
            .expect("paged-attn emitted");
        assert_eq!(ppa.argv[ipa + 1], "off");
        let mut inppa2 = input(&g, &hw, &cfg_pa, &empty);
        inppa2.engine_kind = crate::engine_kind::EngineKind::MistralRs;
        let ppa2 = compile(&inppa2, &TuningOverrides::default()).unwrap();
        assert!(!ppa2.argv.iter().any(|a| a == "--paged-attn"));
        assert_eq!(ppa2.warnings.len(), 1);
    }

    #[test]
    fn unit__compile_mistralrs__extra_args_llama_dialect_refused() {
        // audit GAP-2, post-strict-passthrough contract: extra_args with
        // llama-server dialect on the mistralrs lane hard-fails at
        // compile with manifest teaching (the old warn-and-drop silently
        // disabled engine-only flags — see the ISQ A/B that measured
        // nothing). Unknown-to-the-manifest = Err, never silent.
        let hw = gpu_hw(0, 32_000, 8);
        let g = meta();
        let empty: BTreeSet<String> = BTreeSet::new();
        let ov = ModelOverride {
            extra_args: Some(vec!["--jinja".into(), "--flash-attn".into()]),
            ..DEFAULT_OVERLAY.clone()
        };
        let cfg = Config::default();
        let mut inp = input(&g, &hw, &cfg, &empty);
        inp.engine_kind = crate::engine_kind::EngineKind::MistralRs;
        inp.overlay = &ov;
        let err = compile(&inp, &TuningOverrides::default())
            .expect_err("llama-dialect extra_args must be refused loudly");
        assert!(err.contains("--jinja"), "{err}");
        assert!(
            err.contains("not in this mistralrs engine's probed manifest"),
            "{err}"
        );
        // Empty extra_args stays silent — no warning noise for no action.
        let ov_empty = ModelOverride {
            extra_args: Some(Vec::new()),
            ..DEFAULT_OVERLAY.clone()
        };
        let mut inp2 = input(&g, &hw, &cfg, &empty);
        inp2.engine_kind = crate::engine_kind::EngineKind::MistralRs;
        inp2.overlay = &ov_empty;
        let p2 = compile(&inp2, &TuningOverrides::default()).unwrap();
        assert!(p2.warnings.is_empty(), "{:?}", p2.warnings);
    }

    #[test]
    fn unit__compile__safetensors_on_llamacpp_teaches_working_remedy() {
        // audit GAP-1: the wrong-lane teaching once pointed at
        // `pallama run <model> --engine sglang` — a flag the Run command
        // never had (clap would reject it). The remedy must name the
        // engine install/use lane and the daemon restart.
        let hw = gpu_hw(0, 32_000, 8);
        let g = meta();
        let hf = crate::hfmeta::HfMeta {
            architecture: "Qwen2ForCausalLM".into(),
            ctx_train: Some(32_768),
            dtype: Some("bfloat16".into()),
            quant_bits: None,
            kv: crate::hfmeta::KvGeom {
                layers: Some(28),
                kv_heads: Some(2),
                head_dim: Some(128),
            },
        };
        let empty: BTreeSet<String> = BTreeSet::new();
        let cfg = Config::default();
        let mut inp = input(&g, &hw, &cfg, &empty);
        inp.meta = ModelMeta::Hf(&hf);
        let err = compile(&inp, &TuningOverrides::default()).unwrap_err();
        assert!(err.contains("engine install --kind sglang"), "{err}");
        assert!(err.contains("engine use"), "{err}");
        assert!(err.contains("restart"), "{err}");
        assert!(!err.contains("--engine sglang"), "dead-end flag: {err}");
    }

    #[test]
    fn unit__wire__spec_draft_placement_under_auto() {
        let cfg = Config {
            spec_draft_cpu_range: "0-7".into(),
            spec_draft_ngl: "all".into(),
            spec_draft_p_min: Some(0.2),
            spec: "auto".into(),
            ..Default::default()
        };
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        // spec=auto WITHOUT a resolved draft: no catalog pair for this
        // name -> dense warning; placement stays OFF.
        let mut inp = input(&g, &hw, &cfg, &WIRE_FLAGS);
        inp.model_name = "llama-unknown-8b";
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(
            !p.argv.iter().any(|a| a == "--spec-draft-cpu-range"),
            "{:?}",
            p.argv
        );
        // With the draft resolved (qwen3-8b has a catalog pair): the pair
        // emission plus the full placement battery.
        let mut inp = input(&g, &hw, &cfg, &WIRE_FLAGS);
        let draft = draft_file("wire");
        inp.draft_path = Some(&draft);
        let p2 = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(
            p2.argv
                .windows(2)
                .any(|w| w[0] == "--spec-draft-cpu-range" && w[1] == "0-7"),
            "{:?}",
            p2.argv
        );
        assert!(p2
            .argv
            .windows(2)
            .any(|w| w[0] == "--spec-draft-ngl" && w[1] == "all"));
        assert!(p2
            .argv
            .windows(2)
            .any(|w| w[0] == "--spec-draft-p-min" && w[1].starts_with("0.2")));
    }

    /// `ALL_FLAGS` + the placement flags the sibling tests pin explicitly
    /// (the fixture's flag list predates them).
    fn sibling_flags() -> BTreeSet<String> {
        let mut f = ALL_FLAGS.clone();
        f.insert("--spec-draft-device".to_string());
        f.insert("--mmproj-device".to_string());
        f
    }

    #[test]
    fn unit__siblings__draft_warning_fires_and_pin_silences() {
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        let flags = sibling_flags();
        // spec defaults to "off"; the sibling recommendation only exists
        // for resolved drafts (spec "auto" + draft model present).
        let cfg = Config {
            spec: "auto".into(),
            ..Config::default()
        };
        let mut inp = input(&g, &hw, &cfg, &flags);
        let draft = draft_file("siblings");
        inp.draft_path = Some(&draft);
        inp.sibling_devices = vec!["GPU1".to_string()];
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(
            p.warnings
                .iter()
                .any(|w| w.contains("spec_draft_device = \"GPU1\"")),
            "teaching warning must name the spare card, got: {:?}",
            p.warnings
        );
        // Explicit pin silences the recommendation and emits the flag.
        let pinned_cfg = Config {
            spec: "auto".into(),
            spec_draft_device: "GPU1".into(),
            ..Config::default()
        };
        let mut pinned = input(&g, &hw, &pinned_cfg, &flags);
        pinned.draft_path = Some(&draft);
        pinned.sibling_devices = vec!["GPU1".to_string()];
        let p2 = compile(&pinned, &TuningOverrides::default()).unwrap();
        assert!(!p2.warnings.iter().any(|w| w.contains("spec draft shares")));
        assert!(p2
            .argv
            .windows(2)
            .any(|w| w[0] == "--spec-draft-device" && w[1] == "GPU1"));
    }

    #[test]
    fn unit__auto_tensor_split__emits_with_teaching_and_pin_wins() {
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        let cfg = Config::default();
        let mut inp = input(&g, &hw, &cfg, &ALL_FLAGS);
        inp.auto_tensor_split = Some("2,1".to_string());
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--tensor-split" && w[1] == "2,1"));
        assert!(
            p.warnings
                .iter()
                .any(|w| w.contains("auto tensor-split 2,1") && w.contains("buys capacity")),
            "bandwidth teaching must ride the auto split, got: {:?}",
            p.warnings
        );
        // Manual tensor_split is authoritative: the auto plan is ignored
        // (no duplicate emission, no auto warning).
        let pinned_cfg = Config {
            tensor_split: "3,1".into(),
            ..Config::default()
        };
        let mut pinned = input(&g, &hw, &pinned_cfg, &ALL_FLAGS);
        pinned.auto_tensor_split = Some("2,1".to_string());
        let p2 = compile(&pinned, &TuningOverrides::default()).unwrap();
        assert_eq!(
            p2.argv
                .windows(2)
                .filter(|w| w[0] == "--tensor-split")
                .count(),
            1
        );
        assert!(p2
            .argv
            .windows(2)
            .any(|w| w[0] == "--tensor-split" && w[1] == "3,1"));
    }

    #[test]
    fn unit__auto_tensor_split__engine_without_flag_warns_only() {
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        let cfg = Config::default();
        let mut flags = full_flags();
        flags.remove("--tensor-split");
        let mut inp = input(&g, &hw, &cfg, &flags);
        inp.auto_tensor_split = Some("2,1".to_string());
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(!p.argv.iter().any(|a| a == "--tensor-split"));
        assert!(p
            .warnings
            .iter()
            .any(|w| w.contains("auto tensor-split planned but engine")
                && w.contains("lacks --tensor-split")));
    }

    #[test]
    fn unit__siblings__mmproj_warning_fires_and_pin_silences() {
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        let flags = sibling_flags();
        let cfg = Config::default();
        let mut inp = input(&g, &hw, &cfg, &flags);
        inp.mmproj_path = Some("/models/mmproj.gguf");
        inp.sibling_devices = vec!["GPU1".to_string()];
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(
            p.warnings
                .iter()
                .any(|w| w.contains("mmproj_device = \"GPU1\"")),
            "teaching warning must name the spare card, got: {:?}",
            p.warnings
        );
        let pinned_cfg = Config {
            mmproj_device: "GPU1".into(),
            ..Config::default()
        };
        let mut pinned = input(&g, &hw, &pinned_cfg, &flags);
        pinned.mmproj_path = Some("/models/mmproj.gguf");
        pinned.sibling_devices = vec!["GPU1".to_string()];
        let p2 = compile(&pinned, &TuningOverrides::default()).unwrap();
        assert!(!p2.warnings.iter().any(|w| w.contains("mmproj shares")));
        assert!(p2
            .argv
            .windows(2)
            .any(|w| w[0] == "--mmproj-device" && w[1] == "GPU1"));
    }

    #[test]
    fn unit__siblings__no_sibling_no_warning() {
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        let flags = sibling_flags();
        let cfg = Config {
            spec: "auto".into(),
            ..Config::default()
        };
        let mut inp = input(&g, &hw, &cfg, &flags);
        let draft = draft_file("no-sibling");
        inp.draft_path = Some(&draft);
        inp.mmproj_path = Some("/models/mmproj.gguf");
        // sibling_devices empty (single-card box): stays silent.
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(!p.warnings.iter().any(|w| w.contains("spare discrete card")));
    }

    // ------------------------------------------------------------------
    // sglang: fit ladder + flag contract
    // ------------------------------------------------------------------

    /// The verified sglang 0.5.19 flag surface (`probe_sglang` output shape).
    fn sglang_flags() -> BTreeSet<String> {
        [
            "--context-length",
            "--max-running-requests",
            "--speculative-algorithm",
            "--speculative-draft-model-path",
            "--kv-cache-dtype",
            "--cpu-offload-gb",
            "--mem-fraction-static",
            "--cuda-graph-max-bs",
            "--chunked-prefill-size",
            "--device",
            "--attention-backend",
            "--tool-call-parser",
            "--reasoning-parser",
            "--tokenizer-path",
            "--dtype",
            "--quantization",
            "--schedule-policy",
            "--page-size",
            "--schedule-conservativeness",
            "--max-prefill-tokens",
            "--stream-interval",
            "--random-seed",
            "--enable-metrics",
            "--skip-server-warmup",
            "--enable-torch-compile",
            "--enable-hierarchical-cache",
            "--hicache-ratio",
            "--hicache-size",
        ]
        .iter()
        .map(|f| (*f).to_string())
        .collect()
    }

    /// Qwen2.5-class checkpoint: 28 layers, 2 KV heads, 128 `head_dim`,
    /// 32768 trained ctx, bf16 — KV geometry is the ladder's fuel.
    fn hf_meta() -> crate::hfmeta::HfMeta {
        crate::hfmeta::HfMeta {
            architecture: "Qwen2ForCausalLM".into(),
            ctx_train: Some(32_768),
            dtype: Some("bfloat16".into()),
            quant_bits: None,
            kv: crate::hfmeta::KvGeom {
                layers: Some(28),
                kv_heads: Some(2),
                head_dim: Some(128),
            },
        }
    }

    static SGLANG_FLAGS: LazyLock<BTreeSet<String>> = LazyLock::new(sglang_flags);

    fn sglang_input<'a>(
        hf: &'a crate::hfmeta::HfMeta,
        hw: &'a Hardware,
        cfg: &'a Config,
        weights: u64,
    ) -> ProfileInput<'a> {
        ProfileInput {
            engine_kind: crate::engine_kind::EngineKind::Sglang,
            model_name: "qwen2.5-0.5b",
            instance_key: "qwen2.5-0.5b",
            model_path: "/models/qwen2.5-0.5b.d",
            model_bytes: weights,
            meta: ModelMeta::Hf(hf),
            hardware: hw,
            config: cfg,
            overlay: &DEFAULT_OVERLAY,
            loras: &[],
            draft_path: None,
            draft_gguf: None,
            mmproj_path: None,
            mmproj_force: false,
            engine_tag: "sglang-test",
            supported_flags: &SGLANG_FLAGS,
            spec_types: &[],
            endpoint: Endpoint::Tcp {
                host: "127.0.0.1".into(),
                port: 12345,
            },
            data_dir: "/tmp/pallama-test-data",
            cache_hit_rate: None,
            resident_ram_mib: 0,
            device_hint: None,
            engine_census: hw.gpus.clone(),
            sibling_devices: Vec::new(),
            auto_tensor_split: None,
        }
    }

    /// ctx = 32768 via overlay so KV bytes are deterministic:
    /// kv8 (fp8, 1B/elem) = 28*2*128*2*32768 = 469,762,048; kv16 = 2x;
    /// vram `12_000` MiB = 12,582,912,000 → budget = 92% − 1 GiB = 10,502,537,216.
    fn sglang_ctx_overlay() -> ModelOverride {
        ModelOverride {
            ctx: Some(32_768),
            ..DEFAULT_OVERLAY.clone()
        }
    }

    #[test]
    fn unit__sglang__tier_a_full_gpu_f16_kv() {
        let cfg = Config::default();
        let hw = gpu_hw(12_000, 32_000, 8);
        let hf = hf_meta();
        let overlay = sglang_ctx_overlay();
        let mut inp = sglang_input(&hf, &hw, &cfg, 8_000 * MIB);
        inp.overlay = &overlay;
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert_eq!(p.gpu, "full");
        assert_eq!(p.ctx, 32_768);
        assert_eq!(p.kv_est_bytes, Some(939_524_096));
        // f16 KV is upstream default: no dtype flag in Tier A
        assert!(!p.argv.iter().any(|a| a == "--kv-cache-dtype"));
        assert!(!p.argv.iter().any(|a| a == "--cpu-offload-gb"));
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--context-length" && w[1] == "32768"));
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--mem-fraction-static" && w[1] == "0.741"));
    }

    #[test]
    fn unit__sglang__tier_b_fp8_kv() {
        let cfg = Config::default();
        let hw = gpu_hw(12_000, 32_000, 8);
        let hf = hf_meta();
        let overlay = sglang_ctx_overlay();
        let mut inp = sglang_input(&hf, &hw, &cfg, 9_400 * MIB);
        inp.overlay = &overlay;
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert_eq!(p.gpu, "full");
        assert_eq!(p.kv_est_bytes, Some(469_762_048));
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--kv-cache-dtype" && w[1] == "fp8_e5m2"));
        assert!(p.warnings.iter().any(|w| w.contains("Tier B")));
        // tight-fit knobs engage
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--cuda-graph-max-bs" && w[1] == "4"));
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--chunked-prefill-size" && w[1] == "2048"));
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--mem-fraction-static" && w[1] == "0.821"));
    }

    #[test]
    fn unit__sglang__tier_c_cpu_offload() {
        let cfg = Config::default();
        let hw = gpu_hw(12_000, 32_000, 8);
        let hf = hf_meta();
        let overlay = sglang_ctx_overlay();
        let mut inp = sglang_input(&hf, &hw, &cfg, 10_500 * MIB);
        inp.overlay = &overlay;
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert_eq!(p.gpu, "partial");
        assert_eq!(p.kv_est_bytes, Some(469_762_048));
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--cpu-offload-gb" && w[1] == "1"));
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--kv-cache-dtype" && w[1] == "fp8_e5m2"));
        assert!(p.warnings.iter().any(|w| w.contains("Tier C")));
        // mem-fraction must account for the emitted offload, not re-add it
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--mem-fraction-static" && w[1] == "0.833"));
    }

    #[test]
    fn unit__sglang__tier_d_refusal_not_crash_loop() {
        let cfg = Config::default();
        let hw = gpu_hw(12_000, 8_000, 8);
        let hf = hf_meta();
        let overlay = sglang_ctx_overlay();
        let mut inp = sglang_input(&hf, &hw, &cfg, 13_500 * MIB);
        inp.overlay = &overlay;
        let err = compile(&inp, &TuningOverrides::default()).unwrap_err();
        assert!(err.contains("does not fit this machine"), "got: {err}");
        // teaching content: the user must see the numbers and the levers
        assert!(err.contains("12.53 GiB usable") || err.contains("usable VRAM"));
    }

    #[test]
    fn unit__sglang__cpu_box_device_cpu() {
        let cfg = Config::default();
        let hw = Hardware {
            physical_cores: 4,
            total_ram_mib: 8_000,
            gpus: vec![],
        };
        let hf = hf_meta();
        let overlay = sglang_ctx_overlay();
        let mut inp = sglang_input(&hf, &hw, &cfg, 1_000 * MIB);
        inp.overlay = &overlay;
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert_eq!(p.gpu, "cpu");
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--device" && w[1] == "cpu"));
        assert!(p.warnings.iter().any(|w| w.contains("no GPU detected")));
    }

    #[test]
    fn unit__sglang__gguf_model_refused_with_teaching() {
        let cfg = Config::default();
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        let mut inp = input_with_spec(&g, &hw, &cfg, &SGLANG_FLAGS, &[]);
        inp.engine_kind = crate::engine_kind::EngineKind::Sglang;
        let err = compile(&inp, &TuningOverrides::default()).unwrap_err();
        assert!(err.contains("GGUF file"), "got: {err}");
        assert!(err.contains("safetensors"), "got: {err}");
    }

    #[test]
    fn unit__sglang__hf_model_refused_on_llamacpp() {
        let cfg = Config::default();
        let hw = gpu_hw(12_000, 32_000, 8);
        let hf = hf_meta();
        let mut inp = sglang_input(&hf, &hw, &cfg, 1_000 * MIB);
        inp.engine_kind = crate::engine_kind::EngineKind::default();
        let err = compile(&inp, &TuningOverrides::default()).unwrap_err();
        assert!(err.contains("safetensors"), "got: {err}");
        assert!(err.contains("sglang engine"), "got: {err}");
    }

    #[test]
    fn unit__sglang__reserved_extra_args_hard_error() {
        let cfg = Config::default();
        let hw = gpu_hw(12_000, 32_000, 8);
        let hf = hf_meta();
        let overlay = ModelOverride {
            ctx: Some(32_768),
            extra_args: Some(vec!["--api-key".into(), "evil".into()]),
            ..DEFAULT_OVERLAY.clone()
        };
        let mut inp = sglang_input(&hf, &hw, &cfg, 8_000 * MIB);
        inp.overlay = &overlay;
        let err = compile(&inp, &TuningOverrides::default()).unwrap_err();
        assert!(err.contains("reserved"), "got: {err}");
        // auth is supervisor-minted; the overlay must not be able to touch it
        assert!(err.contains("--api-key"), "got: {err}");
    }

    #[test]
    fn unit__sglang__unknown_extra_args_rejected() {
        let cfg = Config::default();
        let hw = gpu_hw(12_000, 32_000, 8);
        let hf = hf_meta();
        let overlay = ModelOverride {
            ctx: Some(32_768),
            extra_args: Some(vec!["--definitely-not-upstream".into(), "1".into()]),
            ..DEFAULT_OVERLAY.clone()
        };
        let mut inp = sglang_input(&hf, &hw, &cfg, 8_000 * MIB);
        inp.overlay = &overlay;
        let err = compile(&inp, &TuningOverrides::default()).unwrap_err();
        assert!(err.contains("does not support"), "got: {err}");
    }

    #[test]
    fn unit__sglang__valid_extra_args_appended() {
        let cfg = Config::default();
        let hw = gpu_hw(12_000, 32_000, 8);
        let hf = hf_meta();
        let overlay = ModelOverride {
            ctx: Some(32_768),
            extra_args: Some(vec!["--stream-interval".into(), "2".into()]),
            ..DEFAULT_OVERLAY.clone()
        };
        let mut inp = sglang_input(&hf, &hw, &cfg, 8_000 * MIB);
        inp.overlay = &overlay;
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--stream-interval" && w[1] == "2"));
    }

    #[test]
    fn unit__sglang__cache_type_teaches_kv_cache_dtype() {
        let cfg = Config::default();
        let hw = gpu_hw(12_000, 32_000, 8);
        let hf = hf_meta();
        let overlay = ModelOverride {
            ctx: Some(32_768),
            cache_type: Some("q8_0".into()),
            ..DEFAULT_OVERLAY.clone()
        };
        let mut inp = sglang_input(&hf, &hw, &cfg, 8_000 * MIB);
        inp.overlay = &overlay;
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(p
            .warnings
            .iter()
            .any(|w| w.contains("sglang.kv_cache_dtype")));
        assert!(!p.argv.iter().any(|a| a == "--cache-type"));
    }

    #[test]
    fn unit__sglang__ctx_clamped_to_max_position_embeddings() {
        let cfg = Config::default();
        let hw = gpu_hw(12_000, 32_000, 8);
        let hf = hf_meta();
        let overlay = ModelOverride {
            ctx: Some(65_536),
            ..DEFAULT_OVERLAY.clone()
        };
        let mut inp = sglang_input(&hf, &hw, &cfg, 8_000 * MIB);
        inp.overlay = &overlay;
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert_eq!(p.ctx, 32_768);
        assert!(p
            .warnings
            .iter()
            .any(|w| w.contains("clamped to model max_position_embeddings")));
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--context-length" && w[1] == "32768"));
    }

    #[test]
    fn unit__sglang__deterministic_pins_max_running_requests() {
        let cfg = Config::default();
        let hw = gpu_hw(12_000, 32_000, 8);
        let hf = hf_meta();
        let overlay = ModelOverride {
            ctx: Some(32_768),
            slots: Some(8),
            deterministic: Some(true),
            ..DEFAULT_OVERLAY.clone()
        };
        let mut inp = sglang_input(&hf, &hw, &cfg, 8_000 * MIB);
        inp.overlay = &overlay;
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--max-running-requests" && w[1] == "1"));
        assert!(p
            .warnings
            .iter()
            .any(|w| w.contains("pins max-running-requests = 1")));
    }

    #[test]
    fn unit__sglang__eagle3_draft_pair_emitted() {
        let cfg = Config::default();
        let hw = gpu_hw(12_000, 32_000, 8);
        let hf = hf_meta();
        let overlay = sglang_ctx_overlay();
        let draft = draft_file("sglang-eagle3");
        let mut inp = sglang_input(&hf, &hw, &cfg, 8_000 * MIB);
        inp.overlay = &overlay;
        inp.draft_path = Some(&draft);
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--speculative-algorithm" && w[1] == "EAGLE3"));
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--speculative-draft-model-path" && w[1] == draft));
    }

    #[test]
    fn unit__sglang__offload_pin_honored_even_when_it_fits() {
        let cfg = Config::default();
        let hw = gpu_hw(12_000, 32_000, 8);
        let hf = hf_meta();
        let overlay = ModelOverride {
            ctx: Some(32_768),
            sglang: Some(crate::config::SglangTuning {
                cpu_offload_gb: Some(2.0),
                ..crate::config::SglangTuning::default()
            }),
            ..DEFAULT_OVERLAY.clone()
        };
        let mut inp = sglang_input(&hf, &hw, &cfg, 8_000 * MIB);
        inp.overlay = &overlay;
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--cpu-offload-gb" && w[1] == "2"));
        // mem-fraction nets out the pinned offload: (weights − 2GB + kv16)/vram
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--mem-fraction-static" && w[1] == "0.582"));
    }

    #[test]
    fn unit__sglang__missing_kv_geometry_degrades_to_auto() {
        let cfg = Config::default();
        let hw = gpu_hw(12_000, 32_000, 8);
        let mut hf = hf_meta();
        hf.kv = crate::hfmeta::KvGeom {
            layers: None,
            kv_heads: None,
            head_dim: None,
        };
        let overlay = sglang_ctx_overlay();
        let mut inp = sglang_input(&hf, &hw, &cfg, 8_000 * MIB);
        inp.overlay = &overlay;
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert_eq!(p.gpu, "auto");
        assert_eq!(p.kv_est_bytes, None);
        assert!(p.warnings.iter().any(|w| w.contains("lacks KV geometry")));
        // no mem-fraction guess without geometry — sglang defaults apply
        assert!(!p.argv.iter().any(|a| a == "--mem-fraction-static"));
    }

    /// The 0.5.19 surface plus every knob wired after the initial 28-flag
    /// contract: the "new engine" fixture for emission tests.
    fn sglang_flags_extended() -> BTreeSet<String> {
        let mut flags = sglang_flags();
        for flag in [
            "--enable-deterministic-inference",
            "--enable-lora",
            "--lora-paths",
            "--tokenizer-mode",
            "--tokenizer-backend",
            "--tokenizer-worker-num",
            "--detokenizer-worker-num",
            "--enable-dynamic-batch-tokenizer",
            "--dynamic-batch-tokenizer-batch-size",
            "--dynamic-batch-tokenizer-batch-timeout",
            "--grammar-backend",
            "--radix-eviction-policy",
            "--enable-session-radix-cache",
            "--enable-mixed-chunk",
            "--sleep-on-idle",
            "--enable-memory-saver",
            "--watchdog-timeout",
            "--enable-cache-report",
            "--batch-notify-size",
            "--scheduler-recv-interval",
            "--cuda-graph-bs",
            "--cuda-graph-backend-prefill",
            "--max-total-tokens",
            "--tp-size",
            "--dp-size",
            "--pp-size",
            "--ep-size",
            "--max-lora-rank",
            "--lora-backend",
        ] {
            flags.insert(flag.to_string());
        }
        flags
    }

    #[test]
    fn unit__sglang__deterministic_native_flag_additive_with_slots_pin() {
        // Deterministic keeps the slots=1 serialization pin (the actual
        // reproducibility guarantee) AND adds the native
        // --enable-deterministic-inference flag when the engine has it.
        let cfg = Config::default();
        let hw = gpu_hw(12_000, 32_000, 8);
        let hf = hf_meta();
        let flags = sglang_flags_extended();
        let overlay = ModelOverride {
            ctx: Some(32_768),
            deterministic: Some(true),
            ..DEFAULT_OVERLAY.clone()
        };
        let mut inp = sglang_input(&hf, &hw, &cfg, 8_000 * MIB);
        inp.overlay = &overlay;
        inp.supported_flags = &flags;
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--max-running-requests" && w[1] == "1"));
        assert!(p
            .argv
            .contains(&"--enable-deterministic-inference".to_string()));
    }

    #[test]
    fn unit__sglang__deterministic_native_flag_warn_skips_on_old_engine() {
        // Engine without the flag: legacy slots=1 pin still stands, the
        // native flag degrades to a warn-skip (never a spawn error).
        let cfg = Config::default();
        let hw = gpu_hw(12_000, 32_000, 8);
        let hf = hf_meta();
        let overlay = ModelOverride {
            ctx: Some(32_768),
            deterministic: Some(true),
            ..DEFAULT_OVERLAY.clone()
        };
        let mut inp = sglang_input(&hf, &hw, &cfg, 8_000 * MIB);
        inp.overlay = &overlay;
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--max-running-requests" && w[1] == "1"));
        assert!(!p
            .argv
            .contains(&"--enable-deterministic-inference".to_string()));
        assert!(p
            .warnings
            .iter()
            .any(|w| w.contains("--enable-deterministic-inference")));
    }

    #[test]
    fn unit__sglang__new_knobs_emit_across_all_families() {
        // One sweep across every post-28-flag knob: values, bools (flag
        // alone, never a phantom "" positional), and the nargs list.
        let cfg = Config::default();
        let hw = gpu_hw(12_000, 32_000, 8);
        let hf = hf_meta();
        let flags = sglang_flags_extended();
        let overlay = ModelOverride {
            ctx: Some(32_768),
            sglang: Some(crate::config::SglangTuning {
                tokenizer_mode: Some("slow".into()),
                tokenizer_backend: Some("fastokens".into()),
                tokenizer_worker_num: Some(2),
                detokenizer_worker_num: Some(2),
                dynamic_batch_tokenizer: Some(true),
                dynamic_batch_tokenizer_batch_size: Some(8),
                dynamic_batch_tokenizer_batch_timeout: Some(0.01),
                grammar_backend: Some("xgrammar".into()),
                radix_eviction_policy: Some("lfu".into()),
                session_radix_cache: Some(true),
                mixed_chunk: Some(true),
                sleep_on_idle: Some(true),
                memory_saver: Some(true),
                watchdog_timeout: Some(300.0),
                cache_report: Some(true),
                batch_notify_size: Some(32),
                scheduler_recv_interval: Some(2),
                cuda_graph_bs: Some(vec![1, 2, 4]),
                cuda_graph_backend_prefill: Some("disabled".into()),
                max_total_tokens: Some(65_536),
                max_lora_rank: Some(64),
                lora_backend: Some("pytorch".into()),
                ..crate::config::SglangTuning::default()
            }),
            ..DEFAULT_OVERLAY.clone()
        };
        let mut inp = sglang_input(&hf, &hw, &cfg, 8_000 * MIB);
        inp.overlay = &overlay;
        inp.supported_flags = &flags;
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        let pair = |flag: &str, val: &str| {
            assert!(
                p.argv.windows(2).any(|w| w[0] == flag && w[1] == val),
                "missing {flag} {val} in {:?}",
                p.argv
            );
        };
        let bare = |flag: &str| {
            assert!(
                p.argv.contains(&flag.to_string()),
                "missing bool {flag} in {:?}",
                p.argv
            );
        };
        pair("--tokenizer-mode", "slow");
        pair("--tokenizer-backend", "fastokens");
        pair("--tokenizer-worker-num", "2");
        pair("--detokenizer-worker-num", "2");
        bare("--enable-dynamic-batch-tokenizer");
        pair("--dynamic-batch-tokenizer-batch-size", "8");
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--dynamic-batch-tokenizer-batch-timeout" && w[1] == "0.01"));
        pair("--grammar-backend", "xgrammar");
        pair("--radix-eviction-policy", "lfu");
        bare("--enable-session-radix-cache");
        bare("--enable-mixed-chunk");
        bare("--sleep-on-idle");
        bare("--enable-memory-saver");
        pair("--watchdog-timeout", "300");
        bare("--enable-cache-report");
        pair("--batch-notify-size", "32");
        pair("--scheduler-recv-interval", "2");
        pair("--max-total-tokens", "65536");
        pair("--max-lora-rank", "64");
        pair("--lora-backend", "pytorch");
        // nargs list: flag once, then each size as its own token
        let idx = p
            .argv
            .iter()
            .position(|a| a == "--cuda-graph-bs")
            .expect("cuda-graph-bs flag");
        assert_eq!(&p.argv[idx + 1..idx + 4], &["1", "2", "4"]);
        // prefill graph backend rides as a value pair
        pair("--cuda-graph-backend-prefill", "disabled");
        // bool flags never push a phantom empty-string positional
        assert!(
            !p.argv.iter().map(String::as_str).any(str::is_empty),
            "{:?}",
            p.argv
        );
    }

    #[test]
    fn unit__sglang__new_knobs_warn_skip_on_old_engine() {
        // Base fixture lacks every new flag: knobs degrade to warn-skips,
        // argv stays clean — an old engine never receives unknown flags.
        let cfg = Config::default();
        let hw = gpu_hw(12_000, 32_000, 8);
        let hf = hf_meta();
        let overlay = ModelOverride {
            ctx: Some(32_768),
            sglang: Some(crate::config::SglangTuning {
                grammar_backend: Some("xgrammar".into()),
                sleep_on_idle: Some(true),
                cuda_graph_bs: Some(vec![1, 2]),
                cuda_graph_backend_prefill: Some("disabled".into()),
                tp_size: Some(2),
                ..crate::config::SglangTuning::default()
            }),
            ..DEFAULT_OVERLAY.clone()
        };
        let mut inp = sglang_input(&hf, &hw, &cfg, 8_000 * MIB);
        inp.overlay = &overlay;
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        for flag in [
            "--grammar-backend",
            "--sleep-on-idle",
            "--cuda-graph-bs",
            "--cuda-graph-backend-prefill",
            "--tp-size",
        ] {
            assert!(
                !p.argv.contains(&flag.to_string()),
                "{flag} leaked: {:?}",
                p.argv
            );
            assert!(
                p.warnings.iter().any(|w| w.contains(flag)),
                "no warn-skip for {flag}: {:?}",
                p.warnings
            );
        }
    }

    #[test]
    fn unit__sglang__lora_gguf_refused_with_teaching() {
        let cfg = Config::default();
        let hw = gpu_hw(12_000, 32_000, 8);
        let hf = hf_meta();
        let loras = vec![("adapter.gguf".to_string(), 1.0)];
        let overlay = sglang_ctx_overlay();
        let mut inp = sglang_input(&hf, &hw, &cfg, 8_000 * MIB);
        inp.overlay = &overlay;
        inp.loras = &loras;
        let err = compile(&inp, &TuningOverrides::default()).unwrap_err();
        assert!(err.contains(".gguf"), "got: {err}");
        assert!(err.contains("PEFT"), "got: {err}");
    }

    #[test]
    fn unit__sglang__lora_hf_dir_emitted_with_scale_teaching() {
        let cfg = Config::default();
        let hw = gpu_hw(12_000, 32_000, 8);
        let hf = hf_meta();
        let flags = sglang_flags_extended();
        let loras = vec![("/models/adapters/qwen-lora-r16".to_string(), 4.0)];
        let overlay = sglang_ctx_overlay();
        let mut inp = sglang_input(&hf, &hw, &cfg, 8_000 * MIB);
        inp.overlay = &overlay;
        inp.loras = &loras;
        inp.supported_flags = &flags;
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(p.argv.contains(&"--enable-lora".to_string()));
        let idx = p
            .argv
            .iter()
            .position(|a| a == "--lora-paths")
            .expect("lora-paths");
        assert_eq!(
            p.argv[idx + 1],
            "qwen-lora-r16=/models/adapters/qwen-lora-r16"
        );
        // llama.cpp scale has no upstream equivalent — teach, never drop
        assert!(p
            .warnings
            .iter()
            .any(|w| w.contains("scale") && w.contains("ignored")));
    }

    #[test]
    fn unit__sglang__lora_extra_args_reserved() {
        // --enable-lora/--lora-paths are compile-owned now: an extra_args
        // attempt is a hard error pointing at the loras lane.
        let cfg = Config::default();
        let hw = gpu_hw(12_000, 32_000, 8);
        let hf = hf_meta();
        let overlay = ModelOverride {
            ctx: Some(32_768),
            extra_args: Some(vec!["--enable-lora".into()]),
            ..DEFAULT_OVERLAY.clone()
        };
        let mut inp = sglang_input(&hf, &hw, &cfg, 8_000 * MIB);
        inp.overlay = &overlay;
        let err = compile(&inp, &TuningOverrides::default()).unwrap_err();
        assert!(err.contains("reserved"), "got: {err}");
        assert!(err.contains("--enable-lora"), "got: {err}");
    }

    #[test]
    fn unit__sglang__parallel_pins_emit_above_one_and_warn_single_gpu() {
        let cfg = Config::default();
        let hw = gpu_hw(12_000, 32_000, 8); // single GPU
        let hf = hf_meta();
        let flags = sglang_flags_extended();
        let overlay = ModelOverride {
            ctx: Some(32_768),
            sglang: Some(crate::config::SglangTuning {
                tp_size: Some(2),
                dp_size: Some(1), // upstream default: nothing emitted
                ..crate::config::SglangTuning::default()
            }),
            ..DEFAULT_OVERLAY.clone()
        };
        let mut inp = sglang_input(&hf, &hw, &cfg, 8_000 * MIB);
        inp.overlay = &overlay;
        inp.supported_flags = &flags;
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--tp-size" && w[1] == "2"));
        assert!(!p.argv.iter().any(|a| a == "--dp-size"));
        assert!(p
            .warnings
            .iter()
            .any(|w| w.contains("dp_size = 1 is the upstream default")));
        assert!(p
            .warnings
            .iter()
            .any(|w| w.contains("only makes sense multi-GPU")));
    }

    #[test]
    fn unit__sglang__cuda_graph_bs_list_suppresses_tight_fit_scalar() {
        // Tier B engages tight-fit knobs; an explicit capture list wins
        // over the ladder's derived cuda-graph-max-bs scalar.
        let cfg = Config::default();
        let hw = gpu_hw(12_000, 32_000, 8);
        let hf = hf_meta();
        let flags = sglang_flags_extended();
        let overlay = ModelOverride {
            ctx: Some(32_768),
            sglang: Some(crate::config::SglangTuning {
                cuda_graph_bs: Some(vec![1, 2]),
                ..crate::config::SglangTuning::default()
            }),
            ..DEFAULT_OVERLAY.clone()
        };
        let mut inp = sglang_input(&hf, &hw, &cfg, 9_400 * MIB); // Tier B weights
        inp.overlay = &overlay;
        inp.supported_flags = &flags;
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(p.argv.iter().any(|a| a == "--cuda-graph-bs"));
        assert!(!p.argv.iter().any(|a| a == "--cuda-graph-max-bs"));
        // chunked-prefill tight-fit still applies (different knob)
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--chunked-prefill-size" && w[1] == "2048"));
    }
}
