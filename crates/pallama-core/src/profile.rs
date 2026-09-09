//! Profile compiler: deterministic llama-server argv from (model GGUF meta,
//! hardware, config+overlay, engine flag set). Rules 1–12 from the plan,
//! applied in order. Every emitted flag is validated against the installed
//! engine's manifest; a needed-but-missing flag is a hard error naming it.
//!
//! Pure: no I/O, no time, no randomness — table-testable. The supervisor
//! picks the endpoint (free port / socket) and passes it in.

use std::collections::BTreeSet;

use crate::config::{Config, ModelOverride};
use crate::gguf::GgufMeta;
use crate::hardware::Hardware;

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
    pub gguf: &'a GgufMeta,
    pub hardware: &'a Hardware,
    pub config: &'a Config,
    pub overlay: &'a ModelOverride,
    /// (path, scale) pairs from the loras table.
    pub loras: &'a [(String, f64)],
    /// Local path of the pulled draft model when spec=auto resolved one.
    pub draft_path: Option<&'a str>,
    /// Multimodal projector pulled alongside the model (vision/audio-in).
    /// Emitted as `-mm` when the engine supports it; the store's
    /// `mmproj_path` feeds this (rule 19).
    pub mmproj_path: Option<&'a str>,
    pub engine_tag: &'a str,
    /// Capability manifest flag set of the ACTIVE engine.
    pub supported_flags: &'a BTreeSet<String>,
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
    /// Auto-picked GPU id (e.g. "Vulkan1") from `--list-devices` free
    /// memory at spawn time. Only consulted when neither overlay nor
    /// config set `devices` — manual selection always wins. The same
    /// pick scopes `hardware` to that single GPU so every downstream
    /// VRAM estimate (ngl ladder, KV, cache-ram) sizes against the card
    /// the child will actually land on.
    pub device_hint: Option<&'a str>,
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
    /// any KV quantization the profile chose (`q8_0` halves, `q4_0`
    /// quarters).
    /// Feeds the co-residency planner (A15); None when GGUF geometry is
    /// missing.
    pub kv_est_bytes: Option<u64>,
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
        return Ok(compile_mistralrs(input, tuning));
    }
    let mut argv: Vec<String> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();
    let overlay = input.overlay;
    let config = input.config;

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
    let ctx = tuning
        .ctx
        .unwrap_or_else(|| resolve_ctx(input, overlay, &mut warnings));
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
    let vram_bytes = Hardware::bytes(input.hardware.total_vram_mib());
    argv.push("--gpu-layers".into());
    let (gpu_layers, gpu_label) = resolve_gpu_offload(input, ctx, vram_bytes, &mut warnings);
    argv.push(gpu_layers.into());

    // --- 5. prefix-cache chunk reuse
    if config.cache_reuse > 0 {
        argv.push("--cache-reuse".into());
        argv.push(config.cache_reuse.to_string());
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
    if input.gguf.expert_count.unwrap_or(0) > 0
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

    // --- 9. slots: 1 = full-speed single client (default, ollama-parity
    // UX); 0 = upstream auto multi-slot for concurrent clients. Overlay
    // slots shadow the global for this model; the supervisor's LC4
    // adaptive adoption fills in ONLY where the user set nothing.
    let slots = overlay.slots.unwrap_or(input.config.slots);
    argv.push("-np".into());
    if slots == 0 {
        argv.push("-1".into());
    } else {
        argv.push(slots.to_string());
    }

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
    } else if let Some(pooling) = input.gguf.pooling_type {
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
        push_spec_args(input, &mut argv, &mut warnings)?;
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
    }

    // --- 11b. spec-draft placement (only when a draft model is actually
    // resolved: spec = "auto" + pulled pair). Pin the draft to P-cores /
    // a spare GPU / fewer threads so it stops stealing from the target.
    if spec_mode == "auto" && input.draft_path.is_some() {
        if !config.spec_draft_device.is_empty() {
            argv.push("--spec-draft-device".into());
            argv.push(config.spec_draft_device.clone());
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
            argv.push("--spec-draft-cpu-strict".into());
            argv.push("1".into());
        }
        if !config.spec_draft_cpu_range.is_empty() {
            argv.push("--spec-draft-cpu-range".into());
            argv.push(config.spec_draft_cpu_range.clone());
        }
        if !config.spec_draft_ngl.is_empty() {
            argv.push("--spec-draft-ngl".into());
            argv.push(config.spec_draft_ngl.clone());
        }
        if config.spec_draft_threads > 0 {
            argv.push("--spec-draft-threads".into());
            argv.push(config.spec_draft_threads.to_string());
        }
        if let Some(p) = config.spec_draft_p_min {
            argv.push("--spec-draft-p-min".into());
            argv.push(format_trimmed(p));
        }
        if let Some(p) = config.spec_draft_p_split {
            argv.push("--spec-draft-p-split".into());
            argv.push(format_trimmed(p));
        }
        if config.spec_draft_poll.is_some() || config.spec_draft_poll_batch.is_some() {
            // Poll level governs both phases unless batch is explicit.
            if let Some(p) = config.spec_draft_poll {
                argv.push("--spec-draft-poll".into());
                argv.push(p.to_string());
            }
            if let Some(pb) = config.spec_draft_poll_batch {
                argv.push("--spec-draft-poll-batch".into());
                argv.push(if pb { "1".into() } else { "0".into() });
            }
        }
        if config.spec_draft_prio != 0 {
            argv.push("--spec-draft-prio".into());
            argv.push(config.spec_draft_prio.to_string());
        }
        if config.spec_draft_prio_batch != 0 {
            argv.push("--spec-draft-prio-batch".into());
            argv.push(config.spec_draft_prio_batch.to_string());
        }
        if config.spec_draft_cpu_strict_batch {
            argv.push("--spec-draft-cpu-strict-batch".into());
            argv.push("1".into());
        }
        if config.spec_draft_threads_batch > 0 {
            argv.push("--spec-draft-threads-batch".into());
            argv.push(config.spec_draft_threads_batch.to_string());
        }
        if !config.spec_draft_type_k.is_empty() {
            argv.push("--spec-draft-type-k".into());
            argv.push(config.spec_draft_type_k.clone());
        }
        if !config.spec_draft_type_v.is_empty() {
            argv.push("--spec-draft-type-v".into());
            argv.push(config.spec_draft_type_v.clone());
        }
        for ot in &config.spec_draft_override_tensor {
            argv.push("--spec-draft-override-tensor".into());
            argv.push(ot.clone());
        }
        if config.spec_draft_n_cpu_moe > 0 {
            argv.push("--spec-draft-n-cpu-moe".into());
            argv.push(config.spec_draft_n_cpu_moe.to_string());
        }
        if config.spec_draft_cpu_moe {
            argv.push("--spec-draft-cpu-moe".into());
        }
        if !config.spec_draft_backend_sampling {
            argv.push("--no-spec-draft-backend-sampling".into());
        }
        if config.adaptive_decay > 0 {
            argv.push("--adaptive-decay".into());
            argv.push(config.adaptive_decay.to_string());
        }
        if config.adaptive_target > 0.0 {
            argv.push("--adaptive-target".into());
            argv.push(format_trimmed(config.adaptive_target));
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

    // --- 12. prompt-cache budget + vision projector
    if config.cache_ram_mb > 0 {
        // The prompt cache shares physical RAM with everything else on the
        // machine; the upstream-style 8 GiB default starves small-RAM boxes
        // into swap death. Cap at 30% of physical RAM (measured live: 13 GiB
        // box + 8192 budget -> 8.3 GiB child RSS plateau, system-wide thrash).
        // Escape hatches: cache_ram_mb = 0 (unlimited), or a per-model
        // `extra_args = ["--cache-ram", "<MiB>"]` override (appended later,
        // last flag wins upstream).
        let budget = if input.hardware.total_ram_mib > 0 {
            // Adaptive (A16): prefix-heavy traffic (hit rate > 0.5) earns a
            // 40% cap, cache-cold traffic (< 0.1) releases to 20%; the
            // static clamp is 30%. The hint comes from the live daemon's
            // EWMA — CLI/bench compiles pass None and keep 30%.
            let pct = match input.cache_hit_rate {
                Some(h) if h > 0.5 => 40,
                Some(h) if h < 0.1 => 20,
                _ => 30,
            };
            let cap = input.hardware.total_ram_mib * pct / 100;
            match u64::try_from(config.cache_ram_mb) {
                Ok(requested) if requested > cap => {
                    warnings.push(format!(
                        "cache_ram_mb {} clamped to {} ({}% of {} MiB RAM, hit-rate {}); override via model_overrides extra_args --cache-ram or set 0 = unlimited",
                        config.cache_ram_mb, cap, pct, input.hardware.total_ram_mib,
                        input
                            .cache_hit_rate
                            .map_or_else(|| "n/a".to_string(), |h| format!("{h:.2}"))
                    ));
                    i64::try_from(cap).unwrap_or(i64::MAX)
                }
                _ => config.cache_ram_mb,
            }
        } else {
            config.cache_ram_mb
        };
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
    } else if input.supported_flags.contains("--kv-unified") {
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
    }

    // --- 13. latency/affinity passthrough (config-validated, manifest-gated)
    // Semantics verified against upstream arg.cpp b10816: --cpu-range pins
    // child threads to a "lo-hi" CPU set (P/E hybrid boxes: pin to P-cores);
    // --poll 1..100 busy-polls waiting for work (CPU for TTFT);
    // --reasoning-format selects thought-tag extraction in responses.
    if !config.cpu_range.is_empty() {
        argv.push("--cpu-range".into());
        argv.push(config.cpu_range.clone());
    }
    if config.poll > 0 {
        argv.push("--poll".into());
        argv.push(config.poll.to_string());
    }
    if !config.reasoning_format.is_empty() {
        argv.push("--reasoning-format".into());
        argv.push(config.reasoning_format.clone());
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
            argv.push("--reasoning-budget".into());
            argv.push(budget.to_string());
        }
        if !config.reasoning_budget_message.is_empty() {
            argv.push("--reasoning-budget-message".into());
            argv.push(config.reasoning_budget_message.clone());
        }
        let effort = overlay
            .reasoning_effort
            .as_deref()
            .filter(|e| !e.is_empty())
            .unwrap_or_else(|| config.effective_reasoning_effort(input.model_name));
        if !effort.is_empty() {
            argv.push("--reasoning-effort".into());
            argv.push(effort.to_string());
        }
        if let Some(preserve) = config.reasoning_preserve {
            argv.push(if preserve {
                "--reasoning-preserve".into()
            } else {
                "--no-reasoning-preserve".into()
            });
        }
    }

    // --- 13c. scheduling extras (server-level).
    if config.cpu_strict {
        argv.push("--cpu-strict".into());
        argv.push("1".into());
    }
    if config.prio != 0 {
        argv.push("--prio".into());
        argv.push(config.prio.to_string());
    }
    if config.prio_batch != 0 {
        argv.push("--prio-batch".into());
        argv.push(config.prio_batch.to_string());
    }
    if let Some(pb) = config.poll_batch {
        argv.push("--poll-batch".into());
        argv.push(if pb { "1".into() } else { "0".into() });
    }
    if config.threads_http > 0 {
        argv.push("--threads-http".into());
        argv.push(config.threads_http.to_string());
    }

    // --- 12c. vision / multimodal tuning + embeddings normalization.
    if config.image_max_tokens > 0 {
        argv.push("--image-max-tokens".into());
        argv.push(config.image_max_tokens.to_string());
    }
    if config.image_min_tokens > 0 {
        argv.push("--image-min-tokens".into());
        argv.push(config.image_min_tokens.to_string());
    }
    if config.mtmd_batch_max_tokens > 0 {
        argv.push("--mtmd-batch-max-tokens".into());
        argv.push(config.mtmd_batch_max_tokens.to_string());
    }
    if !config.mmproj_offload {
        argv.push("--no-mmproj-offload".into());
    }
    if !config.mmproj_auto {
        argv.push("--no-mmproj-auto".into());
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
    // vision off) — never silently, never fatally.
    if let Some(mmproj) = overlay.extra_args.as_deref().and_then(find_mmproj_arg) {
        argv.push("-mm".into());
        argv.push(mmproj.to_string());
    } else if let Some(mmproj) = input.mmproj_path {
        if input.supported_flags.contains("--mmproj") || input.supported_flags.contains("-mm") {
            argv.push("-mm".into());
            argv.push(mmproj.to_string());
        } else {
            eprintln!(
                "pallama profile: model {} has a multimodal projector but engine {} lacks -mm/--mmproj; vision disabled (engine update)",
                input.model_name, input.engine_tag
            );
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

    // KV estimate for the co-residency planner: f16 bytes scaled by the
    // quantization grade actually emitted above. Hybrid-linear models with
    // a provable recurrent/full split already get the true (fractional) KV
    // from `kv_f16_bytes`; warn exactly when the split is NOT provable, so
    // an inflated estimate is never a surprise (R8).
    if input.gguf.attention_class() == crate::gguf::AttentionClass::HybridLinear
        && !input.gguf.recurrent_split_provable()
    {
        warnings.push(format!(
            "arch {} is hybrid-linear but the GGUF lacks recurrent_layers/full_attention_interval \
             metadata — KV estimate counts all layers (upper bound); a newer GGUF conversion may \
             emit the layer map",
            input.gguf.architecture
        ));
    }
    let kv_est_bytes = kv_f16_bytes(input, ctx).map(|f16| match kv_type.as_deref() {
        Some("q8_0") => f16 / 2,
        Some("q4_0") => f16 / 4,
        Some("q4_1") => f16 * 9 / 20,
        Some("q5_0") => f16 * 11 / 32,
        Some("q5_1") => f16 * 3 / 8,
        _ => f16,
    });

    Ok(Profile {
        argv,
        warnings,
        ctx,
        gpu: gpu_label,
        kv_est_bytes,
    })
}

/// Late-chunking ubatch default: upstream embedding-preset scale
/// (embeddinggemma sets `n_batch` = `n_ubatch` = 2048). One embedding doc
/// must fit a single micro-batch; a full-ctx ubatch OOMs tight GPUs.
pub const LATE_CHUNK_UBATCH_DEFAULT: u32 = 2048;

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

/// Effective `ctx`: overlay > config default, clamped to the model's
/// `context_length` when known. Missing metadata warns, never guesses.
/// Minimal mistral.rs profile: ctx via the same overlay/config/train-cap
/// clamp as llama-server (so `pallama ps` and KV planning share one
/// truth), slots as `-np N` for the argv translator to mine. KV estimate
/// is None: mistral.rs sizes its paged KV from a VRAM fraction, not from
/// ctx math, so an f16 estimate here would be a lie.
fn compile_mistralrs(input: &ProfileInput<'_>, tuning: &TuningOverrides) -> Profile {
    let ctx = tuning
        .ctx
        .unwrap_or_else(|| resolve_ctx(input, input.overlay, &mut Vec::new()));
    let slots = input.overlay.slots.unwrap_or(input.config.slots);
    let mut argv: Vec<String> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();
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
    if let Some(pa) = input.config.mistralrs_paged_attn {
        if input.supported_flags.contains("--paged-attn") {
            argv.push("--paged-attn".into());
            argv.push(if pa { "on".into() } else { "off".into() });
        } else {
            warnings.push(
                "config mistralrs_paged_attn set but this engine lacks \
                 --paged-attn; flag skipped (pallama engine install updates it)"
                    .into(),
            );
        }
    }
    Profile {
        argv,
        warnings,
        ctx,
        gpu: "auto",
        kv_est_bytes: None,
    }
}

fn resolve_ctx(
    input: &ProfileInput<'_>,
    overlay: &ModelOverride,
    warnings: &mut Vec<String>,
) -> u32 {
    let requested = overlay.ctx.unwrap_or(input.config.default_ctx);
    match input.gguf.context_length {
        Some(train) if requested > u32::try_from(train).unwrap_or(u32::MAX) => {
            warnings.push(format!(
                "ctx {requested} clamped to model context_length {train} (GGUF metadata)"
            ));
            u32::try_from(train).unwrap_or(u32::MAX)
        }
        Some(_) => requested,
        None => {
            warnings.push(
                "ctx not clamped: GGUF lacks context_length (rule skipped, not estimated)".into(),
            );
            requested
        }
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
    input.gguf.kv_f16_bytes(u64::from(ctx))
}

/// Public KV estimate for callers outside the compiler (the supervisor's
/// co-residency planner). Same math as the internal ladder input.
#[must_use]
pub fn estimate_kv_f16(input: &ProfileInput<'_>, ctx: Option<u32>) -> Option<u64> {
    let ctx = ctx.unwrap_or_else(|| input.overlay.ctx.unwrap_or(input.config.default_ctx));
    kv_f16_bytes(input, ctx)
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
    if resident.saturating_add(kv) <= vram_bytes / 100 * 85 {
        return ("999", "full");
    }
    ("auto", "auto")
}

/// True for every self-drafting n-gram spec mode ("ngram" is pallama
/// shorthand for upstream "ngram-simple").
fn is_ngram_spec(mode: &str) -> bool {
    matches!(
        mode,
        "ngram" | "ngram-map-k" | "ngram-map-k4v" | "ngram-mod" | "ngram-cache"
    )
}

/// Rule 11: spec=auto draft pairing. Hard error when the catalog pair
/// exists but the draft is not pulled.
fn push_spec_args(
    input: &ProfileInput<'_>,
    argv: &mut Vec<String>,
    warnings: &mut Vec<String>,
) -> Result<(), String> {
    match crate::catalog::spec_pair_for(input.model_name) {
        Some(pair) => {
            if let Some(draft) = input.draft_path {
                argv.push("--spec-type".into());
                argv.push(pair.spec_type.clone());
                argv.push("--spec-draft-model".into());
                argv.push(draft.to_string());
                argv.push("--spec-draft-n-max".into());
                argv.push("3".into());
            } else {
                // Always hard-error on an unpulled draft. The engine's
                // `--spec-draft-hf` auto-download flag exists in b10840+ but
                // resolves the repo to an empty path and the child exits
                // fatally (verified live 2026-09-07) — revisit once upstream
                // fixes draft-side HF resolution.
                return Err(format!(
                    "spec=auto for {} but the draft model is not pulled; run: pallama pull {}",
                    input.model_name, pair.draft_repo
                ));
            }
        }
        None => warnings.push(format!(
            "spec=auto but no draft pair for {} in the catalog; running dense",
            input.model_name
        )),
    }
    Ok(())
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
/// may contain `/` and `:`).
#[must_use]
pub fn path_safe(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
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

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use std::sync::LazyLock;
    const MIB: u64 = 1024 * 1024;
    use super::*;
    use crate::config::SamplerDefaults;
    use crate::gguf::GgufMeta;
    use crate::hardware::GpuInfo;

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
            "--load-mode",
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
            "--lookup-cache-dynamic",
            "--slot-save-path",
            "--rope-scaling",
            "--rope-scale",
            "--n-cpu-moe",
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
            "--embeddings",
            "--pooling",
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
        }
    }

    fn input<'a>(
        gguf: &'a GgufMeta,
        hw: &'a Hardware,
        cfg: &'a Config,
        flags: &'a BTreeSet<String>,
    ) -> ProfileInput<'a> {
        ProfileInput {
            engine_kind: crate::engine_kind::EngineKind::default(),
            model_name: "qwen3-8b",
            instance_key: "qwen3-8b",
            model_path: "/models/qwen3-8b.gguf",
            model_bytes: 5_000 * MIB,
            gguf,
            hardware: hw,
            config: cfg,
            overlay: &DEFAULT_OVERLAY,
            loras: &[],
            draft_path: None,
            mmproj_path: None,
            engine_tag: "b-test",
            supported_flags: flags,
            endpoint: Endpoint::Tcp {
                host: "127.0.0.1".into(),
                port: 12345,
            },
            data_dir: "/tmp/pallama-test-data",
            cache_hit_rate: None,
            device_hint: None,
            sibling_devices: Vec::new(),
            auto_tensor_split: None,
        }
    }

    static ALL_FLAGS: LazyLock<BTreeSet<String>> = LazyLock::new(full_flags);
    static DEFAULT_OVERLAY: ModelOverride = ModelOverride {
        ctx: None,
        slots: None,
        spec: None,
        loras: None,
        extra_args: None,
        cache_type: None,
        kv_unified: None,
        ctx_extend: None,
        cpu_moe_n: None,
        override_tensor: None,
        devices: None,
        warmup: None,
        reasoning_budget: None,
        reasoning_effort: None,
        replicas: None,
        pin: None,
        chat_template: None,
        chat_template_file: None,
        sampler_defaults: None,
        spm_infill: None,
        late_chunking: None,
        rpc_servers: None,
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
        let cfg = Config::default();
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
        // Rule 2: jinja + metrics + flash-attn auto + ctx (default 16384 < 40960)
        assert!(p.argv.contains(&"--jinja".to_string()));
        assert!(p.argv.contains(&"--metrics".to_string()));
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--flash-attn" && w[1] == "auto"));
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--ctx-size" && w[1] == "16384"));
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
        // Rule 5: cache-reuse default 256
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--cache-reuse" && w[1] == "256"));
        // Rule 6: KV = 2*28*8*64*16384*2 = 469MB; +5GB < 0.9*12GB -> NO kv quant
        assert!(!p.argv.contains(&"--cache-type-k".to_string()));
        // Rule 8: sleep (GPU present)
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--sleep-idle-seconds" && w[1] == "300"));
        // Rule 9: default single slot (full-speed single client).
        assert!(p.argv.windows(2).any(|w| w[0] == "-np" && w[1] == "1"));
        // Rule 12: cache-ram default 8192
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--cache-ram" && w[1] == "8192"));
        // Rule 15: sessions dir default-on when the engine supports it
        assert!(p.argv.windows(2).any(|w| w[0] == "--slot-save-path"));
        assert_eq!(p.ctx, 16384);
        assert!(p.warnings.is_empty(), "{:?}", p.warnings);
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
    #[allow(clippy::field_reassign_with_default)]
    fn unit__kv_layout__aux_flags_and_gating() {
        let hw = gpu_hw(24_000, 64_000, 8);
        let mut cfg = Config::default();
        cfg.kv_unified_per_slot = 4096;
        cfg.swa_full = true;
        cfg.ctx_checkpoints = 8;
        cfg.no_kv_offload = true;
        cfg.load_mode = "mlock".into();
        let p = compile(
            &input(&GgufMeta::default(), &hw, &cfg, &ALL_FLAGS),
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
            .any(|w| w[0] == "--cache-ram" && w[1] == "4007"));
        assert!(p
            .warnings
            .iter()
            .any(|w| w.contains("cache_ram_mb 8192 clamped to 4007")));
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
        // Rule 14: persistent lookup cache rides along with ngram mode
        assert!(p.argv.windows(2).any(|w| w[0] == "--lookup-cache-dynamic"
            && w[1] == "/tmp/pallama-test-data/speccache/qwen3-8b.lcache"));
        assert!(p.warnings.is_empty(), "{:?}", p.warnings);
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
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--ctx-size" && w[1] == "40960"));
        assert_eq!(p.ctx, 40_960);
        assert!(p.warnings.iter().any(|w| w.contains("clamped")));
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
        let cfg = Config::default();
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
        let cfg = Config::default();
        let p = compile(
            &input(&g, &hw, &cfg, &ALL_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        // second model with an overlay knob
        let o = ModelOverride {
            ctx: Some(8192),
            ..ModelOverride::default()
        };
        let cfg2 = Config::default();
        let p2 = compile(
            &input(&g, &hw, &cfg2, &ALL_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        let _ = o;
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
        assert!(ini.contains("cache-reuse = 256"));
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
        let cfg = Config {
            spec: "auto".into(),
            ..Config::default()
        };
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        let mut inp = input(&g, &hw, &cfg, &ALL_FLAGS);
        inp.model_name = "qwen3-8b"; // has catalog draft pair
        let err = compile(&inp, &TuningOverrides::default()).unwrap_err();
        assert!(
            err.contains("pallama pull ggml-org/Qwen3-0.6B-GGUF"),
            "{err}"
        );

        // Draft pulled -> flags emitted.
        inp.draft_path = Some("/models/qwen3-0.6b-q4_k_m.gguf");
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--spec-type" && w[1] == "draft-simple"));
        assert!(p
            .argv
            .windows(2)
            .any(|w| w[0] == "--spec-draft-model" && w[1] == "/models/qwen3-0.6b-q4_k_m.gguf"));
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
        // No hint, no overlay: global default (1) still rules.
        let p = compile(
            &input(&g, &hw, &Config::default(), &ALL_FLAGS),
            &TuningOverrides::default(),
        )
        .unwrap();
        assert!(p.argv.windows(2).any(|w| w == ["-np", "1"]));
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
    fn unit__spec_auto__draft_missing__hard_error_even_on_new_engines() {
        // Live evidence (b10840): `--spec-draft-hf` resolves the repo to an
        // empty draft path and the child exits fatally, so an unpulled draft
        // must hard-error with the pull hint regardless of manifest flags.
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
        let err = compile(&inp, &TuningOverrides::default()).unwrap_err();
        assert!(
            err.contains("pallama pull ggml-org/Qwen3-0.6B-GGUF:Q4_0"),
            "{err}"
        );
    }

    #[test]
    fn unit__spec_auto__no_pair_for_model__dense_with_warning() {
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
        assert!(p.warnings.iter().any(|w| w.contains("no draft pair")));
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
        let cfg = Config::default();
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
        let cfg = Config::default();
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
        // Rule 19: a pulled mmproj wires vision automatically (the gap
        // that left downloaded projectors dead on disk).
        let cfg = Config::default();
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        let mut inp = input(&g, &hw, &cfg, &ALL_FLAGS);
        inp.mmproj_path = Some("/models/mmproj-F16.gguf");
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(
            p.argv
                .windows(2)
                .any(|w| w[0] == "-mm" && w[1] == "/models/mmproj-F16.gguf"),
            "pulled projector must reach the engine: {:?}",
            p.argv
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
        let mut f = ALL_FLAGS.clone();
        for flag in [
            "--spec-draft-cpu-range",
            "--spec-draft-cpu-strict",
            "--spec-draft-device",
            "--spec-draft-ngl",
            "--spec-draft-threads",
            "--spec-draft-p-min",
            "--spec-draft-p-split",
            "--spec-draft-poll",
            "--spec-draft-prio",
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
            "--reasoning-budget",
            "--reasoning-budget-message",
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
            "--yarn-orig-ctx",
            "--yarn-ext-factor",
            "--yarn-attn-factor",
            "--yarn-beta-fast",
            "--yarn-beta-slow",
            "--cpu-strict",
            "--prio",
            "--prio-batch",
            "--poll-batch",
            "--threads-http",
            "--no-warmup",
            "--no-repack",
            "--no-cache-idle-slots",
            "--no-host",
            "--op-offload",
            "--no-op-offload",
            "--keep",
            "--override-kv",
            "--control-vector",
            "--control-vector-scaled",
            "--control-vector-layer-range",
        ] {
            f.insert(flag.to_string());
        }
        f
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
        ] {
            assert!(!p.argv.iter().any(|a| a == f), "{f} must stay unset");
        }
    }

    #[test]
    fn unit__wire__reasoning_knobs_and_overlay() {
        let cfg = Config {
            reasoning_budget: 256,
            reasoning_effort: "low".into(),
            reasoning_preserve: Some(false),
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
        // Per-model overlay wins over the global budget.
        let overlay = ModelOverride {
            reasoning_budget: Some(64),
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
        let cfg = Config::default();
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
        let cfg = Config {
            cache_ram_mb: 8192,
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
        // Cold: 20% cap = 3276.
        let mut inp = input(&g, &hw, &cfg, &WIRE_FLAGS);
        inp.cache_hit_rate = Some(0.01);
        let p2 = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(p2
            .argv
            .windows(2)
            .any(|w| w[0] == "--cache-ram" && w[1] == "3276"));
        // None: static 30% = 4915.
        let inp = input(&g, &hw, &cfg, &WIRE_FLAGS);
        let p3 = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(p3
            .argv
            .windows(2)
            .any(|w| w[0] == "--cache-ram" && w[1] == "4915"));
    }

    #[test]
    fn unit__wire__kv_est_bytes_quantized() {
        let cfg = Config::default();
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
        let cfg = Config::default();
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
        let cfg = Config::default();
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
        let cfg = Config::default();
        let hw = gpu_hw(0, 32_000, 8);
        let g = meta();
        let empty: BTreeSet<String> = BTreeSet::new();
        let mut inp = input(&g, &hw, &cfg, &empty);
        inp.engine_kind = crate::engine_kind::EngineKind::MistralRs;
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert_eq!(p.argv, vec!["-np".to_string(), cfg.slots.to_string()]);
        assert!(p.warnings.is_empty());
        assert_eq!(p.kv_est_bytes, None);
        // slots = 0 (upstream auto) emits nothing — the translator's
        // own default applies.
        let cfg0 = Config {
            slots: 0,
            ..Default::default()
        };
        let mut inp0 = input(&g, &hw, &cfg0, &empty);
        inp0.engine_kind = crate::engine_kind::EngineKind::MistralRs;
        let p0 = compile(&inp0, &TuningOverrides::default()).unwrap();
        assert!(p0.argv.is_empty(), "{:?}", p0.argv);
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
        inp.draft_path = Some("/models/draft.gguf");
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
        inp.draft_path = Some("/models/qwen3-0.5b.gguf");
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
        pinned.draft_path = Some("/models/qwen3-0.5b.gguf");
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
        inp.draft_path = Some("/models/qwen3-0.5b.gguf");
        inp.mmproj_path = Some("/models/mmproj.gguf");
        // sibling_devices empty (single-card box): stays silent.
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(!p.warnings.iter().any(|w| w.contains("spare discrete card")));
    }
}
