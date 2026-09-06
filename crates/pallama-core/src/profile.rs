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
    pub endpoint: Endpoint,
    /// Pallama data dir (base for speccache/ + sessions/ paths).
    pub data_dir: &'a str,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Profile {
    pub argv: Vec<String>,
    /// Non-fatal notes (skipped rules, unverifiable clamps). Surfaced via
    /// tracing and `pallama show`.
    pub warnings: Vec<String>,
    /// The ctx actually compiled in (post-clamp).
    pub ctx: u32,
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
    argv.extend(["--jinja".into(), "--metrics".into(), "--flash-attn".into(), fa.into()]);
    argv.push("--ctx-size".into());
    argv.push(ctx.to_string());

    // --- 3. threads
    let threads = tuning
        .threads
        .unwrap_or_else(|| input.hardware.physical_cores.max(1));
    argv.push("--threads".into());
    argv.push(threads.to_string());
    if let Some(b) = tuning.batch {
        argv.push("-b".into());
        argv.push(b.to_string());
    }
    if let Some(ub) = tuning.ubatch {
        argv.push("--ubatch-size".into());
        argv.push(ub.to_string());
    }

    // --- 4. gpu layers: auto everywhere; upstream --fit on (default) shrinks
    argv.push("--gpu-layers".into());
    argv.push("auto".into());

    // --- 5. prefix-cache chunk reuse
    if config.cache_reuse > 0 {
        argv.push("--cache-reuse".into());
        argv.push(config.cache_reuse.to_string());
    }

    // --- 6. KV cache quantization. Bench-adopted tuning wins, then an
    // explicit config/overlay type ("f16"-class = force off), then the
    // capacity ladder: none -> q8_0 (KV/2) -> q4_0 (KV/4).
    let vram_bytes = Hardware::bytes(input.hardware.total_vram_mib());
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
    if let Some(t) = kv_type {
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
    // UX); config 0 = upstream auto multi-slot for concurrent clients.
    argv.push("-np".into());
    if input.config.slots == 0 {
        argv.push("-1".into());
    } else {
        argv.push(input.config.slots.to_string());
    }

    // --- 10. rpc + loras
    if !config.rpc_servers.trim().is_empty() {
        argv.push("--rpc".into());
        argv.push(config.rpc_servers.trim().to_string());
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

    // --- 11. speculative decoding
    let spec_mode = overlay.spec.as_deref().unwrap_or(config.spec.as_str());
    if spec_mode == "auto" {
        push_spec_args(input, &mut argv, &mut warnings)?;
    } else if spec_mode == "ngram" {
        // Self-drafting n-gram speculation: no draft model to pull; drafts
        // from the context's own n-grams. Bench before adopting at scale —
        // verification overhead can regress non-repetitive workloads.
        argv.push("--spec-type".into());
        argv.push("ngram-simple".into());
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
            let cap = input.hardware.total_ram_mib * 30 / 100;
            match u64::try_from(config.cache_ram_mb) {
                Ok(requested) if requested > cap => {
                    warnings.push(format!(
                        "cache_ram_mb {} clamped to {} (30% of {} MiB RAM); override via model_overrides extra_args --cache-ram or set 0 = unlimited",
                        config.cache_ram_mb, cap, input.hardware.total_ram_mib
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
    if spec_mode == "ngram" && config.spec_cache {
        if input.supported_flags.contains("--lookup-cache-dynamic") {
            argv.push("--lookup-cache-dynamic".into());
            argv.push(format!(
                "{}/speccache/{}.lcache",
                input.data_dir,
                path_safe(input.model_name)
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
                path_safe(input.model_name)
            ));
        } else {
            warnings.push(
                "sessions skipped: engine lacks --slot-save-path (engine update recommended)".into(),
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

    // --- overlay extra args (validated like everything else)
    if let Some(extra) = &overlay.extra_args {
        argv.extend(extra.iter().cloned());
    }

    // --- manifest gate: every emitted long flag must exist on this engine
    let mut missing: Vec<String> = Vec::new();
    for a in &argv {
        if a.starts_with("--") {
            let name = a.split('=').next().unwrap_or(a);
            if !input.supported_flags.contains(name)
                && !missing.iter().any(|m| m == name)
            {
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

    Ok(Profile { argv, warnings, ctx })
}

/// Effective `ctx`: overlay > config default, clamped to the model's
/// `context_length` when known. Missing metadata warns, never guesses.
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
    match (
        input.gguf.block_count,
        input.gguf.head_count_kv.or(input.gguf.head_count),
        input.gguf.derived_head_dim(),
    ) {
        (Some(blocks), Some(kv_heads), Some(head_dim)) => {
            let kv = 2u64
                .saturating_mul(blocks)
                .saturating_mul(kv_heads)
                .saturating_mul(head_dim)
                .saturating_mul(u64::from(ctx))
                .saturating_mul(2);
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
        (None, _, _) => {
            warnings.push("KV-quant rule skipped: GGUF lacks block_count".into());
            None
        }
        (_, None, _) => {
            warnings.push("KV-quant rule skipped: GGUF lacks head_count_kv".into());
            None
        }
        (_, _, None) => {
            warnings.push("KV-quant rule skipped: head_dim not derivable".into());
            None
        }
    }
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
            let draft = input.draft_path.ok_or_else(|| {
                format!(
                    "spec=auto for {} but the draft model is not pulled; run: pallama pull {}",
                    input.model_name, pair.draft_repo
                )
            })?;
            argv.push("--spec-type".into());
            argv.push(pair.spec_type.clone());
            argv.push("--spec-draft-model".into());
            argv.push(draft.to_string());
            argv.push("--spec-draft-n-max".into());
            argv.push("3".into());
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
                let value = argv.get(i + 1).filter(|v| !v.starts_with("--") || short_value);
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
    use crate::gguf::GgufMeta;
    use crate::hardware::GpuInfo;

    fn full_flags() -> BTreeSet<String> {
        [
            "-m", "--host", "--port", "--alias", "--jinja", "--metrics", "--flash-attn",
            "--ctx-size", "--threads", "--gpu-layers", "--cache-reuse", "--cache-type-k",
            "--cache-type-v", "--cpu-moe", "--sleep-idle-seconds", "-np", "--rpc", "--lora",
            "--lora-scaled", "--spec-type", "--spec-draft-model", "--spec-draft-n-max",
            "--cache-ram", "-mm", "--mmproj", "--ubatch-size", "--cpu-range", "--poll", "--reasoning-format",
            "--lookup-cache-dynamic", "--slot-save-path", "--rope-scaling", "--rope-scale",
            "--n-cpu-moe", "--override-tensor", "--agent", "--slot-prompt-similarity",
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
            model_name: "qwen3-8b",
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
            endpoint: Endpoint::Tcp { host: "127.0.0.1".into(), port: 12345 },
            data_dir: "/tmp/pallama-test-data",
        }
    }

    static ALL_FLAGS: LazyLock<BTreeSet<String>> = LazyLock::new(full_flags);
    static DEFAULT_OVERLAY: ModelOverride = ModelOverride {
        ctx: None,
        spec: None,
        loras: None,
        extra_args: None,
        cache_type: None,
        ctx_extend: None,
        cpu_moe_n: None,
        override_tensor: None,
    };

    #[test]
    fn unit__profile_base_rules__emitted_in_order() {
        let cfg = Config::default();
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        let p = compile(&input(&g, &hw, &cfg, &ALL_FLAGS), &TuningOverrides::default()).unwrap();
        // Rule 1: model, endpoint, alias
        assert_eq!(p.argv[0], "-m");
        assert_eq!(p.argv[1], "/models/qwen3-8b.gguf");
        assert!(p.argv.windows(2).any(|w| w[0] == "--host" && w[1] == "127.0.0.1"));
        assert!(p.argv.windows(2).any(|w| w[0] == "--port" && w[1] == "12345"));
        assert!(p.argv.windows(2).any(|w| w[0] == "--alias" && w[1] == "qwen3-8b"));
        // Rule 2: jinja + metrics + flash-attn auto + ctx (default 16384 < 40960)
        assert!(p.argv.contains(&"--jinja".to_string()));
        assert!(p.argv.contains(&"--metrics".to_string()));
        assert!(p.argv.windows(2).any(|w| w[0] == "--flash-attn" && w[1] == "auto"));
        assert!(p.argv.windows(2).any(|w| w[0] == "--ctx-size" && w[1] == "16384"));
        // Rule 3: threads = physical cores
        assert!(p.argv.windows(2).any(|w| w[0] == "--threads" && w[1] == "8"));
        // Rule 4: gpu-layers auto
        assert!(p.argv.windows(2).any(|w| w[0] == "--gpu-layers" && w[1] == "auto"));
        // Rule 5: cache-reuse default 256
        assert!(p.argv.windows(2).any(|w| w[0] == "--cache-reuse" && w[1] == "256"));
        // Rule 6: KV = 2*28*8*64*16384*2 = 469MB; +5GB < 0.9*12GB -> NO kv quant
        assert!(!p.argv.contains(&"--cache-type-k".to_string()));
        // Rule 8: sleep (GPU present)
        assert!(p.argv.windows(2).any(|w| w[0] == "--sleep-idle-seconds" && w[1] == "300"));
        // Rule 9: default single slot (full-speed single client).
        assert!(p.argv.windows(2).any(|w| w[0] == "-np" && w[1] == "1"));
        // Rule 12: cache-ram default 8192
        assert!(p.argv.windows(2).any(|w| w[0] == "--cache-ram" && w[1] == "8192"));
        // Rule 15: sessions dir default-on when the engine supports it
        assert!(p.argv.windows(2).any(|w| w[0] == "--slot-save-path"));
        assert_eq!(p.ctx, 16384);
        assert!(p.warnings.is_empty(), "{:?}", p.warnings);
    }

    #[test]
    fn unit__cache_ram_clamped_to_30pct_of_ram_on_small_boxes() {
        // Live case: 13 GiB laptop, default 8192 -> cap 4007 (30% of 13359).
        // Unclamped, the child RSS plateaus at 8.3 GiB and the box swap-thrashes.
        let cfg = Config::default();
        let hw = gpu_hw(12_000, 13_359, 8);
        let g = meta();
        let p = compile(&input(&g, &hw, &cfg, &ALL_FLAGS), &TuningOverrides::default()).unwrap();
        assert!(p.argv.windows(2).any(|w| w[0] == "--cache-ram" && w[1] == "4007"));
        assert!(p.warnings.iter().any(|w| w.contains("cache_ram_mb 8192 clamped to 4007")));
    }

    #[test]
    fn unit__cache_ram_zero_disables_flag_entirely() {
        let cfg = Config { cache_ram_mb: 0, ..Config::default() };
        let hw = gpu_hw(12_000, 13_359, 8);
        let g = meta();
        let p = compile(&input(&g, &hw, &cfg, &ALL_FLAGS), &TuningOverrides::default()).unwrap();
        assert!(!p.argv.contains(&"--cache-ram".to_string()));
    }

    #[test]
    fn unit__latency_affinity_knobs__emitted_when_set_absent_by_default() {
        let g = meta();
        let hw = gpu_hw(12_000, 32_000, 8);
        let p = compile(&input(&g, &hw, &Config::default(), &ALL_FLAGS), &TuningOverrides::default()).unwrap();
        assert!(!p.argv.contains(&"--cpu-range".to_string()));
        assert!(!p.argv.contains(&"--poll".to_string()));
        assert!(!p.argv.contains(&"--reasoning-format".to_string()));

        let cfg = Config {
            cpu_range: "0-15".into(),
            poll: 50,
            reasoning_format: "deepseek".into(),
            ..Config::default()
        };
        let p = compile(&input(&g, &hw, &cfg, &ALL_FLAGS), &TuningOverrides::default()).unwrap();
        assert!(p.argv.windows(2).any(|w| w[0] == "--cpu-range" && w[1] == "0-15"));
        assert!(p.argv.windows(2).any(|w| w[0] == "--poll" && w[1] == "50"));
        assert!(p.argv.windows(2).any(|w| w[0] == "--reasoning-format" && w[1] == "deepseek"));
    }

    #[test]
    fn unit__spec_ngram__emits_self_drafting_no_draft_model() {
        let cfg = Config { spec: "ngram".into(), ..Config::default() };
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        let p = compile(&input(&g, &hw, &cfg, &ALL_FLAGS), &TuningOverrides::default()).unwrap();
        assert!(p.argv.windows(2).any(|w| w[0] == "--spec-type" && w[1] == "ngram-simple"));
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
        let t = TuningOverrides { ubatch: Some(2048), ..TuningOverrides::default() };
        let p = compile(&input(&g, &hw, &Config::default(), &ALL_FLAGS), &t).unwrap();
        assert!(p.argv.windows(2).any(|w| w[0] == "--ubatch-size" && w[1] == "2048"));
        let p2 = compile(&input(&g, &hw, &Config::default(), &ALL_FLAGS), &TuningOverrides::default()).unwrap();
        assert!(!p2.argv.contains(&"--ubatch-size".to_string()));
    }

    #[test]
    fn unit__ctx_clamped_to_train_context() {
        let cfg = Config {
            default_ctx: 131_072, // > ctx_train 40960
            ..Config::default()
        };
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        let p = compile(&input(&g, &hw, &cfg, &ALL_FLAGS), &TuningOverrides::default()).unwrap();
        assert!(p.argv.windows(2).any(|w| w[0] == "--ctx-size" && w[1] == "40960"));
        assert_eq!(p.ctx, 40_960);
        assert!(p.warnings.iter().any(|w| w.contains("clamped")));
    }

    #[test]
    fn unit__ctx_train_missing__warning_not_guess() {
        let mut g = meta();
        g.context_length = None;
        let cfg = Config::default();
        let hw = gpu_hw(12_000, 32_000, 8);
        let p = compile(&input(&g, &hw, &cfg, &ALL_FLAGS), &TuningOverrides::default()).unwrap();
        assert!(p.argv.windows(2).any(|w| w[0] == "--ctx-size" && w[1] == "16384"));
        assert!(p.warnings.iter().any(|w| w.contains("lacks context_length")));
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
        let p = compile(&input(&g, &hw, &cfg, &ALL_FLAGS), &TuningOverrides::default()).unwrap();
        assert!(p.argv.windows(2).any(|w| w[0] == "--cache-type-k" && w[1] == "q8_0"));
        assert!(p.argv.windows(2).any(|w| w[0] == "--cache-type-v" && w[1] == "q8_0"));
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
        let p = compile(&input(&g, &hw, &cfg, &ALL_FLAGS), &TuningOverrides::default()).unwrap();
        assert!(p.argv.windows(2).any(|w| w[0] == "--cache-type-k" && w[1] == "q4_0"));
        assert!(p.argv.windows(2).any(|w| w[0] == "--cache-type-v" && w[1] == "q4_0"));
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
        let p = compile(&input(&g, &hw, &cfg, &ALL_FLAGS), &TuningOverrides::default()).unwrap();
        assert!(p.argv.windows(2).any(|w| w[0] == "--cache-type-k" && w[1] == "q5_0"));
        assert!(p.argv.windows(2).any(|w| w[0] == "--cache-type-v" && w[1] == "q5_0"));
    }

    #[test]
    fn unit__kv_quant__explicit_f16_forces_off() {
        let cfg = Config {
            cache_type: "f16".into(),
            ..Config::default()
        };
        let hw = gpu_hw(5_500, 32_000, 8); // ladder would engage q4_0
        let g = meta();
        let p = compile(&input(&g, &hw, &cfg, &ALL_FLAGS), &TuningOverrides::default()).unwrap();
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
        let p = compile(&input(&g, &hw, &cfg, &ALL_FLAGS), &TuningOverrides::default()).unwrap();
        assert!(p.argv.windows(2).any(|w| w[0] == "--rope-scaling" && w[1] == "yarn"));
        assert!(p.argv.windows(2).any(|w| w[0] == "--rope-scale" && w[1] == "2"));
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
        let p = compile(&input(&g, &hw, &cfg, &ALL_FLAGS), &TuningOverrides::default()).unwrap();
        assert!(p.argv.windows(2).any(|w| w[0] == "--n-cpu-moe" && w[1] == "12"));
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
        let p = compile(&input(&g, &hw, &cfg, &ALL_FLAGS), &TuningOverrides::default()).unwrap();
        assert!(!p.argv.contains(&"--agent".to_string()));
        let cfg2 = Config {
            agent: true,
            ..Config::default()
        };
        let p2 = compile(&input(&g, &hw, &cfg2, &ALL_FLAGS), &TuningOverrides::default()).unwrap();
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
        let p = compile(&input(&g, &hw, &cfg, &ALL_FLAGS), &TuningOverrides::default()).unwrap();
        assert!(!p.argv.contains(&"--lookup-cache-dynamic".to_string()));
        assert!(!p.argv.contains(&"--slot-save-path".to_string()));
        assert!(p.warnings.is_empty(), "{:?}", p.warnings);
    }

    #[test]
    fn unit__router_preset__sections_keys_and_bools() {
        let hw = gpu_hw(24_000, 32_000, 8);
        let g = meta();
        let cfg = Config::default();
        let p = compile(&input(&g, &hw, &cfg, &ALL_FLAGS), &TuningOverrides::default()).unwrap();
        // second model with an overlay knob
        let o = ModelOverride {
            ctx: Some(8192),
            ..ModelOverride::default()
        };
        let cfg2 = Config::default();
        let p2 = compile(&input(&g, &hw, &cfg2, &ALL_FLAGS), &TuningOverrides::default()).unwrap();
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
        assert!(ini.contains("ctx-size = 16384"));
        assert!(ini.contains("gpu-layers = auto"));
        assert!(ini.contains("parallel = 1"));
        assert!(ini.contains("cache-reuse = 256"));
    }

    #[test]
    fn unit__slot_prompt_similarity__emitted_only_when_set() {
        let hw = gpu_hw(24_000, 32_000, 8);
        let g = meta();
        let cfg = Config::default();
        let p = compile(&input(&g, &hw, &cfg, &ALL_FLAGS), &TuningOverrides::default()).unwrap();
        assert!(!p.argv.contains(&"--slot-prompt-similarity".to_string()));
        let cfg2 = Config {
            slot_prompt_similarity: 0.3,
            ..Config::default()
        };
        let p2 = compile(&input(&g, &hw, &cfg2, &ALL_FLAGS), &TuningOverrides::default()).unwrap();
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
        assert!(p.warnings.iter().any(|w| w.contains("--lookup-cache-dynamic")));
    }

    #[test]
    fn unit__kv_quant__head_count_kv_missing_falls_back_to_head_count() {
        // Upstream llama.cpp defaults head_count_kv = head_count when the
        // GGUF omits it (no-GQA models like qwen2.5-0.5b) — same rule here.
        let mut g = meta();
        g.head_count_kv = None; // falls back to head_count 16
        let cfg = Config::default();
        let hw = gpu_hw(5_500, 32_000, 8);
        let p = compile(&input(&g, &hw, &cfg, &ALL_FLAGS), &TuningOverrides::default()).unwrap();
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
        let p = compile(&input(&g, &hw, &cfg, &ALL_FLAGS), &TuningOverrides::default()).unwrap();
        assert!(p.argv.contains(&"--cpu-moe".to_string()));
        // RAM too small -> not emitted
        let hw_small = gpu_hw(2_000, 5_000, 8);
        let p2 = compile(&input(&g, &hw_small, &cfg, &full_flags()), &TuningOverrides::default()).unwrap();
        assert!(!p2.argv.contains(&"--cpu-moe".to_string()));
    }

    #[test]
    fn unit__no_gpu__no_sleep_no_gpu_layers_still_auto() {
        let cfg = Config::default();
        let hw = Hardware { physical_cores: 8, total_ram_mib: 32_000, gpus: vec![] };
        let g = meta();
        let p = compile(&input(&g, &hw, &cfg, &ALL_FLAGS), &TuningOverrides::default()).unwrap();
        assert!(!p.argv.contains(&"--sleep-idle-seconds".to_string()));
        assert!(p.argv.windows(2).any(|w| w[0] == "--gpu-layers" && w[1] == "auto"));
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
        assert!(p.argv.windows(2).any(|w| w[0] == "--rpc" && w[1] == "box1:50052,box2:50052"));
        assert!(p.argv.windows(2).any(|w| w[0] == "--lora" && w[1] == "/loras/a.bin"));
        assert!(p.argv.windows(2).any(|w| w[0] == "--lora-scaled" && w[1] == "/loras/b.bin:0.5"));
    }

    #[test]
    fn unit__unix_socket_endpoint() {
        let cfg = Config::default();
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        let mut inp = input(&g, &hw, &cfg, &ALL_FLAGS);
        inp.endpoint = Endpoint::Unix { socket: "/run/pallama/m.sock".into() };
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(p.argv.windows(2).any(|w| w[0] == "--host" && w[1] == "/run/pallama/m.sock"));
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
        assert!(err.contains("pallama pull ggml-org/Qwen3-0.6B-GGUF"), "{err}");

        // Draft pulled -> flags emitted.
        inp.draft_path = Some("/models/qwen3-0.6b-q4_k_m.gguf");
        let p = compile(&inp, &TuningOverrides::default()).unwrap();
        assert!(p.argv.windows(2).any(|w| w[0] == "--spec-type" && w[1] == "draft-simple"));
        assert!(p.argv.windows(2).any(|w| w[0] == "--spec-draft-model" && w[1] == "/models/qwen3-0.6b-q4_k_m.gguf"));
        assert!(p.argv.windows(2).any(|w| w[0] == "--spec-draft-n-max" && w[1] == "3"));
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
        assert!(p.argv.windows(2).any(|w| w[0] == "--tensor-split" && w[1] == "3,1"));

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
    fn unit__tuning_overrides__fa_and_batch_emitted() {
        let cfg = Config::default();
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        let t = TuningOverrides { fa: Some(false), batch: Some(1024), ..Default::default() };
        let p = compile(&input(&g, &hw, &cfg, &ALL_FLAGS), &t).unwrap();
        assert!(p.argv.windows(2).any(|w| w[0] == "--flash-attn" && w[1] == "off"));
        assert!(p.argv.windows(2).any(|w| w[0] == "-b" && w[1] == "1024"));
    }

    #[test]
    fn unit__tuning_overrides__win_over_heuristics() {
        let cfg = Config::default();
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        let t = TuningOverrides { ctx: Some(8192), threads: Some(6), kv_quant: Some(true), ..Default::default() };
        let p = compile(&input(&g, &hw, &cfg, &ALL_FLAGS), &t).unwrap();
        assert!(p.argv.windows(2).any(|w| w[0] == "--ctx-size" && w[1] == "8192"));
        assert!(p.argv.windows(2).any(|w| w[0] == "--threads" && w[1] == "6"));
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
        assert!(p.argv.windows(2).any(|w| w[0] == "-mm" && w[1] == "/models/mmproj.gguf"));
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
            p.argv.windows(2).any(|w| w[0] == "-mm" && w[1] == "/models/mmproj-F16.gguf"),
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
        assert!(p.argv.windows(2).any(|w| w[0] == "-mm" && w[1] == "/models/custom.gguf"));
        assert!(!p.argv.windows(2).any(|w| w[0] == "-mm" && w[1] == "/models/pulled.gguf"));
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
        assert!(!p.argv.iter().any(|a| a == "-mm"), "no -mm without manifest support");
    }

}
