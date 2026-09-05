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
    pub engine_tag: &'a str,
    /// Capability manifest flag set of the ACTIVE engine.
    pub supported_flags: &'a BTreeSet<String>,
    pub endpoint: Endpoint,
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
    argv.extend(["--jinja".into(), "--metrics".into(), "--flash-attn".into(), "auto".into()]);
    argv.push("--ctx-size".into());
    argv.push(ctx.to_string());

    // --- 3. threads
    let threads = tuning
        .threads
        .unwrap_or_else(|| input.hardware.physical_cores.max(1));
    argv.push("--threads".into());
    argv.push(threads.to_string());

    // --- 4. gpu layers: auto everywhere; upstream --fit on (default) shrinks
    argv.push("--gpu-layers".into());
    argv.push("auto".into());

    // --- 5. prefix-cache chunk reuse
    if config.cache_reuse > 0 {
        argv.push("--cache-reuse".into());
        argv.push(config.cache_reuse.to_string());
    }

    // --- 6. KV quant when GPU-resident and KV would push past 0.9x VRAM
    let vram_bytes = Hardware::bytes(input.hardware.total_vram_mib());
    let kv_quant = tuning
        .kv_quant
        .unwrap_or_else(|| kv_quant_eligible(input, vram_bytes, ctx, &mut warnings));
    if kv_quant {
        argv.extend([
            "--cache-type-k".into(),
            "q8_0".into(),
            "--cache-type-v".into(),
            "q8_0".into(),
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

    // --- 9. multi-slot continuous batching (-1 = auto per upstream).
    argv.push("-np".into());
    argv.push("-1".into());

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
    }

    // --- 12. prompt-cache budget + vision projector
    if config.cache_ram_mb > 0 {
        argv.push("--cache-ram".into());
        argv.push(config.cache_ram_mb.to_string());
    }
    if let Some(mmproj) = overlay.extra_args.as_deref().and_then(find_mmproj_arg) {
        argv.push("-mm".into());
        argv.push(mmproj.to_string());
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

/// Rule 6: KV quant when the model is GPU-resident and estimated KV
/// (`2*blocks*kv_heads*head_dim*ctx*2` bytes f16) + weights would pass
/// 0.9x VRAM. Missing GGUF fields skip the rule with a named warning.
fn kv_quant_eligible(
    input: &ProfileInput<'_>,
    vram_bytes: u64,
    ctx: u32,
    warnings: &mut Vec<String>,
) -> bool {
    let gpu_resident = input.hardware.has_gpu() && input.model_bytes <= vram_bytes;
    if !gpu_resident {
        return false;
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
            kv.saturating_add(input.model_bytes) > vram_bytes / 10 * 9
        }
        (None, _, _) => {
            warnings.push("KV-quant rule skipped: GGUF lacks block_count".into());
            false
        }
        (_, None, _) => {
            warnings.push("KV-quant rule skipped: GGUF lacks head_count_kv".into());
            false
        }
        (_, _, None) => {
            warnings.push("KV-quant rule skipped: head_dim not derivable".into());
            false
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
            "--cache-ram", "-mm", "--mmproj",
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
            engine_tag: "b-test",
            supported_flags: flags,
            endpoint: Endpoint::Tcp { host: "127.0.0.1".into(), port: 12345 },
        }
    }

    static ALL_FLAGS: LazyLock<BTreeSet<String>> = LazyLock::new(full_flags);
    static DEFAULT_OVERLAY: ModelOverride = ModelOverride {
        ctx: None,
        spec: None,
        loras: None,
        extra_args: None,
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
        // Rule 9: -np -1 (upstream auto encoding)
        assert!(p.argv.windows(2).any(|w| w[0] == "-np" && w[1] == "-1"));
        // Rule 12: cache-ram default 8192
        assert!(p.argv.windows(2).any(|w| w[0] == "--cache-ram" && w[1] == "8192"));
        assert_eq!(p.ctx, 16384);
        assert!(p.warnings.is_empty(), "{:?}", p.warnings);
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
    fn unit__kv_quant__engages_when_tight_vram() {
        // 5 GiB model on 5.5 GiB VRAM: KV 469MB pushes past 0.9x -> q8_0.
        let cfg = Config::default();
        let hw = gpu_hw(5_500, 32_000, 8);
        let g = meta();
        let p = compile(&input(&g, &hw, &cfg, &ALL_FLAGS), &TuningOverrides::default()).unwrap();
        assert!(p.argv.windows(2).any(|w| w[0] == "--cache-type-k" && w[1] == "q8_0"));
        assert!(p.argv.windows(2).any(|w| w[0] == "--cache-type-v" && w[1] == "q8_0"));
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
    fn unit__tuning_overrides__win_over_heuristics() {
        let cfg = Config::default();
        let hw = gpu_hw(12_000, 32_000, 8);
        let g = meta();
        let t = TuningOverrides { ctx: Some(8192), threads: Some(6), kv_quant: Some(true) };
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

}
