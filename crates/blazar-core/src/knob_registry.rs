//! Emission-truth registry of the first-class flags Blazar pins per engine
//! lane.
//!
//! The old capability auditor classified flags by grepping source literals,
//! which drifted from reality within one release. This module is the fix at
//! the source: it compiles real [`crate::profile::ProfileInput`] fixtures —
//! one per emission posture a configuration can reach — and collects the
//! flag tokens the profile compiler ACTUALLY emits. A flag that appears
//! here is provably wired end to end (config knob or derivation to argv);
//! a flag absent here is not, no matter what the source looks like.
//!
//! The committed `scripts/first_class_flags.json` is generated from these
//! fixtures by `cargo run -p blazar-runtime --example knob_registry`, and
//! `blazar-runtime/tests/knob_registry.rs` pins the compiled emission
//! against that file in both directions: adding emission without
//! regenerating (or hand-editing the file without changing emission)
//! fails CI. The runtime half (`blazar_runtime::knob_registry`) folds in
//! the per-engine argv builders that run at spawn time, so the JSON is
//! the full first-class surface, not just the compile-time slice.
//!
//! Fixture rule: UNDER-reporting is conservative (an unlisted flag lands
//! in the auditor's `unverified` bucket and gets investigated), so when
//! two knobs are mutually exclusive the fixtures split into variants
//! rather than guessing. Supported-flag sets are supersets of each lane's
//! literal universe; inert extras never emit, which keeps them harmless.

use std::collections::{BTreeMap, BTreeSet};

use crate::config::{
    Config, MistralrsTuning, MlxTuning, ModelOverride, SamplerDefaults, SglangTuning,
};
use crate::engine_kind::EngineKind;
use crate::gguf::GgufMeta;
use crate::hardware::{GpuInfo, Hardware};
use crate::hfmeta::{HfMeta, KvGeom, ModelMeta};
use crate::profile::{ComponentArg, Endpoint, Profile, ProfileInput, TuningOverrides, compile};

const MIB: u64 = 1024 * 1024;

const LLAMACPP_SUPPORTED: &[&str] = &[
    "--adaptive-decay",
    "--adaptive-target",
    "--agent",
    "--alias",
    "--batch-size",
    "--cache-ram",
    "--cache-reuse",
    "--cache-type-k",
    "--cache-type-v",
    "--calibration-file",
    "--chat-template",
    "--chat-template-file",
    "--chat-template-kwargs",
    "--check-tensors",
    "--checkpoint-min-step",
    "--cont-batching",
    "--context-shift",
    "--control-vector",
    "--control-vector-layer-range",
    "--control-vector-scaled",
    "--cpu-moe",
    "--cpu-range",
    "--cpu-strict",
    "--ctx-checkpoints",
    "--ctx-size",
    "--device",
    "--device-layers",
    "--disable-access-log",
    "--disable-metrics",
    "--dry-allowed-length",
    "--dry-base",
    "--dry-multiplier",
    "--dry-penalty-last-n",
    "--embd-normalize",
    "--embeddings",
    "--enable-lora",
    "--encoder-cache-memory-mb",
    "--flash-attn",
    "--frequency-penalty",
    "--gpu-layers",
    "--host",
    "--image-max-tokens",
    "--image-min-tokens",
    "--imatrix",
    "--isq",
    "--isq-organization",
    "--jinja",
    "--keep",
    "--kv-unified",
    "--kv-unified-per-slot",
    "--lazy-mode",
    "--load-mode",
    "--lookup-cache-dynamic",
    "--lookup-cache-static",
    "--lora",
    "--lora-init-without-apply",
    "--lora-max-adapters",
    "--lora-max-bytes",
    "--lora-max-rank",
    "--lora-scaled",
    "--main-gpu",
    "--max-batch-size",
    "--max-decode-steps-before-prefill",
    "--max-image-length",
    "--max-num-images",
    "--max-prefill-chunk-tokens",
    "--mcp-servers-config",
    "--mcp-servers-json",
    "--metrics",
    "--min-p",
    "--mirostat",
    "--mmproj",
    "--mmproj-device",
    "--mtmd-batch-max-tokens",
    "--mtp",
    "--mtp-draft-sampling",
    "--mtp-model",
    "--mtp-n-predict",
    "--n-cpu-ffn",
    "--n-cpu-moe",
    "--no-cache-idle-slots",
    "--no-cont-batching",
    "--no-host",
    "--no-kv-offload",
    "--no-kv-unified",
    "--no-mmproj-auto",
    "--no-mmproj-offload",
    "--no-op-offload",
    "--no-reasoning-preserve",
    "--no-repack",
    "--no-spec-draft-backend-sampling",
    "--no-warmup",
    "--numa",
    "--op-offload",
    "--override-kv",
    "--override-tensor",
    "--pa-block-size",
    "--pa-cache-type",
    "--pa-context-len",
    "--pa-memory-fraction",
    "--pa-memory-mb",
    "--paged-attn",
    "--poll",
    "--poll-batch",
    "--pooling",
    "--port",
    "--prefix-cache-n",
    "--presence-penalty",
    "--prio",
    "--prio-batch",
    "--props",
    "--reasoning",
    "--reasoning-budget",
    "--reasoning-budget-message",
    "--reasoning-effort",
    "--reasoning-format",
    "--reasoning-preserve",
    "--repeat-last-n",
    "--repeat-penalty",
    "--reranking",
    "--reuse-port",
    "--rope-freq-base",
    "--rope-freq-scale",
    "--rope-scale",
    "--rope-scaling",
    "--rpc",
    "--samplers",
    "--seed",
    "--sleep-idle-seconds",
    "--slot-prompt-similarity",
    "--slot-save-path",
    "--spec-draft-backend-sampling",
    "--spec-draft-cpu-moe",
    "--spec-draft-cpu-range",
    "--spec-draft-cpu-strict",
    "--spec-draft-cpu-strict-batch",
    "--spec-draft-device",
    "--spec-draft-model",
    "--spec-draft-n-cpu-moe",
    "--spec-draft-n-max",
    "--spec-draft-ngl",
    "--spec-draft-override-tensor",
    "--spec-draft-p-min",
    "--spec-draft-p-split",
    "--spec-draft-poll",
    "--spec-draft-poll-batch",
    "--spec-draft-prio",
    "--spec-draft-prio-batch",
    "--spec-draft-threads",
    "--spec-draft-threads-batch",
    "--spec-draft-type-k",
    "--spec-draft-type-v",
    "--spec-ngram-map-k-min-hits",
    "--spec-ngram-map-k-size-m",
    "--spec-ngram-map-k-size-n",
    "--spec-ngram-map-k4v-min-hits",
    "--spec-ngram-map-k4v-size-m",
    "--spec-ngram-map-k4v-size-n",
    "--spec-ngram-mod-n-match",
    "--spec-ngram-mod-n-max",
    "--spec-ngram-mod-n-min",
    "--spec-ngram-simple-min-hits",
    "--spec-ngram-simple-size-m",
    "--spec-ngram-simple-size-n",
    "--spec-type",
    "--split-mode",
    "--spm-infill",
    "--sse-ping-interval",
    "--swa-full",
    "--temp",
    "--tensor-split",
    "--threads",
    "--threads-batch",
    "--threads-http",
    "--timeout",
    "--tools",
    "--tools-runtime",
    "--top-k",
    "--top-n-sigma",
    "--top-p",
    "--typical-p",
    "--ubatch-size",
    "--video-ffmpeg-dir",
    "--video-fps",
    "--video-timestamp-interval",
    "--xtc-probability",
    "--xtc-threshold",
    "--yarn-attn-factor",
    "--yarn-beta-fast",
    "--yarn-beta-slow",
    "--yarn-ext-factor",
    "--yarn-orig-ctx",
    "-b",
    "-m",
    "-mm",
    "-np",
];

/// Supported-flag superset for the mistral.rs lane — the manifest a
/// fully-flagged engine build would advertise. Public because the
/// runtime registry half feeds it to `mistralrs_argv`, whose
/// builder-added flags are manifest-gated: the superset keeps builder
/// emission fixture-honest without depending on one installed build.
pub const MISTRALRS_SUPPORTED: &[&str] = &[
    "--calibration-file",
    "--chat-template",
    "--device-layers",
    "--disable-access-log",
    "--disable-metrics",
    "--enable-lora",
    "--encoder-cache-memory-mb",
    "--host",
    "--imatrix",
    "--isq",
    "--isq-organization",
    "--lora",
    "--lora-max-adapters",
    "--lora-max-bytes",
    "--lora-max-rank",
    "--max-batch-size",
    "--max-decode-steps-before-prefill",
    "--max-image-length",
    "--max-model-len",
    "--max-num-batched-tokens",
    "--max-num-images",
    "--max-prefill-chunk-tokens",
    "--max-seqs",
    "--mmproj",
    "--mtp",
    "--mtp-draft-sampling",
    "--mtp-model",
    "--mtp-n-predict",
    "--no-ui",
    "--pa-block-size",
    "--pa-cache-type",
    "--pa-context-len",
    "--pa-memory-fraction",
    "--pa-memory-mb",
    "--paged-attn",
    "--port",
    "--prefix-cache-n",
    "-f",
    "-m",
    "-np",
];

const SDCPP_SUPPORTED: &[&str] = &[
    "--api-key",
    "--audio-encoder",
    "--backend",
    "--cache-mode",
    "--cache-option",
    "--cfg-scale",
    "--clip_g",
    "--clip_l",
    "--clip_vision",
    "--conditioning-cache-size",
    "--context-length",
    "--control-net",
    "--cpu-offload-gb",
    "--diffusion-fa",
    "--diffusion-model",
    "--embd-dir",
    "--enable-lora",
    "--fa",
    "--flow-shift",
    "--height",
    "--high-noise-diffusion-model",
    "--hires-upscalers-dir",
    "--host",
    "--ip-adapter",
    "--listen-ip",
    "--listen-port",
    "--llm",
    "--llm_vision",
    "--lora-model-dir",
    "--lora-paths",
    "--max-vram",
    "--mem-fraction-static",
    "--model",
    "--model-args",
    "--model-path",
    "--motion-module",
    "--offload-to-cpu",
    "--params-backend",
    "--photo-maker",
    "--port",
    "--pulid-weights",
    "--qwen2vl_vision",
    "--rpc-servers",
    "--sage-attn",
    "--sampling-method",
    "--seed",
    "--served-model-name",
    "--split-mode",
    "--t5xxl",
    "--tae",
    "--tensor-type-rules",
    "--threads",
    "--uncond-diffusion-model",
    "--upscale-model",
    "--vae",
    "--vae-tiling",
    "--width",
    "-t",
];

const MLX_SUPPORTED: &[&str] = &[
    "--adapter-path",
    "--chat-template",
    "--chat-template-args",
    "--decode-concurrency",
    "--draft-model",
    "--host",
    "--kv-bits",
    "--kv-group-size",
    "--model",
    "--num-draft-tokens",
    "--port",
    "--prefill-step-size",
    "--prompt-cache-bytes",
    "--prompt-cache-size",
    "--prompt-concurrency",
    "--quantized-kv-start",
    "--trust-remote-code",
    "--use-default-chat-template",
];

const SGLANG_SUPPORTED: &[&str] = &[
    "--attention-backend",
    "--batch-notify-size",
    "--cache-ram",
    "--chunked-prefill-size",
    "--context-length",
    "--cpu-offload-gb",
    "--cuda-graph-backend-prefill",
    "--cuda-graph-bs",
    "--cuda-graph-bs-decode",
    "--cuda-graph-bs-prefill",
    "--cuda-graph-config",
    "--cuda-graph-max-bs",
    "--cuda-graph-max-bs-decode",
    "--cuda-graph-max-bs-prefill",
    "--detokenizer-worker-num",
    "--device",
    "--disable-prefill-cuda-graph",
    "--dp-size",
    "--dtype",
    "--dynamic-batch-tokenizer-batch-size",
    "--dynamic-batch-tokenizer-batch-timeout",
    "--enable-cache-report",
    "--enable-deterministic-inference",
    "--enable-dynamic-batch-tokenizer",
    "--enable-hierarchical-cache",
    "--enable-lora",
    "--enable-memory-saver",
    "--enable-metrics",
    "--enable-mixed-chunk",
    "--enable-priority-scheduling",
    "--enable-session-radix-cache",
    "--enable-tf32-matmul",
    "--enable-torch-compile",
    "--enable-two-batch-overlap",
    "--ep-size",
    "--grammar-backend",
    "--hicache-ratio",
    "--hicache-size",
    "--is-embedding",
    "--kv-cache-dtype",
    "--kv-unified",
    "--lora-backend",
    "--lora-paths",
    "--max-lora-rank",
    "--max-prefill-tokens",
    "--max-running-requests",
    "--max-total-tokens",
    "--mem-fraction-static",
    "--mm-attention-backend",
    "--mmproj",
    "--no-kv-offload",
    "--page-size",
    "--pp-size",
    "--quantization",
    "--radix-eviction-policy",
    "--random-seed",
    "--reasoning-parser",
    "--retraction-policy",
    "--sampling-backend",
    "--schedule-conservativeness",
    "--schedule-policy",
    "--scheduler-recv-interval",
    "--skip-server-warmup",
    "--sleep-on-idle",
    "--spec-draft-model",
    "--spec-draft-n-max",
    "--spec-type",
    "--speculative-accept-threshold-acc",
    "--speculative-accept-threshold-single",
    "--speculative-algorithm",
    "--speculative-draft-model-path",
    "--speculative-eagle-topk",
    "--speculative-num-draft-tokens",
    "--speculative-num-steps",
    "--stream-interval",
    "--tokenizer-backend",
    "--tokenizer-mode",
    "--tokenizer-path",
    "--tokenizer-worker-num",
    "--tool-call-parser",
    "--tp-size",
    "--watchdog-timeout",
    "-b",
    "-cuda",
    "-m",
    "-mm",
    "-np",
];

/// First-class flag tokens per lane, in sorted order, as compiled from the
/// fixture matrix in this module. Keys are the engine lane names used by
/// `scripts/first_class_flags.json` and the capability auditor.
///
/// This is the COMPILE-time slice only: llama.cpp and stable-diffusion.cpp
/// children spawn `profile.argv` verbatim, but mistral.rs, `SGLang` and `MLX`
/// add flags in their spawn-time argv builders —
/// `blazar_runtime::knob_registry::lane_flags` folds those in and is the
/// function behind the committed registry JSON.
#[must_use]
pub fn lane_flags() -> BTreeMap<&'static str, Vec<String>> {
    BTreeMap::from([
        ("llamacpp", llamacpp_lane()),
        ("mistralrs", mistralrs_lane()),
        ("sglang", sglang_lane()),
        ("sdcpp", sdcpp_lane()),
        ("mlx", mlx_lane()),
    ])
}

/// Compiled fixture profiles for the three lanes whose spawn-time argv
/// builders translate or extend the profile (mistral.rs, `SGLang`, `MLX`).
/// llama.cpp and stable-diffusion.cpp children run `profile.argv`
/// verbatim, so for those lanes the compile-time token union in
/// [`lane_flags`] is already the full spawn truth. The runtime registry
/// half feeds these profiles through the real builders and folds in
/// what they add on top.
#[must_use]
pub fn lane_profiles() -> BTreeMap<&'static str, Vec<Profile>> {
    BTreeMap::from([
        ("mistralrs", mistralrs_profiles()),
        ("sglang", sglang_profiles()),
        ("mlx", mlx_profiles()),
    ])
}

/// Flag-token union over a lane's compiled profiles.
fn lane_tokens(profiles: &[Profile]) -> Vec<String> {
    let mut union = BTreeSet::new();
    for profile in profiles {
        union.extend(profile.argv.iter().filter(|t| is_flag_token(t)).cloned());
    }
    union.into_iter().collect()
}

/// A token is a flag when it starts with `--`, or is a short flag like
/// `-m`/`-np` (letter, not a negative-number value such as `-1`).
#[must_use]
pub fn is_flag_token(token: &str) -> bool {
    token.starts_with("--")
        || (token.len() >= 2 && token.starts_with('-') && token.as_bytes()[1].is_ascii_alphabetic())
}

fn supported(flags: &[&str]) -> BTreeSet<String> {
    flags.iter().map(|f| (*f).to_string()).collect()
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

fn gpu4_hw(vram_mib: u64, ram_mib: u64) -> Hardware {
    Hardware {
        physical_cores: 16,
        total_ram_mib: ram_mib,
        gpus: (0..4)
            .map(|_| GpuInfo {
                name: "RTX".into(),
                description: "NVIDIA CUDA".into(),
                total_mib: vram_mib,
                free_mib: vram_mib,
            })
            .collect(),
    }
}

fn gguf_meta() -> GgufMeta {
    GgufMeta {
        architecture: "qwen3".into(),
        name: Some("x".into()),
        basename: Some("Qwen_Qwen3.5".into()),
        size_label: Some("9B".into()),
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

/// Embedding-class GGUF: drives the `--embeddings`/pooling profile rules.
fn embedding_gguf_meta() -> GgufMeta {
    GgufMeta {
        pooling_type: Some(1),
        ..gguf_meta()
    }
}

fn hf_meta() -> HfMeta {
    HfMeta {
        architecture: "Qwen2ForCausalLM".into(),
        ctx_train: Some(32_768),
        dtype: Some("bfloat16".into()),
        quant_bits: None,
        quant_method: None,
        vision_tower: false,
        kv: KvGeom {
            layers: Some(28),
            kv_heads: Some(2),
            head_dim: Some(128),
        },
    }
}

#[allow(clippy::too_many_arguments)]
fn base_input<'a>(
    kind: EngineKind,
    meta: ModelMeta<'a>,
    model_bytes: u64,
    hw: &'a Hardware,
    cfg: &'a Config,
    overlay: &'a ModelOverride,
    flags: &'a BTreeSet<String>,
) -> ProfileInput<'a> {
    ProfileInput {
        engine_kind: kind,
        model_name: "qwen3.5-9b",
        instance_key: "qwen3.5-9b",
        model_path: "/models/qwen3.5-9b.d",
        model_bytes,
        meta,
        hardware: hw,
        config: cfg,
        overlay,
        spec_mode: "off",
        loras: &[],
        draft_path: None,
        draft_gguf: None,
        mmproj_path: None,
        components: &[],
        mmproj_force: false,
        engine_tag: "registry",
        supported_flags: flags,
        spec_types: &[],
        endpoint: Endpoint::Tcp {
            host: "127.0.0.1".into(),
            port: 12345,
        },
        data_dir: "/tmp/blazar-registry",
        cache_hit_rate: None,
        resident_ram_mib: 0,
        device_hint: None,
        engine_census: hw.gpus.clone(),
        sibling_devices: Vec::new(),
        auto_tensor_split: None,
        auto_tp_size: None,
    }
}

/// Compile one fixture and collect it. A registry fixture that fails to
/// compile is a bug in this module, not a condition to tolerate — hence
/// the panic with the compile error.
fn absorb(profiles: &mut Vec<Profile>, inp: &ProfileInput<'_>, tuning: &TuningOverrides) {
    let profile =
        compile(inp, tuning).unwrap_or_else(|e| panic!("registry fixture must compile: {e}"));
    profiles.push(profile);
}

fn llamacpp_knob_cfg() -> Config {
    Config {
        sse_ping_interval: Some(-1),
        server_timeout_secs: Some(600),
        chat_template_kwargs: Some(r#"{"enable_thinking":false}"#.into()),
        cont_batching: Some(false),
        reuse_port: true,
        lora_init_without_apply: true,
        flash_attention: Some(true),
        lookup_cache_static: Some("/lookup/static.bin".into()),
        lookup_cache_dynamic: Some("/lookup/dynamic.bin".into()),
        rope_freq_base: Some(10_000.0),
        rope_freq_scale: Some(1.0),
        swa_full: true,
        no_kv_offload: true,
        poll_batch: Some(true),
        op_offload: Some(true),
        kv_unified: Some(true),
        checkpoint_min_step: Some(16),
        context_shift: true,
        cache_idle_slots: true,
        adaptive_slots: true,
        warmup: false,
        no_host: true,
        spec_draft_p_min: Some(0.9),
        spec_draft_p_split: Some(0.28),
        spec_draft_poll: Some(16),
        spec_draft_poll_batch: Some(true),
        reasoning_preserve: Some(false),
        ..Config::default()
    }
}

fn tuned_overrides() -> TuningOverrides {
    TuningOverrides {
        ctx: Some(8192),
        kv_quant: Some(true),
        threads: Some(8),
        fa: Some(true),
        batch: Some(1024),
        ubatch: Some(256),
    }
}

/// Real (tiny) draft GGUF: compile validates draft existence, and the
/// registry wants the speculative flags emitted, not the stale-row
/// teaching error.
fn draft_file() -> String {
    let p = std::env::temp_dir().join(format!("blazar-registry-draft-{}.gguf", std::process::id()));
    std::fs::write(&p, b"gguf").expect("draft fixture");
    p.to_string_lossy().into_owned()
}

fn llamacpp_lane() -> Vec<String> {
    lane_tokens(&llamacpp_profiles())
}

#[allow(clippy::too_many_lines)] // one flat fixture matrix, not logic
fn llamacpp_profiles() -> Vec<Profile> {
    let g = gguf_meta();
    let g_emb = embedding_gguf_meta();
    let hw = gpu_hw(24_000, 32_000, 8);
    let flags = supported(LLAMACPP_SUPPORTED);
    let base = Config::default();
    let knobs = llamacpp_knob_cfg();
    let plain = ModelOverride::default();
    let mut profiles = Vec::new();

    // Every-plain baseline: geometry-derived pins only.
    absorb(
        &mut profiles,
        &base_input(
            EngineKind::LlamaCpp,
            ModelMeta::Gguf(&g),
            5_000 * MIB,
            &hw,
            &base,
            &plain,
            &flags,
        ),
        &TuningOverrides::default(),
    );

    // Global config knobs (server hygiene, cache, rope, spec posture).
    absorb(
        &mut profiles,
        &base_input(
            EngineKind::LlamaCpp,
            ModelMeta::Gguf(&g),
            5_000 * MIB,
            &hw,
            &knobs,
            &plain,
            &flags,
        ),
        &TuningOverrides::default(),
    );

    // Tuner-adopted pins (bench winner applied at spawn).
    absorb(
        &mut profiles,
        &base_input(
            EngineKind::LlamaCpp,
            ModelMeta::Gguf(&g),
            5_000 * MIB,
            &hw,
            &base,
            &plain,
            &flags,
        ),
        &tuned_overrides(),
    );

    // Per-model overlay knobs: sampler defaults, chat template, LoRA scaling,
    // tensor/device placement, late chunking.
    let loras = vec![("style-lora".to_string(), 0.7)];
    let rich = ModelOverride {
        ctx: Some(4096),
        slots: Some(2),
        cache_type: Some("q8_0".into()),
        chat_template: Some("chatml".into()),
        sampler_defaults: Some(SamplerDefaults {
            temperature: Some(0.8),
            top_k: Some(40),
            top_p: Some(0.9),
            min_p: Some(0.05),
            top_n_sigma: Some(2.0),
            typical_p: Some(0.9),
            repeat_penalty: Some(1.1),
            repeat_last_n: Some(64),
            presence_penalty: Some(0.5),
            frequency_penalty: Some(0.3),
            dry_multiplier: Some(0.8),
            dry_base: Some(1.75),
            dry_allowed_length: Some(2),
            dry_penalty_last_n: Some(64),
            xtc_probability: Some(0.5),
            xtc_threshold: Some(0.1),
            mirostat: Some(1),
            seed: Some(42),
        }),
        reasoning_budget: Some(1024),
        reasoning_effort: Some("medium".into()),
        spm_infill: Some(true),
        override_tensor: Some(vec!["model.=q8_0".into()]),
        devices: Some(vec!["CUDA0".into()]),
        rpc_servers: Some("10.0.0.2:50052".into()),
        tensor_split: Some("3,1".into()),
        late_chunking: Some(true),
        deterministic: Some(true),
        cpu_moe_n: Some(2),
        cpu_ffn_n: Some(2),
        ctx_extend: Some(1.2),
        ..ModelOverride::default()
    };
    let mut overlay_inp = base_input(
        EngineKind::LlamaCpp,
        ModelMeta::Gguf(&g),
        5_000 * MIB,
        &hw,
        &knobs,
        &rich,
        &flags,
    );
    overlay_inp.loras = &loras;
    absorb(&mut profiles, &overlay_inp, &TuningOverrides::default());

    // Speculative draft posture: gguf draft pair + spec knobs.
    let mut spec_inp = base_input(
        EngineKind::LlamaCpp,
        ModelMeta::Gguf(&g),
        5_000 * MIB,
        &hw,
        &knobs,
        &plain,
        &flags,
    );
    spec_inp.spec_mode = "auto";
    let draft = draft_file();
    spec_inp.draft_path = Some(&draft);
    spec_inp.draft_gguf = Some(&g);
    absorb(&mut profiles, &spec_inp, &TuningOverrides::default());

    // Multimodal projector posture.
    let mut mm_inp = base_input(
        EngineKind::LlamaCpp,
        ModelMeta::Gguf(&g),
        5_000 * MIB,
        &hw,
        &knobs,
        &plain,
        &flags,
    );
    mm_inp.mmproj_path = Some("/models/mmproj.gguf");
    let mm_cfg = Config {
        mmproj_offload: false,
        mmproj_auto: false,
        ..knobs.clone()
    };
    absorb(
        &mut profiles,
        &{
            let mut i = mm_inp;
            i.config = &mm_cfg;
            i
        },
        &TuningOverrides::default(),
    );

    // Embedding-class model (pooling/embeddings rules).
    absorb(
        &mut profiles,
        &base_input(
            EngineKind::LlamaCpp,
            ModelMeta::Gguf(&g_emb),
            1_000 * MIB,
            &hw,
            &base,
            &plain,
            &flags,
        ),
        &TuningOverrides::default(),
    );

    profiles
}

fn mistralrs_lane() -> Vec<String> {
    lane_tokens(&mistralrs_profiles())
}

#[allow(clippy::too_many_lines)] // one flat fixture matrix, not logic
fn mistralrs_profiles() -> Vec<Profile> {
    let g = gguf_meta();
    let hf = hf_meta();
    let hw = gpu_hw(12_000, 32_000, 8);
    let flags = supported(MISTRALRS_SUPPORTED);
    let base = Config::default();
    let plain = ModelOverride::default();
    let mut profiles = Vec::new();

    // GGUF loader baseline (geometry + ladder derives).
    absorb(
        &mut profiles,
        &base_input(
            EngineKind::MistralRs,
            ModelMeta::Gguf(&g),
            942 * MIB,
            &hw,
            &base,
            &plain,
            &flags,
        ),
        &TuningOverrides::default(),
    );

    // Full MistralrsTuning knob surface (validated values; the compile
    // error IS the validation oracle here).
    let knobs = Config {
        mistralrs: MistralrsTuning {
            max_batch_size: Some(32),
            max_prefill_chunk_tokens: Some(2048),
            max_decode_steps_before_prefill: Some(16),
            prefix_cache_n: Some(32),
            pa_block_size: Some(32),
            pa_cache_type: Some("f8e4m3".into()),
            pa_context_len: Some(32_768),
            pa_memory_mb: Some(2048),
            enable_lora: Some(true),
            lora_max_rank: Some(32),
            lora_max_adapters: Some(4),
            lora_max_bytes: Some(268_435_456),
            mtp: Some(true),
            mtp_model: Some("mtp-head".into()),
            mtp_n_predict: Some(2),
            mtp_draft_sampling: Some("auto".into()),
            encoder_cache_memory_mb: Some(256),
            max_num_images: Some(2),
            max_image_length: Some(2048),
            disable_metrics: Some(true),
            disable_access_log: Some(true),
            chat_template: Some("test-template".into()),
            isq: Some("q4k".into()),
            imatrix: Some("/models/imatrix.gguf".into()),
            calibration_file: Some("/models/calib.txt".into()),
            isq_organization: Some("default".into()),
            device_layers: Some("0:14".into()),
        },
        ..Config::default()
    };
    absorb(
        &mut profiles,
        &base_input(
            EngineKind::MistralRs,
            ModelMeta::Gguf(&g),
            942 * MIB,
            &hw,
            &knobs,
            &plain,
            &flags,
        ),
        &TuningOverrides::default(),
    );

    // Legacy top-level paged-attention pair (pre-tuning knobs kept as the
    // pinned-fraction path).
    let pa = Config {
        mistralrs_paged_attn: Some(true),
        mistralrs_pa_memory_fraction: Some(0.5),
        ..Config::default()
    };
    absorb(
        &mut profiles,
        &base_input(
            EngineKind::MistralRs,
            ModelMeta::Gguf(&g),
            942 * MIB,
            &hw,
            &pa,
            &plain,
            &flags,
        ),
        &TuningOverrides::default(),
    );

    // HF (safetensors dir) loader geometry.
    absorb(
        &mut profiles,
        &base_input(
            EngineKind::MistralRs,
            ModelMeta::Hf(&hf),
            5_000 * MIB,
            &hw,
            &base,
            &plain,
            &flags,
        ),
        &TuningOverrides::default(),
    );

    // Slot-count posture: compiles `-np N` into the profile, which the
    // spawn-time builder translates to `--max-seqs` (manifest-gated).
    let slotted = ModelOverride {
        slots: Some(2),
        ..ModelOverride::default()
    };
    absorb(
        &mut profiles,
        &base_input(
            EngineKind::MistralRs,
            ModelMeta::Gguf(&g),
            942 * MIB,
            &hw,
            &base,
            &slotted,
            &flags,
        ),
        &TuningOverrides::default(),
    );

    profiles
}

fn sglang_lane() -> Vec<String> {
    lane_tokens(&sglang_profiles())
}

#[allow(clippy::too_many_lines)] // one flat fixture matrix, not logic
fn sglang_profiles() -> Vec<Profile> {
    let hf = hf_meta();
    let hw = gpu_hw(12_000, 32_000, 8);
    let hw4 = gpu4_hw(12_000, 32_000);
    let flags = supported(SGLANG_SUPPORTED);
    let base = Config::default();
    let plain = ModelOverride::default();
    let mut profiles = Vec::new();

    // Baseline ladder derives (ctx clamp, cuda-graph sizing, cache policy).
    absorb(
        &mut profiles,
        &base_input(
            EngineKind::Sglang,
            ModelMeta::Hf(&hf),
            8_000 * MIB,
            &hw,
            &base,
            &plain,
            &flags,
        ),
        &TuningOverrides::default(),
    );

    // Full SglangTuning knob surface. is_embedding and >1 parallel sizes
    // are separate postures (below) because they change the child's shape.
    let knobs = Config {
        sglang: SglangTuning {
            attention_backend: Some("flashinfer".into()),
            sampling_backend: Some("flashinfer".into()),
            tool_call_parser: Some("qwen25".into()),
            reasoning_parser: Some("qwen3".into()),
            tokenizer_path: Some("/models/tokenizer.d".into()),
            dtype: Some("bfloat16".into()),
            quantization: Some("awq".into()),
            kv_cache_dtype: Some("fp8_e4m3".into()),
            mem_fraction_static: Some(0.8),
            cpu_offload_gb: Some(2.0),
            page_size: Some(64),
            schedule_policy: Some("lpm".into()),
            schedule_conservativeness: Some(1.5),
            chunked_prefill_size: Some(4096),
            max_prefill_tokens: Some(16_384),
            stream_interval: Some(4),
            random_seed: Some(42),
            cuda_graph_max_bs: Some(32),
            hicache_enable: Some(true),
            hicache_ratio: Some(2.0),
            hicache_size: Some(4.0),
            metrics: Some(true),
            skip_warmup: Some(true),
            torch_compile: Some(true),
            tokenizer_mode: Some("auto".into()),
            tokenizer_backend: Some("huggingface".into()),
            tokenizer_worker_num: Some(2),
            detokenizer_worker_num: Some(2),
            dynamic_batch_tokenizer: Some(true),
            dynamic_batch_tokenizer_batch_size: Some(64),
            dynamic_batch_tokenizer_batch_timeout: Some(1.0),
            grammar_backend: Some("xgrammar".into()),
            radix_eviction_policy: Some("lru".into()),
            session_radix_cache: Some(true),
            mixed_chunk: Some(true),
            sleep_on_idle: Some(true),
            memory_saver: Some(true),
            watchdog_timeout: Some(300.0),
            cache_report: Some(true),
            batch_notify_size: Some(8),
            scheduler_recv_interval: Some(2),
            cuda_graph_bs: Some(vec![1, 2, 4]),
            cuda_graph_backend_prefill: Some("full".into()),
            max_total_tokens: Some(32_768),
            max_lora_rank: Some(64),
            lora_backend: Some("flashinfer".into()),
            mm_attention_backend: Some("sdpa".into()),
            retraction_policy: Some("retract_decode".into()),
            enable_two_batch_overlap: Some(true),
            enable_tf32_matmul: Some(true),
            enable_priority_scheduling: Some(true),
            ..SglangTuning::default()
        },
        ..Config::default()
    };
    let mut knob_inp = base_input(
        EngineKind::Sglang,
        ModelMeta::Hf(&hf),
        8_000 * MIB,
        &hw,
        &knobs,
        &plain,
        &flags,
    );
    let loras = vec![("style-lora".to_string(), 0.7)];
    knob_inp.loras = &loras;
    absorb(&mut profiles, &knob_inp, &TuningOverrides::default());

    // Draft-free NGRAM speculation posture.
    let ngram = Config {
        sglang: SglangTuning {
            spec_algorithm: Some("ngram".into()),
            spec_num_steps: Some(3),
            spec_num_draft_tokens: Some(4),
            speculative_eagle_topk: Some(4),
            speculative_accept_threshold_single: Some(0.75),
            speculative_accept_threshold_acc: Some(0.85),
            ..SglangTuning::default()
        },
        ..Config::default()
    };
    absorb(
        &mut profiles,
        &base_input(
            EngineKind::Sglang,
            ModelMeta::Hf(&hf),
            8_000 * MIB,
            &hw,
            &ngram,
            &plain,
            &flags,
        ),
        &TuningOverrides::default(),
    );

    // Resolved EAGLE draft pair (real draft file; unknown paths get
    // dropped with a teaching warning instead of emitting spec flags).
    let mut draft_inp = base_input(
        EngineKind::Sglang,
        ModelMeta::Hf(&hf),
        8_000 * MIB,
        &hw,
        &base,
        &plain,
        &flags,
    );
    let draft = draft_file();
    draft_inp.draft_path = Some(&draft);
    absorb(&mut profiles, &draft_inp, &TuningOverrides::default());

    // Embedding posture (dedicated embedding model entries).
    let embed = Config {
        sglang: SglangTuning {
            is_embedding: Some(true),
            ..SglangTuning::default()
        },
        ..Config::default()
    };
    absorb(
        &mut profiles,
        &base_input(
            EngineKind::Sglang,
            ModelMeta::Hf(&hf),
            8_000 * MIB,
            &hw,
            &embed,
            &plain,
            &flags,
        ),
        &TuningOverrides::default(),
    );

    // Parallel sharding posture (tp/dp/pp/ep > 1 needs the census to
    // plausibly host it).
    let parallel = Config {
        sglang: SglangTuning {
            tp_size: Some(2),
            dp_size: Some(2),
            pp_size: Some(2),
            ep_size: Some(2),
            ..SglangTuning::default()
        },
        ..Config::default()
    };
    absorb(
        &mut profiles,
        &base_input(
            EngineKind::Sglang,
            ModelMeta::Hf(&hf),
            8_000 * MIB,
            &hw4,
            &parallel,
            &plain,
            &flags,
        ),
        &TuningOverrides::default(),
    );

    profiles
}

fn sdcpp_lane() -> Vec<String> {
    lane_tokens(&sdcpp_profiles())
}

#[allow(clippy::too_many_lines)] // one flat fixture matrix, not logic
fn sdcpp_profiles() -> Vec<Profile> {
    let g = gguf_meta();
    let hw = gpu_hw(16_384, 32_000, 8);
    let hw_tight = gpu_hw(6_144, 32_000, 8);
    let flags = supported(SDCPP_SUPPORTED);
    let base = Config::default();
    let plain = ModelOverride::default();
    let mut profiles = Vec::new();

    // Component files are real (tiny) fixtures: the offload ladder reads
    // their metadata, and compile treats missing components as a repull
    // teaching case instead of an argv to register.
    let dir = std::env::temp_dir().join(format!("blazar-knob-registry-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("registry fixture dir");
    let vae = dir.join("vae.safetensors");
    let llm = dir.join("te.gguf");
    let t5 = dir.join("t5.gguf");
    let clip = dir.join("clip_l.safetensors");
    for (p, bytes) in [(&vae, b"v"), (&llm, b"t"), (&t5, b"t"), (&clip, b"c")] {
        std::fs::write(p, bytes).expect("component fixture");
    }
    let scan = defer_remove(&dir);

    // Qwen-Image dialect baseline (DiT + --vae + --llm).
    let set_qwen = [
        ComponentArg::new("--vae", vae.to_str().unwrap_or("/vae")),
        ComponentArg::new("--llm", llm.to_str().unwrap_or("/llm")),
    ];
    let mut qwen_inp = base_input(
        EngineKind::SdCpp,
        ModelMeta::Gguf(&g),
        5_000 * MIB,
        &hw,
        &base,
        &plain,
        &flags,
    );
    qwen_inp.components = &set_qwen;
    qwen_inp.device_hint = Some("CUDA0");
    absorb(&mut profiles, &qwen_inp, &TuningOverrides::default());

    // Full sdcpp tuning knob surface over the same component set.
    let knobs = Config {
        sdcpp_cache_mode: Some("easycache".into()),
        sdcpp_flash_attention: true,
        sdcpp_vae_tiling: true,
        sdcpp_rpc_servers: vec!["10.0.0.2:50052".into(), "10.0.0.3:50052".into()],
        sdcpp_sage_attn: true,
        sdcpp_cache_option: Some("threshold=0.25,reset=0".into()),
        sdcpp_max_vram: Some("cuda0=8".into()),
        sdcpp_params_backend: Some("diffusion=disk,clip=cpu".into()),
        sdcpp_split_mode: Some("row".into()),
        sdcpp_tae: Some("/models/tae.gguf".into()),
        sdcpp_conditioning_cache_size: Some(8),
        sdcpp_model_args: Some("qwen_image_2_1_prefix_cache=true".into()),
        sdcpp_tensor_type_rules: Some("model.=q6_k".into()),
        sdcpp_control_net: Some("/models/controlnet.safetensors".into()),
        sdcpp_ip_adapter: Some("/models/ip-adapter.safetensors".into()),
        sdcpp_clip_vision: Some("/models/clip_vision.safetensors".into()),
        sdcpp_motion_module: Some("/models/motion-module.safetensors".into()),
        sdcpp_photo_maker: Some("/models/photomaker-v2.bin".into()),
        sdcpp_pulid_weights: Some("/models/pulid-flux.safetensors".into()),
        sdcpp_upscale_model: Some("/models/realesrgan-x4.pth".into()),
        sdcpp_lora_model_dir: Some("/models/loras".into()),
        sdcpp_hires_upscalers_dir: Some("/models/upscalers".into()),
        sdcpp_embd_dir: Some("/models/embeddings".into()),
        sdcpp_audio_encoder: Some("/models/wav2vec2.bin".into()),
        sdcpp_high_noise_diffusion_model: Some("/models/hn-dit.safetensors".into()),
        sdcpp_uncond_diffusion_model: Some("/models/uncond-dit.safetensors".into()),
        sdcpp_backend: Some("clip=cpu,vae=cuda0".into()),
        sdcpp_qwen_prefix_cache_type: Some("q8_0".into()),
        ..Config::default()
    };
    let mut knob_inp = base_input(
        EngineKind::SdCpp,
        ModelMeta::Gguf(&g),
        5_000 * MIB,
        &hw,
        &knobs,
        &plain,
        &flags,
    );
    knob_inp.components = &set_qwen;
    knob_inp.device_hint = Some("CUDA0");
    absorb(&mut profiles, &knob_inp, &TuningOverrides::default());

    // FLUX dialect (--t5xxl + --clip_l, no --llm) and the tight-VRAM
    // stream posture (--offload-to-cpu).
    let set_flux = [
        ComponentArg::new("--vae", vae.to_str().unwrap_or("/vae")),
        ComponentArg::new("--t5xxl", t5.to_str().unwrap_or("/t5")),
        ComponentArg::new("--clip_l", clip.to_str().unwrap_or("/clip")),
    ];
    for hw_used in [&hw, &hw_tight] {
        let mut flux_inp = base_input(
            EngineKind::SdCpp,
            ModelMeta::Gguf(&g),
            5_000 * MIB,
            hw_used,
            &base,
            &plain,
            &flags,
        );
        flux_inp.components = &set_flux;
        absorb(&mut profiles, &flux_inp, &TuningOverrides::default());
    }

    scan(); // remove fixture dir
    profiles
}

/// RAII-free deferred cleanup: returns a closure that removes the fixture
/// directory; call it when the lane fn is done (panic-safe enough for a
/// registry builder — nextest isolates panics per process).
fn defer_remove(dir: &std::path::Path) -> impl FnOnce() + '_ {
    let dir = dir.to_path_buf();
    move || {
        std::fs::remove_dir_all(&dir).ok();
    }
}

fn mlx_lane() -> Vec<String> {
    lane_tokens(&mlx_profiles())
}

fn mlx_profiles() -> Vec<Profile> {
    let hf = hf_meta();
    let hw = gpu_hw(12_000, 32_000, 8);
    let flags = supported(MLX_SUPPORTED);
    let base = Config::default();
    let plain = ModelOverride::default();
    let mut profiles = Vec::new();

    // Baseline (transport pins + geometry ceiling only).
    absorb(
        &mut profiles,
        &base_input(
            EngineKind::Mlx,
            ModelMeta::Hf(&hf),
            5_000 * MIB,
            &hw,
            &base,
            &plain,
            &flags,
        ),
        &TuningOverrides::default(),
    );

    // Full MlxTuning knob surface, template-path variant (chat_template
    // XOR use_default_chat_template are distinct postures).
    let knobs = Config {
        mlx: MlxTuning {
            draft_model: Some("/models/draft.d".into()),
            num_draft_tokens: Some(4),
            kv_bits: Some(4),
            kv_group_size: Some(64),
            quantized_kv_start: Some(8),
            prefill_step_size: Some(512),
            prompt_cache_size: Some(16),
            prompt_cache_bytes: Some(1_048_576),
            decode_concurrency: Some(4),
            prompt_concurrency: Some(4),
            chat_template: Some("/models/template.jinja".into()),
            trust_remote_code: Some(true),
            adapter_path: Some("/models/adapter".into()),
            ..MlxTuning::default()
        },
        ..Config::default()
    };
    absorb(
        &mut profiles,
        &base_input(
            EngineKind::Mlx,
            ModelMeta::Hf(&hf),
            5_000 * MIB,
            &hw,
            &knobs,
            &plain,
            &flags,
        ),
        &TuningOverrides::default(),
    );

    // Default-template posture.
    let default_tpl = Config {
        mlx: MlxTuning {
            use_default_chat_template: Some(true),
            chat_template_args: Some("{}".into()),
            ..MlxTuning::default()
        },
        ..Config::default()
    };
    absorb(
        &mut profiles,
        &base_input(
            EngineKind::Mlx,
            ModelMeta::Hf(&hf),
            5_000 * MIB,
            &hw,
            &default_tpl,
            &plain,
            &flags,
        ),
        &TuningOverrides::default(),
    );

    profiles
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    #[test]
    fn unit__knob_registry__every_lane_emits_flags() {
        let lanes = lane_flags();
        assert_eq!(lanes.len(), 5, "five compile lanes");
        for (lane, flags) in &lanes {
            assert!(
                !flags.is_empty(),
                "{lane} registry came back empty — fixture matrix broke"
            );
            // Tokens are flags, sorted, unique by construction (BTreeSet).
            assert!(flags.iter().all(|f| is_flag_token(f)));
        }
    }

    #[test]
    fn unit__knob_registry__mlx_first_class_knobs_present() {
        // The stale-auditor bug this module exists to kill: MLX HAS a
        // first-class knob surface. Pin the headline members.
        let lanes = lane_flags();
        let mlx = &lanes["mlx"];
        for expected in [
            "--kv-bits",
            "--kv-group-size",
            "--prompt-cache-size",
            "--decode-concurrency",
            "--draft-model",
            "--adapter-path",
        ] {
            assert!(
                mlx.contains(&expected.to_string()),
                "mlx registry missing {expected}: {mlx:?}"
            );
        }
    }
}
