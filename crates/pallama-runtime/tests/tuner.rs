//! Tuner integration: run `tune_search` against the REAL stub-llama-bench
//! binary and assert the measured argmax is adopted and persisted.

use std::collections::BTreeSet;
use std::path::PathBuf;

use pallama_core::gguf::GgufMeta;
use pallama_core::hardware::{GpuInfo, Hardware};
use pallama_core::profile::{Endpoint, ProfileInput};
use pallama_core::store::Store;
use pallama_core::{Config, ModelOverride, PallamaDirs};
use pallama_runtime::Tuner;

fn bench_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_stub-llama-bench"))
}

fn dirs_setup() -> (tempfile::TempDir, PallamaDirs) {
    let tmp = tempfile::tempdir().unwrap();
    let dirs = PallamaDirs {
        config_dir: tmp.path().join("cfg"),
        data_dir: tmp.path().join("data"),
    };
    dirs.ensure().unwrap();
    (tmp, dirs)
}

fn gpu_hw(cores: u32, vram_mib: u64, ram_mib: u64) -> Hardware {
    Hardware {
        physical_cores: cores,
        total_ram_mib: ram_mib,
        gpus: vec![GpuInfo {
            name: "g".into(),
            description: "CUDA".into(),
            total_mib: vram_mib,
            free_mib: vram_mib,
        }],
    }
}

static META: std::sync::LazyLock<GgufMeta> = std::sync::LazyLock::new(|| GgufMeta {
    architecture: "qwen3".into(),
    name: None,
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
    chat_template: Some("{%- if tools %}{{ tool_calls }}{%- endif %}".into()),
    mtp_layers: None,
    num_loops: None,
});

fn test_input<'a>(
    model_path: &'a str,
    hw: &'a Hardware,
    cfg: &'a Config,
    flags: &'a BTreeSet<String>,
    overlay: &'a ModelOverride,
) -> ProfileInput<'a> {
    ProfileInput {
        engine_kind: pallama_core::engine_kind::EngineKind::default(),
        sibling_devices: Vec::new(),
        auto_tensor_split: None,
        mmproj_path: None,
        model_name: "qwen3-8b",
        instance_key: "qwen3-8b",
        model_path,
        model_bytes: 5_000 * 1024 * 1024,
        meta: pallama_core::ModelMeta::Gguf(&META),
        hardware: hw,
        config: cfg,
        overlay,
        loras: &[],
        draft_path: None,
        draft_gguf: None,
        engine_tag: "b-stub",
        supported_flags: flags,
        spec_types: &[],
        endpoint: Endpoint::Tcp {
            host: "127.0.0.1".into(),
            port: 1,
        },
        data_dir: "/tmp/pallama-test-data",
        cache_hit_rate: None,
        resident_ram_mib: 0,
        device_hint: None,
        mmproj_force: false,
        engine_census: hw.gpus.clone(),
    }
}

fn all_flags() -> BTreeSet<String> {
    [
        "-m",
        "--host",
        "--port",
        "--alias",
        "--jinja",
        "--metrics",
        "--flash-attn",
        "--ctx-size",
        "--threads",
        "--gpu-layers",
        "--cache-reuse",
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
        "-p",
        "-n",
        "-r",
        "-c",
        "-t",
        "-ctk",
        "-ctv",
    ]
    .iter()
    .map(|f| (*f).into())
    .collect()
}

#[test]
#[allow(non_snake_case)]
fn integration__bench_default__parses_stub_rows() {
    let (tmp, dirs) = dirs_setup();
    let model = tmp.path().join("m.gguf");
    std::fs::write(&model, b"x").unwrap();
    let tuner = Tuner {
        dirs: &dirs,
        bench_bin: bench_bin(),
    };
    let rows = tuner.bench_default(&model).unwrap();
    assert!(rows.iter().any(|r| r.test == "tg128"), "{rows:?}");
    assert!(rows.iter().any(|r| r.test.starts_with("pp")));
}

#[test]
#[allow(non_snake_case)]
fn integration__tune_search__adopts_argmax_and_persists() {
    let (tmp, dirs) = dirs_setup();
    let model_path = tmp.path().join("m.gguf");
    std::fs::write(&model_path, b"x").unwrap();
    let model_str = model_path.to_str().unwrap().to_string();

    let cfg = Config::default(); // default_ctx 16384
    let hw = gpu_hw(8, 12_000, 32_000);
    let flags = all_flags();
    let overlay = ModelOverride::default();
    let inp = test_input(&model_str, &hw, &cfg, &flags, &overlay);

    // Grid axes: threads x kv-quant (llama-bench b10816 has no -c axis).
    // Stub scoring: +threads, +50 for q8_0 -> winner (threads 8, q8_0).
    let tuner = Tuner {
        dirs: &dirs,
        bench_bin: bench_bin(),
    };
    let store = Store::open(&dirs).unwrap();
    let (profile, winning, rows) = tuner.tune_search(&store, &inp).unwrap();

    assert!(!rows.is_empty());
    assert!(winning.kv_quant.unwrap(), "stub rewards q8_0 by +50 t/s");
    assert_eq!(winning.threads.unwrap(), 8);
    assert_eq!(winning.fa, Some(true), "stub rewards fa-on by +15 t/s");
    assert_eq!(
        winning.batch,
        Some(1024),
        "stub rewards batch 1024 by +8 t/s"
    );
    assert!(winning.ctx.is_none(), "ctx is not a bench axis");
    assert!(profile
        .argv
        .windows(2)
        .any(|w| w[0] == "--cache-type-k" && w[1] == "q8_0"));
    assert!(profile
        .argv
        .windows(2)
        .any(|w| w[0] == "--threads" && w[1] == "8"));
    assert!(profile
        .argv
        .windows(2)
        .any(|w| w[0] == "--flash-attn" && w[1] == "on"));
    assert!(profile
        .argv
        .windows(2)
        .any(|w| w[0] == "-b" && w[1] == "1024"));

    let stored = store.get_profile("qwen3-8b", "b-stub").unwrap().unwrap();
    let argv: Vec<String> = serde_json::from_str(&stored.args_json).unwrap();
    assert_eq!(argv, profile.argv);
    let payload: serde_json::Value =
        serde_json::from_str(stored.benchmark_json.as_deref().unwrap()).unwrap();
    assert!(payload["score"].as_f64().unwrap() > 0.0);
    assert!(!stored.args_hash.is_empty());
}
