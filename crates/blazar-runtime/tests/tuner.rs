//! Tuner integration: run `tune_search` against the REAL stub-llama-bench
//! binary and assert the measured argmax is adopted and persisted.

use std::collections::BTreeSet;
use std::path::PathBuf;

use blazar_core::gguf::GgufMeta;
use blazar_core::hardware::{GpuInfo, Hardware};
use blazar_core::profile::{Endpoint, ProfileInput};
use blazar_core::store::Store;
use blazar_core::{BlazarDirs, Config, ModelOverride};
use blazar_runtime::Tuner;

fn bench_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_stub-llama-bench"))
}

fn dirs_setup() -> (tempfile::TempDir, BlazarDirs) {
    let tmp = tempfile::tempdir().unwrap();
    let dirs = BlazarDirs {
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
    basename: None,
    size_label: None,
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
    model_name: &'a str,
    model_path: &'a str,
    hw: &'a Hardware,
    cfg: &'a Config,
    flags: &'a BTreeSet<String>,
    overlay: &'a ModelOverride,
) -> ProfileInput<'a> {
    ProfileInput {
        engine_kind: blazar_core::engine_kind::EngineKind::default(),
        sibling_devices: Vec::new(),
        auto_tensor_split: None,
        auto_tp_size: None,
        mmproj_path: None,
        components: &[],
        model_name,
        instance_key: model_name,
        model_path,
        model_bytes: 5_000 * 1024 * 1024,
        meta: blazar_core::ModelMeta::Gguf(&META),
        hardware: hw,
        config: cfg,
        overlay,
        spec_mode: overlay.spec.as_deref().unwrap_or(cfg.spec.as_str()),
        loras: &[],
        draft_path: None,
        draft_gguf: None,
        engine_tag: "b-stub",
        supported_flags: flags,
        spec_types: &[],
        cache_types: &[],
        endpoint: Endpoint::Tcp {
            host: "127.0.0.1".into(),
            port: 1,
        },
        data_dir: "/tmp/blazar-test-data",
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

// the adoption chain is asserted end-to-end in one test so the
// receipt sequence stays visible
#[allow(clippy::too_many_lines)]
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
    // "qwen3.5-9b" carries measured KV-quality receipts (q8_0 and q4_0
    // at ctx 32768) in the compiled-in registry, so the quality gate
    // lets the stub's q4_0 winner through - this test pins the winner
    // contract for an evidenced model. The gate-blocking case has its
    // own test below with an unevidenced name.
    let inp = test_input("qwen3.5-9b", &model_str, &hw, &cfg, &flags, &overlay);

    // Grid axes: threads x kv-quant (llama-bench b10816 has no -c axis).
    // GPU grid: 4-grade KV ladder, single threads/batch. Stub scoring:
    // +threads, grade(k)+grade(v) with q4_0 = +30 per side -> winner
    // (threads 8, q4_0 pair, fa on, batch 2048).
    let tuner = Tuner {
        dirs: &dirs,
        bench_bin: bench_bin(),
    };
    let store = Store::open(&dirs).unwrap();
    let (profile, winning, rows) = tuner.tune_search(&store, &inp).unwrap();

    assert_ne!(rows, [] as [blazar_runtime::BenchRow; 0]);
    assert!(
        rows.iter()
            .any(|r| r.type_k.as_deref() == Some("q5_0") && r.type_v.as_deref() == Some("q5_0")),
        "GPU grid must measure the widened KV ladder (got grades: {:?})",
        rows.iter()
            .map(|r| r.type_k.clone())
            .collect::<std::collections::BTreeSet<_>>()
    );
    assert_eq!(
        winning
            .kv_pair
            .as_ref()
            .map(|(k, v)| (k.as_str(), v.as_str())),
        Some(("q4_0", "q4_0")),
        "stub rewards deeper KV grades (q4_0 = +30/side)"
    );
    assert!(!winning.kv_quant.unwrap(), "legacy bool mirrors the pair");
    assert_eq!(winning.threads.unwrap(), 8);
    assert_eq!(winning.fa, Some(true), "stub rewards fa-on by +15 t/s");
    assert_eq!(
        winning.batch,
        Some(2048),
        "GPU grid pins the single batch leg"
    );
    assert!(winning.ctx.is_none(), "ctx is not a bench axis");
    assert!(
        profile
            .argv
            .windows(2)
            .any(|w| w[0] == "--cache-type-k" && w[1] == "q4_0")
    );
    assert!(
        profile
            .argv
            .windows(2)
            .any(|w| w[0] == "--cache-type-v" && w[1] == "q4_0")
    );
    assert!(
        profile
            .argv
            .windows(2)
            .any(|w| w[0] == "--threads" && w[1] == "8")
    );
    assert!(
        profile
            .argv
            .windows(2)
            .any(|w| w[0] == "--flash-attn" && w[1] == "on")
    );
    assert!(
        profile
            .argv
            .windows(2)
            .any(|w| w[0] == "-b" && w[1] == "2048")
    );

    let stored = store.get_profile("qwen3.5-9b", "b-stub").unwrap().unwrap();
    let argv: Vec<String> = serde_json::from_str(&stored.args_json).unwrap();
    assert_eq!(argv, profile.argv);
    let payload: blazar_runtime::bench::BenchmarkPayload =
        serde_json::from_str(stored.benchmark_json.as_deref().unwrap()).unwrap();
    assert!(payload.score > 0.0);
    assert_ne!(stored.args_hash, "");

    // Optimization-ledger wiring: the adoption must leave an immutable
    // experiment, a measurement row, an adoption event, and the
    // active-profile pointer - the trust base later --plan/--run
    // decisions will be judged against.
    let ledger: (i64, i64, i64) = store
        .conn()
        .query_row(
            "SELECT
                (SELECT COUNT(*) FROM optimization_experiments
                  WHERE model = 'qwen3.5-9b' AND engine_tag = 'b-stub'),
                (SELECT COUNT(*) FROM optimization_measurements m
                  JOIN optimization_experiments e ON e.id = m.experiment_id
                  WHERE e.model = 'qwen3.5-9b' AND m.kind = 'llama-bench-grid'),
                (SELECT COUNT(*) FROM optimization_events v
                  JOIN optimization_experiments e ON e.id = v.experiment_id
                  WHERE e.model = 'qwen3.5-9b' AND v.kind = 'adopted')",
            [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .unwrap();
    assert_eq!(
        ledger,
        (1, 1, 1),
        "experiment + measurement + adopted event"
    );
    let pointer: i64 = store
        .conn()
        .query_row(
            "SELECT experiment_id FROM active_profiles
             WHERE model = 'qwen3.5-9b' AND engine_tag = 'b-stub'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert!(pointer > 0, "active_profiles points at the adoption");
}

#[test]
#[allow(non_snake_case)]
fn integration__tune_search_screen__measures_without_persisting() {
    let (tmp, dirs) = dirs_setup();
    let model_path = tmp.path().join("m.gguf");
    std::fs::write(&model_path, b"x").unwrap();
    let model_str = model_path.to_str().unwrap().to_string();

    let cfg = Config::default();
    let hw = gpu_hw(8, 12_000, 32_000);
    let flags = all_flags();
    let overlay = ModelOverride::default();
    let inp = test_input("qwen3.5-9b", &model_str, &hw, &cfg, &flags, &overlay);

    let tuner = Tuner {
        dirs: &dirs,
        bench_bin: bench_bin(),
    };
    let store = Store::open(&dirs).unwrap();

    // Screen pass: identical winner contract, zero persistence - this
    // is the seam `optimize --run` rides to measure candidates without
    // touching the serving profile.
    let (screen_profile, screen_winning, screen_rows, screen_score) =
        tuner.tune_search_screen(&inp).unwrap();
    assert!(!screen_rows.is_empty());
    assert!(
        screen_score > 0.0,
        "screen returns the grid score for the ledger measurement"
    );
    assert_eq!(
        screen_winning
            .kv_pair
            .as_ref()
            .map(|(k, v)| (k.as_str(), v.as_str())),
        Some(("q4_0", "q4_0")),
        "screen picks the same evidenced winner as the adopting path"
    );
    assert!(
        store.get_profile("qwen3.5-9b", "b-stub").unwrap().is_none(),
        "screen must not write a profile row"
    );
    let persisted: i64 = store
        .conn()
        .query_row("SELECT COUNT(*) FROM optimization_experiments", [], |r| {
            r.get(0)
        })
        .unwrap();
    assert_eq!(persisted, 0, "screen must not open a ledger experiment");

    // Wrapper parity: same search again through the persisting path.
    let (profile, winning, _rows) = tuner.tune_search(&store, &inp).unwrap();
    assert_eq!(profile.argv, screen_profile.argv, "same compiled profile");
    assert_eq!(winning.kv_pair, screen_winning.kv_pair, "same overrides");
    assert!(
        store.get_profile("qwen3.5-9b", "b-stub").unwrap().is_some(),
        "wrapper persists the profile row"
    );
}

#[test]
#[allow(non_snake_case)]
fn integration__tune_search_cpu__keeps_legacy_grid_and_q8_winner() {
    let (tmp, dirs) = dirs_setup();
    let model_path = tmp.path().join("m.gguf");
    std::fs::write(&model_path, b"x").unwrap();
    let model_str = model_path.to_str().unwrap().to_string();

    let cfg = Config::default();
    // No GPU: the CPU lane keeps the legacy grid (threads 2-leg, kv
    // f16/q8_0, fa 2-leg, batch 2-leg) and its winner contract.
    let hw = Hardware {
        physical_cores: 8,
        total_ram_mib: 32_000,
        gpus: vec![],
    };
    let flags = all_flags();
    let overlay = ModelOverride::default();
    // Evidenced model: q8_0 has a measured receipt at ctx 32768, so the
    // legacy q8 winner stays eligible under the quality gate.
    let inp = test_input("qwen3.5-9b", &model_str, &hw, &cfg, &flags, &overlay);

    let tuner = Tuner {
        dirs: &dirs,
        bench_bin: bench_bin(),
    };
    let store = Store::open(&dirs).unwrap();
    let (profile, winning, rows) = tuner.tune_search(&store, &inp).unwrap();

    assert!(
        rows.iter()
            .all(|r| r.type_k.as_deref() != Some("q5_0") && r.type_k.as_deref() != Some("q4_0")),
        "CPU grid must not measure the GPU-only deeper grades"
    );
    assert!(winning.kv_quant.unwrap(), "stub rewards q8_0 by +50 t/s");
    assert_eq!(
        winning
            .kv_pair
            .as_ref()
            .map(|(k, v)| (k.as_str(), v.as_str())),
        Some(("q8_0", "q8_0"))
    );
    assert_eq!(winning.threads.unwrap(), 8);
    assert_eq!(winning.fa, Some(true), "stub rewards fa-on by +15 t/s");
    assert_eq!(
        winning.batch,
        Some(1024),
        "stub rewards batch 1024 by +8 t/s"
    );
    assert!(
        profile
            .argv
            .windows(2)
            .any(|w| w[0] == "--cache-type-k" && w[1] == "q8_0")
    );
    assert!(
        profile
            .argv
            .windows(2)
            .any(|w| w[0] == "--threads" && w[1] == "8")
    );
}

/// The quality gate's blocking side: an unevidenced model must never
/// adopt a quantized KV grade on throughput alone, no matter how fast
/// the stub says it is. The grid still MEASURES the deeper grades (the
/// search is not shrunk - evidence, not exploration, is gated); only
/// the adoption argmax is restricted to f16.
#[test]
#[allow(non_snake_case)]
fn integration__tune_search_gpu__kv_quality_gate_blocks_unevidenced_grades() {
    let (tmp, dirs) = dirs_setup();
    let model_path = tmp.path().join("m.gguf");
    std::fs::write(&model_path, b"x").unwrap();
    let model_str = model_path.to_str().unwrap().to_string();

    let cfg = Config::default(); // default_ctx 16384
    let hw = gpu_hw(8, 12_000, 32_000);
    let flags = all_flags();
    let overlay = ModelOverride::default();
    // "qwen3-8b" has NO entry in the kv-quality registry - every
    // quantized grade is ineligible at any ctx, so the stub's q4_0
    // +30/side bait must lose to the f16 baseline.
    let inp = test_input("qwen3-8b", &model_str, &hw, &cfg, &flags, &overlay);

    let tuner = Tuner {
        dirs: &dirs,
        bench_bin: bench_bin(),
    };
    let store = Store::open(&dirs).unwrap();
    let (profile, winning, rows) = tuner.tune_search(&store, &inp).unwrap();

    // Grid unchanged: the deeper grades are still measured.
    assert!(
        rows.iter().any(|r| r.type_k.as_deref() == Some("q4_0")),
        "the gate filters adoption, not exploration - q4_0 rows must exist"
    );
    // Adoption restricted: f16/f16 is the only eligible pair for an
    // unevidenced model.
    assert_eq!(
        winning
            .kv_pair
            .as_ref()
            .map(|(k, v)| (k.as_str(), v.as_str())),
        Some(("f16", "f16")),
        "unevidenced quantized grades must not win the argmax"
    );
    assert!(!winning.kv_quant.unwrap());
    assert!(
        !profile
            .argv
            .windows(2)
            .any(|w| w[0] == "--cache-type-k" && w[1] != "f16"),
        "compiled argv must not carry a quantized K cache"
    );
}
