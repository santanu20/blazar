//! Supervisor integration tests against the REAL stub-llama-server:
#![allow(unsafe_code)] // audited libc::kill(pid, SIG) on exact child pids
//! ensure/health/argv, cold-start timeout, ladder timing, capacity+evict,
//! crash respawn+circuit, shutdown teardown, concurrent models.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use pallama_core::hardware::{GpuInfo, Hardware};
use pallama_core::store::Store;
use pallama_core::{Config, PallamaDirs};
use pallama_runtime::EventBus;
use pallama_runtime::engine::manifest::{probe as probe_manifest, Manifest};
use pallama_runtime::{LlamaCppEngine, SupervisionError, Supervisor};

fn stub_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_stub-llama-server"))
}

/// Manifest of the stub (real probe): includes --jinja, --sleep-idle-seconds...
fn stub_manifest() -> Manifest {
    probe_manifest(&stub_bin(), "stub").expect("probe stub")
}

fn setup(models: &[(&str, u64)]) -> (tempfile::TempDir, PallamaDirs) {
    let tmp = tempfile::tempdir().unwrap();
    let dirs = PallamaDirs {
        config_dir: tmp.path().join("cfg"),
        data_dir: tmp.path().join("data"),
    };
    dirs.ensure().unwrap();
    // Write real (tiny) GGUF files so read_metadata_file works.
    for (name, bytes) in models {
        let p = dirs.models_dir().join(format!("{name}-q4_k_m.gguf"));
        write_gguf(&p);
        let store = Store::open(&dirs).unwrap();
        store
            .upsert_model(&pallama_core::ModelRow {
                name: (*name).into(),
                repo: format!("o/{name}"),
                quant: "Q4_K_M".into(),
                path: p.display().to_string(),
                bytes: i64::try_from(*bytes).unwrap_or(i64::MAX),
                sha256: None,
                mmproj_path: None,
                shards: 1,
                arch: Some("qwen3".into()),
                params: None,
                ctx_train: Some(40_960),
                pulled_at: 1,
            })
            .unwrap();
    }
    (tmp, dirs)
}

/// Minimal valid GGUF (qwen3-ish metadata) so the profile compiler has
/// real numbers.
fn write_gguf(path: &std::path::Path) {
    let mut b: Vec<u8> = Vec::new();
    b.extend_from_slice(b"GGUF");
    b.extend_from_slice(&3u32.to_le_bytes());
    b.extend_from_slice(&0u64.to_le_bytes());
    let kvs: Vec<(&str, u8, Vec<u8>)> = vec![
        ("general.architecture", 8, payload_str("qwen3")),
        ("qwen3.block_count", 4, 28u32.to_le_bytes().to_vec()),
        ("qwen3.context_length", 4, 40_960u32.to_le_bytes().to_vec()),
        ("qwen3.head_count", 4, 16u32.to_le_bytes().to_vec()),
        ("qwen3.head_count_kv", 4, 8u32.to_le_bytes().to_vec()),
        ("qwen3.embedding_length", 4, 1024u32.to_le_bytes().to_vec()),
    ];
    b.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
    for (k, t, v) in kvs {
        b.extend_from_slice(&(k.len() as u64).to_le_bytes());
        b.extend_from_slice(k.as_bytes());
        b.extend_from_slice(&u32::from(t).to_le_bytes());
        b.extend_from_slice(&v);
    }
    std::fs::write(path, b).unwrap();
}

fn payload_str(s: &str) -> Vec<u8> {
    let mut v = (s.len() as u64).to_le_bytes().to_vec();
    v.extend_from_slice(s.as_bytes());
    v
}

fn supervisor(dirs: &PallamaDirs, config: Config, gpu: bool) -> Arc<Supervisor> {
    let mut engine = LlamaCppEngine::new(stub_manifest());
    engine.child_env = vec![(
        "STUB_ARGV_FILE".into(),
        dirs.run_dir().join("argv.json").display().to_string(),
    )];
    let hw = if gpu {
        Hardware {
            physical_cores: 4,
            total_ram_mib: 16_000,
            gpus: vec![GpuInfo {
                name: "stub-gpu".into(),
                description: "STUB GPU".into(),
                total_mib: 24_000,
                free_mib: 24_000,
            }],
        }
    } else {
        Hardware { physical_cores: 4, total_ram_mib: 16_000, gpus: vec![] }
    };
    let mut s = Supervisor::new(
        dirs.clone(),
        config,
        EventBus::default(),
        hw,
        Arc::new(engine),
    );
    // Fast test timings.
    s.load_timeout = Duration::from_secs(4);
    s.shutdown_grace = Duration::from_secs(2);
    s.circuit_window = Duration::from_secs(4);
    s.max_restarts = 2;
    Arc::new(s)
}

fn base_config() -> Config {
    Config::default()
}

#[tokio::test]
#[allow(non_snake_case)]
async fn integration__ensure_ready__health_and_argv_flags() {
    let (_t, dirs) = setup(&[("m1", 500)]);
    let sup = supervisor(&dirs, base_config(), true);
    let ep = sup.ensure("m1").await.expect("ensure");
    // Endpoint answers health.
    let url = match &ep.endpoint {
        pallama_core::Endpoint::Tcp { host, port } => format!("http://{host}:{port}"),
        pallama_core::Endpoint::Unix { .. } => unreachable!(),
    };
    let body: serde_json::Value = reqwest::Client::new()
        .get(format!("{url}/health"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(body["status"], "ok");

    // argv file records the compiled profile flags.
    tokio::time::sleep(Duration::from_millis(300)).await;
    let argv: Vec<String> =
        serde_json::from_str(&std::fs::read_to_string(dirs.run_dir().join("argv.json")).unwrap())
            .unwrap();
    for flag in ["--jinja", "--metrics", "--flash-attn", "--cache-reuse", "--sleep-idle-seconds", "--cache-ram", "-np"] {
        assert!(argv.contains(&flag.to_string()), "missing {flag} in {argv:?}");
    }
    assert!(argv.windows(2).any(|w| w[0] == "--alias" && w[1] == "m1"));
    assert!(argv.windows(2).any(|w| w[0] == "--ctx-size" && w[1] == "16384"));

    // ps shows ready with ctx.
    let ps = sup.ps();
    assert_eq!(ps.len(), 1);
    assert_eq!(ps[0].state, "ready");
    assert_eq!(ps[0].ctx, 16384);

    sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn integration__ensure_unknown_model__named_error() {
    let (_t, dirs) = setup(&[]);
    let sup = supervisor(&dirs, base_config(), false);
    let err = sup.ensure("nope").await.unwrap_err();
    assert!(err.to_string().contains("no such model"), "{err}");
}

#[tokio::test]
#[allow(non_snake_case)]
async fn integration__capacity_one__second_ensure_evicts_idle_first() {
    let (_t, dirs) = setup(&[("m1", 500), ("m2", 500)]);
    let mut cfg = base_config();
    cfg.max_loaded_models = 1;
    let sup = supervisor(&dirs, cfg, false);
    sup.ensure("m1").await.unwrap();
    sup.ensure("m2").await.unwrap();
    let ps = sup.ps();
    assert_eq!(ps.len(), 1, "capacity 1 enforced");
    assert_eq!(ps[0].name, "m2", "idle m1 evicted for m2");
    // First model respawns on demand.
    let ep = sup.ensure("m1").await.unwrap();
    assert_ne!(ep.endpoint, pallama_core::Endpoint::Tcp { host: "127.0.0.1".into(), port: 0 });
    sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn integration__ladder__sleep_then_evict() {
    let (_t, dirs) = setup(&[("m1", 500)]);
    let mut cfg = base_config();
    cfg.idle_sleep_secs = 1;
    cfg.idle_timeout_secs = 3;
    let mut s = Supervisor::new(
        dirs.clone(),
        cfg,
        EventBus::default(),
        Hardware {
            physical_cores: 4,
            total_ram_mib: 16_000,
            gpus: vec![GpuInfo { name: "g".into(), description: "S".into(), total_mib: 24_000, free_mib: 24_000 }],
        },
        Arc::new(LlamaCppEngine::new(stub_manifest())),
    );
    s.reaper_interval = Duration::from_millis(200);
    s.load_timeout = Duration::from_secs(4);
    s.shutdown_grace = Duration::from_secs(2);
    let sup = Arc::new(s);
    let _reaper = sup.spawn_reaper();

    sup.ensure("m1").await.unwrap();
    assert_eq!(sup.ps()[0].state, "ready");
    // Sleep after ~1s idle.
    tokio::time::sleep(Duration::from_millis(1500)).await;
    let ps = sup.ps();
    assert_eq!(ps[0].state, "sleeping", "ladder: Ready -> Sleeping");
    // Evicted after 3s idle: child gone, map empty.
    let deadline = Instant::now() + Duration::from_secs(6);
    while Instant::now() < deadline {
        if sup.ps().is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    assert!(sup.ps().is_empty(), "ladder: Sleeping -> Evicted");
    assert!(!dirs.run_dir().join("m1.pid").exists(), "pid marker removed");
}

#[tokio::test]
#[allow(non_snake_case)]
async fn integration__crash__respawn_and_circuit() {
    let (_t, dirs) = setup(&[("m1", 500)]);
    let sup = supervisor(&dirs, base_config(), false);
    sup.ensure("m1").await.unwrap();
    let pid = sup.ps()[0].pid;
    // Kill -9 the child by exact pid (simulates an engine crash).
    unsafe { libc::kill(i32::try_from(pid).unwrap_or(-1), libc::SIGKILL) };
    tokio::time::sleep(Duration::from_millis(200)).await;
    // reap_dead_children (also invoked by the reaper in prod) clears it.
    sup.reap_dead_children().await;
    assert!(sup.ps().is_empty(), "crashed instance dropped");
    // Respawn on next ensure.
    sup.ensure("m1").await.unwrap();
    let pid2 = sup.ps()[0].pid;
    assert_ne!(pid, pid2, "respawned a new process");

    // Hammer crashes: circuit opens after max_restarts within window.
    for _ in 0..4 {
        let row = sup.ps().into_iter().next().expect("instance before crash");
        let pid = row.pid;
        unsafe { libc::kill(i32::try_from(pid).unwrap_or(-1), libc::SIGKILL) };
        tokio::time::sleep(Duration::from_millis(150)).await;
        sup.reap_dead_children().await;
        match sup.ensure("m1").await {
            Ok(_) => {}
            Err(e) => {
                assert!(matches!(e, SupervisionError::CircuitOpen(_)), "{e}");
                // Reset re-opens the path.
                sup.reset_circuit(Some("m1"));
                sup.ensure("m1").await.expect("reset reopens circuit");
                sup.shutdown_all().await.unwrap();
                return;
            }
        }
    }
    panic!("circuit never opened after repeated crashes");
}

#[tokio::test]
#[allow(non_snake_case)]
async fn integration__shutdown_kills_children__idempotent() {
    let (_t, dirs) = setup(&[("m1", 500), ("m2", 500)]);
    let mut cfg = base_config();
    cfg.max_loaded_models = 2;
    let sup = supervisor(&dirs, cfg, false);
    sup.ensure("m1").await.unwrap();
    sup.ensure("m2").await.unwrap();
    let pids: Vec<u32> = sup.ps().iter().map(|p| p.pid).collect();
    sup.shutdown_all().await.unwrap();
    sup.shutdown_all().await.unwrap(); // idempotent
    tokio::time::sleep(Duration::from_millis(300)).await;
    for pid in pids {
        // Existence check via signal 0 to the exact pid only.
        let alive = unsafe { libc::kill(i32::try_from(pid).unwrap_or(-1), 0) } == 0;
        assert!(!alive, "child {pid} still alive after shutdown");
    }
    assert!(sup.ps().is_empty());
}

#[tokio::test]
#[allow(non_snake_case)]
async fn integration__concurrent_models_both_ready() {
    let (_t, dirs) = setup(&[("m1", 500), ("m2", 500)]);
    let mut cfg = base_config();
    cfg.max_loaded_models = 2;
    let sup = supervisor(&dirs, cfg, false);
    let (a, b) = tokio::join!(sup.ensure("m1"), sup.ensure("m2"));
    a.expect("m1");
    b.expect("m2");
    assert_eq!(sup.ps().len(), 2);
    sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn integration__load_timeout__never_healthy_stub() {
    let (_t, dirs) = setup(&[("m1", 500)]);
    let mut engine = LlamaCppEngine::new(stub_manifest());
    engine.child_env = vec![("STUB_HEALTH_NEVER".into(), "1".into())];
    let s = Supervisor::new(
        dirs.clone(),
        base_config(),
        EventBus::default(),
        Hardware { physical_cores: 4, total_ram_mib: 16_000, gpus: vec![] },
        Arc::new(engine),
    );
    let mut s = s;
    s.load_timeout = Duration::from_secs(1);
    let sup = Arc::new(s);
    let err = sup.ensure("m1").await.unwrap_err();
    assert!(
        matches!(err, SupervisionError::ModelLoadTimeout(_)),
        "expected ModelLoadTimeout, got {err}"
    );
    assert!(sup.ps().is_empty());
    // No zombie stub left bound to a port.
    sup.shutdown_all().await.unwrap();
}

#[test]
#[allow(non_snake_case)]
fn unit__capacity_auto__cpu_one_gpu_vram_ratio() {
    let (_t, dirs) = setup(&[("m", 500)]);
    let s = Supervisor::new(
        dirs,
        base_config(),
        EventBus::default(),
        Hardware { physical_cores: 4, total_ram_mib: 16_000, gpus: vec![] },
        Arc::new(LlamaCppEngine::new(stub_manifest())),
    );
    assert_eq!(s.capacity_for(1_000_000_000), 1, "CPU-only -> 1");
    let gpu_hw = Hardware {
        physical_cores: 4,
        total_ram_mib: 16_000,
        gpus: vec![GpuInfo { name: "g".into(), description: "S".into(), total_mib: 24_576, free_mib: 24_576 }],
    };
    let s2 = Supervisor::new(
        PallamaDirs { config_dir: std::path::PathBuf::from("/tmp/p-cfg"), data_dir: std::path::PathBuf::from("/tmp/p-data") },
        base_config(),
        EventBus::default(),
        gpu_hw,
        Arc::new(LlamaCppEngine::new(stub_manifest())),
    );
    // 24 GiB VRAM (24576 MiB), 4 GiB model -> floor(6) -> 6.
    assert_eq!(s2.capacity_for(4 * 1024 * 1024 * 1024), 6);
    // 23.4 GiB VRAM would floor to 5 — auto capacity is conservative.
    let hw_234 = Hardware {
        physical_cores: 4,
        total_ram_mib: 16_000,
        gpus: vec![GpuInfo { name: "g".into(), description: "S".into(), total_mib: 24_000, free_mib: 24_000 }],
    };
    let s3 = Supervisor::new(
        PallamaDirs { config_dir: std::path::PathBuf::from("/tmp/p-cfg"), data_dir: std::path::PathBuf::from("/tmp/p-data") },
        base_config(),
        EventBus::default(),
        hw_234,
        Arc::new(LlamaCppEngine::new(stub_manifest())),
    );
    assert_eq!(s3.capacity_for(4 * 1024 * 1024 * 1024), 5);
}
