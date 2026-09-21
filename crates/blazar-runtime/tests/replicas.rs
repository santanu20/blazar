//! B1 replica integration tests against the REAL stub-llama-server:
//! multi-instance spawn, prefix-affinity stickiness, legacy single-key
//! behavior, model-level evict, per-replica pidfiles.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use blazar_core::hardware::{GpuInfo, Hardware};
use blazar_core::store::Store;
use blazar_core::{BlazarDirs, Config, ModelOverride};
use blazar_runtime::engine::manifest::{probe as probe_manifest, Manifest};
use blazar_runtime::supervisor::PrefixKey;
use blazar_runtime::{EventBus, LlamaCppEngine, Supervisor};

fn stub_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_stub-llama-server"))
}

fn stub_manifest() -> Manifest {
    probe_manifest(&stub_bin(), "stub").expect("probe stub")
}

fn setup(model: &str) -> (tempfile::TempDir, BlazarDirs) {
    let tmp = tempfile::tempdir().unwrap();
    let dirs = BlazarDirs {
        config_dir: tmp.path().join("cfg"),
        data_dir: tmp.path().join("data"),
    };
    dirs.ensure().unwrap();
    let p = dirs.models_dir().join(format!("{model}-q4_k_m.gguf"));
    write_gguf(&p);
    let store = Store::open(&dirs).unwrap();
    store
        .upsert_model(&blazar_core::ModelRow {
            name: model.into(),
            repo: format!("o/{model}"),
            quant: "Q4_K_M".into(),
            path: p.display().to_string(),
            bytes: 500,
            sha256: None,
            mmproj_path: None,
            shards: 1,
            arch: Some("qwen3".into()),
            params: None,
            ctx_train: Some(40_960),
            pulled_at: 1,
        })
        .unwrap();
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

fn supervisor(dirs: &BlazarDirs, replicas: Option<u32>) -> Arc<Supervisor> {
    let mut engine = LlamaCppEngine::new(stub_manifest());
    engine.child_env = vec![(
        "STUB_ARGV_FILE".into(),
        dirs.run_dir().join("argv.json").display().to_string(),
    )];
    let hw = Hardware {
        physical_cores: 4,
        total_ram_mib: 16_000,
        gpus: vec![GpuInfo {
            name: "stub-gpu".into(),
            description: "STUB GPU".into(),
            total_mib: 24_000,
            free_mib: 24_000,
        }],
    };
    let mut config = Config::default();
    if let Some(r) = replicas {
        config.model_overrides.insert(
            "m".into(),
            ModelOverride {
                replicas: Some(r),
                ..Default::default()
            },
        );
    }
    let mut s = Supervisor::new(
        dirs.clone(),
        config,
        EventBus::default(),
        hw,
        Arc::new(engine),
    );
    s.load_timeout_secs = Some(8);
    s.shutdown_grace = Duration::from_secs(2);
    Arc::new(s)
}

#[tokio::test]
#[allow(non_snake_case)]
async fn integration__replicas2_distinct_prefixes__two_children() {
    let (_t, dirs) = setup("m");
    let sup = supervisor(&dirs, Some(2));

    let a = sup
        .ensure_routed("m", Some(PrefixKey { sys: 1, convo: 1 }))
        .await
        .expect("replica 1");
    let b = sup
        .ensure_routed("m", Some(PrefixKey { sys: 2, convo: 2 }))
        .await
        .expect("replica 2");
    assert_eq!(a.name, "m#1", "first distinct prefix grows #1");
    assert_eq!(b.name, "m#2", "second distinct prefix grows #2");
    assert_ne!(a.endpoint, b.endpoint, "replicas own separate children");

    let ps = sup.ps();
    assert_eq!(ps.len(), 2, "two live replicas");
    let idx: std::collections::BTreeSet<u32> = ps.iter().filter_map(|r| r.replica).collect();
    assert_eq!(idx, [1, 2].into_iter().collect());
    assert!(
        ps.iter().all(|r| r.name == "m"),
        "ps names stay model-level"
    );

    // Per-replica pidfiles exist under the instance keys.
    assert!(dirs.run_dir().join("m#1.pid").exists());
    assert!(dirs.run_dir().join("m#2.pid").exists());

    sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn integration__same_prefix__sticky_replica() {
    let (_t, dirs) = setup("m");
    let sup = supervisor(&dirs, Some(2));

    let first = sup
        .ensure_routed("m", Some(PrefixKey { sys: 42, convo: 42 }))
        .await
        .expect("first");
    let second = sup
        .ensure_routed("m", Some(PrefixKey { sys: 42, convo: 42 }))
        .await
        .expect("second");
    assert_eq!(
        first.name, second.name,
        "same conversation prefix sticks to its warm replica"
    );
    assert_eq!(sup.ps().len(), 1, "sticky traffic never scales out");

    sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn integration__no_overlay__legacy_single_key() {
    let (_t, dirs) = setup("m");
    let sup = supervisor(&dirs, None);

    let ep = sup.ensure("m").await.expect("legacy ensure");
    assert_eq!(ep.name, "m", "byte-identical legacy key");
    let ps = sup.ps();
    assert_eq!(ps.len(), 1);
    assert_eq!(ps[0].replica, None, "plain models carry no replica idx");
    assert!(dirs.run_dir().join("m.pid").exists());

    sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn integration__evict_model__kills_all_replicas() {
    let (_t, dirs) = setup("m");
    let sup = supervisor(&dirs, Some(2));

    let a = sup
        .ensure_routed("m", Some(PrefixKey { sys: 1, convo: 1 }))
        .await
        .expect("replica 1");
    let b = sup
        .ensure_routed("m", Some(PrefixKey { sys: 2, convo: 2 }))
        .await
        .expect("replica 2");
    let pid_a = pid_of_replica(&sup, &a.name);
    let pid_b = pid_of_replica(&sup, &b.name);

    sup.evict_model("m").await.expect("evict model");
    assert!(sup.ps().is_empty(), "model-level evict reaps every replica");
    tokio::time::sleep(Duration::from_millis(300)).await;
    for pid in [pid_a, pid_b] {
        assert!(
            !blazar_runtime::process_alive_by_pid(pid),
            "replica child {pid} gone after evict_model"
        );
    }

    sup.shutdown_all().await.unwrap();
}

fn pid_of_replica(sup: &Supervisor, key: &str) -> u32 {
    sup.ps()
        .iter()
        .find(|r| r.replica.is_some() && format!("{}#{}", r.name, r.replica.unwrap()) == key)
        .map_or_else(|| panic!("no ps row for {key}"), |r| r.pid)
}
