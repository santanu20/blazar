//! Shared harness for gateway integration tests: a real axum server over
//! a real supervisor spawning stub-llama-server children (same pattern
//! as gateway.rs, factored for the compat suite).

#![allow(dead_code)]

use std::sync::Arc;
use std::time::Duration;

use pallama_core::{ApiKey, Config, PallamaDirs};
use pallama_gateway::router;
use pallama_gateway::state::AppState;
use pallama_runtime::engine_impl::LlamaCppEngine;
use pallama_runtime::{EventBus, Supervisor};

pub struct TestServer {
    pub base: String,
    _tmp: tempfile::TempDir,
    pub dirs: PallamaDirs,
    pub state: Arc<AppState>,
    _sup_reaper: tokio::task::JoinHandle<()>,
}

fn find_stub() -> std::path::PathBuf {
    let exe = std::env::current_exe().unwrap();
    for dir in exe.ancestors().skip(1) {
        let candidate = dir.join("stub-llama-server");
        if candidate.exists() {
            return candidate;
        }
    }
    panic!("stub-llama-server not found near {}", exe.display());
}

fn write_gguf(path: &std::path::Path) {
    let mut b: Vec<u8> = Vec::new();
    b.extend_from_slice(b"GGUF");
    b.extend_from_slice(&3u32.to_le_bytes());
    b.extend_from_slice(&0u64.to_le_bytes());
    let kvs: Vec<(&str, u8, Vec<u8>)> = vec![
        ("general.architecture", 8, pstr("qwen3")),
        ("qwen3.block_count", 4, 28u32.to_le_bytes().to_vec()),
        ("qwen3.context_length", 4, 40_960u32.to_le_bytes().to_vec()),
        ("qwen3.head_count", 4, 16u32.to_le_bytes().to_vec()),
        ("qwen3.head_count_kv", 4, 8u32.to_le_bytes().to_vec()),
        ("qwen3.embedding_length", 4, 1024u32.to_le_bytes().to_vec()),
        // ChatML template with a thinking marker: exercises the
        // evidence-based `thinking` capability in /api/show.
        (
            "tokenizer.chat_template",
            8,
            pstr("{% if enable_thinking %}{{ content }}{% endif %}"),
        ),
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

fn pstr(s: &str) -> Vec<u8> {
    let mut v = (s.len() as u64).to_le_bytes().to_vec();
    v.extend_from_slice(s.as_bytes());
    v
}

/// Config with two keys: an unscoped admin + a scoped "ci" key.
#[must_use]
pub fn config_with_keys() -> Config {
    Config {
        keys: vec![
            ApiKey {
                name: "admin".into(),
                key: "plm_admin".into(),
                ..ApiKey::default()
            },
            ApiKey {
                name: "ci".into(),
                key: "plm_ci".into(),
                models: vec!["m1".into()],
                ..ApiKey::default()
            },
        ],
        ..Config::default()
    }
}

pub async fn start(config: Config) -> TestServer {
    let tmp = tempfile::tempdir().unwrap();
    let dirs = PallamaDirs {
        config_dir: tmp.path().join("c"),
        data_dir: tmp.path().join("d"),
    };
    dirs.ensure().unwrap();
    let gguf = dirs.models_dir().join("m1-q4_k_m.gguf");
    write_gguf(&gguf);
    let store = pallama_core::Store::open(&dirs).unwrap();
    store
        .upsert_model(&pallama_core::ModelRow {
            name: "m1".into(),
            repo: "o/m1".into(),
            quant: "Q4_K_M".into(),
            path: gguf.display().to_string(),
            bytes: 500_000_000,
            sha256: None,
            mmproj_path: None,
            shards: 1,
            arch: Some("qwen3".into()),
            params: Some(0.5),
            ctx_train: Some(40_960),
            pulled_at: 1,
        })
        .unwrap();
    // Production always has an active engine row (engine-manager); the
    // #20 identity manifest build reads it.
    store
        .upsert_engine(&pallama_core::EngineRow {
            tag: "stub-1".into(),
            asset: "stub".into(),
            sha256: "stub-sha".into(),
            installed_at: 1,
            active: true,
            manifest: "{}".into(),
            kind: pallama_core::engine_kind::EngineKind::default(),
        })
        .unwrap();
    store.set_active_engine("stub-1").unwrap();
    let manifest = pallama_runtime::probe_manifest(&find_stub(), "stub").unwrap();
    let engine = LlamaCppEngine::new(manifest);
    let mut s = Supervisor::new(
        dirs.clone(),
        config.clone(),
        EventBus::default(),
        pallama_core::hardware::Hardware {
            physical_cores: 4,
            total_ram_mib: 16_000,
            gpus: vec![pallama_core::hardware::GpuInfo {
                name: "g".into(),
                description: "STUB".into(),
                total_mib: 24_000,
                free_mib: 24_000,
            }],
        },
        Arc::new(engine),
    );
    s.reaper_interval = Duration::from_hours(1);
    let sup = Arc::new(s);
    let reaper = sup.spawn_reaper();
    let state = Arc::new(AppState::new(
        dirs.clone(),
        config,
        sup.clone(),
        EventBus::default(),
    ));
    let app = router(state.clone());
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
        .await
        .unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    TestServer {
        base: format!("http://127.0.0.1:{port}"),
        _tmp: tmp,
        dirs,
        state,
        _sup_reaper: reaper,
    }
}

pub fn client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .unwrap()
}

/// A standalone stub-llama-server playing an OpenAI-compatible remote.
pub struct RemoteStub {
    pub base: String,
    child: tokio::process::Child,
}

impl RemoteStub {
    pub async fn shutdown(mut self) {
        self.child.start_kill().ok();
        let _ = self.child.wait().await;
    }
}

/// Spawn the stub on an ephemeral port with `--alias m1` (any `OpenAI`
/// server shape works as a remote); polls /health until ready.
pub async fn spawn_remote_stub() -> RemoteStub {
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
        .await
        .unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    let child = tokio::process::Command::new(find_stub())
        .args([
            "--host",
            "127.0.0.1",
            "--port",
            &port.to_string(),
            "--alias",
            "m1",
            "-m",
            "/dev/null",
        ])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .expect("spawn remote stub");
    let base = format!("http://127.0.0.1:{port}");
    for _ in 0..100 {
        if reqwest::Client::new()
            .get(format!("{base}/health"))
            .timeout(Duration::from_secs(1))
            .send()
            .await
            .is_ok_and(|r| r.status().is_success())
        {
            return RemoteStub { base, child };
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    panic!("remote stub never became ready at {base}");
}

/// Attach a [[remotes]] entry to a config.
#[must_use]
pub fn with_remote(mut cfg: Config, name: &str, url: &str) -> Config {
    cfg.remotes.push(pallama_core::Remote {
        name: name.to_string(),
        url: url.to_string(),
        key: String::new(),
    });
    cfg
}
