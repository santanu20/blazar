//! Gateway end-to-end: real axum server over a real supervisor spawning
//! stub-llama-server children. Exercises both API surfaces, translation,
//! auth, `keep_alive` eviction, `num_ctx` restart, and cancellation.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use pallama_core::hardware::{GpuInfo, Hardware};
use pallama_core::store::Store;
use pallama_core::{Config, PallamaDirs};
use pallama_runtime::EventBus;
use pallama_gateway::state::AppState;
use pallama_gateway::{router, translate as tr};
use pallama_runtime::{LlamaCppEngine, Supervisor};

/// Locate the stub binary by walking up from this test executable to the
/// cargo target dir (works under deps/, debug/, release/).
fn stub_bin() -> PathBuf {
    let exe = std::env::current_exe().expect("test exe path");
    for dir in exe.ancestors() {
        let candidate = dir.join("stub-llama-server");
        if candidate.exists() {
            return candidate;
        }
        if dir.file_name().is_some_and(|n| n == "target") {
            break;
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

struct TestServer {
    base: String,
    _tmp: tempfile::TempDir,
    _dirs: PallamaDirs,
    state: Arc<AppState>,
    _sup_reaper: tokio::task::JoinHandle<()>,
}

async fn start(config: Config) -> TestServer {
    start_with(config, Vec::new()).await
}

async fn start_with(config: Config, child_env: Vec<(String, String)>) -> TestServer {
    let tmp = tempfile::tempdir().unwrap();
    let dirs = PallamaDirs { config_dir: tmp.path().join("c"), data_dir: tmp.path().join("d") };
    dirs.ensure().unwrap();
    let gguf = dirs.models_dir().join("m1-q4_k_m.gguf");
    write_gguf(&gguf);
    let store = Store::open(&dirs).unwrap();
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

    let mut engine = LlamaCppEngine::new(
        pallama_runtime::probe_manifest(&stub_bin(), "stub").unwrap(),
    );
    engine.child_env = child_env;
    let mut s = Supervisor::new(
        dirs.clone(),
        config.clone(),
        EventBus::default(),
        Hardware {
            physical_cores: 4,
            total_ram_mib: 16_000,
            gpus: vec![GpuInfo {
                name: "g".into(),
                description: "STUB".into(),
                total_mib: 24_000,
                free_mib: 24_000,
            }],
        },
        Arc::new(engine),
    );
    s.reaper_interval = Duration::from_hours(1); // reaper off; tests drive lifecycle
    let sup = Arc::new(s);
    let reaper = sup.spawn_reaper();
    let state = Arc::new(AppState::new(dirs.clone(), config, sup.clone(), EventBus::default()));
    let app = router(state.clone());
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    TestServer { base: format!("http://127.0.0.1:{port}"), _tmp: tmp, _dirs: dirs, state, _sup_reaper: reaper }
}

fn client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .unwrap()
}

#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__healthz_version_tags_models() {
    let ts = start(Config::default()).await;
    let c = client();
    assert_eq!(c.get(format!("{}/healthz", ts.base)).send().await.unwrap().status(), 200);
    let v: serde_json::Value = c.get(format!("{}/api/version", ts.base)).send().await.unwrap().json().await.unwrap();
    assert!(v["version"].as_str().is_some_and(|s| !s.is_empty()));
    let t: serde_json::Value = c.get(format!("{}/api/tags", ts.base)).send().await.unwrap().json().await.unwrap();
    assert_eq!(t["models"][0]["name"], "m1:q4_k_m");
    assert_eq!(t["models"][0]["details"]["quantization_level"], "Q4_K_M");
    let m: serde_json::Value = c.get(format!("{}/v1/models", ts.base)).send().await.unwrap().json().await.unwrap();
    assert_eq!(m["data"][0]["id"], "m1:q4_k_m");
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__openai_chat_nonstream_and_stream() {
    let ts = start(Config::default()).await;
    let c = client();
    let body = serde_json::json!({
        "model": "m1",
        "messages": [{"role": "user", "content": "hello gateway"}],
        "stream": false,
    });
    let r: serde_json::Value = c
        .post(format!("{}/v1/chat/completions", ts.base))
        .json(&body)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    // Byte-faithful: the stub's own reply shape, untouched.
    assert_eq!(r["choices"][0]["message"]["content"], "stub:m1:hello gateway");
    assert!(r["usage"]["completion_tokens"].as_i64().unwrap_or(0) > 0);

    // Stream: >= 2 data chunks + [DONE].
    let sbody = serde_json::json!({
        "model": "m1", "stream": true,
        "messages": [{"role": "user", "content": "stream me"}],
    });
    let text = c
        .post(format!("{}/v1/chat/completions", ts.base))
        .json(&sbody)
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    let chunks = text.matches("data: ").count();
    assert!(chunks >= 3, "expected role+content+finish chunks: {chunks}");
    assert!(text.contains("[DONE]"));
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__ollama_chat_nonstream_translation() {
    let ts = start(Config::default()).await;
    let c = client();
    let body = serde_json::json!({
        "model": "m1",
        "messages": [{"role": "user", "content": "translate me"}],
        "stream": false,
        "options": {"temperature": 0.5, "num_predict": 50},
    });
    let r: serde_json::Value = c
        .post(format!("{}/api/chat", ts.base))
        .json(&body)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(r["message"]["role"], "assistant");
    assert_eq!(r["message"]["content"], "stub:m1:translate me");
    assert_eq!(r["done"], true);
    assert!(r["eval_count"].as_i64().unwrap_or(0) > 0, "final counts present");
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__ollama_chat_stream_ndjson_with_final_counts() {
    let ts = start(Config::default()).await;
    let c = client();
    let body = serde_json::json!({
        "model": "m1",
        "messages": [{"role": "user", "content": "one two"}],
        "stream": true,
    });
    let text = c
        .post(format!("{}/api/chat", ts.base))
        .json(&body)
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    let lines: Vec<serde_json::Value> = text
        .lines()
        .filter(|l| !l.is_empty())
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()
        .unwrap();
    assert!(lines.len() >= 2, "content chunks + final: {lines:?}");
    let last = lines.last().unwrap();
    assert_eq!(last["done"], true);
    assert!(last["eval_count"].as_i64().unwrap_or(0) > 0);
    let joined: String = lines
        .iter()
        .filter_map(|l| l["message"]["content"].as_str())
        .collect();
    assert!(joined.contains("stub:m1:one two"), "reassembled content: {joined}");
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__ollama_unknown_option_and_unknown_model() {
    let ts = start(Config::default()).await;
    let c = client();
    let bad = serde_json::json!({
        "model": "m1", "messages": [],
        "options": {"vram_magic": 3},
    });
    let resp = c.post(format!("{}/api/chat", ts.base)).json(&bad).send().await.unwrap();
    assert_eq!(resp.status(), 400);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(body["error"].as_str().unwrap().contains("vram_magic"));

    let unknown = serde_json::json!({"model": "zzz", "messages": []});
    let resp = c.post(format!("{}/api/chat", ts.base)).json(&unknown).send().await.unwrap();
    assert_eq!(resp.status(), 404);

    // Near-miss model gets a suggestion (complaint #6 family UX).
    let near = serde_json::json!({"model": "mX", "messages": []});
    let resp = c.post(format!("{}/api/chat", ts.base)).json(&near).send().await.unwrap();
    assert_eq!(resp.status(), 404);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(body["error"].as_str().unwrap().contains("did you mean"));
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__keep_alive_zero_evicts() {
    let ts = start(Config::default()).await;
    let c = client();
    let body = serde_json::json!({
        "model": "m1", "stream": false, "keep_alive": 0,
        "messages": [{"role": "user", "content": "bye"}],
    });
    let r: serde_json::Value = c.post(format!("{}/api/chat", ts.base)).json(&body).send().await.unwrap().json().await.unwrap();
    assert_eq!(r["done"], true);
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(ts.state.sup.ps().is_empty(), "keep_alive=0 evicts the instance");
}

#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__num_ctx_restarts_instance_at_requested_size() {
    let ts = start(Config::default()).await;
    let c = client();
    // First request at default ctx (16384).
    let body = serde_json::json!({
        "model": "m1", "stream": false,
        "messages": [{"role": "user", "content": "a"}],
    });
    let _: serde_json::Value = c.post(format!("{}/api/chat", ts.base)).json(&body).send().await.unwrap().json().await.unwrap();
    assert_eq!(ts.state.sup.ps()[0].ctx, 16384);

    // Request 32768 -> instance restarts at 32768 (complaint #13).
    let body = serde_json::json!({
        "model": "m1", "stream": false,
        "messages": [{"role": "user", "content": "b"}],
        "options": {"num_ctx": 32768},
    });
    let _: serde_json::Value = c.post(format!("{}/api/chat", ts.base)).json(&body).send().await.unwrap().json().await.unwrap();
    let ps = ts.state.sup.ps();
    assert!(!ps.is_empty(), "instance respawned at requested ctx");
    assert_eq!(ps[0].ctx, 32768, "ctx honored, never silently truncated");
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__auth_bearer_required_when_keys_set() {
    let cfg = Config {
        api_keys: vec!["secret-1".into()],
        ..Config::default()
    };
    let ts = start(cfg).await;
    let c = client();
    let body = serde_json::json!({"model": "m1", "messages": []});
    let no_auth = c.post(format!("{}/api/chat", ts.base)).json(&body).send().await.unwrap();
    assert_eq!(no_auth.status(), 401);
    let tags = c.get(format!("{}/api/tags", ts.base)).bearer_auth("secret-1").send().await.unwrap();
    assert_eq!(tags.status(), 200);
    let with_auth = c
        .post(format!("{}/api/chat", ts.base))
        .bearer_auth("secret-1")
        .json(&serde_json::json!({
            "model": "m1", "stream": false,
            "messages": [{"role": "user", "content": "x"}],
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(with_auth.status(), 200);
    // healthz stays open.
    assert_eq!(c.get(format!("{}/healthz", ts.base)).send().await.unwrap().status(), 200);
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__generate_raw_ok_templated_400_and_embeddings() {
    let ts = start(Config::default()).await;
    let c = client();
    let raw = serde_json::json!({"model": "m1", "prompt": "complete this", "stream": false});
    let r: serde_json::Value = c.post(format!("{}/api/generate", ts.base)).json(&raw).send().await.unwrap().json().await.unwrap();
    assert!(r["done"] == true, "generate done: {r}");
    assert!(r["response"].as_str().unwrap().contains("complete this"));

    let templated = serde_json::json!({"model": "m1", "prompt": "x", "system": "sys"});
    let resp = c.post(format!("{}/api/generate", ts.base)).json(&templated).send().await.unwrap();
    assert_eq!(resp.status(), 400);
    let b: serde_json::Value = resp.json().await.unwrap();
    assert!(b["error"].as_str().unwrap().contains("/api/chat"));

    let emb = serde_json::json!({"model": "m1", "prompt": "embed me"});
    let r: serde_json::Value = c.post(format!("{}/api/embeddings", ts.base)).json(&emb).send().await.unwrap().json().await.unwrap();
    assert!(r["embedding"].as_array().is_some(), "embedding array: {r}");
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__ps_and_show() {
    let ts = start(Config::default()).await;
    let c = client();
    let body = serde_json::json!({"model": "m1", "stream": false, "messages": [{"role": "user", "content": "s"}]});
    let _: serde_json::Value = c.post(format!("{}/api/chat", ts.base)).json(&body).send().await.unwrap().json().await.unwrap();
    let ps: serde_json::Value = c.get(format!("{}/api/ps", ts.base)).send().await.unwrap().json().await.unwrap();
    assert_eq!(ps["models"][0]["name"], "m1");
    assert_eq!(ps["models"][0]["pallama_state"], "ready");
    assert_eq!(ps["models"][0]["pallama_ctx"], 16384);

    let show_body = serde_json::json!({"model": "m1"});
    let s: serde_json::Value = c.post(format!("{}/api/show", ts.base)).json(&show_body).send().await.unwrap().json().await.unwrap();
    assert_eq!(s["details"]["quantization_level"], "Q4_K_M");
    assert_eq!(s["model_info"]["general.architecture"], "qwen3");
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__client_disconnect_frees_slot() {
    let ts = start_with(
        Config::default(),
        vec![("STUB_DELAY_MS".into(), "10000".into())],
    )
    .await;
    let c = client();
    // Start a slow streamed request and drop it mid-flight.
    let body = serde_json::json!({
        "model": "m1", "stream": true,
        "messages": [{"role": "user", "content": "slow"}],
    });
    let req = c.post(format!("{}/v1/chat/completions", ts.base)).json(&body);
    let resp = req.send().await.unwrap();
    drop(resp); // client goes away
    tokio::time::sleep(Duration::from_millis(400)).await;
    let ps = ts.state.sup.ps();
    assert!(!ps.is_empty());
    assert_eq!(ps[0].in_flight, 0, "slot freed after client disconnect");
    ts.state.sup.shutdown_all().await.unwrap();
}

// Pure-function coverage lives next to translate.rs; one sanity echo here.
#[test]
#[allow(non_snake_case)]
fn unit__translation_module_reachable() {
    let (req, ctx) = tr::chat_to_openai(&serde_json::json!({
        "model": "m", "messages": [], "options": {"num_ctx": 123}
    }))
    .unwrap();
    assert_eq!(ctx, Some(123));
    assert_eq!(req["model"], "m");
}
