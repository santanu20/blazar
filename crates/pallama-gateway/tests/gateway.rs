//! Gateway end-to-end: real axum server over a real supervisor spawning
//! stub-llama-server children. Exercises both API surfaces, translation,
//! auth, `keep_alive` eviction, `num_ctx` restart, and cancellation.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use pallama_core::hardware::{GpuInfo, Hardware};
use pallama_core::store::Store;
use pallama_core::{Config, PallamaDirs};
use pallama_gateway::state::AppState;
use pallama_gateway::{router, translate as tr};
use pallama_runtime::EventBus;
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
    pub dirs: PallamaDirs,
    state: Arc<AppState>,
    _sup_reaper: tokio::task::JoinHandle<()>,
}

async fn start(config: Config) -> TestServer {
    start_with(config, Vec::new()).await
}

async fn start_with(config: Config, child_env: Vec<(String, String)>) -> TestServer {
    let tmp = tempfile::tempdir().unwrap();
    let dirs = PallamaDirs {
        config_dir: tmp.path().join("c"),
        data_dir: tmp.path().join("d"),
    };
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
    // Second model for router-mode tests (unused by single-model tests).
    let gguf2 = dirs.models_dir().join("m2-q4_k_m.gguf");
    write_gguf(&gguf2);
    store
        .upsert_model(&pallama_core::ModelRow {
            name: "m2".into(),
            repo: "o/m2".into(),
            quant: "Q4_K_M".into(),
            path: gguf2.display().to_string(),
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

    let mut engine =
        LlamaCppEngine::new(pallama_runtime::probe_manifest(&stub_bin(), "stub").unwrap());
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
    assert_eq!(
        c.get(format!("{}/healthz", ts.base))
            .send()
            .await
            .unwrap()
            .status(),
        200
    );
    let v: serde_json::Value = c
        .get(format!("{}/api/version", ts.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(v["version"].as_str().is_some_and(|s| !s.is_empty()));
    let t: serde_json::Value = c
        .get(format!("{}/api/tags", ts.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(t["models"][0]["name"], "m1:q4_k_m");
    assert_eq!(t["models"][0]["details"]["quantization_level"], "Q4_K_M");
    let m: serde_json::Value = c
        .get(format!("{}/v1/models", ts.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
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
    assert_eq!(
        r["choices"][0]["message"]["content"],
        "stub:m1:hello gateway"
    );
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
    assert!(
        r["eval_count"].as_i64().unwrap_or(0) > 0,
        "final counts present"
    );
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
    assert!(
        joined.contains("stub:m1:one two"),
        "reassembled content: {joined}"
    );
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
    let resp = c
        .post(format!("{}/api/chat", ts.base))
        .json(&bad)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(body["error"].as_str().unwrap().contains("vram_magic"));

    let unknown = serde_json::json!({"model": "zzz", "messages": []});
    let resp = c
        .post(format!("{}/api/chat", ts.base))
        .json(&unknown)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);

    // Near-miss model gets a suggestion (complaint #6 family UX).
    let near = serde_json::json!({"model": "mX", "messages": []});
    let resp = c
        .post(format!("{}/api/chat", ts.base))
        .json(&near)
        .send()
        .await
        .unwrap();
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
    let r: serde_json::Value = c
        .post(format!("{}/api/chat", ts.base))
        .json(&body)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(r["done"], true);
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(
        ts.state.sup.ps().is_empty(),
        "keep_alive=0 evicts the instance"
    );
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
    let _: serde_json::Value = c
        .post(format!("{}/api/chat", ts.base))
        .json(&body)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(ts.state.sup.ps()[0].ctx, 16384);

    // Request 32768 -> instance restarts at 32768 (complaint #13).
    let body = serde_json::json!({
        "model": "m1", "stream": false,
        "messages": [{"role": "user", "content": "b"}],
        "options": {"num_ctx": 32768},
    });
    let _: serde_json::Value = c
        .post(format!("{}/api/chat", ts.base))
        .json(&body)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let ps = ts.state.sup.ps();
    assert!(!ps.is_empty(), "instance respawned at requested ctx");
    assert_eq!(ps[0].ctx, 32768, "ctx honored, never silently truncated");
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__auth_bearer_required_when_keys_set() {
    let cfg = Config {
        keys: vec![pallama_core::ApiKey {
            name: "admin".into(),
            key: "secret-1".into(),
            ..pallama_core::ApiKey::default()
        }],
        ..Config::default()
    };
    let ts = start(cfg).await;
    let c = client();
    let body = serde_json::json!({"model": "m1", "messages": []});
    let no_auth = c
        .post(format!("{}/api/chat", ts.base))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(no_auth.status(), 401);
    let tags = c
        .get(format!("{}/api/tags", ts.base))
        .bearer_auth("secret-1")
        .send()
        .await
        .unwrap();
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
    assert_eq!(
        c.get(format!("{}/healthz", ts.base))
            .send()
            .await
            .unwrap()
            .status(),
        200
    );
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__keys_scoped_rate_and_accounting() {
    use pallama_core::ApiKey;
    let scoped = ApiKey {
        name: "ci".into(),
        key: "plm_scoped".into(),
        models: vec!["m1".into()],
        rpm: 2,
        tpm: 0,
        daily_tokens: 0,
        max_concurrent: 0,
    };
    let cfg = Config {
        keys: vec![
            ApiKey {
                name: "admin".into(),
                key: "plm_admin".into(),
                ..ApiKey::default()
            },
            scoped,
        ],
        ..Config::default()
    };
    let ts = start(cfg).await;
    let c = client();

    // Scope: wrong model -> 403 naming the key.
    let wrong = c
        .post(format!("{}/v1/chat/completions", ts.base))
        .bearer_auth("plm_scoped")
        .json(&serde_json::json!({"model": "other", "messages": []}))
        .send()
        .await
        .unwrap();
    assert_eq!(wrong.status(), 403);

    // rpm=2 exhaustion needs model-routed charges; /api/tags (GET, not
    // model-routed) never consumes budget — prove it stays 200.
    let limited = c
        .get(format!("{}/api/tags", ts.base))
        .bearer_auth("plm_scoped")
        .send()
        .await
        .unwrap();
    assert_eq!(limited.status(), 200);
    let chat = serde_json::json!({
        "model": "m1", "stream": false,
        "messages": [{"role": "user", "content": "hello"}],
    });
    for _ in 0..2 {
        let r = c
            .post(format!("{}/api/chat", ts.base))
            .bearer_auth("plm_scoped")
            .json(&chat)
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 200, "within rpm=2");
    }
    let over = c
        .post(format!("{}/api/chat", ts.base))
        .bearer_auth("plm_scoped")
        .json(&chat)
        .send()
        .await
        .unwrap();
    assert_eq!(over.status(), 429, "rpm=2 exhausted");
    assert!(over.headers().contains_key("retry-after"));

    // Management: admin lists (secrets redacted), scoped key refused.
    let listed: serde_json::Value = c
        .get(format!("{}/api/keys", ts.base))
        .bearer_auth("plm_admin")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let keys = listed["keys"].as_array().cloned().unwrap_or_default();
    assert_eq!(keys.len(), 2);
    assert!(keys
        .iter()
        .all(|k| !k["key"].as_str().unwrap_or("").contains("plm_admin")));
    let refused = c
        .get(format!("{}/api/keys", ts.base))
        .bearer_auth("plm_scoped")
        .send()
        .await
        .unwrap();
    assert_eq!(refused.status(), 403);
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__keys_concurrency_cap() {
    use pallama_core::ApiKey;
    // Held streaming response keeps the slot leased (GuardedBody Drop fires
    // only when the body drains or the client disconnects). The stub delays
    // its SSE chunks via env, inherited by the spawned stub child; tests run
    // with --test-threads=1 in the gate, so the global env is safe here.
    std::env::set_var("STUB_DELAY_CHUNK_MS", "400");
    let cfg = Config {
        keys: vec![ApiKey {
            name: "cap".into(),
            key: "plm_cap".into(),
            models: vec!["m1".into()],
            max_concurrent: 1,
            ..ApiKey::default()
        }],
        ..Config::default()
    };
    let ts = start(cfg).await;
    let c = client();
    let sbody = serde_json::json!({
        "model": "m1", "stream": true,
        "messages": [{"role": "user", "content": "hold my slot"}],
    });

    // First request leases the only slot; headers arrive before the
    // (delayed) body drains.
    let r1 = c
        .post(format!("{}/v1/chat/completions", ts.base))
        .bearer_auth("plm_cap")
        .json(&sbody)
        .send()
        .await
        .unwrap();
    assert_eq!(r1.status(), 200);

    // Second request while r1's body is still streaming -> 429 concurrent.
    let r2 = c
        .post(format!("{}/v1/chat/completions", ts.base))
        .bearer_auth("plm_cap")
        .json(&sbody)
        .send()
        .await
        .unwrap();
    assert_eq!(r2.status(), 429, "cap=1 must reject the parallel stream");
    assert!(r2.headers().contains_key("retry-after"));
    let body: serde_json::Value = r2.json().await.unwrap();
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap_or("")
            .contains("concurrent"),
        "rejection names the concurrency budget: {body}"
    );

    // Dropping r1 disconnects the client; hyper drops the wrapped body and
    // the lease releases. Poll until the slot frees (bounded by chunk delay).
    drop(r1);
    let mut freed = None;
    for _ in 0..40 {
        tokio::time::sleep(std::time::Duration::from_millis(125)).await;
        let r3 = c
            .post(format!("{}/v1/chat/completions", ts.base))
            .bearer_auth("plm_cap")
            .json(&serde_json::json!({
                "model": "m1", "stream": false,
                "messages": [{"role": "user", "content": "after release"}],
            }))
            .send()
            .await
            .unwrap();
        if r3.status() == 200 {
            freed = Some(r3);
            break;
        }
        assert_eq!(r3.status(), 429, "only the cap can reject here");
    }
    assert!(
        freed.is_some(),
        "lease must release after client disconnect"
    );

    std::env::remove_var("STUB_DELAY_CHUNK_MS");
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__tls_serves_https_and_cors_headers() {
    let cfg = Config {
        tls_cert: "tests/fixtures/tls-cert.pem".into(),
        tls_key: "tests/fixtures/tls-key.pem".into(),
        cors_origins: vec!["https://chat.example".into()],
        ..Config::default()
    };
    let ts = start(cfg).await;
    // start() serves plain HTTP; TLS needs the full serve() path — bind
    // a second ephemeral port through gateway::serve with the same state.
    let state = Arc::clone(&ts.state);
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
        .await
        .unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    tokio::spawn(async move {
        pallama_gateway::serve(
            state,
            "127.0.0.1",
            port,
            Box::pin(async move {
                let _ = rx.await;
            }),
        )
        .await
        .unwrap();
    });
    // Wait for the TLS listener to accept.
    let c = reqwest::Client::builder()
        .danger_accept_invalid_certs(true)
        .build()
        .unwrap();
    let base = format!("https://127.0.0.1:{port}");
    let mut ok = false;
    for _ in 0..50 {
        if c.get(format!("{base}/healthz"))
            .send()
            .await
            .is_ok_and(|r| r.status() == 200)
        {
            ok = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(ok, "TLS server never became ready");
    // CORS: allowed origin echoed, disallowed origin gets no ACAO header.
    let allowed = c
        .get(format!("{base}/healthz"))
        .header("origin", "https://chat.example")
        .send()
        .await
        .unwrap();
    assert_eq!(
        allowed
            .headers()
            .get("access-control-allow-origin")
            .and_then(|v| v.to_str().ok()),
        Some("https://chat.example")
    );
    let denied = c
        .get(format!("{base}/healthz"))
        .header("origin", "https://evil.example")
        .send()
        .await
        .unwrap();
    assert!(denied
        .headers()
        .get("access-control-allow-origin")
        .is_none());
    let _ = tx.send(());
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__responses_chaining_and_store() {
    let ts = start(Config::default()).await;
    let c = client();
    let r1: serde_json::Value = c
        .post(format!("{}/v1/responses", ts.base))
        .json(&serde_json::json!({
            "model": "m1",
            "input": [{"role": "user", "content": "hello"}],
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let id = r1["id"].as_str().unwrap().to_string();
    assert!(id.starts_with("resp_"), "gateway re-id'd: {id}");
    assert_eq!(r1["debug_input_items"].as_array().unwrap().len(), 1);
    assert_eq!(r1["usage"]["input_tokens"], 3);

    // Chain: previous_response_id + new input -> full history to the child.
    let r2: serde_json::Value = c
        .post(format!("{}/v1/responses", ts.base))
        .json(&serde_json::json!({
            "model": "m1",
            "previous_response_id": id,
            "input": [{"role": "user", "content": "again"}],
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let items = r2["debug_input_items"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    assert_eq!(
        items.len(),
        3,
        "stored input + stored output + new input: {items:?}"
    );
    assert_eq!(items[2]["content"], "again");
    let id2 = r2["id"].as_str().unwrap().to_string();
    assert!(
        id2.starts_with("resp_") && id2 != id,
        "fresh id per response"
    );

    // Unknown id: named 404 teaching the expiry semantics.
    let missing = c
        .post(format!("{}/v1/responses", ts.base))
        .json(&serde_json::json!({
            "model": "m1",
            "previous_response_id": "resp_nope",
            "input": "x",
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(missing.status(), 404);
    let b: serde_json::Value = missing.json().await.unwrap();
    assert!(
        b["error"]["message"].as_str().unwrap().contains("expired"),
        "{b}"
    );

    // store:false -> NOT stored (chaining from it 404s).
    let unstored: serde_json::Value = c
        .post(format!("{}/v1/responses", ts.base))
        .json(&serde_json::json!({
            "model": "m1", "store": false, "input": "one-shot",
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let uid = unstored["id"].as_str().unwrap().to_string();
    let gone = c
        .post(format!("{}/v1/responses", ts.base))
        .json(&serde_json::json!({
            "model": "m1",
            "previous_response_id": uid,
            "input": "x",
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(gone.status(), 404);
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__generate_raw_ok_templated_400_and_embeddings() {
    let ts = start(Config::default()).await;
    let c = client();
    let raw = serde_json::json!({"model": "m1", "prompt": "complete this", "stream": false});
    let r: serde_json::Value = c
        .post(format!("{}/api/generate", ts.base))
        .json(&raw)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(r["done"] == true, "generate done: {r}");
    assert!(r["response"].as_str().unwrap().contains("complete this"));

    let templated = serde_json::json!({"model": "m1", "prompt": "x", "system": "sys"});
    let resp = c
        .post(format!("{}/api/generate", ts.base))
        .json(&templated)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
    let b: serde_json::Value = resp.json().await.unwrap();
    assert!(b["error"].as_str().unwrap().contains("/api/chat"));

    let emb = serde_json::json!({"model": "m1", "prompt": "embed me"});
    let r: serde_json::Value = c
        .post(format!("{}/api/embeddings", ts.base))
        .json(&emb)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(r["embedding"].as_array().is_some(), "embedding array: {r}");
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__ps_and_show() {
    let ts = start(Config::default()).await;
    let c = client();
    let body = serde_json::json!({"model": "m1", "stream": false, "messages": [{"role": "user", "content": "s"}]});
    let _: serde_json::Value = c
        .post(format!("{}/api/chat", ts.base))
        .json(&body)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let ps: serde_json::Value = c
        .get(format!("{}/api/ps", ts.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(ps["models"][0]["name"], "m1");
    assert_eq!(ps["models"][0]["pallama_state"], "ready");
    assert_eq!(ps["models"][0]["pallama_ctx"], 16384);

    let show_body = serde_json::json!({"model": "m1"});
    let s: serde_json::Value = c
        .post(format!("{}/api/show", ts.base))
        .json(&show_body)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(s["details"]["quantization_level"], "Q4_K_M");
    assert_eq!(s["model_info"]["general.architecture"], "qwen3");
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__pull_failure_emits_error_line_not_silence() {
    // Regression (found by scripts/validate.py): take_while dropped the
    // terminal line, so failed pulls were silent empty 200s (H1). An
    // invalid target fails before any network — deterministic, offline.
    let ts = start(Config::default()).await;
    let c = client();
    let resp = c
        .post(format!("{}/api/pull", ts.base))
        .json(&serde_json::json!({"model": "definitely-not-a-real-owner-zzz/nope:Q4_0", "stream": true}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let text = resp.text().await.unwrap();
    assert!(
        text.contains("\"error\"") || text.contains("error"),
        "pull failure must reach the client, got: {text:?}"
    );
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__responses_api_proxied_byte_faithful() {
    let ts = start(Config::default()).await;
    let c = client();
    let body = serde_json::json!({
        "model": "m1",
        "input": "hello responses",
        "stream": false,
    });
    let r: serde_json::Value = c
        .post(format!("{}/v1/responses", ts.base))
        .json(&body)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(r["object"], "response");
    assert_eq!(r["model"], "m1");
    // missing model -> 400 naming the requirement
    let resp = c
        .post(format!("{}/v1/responses", ts.base))
        .json(&serde_json::json!({"input": "x"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__audio_transcriptions_no_backend_teaching_501() {
    let ts = start(Config::default()).await;
    let c = client();
    // OpenAI-style multipart: file part (with filename) + model field.
    // Without a whisper lane installed and without remote intent the
    // gateway answers a teaching 501 (llama.cpp children cannot STT).
    let boundary = "pallama-test-boundary";
    let mp = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\nContent-Type: audio/wav\r\n\r\nRIFF-binary-bytes-here\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nm1\r\n--{boundary}--\r\n"
    );
    let resp = c
        .post(format!("{}/v1/audio/transcriptions", ts.base))
        .header(
            "content-type",
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(mp)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 501);
    let r: serde_json::Value = resp.json().await.unwrap();
    let msg = r["error"]["message"].as_str().unwrap_or_default();
    assert!(
        msg.contains("pallama whisper install"),
        "teaching error should name the fix: {r}"
    );
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__infill_control_tokenize_proxied() {
    let ts = start(Config::default()).await;
    let c = client();
    let infill: serde_json::Value = c
        .post(format!("{}/infill", ts.base))
        .json(&serde_json::json!({"model": "m1", "input": "prefix"}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(infill["content"], "stub infill");

    let ctrl: serde_json::Value = c
        .post(format!("{}/v1/chat/completions/control", ts.base))
        .json(&serde_json::json!({
            "model": "m1",
            "control_vectors": [{"id": 1}, {"id": 2}],
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(ctrl["control_vectors_applied"], 2);

    let tok: serde_json::Value = c
        .post(format!("{}/tokenize", ts.base))
        .json(&serde_json::json!({"model": "m1", "content": "a b c"}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(tok["tokens"].as_array().unwrap().len(), 3);
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__session_save_restore_erase_roundtrip() {
    let ts = start(Config::default()).await;
    let c = client();
    // warm the instance (save needs live slot state upstream; ours just
    // needs the child reachable).
    let _: serde_json::Value = c
        .post(format!("{}/api/chat", ts.base))
        .json(&serde_json::json!({"model": "m1", "stream": false, "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();

    let save: serde_json::Value = c
        .post(format!("{}/api/session", ts.base))
        .json(&serde_json::json!({"model": "m1", "action": "save", "filename": "ckpt1"}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(save["status"], "ok", "{save}");

    // file landed in the per-model sessions dir
    let sess_dir = ts.dirs.sessions_dir().join("m1");
    assert!(sess_dir.join("ckpt1").exists(), "checkpoint file missing");

    let restore: serde_json::Value = c
        .post(format!("{}/api/session", ts.base))
        .json(&serde_json::json!({"model": "m1", "action": "restore", "filename": "ckpt1"}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(restore["status"], "ok", "{restore}");

    // listing sees it
    let list: serde_json::Value = c
        .get(format!("{}/api/session?model=m1", ts.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(list["sessions"].as_array().unwrap().len(), 1);
    assert_eq!(list["sessions"][0]["filename"], "ckpt1");

    // path traversal rejected
    let bad = c
        .post(format!("{}/api/session", ts.base))
        .json(&serde_json::json!({"model": "m1", "action": "save", "filename": "../escape"}))
        .send()
        .await
        .unwrap();
    assert_eq!(bad.status(), 400);

    // restore of missing checkpoint -> 404 from the child
    let missing = c
        .post(format!("{}/api/session", ts.base))
        .json(&serde_json::json!({"model": "m1", "action": "restore", "filename": "nope"}))
        .send()
        .await
        .unwrap();
    assert_eq!(missing.status(), 404);

    let erase: serde_json::Value = c
        .post(format!("{}/api/session", ts.base))
        .json(&serde_json::json!({"model": "m1", "action": "erase", "filename": "ckpt1"}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(erase["status"], "ok");
    assert!(!sess_dir.join("ckpt1").exists());
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__router_mode_one_child_serves_both_models() {
    let cfg = Config {
        router: true,
        router_max_models: 2,
        ..Config::default()
    };
    let ts = start(cfg).await;
    let c = client();

    // Both models route through the single router child.
    for m in ["m1", "m2"] {
        let body = serde_json::json!({
            "model": m,
            "stream": false,
            "messages": [{"role": "user", "content": "hi"}],
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
        assert_eq!(r["model"], m, "router routed {m}: {r}");
    }

    // ONE engine child process (the router), not one per model.
    assert_eq!(
        ts.state.sup.ps().len(),
        1,
        "router must be a single instance"
    );
    assert_eq!(ts.state.sup.ps()[0].name, "_router");

    // ps translates the child's /models listing.
    let ps: serde_json::Value = c
        .get(format!("{}/api/ps", ts.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let names: Vec<&str> = ps["models"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|m| m["name"].as_str())
        .collect();
    assert!(
        names.contains(&"m1") && names.contains(&"m2"),
        "ps rows: {ps}"
    );

    // Unknown model fails fast against the store (never spawns).
    let resp = c
        .post(format!("{}/v1/chat/completions", ts.base))
        .json(&serde_json::json!({"model": "nope", "messages": []}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);

    // evict forwards the engine unload (model stays preset-listed).
    let ev: serde_json::Value = c
        .post(format!("{}/api/evict", ts.base))
        .json(&serde_json::json!({"model": "m1"}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(ev["status"], "ok");
    assert_eq!(
        ts.state.sup.ps().len(),
        1,
        "unload must not kill the router"
    );

    // num_ctx in router mode: explicit 400 with the overlay hint.
    let nc = c
        .post(format!("{}/api/chat", ts.base))
        .json(&serde_json::json!({
            "model": "m1",
            "stream": false,
            "messages": [{"role": "user", "content": "x"}],
            "options": {"num_ctx": 8192},
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(nc.status(), 400);
    let nb: serde_json::Value = nc.json().await.unwrap();
    assert!(nb["error"].as_str().unwrap().contains("model_overrides"));

    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__router_mode_sessions_route_by_model() {
    let cfg = Config {
        router: true,
        ..Config::default()
    };
    let ts = start(cfg).await;
    let c = client();
    // warm the router
    let _: serde_json::Value = c
        .post(format!("{}/v1/chat/completions", ts.base))
        .json(&serde_json::json!({"model": "m1", "stream": false, "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let save: serde_json::Value = c
        .post(format!("{}/api/session", ts.base))
        .json(&serde_json::json!({"model": "m1", "action": "save", "filename": "r1"}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(save["status"], "ok", "{save}");
    assert!(ts.dirs.sessions_dir().join("m1").join("r1").exists());
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
    let req = c
        .post(format!("{}/v1/chat/completions", ts.base))
        .json(&body);
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
