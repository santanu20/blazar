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
    let _ = state.http_addr.set(("127.0.0.1".to_string(), port));
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
async fn e2e__num_ctx_refuses_oom_shape_with_q8_hint() {
    // I5 preflight, classic (f16) lane: the stub manifest lacks
    // --kv-unified, so the charge is full f16 KV — 500k ctx on the 24 GiB
    // stub card cannot fit and must refuse with the q8_0 teaching hint.
    // (The unified 512 MiB-floor branch is pinned at the core level in
    // unit__kv_unified_for__truth_table_for_offline_callers: adding the
    // flag to the stub --help would cascade into every spawn-argv
    // battery.)
    let ts = start(Config::default()).await;
    let c = client();
    let body = serde_json::json!({
        "model": "m1", "stream": false,
        "messages": [{"role": "user", "content": "a"}],
        "options": {"num_ctx": 500_000},
    });
    let r = c
        .post(format!("{}/api/chat", ts.base))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 400);
    let v: serde_json::Value = r.json().await.unwrap();
    let msg = format!("{}", v["error"]);
    assert!(msg.contains("num_ctx 500000"), "{msg}");
    assert!(msg.contains("q8_0"), "teaching hint present: {msg}");
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
        weight: 1,
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
async fn e2e__keep_alive_zero_unload_ping_idempotent() {
    let ts = start(Config::default()).await;
    let c = client();
    // The ollama unload idiom: {model, keep_alive: 0} with NO payload —
    // idempotent 200 even when the model was never loaded (a 400 here
    // breaks _unload_all-style harness sweeps).
    for (path, lane) in [("/api/generate", "response"), ("/api/chat", "message")] {
        let r: serde_json::Value = c
            .post(format!("{}{}", ts.base, path))
            .json(&serde_json::json!({"model": "m1", "keep_alive": 0}))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert!(r["done"] == true, "{lane} ping done: {r}");
        assert!(r.get(lane).is_some(), "{lane} ping shape: {r}");
    }
    // Prompt-less WITHOUT keep_alive 0 is still a hard 400 — inference
    // genuinely requires the field; only the release ping is exempt.
    let r = c
        .post(format!("{}/api/generate", ts.base))
        .json(&serde_json::json!({"model": "m1"}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 400, "no-prompt non-ping must fail fast");
    // Unknown model in a ping resolves like any request: 404.
    let r = c
        .post(format!("{}/api/generate", ts.base))
        .json(&serde_json::json!({"model": "no-such", "keep_alive": 0}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 404, "unknown ping model must 404");
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
    assert!(
        r.get("message").is_none(),
        "generate shape, not chat shape: {r}"
    );

    // system now rides the chat bus (ollama templated-generate parity).
    let with_system = serde_json::json!({
        "model": "m1", "prompt": "say hi", "system": "be terse", "stream": false
    });
    let r2: serde_json::Value = c
        .post(format!("{}/api/generate", ts.base))
        .json(&with_system)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(r2["done"] == true, "system generate works: {r2}");
    assert!(r2["response"].as_str().unwrap().contains("say hi"));

    // template/suffix stay rejected — the engine owns the template.
    let templated = serde_json::json!({"model": "m1", "prompt": "x", "template": "{{.System}}"});
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
async fn e2e__generate_images_and_streaming_chat_bus() {
    let ts = start(Config::default()).await;
    let c = client();
    // pdf_ocr-shaped payload: images[] (b64 of PNG magic + padding) +
    // system + options.
    let png_b64 = "iVBORw0KGgoAAAA";
    let vision = serde_json::json!({
        "model": "m1",
        "prompt": "describe",
        "system": "be terse",
        "images": [png_b64],
        "stream": false,
        "options": {"temperature": 0.2, "num_predict": 64}
    });
    let r: serde_json::Value = c
        .post(format!("{}/api/generate", ts.base))
        .json(&vision)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(r["done"] == true, "vision generate works: {r}");
    assert!(r["response"].as_str().unwrap().contains("describe"));

    // Bad image magic: fail fast 400 (never a silent text-only answer).
    let bad = serde_json::json!({
        "model": "m1", "prompt": "x", "images": ["AAAAAAAA"], "stream": false
    });
    let resp = c
        .post(format!("{}/api/generate", ts.base))
        .json(&bad)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
    let b: serde_json::Value = resp.json().await.unwrap();
    assert!(
        b["error"]
            .as_str()
            .unwrap()
            .contains("unsupported image format"),
        "{b}"
    );

    // Streaming generate (websearch shape): NDJSON response deltas ending
    // in a done:true line carrying counts.
    let stream_req = serde_json::json!({
        "model": "m1", "prompt": "stream me", "system": "sys", "stream": true, "think": false
    });
    let resp = c
        .post(format!("{}/api/generate", ts.base))
        .json(&stream_req)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    assert!(resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .is_some_and(|ct| ct.contains("ndjson")));
    let body = resp.text().await.unwrap();
    let mut saw_delta = false;
    let mut final_line: Option<serde_json::Value> = None;
    for line in body.lines().filter(|l| !l.is_empty()) {
        let v: serde_json::Value = serde_json::from_str(line).unwrap();
        if v["done"] == false {
            saw_delta = true;
            assert!(v.get("response").is_some(), "delta shape: {v}");
            assert!(v.get("message").is_none(), "generate lane: {v}");
        } else {
            final_line = Some(v);
        }
    }
    assert!(saw_delta, "stream produced deltas: {body}");
    let fin = final_line.expect("stream ends with a done:true line");
    assert!(fin["done"] == true);
    assert!(fin["done_reason"].as_str().is_some());
    assert!(
        fin["eval_count"].as_i64().is_some(),
        "counts ride final: {fin}"
    );
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__chat_message_images_translate_to_parts() {
    let ts = start(Config::default()).await;
    let c = client();
    // b64 of PNG magic + 4 zero bytes (decodes cleanly at 12 bytes).
    let png_b64 = "iVBORw0KGgoAAAA";
    let body = serde_json::json!({
        "model": "m1",
        "stream": false,
        "messages": [
            {"role": "user", "content": "what is this?", "images": [png_b64]}
        ]
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
    // The stub echoes the last user content — parts were forwarded (not
    // silently dropped): the text part survives in the echo.
    let content = r["message"]["content"].as_str().unwrap_or_default();
    assert!(
        content.contains("what is this?"),
        "vision parts forwarded: {r}"
    );
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
    // ollama-parity capability discovery (geokit vision + thinking
    // detection): m1 has no mmproj but its fixture template carries an
    // enable_thinking marker -> completion + thinking, never a vision lie.
    let caps = s["capabilities"].as_array().expect("capabilities array");
    assert!(caps.iter().any(|c| c == "completion"), "{s}");
    assert!(caps.iter().any(|c| c == "thinking"), "{s}");
    assert!(!caps.iter().any(|c| c == "vision"), "{s}");
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
        msg.contains("pallama whisper --install"),
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

    // file landed in the per-model sessions dir (path_safe adds the
    // stable FNV suffix — derive, never hardcode)
    let sess_dir = ts
        .dirs
        .sessions_dir()
        .join(pallama_core::profile::path_safe("m1"));
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
    assert!(ts
        .dirs
        .sessions_dir()
        .join(pallama_core::profile::path_safe("m1"))
        .join("r1")
        .exists());
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

#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__client_disconnect_frees_slot_ollama_lane() {
    // F29 pin: the ollama NDJSON lanes must hold accounting for the BODY
    // lifetime — dropping the response mid-stream frees the slot (the old
    // with_accounting future-bracket leaked the counter forever on abort).
    let ts = start_with(
        Config::default(),
        vec![("STUB_DELAY_MS".into(), "10000".into())],
    )
    .await;
    let c = client();
    let body = serde_json::json!({
        "model": "m1", "stream": true,
        "messages": [{"role": "user", "content": "slow"}],
    });
    let resp = c
        .post(format!("{}/api/chat", ts.base))
        .json(&body)
        .send()
        .await
        .unwrap();
    drop(resp); // client goes away mid-NDJSON
    tokio::time::sleep(Duration::from_millis(400)).await;
    let ps = ts.state.sup.ps();
    assert!(!ps.is_empty());
    assert_eq!(
        ps[0].in_flight, 0,
        "ollama lane slot freed after disconnect"
    );
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

/// Scoped surfaces (/props, /slots, /v1/stream*, /v1/streams/lookup)
/// forward to the single hot child without any explicit target.
#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__scoped_routes_single_child_passthrough() {
    let ts = start(Config::default()).await;
    let c = client();
    // Warm m1 so exactly one child is hot.
    let _: serde_json::Value = c
        .post(format!("{}/v1/chat/completions", ts.base))
        .json(&serde_json::json!({"model": "m1", "stream": false,
            "messages": [{"role": "user", "content": "warm"}]}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();

    let props: serde_json::Value = c
        .get(format!("{}/props", ts.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        props["model_alias"], "m1",
        "single hot child resolved: {props}"
    );

    // POST /props carries settings through (stub echoes them back).
    let posted: serde_json::Value = c
        .post(format!("{}/props", ts.base))
        .json(&serde_json::json!({"temperature": 0.1}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(posted["user_props"]["temperature"], 0.1);

    let slots: serde_json::Value = c
        .get(format!("{}/slots", ts.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(
        slots[0]["id"].as_u64().is_some(),
        "upstream bare-array shape: {slots}"
    );

    let streams: serde_json::Value = c
        .get(format!("{}/v1/stream", ts.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(streams["streams"][0]["stream_id"], "s0", "{streams}");

    let del = c
        .delete(format!("{}/v1/stream?stream_id=s0", ts.base))
        .send()
        .await
        .unwrap();
    assert_eq!(del.status(), 200);

    // /v1/streams/lookup carries model in its body (chat-shaped).
    let lookup: serde_json::Value = c
        .post(format!("{}/v1/streams/lookup", ts.base))
        .json(&serde_json::json!({"model": "m1", "prompt": "warm"}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(lookup["matched_prompt"], "warm", "{lookup}");
    assert_eq!(lookup["model_alias"], "m1");
    ts.state.sup.shutdown_all().await.unwrap();
}

/// With zero hot children or an ambiguous multi-child state the scoped
/// surfaces demand an explicit target (400 teaching error); the
/// X-Pallama-Model header disambiguates.
#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__scoped_routes_require_target() {
    let ts = start(Config::default()).await;
    let c = client();
    // No child loaded yet: no implicit resolution possible.
    let none = c.get(format!("{}/props", ts.base)).send().await.unwrap();
    assert_eq!(none.status(), 400);
    let none_body: serde_json::Value = none.json().await.unwrap();
    let msg = none_body["error"]["message"]
        .as_str()
        .or_else(|| none_body["error"].as_str())
        .unwrap_or_default();
    assert!(
        msg.contains("X-Pallama-Model"),
        "teaching error: {none_body}"
    );

    // Warm two children: single-child shortcut no longer applies.
    for m in ["m1", "m2"] {
        let _: serde_json::Value = c
            .post(format!("{}/v1/chat/completions", ts.base))
            .json(&serde_json::json!({"model": m, "stream": false,
                "messages": [{"role": "user", "content": "warm"}]}))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
    }
    let amb = c.get(format!("{}/props", ts.base)).send().await.unwrap();
    assert_eq!(amb.status(), 400, "ambiguous without target");

    // Header resolves; query param resolves too.
    let hdr: serde_json::Value = c
        .get(format!("{}/props", ts.base))
        .header("x-pallama-model", "m2")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(hdr["model_alias"], "m2", "{hdr}");
    let q: serde_json::Value = c
        .get(format!("{}/slots?model=m1", ts.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(q.as_array().is_some_and(|a| !a.is_empty()), "{q}");
    ts.state.sup.shutdown_all().await.unwrap();
}

/// /v1/reranking is the Jina-style alias of /v1/rerank: body carries the
/// model, response passes through untouched.
#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__v1_reranking_alias() {
    let ts = start(Config::default()).await;
    let c = client();
    let r: serde_json::Value = c
        .post(format!("{}/v1/reranking", ts.base))
        .json(&serde_json::json!({
            "model": "m1",
            "query": "q",
            "documents": ["short", "a much longer document"],
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let results = r["results"].as_array().expect("results array");
    assert!(!results.is_empty(), "{r}");
    // Stub ranks by length: longer doc first.
    assert_eq!(results[0]["index"], 1, "{r}");
    ts.state.sup.shutdown_all().await.unwrap();
}

/// Child-auth hardening (auto: TCP children get `--api-key-file`): the
/// gateway stamps every child-bound call, so normal lanes keep working,
/// while DIRECT access to the child port is 401 without the secret and
/// 200 with it. Proves both the choke-point coverage (a missed site
/// would fail loudly here) and the bypass closure.
#[allow(non_snake_case)]
#[tokio::test]
async fn e2e__child_auth__gateway_stamps_and_direct_rejected() {
    let ts = start(Config::default()).await;
    let c = client();
    // Warm m1; any successful child-bound lane proves the stamping.
    let chat: serde_json::Value = c
        .post(format!("{}/v1/chat/completions", ts.base))
        .json(&serde_json::json!({"model": "m1", "stream": false,
            "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(chat["choices"].is_array(), "gateway lane served: {chat}");

    // Keyfile minted next to the pidfile, 0600, plm_-prefixed secret.
    let keyfile = ts.dirs.run_dir().join("m1.apikey");
    let secret = std::fs::read_to_string(&keyfile)
        .unwrap_or_else(|e| panic!("keyfile {}: {e}", keyfile.display()));
    assert!(
        secret.starts_with("plm_") && secret.len() >= 32,
        "{secret:?}"
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(&keyfile).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600, "keyfile must be owner-only");
    }

    // Direct child access: blocked without the secret, open with it.
    let ps: serde_json::Value = c
        .get(format!("{}/api/ps", ts.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let endpoint = ps["models"][0]["pallama_endpoint"]
        .as_str()
        .expect("ps endpoint")
        .to_string();
    let direct = reqwest::Client::new();
    let no_key = direct
        .get(format!("http://{endpoint}/v1/models"))
        .send()
        .await
        .unwrap();
    assert_eq!(no_key.status(), 401, "unauthenticated direct access");
    let with_key = direct
        .get(format!("http://{endpoint}/v1/models"))
        .bearer_auth(&secret)
        .send()
        .await
        .unwrap();
    assert_eq!(with_key.status(), 200, "bearer-stamped direct access");

    // Teardown removes the keyfile alongside the pidfile.
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__remote_failover_circuit_and_lb() {
    // Remote target = a second healthy pallama server (stub child m1).
    let target = start(Config::default()).await;
    // Main server: pool "r" = [dead (port 1 refuses instantly), target].
    // Tie-break min_by_key picks the FIRST member while both are idle, so
    // requests 1..3 hit the dead one, rack up consecutive failures and mark
    // it down (circuit open); request 4 must fail over to the target.
    let cfg = Config {
        remotes: vec![
            pallama_core::config::Remote {
                name: "r".into(),
                url: "http://127.0.0.1:1".into(),
                key: String::new(),
            },
            pallama_core::config::Remote {
                name: "r".into(),
                url: target.base.clone(),
                key: String::new(),
            },
        ],
        ..Config::default()
    };
    let ts = start(cfg).await;
    let body = r#"{"model":"r:m1","messages":[{"role":"user","content":"hi"}],"max_tokens":4}"#;
    for i in 1..=3u32 {
        let resp = client()
            .post(format!("{}/v1/chat/completions", ts.base))
            .header("content-type", "application/json")
            .body(body)
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 502, "dead member, attempt {i}");
    }
    // Circuit open on the dead member: the next request routes to target.
    let resp = client()
        .post(format!("{}/v1/chat/completions", ts.base))
        .header("content-type", "application/json")
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "failover to healthy member");
    let v: serde_json::Value = resp.json().await.unwrap();
    assert!(
        v["choices"][0]["message"]["content"]
            .as_str()
            .unwrap()
            .contains("stub:m1:"),
        "answered by target's stub child: {v:?}"
    );
    ts.state.sup.shutdown_all().await.unwrap();
    target.state.sup.shutdown_all().await.unwrap();
}

/// Opt-out lane: `child_auth = false` keeps children open (legacy
/// behavior) — no keyfile, direct access unauthenticated.
#[allow(non_snake_case)]
#[tokio::test]
async fn e2e__child_auth__disabled_keeps_children_open() {
    let ts = start(Config {
        child_auth: Some(false),
        ..Config::default()
    })
    .await;
    let c = client();
    let chat: serde_json::Value = c
        .post(format!("{}/v1/chat/completions", ts.base))
        .json(&serde_json::json!({"model": "m1", "stream": false,
            "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(chat["choices"].is_array(), "gateway lane served: {chat}");
    assert!(!ts.dirs.run_dir().join("m1.apikey").exists());
    let ps: serde_json::Value = c
        .get(format!("{}/api/ps", ts.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let endpoint = ps["models"][0]["pallama_endpoint"]
        .as_str()
        .unwrap()
        .to_string();
    let resp = reqwest::Client::new()
        .get(format!("http://{endpoint}/v1/models"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "opt-out keeps the child open");
    ts.state.sup.shutdown_all().await.unwrap();
}

// ---- D7 Anthropic /v1/messages native translate lane ----

/// Non-stream: Claude-dialect request in, Anthropic message shape out.
#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__anthropic_messages_translate_non_stream() {
    let ts = start(Config::default()).await;
    let c = client();
    let v: serde_json::Value = c
        .post(format!("{}/v1/messages", ts.base))
        .json(&serde_json::json!({
            "model": "m1",
            "max_tokens": 64,
            "system": "be terse",
            "messages": [{"role": "user", "content": "hi there"}],
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(v["type"], "message", "body: {v}");
    assert_eq!(v["role"], "assistant");
    assert!(
        v["id"].as_str().unwrap().starts_with("msg_"),
        "anthropic id namespace: {}",
        v["id"]
    );
    assert_eq!(v["content"][0]["type"], "text");
    assert_eq!(v["content"][0]["text"], "stub:m1:hi there");
    assert_eq!(v["stop_reason"], "end_turn");
    assert!(v["usage"]["input_tokens"].as_i64().unwrap() > 0);
    assert!(v["usage"]["output_tokens"].as_i64().unwrap() > 0);

    // Missing model -> Anthropic error envelope, not OpenAI's.
    let e: serde_json::Value = c
        .post(format!("{}/v1/messages", ts.base))
        .json(&serde_json::json!({"max_tokens": 8, "messages": []}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(e["type"], "error");
    assert_eq!(e["error"]["type"], "invalid_request_error");
    ts.state.sup.shutdown_all().await.unwrap();
}

/// Stream: `OpenAI` child SSE -> Anthropic event family.
#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__anthropic_messages_translate_stream() {
    let ts = start(Config::default()).await;
    let c = client();
    let body = c
        .post(format!("{}/v1/messages", ts.base))
        .json(&serde_json::json!({
            "model": "m1",
            "max_tokens": 32,
            "stream": true,
            "messages": [{"role": "user", "content": "hello"}],
        }))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(body.contains("event: message_start"), "body: {body}");
    assert!(body.contains("event: content_block_start"));
    assert!(body.contains("event: content_block_delta"));
    assert!(body.contains("\"text_delta\""));
    assert!(body.contains("event: content_block_stop"));
    assert!(body.contains("event: message_delta"));
    assert!(body.contains("event: message_stop"));
    ts.state.sup.shutdown_all().await.unwrap();
}

/// Tool-call child response -> `tool_use` blocks in the Anthropic shape.
#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__anthropic_messages_translate_tool_use() {
    let ts = start_with(
        Config::default(),
        vec![("STUB_BAD_TOOL_ARGS".to_string(), "1".to_string())],
    )
    .await;
    let c = client();
    let v: serde_json::Value = c
        .post(format!("{}/v1/messages", ts.base))
        .json(&serde_json::json!({
            "model": "m1",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "call the tool"}],
            "tools": [{"name": "echo", "description": "echoes",
                       "input_schema": {"type": "object", "properties": {}}}],
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let blocks = v["content"].as_array().unwrap();
    let tu = blocks
        .iter()
        .find(|b| b["type"] == "tool_use")
        .expect("tool_use block present: {blocks:?}");
    assert_eq!(tu["name"], "echo");
    assert_eq!(tu["input"], serde_json::json!({}), "broken json -> {{}}");
    ts.state.sup.shutdown_all().await.unwrap();
}

/// Batch API end-to-end: upload JSONL, run batch, poll to completion,
/// download output file with per-custom_id results.
#[tokio::test]
#[allow(non_snake_case)]
#[allow(clippy::too_many_lines)] // one linear scenario, assertions inline
async fn e2e__batch_jsonl_end_to_end() {
    let ts = start(Config::default()).await;
    let c = client();
    let boundary = "pallamaBatchTest7f2a";
    let line1 = r#"{"custom_id":"task-1","body":{"model":"m1","messages":[{"role":"user","content":"hello batch"}],"max_tokens":5}}"#;
    let line2 = r#"{"custom_id":"task-2","body":{"model":"m1","messages":[{"role":"user","content":"second line"}],"max_tokens":5}}"#;
    let file_body = line1.to_string() + "\n" + line2 + "\n";
    let multipart = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n\
         --{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"reqs.jsonl\"\r\n\
         Content-Type: application/jsonl\r\n\r\n{file_body}\r\n--{boundary}--\r\n"
    );
    let file: serde_json::Value = c
        .post(format!("{}/v1/files", ts.base))
        .header(
            "content-type",
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(multipart)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let file_id = file["id"].as_str().unwrap().to_string();
    assert!(file_id.starts_with("file-"), "file id: {file:?}");
    assert_eq!(file["object"], "file");
    assert_eq!(file["purpose"], "batch");

    // Unknown file -> 404; wrong endpoint -> 400 teaching error.
    let bad = c
        .post(format!("{}/v1/batches", ts.base))
        .json(
            &serde_json::json!({"input_file_id": "file-none", "endpoint": "/v1/chat/completions"}),
        )
        .send()
        .await
        .unwrap();
    assert_eq!(bad.status(), 404);
    let bad = c
        .post(format!("{}/v1/batches", ts.base))
        .json(&serde_json::json!({"input_file_id": file_id, "endpoint": "/v1/embeddings"}))
        .send()
        .await
        .unwrap();
    assert_eq!(bad.status(), 400);

    let batch: serde_json::Value = c
        .post(format!("{}/v1/batches", ts.base))
        .json(&serde_json::json!({"input_file_id": file_id, "endpoint": "/v1/chat/completions"}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let batch_id = batch["id"].as_str().unwrap().to_string();
    assert!(batch_id.starts_with("batch-"), "batch id: {batch:?}");
    assert_eq!(batch["status"], "in_progress");
    assert_eq!(batch["request_counts"]["total"], 2);

    // Poll to completion (sequential worker; stub child is instant).
    let mut final_batch = None;
    for _ in 0..100 {
        let v: serde_json::Value = c
            .get(format!("{}/v1/batches/{batch_id}", ts.base))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        if v["status"] == "completed" || v["status"] == "failed" || v["status"] == "cancelled" {
            final_batch = Some(v);
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(250)).await;
    }
    let done = final_batch.expect("batch reaches a terminal status");
    assert_eq!(done["status"], "completed", "batch: {done:?}");
    assert_eq!(done["request_counts"]["completed"], 2);
    assert_eq!(done["request_counts"]["failed"], 0);

    let out_id = done["output_file_id"].as_str().unwrap().to_string();
    assert!(out_id.starts_with("file-"), "output file id: {done:?}");
    let content = c
        .get(format!("{}/v1/files/{out_id}/content", ts.base))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    let rows: Vec<serde_json::Value> = content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(rows.len(), 2, "two output lines: {content}");
    let ids: Vec<&str> = rows
        .iter()
        .map(|r| r["custom_id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, vec!["task-1", "task-2"], "custom_id order preserved");
    for r in &rows {
        assert_eq!(r["response"]["status_code"], 200, "row: {r:?}");
        let body = &r["response"]["body"];
        assert!(
            body["choices"][0]["message"]["content"]
                .as_str()
                .unwrap()
                .contains("stub:m1:"),
            "stub reply present: {r:?}"
        );
    }
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__llamacpp_only_gate__non_llamacpp_kinds_get_teaching_400() {
    // audit GAP-3: the gate used to fire only for mistralrs — an sglang
    // child proxied /props, /slots, /api/session-save class requests to a
    // bare upstream 404. Every NON-llamacpp kind must get the teaching
    // 400 naming its kind; llamacpp passes the gate untouched. The gates
    // read the store per request (only proxy's model-rewrite memoizes
    // kind — F34 — and never these), so a post-boot row flip is live.
    let ts = start(Config::default()).await;
    let c = client();

    // llamacpp active: the gate must stay silent. /props then fails on
    // model resolution (no child, no header) — a DIFFERENT error that
    // must not carry the gate's teaching.
    let r = c.get(format!("{}/props", ts.base)).send().await.unwrap();
    let text = r.text().await.unwrap();
    assert!(
        !text.contains("llama-server-only"),
        "llamacpp must pass the gate: {text}"
    );

    let flip = |tag: &str, kind: pallama_core::engine_kind::EngineKind| {
        ts.state.with_store(|s| {
            s.upsert_engine(&pallama_core::EngineRow {
                tag: tag.into(),
                asset: "stub".into(),
                sha256: "flip".into(),
                installed_at: 2,
                active: true,
                manifest: "{}".into(),
                kind,
            })
            .unwrap();
            s.set_active_engine(tag).unwrap();
        })
    };

    for (tag, kind, name) in [
        (
            "sglang-t",
            pallama_core::engine_kind::EngineKind::Sglang,
            "sglang",
        ),
        (
            "mistralrs-t",
            pallama_core::engine_kind::EngineKind::MistralRs,
            "mistralrs",
        ),
    ] {
        flip(tag, kind);
        // openai surface: /props and friends teach with the kind named.
        let r = c.get(format!("{}/props", ts.base)).send().await.unwrap();
        assert_eq!(r.status(), 400, "{name} /props");
        let v: serde_json::Value = r.json().await.unwrap();
        let msg = v["error"]["message"].as_str().unwrap_or_default();
        assert!(
            msg.contains("llama-server-only") && msg.contains(name),
            "{name} /props teaching: {msg}"
        );
        // ollama surface: slot KV checkpoint actions teach too...
        let r = c
            .post(format!("{}/api/session", ts.base))
            .json(&serde_json::json!({"action": "save", "session": "s"}))
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 400, "{name} session save");
        let v: serde_json::Value = r.json().await.unwrap();
        let msg = v.to_string();
        assert!(
            msg.contains("llama-server-only") && msg.contains(name),
            "{name} session save teaching: {msg}"
        );
        // ...but `close` stays open — it releases a gateway-side pin,
        // no slot surface involved (fails on unknown session instead).
        let r = c
            .post(format!("{}/api/session", ts.base))
            .json(&serde_json::json!({"action": "close", "session": "nope"}))
            .send()
            .await
            .unwrap();
        let v: serde_json::Value = r.json().await.unwrap();
        assert!(
            !v.to_string().contains("llama-server-only"),
            "{name} close must bypass the gate: {v}"
        );
    }

    ts.state.sup.shutdown_all().await.unwrap();
}
