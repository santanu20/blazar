//! Gateway end-to-end: real axum server over a real supervisor spawning
//! stub-llama-server children. Exercises both API surfaces, translation,
//! auth, `keep_alive` eviction, `num_ctx` restart, and cancellation.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use blazar_core::hardware::{GpuInfo, Hardware};
use blazar_core::store::Store;
use blazar_core::{BlazarDirs, Config};
use blazar_gateway::state::AppState;
use blazar_gateway::{router, translate as tr};
use blazar_runtime::EventBus;
use blazar_runtime::{LlamaCppEngine, Supervisor};

/// Locate the stub binary by walking up from this test executable to the
/// cargo target dir (works under deps/, debug/, release/).
fn stub_bin() -> PathBuf {
    let exe = std::env::current_exe().expect("test exe path");
    // Windows binaries carry the .exe suffix cargo appends at build time.
    let name = if cfg!(windows) {
        "stub-llama-server.exe"
    } else {
        "stub-llama-server"
    };
    for dir in exe.ancestors() {
        let candidate = dir.join(name);
        if candidate.exists() {
            return candidate;
        }
        if dir.file_name().is_some_and(|n| n == "target") {
            break;
        }
    }
    panic!("stub-llama-server not found near {}", exe.display());
}

// Edition 2024 makes env mutation unsafe (std demands no concurrent
// reader/writer thread). These tests run one-per-process under nextest and
// only the spawning test thread and its stub children touch the STUB_*
// knobs — the condition std requires.
fn set_env(key: &str, value: &str) {
    #[expect(unsafe_code)]
    unsafe {
        std::env::set_var(key, value);
    }
}

fn remove_env(key: &str) {
    #[expect(unsafe_code)]
    unsafe {
        std::env::remove_var(key);
    }
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
    pub dirs: BlazarDirs,
    state: Arc<AppState>,
    _sup_reaper: tokio::task::JoinHandle<()>,
}

async fn start(config: Config) -> TestServer {
    start_with(config, Vec::new()).await
}

async fn start_with(config: Config, child_env: Vec<(String, String)>) -> TestServer {
    let tmp = tempfile::tempdir().unwrap();
    let dirs = BlazarDirs {
        config_dir: tmp.path().join("c"),
        data_dir: tmp.path().join("d"),
    };
    dirs.ensure().unwrap();
    let gguf = dirs.models_dir().join("m1-q4_k_m.gguf");
    write_gguf(&gguf);
    let store = Store::open(&dirs).unwrap();
    store
        .upsert_model(&blazar_core::ModelRow {
            name: "m1".into(),
            repo: "o/m1".into(),
            quant: "Q4_K_M".into(),
            path: gguf.display().to_string(),
            bytes: 500_000_000,
            sha256: None,
            mmproj_path: None,
            components: vec![],
            shards: 1,
            arch: Some("qwen3".into()),
            params: Some(0.5),
            ctx_train: Some(40_960),
            pulled_at: 1,
            last_used_at: 1,
        })
        .unwrap();
    // Second model for router-mode tests (unused by single-model tests).
    let gguf2 = dirs.models_dir().join("m2-q4_k_m.gguf");
    write_gguf(&gguf2);
    store
        .upsert_model(&blazar_core::ModelRow {
            name: "m2".into(),
            repo: "o/m2".into(),
            quant: "Q4_K_M".into(),
            path: gguf2.display().to_string(),
            bytes: 500_000_000,
            sha256: None,
            mmproj_path: None,
            components: vec![],
            shards: 1,
            arch: Some("qwen3".into()),
            params: Some(0.5),
            ctx_train: Some(40_960),
            pulled_at: 1,
            last_used_at: 1,
        })
        .unwrap();

    let mut engine =
        LlamaCppEngine::new(blazar_runtime::probe_manifest(&stub_bin(), "stub").unwrap());
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

/// m1's vision-capable sibling: same row plus a projector sidecar. The
/// admission gate 400s image requests against a sidecar-less model, so
/// vision-pipeline e2e tests ride this row (the stub child never reads
/// the sidecar path).
fn upsert_vision_sibling(ts: &TestServer) {
    ts.state
        .with_store(|s| {
            let mut row = s.get_model("m1").unwrap().expect("m1 fixture row");
            row.name = "m1v".into();
            row.mmproj_path = Some("mmproj.gguf".into());
            s.upsert_model(&row).unwrap();
        })
        .unwrap();
}

fn client() -> reqwest::Client {
    blazar_core::tls::ensure_tls_provider();
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
    // The name marker is doctor's port-identity signal — pin it.
    assert_eq!(v["name"].as_str(), Some("blazar"));
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
async fn e2e__ollama_delete_accepts_the_tags_rendered_name() {
    let ts = start(Config::default()).await;
    let c = client();
    // GET /api/tags renders `name:quant`; a client deleting by that exact
    // string must succeed. The raw store-key probe this replaces 404'd on
    // the tags-displayed name (store key is the bare `m1`).
    let status = c
        .post(format!("{}/api/delete", ts.base))
        .json(&serde_json::json!({"model": "m1:q4_k_m"}))
        .send()
        .await
        .unwrap()
        .status();
    assert_eq!(status, 200);
    // Row and file are gone; the untouched sibling stays.
    let t: serde_json::Value = c
        .get(format!("{}/api/tags", ts.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let names: Vec<&str> = t["models"]
        .as_array()
        .map(|a| a.iter().filter_map(|m| m["name"].as_str()).collect())
        .unwrap_or_default();
    assert!(!names.contains(&"m1:q4_k_m"), "deleted row gone: {names:?}");
    assert!(names.contains(&"m2:q4_k_m"), "sibling untouched: {names:?}");
    // Already-deleted (by any colon form) and unknown names still teach
    // 404 through the resolver, which carries the suggestion ladder.
    for probe in ["m1:q4_k_m", "m1", "m1:latest", "no-such-model"] {
        let status = c
            .post(format!("{}/api/delete", ts.base))
            .json(&serde_json::json!({"model": probe}))
            .send()
            .await
            .unwrap()
            .status();
        assert_eq!(status, 404, "probe {probe:?}");
    }
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__ollama_evict_accepts_the_tags_rendered_name() {
    let ts = start(Config::default()).await;
    let c = client();
    // GET /api/tags renders `name:quant`; a client evicting by that exact
    // string must reach the supervisor. The raw passthrough this replaces
    // no-op'd 200 on the tags form — instance keys hold bare store names,
    // so `model_of_key(k) == model` never matched and the child kept
    // serving (live-repro'd: bge-reranker-v2-m3:q8_0 evicted, still ready).
    let status = c
        .post(format!("{}/api/evict", ts.base))
        .json(&serde_json::json!({"model": "m1:q4_k_m"}))
        .send()
        .await
        .unwrap()
        .status();
    assert_eq!(status, 200, "tags form resolves and unloads");
    // Unloading a not-loaded model stays a 200 no-op (ollama keep_alive:0
    // semantics) — but only for names that RESOLVE. Garbage teaches 404
    // through the resolver instead of silently ok-ing.
    let status = c
        .post(format!("{}/api/evict", ts.base))
        .json(&serde_json::json!({"model": "m1"}))
        .send()
        .await
        .unwrap()
        .status();
    assert_eq!(status, 200, "bare form resolves too");
    for probe in ["no-such-model", "zz:q4_k_m"] {
        let status = c
            .post(format!("{}/api/evict", ts.base))
            .json(&serde_json::json!({"model": probe}))
            .send()
            .await
            .unwrap()
            .status();
        assert_eq!(status, 404, "unresolvable {probe:?} teaches 404");
    }
    let status = c
        .post(format!("{}/api/evict", ts.base))
        .json(&serde_json::json!({"model": 42}))
        .send()
        .await
        .unwrap()
        .status();
    assert_eq!(status, 400, "non-string model is a 400");
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__ollama_evict_unresolvable_name_finds_the_whisper_lane() {
    let ts = start(Config::default()).await;
    let c = client();
    // The whisper child keys by its size name ("base"), which is never
    // a store row: before the lane fallback, evicting "base" 404'd with
    // a levenshtein guess ("laya") while the audio child stayed loaded.
    // Plant a live sleeper as the lane child, then evict by size name.
    let child = tokio::process::Command::new("sleep")
        .arg("60")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("spawn sleep");
    let pid = child.id().expect("pid");
    *ts.state.whisper.child.lock().await = Some(blazar_runtime::whisper::WhisperChild {
        child,
        port: 1,
        loaded: "base".into(),
        last_used: tokio::time::Instant::now(),
    });
    // While the lane is live but the name does NOT match: the 404 must
    // teach with the live lanes listed, not a bare store suggestion.
    let resp = c
        .post(format!("{}/api/evict", ts.base))
        .json(&serde_json::json!({"model": "no-such-model"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
    let v: serde_json::Value = resp.json().await.unwrap();
    let msg = v["error"].as_str().expect("message");
    assert!(msg.contains("whisper:base"), "404 names live lanes: {msg}");
    // The size name resolves to the lane and tears it down for real.
    let resp = c
        .post(format!("{}/api/evict", ts.base))
        .json(&serde_json::json!({"model": "base"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "whisper size name resolves to the lane");
    let v: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(v["lane"], "whisper");
    assert_eq!(v["model"], "base");
    assert!(ts.state.whisper.status().await.is_none(), "slot drained");
    assert!(
        !std::path::Path::new(&format!("/proc/{pid}")).exists(),
        "whisper child {pid} leaked past evict"
    );
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__num_ctx_header_resolves_the_tags_rendered_name() {
    let ts = start(Config::default()).await;
    let c = client();
    // X-Blazar-Num-Ctx (and X-Blazar-Spec) feed the RAW body model into
    // apply_num_ctx/apply_spec; the tags-rendered alias must behave like
    // the exact name. The raw-key probe this replaces silently skipped
    // the KV preflight AND queued a phantom ctx under the display alias.
    // An absurd ctx makes resolution observable: the preflight refuses
    // loudly through the resolver instead of no-op'ing.
    let chat = |model: &str| {
        serde_json::json!({
            "model": model,
            "messages": [{"role": "user", "content": "hi"}],
        })
    };
    for model in ["m1", "m1:q4_k_m"] {
        let resp = c
            .post(format!("{}/v1/chat/completions", ts.base))
            .header("x-blazar-num-ctx", "999999999999")
            .json(&chat(model))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 400, "model {model:?}: preflight must fire");
        let body: serde_json::Value = resp.json().await.unwrap();
        let msg = body.to_string();
        assert!(
            msg.contains("cannot fit the GPU"),
            "model {model:?}: KV refuse teaching, got {msg}"
        );
    }
    // Unknown names fail fast from the header path with the ladder's
    // teaching instead of queueing a phantom ctx.
    let resp = c
        .post(format!("{}/v1/chat/completions", ts.base))
        .header("x-blazar-num-ctx", "8192")
        .json(&chat("no-such-model"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__n_choices_engine_ignoring_n_gets_a_teaching_502() {
    // Default stub returns ONE choice no matter what was asked: the
    // gateway must detect "lane ignored n" and refuse loudly instead of
    // letting a silent 1-of-N answer through. n=2 is the llamacpp lane
    // ceiling, so this is the largest n that still reaches the engine.
    let ts = start(Config::default()).await;
    let c = client();
    let resp = c
        .post(format!("{}/v1/chat/completions", ts.base))
        .json(&serde_json::json!({
            "model": "m1",
            "messages": [{"role": "user", "content": "pick one"}],
            "n": 2,
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 502);
    let body: serde_json::Value = resp.json().await.unwrap();
    let msg = body.to_string();
    assert!(
        msg.contains("asked n=2 choices") && msg.contains("engine returned 1"),
        "under-count teaching must name both numbers, got {msg}"
    );
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__n_choices_honored_passes_through_with_all_choices() {
    set_env("STUB_CHOICE_COUNT", "2");
    let ts = start(Config::default()).await;
    let c = client();
    let r: serde_json::Value = c
        .post(format!("{}/v1/chat/completions", ts.base))
        .json(&serde_json::json!({
            "model": "m1",
            "messages": [{"role": "user", "content": "variants"}],
            "n": 2,
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let choices = r["choices"].as_array().expect("choices array");
    assert_eq!(choices.len(), 2, "both choices must arrive");
    let indexes: Vec<i64> = choices
        .iter()
        .map(|ch| ch["index"].as_i64().unwrap())
        .collect();
    assert_eq!(indexes, vec![0, 1]);
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__n_choices_out_of_range_fails_fast_unbilled() {
    let ts = start(Config::default()).await;
    let c = client();
    let resp = c
        .post(format!("{}/v1/chat/completions", ts.base))
        .json(&serde_json::json!({
            "model": "m1",
            "messages": [{"role": "user", "content": "too many"}],
            "n": 9,
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
    let body: serde_json::Value = resp.json().await.unwrap();
    let msg = body.to_string();
    assert!(
        msg.contains("n must be an integer in 2..=8") || msg.contains("1..=8"),
        "range teaching must name the cap, got {msg}"
    );
    // Strictness: a float or string n is a contract violation, not a
    // silent default-to-1.
    for bad in [serde_json::json!(2.0), serde_json::json!("2")] {
        let resp = c
            .post(format!("{}/v1/chat/completions", ts.base))
            .json(&serde_json::json!({
                "model": "m1",
                "messages": [{"role": "user", "content": "strict"}],
                "n": bad,
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 400, "n={bad} must be rejected");
    }
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__systemone_is_llamacpp_only_and_teaches_on_other_lanes() {
    // TypeSafe System One is a llama-server child surface; with a non-llamacpp
    // lane active the gate must teach (naming the kind), and with llamacpp
    // active the gate stays silent (any later error must not carry the
    // teaching) — mirrors the /props gate pin.
    let ts = start(Config::default()).await;
    let c = client();
    let body = serde_json::json!({
        "state": "The build is green and tests pass.",
        "questions": { "ship": { "type": "choice", "instructions": "Decide.",
            "criteria": { "yes": "Ship it.", "no": "Hold." } } }
    });

    let r = c
        .post(format!("{}/v1/systemone", ts.base))
        .header("content-type", "application/json")
        .json(&body)
        .send()
        .await
        .unwrap();
    let text = r.text().await.unwrap();
    assert!(
        !text.contains("llama-server-only"),
        "llamacpp must pass the gate: {text}"
    );

    ts.state
        .with_store(|s| {
            s.upsert_engine(&blazar_core::EngineRow {
                tag: "sglang-t".into(),
                asset: "stub".into(),
                sha256: "flip".into(),
                installed_at: 2,
                active: true,
                manifest: "{}".into(),
                kind: blazar_core::engine_kind::EngineKind::Sglang,
            })
            .unwrap();
            s.set_active_engine("sglang-t").unwrap();
        })
        .unwrap();
    let r = c
        .post(format!("{}/v1/systemone", ts.base))
        .header("content-type", "application/json")
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 400);
    let v: serde_json::Value = r.json().await.unwrap();
    let msg = v["error"]["message"].as_str().unwrap_or_default();
    assert!(
        msg.contains("llama-server-only") && msg.contains("sglang"),
        "sglang systemone teaching: {msg}"
    );
}

#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__n_choices_lane_ceiling_teaches_before_the_engine_can() {
    // The plane-wide range admits n=3, but the llamacpp lane (the stub)
    // serves at most 2. The gateway must 400 in its own voice before
    // the engine's raw "Value must be between 1 <= value <= 2" leaks.
    let ts = start(Config::default()).await;
    let c = client();
    let resp = c
        .post(format!("{}/v1/chat/completions", ts.base))
        .json(&serde_json::json!({
            "model": "m1",
            "messages": [{"role": "user", "content": "three ways"}],
            "n": 3,
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
    let body: serde_json::Value = resp.json().await.unwrap();
    let msg = body.to_string();
    assert!(
        msg.contains("exceeds this lane's ceiling"),
        "lane teaching must be the gateway's own voice, got {msg}"
    );
    assert!(
        msg.contains("at most 2 choices"),
        "must name the llamacpp cap, got {msg}"
    );
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__openai_chat_nonstream_and_stream() {
    // Pin the local-lane framing contract: the gateway must forward an exact
    // Content-Length, never chunked (mlx_lm.server class children 411 on it).
    set_env("STUB_REJECT_CHUNKED", "1");
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

/// `child_transport = "unix"` (upstream #28690 surface): the whole lane —
/// spawn argv (`--host <path>.sock`, no --port), stub bind, readiness,
/// proxy forward, reply — rides the socket, and teardown reaps the file.
#[tokio::test]
#[cfg(unix)]
#[allow(non_snake_case)]
async fn e2e__openai_chat_over_unix_socket_transport() {
    let config = Config {
        child_transport: "unix".into(),
        ..Config::default()
    };
    let ts = start(config).await;
    let c = client();

    let r: serde_json::Value = c
        .post(format!("{}/v1/chat/completions", ts.base))
        .json(&serde_json::json!({
            "model": "m1",
            "messages": [{"role": "user", "content": "unix lane"}],
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(r["choices"][0]["message"]["content"], "stub:m1:unix lane");

    // The child really is socket-backed: exactly one live .sock in the
    // run dir, named for the sanitized instance key.
    let live_socks = |dir: &std::path::Path| -> Vec<std::fs::DirEntry> {
        std::fs::read_dir(dir)
            .unwrap()
            .filter_map(Result::ok)
            .filter(|e| e.path().extension().is_some_and(|x| x == "sock"))
            .collect()
    };
    let socks = live_socks(&ts.dirs.run_dir());
    assert_eq!(socks.len(), 1, "one socket per live child");
    assert!(socks[0].path().to_string_lossy().contains("m1"));

    // Teardown unlinks the socket with the child (H19: acquire/release).
    ts.state.sup.shutdown_all().await.unwrap();
    assert!(
        live_socks(&ts.dirs.run_dir()).is_empty(),
        "socket must die with the child"
    );
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
    // I5 preflight, beyond-the-ladder lane: the stub manifest lacks
    // --kv-unified, so the charge is full f16 KV — 2M ctx on the 24 GiB
    // stub card cannot fit at ANY ladder rung (f16/q8_0/q4_0) and must
    // refuse with the q8_0 teaching hint. (The 500k shape one rung
    // below now SERVES via the ladder — pinned in
    // e2e__num_ctx_ladder_serves_q4_hostable_shape.)
    // (The unified 512 MiB-floor branch is pinned at the core level in
    // unit__kv_unified_for__truth_table_for_offline_callers: adding the
    // flag to the stub --help would cascade into every spawn-argv
    // battery.)
    let ts = start(Config::default()).await;
    let c = client();
    let body = serde_json::json!({
        "model": "m1", "stream": false,
        "messages": [{"role": "user", "content": "a"}],
        "options": {"num_ctx": 2_000_000},
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
    assert!(msg.contains("num_ctx 2000000"), "{msg}");
    assert!(msg.contains("q8_0"), "teaching hint present: {msg}");
}

#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__num_ctx_ladder_serves_q4_hostable_shape() {
    // Preflight ladder band: 500k ctx overflows f16 and q8_0 on the
    // 24 GiB stub card but FITS at q4_0 — the unpinned cache_type
    // ladders (mirroring the spawn compiler's kv_quant_ladder) and the
    // request SERVES instead of refusing a shape the spawn itself
    // would host.
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
    assert_eq!(r.status(), 200, "ladder must host the q4_0-fittable shape");
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__auth_bearer_required_when_keys_set() {
    let cfg = Config {
        keys: vec![blazar_core::ApiKey {
            name: "admin".into(),
            key: "secret-1".into(),
            ..blazar_core::ApiKey::default()
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
    use blazar_core::ApiKey;
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
    assert!(
        keys.iter()
            .all(|k| !k["key"].as_str().unwrap_or("").contains("plm_admin"))
    );
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
async fn e2e__local_tts_charges_key_admission() {
    use blazar_core::ApiKey;
    // Audit MM1: the LOCAL piper lane must pass key admission BEFORE its
    // not-installed teaching — the first request charges rpm, the second
    // within the window exhausts it, and a wrongly-scoped key never
    // reaches the lane at all.
    let cfg = Config {
        keys: vec![ApiKey {
            name: "ci".into(),
            key: "plm_tts".into(),
            rpm: 1,
            ..ApiKey::default()
        }],
        ..Config::default()
    };
    let ts = start(cfg).await;
    let c = client();
    let speech = serde_json::json!({
        "model": "en_US-amy-medium",
        "input": "hello from the local lane",
    });
    // 404 = tts_error not-installed teaching — admission PASSED (a
    // bypassed lane would 404 twice without ever exhausting rpm).
    let first = c
        .post(format!("{}/v1/audio/speech", ts.base))
        .bearer_auth("plm_tts")
        .json(&speech)
        .send()
        .await
        .unwrap();
    assert_eq!(first.status(), 404, "teaching error, not a crash");
    let second = c
        .post(format!("{}/v1/audio/speech", ts.base))
        .bearer_auth("plm_tts")
        .json(&speech)
        .send()
        .await
        .unwrap();
    assert_eq!(second.status(), 429, "rpm=1 exhausted by the local lane");
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__local_whisper_charges_key_admission() {
    use blazar_core::ApiKey;
    // Audit MM1: local whisper transcriptions admit through the key
    // store like every other generation lane (501 teaching below proves
    // the request reached the lane AFTER admission charged it).
    let cfg = Config {
        keys: vec![ApiKey {
            name: "ci".into(),
            key: "plm_whisper".into(),
            rpm: 1,
            ..ApiKey::default()
        }],
        ..Config::default()
    };
    let ts = start(cfg).await;
    let c = client();
    let boundary = "X-BLAZAR-AUDIT-MM1";
    let multipart = format!(
        "--{boundary}\r\ncontent-disposition: form-data; name=\"file\"; \
         filename=\"clip.wav\"\r\ncontent-type: audio/wav\r\n\r\nAAAA\r\n--{boundary}--\r\n"
    );
    let post_multipart = |key: &'static str| {
        let c = c.clone();
        let url = format!("{}/v1/audio/transcriptions", ts.base);
        let body = multipart.clone();
        async move {
            c.post(url)
                .bearer_auth(key)
                .header(
                    reqwest::header::CONTENT_TYPE,
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .body(body)
                .send()
                .await
                .unwrap()
        }
    };
    // First: 501 teaching (no whisper engine/model in the test dirs) —
    // admission already charged the request.
    let first = post_multipart("plm_whisper").await;
    assert_eq!(first.status(), 501, "half-missing teaching");
    let second = post_multipart("plm_whisper").await;
    assert_eq!(second.status(), 429, "rpm=1 exhausted by the local lane");
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__whisper_stream_and_async_are_mutually_exclusive() {
    // F6: `stream` + `async` on one body is a request-shape error — it
    // must 400 before lane availability is even considered.
    let ts = start(Config::default()).await;
    let c = client();
    let boundary = "X-BLAZAR-F6-MM";
    let multipart = format!(
        "--{boundary}\r\ncontent-disposition: form-data; name=\"file\"; \
         filename=\"clip.wav\"\r\ncontent-type: audio/wav\r\n\r\nAAAA\r\n\
         --{boundary}\r\ncontent-disposition: form-data; name=\"stream\"\r\n\r\ntrue\r\n\
         --{boundary}\r\ncontent-disposition: form-data; name=\"async\"\r\n\r\ntrue\r\n\
         --{boundary}--\r\n"
    );
    let resp = c
        .post(format!("{}/v1/audio/transcriptions", ts.base))
        .header(
            reqwest::header::CONTENT_TYPE,
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(multipart)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
    let body = resp.text().await.unwrap();
    assert!(body.contains("mutually exclusive"), "body: {body}");
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__whisper_stream_without_engine_keeps_teaching() {
    // F6: stream on a rig with no whisper engine falls through to the
    // lane-absent 501 teaching — the streaming branch never bypasses
    // the teaching ladder.
    let ts = start(Config::default()).await;
    let c = client();
    let boundary = "X-BLAZAR-F6-ST";
    let multipart = format!(
        "--{boundary}\r\ncontent-disposition: form-data; name=\"file\"; \
         filename=\"clip.wav\"\r\ncontent-type: audio/wav\r\n\r\nAAAA\r\n\
         --{boundary}\r\ncontent-disposition: form-data; name=\"stream\"\r\n\r\ntrue\r\n\
         --{boundary}--\r\n"
    );
    let resp = c
        .post(format!("{}/v1/audio/transcriptions", ts.base))
        .header(
            reqwest::header::CONTENT_TYPE,
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(multipart)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 501);
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__models_capabilities_field_matches_mmproj() {
    // Audit MM13: /v1/models rows carry a capabilities array (blazar-
    // native shape mirroring /api/show) — [] for text-only rows.
    let ts = start(Config::default()).await;
    let c = client();
    let listed: serde_json::Value = c
        .get(format!("{}/v1/models", ts.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let data = listed["data"].as_array().cloned().unwrap_or_default();
    assert!(!data.is_empty(), "harness model m1 present");
    for row in &data {
        assert!(
            row.get("capabilities")
                .is_some_and(serde_json::Value::is_array),
            "every row carries a capabilities array: {row}"
        );
    }
    // The harness rows have no mmproj attached.
    assert!(
        data.iter()
            .all(|r| r["capabilities"].as_array().is_some_and(Vec::is_empty))
    );
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__keys_concurrency_cap() {
    use blazar_core::ApiKey;
    // Held streaming response keeps the slot leased (GuardedBody Drop fires
    // only when the body drains or the client disconnects). The stub delays
    // its SSE chunks via env, inherited by the spawned stub child; tests run
    // with --test-threads=1 in the gate, so the global env is safe here.
    set_env("STUB_DELAY_CHUNK_MS", "400");
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

    remove_env("STUB_DELAY_CHUNK_MS");
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
        blazar_gateway::serve(
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
    blazar_core::tls::ensure_tls_provider();
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
    assert!(
        denied
            .headers()
            .get("access-control-allow-origin")
            .is_none()
    );
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
#[allow(clippy::too_many_lines)]
async fn e2e__responses__background_conversations_and_cloud_tools() {
    let ts = start(Config::default()).await;
    let c = client();

    // Background mode: 202 + queued, then polls to the completed body.
    let queued = c
        .post(format!("{}/v1/responses", ts.base))
        .json(&serde_json::json!({
            "model": "m1",
            "input": "in the background",
            "store": true,
            "background": true,
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(queued.status(), 202);
    let q: serde_json::Value = queued.json().await.unwrap();
    assert_eq!(q["status"], "queued");
    assert_eq!(q["background"], true);
    let bg_id = q["id"].as_str().unwrap().to_string();

    // The job plane carries the same id (kind=responses).
    let job: serde_json::Value = c
        .get(format!("{}/v1/jobs/{}", ts.base, bg_id))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(job["kind"], "responses");

    // Poll until terminal (stub completes immediately; bound anyway).
    let mut final_body = serde_json::Value::Null;
    for _ in 0..100 {
        let r: serde_json::Value = c
            .get(format!("{}/v1/responses/{}", ts.base, bg_id))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        if matches!(r["status"].as_str(), Some("queued" | "in_progress")) {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        } else {
            final_body = r;
            break;
        }
    }
    assert_eq!(
        final_body["status"], "completed",
        "background response completes: {final_body}"
    );
    assert_eq!(final_body["id"].as_str().unwrap(), bg_id);
    // Body tier, not just the job tier: the verbatim stored body must be
    // served under the SAME id the 202 returned (guards against a shadowed
    // re-mint storing it under an unreachable id).
    assert!(
        final_body.get("output").is_some(),
        "polled body is the stored engine body: {final_body}"
    );

    // Background + stream is a contract error, not a surprise.
    let bad = c
        .post(format!("{}/v1/responses", ts.base))
        .json(&serde_json::json!({
            "model": "m1", "input": "x", "store": true,
            "background": true, "stream": true,
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(bad.status(), 400);
    let b: serde_json::Value = bad.json().await.unwrap();
    assert!(
        b["error"]["message"]
            .as_str()
            .unwrap()
            .contains("background")
    );

    // Cloud builtin tools teach, never silently pretend.
    let cloud = c
        .post(format!("{}/v1/responses", ts.base))
        .json(&serde_json::json!({
            "model": "m1", "input": "x",
            "tools": [{"type": "web_search"}],
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(cloud.status(), 400);
    let b: serde_json::Value = cloud.json().await.unwrap();
    assert!(
        b["error"]["message"]
            .as_str()
            .unwrap()
            .contains("web_search"),
        "{}",
        b
    );

    // Conversations: two stored responses in conv-e2e, the second via
    // previous_response_id inheriting the conversation.
    let c1: serde_json::Value = c
        .post(format!("{}/v1/responses", ts.base))
        .json(&serde_json::json!({
            "model": "m1", "input": "first", "store": true,
            "conversation": "conv-e2e",
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let first_id = c1["id"].as_str().unwrap().to_string();
    let _c2: serde_json::Value = c
        .post(format!("{}/v1/responses", ts.base))
        .json(&serde_json::json!({
            "model": "m1", "input": "second", "store": true,
            "previous_response_id": first_id,
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();

    let conv: serde_json::Value = c
        .get(format!("{}/v1/conversations/conv-e2e", ts.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(conv["count"], 2, "both responses stamped/inherited: {conv}");
    assert_eq!(conv["object"], "conversation");

    let del: serde_json::Value = c
        .delete(format!("{}/v1/conversations/conv-e2e", ts.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(del["deleted"], 2);

    let gone = c
        .get(format!("{}/v1/conversations/conv-e2e", ts.base))
        .send()
        .await
        .unwrap();
    assert_eq!(gone.status(), 404);
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
    upsert_vision_sibling(&ts);
    // pdf_ocr-shaped payload: images[] (b64 of PNG magic + padding) +
    // system + options.
    let png_b64 = "iVBORw0KGgoAAAA";
    let vision = serde_json::json!({
        "model": "m1v",
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

    // No-projector contract (e2e): m1 has no sidecar, so the same
    // payload teaches instead of reaching a child that would 500.
    let refused = c
        .post(format!("{}/api/generate", ts.base))
        .json(&serde_json::json!({
            "model": "m1", "prompt": "describe", "images": [png_b64], "stream": false
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(refused.status(), 400);
    let rb: serde_json::Value = refused.json().await.unwrap();
    assert!(
        rb["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("no projector sidecar"),
        "teaching refusal: {rb}"
    );

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
    assert!(
        resp.headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .is_some_and(|ct| ct.contains("ndjson"))
    );
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
    assert_eq!(fin["done"], true);
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
    upsert_vision_sibling(&ts);
    // b64 of PNG magic + 4 zero bytes (decodes cleanly at 12 bytes).
    let png_b64 = "iVBORw0KGgoAAAA";
    let body = serde_json::json!({
        "model": "m1v",
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
    assert_eq!(ps["models"][0]["blazar_state"], "ready");
    assert_eq!(ps["models"][0]["blazar_ctx"], 16384);
    // Compiled lanes (this stub runs the llama dialect) must stay
    // byte-identical: the capacity key is additive, runtime-managed
    // lanes only.
    assert!(ps["models"][0].get("blazar_capacity").is_none());

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
    // ollama-parity capability discovery (client vision + thinking
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
    let boundary = "blazar-test-boundary";
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
        msg.contains("blazar whisper --install"),
        "teaching error should name the fix: {r}"
    );
    ts.state.sup.shutdown_all().await.unwrap();
}

/// The async knob never bypasses the lane checks: with no whisper lane
/// installed an `async=true` submit gets the SAME teaching 501 as the
/// sync path (the job machinery only runs on the local lane).
#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__audio_transcriptions_async_without_lane_teaching_501() {
    let ts = start(Config::default()).await;
    let c = client();
    let boundary = "blazar-test-boundary";
    let mp = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\nContent-Type: audio/wav\r\n\r\nRIFF\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nm1\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"async\"\r\n\r\ntrue\r\n--{boundary}--\r\n"
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
        msg.contains("blazar whisper --install"),
        "async submit teaches the same fix: {r}"
    );
    ts.state.sup.shutdown_all().await.unwrap();
}

/// Capabilities are gateway-derived: on an empty install the route 200s
/// with engine.installed=false, no models, no child — and boots nothing.
#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__audio_capabilities__empty_install_reports_gaps_without_booting() {
    let ts = start(Config::default()).await;
    let c = client();
    let resp = c
        .get(format!("{}/v1/audio/capabilities", ts.base))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let r: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(r["object"], "whisper.capabilities");
    assert_eq!(r["engine"]["kind"], "whisper");
    assert_eq!(r["engine"]["installed"], false);
    assert_eq!(r["models"], serde_json::json!([]));
    assert!(r["child"].is_null(), "no child on an empty install: {r}");
    assert_eq!(r["async"]["field"], "async");
    assert!(
        r["endpoints"]
            .as_array()
            .expect("endpoints array")
            .iter()
            .any(|e| e == "/v1/audio/jobs/{id}"),
        "job endpoints advertised: {r}"
    );
    assert_eq!(r["idle_timeout_secs"], 900, "config default shows up");
    // Boots nothing: the lane stays absent after the read.
    assert!(ts.state.whisper.status().await.is_none());
    ts.state.sup.shutdown_all().await.unwrap();
}

/// Job routes: unknown id 404s with the durable-ledger lifetime truth;
/// a hostile id (path/meta characters) 400s before any lookup. The
/// unified `/v1/jobs` plane teaches the same contract on its own routes.
#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__audio_jobs__unknown_404_and_invalid_id_400() {
    let ts = start(Config::default()).await;
    let c = client();
    let resp = c
        .get(format!("{}/v1/audio/jobs/aj-does-not-exist", ts.base))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
    let r: serde_json::Value = resp.json().await.unwrap();
    let msg = r["error"]["message"].as_str().unwrap_or_default();
    assert!(
        msg.contains("durable records live at /v1/jobs"),
        "lifetime truth in the 404: {r}"
    );
    // Unified plane: empty ledger lists cleanly, unknown ids 404.
    let resp = c
        .get(format!("{}/v1/jobs?kind=audio&limit=5", ts.base))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let list: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(list["object"], "blazar.job.list", "list shape: {list}");
    let resp = c
        .get(format!("{}/v1/jobs/job-does-not-exist", ts.base))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
    let resp = c
        .post(format!("{}/v1/audio/jobs/..%2Fetc/cancel", ts.base))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
    ts.state.sup.shutdown_all().await.unwrap();
}

/// Request lifecycle (v0.15): a tracked generation gets a card (model
/// stamped from the body sniff, request-id header echoed), lands in the
/// terminal ring after the body drains, and the stop routes teach their
/// lifetime contract on unknown ids. A hostile id 400s before any lookup.
#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__requests__card_lifecycle_and_stop_contracts() {
    let ts = start(Config::default()).await;
    let c = client();
    let resp = c
        .post(format!("{}/api/chat", ts.base))
        .json(&serde_json::json!({
            "model": "m1",
            "stream": false,
            "messages": [{"role": "user", "content": "hi"}],
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let request_id = resp
        .headers()
        .get("x-blazar-request-id")
        .and_then(|v| v.to_str().ok())
        .expect("tracked lane stamps x-blazar-request-id")
        .to_string();
    // Drain the body so the card reaches its terminal state.
    let _body: serde_json::Value = resp.json().await.unwrap();

    let list: serde_json::Value = c
        .get(format!("{}/v1/requests?state=done&model=m1", ts.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(list["object"], "blazar.request.list", "list shape: {list}");
    let card = list["data"]
        .as_array()
        .and_then(|d| d.iter().find(|c| c["id"].as_str() == Some(&request_id)))
        .unwrap_or_else(|| panic!("card {request_id} in terminal list: {list}"));
    assert_eq!(card["state"], "done");
    assert_eq!(card["model"], "m1");
    assert_eq!(card["status"], 200);

    let got: serde_json::Value = c
        .get(format!("{}/v1/requests/{request_id}", ts.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(got["id"], *card["id"].as_str().unwrap());
    // Terminal card: no live stop URLs anymore.
    assert!(got.get("cancel_url").is_none());

    // Unknown id teaches the lifetime; traversal 400s.
    let resp = c
        .post(format!("{}/v1/requests/req_nope/cancel", ts.base))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
    let resp = c
        .post(format!("{}/v1/requests/..%2Fetc/interrupt", ts.base))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400, "encoded traversal is an invalid id");
    ts.state.sup.shutdown_all().await.unwrap();
}

/// Capacity endpoint: shape contract only (device count depends on the
/// host — CI boxes may have no nvidia-smi; the endpoint must answer with
/// an honest empty census + note, never error).
#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__capacity__shape_contract() {
    let ts = start(Config::default()).await;
    let c = client();
    let resp = c
        .get(format!("{}/api/capacity", ts.base))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let v: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(v["object"], "blazar.capacity", "shape: {v}");
    assert!(v["devices"].is_array());
    assert!(v["residents"].is_array());
    assert!(v["external"].is_array());
    assert!(
        !v["notes"].as_array().unwrap().is_empty(),
        "honesty notes present: {v}"
    );
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__warm__loads_model_and_route_agrees() {
    let ts = start(Config::default()).await;
    let c = client();

    // Warm the seeded model: the admission path spawns the stub child,
    // then the response reports the live resident state.
    let resp = c
        .post(format!("{}/api/warm", ts.base))
        .json(&serde_json::json!({ "model": "m1", "wait": true }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "warm must load m1");
    let v: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(v["object"], "blazar.warm", "shape: {v}");
    assert_eq!(v["model"], "m1");
    assert!(v["lane"].as_str().is_some(), "lane named: {v}");
    assert!(
        v["resident"]["state"].as_str().is_some(),
        "resident row after ensure: {v}"
    );
    assert!(
        v["note"].as_str().is_some_and(|n| n.contains("idle TTL")),
        "residency honesty note: {v}"
    );

    // The scheduling explainer must now agree the model is local and
    // resident — the decision a chat request would take.
    let resp = c
        .get(format!("{}/api/route/m1", ts.base))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let r: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(r["object"], "blazar.route", "shape: {r}");
    assert_eq!(r["decision"]["target"], "local", "local model: {r}");
    assert!(
        r["local"]["resident"]["state"].as_str().is_some(),
        "resident after warm: {r}"
    );
    assert!(r["peers"].as_array().is_some_and(std::vec::Vec::is_empty));

    // Unknown model: a 200 decision of "none" with the teaching reason —
    // absence is a valid answer from an explainer, never a fake target.
    let resp = c
        .get(format!("{}/api/route/nope", ts.base))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let r: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(r["decision"]["target"], "none", "unknown everywhere: {r}");
    assert!(
        r["decision"]["reason"]
            .as_str()
            .is_some_and(|s| s.contains("no peer lists it")),
        "teaching reason: {r}"
    );
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__replicate__teaches_without_remotes_and_bad_bodies() {
    let ts = start(Config::default()).await;
    let c = client();

    let resp = c
        .post(format!("{}/api/replicate", ts.base))
        .json(&serde_json::json!({ "model": "m1" }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400, "no remotes configured");
    let text = resp.text().await.unwrap();
    assert!(
        text.contains("no remotes configured"),
        "teaching 400: {text}"
    );

    let resp = c
        .post(format!("{}/api/replicate", ts.base))
        .json(&serde_json::json!({ "peers": ["a"] }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400, "missing model is a 400");
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__warm__notify_mode_keeps_pull_contract() {
    let ts = start(Config::default()).await;
    let c = client();
    let resp = c
        .post(format!("{}/api/warm", ts.base))
        .json(&serde_json::json!({ "model": "m1" }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "notify mode answers immediately");
    let v: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(v["status"], "ok", "pull-notify contract: {v}");
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__warm__unknown_model_teaches_404() {
    let ts = start(Config::default()).await;
    let c = client();
    let resp = c
        .post(format!("{}/api/warm", ts.base))
        .json(&serde_json::json!({ "model": "definitely-not-here", "wait": true }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
    let text = resp.text().await.unwrap();
    assert!(
        text.contains("unknown model") || text.contains("not found"),
        "teaching 404: {text}"
    );
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__explain__card_provenance_and_unknown_404() {
    let ts = start(Config::default()).await;
    let c = client();

    // Known model (the harness seeds m1 as a GGUF row with the stub
    // engine active): the card must name its sources, not just values.
    let resp = c
        .get(format!("{}/api/explain/m1", ts.base))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let v: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(v["object"], "blazar.explain", "shape: {v}");
    assert_eq!(v["model"]["name"], "m1");
    assert_eq!(v["model"]["format"], "gguf");
    // This harness store seeds models but NO engine rows — the card must
    // say so honestly (teaching reason), never fabricate a lane.
    assert_eq!(v["engine"]["source"], "error", "no engine rows: {v}");
    assert!(
        v["engine"]["reason"]
            .as_str()
            .is_some_and(|s| s.contains("no engine installed")),
        "teaching reason for the engineless store: {v}"
    );
    let ctx = &v["context"];
    assert!(
        ctx["requested_source"]
            .as_str()
            .is_some_and(|s| s.starts_with("config default_ctx = ")),
        "no overlay on m1 — requested ctx must cite the config default: {ctx}"
    );
    assert!(
        ctx["effective_source"]
            .as_str()
            .is_some_and(|s| s.contains("not resident")),
        "m1 is not spawned in the harness — effective ctx must say so: {ctx}"
    );
    assert!(
        v["speculation"]["mode"].as_str().is_some(),
        "spec mode always resolves (static tier): {v}"
    );
    assert!(
        v["cache"]["kv_k"].is_string(),
        "kv grade or auto ladder: {v}"
    );
    assert!(
        v["residents"]
            .as_array()
            .is_some_and(std::vec::Vec::is_empty)
    );

    // Tag-form alias (`name:quant`, the display shape /api/tags emits)
    // resolves through the canonical ladder — regression pin for the
    // explain 404 on colon forms.
    let resp = c
        .get(format!("{}/api/explain/m1:bf16", ts.base))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "tag form must resolve: ");
    let card = resp.json::<serde_json::Value>().await.unwrap();
    assert_eq!(card["model"]["name"].as_str(), Some("m1"));

    // Unknown model: teaching 404, never a half-empty card.
    let resp = c
        .get(format!("{}/api/explain/nope", ts.base))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
    let text = resp.text().await.unwrap();
    assert!(
        text.contains("not found") || text.contains("unknown model"),
        "teaching 404: {text}"
    );
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__model_doctor__job_cert_and_stored_caps() {
    let ts = start(Config::default()).await;
    let c = client();

    // Unknown model: teaching 404 BEFORE any job row exists.
    let resp = c
        .post(format!("{}/api/model-doctor", ts.base))
        .json(&serde_json::json!({"model": "nope"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
    let text = resp.text().await.unwrap();
    assert!(
        text.contains("not found") || text.contains("unknown model"),
        "teaching 404: {text}"
    );

    // Known model: a durable doctor job is created and runs the probes
    // through the real gateway path (stub engines — verdicts are theirs,
    // the CONTRACT here is the shape: every cap classified, terminal
    // state reached, cert queryable after).
    let resp = c
        .post(format!("{}/api/model-doctor", ts.base))
        .json(&serde_json::json!({"model": "m1"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "accepted: body checked next");
    let started: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(started["object"], "blazar.doctor");
    let id = started["id"].as_str().expect("job id").to_string();
    assert!(
        started["poll_url"]
            .as_str()
            .is_some_and(|u| u.contains(&id))
    );

    let cert;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    loop {
        let resp = c
            .get(format!("{}/v1/jobs/{id}", ts.base))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let job: serde_json::Value = resp.json().await.unwrap();
        match job["status"].as_str().unwrap_or("") {
            "completed" => {
                cert = job["result"].clone();
                break;
            }
            "failed" | "cancelled" => panic!("doctor job ended {job}"),
            _ => {}
        }
        assert!(
            std::time::Instant::now() < deadline,
            "doctor job did not finish in 60s: {job}"
        );
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    }
    assert_eq!(cert["object"], "blazar.model-doctor", "cert shape: {cert}");
    assert_eq!(cert["model"], "m1");
    let caps = cert["caps"].as_object().expect("caps map");
    for name in [
        "chat",
        "stream",
        "json",
        "tools",
        "embeddings",
        "vision",
        "think",
    ] {
        let v = caps
            .get(name)
            .unwrap_or_else(|| panic!("cap {name} present: {cert}"));
        let status = v["status"]
            .as_str()
            .unwrap_or_else(|| panic!("{name} status"));
        assert!(
            ["PASS", "FAIL", "N/A"].contains(&status),
            "{name} classified honestly, got {status}"
        );
        assert!(
            v["receipt"].as_str().is_some_and(|r| !r.is_empty()),
            "{name} carries a receipt"
        );
    }

    // The certificate is queryable by model after the run.
    let resp = c
        .get(format!("{}/api/model-doctor/m1", ts.base))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let stored: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(stored["object"], "blazar.model-doctor");
    assert!(stored["caps"].is_object());
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
        .join(blazar_core::profile::path_safe("m1"));
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
    assert!(
        ts.dirs
            .sessions_dir()
            .join(blazar_core::profile::path_safe("m1"))
            .join("r1")
            .exists()
    );
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
async fn e2e__child_header_stall_bounded_evicted_and_504() {
    // Pin (live evidence): a banked-session slot restore raced traffic
    // and one request hung 300s PRE-FIRST-BYTE while the child served
    // every later request — invisible to the request log and sentinel
    // (both observe from response start) and unbounded up to the blanket
    // transport ceiling. The header phase must instead be bounded,
    // warned, evicted, and retried once; terminal failure is a 504.
    let cfg = Config {
        child_header_timeout_secs: 1,
        ..Config::default()
    };
    let ts = start_with(cfg, vec![("STUB_HANG_ON".into(), "hang-token-9".into())]).await;
    let c = client();
    let began = std::time::Instant::now();
    let resp = c
        .post(format!("{}/api/chat", ts.base))
        .json(&serde_json::json!({
            "model": "m1",
            "messages": [{"role": "user", "content": "hang-token-9"}],
        }))
        .send()
        .await
        .expect("bounded: the request must return, never park to the blanket ceiling");
    let elapsed = began.elapsed();
    let status = resp.status().as_u16();
    // Two bounded attempts (1s each) + eviction/respawn overhead. The
    // spawn overhead is platform-dependent: windows CI runners pay a
    // multi-second real-time-scanning tax per child process spawn
    // (measured 23.3s end-to-end on windows-latest vs ~3s locally).
    // 60s still pins the contract — bounded far below the ~300s
    // blanket transport ceiling this test exists to guard against.
    assert!(
        elapsed < Duration::from_secs(60),
        "header-stall took {elapsed:?} — phase unbounded"
    );
    assert_eq!(
        status, 504,
        "terminal header-stall status, got {status} in {elapsed:?}"
    );
    // Wedge branch (slot probe said idle): the body must say the child
    // was evicted — the discriminator against the busy-child branch.
    let body: serde_json::Value = resp.json().await.unwrap();
    let msg = body["error"]["message"]
        .as_str()
        .or_else(|| body["error"].as_str())
        .unwrap_or_default();
    assert!(
        msg.contains("wedged child evicted"),
        "wedge 504 must name the eviction, got: {msg}"
    );
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__child_header_timeout_busy_child_kept_teaching_504() {
    // Pin (live evidence, 2026-10-05): a buffered (non-stream) think-mode
    // generation ran past child_header_timeout_secs and drew a 504 while
    // the child was healthy mid-generation — the synchronous eviction
    // then killed the child (and would kill sibling streams on a
    // multi-slot instance). vLLM and ollama cap no generation. The guard
    // now probes /slots on expiry: slots busy → child KEPT, request
    // fails alone with a teaching body naming the two remedies.
    let cfg = Config {
        child_header_timeout_secs: 1,
        ..Config::default()
    };
    let ts = start_with(
        cfg,
        vec![("STUB_HANG_BUSY_ON".into(), "busy-token-7".into())],
    )
    .await;
    let c = client();
    let resp = c
        .post(format!("{}/api/chat", ts.base))
        .json(&serde_json::json!({
            "model": "m1",
            "messages": [{"role": "user", "content": "busy-token-7"}],
        }))
        .send()
        .await
        .expect("bounded: the request must return, never park to the blanket ceiling");
    assert_eq!(resp.status().as_u16(), 504);
    let body: serde_json::Value = resp.json().await.unwrap();
    let msg = body["error"]["message"]
        .as_str()
        .or_else(|| body["error"].as_str())
        .unwrap_or_default();
    assert!(
        msg.contains("still generating"),
        "busy-child 504 must say the child is healthy, got: {msg}"
    );
    assert!(
        msg.contains("\"stream\": true"),
        "busy-child 504 must teach the streaming remedy, got: {msg}"
    );
    assert!(
        msg.contains("child_header_timeout_secs"),
        "busy-child 504 must name the knob, got: {msg}"
    );
    assert!(
        !msg.contains("wedged child evicted"),
        "busy-child 504 must NOT claim an eviction, got: {msg}"
    );
    assert!(
        !msg.contains("retry on respawned child"),
        "busy-child 504 must NOT retry onto a respawned lane (the retry would \
         re-run the same oversized generation into the same ceiling), got: {msg}"
    );
    assert_eq!(
        msg.matches("still generating").count(),
        1,
        "teaching message must appear exactly once (no duplicated retry text), got: {msg}"
    );
    // The child was NOT evicted: ps still shows m1 resident and ready
    // (nothing respawns it without traffic), and a follow-up request
    // serves from the same live child.
    let ps: serde_json::Value = c
        .get(format!("{}/api/ps", ts.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        ps["models"][0]["name"], "m1",
        "child must stay resident after the busy-child teaching 504"
    );
    assert_eq!(
        ps["models"][0]["blazar_state"], "ready",
        "child must stay ready after the busy-child teaching 504"
    );
    let follow_up = c
        .post(format!("{}/api/chat", ts.base))
        .json(&serde_json::json!({
            "model": "m1",
            "messages": [{"role": "user", "content": "are you alive"}],
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(follow_up.status(), 200, "same child keeps serving");
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__ollama_compat_recipe_raw_completion_lane_round_trip() {
    // prompt_recipe = "ollama_compat": the gateway renders the prompt
    // (chatml + JSON tool grammar system block) and drives the child
    // through /v1/completions, adapting text back to chat shape. The
    // stub replies with a fixed completion; the response must arrive
    // in ollama chat shape with the recipe's stop honored end-to-end.
    let cfg = Config {
        prompt_recipe: "ollama_compat".into(),
        ..Config::default()
    };
    let ts = start(cfg).await;
    let c = client();
    let r = c
        .post(format!("{}/api/chat", ts.base))
        .json(&serde_json::json!({
            "model": "m1",
            "stream": false,
            "messages": [{"role": "system", "content": "be terse"},
                         {"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {
                "name": "compute", "description": "math",
                "parameters": {"type": "object", "properties": {"x": {"type": "number"}}}}}],
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let v: serde_json::Value = r.json().await.unwrap();
    let msg = &v["message"];
    assert!(msg.is_object(), "chat shape preserved: {v}");
    assert!(msg.get("role").is_some_and(|r| r == "assistant"));
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__sentinel__body_stall_fires_detection_and_bounded_close() {
    // Pin (live evidence: two 600 s zero-chunk hangs finished with NO
    // stall detection — finish null, detections [], prompt_tokens
    // null): a child that opens the SSE body then never sends a byte
    // must (1) fire the stall detector, (2) evict the wedged child
    // (which terminates the upstream body), and (3) close the client
    // stream within a bounded window instead of parking until the
    // client gives up.
    let cfg = Config {
        sentinel_stall_secs: 5,
        ..Config::default()
    };
    let ts = start_with(
        cfg,
        vec![
            ("STUB_HANG_ON".into(), "stall-body-token".into()),
            ("STUB_STALL_BODY".into(), "1".into()),
        ],
    )
    .await;
    let c = client();
    let began = std::time::Instant::now();
    let resp = c
        .post(format!("{}/api/chat", ts.base))
        .json(&serde_json::json!({
            "model": "m1",
            "stream": true,
            "messages": [{"role": "user", "content": "stall-body-token"}],
        }))
        .send()
        .await
        .expect("headers must arrive (body-stall, not header-stall)");
    // Stall (5 s) + eviction grace + respawn overhead — never minutes.
    // Same windows spawn-tax note as the header-stall pin above: child
    // respawn costs seconds per spawn on windows CI runners, so the
    // ceiling guards "bounded, not parked until client give-up" rather
    // than raw platform speed.
    let body = tokio::time::timeout(Duration::from_secs(60), resp.bytes())
        .await
        .expect("body must close within the stall+evict window")
        .expect("body read");
    let elapsed = began.elapsed();
    assert!(
        elapsed < Duration::from_secs(60),
        "body-stall closed in {elapsed:?}"
    );
    let _ = body;
    // The record must carry the stall detection (answerable via why).
    // Poll: the analyzer commits asynchronously once the dropped feed
    // signals End — the record lands within moments, not instantly.
    let mut stalled = false;
    for _ in 0..25 {
        let why: serde_json::Value = c
            .get(format!("{}/api/why", ts.base))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        let records = why["records"].as_array().cloned().unwrap_or_default();
        if records.iter().any(|r| {
            r.get("detections")
                .and_then(|d| d.as_array())
                .is_some_and(|d| {
                    d.iter()
                        .any(|x| x.get("code").and_then(|c| c.as_str()) == Some("stalled_stream"))
                })
        }) {
            stalled = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    assert!(stalled, "no stalled_stream detection in why records");
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
/// X-Blazar-Model header disambiguates.
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
        msg.contains("X-Blazar-Model"),
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
        .header("x-blazar-model", "m2")
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
    let endpoint = ps["models"][0]["blazar_endpoint"]
        .as_str()
        .expect("ps endpoint")
        .to_string();
    blazar_core::tls::ensure_tls_provider();
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
    // Remote target = a second healthy blazar server (stub child m1).
    let target = start(Config::default()).await;
    // Main server: pool "r" = [dead (port 1 refuses instantly), target].
    // Tie-break min_by_key picks the FIRST member while both are idle, so
    // requests 1..3 hit the dead one, rack up consecutive failures and mark
    // it down (circuit open); request 4 must fail over to the target.
    let cfg = Config {
        remotes: vec![
            blazar_core::config::Remote {
                name: "r".into(),
                url: "http://127.0.0.1:1".into(),
                key: String::new(),
                allow_insecure_http: false,
            },
            blazar_core::config::Remote {
                name: "r".into(),
                url: target.base.clone(),
                key: String::new(),
                allow_insecure_http: false,
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
    let endpoint = ps["models"][0]["blazar_endpoint"]
        .as_str()
        .unwrap()
        .to_string();
    blazar_core::tls::ensure_tls_provider();
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
    let boundary = "blazarBatchTest7f2a";
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

/// Anthropic Message Batches end-to-end: inline requests, replay through
/// the gateway's own /v1/messages, five-way counts, jsonl results, cancel
/// and validation teaching errors.
#[tokio::test]
#[allow(non_snake_case)]
#[allow(clippy::too_many_lines)] // one linear scenario, assertions inline
async fn e2e__anthropic_batch__create_replay_and_results() {
    let ts = start(Config::default()).await;
    let c = client();

    // Validation teaching errors: empty, bad custom_id, duplicate, and
    // missing model.
    let bad = c
        .post(format!("{}/v1/messages/batches", ts.base))
        .json(&serde_json::json!({"requests": []}))
        .send()
        .await
        .unwrap();
    assert_eq!(bad.status(), 400);
    let bad = c
        .post(format!("{}/v1/messages/batches", ts.base))
        .json(&serde_json::json!({"requests": [
            {"custom_id": "no spaces!", "params": {"model": "m1", "messages": []}}
        ]}))
        .send()
        .await
        .unwrap();
    assert_eq!(bad.status(), 400);
    let bad = c
        .post(format!("{}/v1/messages/batches", ts.base))
        .json(&serde_json::json!({"requests": [
            {"custom_id": "a", "params": {"model": "m1", "messages": []}},
            {"custom_id": "a", "params": {"model": "m1", "messages": []}}
        ]}))
        .send()
        .await
        .unwrap();
    assert_eq!(bad.status(), 400);
    let bad = c
        .post(format!("{}/v1/messages/batches", ts.base))
        .json(&serde_json::json!({"requests": [
            {"custom_id": "a", "params": {"messages": []}}
        ]}))
        .send()
        .await
        .unwrap();
    assert_eq!(bad.status(), 400);

    // Create: two good requests + one that the lane refuses at replay
    // (max_tokens missing -> /v1/messages 400 -> errored row, not a dead
    // batch — the documented per-request error contract).
    let batch: serde_json::Value = c
        .post(format!("{}/v1/messages/batches", ts.base))
        .json(&serde_json::json!({"requests": [
            {"custom_id": "a-1", "params": {
                "model": "m1", "max_tokens": 5,
                "messages": [{"role": "user", "content": "hello"}]}},
            {"custom_id": "a-2", "params": {
                "model": "m1", "max_tokens": 5,
                "messages": [{"role": "user", "content": "second"}]}},
            {"custom_id": "bad-1", "params": {
                "model": "m1",
                "messages": [{"role": "user", "content": "no max tokens"}]}}
        ]}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let id = batch["id"].as_str().unwrap().to_string();
    assert!(id.starts_with("msgbatch-"), "batch id: {batch:?}");
    assert_eq!(batch["type"], "message_batch");
    assert_eq!(batch["processing_status"], "in_progress");
    assert_eq!(batch["request_counts"]["processing"], 3);
    assert_eq!(
        batch["results_url"],
        format!("/v1/messages/batches/{id}/results")
    );
    for key in ["created_at", "expires_at"] {
        let s = batch[key].as_str().unwrap();
        assert!(
            s.len() == 20 && s.ends_with('Z') && s.contains('T'),
            "{key} must be ISO-8601 UTC: {s}"
        );
    }

    // Poll to ended.
    let mut done = None;
    for _ in 0..100 {
        let v: serde_json::Value = c
            .get(format!("{}/v1/messages/batches/{id}", ts.base))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        if v["processing_status"] == "ended" {
            done = Some(v);
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(250)).await;
    }
    let done = done.expect("anthropic batch reaches ended");
    assert_eq!(done["request_counts"]["succeeded"], 2, "batch: {done:?}");
    assert_eq!(done["request_counts"]["errored"], 1);
    assert_eq!(done["request_counts"]["processing"], 0);
    assert!(done["ended_at"].is_string(), "ended_at stamped: {done:?}");

    // Results jsonl: match rows by custom_id (order not guaranteed).
    let results = c
        .get(format!("{}/v1/messages/batches/{id}/results", ts.base))
        .send()
        .await
        .unwrap();
    assert_eq!(results.status(), 200);
    let body = results.text().await.unwrap();
    let rows: std::collections::HashMap<String, serde_json::Value> = body
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let v: serde_json::Value = serde_json::from_str(l).unwrap();
            (v["custom_id"].as_str().unwrap().to_string(), v)
        })
        .collect();
    assert_eq!(rows.len(), 3, "three result rows: {body}");
    for cid in ["a-1", "a-2"] {
        let r = &rows[cid];
        assert_eq!(r["result"]["type"], "message", "row {cid}: {r:?}");
        assert_eq!(r["result"]["message"]["role"], "assistant", "row {cid}");
        assert!(
            r["result"]["message"]["content"].is_array(),
            "anthropic content blocks: {r:?}"
        );
    }
    let bad_row = &rows["bad-1"];
    assert_eq!(bad_row["result"]["type"], "errored", "row: {bad_row:?}");
    assert!(
        bad_row["result"]["error"]["message"].is_string(),
        "teaching error carried: {bad_row:?}"
    );

    // List contains the batch with first/last ids.
    let list: serde_json::Value = c
        .get(format!("{}/v1/messages/batches", ts.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(
        list["data"]
            .as_array()
            .unwrap()
            .iter()
            .any(|v| v["id"].as_str() == Some(id.as_str())),
        "list contains the batch: {list:?}"
    );
    assert!(list["first_id"].is_string() || list["data"].as_array().unwrap().is_empty());

    // Cancel after ended and unknown id both teach, not crash.
    let r = c
        .post(format!("{}/v1/messages/batches/{id}/cancel", ts.base))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 400);
    let r = c
        .get(format!(
            "{}/v1/messages/batches/msgbatch-none/results",
            ts.base
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 404);

    // Archive marks archived_at and keeps the object readable.
    let archived: serde_json::Value = c
        .delete(format!("{}/v1/messages/batches/{id}", ts.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(
        archived["archived_at"].is_string(),
        "archived: {archived:?}"
    );
    let r = c
        .get(format!("{}/v1/messages/batches/{id}", ts.base))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);

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

    let flip = |tag: &str, kind: blazar_core::engine_kind::EngineKind| {
        ts.state.with_store(|s| {
            s.upsert_engine(&blazar_core::EngineRow {
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
            blazar_core::engine_kind::EngineKind::Sglang,
            "sglang",
        ),
        (
            "mistralrs-t",
            blazar_core::engine_kind::EngineKind::MistralRs,
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

#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__llamacpp_only_gates__routed_lane_beats_active_row() {
    // Auto-routing: a GGUF model serves on a llamacpp child even when a
    // mistralrs engine is globally active (JIT installs activate their
    // engine, so the mismatch is a default-config reality). The surface
    // gates must key on the ROUTED lane, not the active row.
    let ts = start(Config::default()).await;
    let c = client();

    // A safetensors-dir model so routing has a mistralrs candidate; the
    // dir must exist — the router checks `path.is_dir()`.
    let st_dir = ts.dirs.models_dir().join("st1.d");
    std::fs::create_dir_all(&st_dir).unwrap();
    ts.state.with_store(|s| {
        // The GGUF lane needs an installed llamacpp row to route to —
        // this harness seeds no engine rows by default (unlike the
        // support/ harness), so install both lanes explicitly.
        s.upsert_engine(&blazar_core::EngineRow {
            tag: "stub-l".into(),
            asset: "stub".into(),
            sha256: "l".into(),
            installed_at: 1,
            active: true,
            manifest: "{}".into(),
            kind: blazar_core::engine_kind::EngineKind::default(),
        })
        .unwrap();
        s.upsert_engine(&blazar_core::EngineRow {
            tag: "mistralrs-t".into(),
            asset: "stub".into(),
            sha256: "flip".into(),
            installed_at: 2,
            active: true,
            manifest: "{}".into(),
            kind: blazar_core::engine_kind::EngineKind::MistralRs,
        })
        .unwrap();
        s.set_active_engine("mistralrs-t").unwrap();
        s.upsert_model(&blazar_core::ModelRow {
            name: "st1".into(),
            repo: "o/st1".into(),
            quant: "BF16".into(),
            path: st_dir.display().to_string(),
            bytes: 1,
            sha256: None,
            mmproj_path: None,
            components: vec![],
            shards: 1,
            arch: None,
            params: None,
            ctx_train: None,
            pulled_at: 1,
            last_used_at: 1,
        })
        .unwrap();
    });

    // GGUF m1 routes llamacpp: /tokenize passes the gate and the stub
    // child answers (pre-fix this 400'd whenever mistralrs was active).
    let r = c
        .post(format!("{}/tokenize", ts.base))
        .json(&serde_json::json!({"model": "m1", "content": "hello"}))
        .send()
        .await
        .unwrap();
    let status = r.status();
    let body = r.text().await.unwrap();
    assert_eq!(
        status, 200,
        "GGUF model must keep llama-only surfaces (body: {body})"
    );
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(v.get("tokens").is_some(), "stub tokenize shape: {v}");

    // Safetensors st1 routes mistralrs: the gate teaches with the
    // routed kind named.
    let r = c
        .post(format!("{}/tokenize", ts.base))
        .json(&serde_json::json!({"model": "st1", "content": "hello"}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 400, "st1 /tokenize");
    let v: serde_json::Value = r.json().await.unwrap();
    let msg = v.to_string();
    assert!(
        msg.contains("llama-server-only") && msg.contains("mistralrs"),
        "st1 /tokenize teaching: {msg}"
    );

    // Same contract on the session lane: slot checkpoints follow the
    // routed lane, and the GGUF model bypasses the gate entirely.
    let r = c
        .post(format!("{}/api/session", ts.base))
        .json(&serde_json::json!({"model": "st1", "action": "save", "filename": "f"}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 400, "st1 session save");
    let v: serde_json::Value = r.json().await.unwrap();
    assert!(
        v.to_string().contains("llama-server-only"),
        "st1 session save teaching: {v}"
    );
    let r = c
        .post(format!("{}/api/session", ts.base))
        .json(&serde_json::json!({"model": "m1", "action": "save", "filename": "f"}))
        .send()
        .await
        .unwrap();
    let v: serde_json::Value = r.json().await.unwrap();
    assert!(
        !v.to_string().contains("llama-server-only"),
        "m1 session save must bypass the gate: {v}"
    );

    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__tool_wins_admission_over_earlier_queued_interactive() {
    // Admission contract, wired end to end: with the single slot held,
    // a Tool-class chat must be admitted before an Interactive chat
    // that queued EARLIER — class rank (plus the Tool deadline
    // tightening) decides, not arrival order. Order-based assertion:
    // response completion instants, no duration guessing.
    let mut cfg = Config::default();
    cfg.model_overrides.insert(
        "m1".into(),
        blazar_core::ModelOverride {
            slots: Some(1),
            ..Default::default()
        },
    );
    let ts = start_with(cfg, vec![("STUB_DELAY_MS".into(), "6000".into())]).await;
    let chat = |tools: bool| {
        let base = ts.base.clone();
        async move {
            let mut body = serde_json::json!({
                "model": "m1", "stream": false,
                "messages": [{"role": "user", "content": "hi"}],
            });
            if tools {
                body["tools"] = serde_json::json!([{
                    "type": "function",
                    "function": {"name": "t", "parameters": {"type": "object"}}
                }]);
            }
            let r = client()
                .post(format!("{base}/api/chat"))
                .json(&body)
                .send()
                .await
                .unwrap();
            assert_eq!(r.status(), 200);
            // Non-stream: the gateway buffers the full translated body,
            // so send() resolving = the request COMPLETED. The pre-send
            // instant this closure used to return only measured spawn
            // order, which made the ordering asserts below vacuous.
            std::time::Instant::now()
        }
    };
    // A: plain chat, admitted immediately, holds the slot ~6 s.
    // Deterministic hold: wait until the daemon CONFIRMS m1 is in
    // flight before queueing the waiters. A fixed stagger let slow
    // runners (Windows CI receipt: run 36806193764) queue C before A
    // reached the gate — C then admitted first and the scenario under
    // test flipped (tool could no longer jump the earlier-queued
    // interactive because it was not queued behind a held slot).
    let a = tokio::spawn(chat(false));
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
    loop {
        let inflight = match client().get(format!("{}/api/ps", ts.base)).send().await {
            Ok(r) if r.status().is_success() => r
                .json::<serde_json::Value>()
                .await
                .ok()
                .and_then(|v| {
                    v["models"].as_array().map(|rows| {
                        rows.iter()
                            .filter(|r| r["model"].as_str() == Some("m1"))
                            .filter_map(|r| r["blazar_in_flight"].as_i64())
                            .sum::<i64>()
                    })
                })
                .unwrap_or(0),
            _ => 0,
        };
        if inflight >= 1 {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "A never reached the admission gate"
        );
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    // C: plain chat (Interactive) — queues FIRST.
    let c = tokio::spawn(chat(false));
    tokio::time::sleep(std::time::Duration::from_millis(400)).await;
    // B: chat with tools (Tool class) — queues SECOND.
    let b = tokio::spawn(chat(true));
    let (ra, rc, rb) = tokio::join!(a, c, b);
    let (done_a, done_c, done_b) = (ra.unwrap(), rc.unwrap(), rb.unwrap());
    assert!(
        done_b < done_c,
        "tool request must admit before earlier-queued interactive"
    );
    assert!(done_a < done_b);
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__deterministic_isolate__parks_normal_request_until_exclusive_finishes() {
    // F4 contract, wired end to end: with deterministic_isolate on, a
    // temp-0 seeded request is granted EXCLUSIVE admission — a normal
    // request for the same model parks until the exclusive one finishes,
    // even though a free slot exists (slots = 2 here so the plain
    // ceiling would happily admit both concurrently — only the
    // exclusive gate can produce the observed serialization).
    let mut cfg = Config::default();
    cfg.model_overrides.insert(
        "m1".into(),
        blazar_core::ModelOverride {
            slots: Some(2),
            deterministic_isolate: Some(true),
            ..Default::default()
        },
    );
    let ts = start_with(cfg, vec![("STUB_DELAY_MS".into(), "6000".into())]).await;
    let chat = |temperature: Option<f64>, seed: Option<i64>| {
        let base = ts.base.clone();
        async move {
            let mut body = serde_json::json!({
                "model": "m1", "stream": false,
                "messages": [{"role": "user", "content": "hi"}],
            });
            if let Some(t) = temperature {
                body["options"]["temperature"] = serde_json::json!(t);
            }
            if let Some(s) = seed {
                body["options"]["seed"] = serde_json::json!(s);
            }
            let r = client()
                .post(format!("{base}/api/chat"))
                .json(&body)
                .send()
                .await
                .unwrap();
            assert_eq!(r.status(), 200);
            // Non-stream: gateway buffers the full body, so send()
            // resolving = the request COMPLETED (post-send instant).
            std::time::Instant::now()
        }
    };
    // A: temp-0 + seeded -> exclusive lane, holds it ~6 s.
    let a = tokio::spawn(chat(Some(0.0), Some(42)));
    tokio::time::sleep(std::time::Duration::from_millis(400)).await;
    // B: normal sampling request — would co-run on the second slot
    // without the exclusive gate; must park behind A instead.
    let b = tokio::spawn(chat(None, None));
    let (ra, rb) = tokio::join!(a, b);
    let (done_a, done_b) = (ra.unwrap(), rb.unwrap());
    assert!(
        done_b.duration_since(done_a) >= std::time::Duration::from_millis(3_000),
        "normal request must complete well after the exclusive holder (gate off would co-run both at ~6 s)"
    );
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__unknown_route__404_json_envelope_with_teaching_pointer() {
    let ts = start(Config::default()).await;
    // A route no router arm matches must still answer in the standard
    // error envelope (not axum's empty body), name method + path, and
    // point at the /.well-known/blazar census.
    let resp = client()
        .post(format!("{}/definitely/not/mounted", ts.base))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::NOT_FOUND);
    let body: serde_json::Value = resp.json().await.unwrap();
    let err = body
        .get("error")
        .and_then(|e| e.as_object())
        .unwrap_or_else(|| panic!("expected error envelope, got {body}"));
    assert_eq!(
        err.get("type").and_then(|t| t.as_str()),
        Some("blazar_error")
    );
    assert_eq!(
        err.get("code").and_then(serde_json::Value::as_u64),
        Some(404)
    );
    let msg = err.get("message").and_then(|m| m.as_str()).unwrap();
    assert!(
        msg.contains("POST") && msg.contains("/definitely/not/mounted"),
        "message names method+path: {msg}"
    );
    assert!(
        msg.contains("/.well-known/blazar"),
        "message teaches the census: {msg}"
    );
    ts.state.sup.shutdown_all().await.unwrap();
}

/// Failover chains (Wave B): a virtual alias escalates from a failing
/// primary to the healthy secondary on the same request, stamps
/// provenance headers, records the switch in the registry, and the
/// admin plane (list/detail/pin/unpin) controls resolution.
#[allow(non_snake_case)]
#[allow(clippy::too_many_lines)] // one end-to-end receipt per concern
#[tokio::test]
async fn e2e__failover_chain__escalates_stamps_and_admin_plane() {
    use blazar_core::config::{FailoverChain, FailoverTarget};
    let cfg = Config {
        failover: vec![FailoverChain {
            name: "receipt".into(),
            alias: "assistant".into(),
            min_residence_secs: 30,
            targets: vec![
                FailoverTarget {
                    remote: None,
                    model: "missing-model".into(),
                },
                FailoverTarget {
                    remote: None,
                    model: "m1".into(),
                },
            ],
        }],
        ..Config::default()
    };
    let ts = start(cfg).await;
    let c = client();

    // OpenAI lane: alias enters the chain, primary 404s, secondary serves.
    let resp = c
        .post(format!("{}/v1/chat/completions", ts.base))
        .json(&serde_json::json!({"model": "assistant", "stream": false,
            "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "escalated to the healthy target");
    assert_eq!(
        resp.headers()
            .get("x-blazar-failover")
            .and_then(|v| v.to_str().ok()),
        Some("receipt")
    );
    assert_eq!(
        resp.headers()
            .get("x-blazar-served-model")
            .and_then(|v| v.to_str().ok()),
        Some("local:m1")
    );
    let v: serde_json::Value = resp.json().await.unwrap();
    assert!(
        v["choices"][0]["message"]["content"]
            .as_str()
            .unwrap()
            .contains("stub:m1:"),
        "answered by the stub child: {v:?}"
    );

    // Anti-flap: the failed primary is benched, the ollama lane resolves
    // straight to the secondary.
    let resp = c
        .post(format!("{}/api/chat", ts.base))
        .json(&serde_json::json!({"model": "assistant", "stream": false,
            "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "ollama lane rides the benched chain");
    assert_eq!(
        resp.headers()
            .get("x-blazar-served-model")
            .and_then(|v| v.to_str().ok()),
        Some("local:m1"),
        "benched primary skipped without a second failure"
    );

    // Admin plane: status shows the sticky switch; detail accepts name
    // AND alias; pin forces; unpin releases.
    let st: serde_json::Value = c
        .get(format!("{}/api/failover", ts.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let chain = &st["chains"][0];
    assert_eq!(chain["name"], "receipt");
    assert_eq!(
        chain["sticky"], 1,
        "sticky moved to the serving target: {chain}"
    );
    assert!(chain["switches"].as_u64().unwrap() >= 1);
    // 30s bench minus the seconds elapsed since the failure landed.
    let bench = chain["targets"][0]["benched_for_secs"].as_u64().unwrap();
    assert!(
        (25..=30).contains(&bench),
        "primary benched for ~30s (anti-flap): {bench}"
    );

    let det: serde_json::Value = c
        .get(format!("{}/api/failover/assistant", ts.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(det["alias"], "assistant", "detail by alias");
    let det2: serde_json::Value = c
        .get(format!("{}/api/failover/receipt", ts.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(det2["name"], "receipt", "detail by name");

    let pin: serde_json::Value = c
        .post(format!("{}/api/failover/receipt/pin", ts.base))
        .json(&serde_json::json!({"target": 1}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(pin["pinned"], 1);
    let bad = c
        .post(format!("{}/api/failover/receipt/pin", ts.base))
        .json(&serde_json::json!({"target": 9}))
        .send()
        .await
        .unwrap();
    assert_eq!(bad.status(), 400, "out-of-range pin teaches");
    let unpin: serde_json::Value = c
        .post(format!("{}/api/failover/receipt/unpin", ts.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(unpin["pinned"].is_null());
    let unknown = c
        .get(format!("{}/api/failover/nope", ts.base))
        .send()
        .await
        .unwrap();
    assert_eq!(unknown.status(), 404);

    ts.state.sup.shutdown_all().await.unwrap();
}

/// MCP tool catalog: the status plane lists configured servers
/// report-only (never spawning one), and the buffered-tool-round guard
/// teaches instead of guessing when a client asks for streaming.
#[allow(non_snake_case)]
#[tokio::test]
async fn e2e__mcp__status_plane_and_stream_guard() {
    use blazar_core::config::McpServer;
    let cfg = Config {
        mcp: vec![McpServer {
            name: "demo".into(),
            command: vec!["sleep".into(), "600".into()],
            url: None,
            env: std::collections::HashMap::new(),
            timeout_secs: 30,
            env_clear: false,
            allowed_env: Vec::new(),
            cwd: None,
            max_output_bytes: None,
        }],
        ..Config::default()
    };
    let ts = start(cfg).await;
    let c = client();

    // Status plane: report-only — `sleep` is not an MCP server and must
    // never have been spawned to answer this.
    let st: serde_json::Value = c
        .get(format!("{}/api/mcp", ts.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(st["servers"][0]["name"], "demo");
    assert_eq!(
        st["servers"][0]["timeout_secs"], 30,
        "config carried through: {st}"
    );
    assert!(
        st["servers"][0]["alive"].is_boolean(),
        "alive reports without probing (report-only plane): {st}"
    );

    // Stream guard on the ollama lane: teaching 400 before any child or
    // MCP process is touched.
    let resp = c
        .post(format!("{}/api/chat", ts.base))
        .json(
            &serde_json::json!({"model": "m1", "stream": true, "mcp": "demo",
            "messages": [{"role": "user", "content": "hi"}]}),
        )
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400, "streaming + mcp teaches");
    let v: serde_json::Value = resp.json().await.unwrap();
    // ollama dialect carries the message as a bare error string.
    let msg = v["error"]["message"]
        .as_str()
        .or_else(|| v["error"].as_str())
        .unwrap_or_default();
    assert!(msg.contains("stream:false"), "message names the fix: {msg}");

    // Selector validation: an unconfigured server name teaches with the
    // configured list, not a bare 500.
    let resp = c
        .post(format!("{}/api/chat", ts.base))
        .json(
            &serde_json::json!({"model": "m1", "stream": false, "mcp": "nope",
            "messages": [{"role": "user", "content": "hi"}]}),
        )
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400, "unknown server teaches");
    let v: serde_json::Value = resp.json().await.unwrap();
    let msg = v["error"]["message"]
        .as_str()
        .or_else(|| v["error"].as_str())
        .unwrap_or_default();
    assert!(
        msg.contains("demo"),
        "teaching lists the configured servers: {msg}"
    );
    ts.state.sup.shutdown_all().await.unwrap();
}

/// The Anthropic translate lane teaches (never guesses) when a chain
/// carries remote targets: local-only chains serve, remote ones 400.
#[allow(non_snake_case)]
#[tokio::test]
async fn e2e__failover_chain__anthropic_lane_local_targets_only() {
    use blazar_core::config::{FailoverChain, FailoverTarget};
    let cfg = Config {
        failover: vec![FailoverChain {
            name: "mixed".into(),
            alias: "mixed-assistant".into(),
            min_residence_secs: 30,
            targets: vec![FailoverTarget {
                remote: Some("elsewhere".into()),
                model: "m1".into(),
            }],
        }],
        ..Config::default()
    };
    let ts = start(cfg).await;
    let resp = client()
        .post(format!("{}/v1/messages", ts.base))
        .json(
            &serde_json::json!({"model": "mixed-assistant", "max_tokens": 8,
            "messages": [{"role": "user", "content": "hi"}]}),
        )
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400, "remote target on the translate lane");
    let v: serde_json::Value = resp.json().await.unwrap();
    let msg = v["error"]["message"].as_str().unwrap_or_default();
    assert!(
        msg.contains("local targets only") && msg.contains("mixed"),
        "teaching names the chain and the lane constraint: {msg}"
    );
    ts.state.sup.shutdown_all().await.unwrap();
}
