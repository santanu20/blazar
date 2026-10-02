//! Thinking-dialect end-to-end: the real gateway over a real supervisor
//! spawning stub-llama-server children, with the stub shaped as a
//! thinking-capable model. Proves the two load-bearing contracts:
//!
//! 1. raw think-tag suppression on the `OpenAI` chat lane — a child that
//!    burns reasoning tokens by default must not leak them to a client
//!    that never asked for thinking (buffered and streamed), and must
//!    stay untouched when the caller explicitly asks;
//! 2. mistral.rs dialect normalization — a mistralrs-routed lane
//!    receives top-level think controls it actually understands
//!    (`enable_thinking` / mapped `reasoning_effort`), while explicit
//!    caller controls pass through unchanged.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use blazar_core::hardware::{GpuInfo, Hardware};
use blazar_core::store::Store;
use blazar_core::{BlazarDirs, Config};
use blazar_gateway::router;
use blazar_gateway::state::AppState;
use blazar_runtime::{EventBus, LlamaCppEngine, Supervisor};

fn stub_bin() -> PathBuf {
    let exe = std::env::current_exe().expect("test exe path");
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

fn pstr(s: &str) -> Vec<u8> {
    let mut v = (s.len() as u64).to_le_bytes().to_vec();
    v.extend_from_slice(s.as_bytes());
    v
}

/// GGUF fixture with a tool-capable template (routing only cares that a
/// valid qwen3 GGUF exists).
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
        (
            "tokenizer.chat_template",
            8,
            pstr("{%- if tools %}{{ tool_calls }}{%- endif %}"),
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

struct TestServer {
    base: String,
    argv_path: PathBuf,
    _tmp: tempfile::TempDir,
    state: Arc<AppState>,
    _sup_reaper: tokio::task::JoinHandle<()>,
}

/// `think_wrap` lands in the child env of BOTH lanes: the primary
/// llamacpp stub (`engine.child_env`) and every routed spawn
/// (`config.engine_env`, which the supervisor forwards to adapters).
#[allow(clippy::too_many_lines)] // one cohesive server-bring-up, like the sentinel harness
async fn start(think_wrap: &str) -> TestServer {
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

    // Both lanes registered the way real installs are: a stub binary
    // under a data-dir-relative server path per row. The llamacpp row
    // claims the GGUF model; the mistralrs row (kept active) claims the
    // safetensors dir.
    let register_stub =
        |rel_dir: &str, tag: &str, kind: blazar_core::engine_kind::EngineKind, active: bool| {
            let rel_server = format!(
                "{}/{}",
                rel_dir,
                if cfg!(windows) {
                    "stub-llama-server.exe"
                } else {
                    "stub-llama-server"
                }
            );
            let server_abs = dirs.data_dir.join(&rel_server);
            std::fs::create_dir_all(server_abs.parent().unwrap()).unwrap();
            std::fs::copy(stub_bin(), &server_abs).unwrap();
            let manifest = blazar_runtime::engine::manifest::Manifest {
                tag: "v0.9.3".into(),
                build_number: 9003,
                version_raw: "v0.9.3".into(),
                devices: Vec::new(),
                flags: BTreeSet::default(),
                spec_types: Vec::new(),
                server_path: rel_server,
                ..Default::default()
            };
            store
                .upsert_engine(&blazar_core::EngineRow {
                    tag: tag.into(),
                    asset: "stub".into(),
                    sha256: "think".into(),
                    installed_at: 2,
                    active,
                    manifest: serde_json::to_string(&manifest).unwrap(),
                    kind,
                })
                .unwrap();
        };
    register_stub(
        "engines/stub-l",
        "stub-l",
        blazar_core::engine_kind::EngineKind::LlamaCpp,
        false,
    );
    register_stub(
        "engines/mistralrs-stub",
        "mistralrs-t",
        blazar_core::engine_kind::EngineKind::MistralRs,
        true,
    );
    store.set_active_engine("mistralrs-t").unwrap();

    // A safetensors-dir model so routing has a mistralrs candidate; the
    // router checks `path.is_dir()` and the supervisor reads config.json
    // (read_hf_config) before spawning the mistral.rs child.
    let st_dir = dirs.models_dir().join("st1.d");
    std::fs::create_dir_all(&st_dir).unwrap();
    std::fs::write(
        st_dir.join("config.json"),
        serde_json::json!({"model_type": "qwen2"}).to_string(),
    )
    .unwrap();
    store
        .upsert_model(&blazar_core::ModelRow {
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
    drop(store);

    let argv_path = tmp.path().join("argv.json");
    let mut engine =
        LlamaCppEngine::new(blazar_runtime::probe_manifest(&stub_bin(), "stub").unwrap());
    engine.child_env = vec![("STUB_THINK_WRAP".to_string(), think_wrap.to_string())];

    let config = Config {
        engine_env: BTreeMap::from([
            ("STUB_THINK_WRAP".to_string(), think_wrap.to_string()),
            (
                "STUB_ARGV_FILE".to_string(),
                argv_path.display().to_string(),
            ),
        ]),
        ..Default::default()
    };

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
    s.reaper_interval = Duration::from_secs(3600);
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
        argv_path: argv_path.clone(),
        _tmp: tmp,
        state,
        _sup_reaper: reaper,
    }
}

fn client() -> reqwest::Client {
    blazar_core::tls::ensure_tls_provider();
    reqwest::Client::builder()
        .timeout(Duration::from_secs(60))
        .build()
        .unwrap()
}

async fn chat(
    c: &reqwest::Client,
    base: &str,
    model: &str,
    stream: bool,
    extra: serde_json::Value,
) -> (reqwest::StatusCode, String) {
    let mut body = serde_json::json!({
        "model": model,
        "stream": stream,
        "messages": [{"role": "user", "content": "hello there"}],
    });
    if let (Some(obj), Some(add)) = (body.as_object_mut(), extra.as_object()) {
        for (k, v) in add {
            obj.insert(k.clone(), v.clone());
        }
    }
    let r = c
        .post(format!("{base}/v1/chat/completions"))
        .json(&body)
        .send()
        .await
        .unwrap();
    let status = r.status();
    (status, r.text().await.unwrap())
}

#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__think_suppression__openai_lane_strips_unasked_and_keeps_asked() {
    // The stub plays a model that burns reasoning tokens by default and
    // reflects the think controls it received: every reply is prefixed
    // `<think>stub-plan</think>`. The gateway owes the caller the bare
    // answer unless they asked for thinking.
    let ts = start("always-reflect").await;
    let c = client();

    // Buffered, no think controls: reasoning must not leak, and the
    // reflection proves the no-controls request arrived that way.
    let (status, body) = chat(&c, &ts.base, "m1", false, serde_json::json!({})).await;
    assert_eq!(status, 200, "buffered chat: {body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    let content = v["choices"][0]["message"]["content"].as_str().unwrap();
    assert!(
        !content.contains("<think") && !content.contains("stub-plan"),
        "buffered: think must be suppressed, got: {content:?} argv={:?}",
        std::fs::read_to_string(&ts.argv_path)
    );
    assert!(
        content.contains("[think-reflect et=unset kwargs=unset effort=unset]"),
        "buffered: llamacpp lane keeps the caller's no-controls body, got: {content:?} argv={:?}",
        std::fs::read_to_string(&ts.argv_path)
    );
    assert!(content.contains("stub:"), "answer body kept: {content:?}");

    // Streamed, no think controls: the SSE split rides the think block
    // across frame halves — the filter must hold and re-emit safely.
    let (status, body) = chat(&c, &ts.base, "m1", true, serde_json::json!({})).await;
    assert_eq!(status, 200, "streamed chat: {body}");
    assert!(
        !body.contains("<think") && !body.contains("stub-plan"),
        "streamed: think must be suppressed, got SSE: {body}"
    );
    assert!(body.contains("stub:"), "streamed answer kept: {body}");
    assert!(body.contains("[DONE]"), "stream must terminate: {body}");

    // Asked for thinking (the kwargs dialect the OpenAI lane speaks):
    // the raw reasoning must pass through untouched — reflection in the
    // same reply proves the kwargs also reached the child verbatim.
    let asked = serde_json::json!({
        "chat_template_kwargs": {"enable_thinking": true}
    });
    let (status, body) = chat(&c, &ts.base, "m1", false, asked).await;
    assert_eq!(status, 200, "asked-thinking chat: {body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    let content = v["choices"][0]["message"]["content"].as_str().unwrap();
    assert!(
        content.contains("kwargs=on"),
        "asked: kwargs must reach the child, got: {content:?}"
    );
    assert!(
        content.contains("<think>stub-plan</think>"),
        "asked: think must be preserved, got: {content:?}"
    );

    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__think_dialect__mistralrs_lane_receives_top_level_controls() {
    // mistral.rs reads TOP-LEVEL `enable_thinking` / `reasoning_effort`
    // and thinks BY DEFAULT when both are absent. The stub reflects the
    // controls it received so the assertion is on the forwarded bytes'
    // effect, not on internals.
    let ts = start("reflect").await;
    let c = client();

    // No think controls: the gateway must pin the child OFF.
    let (status, body) = chat(&c, &ts.base, "st1", false, serde_json::json!({})).await;
    assert_eq!(status, 200, "mistralrs default-off chat: {body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    let content = v["choices"][0]["message"]["content"].as_str().unwrap();
    assert!(
        content.contains("[think-reflect et=off kwargs=unset effort=unset]"),
        "default must arrive as top-level enable_thinking=false, got: {content:?}"
    );

    // OpenAI effort vocabulary maps onto mistral.rs's: max -> xhigh,
    // with thinking pinned on.
    let max_effort = serde_json::json!({"reasoning_effort": "max"});
    let (status, body) = chat(&c, &ts.base, "st1", false, max_effort).await;
    assert_eq!(status, 200, "mistralrs max-effort chat: {body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    let content = v["choices"][0]["message"]["content"].as_str().unwrap();
    assert!(
        content.contains("et=on") && content.contains("effort=xhigh"),
        "max must map to xhigh with thinking on, got: {content:?}"
    );

    // Explicit kwargs controls pass through and the top level agrees.
    let kwargs_on = serde_json::json!({
        "chat_template_kwargs": {"enable_thinking": true}
    });
    let (status, body) = chat(&c, &ts.base, "st1", false, kwargs_on).await;
    assert_eq!(status, 200, "mistralrs kwargs-on chat: {body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    let content = v["choices"][0]["message"]["content"].as_str().unwrap();
    assert!(
        content.contains("kwargs=on") && content.contains("et=on"),
        "explicit kwargs must pass with top level agreeing, got: {content:?}"
    );

    // The llamacpp lane keeps its own dialect: no top-level controls are
    // invented for a kind that does not speak them.
    let (status, body) = chat(&c, &ts.base, "m1", false, serde_json::json!({})).await;
    assert_eq!(status, 200, "llamacpp chat on shared server: {body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    let content = v["choices"][0]["message"]["content"].as_str().unwrap();
    assert!(
        content.contains("et=unset"),
        "llamacpp lane must not receive mistral.rs controls, got: {content:?}"
    );

    ts.state.sup.shutdown_all().await.unwrap();
}
