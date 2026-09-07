//! Sentinel end-to-end: real gateway over a real supervisor spawning
//! stub-llama-server children with env-shaped response semantics.
//! Proves the two load-bearing invariants together:
//! 1. detections land in the ring (`pallama why` data), and
//! 2. response BYTES are untouched — parity under observation.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use pallama_core::hardware::{GpuInfo, Hardware};
use pallama_core::store::Store;
use pallama_core::{Config, PallamaDirs};
use pallama_gateway::router;
use pallama_gateway::state::AppState;
use pallama_runtime::{EventBus, LlamaCppEngine, Supervisor};

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

fn pstr(s: &str) -> Vec<u8> {
    let mut v = (s.len() as u64).to_le_bytes().to_vec();
    v.extend_from_slice(s.as_bytes());
    v
}

/// GGUF fixture WITH a tool-capable template unless `bare` (no template:
/// the precheck's `TemplateSupport::Missing` case).
fn write_gguf(path: &std::path::Path, bare: bool) {
    let mut b: Vec<u8> = Vec::new();
    b.extend_from_slice(b"GGUF");
    b.extend_from_slice(&3u32.to_le_bytes());
    b.extend_from_slice(&0u64.to_le_bytes());
    let mut kvs: Vec<(&str, u8, Vec<u8>)> = vec![
        ("general.architecture", 8, pstr("qwen3")),
        ("qwen3.block_count", 4, 28u32.to_le_bytes().to_vec()),
        ("qwen3.context_length", 4, 40_960u32.to_le_bytes().to_vec()),
        ("qwen3.head_count", 4, 16u32.to_le_bytes().to_vec()),
        ("qwen3.head_count_kv", 4, 8u32.to_le_bytes().to_vec()),
        ("qwen3.embedding_length", 4, 1024u32.to_le_bytes().to_vec()),
    ];
    if !bare {
        let template = "{%- if tools %}{{ tool_calls }}{%- endif %}";
        kvs.push(("tokenizer.chat_template", 8, pstr(template)));
    }
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
    _tmp: tempfile::TempDir,
    state: Arc<AppState>,
    _sup_reaper: tokio::task::JoinHandle<()>,
}

async fn start(config: Config, stub_env: &[(&str, &str)], bare_template: bool) -> TestServer {
    let tmp = tempfile::tempdir().unwrap();
    let dirs = PallamaDirs {
        config_dir: tmp.path().join("c"),
        data_dir: tmp.path().join("d"),
    };
    dirs.ensure().unwrap();
    let gguf = dirs.models_dir().join("m1-q4_k_m.gguf");
    write_gguf(&gguf, bare_template);
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

    let mut engine =
        LlamaCppEngine::new(pallama_runtime::probe_manifest(&stub_bin(), "stub").unwrap());
    engine.child_env = stub_env
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
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
        state,
        _sup_reaper: reaper,
    }
}

fn client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(Duration::from_mins(1))
        .build()
        .unwrap()
}

/// The analyzer finalizes off the response path: poll the ring until a
/// record for `route` exists (or fail loud).
async fn await_record(ts: &TestServer, route: &str) -> pallama_gateway::sentinel::SentinelRecord {
    for _ in 0..200 {
        let recs = ts.state.sentinel.why(None, 50);
        if let Some(r) = recs.iter().find(|r| r.route == route) {
            return r.clone();
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!(
        "no sentinel record for route {route}: {:?}",
        ts.state.sentinel.why(None, 5)
    );
}

fn codes(rec: &pallama_gateway::sentinel::SentinelRecord) -> Vec<&'static str> {
    rec.detections.iter().map(|d| d.code.as_str()).collect()
}

#[tokio::test]
#[allow(non_snake_case)]
async fn integration__sentinel__truncation_and_near_limit() {
    // Contract 1 (default): prompt_preflight refuses the over-ctx prompt
    // pre-hoc with a teaching 400 — the engine never truncates silently.
    let cfg = Config {
        default_ctx: 8,
        ..Config::default()
    };
    let ts = start(cfg, &[("STUB_FINISH", "length")], false).await;
    let c = client();
    let long: String = "word ".repeat(50);
    let refused = c
        .post(format!("{}/v1/chat/completions", ts.base))
        .json(&serde_json::json!({
            "model": "m1",
            "stream": false,
            "messages": [{"role": "user", "content": long}],
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(refused.status(), 400);
    let body: serde_json::Value = refused.json().await.unwrap();
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("truncat"),
        "teaching error: {body}"
    );
    ts.state.sup.shutdown_all().await.unwrap();

    // Contract 2 (preflight off): the sentinel catches the engine's
    // silent truncation post-hoc — the safety net stays pinned.
    let cfg = Config {
        default_ctx: 8,
        prompt_preflight: false,
        ..Config::default()
    };
    let ts = start(cfg, &[("STUB_FINISH", "length")], false).await;
    let r: serde_json::Value = c
        .post(format!("{}/v1/chat/completions", ts.base))
        .json(&serde_json::json!({
            "model": "m1",
            "stream": false,
            "messages": [{"role": "user", "content": long}],
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    // Bytes untouched: the stub's finish_reason rides through verbatim.
    assert_eq!(r["choices"][0]["finish_reason"], "length");
    let rec = await_record(&ts, "openai-chat").await;
    let cs = codes(&rec);
    assert!(cs.contains(&"ctx_truncated"), "{cs:?}");
    assert!(cs.contains(&"ctx_near_limit"), "{cs:?}");
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn integration__sentinel__tool_args_and_parity_under_observation() {
    let ts = start(Config::default(), &[("STUB_BAD_TOOL_ARGS", "1")], false).await;
    let c = client();
    let body = serde_json::json!({
        "model": "m1",
        "stream": true,
        "messages": [{"role": "user", "content": "call the tool"}],
        "tools": [{"function": {"name": "echo", "parameters": {"type": "object"}}}],
    });
    let text1 = c
        .post(format!("{}/v1/chat/completions", ts.base))
        .json(&body)
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    let text2 = c
        .post(format!("{}/v1/chat/completions", ts.base))
        .json(&body)
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    // Parity under observation: identical requests, byte-identical streams.
    assert_eq!(text1, text2, "sentinel must not alter response bytes");
    let rec = await_record(&ts, "openai-chat").await;
    let cs = codes(&rec);
    assert!(cs.contains(&"tool_args_invalid_json"), "{cs:?}");
    // Template HAS tools in this fixture: no precheck false positive.
    assert!(!cs.contains(&"template_no_tools"), "{cs:?}");
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn integration__sentinel__precheck_header_on_bare_template() {
    // Fixture GGUF carries NO chat template + tools in request:
    // x-pallama-warnings must name it BEFORE inference (header, not just why).
    let ts = start(Config::default(), &[], true).await;
    let c = client();
    let resp = c
        .post(format!("{}/v1/chat/completions", ts.base))
        .json(&serde_json::json!({
            "model": "m1",
            "stream": false,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"function": {"name": "echo", "parameters": {"type": "object"}}}],
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    assert_eq!(
        resp.headers()
            .get("x-pallama-warnings")
            .and_then(|v| v.to_str().ok()),
        Some("template_no_tools")
    );
    let rec = await_record(&ts, "openai-chat").await;
    assert!(codes(&rec).contains(&"template_no_tools"));
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn integration__sentinel__empty_and_schema_violation() {
    let ts = start(Config::default(), &[("STUB_EMPTY", "1")], false).await;
    let c = client();
    let r: serde_json::Value = c
        .post(format!("{}/v1/chat/completions", ts.base))
        .json(&serde_json::json!({"model": "m1", "stream": false,
            "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(r["choices"][0]["message"]["content"], "");
    let rec = await_record(&ts, "openai-chat").await;
    assert!(codes(&rec).contains(&"empty_response"), "{:?}", codes(&rec));
    ts.state.sup.shutdown_all().await.unwrap();

    let ts = start(Config::default(), &[("STUB_SCHEMA_VIOLATION", "1")], false).await;
    let c = client();
    let _: serde_json::Value = c
        .post(format!("{}/v1/chat/completions", ts.base))
        .json(&serde_json::json!({"model": "m1", "stream": false,
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "json_object"}}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let rec = await_record(&ts, "openai-chat").await;
    assert!(
        codes(&rec).contains(&"schema_violation"),
        "{:?}",
        codes(&rec)
    );
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn integration__sentinel__stalled_stream_detected_and_stream_survives() {
    let cfg = Config {
        sentinel_stall_secs: 5,
        ..Config::default()
    }; // validation floor
    let ts = start(cfg, &[("STUB_DELAY_CHUNK_MS", "5500")], false).await;
    let c = client();
    let text = c
        .post(format!("{}/v1/chat/completions", ts.base))
        .json(&serde_json::json!({"model": "m1", "stream": true,
            "messages": [{"role": "user", "content": "slow"}]}))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    // The stall warn must NEVER break the stream itself.
    assert!(text.contains("[DONE]"), "stream completed: {text}");
    let rec = await_record(&ts, "openai-chat").await;
    assert!(codes(&rec).contains(&"stalled_stream"), "{:?}", codes(&rec));
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn integration__sentinel__ollama_chat_path_observed() {
    let ts = start(Config::default(), &[("STUB_FINISH", "length")], false).await;
    let c = client();
    let r: serde_json::Value = c
        .post(format!("{}/api/chat", ts.base))
        .json(&serde_json::json!({"model": "m1", "stream": false,
            "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(r["done_reason"], "length");
    let rec = await_record(&ts, "ollama-chat").await;
    assert!(codes(&rec).contains(&"ctx_truncated"), "{:?}", codes(&rec));
    // /api/why shape: sentinel flag + records with machine codes.
    let why: serde_json::Value = c
        .get(format!("{}/api/why", ts.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(why["sentinel"], true);
    assert!(why["records"].as_array().is_some_and(|a| !a.is_empty()));
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn integration__sentinel__responses_api_stream_and_nonstream() {
    // Responses grammar: streamed function_call fragments merge into an
    // invalid-args detection; non-stream incomplete maps to truncation.
    let ts = start(Config::default(), &[("STUB_BAD_TOOL_ARGS", "1")], false).await;
    let c = client();
    let text = c
        .post(format!("{}/v1/responses", ts.base))
        .json(&serde_json::json!({"model": "m1", "stream": true,
            "tools": [{"type": "function", "name": "echo",
                "parameters": {"type": "object"}}]}))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(text.contains("[DONE]"), "stream completed: {text}");
    let rec = await_record(&ts, "openai-responses").await;
    let cs = codes(&rec);
    assert!(cs.contains(&"tool_args_invalid_json"), "{cs:?}");
    ts.state.sup.shutdown_all().await.unwrap();

    let ts = start(Config::default(), &[("STUB_FINISH", "length")], false).await;
    let c = client();
    let r: serde_json::Value = c
        .post(format!("{}/v1/responses", ts.base))
        .json(&serde_json::json!({"model": "m1", "stream": false, "input": "hi"}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(r["status"], "incomplete");
    let rec = await_record(&ts, "openai-responses").await;
    assert!(codes(&rec).contains(&"ctx_truncated"), "{:?}", codes(&rec));
    assert_eq!(
        rec.prompt_tokens,
        Some(u64::from(Config::default().default_ctx) - 4),
        "usage mapped input_tokens->prompt"
    );
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn integration__sentinel__ring_survives_restart_via_jsonl() {
    let tmp = tempfile::tempdir().unwrap();
    let run_dir = tmp.path().join("run");
    std::fs::create_dir_all(&run_dir).unwrap();
    let sentinel = pallama_gateway::sentinel::Sentinel::new(true, 0, Some(&run_dir));
    // Persistence is driven through the public judge (records + commits).
    let ctx = pallama_gateway::sentinel::RequestCtx {
        trace: "plm-persist-1".into(),
        route: "openai-chat".into(),
        model: "m1".into(),
        ..Default::default()
    };
    let body = serde_json::to_vec(&serde_json::json!({
        "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]
    }))
    .unwrap();
    let _ = sentinel.judge(&ctx, &body, 200);
    assert_eq!(sentinel.why(Some("plm-persist-1"), 10).len(), 1);
    drop(sentinel);

    // Fresh instance on the same run dir loads the history.
    let revived = pallama_gateway::sentinel::Sentinel::new(true, 0, Some(&run_dir));
    let loaded = revived.why(Some("plm-persist-1"), 10);
    assert_eq!(loaded.len(), 1, "record survived the restart");
    assert_eq!(loaded[0].model, "m1");
    // Corrupt trailing line: skipped, not fatal.
    {
        use std::io::Write as _;
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(run_dir.join("sentinel.jsonl"))
            .unwrap();
        writeln!(f, "{{not json").unwrap();
    }
    let tolerant = pallama_gateway::sentinel::Sentinel::new(true, 0, Some(&run_dir));
    assert_eq!(tolerant.why(Some("plm-persist-1"), 10).len(), 1);
}

#[tokio::test]
#[allow(non_snake_case)]
async fn integration__sentinel__enforce_422_on_nonstream_only() {
    // Opt-in header: non-stream gets a named 422; stream stays 200
    // (bytes already on the wire); default (no header) stays warn-only.
    let ts = start(Config::default(), &[("STUB_BAD_TOOL_ARGS", "1")], false).await;
    let c = client();
    let tools = serde_json::json!({"model": "m1", "stream": false,
        "messages": [{"role": "user", "content": "call it"}],
        "tools": [{"function": {"name": "echo", "parameters": {"type": "object"}}}]});

    let enforced = c
        .post(format!("{}/v1/chat/completions", ts.base))
        .header("x-pallama-enforce", "1")
        .json(&tools)
        .send()
        .await
        .unwrap();
    assert_eq!(enforced.status(), 422);
    let ebody: serde_json::Value = enforced.json().await.unwrap();
    let msg = ebody["error"]["message"].as_str().unwrap_or_default();
    assert!(msg.contains("tool_args_invalid_json"), "{msg}");

    let warn_only = c
        .post(format!("{}/v1/chat/completions", ts.base))
        .json(&tools)
        .send()
        .await
        .unwrap();
    assert_eq!(warn_only.status(), 200, "default stays warn-only");

    let mut streamed = tools.clone();
    streamed["stream"] = serde_json::json!(true);
    let stream_enforced = c
        .post(format!("{}/v1/chat/completions", ts.base))
        .header("x-pallama-enforce", "1")
        .json(&streamed)
        .send()
        .await
        .unwrap();
    assert_eq!(
        stream_enforced.status(),
        200,
        "streaming is never hard-failed"
    );
    let stext = stream_enforced.text().await.unwrap();
    assert!(stext.contains("[DONE]"));
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn integration__sentinel__enforce_config_global_and_ollama_path() {
    let cfg = Config {
        sentinel_enforce: true,
        ..Config::default()
    };
    let ts = start(cfg, &[("STUB_BAD_TOOL_ARGS", "1")], false).await;
    let c = client();
    // OpenAI path, no header: config enforces.
    let r = c
        .post(format!("{}/v1/chat/completions", ts.base))
        .json(&serde_json::json!({"model": "m1", "stream": false,
            "messages": [{"role": "user", "content": "x"}],
            "tools": [{"function": {"name": "echo"}}]}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 422);
    // Header 0 escapes the global enforce.
    let escape = c
        .post(format!("{}/v1/chat/completions", ts.base))
        .header("x-pallama-enforce", "0")
        .json(&serde_json::json!({"model": "m1", "stream": false,
            "messages": [{"role": "user", "content": "x"}],
            "tools": [{"function": {"name": "echo"}}]}))
        .send()
        .await
        .unwrap();
    assert_eq!(escape.status(), 200, "explicit 0 overrides config true");
    // Ollama path honors it too (ollama-shaped error body).
    let o = c
        .post(format!("{}/api/chat", ts.base))
        .json(&serde_json::json!({"model": "m1", "stream": false,
            "messages": [{"role": "user", "content": "x"}],
            "tools": [{"function": {"name": "echo"}}]}))
        .send()
        .await
        .unwrap();
    assert_eq!(o.status(), 422);
    let obody: serde_json::Value = o.json().await.unwrap();
    assert!(obody["error"]
        .as_str()
        .unwrap_or_default()
        .contains("sentinel enforce"));
    // Enforce decisions land in the ring (answerable via why).
    let rec = await_record(&ts, "ollama-chat").await;
    assert!(
        codes(&rec).contains(&"tool_args_invalid_json"),
        "{:?}",
        codes(&rec)
    );
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn integration__watch_sse_streams_live_detections() {
    let ts = start(Config::default(), &[("STUB_FINISH", "length")], false).await;
    let c = client();
    // Open the live tail BEFORE triggering the request.
    let stream = c
        .get(format!("{}/api/watch", ts.base))
        .timeout(Duration::from_mins(1))
        .send()
        .await
        .unwrap();
    assert_eq!(stream.status(), 200);
    assert!(stream
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.contains("text/event-stream")));
    let reader = stream;
    let (tx_lines, rx_lines) = tokio::sync::oneshot::channel::<String>();
    let handle = tokio::spawn(async move {
        let mut collected = String::new();
        let mut s = reader;
        while let Some(chunk) = s.chunk().await.unwrap_or(None) {
            collected.push_str(&String::from_utf8_lossy(&chunk));
            if collected.contains("ctx_truncated") {
                break;
            }
        }
        let _ = tx_lines.send(collected);
    });
    // Trigger a truncation-detected request.
    let _: serde_json::Value = c
        .post(format!("{}/v1/chat/completions", ts.base))
        .json(&serde_json::json!({"model": "m1", "stream": false,
            "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let collected = tokio::time::timeout(Duration::from_secs(20), rx_lines)
        .await
        .expect("watch stream delivered the record")
        .expect("reader task alive");
    assert!(collected.contains("data: "), "{collected}");
    assert!(collected.contains("ctx_truncated"), "{collected}");
    handle.abort();
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn integration__openai_num_ctx_header__restarts_at_requested_size() {
    // Gap-analysis row 15 closure: the OpenAI protocol has no ctx field;
    // the X-Pallama-Num-Ctx extension header fills it with the same
    // restart-once semantics as options.num_ctx.
    let ts = start(Config::default(), &[], false).await;
    let c = client();
    let first: serde_json::Value = c
        .post(format!("{}/v1/chat/completions", ts.base))
        .json(&serde_json::json!({"model": "m1", "stream": false,
            "messages": [{"role": "user", "content": "a"}]}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(first["choices"][0]["message"]["content"].as_str().is_some());
    assert_eq!(ts.state.sup.ps()[0].ctx, 16384);

    let r: serde_json::Value = c
        .post(format!("{}/v1/chat/completions", ts.base))
        .header("x-pallama-num-ctx", "32768")
        .json(&serde_json::json!({"model": "m1", "stream": false,
            "messages": [{"role": "user", "content": "b"}]}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(r["choices"][0]["message"]["content"].as_str().is_some());
    let ps = ts.state.sup.ps();
    assert!(!ps.is_empty(), "instance respawned at requested ctx");
    assert_eq!(ps[0].ctx, 32768, "header honored, never silently truncated");

    // Non-numeric value: named 400, never silent.
    let bad = c
        .post(format!("{}/v1/chat/completions", ts.base))
        .header("x-pallama-num-ctx", "big")
        .json(&serde_json::json!({"model": "m1", "stream": false,
            "messages": [{"role": "user", "content": "c"}]}))
        .send()
        .await
        .unwrap();
    assert_eq!(bad.status(), 400);
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn integration__sentinel__disabled_is_a_full_kill_switch() {
    let cfg = Config {
        sentinel: false,
        ..Config::default()
    };
    let ts = start(cfg, &[("STUB_FINISH", "length")], true).await;
    let c = client();
    let resp = c
        .post(format!("{}/v1/chat/completions", ts.base))
        .json(&serde_json::json!({"model": "m1", "stream": false,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"function": {"name": "echo"}}]}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    assert!(resp.headers().get("x-pallama-warnings").is_none());
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(
        ts.state.sentinel.why(None, 10).is_empty(),
        "kill switch must disable every hook"
    );
    let why: serde_json::Value = c
        .get(format!("{}/api/why", ts.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(why["sentinel"], false);
    ts.state.sup.shutdown_all().await.unwrap();
}
