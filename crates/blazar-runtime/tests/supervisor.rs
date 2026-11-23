// Crash/teardown pins send real signals (SIGKILL, signal-0 liveness) to
// exact child pids - a unix contract; the gateway crash-retry suite
// carries the portable taskkill variant.
#![cfg(unix)]
//! Supervisor integration tests against the REAL stub-llama-server:
#![allow(unsafe_code)] // audited libc::kill(pid, SIG) on exact child pids
//! ensure/health/argv, cold-start timeout, ladder timing, capacity+evict,
//! crash respawn+circuit, shutdown teardown, concurrent models.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use blazar_core::hardware::{GpuInfo, Hardware};
use blazar_core::store::Store;
use blazar_core::{BlazarDirs, Config};
use blazar_runtime::EventBus;
use blazar_runtime::engine::manifest::{Manifest, probe as probe_manifest};
use blazar_runtime::{LlamaCppEngine, SupervisionError, Supervisor};

fn stub_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_stub-llama-server"))
}

/// Manifest of the stub (real probe): includes --jinja, --sleep-idle-seconds...
fn stub_manifest() -> Manifest {
    probe_manifest(&stub_bin(), "stub").expect("probe stub")
}

fn setup(models: &[(&str, u64)]) -> (tempfile::TempDir, BlazarDirs) {
    let tmp = tempfile::tempdir().unwrap();
    let dirs = BlazarDirs {
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
            .upsert_model(&blazar_core::ModelRow {
                name: (*name).into(),
                repo: format!("o/{name}"),
                quant: "Q4_K_M".into(),
                path: p.display().to_string(),
                bytes: i64::try_from(*bytes).unwrap_or(i64::MAX),
                sha256: None,
                mmproj_path: None,
                components: vec![],
                shards: 1,
                arch: Some("qwen3".into()),
                params: None,
                ctx_train: Some(40_960),
                pulled_at: 1,
                last_used_at: 1,
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

fn supervisor(dirs: &BlazarDirs, config: Config, gpu: bool) -> Arc<Supervisor> {
    let mut engine = LlamaCppEngine::new(stub_manifest());
    engine.child_env = vec![
        (
            "STUB_ARGV_FILE".into(),
            dirs.run_dir().join("argv.json").display().to_string(),
        ),
        // Match the harness GpuInfo name so device mapping keeps it.
        (
            "STUB_DEVICES".into(),
            "stub-gpu: STUB GPU (24000 MiB, 24000 MiB free)".into(),
        ),
    ];
    let hw = if gpu {
        Hardware {
            physical_cores: 4,
            total_ram_mib: 16_000,
            mem_available_mib: None,
            gpus: vec![GpuInfo {
                name: "stub-gpu".into(),
                description: "STUB GPU".into(),
                total_mib: 24_000,
                free_mib: 24_000,
            }],
        }
    } else {
        Hardware {
            physical_cores: 4,
            total_ram_mib: 16_000,
            mem_available_mib: None,
            gpus: vec![],
        }
    };
    let mut s = Supervisor::new(
        dirs.clone(),
        config,
        EventBus::default(),
        hw,
        Arc::new(engine),
    );
    // Fast test timings.
    s.load_timeout_secs = Some(4);
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
        blazar_core::Endpoint::Tcp { host, port } => format!("http://{host}:{port}"),
        blazar_core::Endpoint::Unix { .. } => unreachable!(),
    };
    blazar_core::tls::ensure_tls_provider();
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
    for flag in [
        "--jinja",
        "--metrics",
        "--flash-attn",
        "--sleep-idle-seconds",
        "--cache-ram",
        "-np",
        "--slot-save-path",
    ] {
        assert!(
            argv.contains(&flag.to_string()),
            "missing {flag} in {argv:?}"
        );
    }
    // cache-reuse defaults OFF (native slot cache covers identical and
    // extended prefixes; the engine force-disables it at load on
    // hybrid-linear archs, so it only engages on full-attention models
    // - the retired "~0.6s cold" claim was sweep noise, unreproducible).
    assert!(
        !argv.contains(&"--cache-reuse".to_string()),
        "cache-reuse must not ride the default argv: {argv:?}"
    );
    // sessions dir is per-model under the data dir (path_safe suffix - derive)
    let sess_subdir = format!("sessions/{}/", blazar_core::profile::path_safe("m1"));
    assert!(
        argv.windows(2)
            .any(|w| w[0] == "--slot-save-path" && w[1].ends_with(&sess_subdir)),
        "slot-save-path per-model dir: {argv:?}"
    );
    assert!(argv.windows(2).any(|w| w[0] == "--alias" && w[1] == "m1"));
    // Default slots=0 auto: train ctx 40960 / base 16384 -> np 2 with the
    // total ctx scaled; ps still reports the per-slot ctx below.
    assert!(
        argv.windows(2)
            .any(|w| w[0] == "--ctx-size" && w[1] == "32768")
    );
    assert!(argv.windows(2).any(|w| w[0] == "-np" && w[1] == "2"));

    // ps shows ready with ctx.
    let ps = sup.ps();
    assert_eq!(ps.len(), 1);
    assert_eq!(ps[0].state, "ready");
    assert_eq!(ps[0].ctx, 16384, "ps ctx stays per-slot");

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

/// Workload-derived routing policy: 20 interactive requests (floor met,
/// latency-dominant) must flip the effective policy from the configured
/// default to Latency - and a GGUF model must STILL land on llamacpp,
/// because the policy only forks safetensors lanes. The spawn path and
/// the plan prediction share `derive_policy_from_store`, so pinning the
/// spawn side here covers both consumers' lane choice.
#[tokio::test]
#[allow(non_snake_case)]
async fn integration__routed_engine__workload_derived_policy_keeps_gguf_lane_stable() {
    let (_t, dirs) = setup(&[("m1", 500)]);
    // Seed an unambiguous interactive mix for m1. Day key from SQLite's
    // own clock so the seed always sits inside the 30d read window.
    let store = Store::open(&dirs).unwrap();
    let day: i64 = store
        .conn()
        .query_row(
            "SELECT CAST(strftime('%Y%m%d','now') AS INTEGER)",
            [],
            |r| r.get(0),
        )
        .unwrap();
    store
        .add_usage_workload_deltas(&[(day, "m1".into(), "interactive_chat".into(), 20)])
        .unwrap();
    let decision = blazar_core::routing_policy::derive_policy_from_store(
        &store,
        blazar_core::config::RoutingPolicy::Quality,
        true,
        "m1",
    );
    assert!(decision.derived, "reason: {}", decision.reason);
    assert_eq!(decision.policy, blazar_core::config::RoutingPolicy::Latency);

    // The supervisor consumes the same derivation at spawn: the derived
    // Latency policy must not destabilize the GGUF lane. In this harness
    // the llamacpp lane is served by the stub binary (LlamaCppEngine on
    // a probed stub manifest), so the lane tag is "stub".
    let sup = supervisor(&dirs, base_config(), true);
    sup.ensure("m1").await.expect("ensure under derived policy");
    let ps = sup.ps();
    assert_eq!(ps.len(), 1);
    assert_eq!(ps[0].engine, "stub");
    assert_eq!(ps[0].state, "ready");
    sup.shutdown_all().await.unwrap();
}

/// `autotune = "scheduled"` end-to-end against the real stub engine:
/// the reaper tick detaches a race pass that spawns real children for
/// the advertised spec arms, lands receipts carrying parity evidence,
/// publishes the `AutotuneRace` event, and releases the single-flight
/// guard. The engine row + engine dir wear the stub binary as both
/// llama-bench and llama-server - exactly the asset layout
/// `find_bench_bin` demands.
#[tokio::test]
#[allow(non_snake_case)]
async fn integration__autotune_schedule__lands_parity_receipts_and_event() {
    let (_t, dirs) = setup(&[("m1", 500)]);
    autotune_engine_assets(&dirs);
    let store = Store::open(&dirs).unwrap();

    // Evidence floor: the scheduled pass only races models with
    // chat-shaped workload rows in the window (day key packed int,
    // matching the store's YYYYMMDD convention).
    let day: i64 = store
        .conn()
        .query_row(
            "SELECT CAST(strftime('%Y%m%d','now') AS INTEGER)",
            [],
            |r| r.get(0),
        )
        .unwrap();
    store
        .add_usage_workload_deltas(&[(day, "m1".into(), "interactive_chat".into(), 5)])
        .unwrap();

    let mut cfg = base_config();
    cfg.autotune = Some("scheduled".into());
    let mut engine = LlamaCppEngine::new(stub_manifest());
    engine.child_env = vec![
        (
            "STUB_ARGV_FILE".into(),
            dirs.run_dir().join("argv.json").display().to_string(),
        ),
        (
            "STUB_DEVICES".into(),
            "stub-gpu: STUB GPU (24000 MiB, 24000 MiB free)".into(),
        ),
    ];
    let bus = EventBus::default();
    let mut events = bus.subscribe();
    let mut s = Supervisor::new(
        dirs.clone(),
        cfg,
        bus,
        Hardware {
            physical_cores: 4,
            total_ram_mib: 16_000,
            mem_available_mib: None,
            gpus: vec![GpuInfo {
                name: "stub-gpu".into(),
                description: "STUB GPU".into(),
                total_mib: 24_000,
                free_mib: 24_000,
            }],
        },
        Arc::new(engine),
    );
    s.load_timeout_secs = Some(4);
    let sup = Arc::new(s);

    sup.autotune_schedule_tick();
    // The pass is detached; poll the store until the dense baseline
    // and at least one spec arm land (the stub advertises ngram-simple,
    // so the ngram arm must race) or the deadline faults.
    let deadline = Instant::now() + Duration::from_secs(60);
    let mut receipts;
    loop {
        receipts = store.list_spec_receipts("m1", "stub").unwrap();
        let dense = receipts.iter().any(|r| r.mode == "dense");
        let spec_arm = receipts
            .iter()
            .any(|r| r.mode != "dense" && r.spec_params.is_some());
        if dense && spec_arm {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "scheduled race never landed receipts; got: {receipts:?}"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    // Every spec-arm receipt carries the parity verdict the adoption
    // gate reads.
    for r in receipts.iter().filter(|r| r.mode != "dense") {
        let params: serde_json::Value =
            serde_json::from_str(r.spec_params.as_deref().unwrap_or("null")).unwrap();
        assert!(
            params.get("parity_total").is_some(),
            "spec arm {} lacks parity evidence: {params}",
            r.mode
        );
    }
    // The event lane saw the pass summary.
    let mut saw_race = false;
    while let Ok(ev) = events.try_recv() {
        if let blazar_runtime::BlazarEvent::AutotuneRace { model, arms, .. } = ev {
            assert_eq!(model, "m1");
            assert!(arms >= 2, "dense + at least one spec arm, got {arms}");
            saw_race = true;
        }
    }
    assert!(saw_race, "AutotuneRace event never published");
    // Single-flight released after the pass.
    assert!(!sup.autotune_race_busy_for_test());
    sup.shutdown_all().await.unwrap();
}

/// Shared scheduled-race fixtures: engine row (llamacpp kind, stub
/// manifest) plus the on-disk asset pair `find_bench_bin` demands -
/// llama-bench anchors the dir, llama-server (the stub) is what
/// actually spawns.
fn autotune_engine_assets(dirs: &BlazarDirs) {
    let engine_dir = dirs.engines_dir().join("stub");
    std::fs::create_dir_all(&engine_dir).unwrap();
    for name in ["llama-server", "llama-bench"] {
        let p = engine_dir.join(name);
        std::fs::copy(stub_bin(), &p).unwrap();
        // chmod on a fresh tempdir copy: PermissionsExt::from_mode and
        // set_permissions are safe APIs - no unsafe block needed.
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
    }
    Store::open(dirs)
        .unwrap()
        .upsert_engine(&blazar_core::store::EngineRow {
            tag: "stub".into(),
            asset: "stub".into(),
            sha256: String::new(),
            installed_at: 1,
            active: true,
            manifest: serde_json::to_string(&stub_manifest()).unwrap(),
            kind: blazar_core::engine_kind::EngineKind::LlamaCpp,
        })
        .unwrap();
}

/// The scheduled pass never races under traffic: a model carrying an
/// in-flight request is skipped wholesale - no evict, no receipts, the
/// serving instance untouched.
#[tokio::test]
#[allow(non_snake_case)]
async fn integration__autotune_schedule__busy_lane_skips_model() {
    let (_t, dirs) = setup(&[("m1", 500)]);
    autotune_engine_assets(&dirs);
    // Evidence seeded so the busy-lane gate (not the evidence floor)
    // is what skips this model.
    let store = Store::open(&dirs).unwrap();
    let day: i64 = store
        .conn()
        .query_row(
            "SELECT CAST(strftime('%Y%m%d','now') AS INTEGER)",
            [],
            |r| r.get(0),
        )
        .unwrap();
    store
        .add_usage_workload_deltas(&[(day, "m1".into(), "interactive_chat".into(), 5)])
        .unwrap();
    let mut cfg = base_config();
    cfg.autotune = Some("scheduled".into());
    let sup = supervisor(&dirs, cfg, true);
    sup.ensure("m1").await.expect("serve m1");
    // One in-flight request pins the lane busy.
    sup.begin_request("m1");
    sup.autotune_schedule_tick();
    tokio::time::sleep(Duration::from_millis(500)).await;
    let store = Store::open(&dirs).unwrap();
    assert!(
        store.list_spec_receipts("m1", "stub").unwrap().is_empty(),
        "busy lane must not race"
    );
    assert_eq!(sup.ps().len(), 1, "serving instance must survive");
    assert_eq!(sup.ps()[0].state, "ready");
    sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn integration__autotune_schedule__no_chat_evidence_skips_model() {
    // The evidence floor: a GGUF model the gateway has never served
    // chat traffic for (embedder, TTS voice, fresh install) must never
    // be raced - no evictions, no children, no receipts.
    let (_t, dirs) = setup(&[("m1", 500)]);
    autotune_engine_assets(&dirs);
    let mut cfg = base_config();
    cfg.autotune = Some("scheduled".into());
    let sup = supervisor(&dirs, cfg, true);
    sup.autotune_schedule_tick();
    // Detached pass: give it a window to (wrongly) start racing, then
    // assert nothing landed and nothing was evicted or spawned.
    tokio::time::sleep(Duration::from_millis(750)).await;
    let store = Store::open(&dirs).unwrap();
    assert!(
        store.list_spec_receipts("m1", "stub").unwrap().is_empty(),
        "unevidenced model must not race"
    );
    assert!(sup.ps().is_empty(), "no instance may be spawned for a race");
    assert!(
        !sup.autotune_race_busy_for_test(),
        "pass must release the single-flight flag"
    );
}

#[tokio::test]
#[allow(non_snake_case)]
async fn regression__dropped_loader_future_does_not_wedge_next_ensure() {
    // Live-repro'd wedge: a request future dropped mid-spawn (client
    // timeout during cold load) used to strand `loading[key]` - every
    // later ensure parked forever on a Notify that never fired. The
    // LoadAbort guard must clear the slot so the NEXT loader runs.
    let (_t, dirs) = setup(&[("m1", 500)]);
    let sup = supervisor(&dirs, base_config(), true);
    // Drop a loader mid-spawn: 5ms is inside process spawn + health poll.
    {
        let mut fut = Box::pin(sup.ensure("m1"));
        tokio::select! {
            () = tokio::time::sleep(Duration::from_millis(5)) => {},
            _ = &mut fut => { /* load finished instantly - pin vacuous */ }
        }
    } // fut dropped HERE = the client-disconnect cancellation
    let budget = Duration::from_secs(
        sup.load_timeout_secs
            .expect("helper pins a fast load timeout"),
    ) * 3;
    let second = tokio::time::timeout(budget, sup.ensure("m1")).await;
    let ep = second
        .expect("subsequent ensure wedged after dropped loader")
        .expect("ensure");
    let url = match &ep.endpoint {
        blazar_core::Endpoint::Tcp { host, port } => format!("http://{host}:{port}"),
        blazar_core::Endpoint::Unix { .. } => unreachable!(),
    };
    blazar_core::tls::ensure_tls_provider();
    let body: serde_json::Value = reqwest::Client::new()
        .get(format!("{url}/health"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(body["status"], "ok");
    sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn integration__capacity_two__hot_cache_survives_cold_eviction() {
    // Prefix heat biases capacity eviction: with m1 hot (agent traffic)
    // and m2 cold, admitting m3 must evict m2 - NOT the hot m1 - even
    // though both are idle. Recency weighting: heat = hits/2 + 1.
    let (_t, dirs) = setup(&[("m1", 500), ("m2", 500), ("m3", 500)]);
    let mut cfg = base_config();
    cfg.max_loaded_models = 2;
    let sup = supervisor(&dirs, cfg, false);
    sup.ensure("m1").await.unwrap();
    sup.ensure("m2").await.unwrap();
    // m1 gets the traffic (three hits - decisively above cold 0).
    sup.note_prefix_hit("m1");
    sup.note_prefix_hit("m1");
    sup.note_prefix_hit("m1");
    assert!(sup.heat_of("m1") >= 2);
    assert_eq!(sup.heat_of("m2"), 0);
    sup.ensure("m3").await.unwrap();
    let names: Vec<String> = sup.ps().iter().map(|p| p.name.clone()).collect();
    assert!(
        names.contains(&"m1".to_string()),
        "hot m1 survived: {names:?}"
    );
    assert!(
        !names.contains(&"m2".to_string()),
        "cold m2 evicted: {names:?}"
    );
    sup.shutdown_all().await.unwrap();
}

#[test]
#[allow(non_snake_case)]
fn unit__prefix_heat__saturating_add_and_decay_semantics() {
    let (_t, dirs) = setup(&[]);
    let sup = supervisor(&dirs, base_config(), false);
    for _ in 0..10 {
        sup.note_prefix_hit("m");
    }
    // Saturating add: monotonically hot within a half-life window.
    assert_eq!(sup.heat_of("m"), 10);
    assert_eq!(sup.heat_of("never-seen"), 0);
    // Cap exists (bounded counters).
    for _ in 0..2000 {
        sup.note_prefix_hit("m");
    }
    assert!(sup.heat_of("m") <= 1000);
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
    assert_ne!(
        ep.endpoint,
        blazar_core::Endpoint::Tcp {
            host: "127.0.0.1".into(),
            port: 0
        }
    );
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
            mem_available_mib: None,
            gpus: vec![GpuInfo {
                name: "g".into(),
                description: "S".into(),
                total_mib: 24_000,
                free_mib: 24_000,
            }],
        },
        Arc::new(LlamaCppEngine::new(stub_manifest())),
    );
    s.reaper_interval = Duration::from_millis(200);
    s.load_timeout_secs = Some(4);
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
    assert!(
        !dirs.run_dir().join("m1.pid").exists(),
        "pid marker removed"
    );
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
async fn integration__crash__spawn_phase_death_counts_toward_circuit() {
    // num_ctx-storm shape (live 2026-09-13): a child that dies during
    // LOAD never becomes a tracked instance, so record_restart never
    // fired and the breaker stayed closed through 21 consecutive
    // spawn-retry 502s - each request re-pinned a doomed ctx and the
    // child died at context creation on every attempt. A death during
    // load is a crash-class restart; clean churn and pure load
    // timeouts stay uncounted (the wave-7 contract).
    let (_t, dirs) = setup(&[("m1", 500)]);
    let mut engine = LlamaCppEngine::new(stub_manifest());
    engine.child_env = vec![
        (
            "STUB_ARGV_FILE".into(),
            dirs.run_dir().join("argv.json").display().to_string(),
        ),
        (
            "STUB_DEVICES".into(),
            "stub-gpu: STUB GPU (24000 MiB, 24000 MiB free)".into(),
        ),
        // Health can never win the race (503 forever) and the child
        // dies (exit 3) mid-load on EVERY attempt: the 2-attempt spawn
        // loop exhausts with child_died set - the storm's exact shape.
        ("STUB_HEALTH_NEVER".into(), "1".into()),
        ("STUB_DIE_MS".into(), "300".into()),
    ];
    let hw = Hardware {
        physical_cores: 4,
        total_ram_mib: 16_000,
        mem_available_mib: None,
        gpus: vec![],
    };
    let mut s = Supervisor::new(
        dirs.clone(),
        base_config(),
        EventBus::default(),
        hw,
        Arc::new(engine),
    );
    s.load_timeout_secs = Some(3);
    s.shutdown_grace = Duration::from_secs(2);
    s.circuit_window = Duration::from_secs(30);
    s.max_restarts = 2;
    let sup = Arc::new(s);
    // Three doomed spawn rounds -> three counted restarts; the fourth
    // ensure must hit CircuitOpen (recent 3 > max 2) instead of another
    // full spawn-retry cycle.
    for round in 0..3 {
        let err = sup.ensure("m1").await.expect_err("child dies mid-load");
        assert!(
            matches!(err, SupervisionError::EngineCrashed(_)),
            "round {round}: {err}"
        );
    }
    match sup.ensure("m1").await {
        Ok(_) => panic!("breaker must open after repeated load-phase deaths"),
        Err(e) => assert!(matches!(e, SupervisionError::CircuitOpen(_)), "{e}"),
    }
    // Reset re-opens the path (unchanged breaker contract).
    sup.reset_circuit(Some("m1"));
    sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn integration__evict__concurrent_marks_clear_without_starving() {
    // The evicting mark is a tokio Mutex with an explicit clear (no
    // Drop): two concurrent evicts of one name must both complete and
    // leave the set empty - a leaked mark would defer every future
    // spawn of that name (starvation), the wedge class observed once
    // live as a /api/evict timeout.
    let (_t, dirs) = setup(&[("m1", 500)]);
    let sup = supervisor(&dirs, base_config(), false);
    sup.ensure("m1").await.unwrap();
    let (ra, rb) = tokio::join!(sup.evict("m1"), sup.evict("m1"));
    ra.unwrap();
    rb.unwrap();
    assert!(
        sup.evicting_is_empty().await,
        "evict marks must clear under concurrency"
    );
    // And the name is immediately spawnable again.
    sup.ensure("m1").await.unwrap();
    sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn integration__churn__clean_stop_run_never_trips_circuit() {
    // Live-repro'd bug: record_restart fired on EVERY successful spawn,
    // so a user stop->run churn x4 inside the breaker window opened the
    // circuit on the 5th and 503'd until `ps --reset`. A clean teardown
    // is a cold start, not a crash restart - churn must never trip.
    let (_t, dirs) = setup(&[("m1", 500)]);
    let sup = supervisor(&dirs, base_config(), false);
    for round in 0..6 {
        sup.evict("m1").await.unwrap();
        match sup.ensure("m1").await {
            Ok(_) => {}
            Err(e) => panic!("clean churn round {round} must not trip the breaker: {e}"),
        }
    }
    sup.shutdown_all().await.unwrap();
}

/// Scan /proc for stub children whose argv carries `marker`, returning
/// (pid, null-separated cmdline joined for the failure message).
fn live_children_with(marker: &str) -> Vec<(u32, String)> {
    let mut found = Vec::new();
    let Ok(entries) = std::fs::read_dir("/proc") else {
        return found;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        if !name.bytes().all(|b| b.is_ascii_digit()) {
            continue;
        }
        let Ok(cmdline) = std::fs::read(entry.path().join("cmdline")) else {
            continue;
        };
        if String::from_utf8_lossy(&cmdline).contains(marker) {
            let pid = name.parse::<u32>().unwrap_or(0);
            found.push((pid, String::from_utf8_lossy(&cmdline).replace('\0', " ")));
        }
    }
    found
}

#[tokio::test]
#[allow(non_snake_case)]
async fn integration__churn__no_orphaned_children_after_concurrent_lifecycle() {
    // The orphan gate: concurrent loads, capacity evictions and a
    // mid-flight crash must leave zero engine children alive once the
    // supervisor shuts down - proven against the real process table,
    // not the supervisor's own state. Model names embed this test's pid
    // so sibling test binaries running stubs cannot cross-match.
    let tag = format!("orphan-{}", std::process::id());
    let models = [format!("{tag}-a"), format!("{tag}-b"), format!("{tag}-c")];
    let named: Vec<(&str, u64)> = models.iter().map(|m| (m.as_str(), 500)).collect();
    let (_t, dirs) = setup(&named);
    let mut cfg = base_config();
    // Capacity 1: every racer's ensure evicts a live instance under the
    // others' feet, so spawn/evict/respawn all interleave.
    cfg.max_loaded_models = 1;
    let sup = supervisor(&dirs, cfg, true);

    let mut racers = Vec::new();
    for round in 0..3 {
        for m in &models {
            let sup = sup.clone();
            let m = m.clone();
            racers.push(tokio::spawn(async move {
                let _ = sup.ensure(&m).await;
            }));
        }
        // One hard crash mid-churn routes some teardown through the
        // reaper instead of the graceful path.
        if round == 1
            && let Some(row) = sup.ps().first()
        {
            unsafe { libc::kill(i32::try_from(row.pid).unwrap_or(-1), libc::SIGKILL) };
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    for r in racers {
        let _ = tokio::time::timeout(Duration::from_secs(30), r).await;
    }

    // Mid-run honesty: whatever is still alive must be an instance the
    // supervisor knows - an evicted name's child surviving here IS an
    // orphan even before shutdown. ps rows can trail the OS by one
    // reaper tick, so poll briefly before declaring a mismatch.
    let mut unknown_after_grace: Vec<(u32, String)> = Vec::new();
    for _ in 0..10 {
        let known: Vec<u32> = sup.ps().iter().map(|r| r.pid).collect();
        unknown_after_grace = live_children_with(&tag)
            .into_iter()
            .filter(|(pid, _)| !known.contains(pid))
            .collect();
        if unknown_after_grace.is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    assert!(
        unknown_after_grace.is_empty(),
        "children alive but unknown to the supervisor: {unknown_after_grace:?}"
    );

    sup.shutdown_all().await.unwrap();
    // SIGKILL delivery and reap need a beat before the scan is honest.
    tokio::time::sleep(Duration::from_millis(500)).await;
    let survivors = live_children_with(&tag);
    assert!(
        survivors.is_empty(),
        "orphaned stub children after shutdown: {survivors:?}"
    );
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
        Hardware {
            physical_cores: 4,
            total_ram_mib: 16_000,
            mem_available_mib: None,
            gpus: vec![],
        },
        Arc::new(engine),
    );
    let mut s = s;
    s.load_timeout_secs = Some(1);
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

/// Fail-fast contract: a child that DIES mid-load must surface as
/// `EngineCrashed` in seconds - never wait out the full `model_load_timeout`
/// polling a corpse (live bug: 0.5B model "timing out" for 180s while the
/// child had rejected `--device` and exited at 2s).
#[tokio::test]
#[allow(non_snake_case)]
async fn integration__ensure_fail_fast__child_death_beats_load_timeout() {
    let (_t, dirs) = setup(&[("m1", 500)]);
    let mut engine = LlamaCppEngine::new(stub_manifest());
    engine.child_env = vec![
        ("STUB_HEALTH_NEVER".into(), "1".into()),
        ("STUB_DIE_MS".into(), "300".into()),
    ];
    let s = Supervisor::new(
        dirs.clone(),
        base_config(),
        EventBus::default(),
        Hardware {
            physical_cores: 4,
            total_ram_mib: 16_000,
            mem_available_mib: None,
            gpus: vec![],
        },
        Arc::new(engine),
    );
    let mut s = s;
    s.load_timeout_secs = Some(30); // 2 attempts = 60s without the race
    s.shutdown_grace = Duration::from_secs(2);
    let sup = Arc::new(s);
    let t0 = Instant::now();
    let err = sup.ensure("m1").await.unwrap_err();
    assert!(
        t0.elapsed() < Duration::from_secs(10),
        "fail-fast violated: took {:?}",
        t0.elapsed()
    );
    assert!(
        matches!(err, SupervisionError::EngineCrashed(_)),
        "expected EngineCrashed, got {err}"
    );
    assert!(sup.ps().is_empty());
    sup.shutdown_all().await.unwrap();
}

#[test]
#[allow(non_snake_case)]
fn unit__capacity_auto__cpu_one_gpu_bytes_admission() {
    let (_t, dirs) = setup(&[("m", 500)]);
    let s = Supervisor::new(
        dirs,
        base_config(),
        EventBus::default(),
        Hardware {
            physical_cores: 4,
            total_ram_mib: 16_000,
            mem_available_mib: None,
            gpus: vec![],
        },
        Arc::new(LlamaCppEngine::new(stub_manifest())),
    );
    // CPU-only: pinned to a single instance; bytes admission inactive
    // (page cache is shared across processes, the heat model does not
    // reason about it).
    assert_eq!(s.instance_cap(), Some(1), "CPU-only -> 1");
    assert!(!s.bytes_admission_active());
    let gpu_hw = Hardware {
        physical_cores: 4,
        total_ram_mib: 16_000,
        mem_available_mib: None,
        gpus: vec![GpuInfo {
            name: "g".into(),
            description: "S".into(),
            total_mib: 24_576,
            free_mib: 24_576,
        }],
    };
    let s2 = Supervisor::new(
        BlazarDirs {
            config_dir: std::path::PathBuf::from("/tmp/p-cfg"),
            data_dir: std::path::PathBuf::from("/tmp/p-data"),
        },
        base_config(),
        EventBus::default(),
        gpu_hw,
        Arc::new(LlamaCppEngine::new(stub_manifest())),
    );
    // GPU + auto: no count cap; bytes admission against 24 GiB VRAM.
    assert_eq!(s2.instance_cap(), None);
    assert!(s2.bytes_admission_active());
    assert_eq!(s2.vram_budget_bytes(), 24_576 * 1024 * 1024);
    // Heterogeneous co-residency the old floor(VRAM/largest) formula
    // rejected: 23.4 GiB VRAM, a 4 GiB + a 3 GiB model = 7 GiB sum -
    // admitted (the old formula said capacity 5 by count but capacity 1
    // whenever the LARGEST model alone crossed VRAM/model).
    let hw_234 = Hardware {
        physical_cores: 4,
        total_ram_mib: 16_000,
        mem_available_mib: None,
        gpus: vec![GpuInfo {
            name: "g".into(),
            description: "S".into(),
            total_mib: 24_000,
            free_mib: 24_000,
        }],
    };
    let s3 = Supervisor::new(
        BlazarDirs {
            config_dir: std::path::PathBuf::from("/tmp/p-cfg"),
            data_dir: std::path::PathBuf::from("/tmp/p-data"),
        },
        base_config(),
        EventBus::default(),
        hw_234.clone(),
        Arc::new(LlamaCppEngine::new(stub_manifest())),
    );
    assert_eq!(s3.vram_budget_bytes(), 24_000 * 1024 * 1024);
    // Explicit count override wins over every heuristic.
    let mut cfg_n = base_config();
    cfg_n.max_loaded_models = 3;
    let s4 = Supervisor::new(
        BlazarDirs {
            config_dir: std::path::PathBuf::from("/tmp/p-cfg"),
            data_dir: std::path::PathBuf::from("/tmp/p-data"),
        },
        cfg_n,
        EventBus::default(),
        hw_234.clone(),
        Arc::new(LlamaCppEngine::new(stub_manifest())),
    );
    assert_eq!(s4.instance_cap(), Some(3));
    assert!(!s4.bytes_admission_active());
}

#[test]
#[allow(non_snake_case)]
fn unit__bytes_admission__heterogeneous_pair_coresides() {
    // The scenario the floor-division heuristic got wrong: an 8 GiB card,
    // a 0.5 GiB model live, a 5.8 GiB model incoming. Old formula:
    // floor(8188 MiB / 5800 MiB) = 1 -> evict the small. Sum admission:
    // 0.5 + 5.8 = 6.3 GiB <= 8 GiB budget -> co-reside.
    let mib = |m: u64| m * 1024 * 1024;
    let (_t, dirs) = setup(&[("small", mib(500)), ("big", mib(5800))]);
    let s = Supervisor::new(
        dirs,
        base_config(),
        EventBus::default(),
        Hardware {
            physical_cores: 4,
            total_ram_mib: 16_000,
            mem_available_mib: None,
            gpus: vec![GpuInfo {
                name: "g".into(),
                description: "S".into(),
                total_mib: 8188,
                free_mib: 8188,
            }],
        },
        Arc::new(LlamaCppEngine::new(stub_manifest())),
    );
    assert!(s.bytes_admission_active());

    let store = blazar_core::store::Store::open(&s.dirs).unwrap();
    let small = u64::try_from(store.get_model("small").unwrap().unwrap().bytes.max(0)).unwrap();
    let big = u64::try_from(store.get_model("big").unwrap().unwrap().bytes.max(0)).unwrap();
    drop(store);
    let budget = s.vram_budget_bytes();

    // Cold box: resident 0, any single model admitted (J3 spawn guard
    // owns the honest refusal for loads that cannot fit at all).
    assert_eq!(s.resident_bytes(), 0);
    assert!(big <= budget, "cold box must admit any single model");

    // Small resident + big incoming co-resides; a second big crosses.
    assert!(small + big <= budget, "pair must co-reside");
    assert!(
        small + big + big > budget,
        "second heavyweight must not fit"
    );
}

#[tokio::test]
#[allow(non_snake_case)]
#[allow(clippy::too_many_lines)] // one live-lifecycle story: spawn -> GET -> attach -> detach -> name-only detach
async fn integration__lora_hot_attach__llamacpp_reconciles_scales_over_spawn_set() {
    let (_t, dirs) = setup(&[("m1", 500)]);
    // Adapter on disk + store row: the spawn attaches it (--lora argv),
    // and the stub seeds its /lora-adapters state from that argv.
    let adapter = dirs.data_dir.join("anonymizer.gguf");
    std::fs::write(&adapter, b"gguf").unwrap();
    let store = Store::open(&dirs).unwrap();
    let row_id = store
        .add_lora("m1", &adapter.display().to_string(), 1.0, None)
        .unwrap();
    let sup = supervisor(&dirs, base_config(), true);
    let ep = sup.ensure("m1").await.expect("ensure");
    let url = match &ep.endpoint {
        blazar_core::Endpoint::Tcp { host, port } => format!("http://{host}:{port}"),
        blazar_core::Endpoint::Unix { .. } => unreachable!(),
    };
    // Spawn argv carried the adapter.
    tokio::time::sleep(Duration::from_millis(300)).await;
    let argv: Vec<String> =
        serde_json::from_str(&std::fs::read_to_string(dirs.run_dir().join("argv.json")).unwrap())
            .unwrap();
    assert!(
        argv.windows(2)
            .any(|w| w[0] == "--lora" && w[1] == adapter.display().to_string()),
        "--lora missing from spawn argv: {argv:?}"
    );
    blazar_core::tls::ensure_tls_provider();
    let client = reqwest::Client::new();
    // Children are bearer-gated: pull the per-child secret the gateway
    // itself uses (EngineRef.auth) for the direct probes below.
    let auth = sup
        .live_http_endpoints()
        .into_iter()
        .find(|e| e.name == "m1")
        .and_then(|e| e.auth);
    let lora_get = |client: &reqwest::Client| {
        let mut r = client.get(format!("{url}/lora-adapters"));
        if let Some(sec) = &auth {
            r = r.bearer_auth(sec);
        }
        r
    };
    // GET answers the upstream bare-array shape with the loaded row.
    let get: serde_json::Value = lora_get(&client)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let arr = get.as_array().expect("bare array (upstream shape)");
    assert_eq!(arr.len(), 1, "{arr:?}");
    assert_eq!(
        arr[0]["path"],
        serde_json::json!(adapter.display().to_string())
    );
    assert_eq!(arr[0]["id"], serde_json::json!(0));
    assert_eq!(arr[0]["scale"], serde_json::json!(1.0));

    // Hot attach of an already-loaded adapter: reconcile keeps it live.
    let verdict = sup
        .lora_hot_attach("m1", true, "anonymizer", &adapter.display().to_string())
        .await
        .unwrap();
    assert!(
        verdict.starts_with("hot: adapter anonymizer live on"),
        "{verdict}"
    );

    // Hot attach of an adapter NOT loaded: teaching deferral (files only
    // load at spawn), never a silent fall-back.
    let absent = dirs.data_dir.join("absent.gguf");
    std::fs::write(&absent, b"gguf").unwrap();
    let verdict = sup
        .lora_hot_attach("m1", true, "absent", &absent.display().to_string())
        .await
        .unwrap();
    assert!(verdict.contains("cannot hot-load"), "{verdict}");

    // Hot detach after a store rm: the reconcile POST drops the row's id
    // from the body, deactivating it at 0.0 on the resident.
    store.delete_lora(row_id).unwrap();
    let verdict = sup
        .lora_hot_attach("m1", false, "anonymizer", &adapter.display().to_string())
        .await
        .unwrap();
    assert!(verdict.contains("detached from"), "{verdict}");
    let get: serde_json::Value = lora_get(&client)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        get[0]["scale"],
        serde_json::json!(0.0),
        "deactivated, not unloaded: {get}"
    );

    // Re-activate, then detach by NAME ONLY (empty path) - callers that
    // omit the path (older CLIs) must still match the loaded adapter by
    // its file stem. The row leaves the store first, mirroring the CLI.
    let row_id = store
        .add_lora("m1", &adapter.display().to_string(), 1.0, None)
        .unwrap();
    let verdict = sup
        .lora_hot_attach("m1", true, "anonymizer", &adapter.display().to_string())
        .await
        .unwrap();
    assert!(verdict.starts_with("hot:"), "{verdict}");
    store.delete_lora(row_id).unwrap();
    let verdict = sup
        .lora_hot_attach("m1", false, "anonymizer", "")
        .await
        .unwrap();
    assert!(verdict.contains("detached from"), "{verdict}");
    let get: serde_json::Value = lora_get(&client)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(get[0]["scale"], serde_json::json!(0.0), "{get}");

    sup.shutdown_all().await.unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn integration__prewarm_reshape__cutover_same_key_new_argv() {
    // End-to-end prewarm against the real stub child: a queued reshape
    // spawns the replacement under `m1#warm` (old child keeps serving,
    // zero admission gap), then swaps it under `m1`. The stub's
    // STUB_ARGV_FILE is overwritten per spawn, so after the cutover it
    // holds the replacement's argv - pinning that the adopted slot
    // count (-np 4) reached a REAL child, not just the maps.
    let (_t, dirs) = setup(&[("m1", 500)]);
    let mut cfg = base_config();
    cfg.adaptive_slots = true;
    // Pin + override - the exact production shape that live-exposed the
    // LC4 precedence bug: without `adaptive_override`, the pin would
    // exclude m1 from adaptation entirely; with it, the pin seeds the
    // shape and the adopted value must supersede it at the prewarm
    // spawn (the spawn-side guard keys on exactly this delegation).
    cfg.model_overrides.insert(
        "m1".to_string(),
        blazar_core::ModelOverride {
            slots: Some(1),
            adaptive_override: Some(true),
            ..Default::default()
        },
    );
    let sup = supervisor(&dirs, cfg, true);
    let ep1 = sup.ensure("m1").await.expect("ensure");
    let old_port = match &ep1.endpoint {
        blazar_core::Endpoint::Tcp { port, .. } => *port,
        blazar_core::Endpoint::Unix { .. } => panic!("expected tcp endpoint"),
    };

    // Sustained demand (3 parks deep over a 1-slot shape -> peak 3 >=
    // 2x1) drives the fast-path adoption inside the production cycle;
    // the same reap call runs the drain that pre-warms and cuts over.
    // No decode hints exist yet, so the D5 decode-bound gate stands
    // down (None) - the adoption proceeds, as it must on a cold model.
    sup.note_slot_pressure("m1");
    sup.note_slot_pressure("m1");
    sup.note_slot_pressure("m1");
    let mut rx = sup.bus.subscribe();
    sup.adaptive_slots_reap().await; // tick 1: streak builds
    sup.adaptive_slots_reap().await; // tick 2: fast-path adopt + drain + cutover

    // Cutover landed: exactly one resident, still named `m1`, at the
    // adopted shape on a replacement child (new port).
    let rows = sup.ps();
    assert_eq!(rows.len(), 1, "single resident: {rows:?}");
    assert_eq!(rows[0].name, "m1");
    assert_eq!(rows[0].slots, Some(4), "adopted shape visible: {rows:?}");
    let rows_port: u16 = rows[0]
        .endpoint
        .rsplit(':')
        .next()
        .and_then(|p| p.parse().ok())
        .expect("row endpoint port");
    assert_ne!(rows_port, old_port, "replacement child, not the original");

    // The last spawn's argv carries the adopted shape.
    let argv: Vec<String> =
        serde_json::from_str(&std::fs::read_to_string(dirs.run_dir().join("argv.json")).unwrap())
            .unwrap();
    let pos = argv
        .iter()
        .position(|a| a == "-np")
        .expect("-np in spawn argv");
    assert_eq!(
        argv[pos + 1],
        "4",
        "adopted slots on the real child: {argv:?}"
    );

    // Both the path-specific and the completion-contract events fired.
    let mut prewarm = false;
    let mut reshaped = false;
    while let Ok(ev) = rx.try_recv() {
        match ev {
            blazar_runtime::BlazarEvent::SlotsPrewarmCutover { model, slots } => {
                assert_eq!(model, "m1");
                assert_eq!(slots, 4);
                prewarm = true;
            }
            blazar_runtime::BlazarEvent::SlotsReshaped { model, slots } => {
                assert_eq!(model, "m1");
                assert_eq!(slots, 4);
                reshaped = true;
            }
            _ => {}
        }
    }
    assert!(prewarm, "SlotsPrewarmCutover received");
    assert!(reshaped, "SlotsReshaped received (path-agnostic contract)");

    sup.shutdown_all().await.unwrap();
}

// --------------------------------------------------------------- openaihost

/// The openaihost lane (LiteRT-LM class): every `.litertlm` model must
/// collapse onto ONE shared registry child (`_openaihost`), never a
/// per-model instance, and the lane child must die with `shutdown_all`.
/// The stub doubles as the host: its `/v1/models` data-array answer is
/// exactly the lane health contract, and its loose argv parser accepts
/// the lane's `serve --host --port` grammar.
#[tokio::test]
#[allow(non_snake_case)]
async fn integration__openaihost_lane__shared_child_collapse_and_teardown() {
    let (_t, dirs) = setup(&[]);
    let store = Store::open(&dirs).unwrap();
    for name in ["x", "x2"] {
        let p = dirs.models_dir().join(format!("{name}.litertlm"));
        std::fs::write(&p, b"litert stub bundle").unwrap();
        store
            .upsert_model(&blazar_core::ModelRow {
                name: name.into(),
                repo: format!("litert-community/{name}"),
                quant: "LITERT".into(),
                path: p.display().to_string(),
                bytes: 1000,
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
    }
    // Engine row for the lane. Auth flags are stripped from the stub
    // manifest so the lane shape matches production litert: no child
    // auth is minted, the child stays on loopback + random port.
    let mut manifest = stub_manifest();
    manifest.flags.remove("--api-key-file");
    manifest.flags.remove("--api-key");
    store
        .upsert_engine(&blazar_core::store::EngineRow {
            tag: "litert-0".into(),
            asset: "pip:litert-lm==test".into(),
            sha256: String::new(),
            installed_at: 2,
            active: false,
            manifest: serde_json::to_string(&manifest).unwrap(),
            kind: blazar_core::engine_kind::EngineKind::OpenAiHost,
        })
        .unwrap();

    let sup = supervisor(&dirs, base_config(), false);
    let lane_key = blazar_runtime::supervisor::OPENAIHOST_KEY.to_string();

    // Cold path: a litertlm model ensure collapses onto the lane child.
    let e1 = sup.ensure_routed("x", None).await.expect("lane child");
    assert_eq!(e1.key, lane_key, "engine ref addresses the lane key");
    assert_eq!(e1.kind, blazar_core::engine_kind::EngineKind::OpenAiHost);
    assert!(e1.auth.is_none(), "no auth flags on the litert lane");
    let names: Vec<String> = sup.ps().into_iter().map(|p| p.name).collect();
    assert_eq!(names, vec![lane_key], "no per-model instance rows");
    let port_of = |e: &blazar_runtime::EngineRef| match &e.endpoint {
        blazar_core::Endpoint::Tcp { port, .. } => *port,
        blazar_core::Endpoint::Unix { .. } => panic!("lane child must be tcp"),
    };
    let port = port_of(&e1);

    // Warm path: a second bundle model rides the SAME child (one
    // registry server, one port - no second spawn).
    let e2 = sup
        .ensure_routed("x2", None)
        .await
        .expect("second model joins the shared child");
    assert_eq!(port_of(&e2), port, "both bundle models share one child");
    assert_eq!(sup.ps().len(), 1, "still exactly one lane instance");

    // Teardown: the lane child dies with the supervisor and its port
    // stops answering (V5: no orphaned child survives shutdown).
    sup.shutdown_all().await.unwrap();
    assert!(sup.ps().is_empty());
    let addr = format!("127.0.0.1:{port}");
    let mut refused = false;
    for _ in 0..50 {
        if tokio::net::TcpStream::connect(&addr).await.is_err() {
            refused = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(refused, "lane child port closed after shutdown_all");
}
