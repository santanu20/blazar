// Crash/teardown pins send real signals (SIGKILL, signal-0 liveness) to
// exact child pids — a unix contract; the gateway crash-retry suite
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
use blazar_runtime::engine::manifest::{probe as probe_manifest, Manifest};
use blazar_runtime::EventBus;
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
                vae_path: None,
                llm_path: None,
                llm_vision_path: None,
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
    // cache-reuse defaults OFF (elim sweep 2026-09-12: native slot cache
    // covers identical prefixes; --cache-reuse cost ~0.6s cold).
    assert!(
        !argv.contains(&"--cache-reuse".to_string()),
        "cache-reuse must not ride the default argv: {argv:?}"
    );
    // sessions dir is per-model under the data dir (path_safe suffix — derive)
    let sess_subdir = format!("sessions/{}/", blazar_core::profile::path_safe("m1"));
    assert!(
        argv.windows(2)
            .any(|w| w[0] == "--slot-save-path" && w[1].ends_with(&sess_subdir)),
        "slot-save-path per-model dir: {argv:?}"
    );
    assert!(argv.windows(2).any(|w| w[0] == "--alias" && w[1] == "m1"));
    // Default slots=0 auto: train ctx 40960 / base 16384 -> np 2 with the
    // total ctx scaled; ps still reports the per-slot ctx below.
    assert!(argv
        .windows(2)
        .any(|w| w[0] == "--ctx-size" && w[1] == "32768"));
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

#[tokio::test]
#[allow(non_snake_case)]
async fn regression__dropped_loader_future_does_not_wedge_next_ensure() {
    // Live-repro'd wedge: a request future dropped mid-spawn (client
    // timeout during cold load) used to strand `loading[key]` — every
    // later ensure parked forever on a Notify that never fired. The
    // LoadAbort guard must clear the slot so the NEXT loader runs.
    let (_t, dirs) = setup(&[("m1", 500)]);
    let sup = supervisor(&dirs, base_config(), true);
    // Drop a loader mid-spawn: 5ms is inside process spawn + health poll.
    {
        let mut fut = Box::pin(sup.ensure("m1"));
        tokio::select! {
            () = tokio::time::sleep(Duration::from_millis(5)) => {},
            _ = &mut fut => { /* load finished instantly — pin vacuous */ }
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
    // and m2 cold, admitting m3 must evict m2 — NOT the hot m1 — even
    // though both are idle. Recency weighting: heat = hits/2 + 1.
    let (_t, dirs) = setup(&[("m1", 500), ("m2", 500), ("m3", 500)]);
    let mut cfg = base_config();
    cfg.max_loaded_models = 2;
    let sup = supervisor(&dirs, cfg, false);
    sup.ensure("m1").await.unwrap();
    sup.ensure("m2").await.unwrap();
    // m1 gets the traffic (three hits — decisively above cold 0).
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
    // spawn-retry 502s — each request re-pinned a doomed ctx and the
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
        // loop exhausts with child_died set — the storm's exact shape.
        ("STUB_HEALTH_NEVER".into(), "1".into()),
        ("STUB_DIE_MS".into(), "300".into()),
    ];
    let hw = Hardware {
        physical_cores: 4,
        total_ram_mib: 16_000,
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
    // Three doomed spawn rounds → three counted restarts; the fourth
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
    // leave the set empty — a leaked mark would defer every future
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
    // so a user stop→run churn ×4 inside the breaker window opened the
    // circuit on the 5th and 503'd until `ps --reset`. A clean teardown
    // is a cold start, not a crash restart — churn must never trip.
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
/// `EngineCrashed` in seconds — never wait out the full `model_load_timeout`
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
    // rejected: 23.4 GiB VRAM, a 4 GiB + a 3 GiB model = 7 GiB sum —
    // admitted (the old formula said capacity 5 by count but capacity 1
    // whenever the LARGEST model alone crossed VRAM/model).
    let hw_234 = Hardware {
        physical_cores: 4,
        total_ram_mib: 16_000,
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
