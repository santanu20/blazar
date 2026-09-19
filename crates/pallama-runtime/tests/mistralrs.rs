//! mistral.rs engine lane integration tests: argv dialect (forced
//! loopback, flag-gated ctx/seqs/mmproj), `/v1/models`-loaded health
//! gate, and the install e2e (wiremock GitHub + a real tar.gz holding
//! the stub binary — exercises the tolerant probe + register flow).
#![allow(non_snake_case)] // house test-naming: unit__subject__behavior

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::time::Duration;

use pallama_core::engine_kind::EngineKind;
use pallama_core::profile::{Endpoint, Profile};
use pallama_core::store::ModelRow;
use pallama_core::PallamaDirs;
use pallama_runtime::engine::gh::GhClient;
use pallama_runtime::engine::manifest::Manifest;
use pallama_runtime::engine::{EngineManager, LOCAL_TAG};
use pallama_runtime::engine_impl::Engine as _;
use pallama_runtime::engine_impl::{mistralrs_argv, MistralRsEngine};
use pallama_runtime::EventBus;
use sha2::Digest as _;
use std::io::Write as _;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn model(path: &str, mmproj: Option<&str>) -> ModelRow {
    ModelRow {
        name: "m".into(),
        repo: "o/r".into(),
        quant: "Q4_K_M".into(),
        path: path.into(),
        bytes: 1000,
        sha256: None,
        mmproj_path: mmproj.map(str::to_string),
        shards: 1,
        arch: None,
        params: None,
        ctx_train: None,
        pulled_at: 0,
    }
}

fn profile(ctx: u32, argv: &[&str]) -> Profile {
    Profile {
        argv: argv.iter().map(std::string::ToString::to_string).collect(),
        warnings: Vec::new(),
        ctx,
        gpu: "auto",
        kv_est_bytes: None,
        ctx_autofit: None,
    }
}

fn flags(all: bool) -> BTreeSet<String> {
    let mut f = BTreeSet::new();
    if all {
        f.insert("--max-model-len".to_string());
        f.insert("--max-seqs".to_string());
        f.insert("--mmproj".to_string());
    }
    f
}

fn manifest() -> Manifest {
    Manifest {
        tag: "v0.9.3".into(),
        build_number: 9003,
        version_raw: "v0.9.3".into(),
        devices: Vec::new(),
        flags: flags(true),
        spec_types: Vec::new(),
        server_path: "/nonexistent/mistralrs".into(),
    }
}

#[test]
fn unit__mistralrs_argv__base_shape_forced_loopback() {
    let m = model("/data/m.gguf", Some("/data/mmproj.gguf"));
    let p = profile(8192, &["-np", "4"]);
    let argv = mistralrs_argv(
        &m,
        &p,
        &Endpoint::Tcp {
            host: "0.0.0.0".into(),
            port: 1234,
        },
        &flags(true),
    );
    // Loopback is forced regardless of the endpoint's host: the child is
    // unauthenticated; the gateway is the public face.
    assert_eq!(
        argv,
        vec![
            "serve",
            "-f",
            "/data/m.gguf",
            "--host",
            "127.0.0.1",
            "--port",
            "1234",
            "--no-ui",
            "--max-model-len",
            "8192",
            "--max-seqs",
            "4",
            "--mmproj",
            "/data/mmproj.gguf",
        ]
    );
}

#[test]
fn unit__mistralrs_argv__flag_gated_rules_dropped_when_unsupported() {
    let m = model("/data/m.gguf", Some("/data/mmproj.gguf"));
    let p = profile(8192, &["--parallel=7"]);
    let argv = mistralrs_argv(
        &m,
        &p,
        &Endpoint::Tcp {
            host: "127.0.0.1".into(),
            port: 8080,
        },
        &flags(false),
    );
    assert_eq!(
        argv,
        vec![
            "serve",
            "-f",
            "/data/m.gguf",
            "--host",
            "127.0.0.1",
            "--port",
            "8080",
            "--no-ui"
        ]
    );
}

#[test]
fn unit__mistralrs_argv__parallel_eq_form_mined() {
    let m = model("/m.gguf", None);
    let p = profile(0, &["--parallel=7"]);
    let argv = mistralrs_argv(
        &m,
        &p,
        &Endpoint::Tcp {
            host: "h".into(),
            port: 9,
        },
        &flags(true),
    );
    let i = argv
        .iter()
        .position(|a| a == "--max-seqs")
        .expect("max-seqs present");
    assert_eq!(argv[i + 1], "7");
    assert!(!argv.contains(&"--mmproj".to_string()));
}

#[test]
fn unit__mistralrs_argv__pa_memory_fraction_passthrough_gated() {
    // the profile dialect's tuning pairs ride through verbatim when the
    // binary has the flag, and are dropped when it does not
    let m = model("/m.gguf", None);
    let mut f = flags(true);
    f.insert("--pa-memory-fraction".to_string());
    let p = profile(8192, &["-np", "2", "--pa-memory-fraction", "0.2"]);
    let argv = mistralrs_argv(
        &m,
        &p,
        &Endpoint::Tcp {
            host: "h".into(),
            port: 9,
        },
        &f,
    );
    let i = argv
        .iter()
        .position(|a| a == "--pa-memory-fraction")
        .expect("fraction forwarded");
    assert_eq!(argv[i + 1], "0.2");

    let argv_off = mistralrs_argv(
        &m,
        &p,
        &Endpoint::Tcp {
            host: "h".into(),
            port: 9,
        },
        &flags(true), // no --pa-memory-fraction
    );
    assert!(!argv_off.contains(&"--pa-memory-fraction".to_string()));
}

#[test]
fn unit__mistralrs_argv__unix_endpoint_port_placeholder() {
    let m = model("/m.gguf", None);
    let argv = mistralrs_argv(
        &m,
        &profile(0, &[]),
        &Endpoint::Unix {
            socket: "/sock".into(),
        },
        &flags(true),
    );
    let i = argv.iter().position(|a| a == "--port").expect("port token");
    assert_eq!(argv[i + 1], "0");
}

#[tokio::test]
async fn integration__mistralrs_health__waits_for_models_loaded() {
    let api = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "data": [{"id": "m", "status": "loaded"}]
        })))
        .mount(&api)
        .await;
    let (host, port) = host_port(&api.uri());
    let e = MistralRsEngine::new(manifest());
    e.health_check(&Endpoint::Tcp { host, port }, Duration::from_secs(2))
        .await
        .expect("loaded status passes the health gate");
}

#[tokio::test]
async fn integration__mistralrs_health__missing_status_tolerated_as_loaded() {
    let api = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "data": [{"id": "m"}]
        })))
        .mount(&api)
        .await;
    let (host, port) = host_port(&api.uri());
    let e = MistralRsEngine::new(manifest());
    e.health_check(&Endpoint::Tcp { host, port }, Duration::from_secs(2))
        .await
        .expect("missing status field tolerated");
}

#[tokio::test]
async fn integration__mistralrs_health__never_loaded_times_out() {
    let api = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "data": [{"id": "m", "status": "loading"}]
        })))
        .mount(&api)
        .await;
    let (host, port) = host_port(&api.uri());
    let e = MistralRsEngine::new(manifest());
    let err = e
        .health_check(&Endpoint::Tcp { host, port }, Duration::from_millis(400))
        .await
        .expect_err("loading never becomes loaded within the window");
    let msg = format!("{err}");
    assert!(
        msg.contains("/v1/models"),
        "error names the probe endpoint: {msg}"
    );
    assert!(
        msg.contains("model_load_timeout") || msg.contains("timeout"),
        "error names the timeout: {msg}"
    );
}

fn host_port(uri: &str) -> (String, u16) {
    let u = uri.trim_start_matches("http://");
    let (h, p) = u.rsplit_once(':').expect("host:port");
    (h.to_string(), p.parse().expect("port"))
}

// ---- install e2e (wiremock GitHub + real tar.gz + stub binary) ----

fn sha256_hex(b: &[u8]) -> String {
    format!("{:x}", sha2::Sha256::digest(b))
}

/// mistral.rs-style release tar.gz: `mistralrs` executable at the root
/// (release archives carry the binary without a wrapper directory; the
/// finder's exact-name walk handles either layout).
fn fixture_mistralrs_tarball() -> Vec<u8> {
    let bin = std::fs::read(PathBuf::from(env!("CARGO_BIN_EXE_stub-llama-server")))
        .expect("read stub binary");
    let mut tarbuf = Vec::new();
    {
        let mut builder = tar::Builder::new(&mut tarbuf);
        let mut header = tar::Header::new_gnu();
        header.set_size(bin.len() as u64);
        header.set_entry_type(tar::EntryType::Regular);
        header.set_mode(0o755);
        header.set_cksum();
        builder
            .append_data(&mut header, "mistralrs", bin.as_slice())
            .unwrap();
        builder.finish().unwrap();
    }
    let mut gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    gz.write_all(&tarbuf).unwrap();
    gz.finish().unwrap()
}

/// The asset this host's mistral.rs lane requests (mirrors
/// `gh::mistralrs_asset_picks` for GPU-less machines): the suite runs on
/// linux x64, linux arm64, macOS arm64 and Windows x64, so every CI lane
/// asks for the CPU/Metal build, never a CUDA pick.
fn host_asset() -> (&'static str, &'static str) {
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        ("mistralrs-metal-aarch64-apple-darwin.tar.gz", "metal")
    } else if cfg!(all(target_os = "windows", target_arch = "x86_64")) {
        ("mistralrs-cpu-x86_64-pc-windows-msvc.zip", "cpu")
    } else if cfg!(all(target_os = "linux", target_arch = "aarch64")) {
        ("mistralrs-cpu-aarch64-unknown-linux-gnu.tar.gz", "cpu")
    } else {
        ("mistralrs-cpu-x86_64-unknown-linux-gnu.tar.gz", "cpu")
    }
}

/// Fixture archive in the format the host lane downloads: tar.gz on the
/// unix lanes, zip for the Windows CPU lane.
fn fixture_mistralrs_archive() -> Vec<u8> {
    if cfg!(windows) {
        let bin = std::fs::read(PathBuf::from(env!("CARGO_BIN_EXE_stub-llama-server")))
            .expect("read stub binary");
        let mut buf = std::io::Cursor::new(Vec::new());
        let mut zip = zip::ZipWriter::new(&mut buf);
        let opts: zip::write::SimpleFileOptions =
            zip::write::SimpleFileOptions::default().unix_permissions(0o755);
        zip.start_file("mistralrs", opts).unwrap();
        zip.write_all(&bin).unwrap();
        zip.finish().unwrap();
        buf.into_inner()
    } else {
        fixture_mistralrs_tarball()
    }
}

async fn mount_mistralrs_release(api: &MockServer, tag: &str, assets: &[(&str, Vec<u8>)]) {
    let body = serde_json::json!({
        "tag_name": tag,
        "prerelease": false,
        // Old timestamp: a missing asset must NOT trigger the fresh-release
        // upload retry loop (tests would stall 2 minutes).
        "published_at": "2020-01-01T00:00:00Z",
        "assets": assets.iter().map(|(name, bytes)| serde_json::json!({
            "name": name,
            "digest": format!("sha256:{}", sha256_hex(bytes)),
            "size": u64::try_from(bytes.len()).unwrap_or(u64::MAX),
            "browser_download_url": format!("{}/download/{tag}/{name}", api.uri()),
        })).collect::<Vec<_>>(),
    });
    Mock::given(method("GET"))
        .and(path(format!(
            "/repos/EricLBuehler/mistral.rs/releases/tags/{tag}"
        )))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(api)
        .await;
    for (name, bytes) in assets {
        Mock::given(method("GET"))
            .and(path(format!("/download/{tag}/{name}")))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(bytes.clone()))
            .mount(api)
            .await;
    }
}

fn manager(dirs: &PallamaDirs, api_uri: &str) -> EngineManager {
    let gh = GhClient::with_base(api_uri, None).unwrap();
    EngineManager {
        dirs: dirs.clone(),
        gh,
        bus: EventBus::default(),
        asset_override: "auto".into(),
    }
}

fn tmp_dirs() -> (tempfile::TempDir, PallamaDirs) {
    let t = tempfile::tempdir().unwrap();
    let dirs = PallamaDirs {
        config_dir: t.path().join("cfg"),
        data_dir: t.path().join("data"),
    };
    dirs.ensure().unwrap();
    (t, dirs)
}

#[tokio::test]
async fn integration__install_mistralrs__latest_resolves_registers_activates() {
    let api = MockServer::start().await;
    let (asset_name, asset_label) = host_asset();
    let asset = fixture_mistralrs_archive();
    mount_mistralrs_release(&api, "v0.9.3", &[(asset_name, asset)]).await;
    Mock::given(method("GET"))
        .and(path("/repos/EricLBuehler/mistral.rs/releases"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!([
            {"tag_name": "v0.9.3", "prerelease": false, "assets": []}
        ])))
        .mount(&api)
        .await;

    let (t, dirs) = tmp_dirs();
    let mgr = manager(&dirs, &api.uri());
    let row = mgr.update_mistralrs(None).await.expect("install succeeds");

    assert_eq!(row.tag, "v0.9.3");
    assert_eq!(row.kind, EngineKind::MistralRs);
    assert!(row.active, "install activates the engine");
    assert_eq!(row.asset, asset_label);
    // vtag serial: 0*1_000_000 + 9*1_000 + 3
    let m: Manifest = serde_json::from_str(&row.manifest).unwrap();
    assert_eq!(m.build_number, 9003);
    let bin = dirs.engines_dir().join("v0.9.3").join("mistralrs");
    assert!(bin.is_file(), "binary installed at {}", bin.display());
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        let mode = std::fs::metadata(&bin).unwrap().permissions().mode();
        assert!(mode & 0o111 != 0, "binary is executable");
    }
    drop(mgr);
    drop(t);
}

#[tokio::test]
async fn integration__install_mistralrs__explicit_tag_uses_tags_endpoint() {
    let api = MockServer::start().await;
    let (asset_name, _asset_label) = host_asset();
    let asset = fixture_mistralrs_archive();
    mount_mistralrs_release(&api, "v0.9.2", &[(asset_name, asset)]).await;

    let (t, dirs) = tmp_dirs();
    let mgr = manager(&dirs, &api.uri());
    let row = mgr
        .update_mistralrs(Some("v0.9.2"))
        .await
        .expect("explicit tag installs");
    assert_eq!(row.tag, "v0.9.2");
    drop(mgr);
    drop(t);
}

#[tokio::test]
async fn integration__install_mistralrs__missing_asset_error_lists_wanted() {
    let api = MockServer::start().await;
    // Only a never-pickable CUDA asset exists; the wanted-list error must
    // name the asset this host asked for so the user can see what was
    // searched. cuda999 sorts above any real driver floor, and GPU-less
    // hosts never request CUDA picks at all, so the decoy is never chosen
    // on any runner or dev box.
    mount_mistralrs_release(
        &api,
        "v0.9.3",
        &[(
            "mistralrs-cuda999-sm89-x86_64-unknown-linux-gnu.tar.gz",
            vec![1, 2, 3],
        )],
    )
    .await;

    let (t, dirs) = tmp_dirs();
    let mgr = manager(&dirs, &api.uri());
    let err = mgr
        .update_mistralrs(Some("v0.9.3"))
        .await
        .expect_err("no matching asset must fail fast");
    let msg = format!("{err}");
    assert!(
        msg.contains(host_asset().0),
        "error lists wanted picks: {msg}"
    );
    // Nothing registered on failure.
    let store = pallama_core::store::Store::open(&dirs).unwrap();
    assert_eq!(store.list_engines().unwrap().len(), 0);
    assert_ne!(LOCAL_TAG, "");
    drop(mgr);
    drop(t);
}

// --- staging view ---------------------------------------------------------
// mistralrs discovers projectors by scanning the model file's directory
// (verified in upstream v0.9.3 gguf_discovery.rs: list_local_gguf_
// companions reads the whole dir; one mmproj* = auto-selected, several =
// hard error, '' override = rejected). Pallama's flat shared models dir
// must therefore be narrowed to a per-model view at spawn.

fn touch(p: &std::path::Path) {
    std::fs::write(p, b"gguf").unwrap();
}

#[test]
fn staging__stray_projector_excluded_from_view() {
    let t = tempfile::tempdir().unwrap();
    let models = t.path().join("models");
    std::fs::create_dir_all(&models).unwrap();
    touch(&models.join("m.gguf"));
    // Another model's projector living in the shared dir — the poison.
    touch(&models.join("mmproj-F16.gguf"));
    let staging = t.path().join("staging");
    std::fs::create_dir_all(&staging).unwrap();

    let engine = MistralRsEngine::with_staging(manifest(), Vec::new(), Some(staging));
    let m = model(&models.join("m.gguf").to_string_lossy(), None);
    let argv = engine.build_argv(&m, &profile(0, &[]), &tcp());
    let staged_f = argv
        .windows(2)
        .find(|w| w[0] == "-f")
        .map(|w| w[1].clone())
        .unwrap();
    assert!(staged_f.contains("staging"), "{argv:?}");
    let view = std::path::Path::new(&staged_f).parent().unwrap();
    let mut names: Vec<String> = std::fs::read_dir(view)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    assert_eq!(names, vec!["m.gguf".to_string()], "view must isolate");
    assert!(!argv.iter().any(|a| a == "--mmproj"), "{argv:?}");
}

#[test]
fn staging__own_projector_linked_and_passed() {
    let t = tempfile::tempdir().unwrap();
    let models = t.path().join("models");
    std::fs::create_dir_all(&models).unwrap();
    touch(&models.join("m.gguf"));
    touch(&models.join("mmproj-F16.gguf"));
    let staging = t.path().join("staging");
    std::fs::create_dir_all(&staging).unwrap();

    let engine =
        MistralRsEngine::with_staging(manifest_flags(&["--mmproj"]), Vec::new(), Some(staging));
    let m = model(
        &models.join("m.gguf").to_string_lossy(),
        Some(&models.join("mmproj-F16.gguf").to_string_lossy()),
    );
    let argv = engine.build_argv(&m, &profile(0, &[]), &tcp());
    let staged_mm = argv
        .windows(2)
        .find(|w| w[0] == "--mmproj")
        .map(|w| std::path::PathBuf::from(&w[1]).clone())
        .unwrap();
    assert!(staged_mm.to_string_lossy().contains("staging"), "{argv:?}");
    assert!(staged_mm.is_file(), "symlink resolves to content");
}

#[test]
fn staging__shard_set_grouped_but_neighbors_excluded() {
    let t = tempfile::tempdir().unwrap();
    let models = t.path().join("models");
    std::fs::create_dir_all(&models).unwrap();
    touch(&models.join("big-00001-of-00002.gguf"));
    touch(&models.join("big-00002-of-00002.gguf"));
    touch(&models.join("big-v2.gguf")); // prefix-sharing impostor
    touch(&models.join("other.gguf"));
    let staging = t.path().join("staging");
    std::fs::create_dir_all(&staging).unwrap();

    let engine = MistralRsEngine::with_staging(manifest(), Vec::new(), Some(staging));
    let m = model(
        &models.join("big-00001-of-00002.gguf").to_string_lossy(),
        None,
    );
    let argv = engine.build_argv(&m, &profile(0, &[]), &tcp());
    let staged_f = argv
        .windows(2)
        .find(|w| w[0] == "-f")
        .map(|w| std::path::PathBuf::from(&w[1]))
        .unwrap();
    let view = staged_f.parent().unwrap();
    let mut names: Vec<String> = std::fs::read_dir(view)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    assert_eq!(
        names,
        vec![
            "big-00001-of-00002.gguf".to_string(),
            "big-00002-of-00002.gguf".to_string(),
        ],
        "shards grouped, impostors excluded"
    );
}

#[test]
fn staging__none_falls_back_to_raw_store_paths() {
    let engine = MistralRsEngine::with_env(manifest(), Vec::new());
    let m = model("/store/models/m.gguf", None);
    let argv = engine.build_argv(&m, &profile(0, &[]), &tcp());
    assert!(
        argv.windows(2)
            .any(|w| w[0] == "-f" && w[1] == "/store/models/m.gguf"),
        "{argv:?}"
    );
}

fn tcp() -> Endpoint {
    Endpoint::Tcp {
        host: "127.0.0.1".into(),
        port: 1234,
    }
}

fn manifest_flags(extra: &[&str]) -> Manifest {
    let mut m = manifest();
    for f in extra {
        m.flags.insert(f.to_string());
    }
    m
}

#[test]
fn unit__mistralrs_argv__tuning_passthrough_pairs_and_bools() {
    // compile_mistralrs emits tuning tokens into profile.argv; the argv
    // builder must forward every pair (and bare bool) the manifest knows,
    // order-preserving, and drop the rest.
    let m = model("/data/m.gguf", None);
    let p = profile(
        8192,
        &[
            "--max-batch-size",
            "4",
            "--prefix-cache-n",
            "0",
            "--lora",
            "style=/loras/style.gguf",
            "--mtp",
            "--mtp-model",
            "mtp-draft",
            "--mtp-draft-sampling",
            "greedy",
            "--device-layers",
            "0:12",
            "--disable-metrics",
        ],
    );
    let engine = MistralRsEngine::with_env(
        manifest_flags(&[
            "--max-batch-size",
            "--prefix-cache-n",
            "--lora",
            "--mtp",
            "--mtp-model",
            "--mtp-draft-sampling",
            "--device-layers",
            "--disable-metrics",
        ]),
        Vec::new(),
    );
    let argv = engine.build_argv(&m, &p, &tcp());
    for pair in [
        ("--max-batch-size", "4"),
        ("--prefix-cache-n", "0"),
        ("--lora", "style=/loras/style.gguf"),
        ("--mtp-model", "mtp-draft"),
        ("--mtp-draft-sampling", "greedy"),
        ("--device-layers", "0:12"),
    ] {
        assert!(
            argv.windows(2).any(|w| w == [pair.0, pair.1]),
            "missing {pair:?}: {argv:?}"
        );
    }
    for bare in ["--mtp", "--disable-metrics"] {
        assert!(argv.iter().any(|t| t == bare), "missing {bare}: {argv:?}");
    }
    let mtp = argv.iter().position(|t| t == "--mtp").unwrap();
    let mtp_model = argv.iter().position(|t| t == "--mtp-model").unwrap();
    assert!(
        mtp < mtp_model,
        "family switch must lead dependents: {argv:?}"
    );
}

#[test]
fn unit__mistralrs_argv__tuning_passthrough_gated_by_manifest() {
    // Old engine (manifest lacks the tuning flags): every tuning token is
    // dropped from the child argv — the compile-time warning already taught.
    let m = model("/data/m.gguf", None);
    let p = profile(
        8192,
        &[
            "--max-batch-size",
            "4",
            "--mtp",
            "--device-layers",
            "0:12",
            "--disable-access-log",
        ],
    );
    let engine = MistralRsEngine::with_env(manifest(), Vec::new());
    let argv = engine.build_argv(&m, &p, &tcp());
    for flag in [
        "--max-batch-size",
        "--mtp",
        "--device-layers",
        "--disable-access-log",
    ] {
        assert!(!argv.iter().any(|t| t == flag), "{flag} leaked: {argv:?}");
    }
}

#[test]
fn unit__mistralrs_argv__extra_args_forwarded_manifest_gated() {
    // extra_args ride profile.argv after compile-time strict validation;
    // the generic passthrough forwards manifest-known pairs verbatim
    // (--isq q4k) and drops tokens an old engine lacks.
    let m = model("/data/m.d", None);
    let p = profile(8192, &["-np", "2", "--isq", "q4k"]);
    let engine = MistralRsEngine::with_env(
        manifest_flags(&["--max-model-len", "--isq", "--no-ui"]),
        Vec::new(),
    );
    let argv = engine.build_argv(&m, &p, &tcp());
    assert!(
        argv.windows(2).any(|w| w == ["--isq", "q4k"]),
        "isq pair must reach the child argv: {argv:?}"
    );

    let stale =
        MistralRsEngine::with_env(manifest_flags(&["--max-model-len", "--no-ui"]), Vec::new());
    let argv = stale.build_argv(&m, &p, &tcp());
    assert!(
        !argv.iter().any(|t| t == "--isq"),
        "engine without the flag must not receive it: {argv:?}"
    );
}
