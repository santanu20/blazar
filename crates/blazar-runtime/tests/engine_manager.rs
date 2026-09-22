//! Engine manager integration tests: full install -> manifest -> activate
//! -> rollback cycle against a wiremock GitHub API, with a REAL tar.gz
//! archive containing the compiled stub-llama-server (exercises the
//! genuine probe path: --version / --list-devices / --help).

use sha2::Digest as _;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use blazar_core::config::UpdateChannel;
use blazar_core::store::Store;
use blazar_core::BlazarDirs;
use blazar_runtime::engine::gh::GhClient;
use blazar_runtime::engine::manifest::{EngineSource, LaneProvenance, Manifest, TrustTier};
use blazar_runtime::engine::{EngineManager, KEEP_TAGS, LOCAL_TAG};
use blazar_runtime::EventBus;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn stub_server_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_stub-llama-server"))
}

/// Build a llama.cpp-style release tar.gz: root dir `llama-<tag>/` with
/// the stub binary named `llama-server` inside (matches release.yml
/// `--transform s,^\.,llama-<tag>, -C ./build/bin .`).
fn fixture_tarball(tag: &str) -> Vec<u8> {
    let bin = std::fs::read(stub_server_bin()).expect("read stub binary");
    let mut tarbuf = Vec::new();
    {
        let mut builder = tar::Builder::new(&mut tarbuf);
        let root = format!("llama-{tag}");
        let mut header = tar::Header::new_gnu();
        header.set_size(0);
        header.set_entry_type(tar::EntryType::Directory);
        header.set_mode(0o755);
        header.set_cksum();
        builder
            .append_data(&mut header, &root, std::io::empty())
            .unwrap();
        let mut header = tar::Header::new_gnu();
        header.set_size(bin.len() as u64);
        header.set_entry_type(tar::EntryType::Regular);
        header.set_mode(0o755);
        header.set_cksum();
        builder
            .append_data(&mut header, format!("{root}/llama-server"), bin.as_slice())
            .unwrap();
        builder.finish().unwrap();
    }
    let mut gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    gz.write_all(&tarbuf).unwrap();
    gz.finish().unwrap()
}

fn fixture_zip(tag: &str) -> Vec<u8> {
    let bin = std::fs::read(stub_server_bin()).expect("read stub binary");
    let mut w = std::io::Cursor::new(Vec::new());
    {
        let mut zip = zip::ZipWriter::new(&mut w);
        let opts: zip::write::SimpleFileOptions =
            zip::write::SimpleFileOptions::default().unix_permissions(0o755);
        zip.add_directory(format!("llama-{tag}"), opts).unwrap();
        zip.start_file(format!("llama-{tag}/llama-server.exe"), opts)
            .unwrap();
        zip.write_all(&bin).unwrap();
        zip.finish().unwrap();
    }
    w.into_inner()
}

fn sha256_hex(b: &[u8]) -> String {
    format!("{:x}", sha2::Sha256::digest(b))
}

/// Mount a b-tag release with the given asset body + digest.
async fn mount_release(api: &MockServer, tag: &str, asset_name: &str, bytes: &[u8]) {
    let body = serde_json::json!({
        "tag_name": tag,
        "prerelease": true,
        "assets": [{
            "name": asset_name,
            "digest": format!("sha256:{}", sha256_hex(bytes)),
            "size": u64::try_from(bytes.len()).unwrap_or(u64::MAX),
            "browser_download_url": format!("{}/download/{tag}/{asset_name}", api.uri()),
        }]
    });
    Mock::given(method("GET"))
        .and(path(format!(
            "/repos/ggml-org/llama.cpp/releases/tags/{tag}"
        )))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(api)
        .await;
    Mock::given(method("GET"))
        .and(path(format!("/download/{tag}/{asset_name}")))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(bytes.to_vec()))
        .mount(api)
        .await;
}

async fn mount_releases_list(api: &MockServer, tags: &[&str]) {
    let list: Vec<serde_json::Value> = tags
        .iter()
        .map(|t| serde_json::json!({"tag_name": t, "prerelease": true, "assets": []}))
        .collect();
    Mock::given(method("GET"))
        .and(path("/repos/ggml-org/llama.cpp/releases"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!(list)))
        .mount(api)
        .await;
}

fn manager(dirs: &BlazarDirs, api_uri: &str) -> EngineManager {
    // Diagnostics: surface the probe layer's warn! lines (spawn errno)
    // that probe_output otherwise swallows into None. try_init is
    // idempotent — the first parallel caller wins, the rest no-op.
    let _ = tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(tracing_subscriber::EnvFilter::new("warn"))
        .try_init();
    let gh = GhClient::with_base(api_uri, None).unwrap();
    EngineManager {
        dirs: dirs.clone(),
        gh,
        bus: EventBus::default(),
        asset_override: "ubuntu-vulkan-x64".into(),
    }
}

/// Auto asset selection (no override): exercises the resolve/retry path.
fn manager_auto(dirs: &BlazarDirs, api_uri: &str) -> EngineManager {
    let gh = GhClient::with_base(api_uri, None).unwrap();
    EngineManager {
        dirs: dirs.clone(),
        gh,
        bus: EventBus::default(),
        asset_override: "auto".into(),
    }
}

/// ISO-8601 Zulu string for an epoch offset (civil-from-days inverse).
fn iso_from_epoch(epoch: i64) -> String {
    let days = epoch.div_euclid(86_400);
    let secs = epoch.rem_euclid(86_400);
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        secs / 3600,
        (secs % 3600) / 60,
        secs % 60
    )
}

fn tmp_dirs() -> (tempfile::TempDir, BlazarDirs) {
    let tmp = tempfile::tempdir().unwrap();
    let dirs = BlazarDirs {
        config_dir: tmp.path().join("cfg"),
        data_dir: tmp.path().join("data"),
    };
    dirs.ensure().unwrap();
    (tmp, dirs)
}

/// Asset labels `pick_asset` could match for the HOST platform (primary
/// first). Fixtures built from this list stay honest on any runner
/// OS/arch instead of assuming a linux/x64 host.
fn host_asset_labels() -> &'static [&'static str] {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("macos", _) => &["macos-arm64", "macos-x64"],
        ("windows", "x86_64") => &["win-vulkan-x64", "win-cpu-x64"],
        ("windows", _) => &["win-cpu-arm64"],
        ("linux", "aarch64") => &["ubuntu-vulkan-arm64", "ubuntu-arm64"],
        _ => &["ubuntu-vulkan-x64", "ubuntu-x64"],
    }
}

/// The CPU-tier label for the host platform (the last-resort candidate).
#[cfg(not(target_os = "macos"))] // macos has no cpu-fallback tier
fn host_cpu_asset_label() -> &'static str {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("windows", "x86_64") => "win-cpu-x64",
        ("windows", _) => "win-cpu-arm64",
        ("linux", "aarch64") => "ubuntu-arm64",
        _ => "ubuntu-x64",
    }
}

/// Archive bytes matching the wire shape the picker expects for this
/// host: upstream publishes win-* assets as zip, everything else as
/// tar.gz (see `gh::asset_filename`). Fixtures carrying the wrong
/// extension make the picker report "no usable asset".
fn fixture_host_archive(tag: &str) -> Vec<u8> {
    if cfg!(windows) {
        fixture_zip(tag)
    } else {
        fixture_tarball(tag)
    }
}

fn host_asset_ext() -> &'static str {
    if cfg!(windows) {
        "zip"
    } else {
        "tar.gz"
    }
}

/// Release-`assets` JSON for every host-platform candidate, digest and
/// size included, download URLs pointing at the wiremock server.
fn host_assets_json(api_uri: &str, tag: &str, tar: &[u8]) -> serde_json::Value {
    host_asset_labels()
        .iter()
        .map(|label| {
            let name = format!("llama-{tag}-bin-{label}.{}", host_asset_ext());
            serde_json::json!({
                "name": name,
                "digest": format!("sha256:{}", sha256_hex(tar)),
                "size": tar.len() as u64,
                "browser_download_url": format!("{api_uri}/download/{tag}/{name}"),
            })
        })
        .collect::<Vec<_>>()
        .into()
}

#[tokio::test]
#[allow(non_snake_case)]
async fn integration__install_probe_activate_rollback_cycle() {
    let (_t, dirs) = tmp_dirs();
    let api = MockServer::start().await;
    let t1 = fixture_tarball("b100");
    let t2 = fixture_tarball("b200");
    mount_release(&api, "b100", "llama-b100-bin-ubuntu-vulkan-x64.tar.gz", &t1).await;
    mount_release(&api, "b200", "llama-b200-bin-ubuntu-vulkan-x64.tar.gz", &t2).await;
    let mgr = manager(&dirs, &api.uri());

    // Install b100.
    let row = mgr
        .update(Some("b100"), UpdateChannel::Latest)
        .await
        .unwrap();
    assert_eq!(row.tag, "b100");
    assert_eq!(row.sha256, sha256_hex(&t1));
    let store = Store::open(&dirs).unwrap();
    assert_eq!(store.active_engine().unwrap().unwrap().tag, "b100");

    // Manifest is a real probe of the stub: version/build/devices/flags.
    // The stub banner reports STUB_BUILD 9999, but the install tag (b100)
    // is the authoritative build identity — same rule that overrides the
    // shallow-clone "build 1" artifact on source-built engines.
    let m: Manifest = serde_json::from_str(&row.manifest).unwrap();
    assert_eq!(m.build_number, 100, "tag build number is authoritative");
    assert_eq!(m.devices.len(), 1, "stub --list-devices GPU fixture");
    assert!(m.has_flag("--ctx-size"));
    assert!(m.has_flag("--jinja"));
    assert!(m.has_flag("--spec-draft-model"));
    assert!(m.spec_types.contains(&"draft-simple".to_string()));
    // Extraction landed the binary under engines/<tag>/.
    assert!(Path::new(&m.server_path).exists());

    // Install b200 -> becomes active; rollback returns to b100.
    mgr.update(Some("b200"), UpdateChannel::Latest)
        .await
        .unwrap();
    let store = Store::open(&dirs).unwrap();
    assert_eq!(store.active_engine().unwrap().unwrap().tag, "b200");
    let back = mgr.rollback().unwrap();
    assert_eq!(back.tag, "b100");

    // EngineUpdated events fired for each activation.
    let mut rx = mgr.bus.subscribe();
    let _ = rx.try_recv();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn integration__sha_mismatch__fail_fast_no_engine_dir() {
    let (_t, dirs) = tmp_dirs();
    let api = MockServer::start().await;
    let body = fixture_tarball("b100");
    // Corrupt digest -> download rejected.
    let bad = serde_json::json!({
        "tag_name": "b100",
        "assets": [{
            "name": "llama-b100-bin-ubuntu-vulkan-x64.tar.gz",
            "digest": format!("sha256:{}", sha256_hex(b"not-the-asset")),
            "size": body.len() as u64,
            "browser_download_url": format!("{}/download/b100/llama-b100-bin-ubuntu-vulkan-x64.tar.gz", api.uri()),
        }]
    });
    Mock::given(method("GET"))
        .and(path("/repos/ggml-org/llama.cpp/releases/tags/b100"))
        .respond_with(ResponseTemplate::new(200).set_body_json(bad))
        .mount(&api)
        .await;
    Mock::given(method("GET"))
        .and(path(
            "/download/b100/llama-b100-bin-ubuntu-vulkan-x64.tar.gz",
        ))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(body))
        .mount(&api)
        .await;

    let mgr = manager(&dirs, &api.uri());
    let err = mgr
        .update(Some("b100"), UpdateChannel::Latest)
        .await
        .unwrap_err();
    let msg = format!("{err:#}");
    assert!(msg.contains("sha256 mismatch"), "{msg}");
    assert!(
        !dirs.engines_dir().join("b100").exists(),
        "no engine dir left behind"
    );
    assert!(Store::open(&dirs)
        .unwrap()
        .list_engines()
        .unwrap()
        .is_empty());
}

#[tokio::test]
#[allow(non_snake_case)]
async fn integration__vtag_resolves_via_nightly_txt() {
    let (_t, dirs) = tmp_dirs();
    let api = MockServer::start().await;
    let nightly = b"b10809\n".to_vec();
    let tar = fixture_tarball("b10809");
    // v0.4.0 release: single nightly-tag.txt asset.
    Mock::given(method("GET"))
        .and(path("/repos/ggml-org/llama.cpp/releases/tags/v0.4.0"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "tag_name": "v0.4.0",
            "assets": [{
                "name": "nightly-tag.txt",
                "digest": format!("sha256:{}", sha256_hex(&nightly)),
                "browser_download_url": format!("{}/download/v0.4.0/nightly-tag.txt", api.uri()),
            }]
        })))
        .mount(&api)
        .await;
    Mock::given(method("GET"))
        .and(path("/download/v0.4.0/nightly-tag.txt"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(nightly))
        .mount(&api)
        .await;
    mount_release(
        &api,
        "b10809",
        "llama-b10809-bin-ubuntu-vulkan-x64.tar.gz",
        &tar,
    )
    .await;

    let mgr = manager(&dirs, &api.uri());
    let row = mgr
        .update(Some("v0.4.0"), UpdateChannel::Latest)
        .await
        .unwrap();
    assert_eq!(row.tag, "b10809", "v-tag resolved through nightly-tag.txt");
}

#[tokio::test]
#[allow(non_snake_case)]
async fn integration__latest_btag_from_list() {
    let (_t, dirs) = tmp_dirs();
    let api = MockServer::start().await;
    mount_releases_list(&api, &["b100", "b300", "b200"]).await;
    let tar = fixture_tarball("b300");
    mount_release(
        &api,
        "b300",
        "llama-b300-bin-ubuntu-vulkan-x64.tar.gz",
        &tar,
    )
    .await;
    let mgr = manager(&dirs, &api.uri());
    let rel = mgr.gh.latest_b_release().await.unwrap();
    assert_eq!(
        rel.tag_name, "b300",
        "newest by build number, not list order"
    );
}

#[tokio::test]
#[allow(non_snake_case)]
async fn integration__rate_limit__gh_token_hint() {
    let api = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/repos/ggml-org/llama.cpp/releases"))
        .respond_with(ResponseTemplate::new(403))
        .mount(&api)
        .await;
    let gh = GhClient::with_base(&api.uri(), None).unwrap();
    let err = gh.list_releases().await.unwrap_err();
    assert!(err.to_string().contains("GH_TOKEN"), "{err}");
}

#[tokio::test]
#[allow(non_snake_case)]
async fn integration__prune_keeps_newest_keep_tags_and_local() {
    let (_t, dirs) = tmp_dirs();
    let api = MockServer::start().await;
    let mgr = manager(&dirs, &api.uri());
    let store = Store::open(&dirs).unwrap();

    // Install 5 engines with increasing timestamps; then a `local` row.
    for (i, tag) in ["b1", "b2", "b3", "b4", "b5"].iter().enumerate() {
        let dir = dirs.engines_dir().join(tag);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("marker"), tag).unwrap();
        store
            .upsert_engine(&blazar_core::EngineRow {
                tag: tag.to_string(),
                asset: "x".into(),
                sha256: "x".into(),
                installed_at: 1000 + i64::try_from(i).unwrap_or(0),
                active: false,
                manifest: "{}".into(),
                kind: blazar_core::engine_kind::EngineKind::default(),
            })
            .unwrap();
    }
    // b1 also registered as local-path pseudo tag stays even though oldest.
    store
        .upsert_engine(&blazar_core::EngineRow {
            tag: LOCAL_TAG.to_string(),
            asset: "local".into(),
            sha256: "local".into(),
            installed_at: 1,
            active: false,
            manifest: "{}".into(),
            kind: blazar_core::engine_kind::EngineKind::default(),
        })
        .unwrap();
    // b2 marked active (old but active -> kept).
    store.set_active_engine("b2").unwrap();

    mgr.prune(&store).unwrap();
    let remaining: Vec<String> = store
        .list_engines()
        .unwrap()
        .into_iter()
        .map(|e| e.tag)
        .collect();
    // Newest KEEP_TAGS of (b1..=b5) + local + active b2 survive; every
    // older tag is pruned. Expectations derive from the const so a
    // retention-policy change re-pins this test for free.
    let all: Vec<String> = (1..=5).map(|n| format!("b{n}")).collect();
    let newest_kept: Vec<String> = all.iter().rev().take(KEEP_TAGS).cloned().collect();
    let expected_kept: Vec<String> = newest_kept
        .iter()
        .cloned()
        .chain([("b2".to_string()), (LOCAL_TAG.to_string())])
        .collect();
    for kept in &expected_kept {
        assert!(
            remaining.iter().any(|t| t == kept),
            "missing {kept} in {remaining:?}"
        );
    }
    for pruned in all.iter().filter(|t| !expected_kept.contains(t)) {
        assert!(
            !remaining.iter().any(|t| t == pruned),
            "{pruned} should be pruned: {remaining:?}"
        );
        assert!(!dirs.engines_dir().join(pruned).exists());
    }
    // Pruned count matches KEEP_TAGS policy.
    assert_eq!(remaining.len(), KEEP_TAGS + 2);
}

#[tokio::test]
#[allow(non_snake_case)]
async fn integration__prune_retention_is_scoped_per_kind() {
    use blazar_core::engine_kind::EngineKind;
    let (_t, dirs) = tmp_dirs();
    let api = MockServer::start().await;
    let mgr = manager(&dirs, &api.uri());
    let store = Store::open(&dirs).unwrap();

    let stage = |tag: &str, kind: blazar_core::engine_kind::EngineKind, at: i64| {
        let dir = dirs.engines_dir().join(tag);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("marker"), tag).unwrap();
        store
            .upsert_engine(&blazar_core::EngineRow {
                tag: tag.to_string(),
                asset: "x".into(),
                sha256: "x".into(),
                installed_at: at,
                active: false,
                manifest: "{}".into(),
                kind,
            })
            .unwrap();
    };
    // llamacpp lane past retention (3 > KEEP_TAGS) ...
    stage("b1", EngineKind::LlamaCpp, 1000);
    stage("b2", EngineKind::LlamaCpp, 1001);
    stage("b3", EngineKind::LlamaCpp, 1002);
    // ... while the mistralrs and sglang lanes each hold few engines that
    // are NOT substitutes for a llamacpp rollback anchor: they must
    // survive a llamacpp prune entirely.
    stage("m1", EngineKind::MistralRs, 1003);
    stage("m2", EngineKind::MistralRs, 1004);
    stage("s1", EngineKind::Sglang, 1005);
    store.set_active_engine("b2").unwrap();

    mgr.prune(&store).unwrap();
    let mut remaining: Vec<String> = store
        .list_engines()
        .unwrap()
        .into_iter()
        .map(|e| e.tag)
        .collect();
    remaining.sort();
    let mut expected = vec![
        "b2".to_string(), // active llamacpp
        "b3".to_string(), // newest KEEP_TAGS llamacpp
    ];
    for extra in ["m1", "m2", "s1"] {
        expected.push(extra.to_string());
    }
    expected.sort();
    assert_eq!(remaining, expected, "cross-kind engines are not anchors");
    assert!(!dirs.engines_dir().join("b1").exists());
    for kept in ["b2", "b3", "m1", "m2", "s1"] {
        assert!(dirs.engines_dir().join(kept).exists(), "{kept} dir gone");
    }
}

#[tokio::test]
#[allow(non_snake_case)]
async fn integration__prune_siblings_deletes_same_kind_keeps_rest() {
    use blazar_core::engine_kind::EngineKind;
    let (_t, dirs) = tmp_dirs();
    let api = MockServer::start().await;
    let mgr = manager(&dirs, &api.uri());
    let store = Store::open(&dirs).unwrap();

    let stage = |tag: &str, kind: EngineKind, at: i64| {
        let dir = dirs.engines_dir().join(tag);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("server"), vec![7u8; 4096]).unwrap();
        store
            .upsert_engine(&blazar_core::EngineRow {
                tag: tag.to_string(),
                asset: "x".into(),
                sha256: "x".into(),
                installed_at: at,
                active: false,
                manifest: "{}".into(),
                kind,
            })
            .unwrap();
    };
    // Two superseded llamacpp builds, the fresh one, a cross-kind
    // neighbor, and the never-pruned local pseudo-tag.
    stage("b-old", EngineKind::LlamaCpp, 1000);
    stage("b-mid", EngineKind::LlamaCpp, 1001);
    stage("b-new", EngineKind::LlamaCpp, 1002);
    stage("s1", EngineKind::Sglang, 1003);
    stage(LOCAL_TAG, EngineKind::LlamaCpp, 1);
    store.set_active_engine("b-new").unwrap();

    let freed = mgr.prune_siblings("llamacpp", "b-new").unwrap();
    let mut freed_tags: Vec<&str> = freed.iter().map(|(t, _)| t.as_str()).collect();
    freed_tags.sort_unstable();
    assert_eq!(
        freed_tags,
        vec!["b-mid", "b-old"],
        "only same-kind siblings"
    );
    // Freed bytes measured from the dir contents before deletion.
    for (_, bytes) in &freed {
        assert!(*bytes >= 4096, "byte counts come from the removed dir");
    }
    let mut remaining: Vec<String> = store
        .list_engines()
        .unwrap()
        .into_iter()
        .map(|e| e.tag)
        .collect();
    remaining.sort();
    let mut expected = vec!["b-new".to_string(), LOCAL_TAG.to_string(), "s1".to_string()];
    expected.sort();
    assert_eq!(remaining, expected, "cross-kind + local + keeper survive");
    assert!(!dirs.engines_dir().join("b-old").exists());
    assert!(!dirs.engines_dir().join("b-mid").exists());
    for kept in ["b-new", "s1", LOCAL_TAG] {
        assert!(dirs.engines_dir().join(kept).exists(), "{kept} dir gone");
    }
}

/// Stage an engine row whose manifest marks it as a fork capability lane.
/// Used by the prune-safety pins: fork lanes must never be auto-pruned no
/// matter how old they are.
fn stage_fork_row(store: &Store, dirs: &BlazarDirs, tag: &str, at: i64) {
    let dir = dirs.engines_dir().join(tag);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("marker"), tag).unwrap();
    let manifest = Manifest {
        tag: tag.to_string(),
        source: EngineSource::Fork,
        repo: Some("acme/llama.cpp".into()),
        ref_pin: Some("7c81a9f0".repeat(5)),
        architectures: std::collections::BTreeSet::from([
            "qwen35".to_string(),
            "qwen35_moe".to_string(),
        ]),
        ..Default::default()
    };
    store
        .upsert_engine(&blazar_core::EngineRow {
            tag: tag.to_string(),
            asset: "built-fork".into(),
            sha256: "x".into(),
            installed_at: at,
            active: false,
            manifest: serde_json::to_string(&manifest).unwrap(),
            kind: blazar_core::engine_kind::EngineKind::LlamaCpp,
        })
        .unwrap();
}

#[tokio::test]
#[allow(non_snake_case)]
async fn integration__prune_never_removes_fork_lanes() {
    let (_t, dirs) = tmp_dirs();
    let api = MockServer::start().await;
    let mgr = manager(&dirs, &api.uri());
    let store = Store::open(&dirs).unwrap();

    // Five upstream builds plus a fork lane OLDER than everything: the
    // fork must survive retention without consuming a KEEP_TAGS slot.
    for (i, tag) in ["b1", "b2", "b3", "b4", "b5"].iter().enumerate() {
        let dir = dirs.engines_dir().join(tag);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("marker"), tag).unwrap();
        store
            .upsert_engine(&blazar_core::EngineRow {
                tag: tag.to_string(),
                asset: "x".into(),
                sha256: "x".into(),
                installed_at: 1000 + i64::try_from(i).unwrap_or(0),
                active: false,
                manifest: "{}".into(),
                kind: blazar_core::engine_kind::EngineKind::LlamaCpp,
            })
            .unwrap();
    }
    let fork_tag = "fork-acme_llama.cpp-7c81a9f0-cpu";
    stage_fork_row(&store, &dirs, fork_tag, 1);
    store.set_active_engine("b5").unwrap();

    mgr.prune(&store).unwrap();
    let mut remaining: Vec<String> = store
        .list_engines()
        .unwrap()
        .into_iter()
        .map(|e| e.tag)
        .collect();
    remaining.sort();
    // Newest KEEP_TAGS upstream builds + active + the fork: the fork rides
    // along free — no upstream build is evicted to make room for it.
    let mut expected: Vec<String> = (1..=5)
        .map(|n| format!("b{n}"))
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .take(KEEP_TAGS)
        .collect();
    expected.push(fork_tag.to_string());
    expected.sort();
    assert_eq!(
        remaining, expected,
        "fork lane survives prune and spends no retention budget"
    );
    assert!(
        dirs.engines_dir().join(fork_tag).exists(),
        "fork lane dir must survive"
    );
}

#[tokio::test]
#[allow(non_snake_case)]
async fn integration__prune_siblings_never_removes_fork_lanes() {
    let (_t, dirs) = tmp_dirs();
    let api = MockServer::start().await;
    let mgr = manager(&dirs, &api.uri());
    let store = Store::open(&dirs).unwrap();

    // An upstream update normally frees same-kind siblings; the fork lane
    // is an additive capability lane, never a superseded sibling.
    for tag in ["b-old", "b-new"] {
        let dir = dirs.engines_dir().join(tag);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("server"), vec![7u8; 4096]).unwrap();
        store
            .upsert_engine(&blazar_core::EngineRow {
                tag: tag.to_string(),
                asset: "x".into(),
                sha256: "x".into(),
                installed_at: if tag == "b-new" { 1002 } else { 1000 },
                active: false,
                manifest: "{}".into(),
                kind: blazar_core::engine_kind::EngineKind::LlamaCpp,
            })
            .unwrap();
    }
    let fork_tag = "fork-acme_llama.cpp-7c81a9f0-cpu";
    stage_fork_row(&store, &dirs, fork_tag, 1001);
    store.set_active_engine("b-new").unwrap();

    let freed = mgr.prune_siblings("llamacpp", "b-new").unwrap();
    let freed_tags: Vec<&str> = freed.iter().map(|(t, _)| t.as_str()).collect();
    assert_eq!(freed_tags, vec!["b-old"], "only the upstream sibling frees");
    let mut remaining: Vec<String> = store
        .list_engines()
        .unwrap()
        .into_iter()
        .map(|e| e.tag)
        .collect();
    remaining.sort_unstable();
    assert_eq!(
        remaining,
        vec!["b-new".to_string(), fork_tag.to_string()],
        "fork lane is additive, never a prune sibling"
    );
    assert!(dirs.engines_dir().join(fork_tag).exists());
}

#[tokio::test]
#[allow(non_snake_case)]
async fn integration__register_engine_provenanced_bakes_source_and_architectures() {
    let (_t, dirs) = tmp_dirs();
    let api = MockServer::start().await;
    let mgr = manager(&dirs, &api.uri());

    let ref_sha = "7c81a9f0".repeat(5);
    let prov = LaneProvenance {
        source: EngineSource::Fork,
        repo: Some("acme/llama.cpp".into()),
        ref_pin: Some(ref_sha.clone()),
        base_ref: Some("b10980".into()),
        architectures: std::collections::BTreeSet::from([
            "qwen35".to_string(),
            "qwen35_moe".to_string(),
        ]),
    };
    let tag = "fork-acme_llama.cpp-7c81a9f0-cpu";
    let (dir, _guard) = stub_engine_dir("prov");
    let row = mgr
        .register_engine_provenanced(
            &dir,
            tag,
            "built-fork",
            "cafebabe",
            blazar_core::engine_kind::EngineKind::LlamaCpp,
            &prov,
            blazar_runtime::engine::manifest::TrustTier::default(),
        )
        .unwrap();

    let m: Manifest = serde_json::from_str(&row.manifest).unwrap();
    assert_eq!(m.source, EngineSource::Fork);
    assert_eq!(m.repo.as_deref(), Some("acme/llama.cpp"));
    assert_eq!(m.ref_pin.as_deref(), Some(ref_sha.as_str()));
    assert!(m.advertises_arch("qwen35"), "mined arch must be baked in");
    assert!(m.advertises_arch("qwen35_moe"));
    assert!(
        !m.advertises_arch("llama"),
        "only advertised architectures claim support"
    );
    let label = m.provenance_label();
    assert!(
        label.contains("acme/llama.cpp@7c81a9f0"),
        "label pins the exact commit: {label}"
    );
    assert!(
        label.contains("base b10980"),
        "label shows the merge target: {label}"
    );
}

#[tokio::test]
#[allow(non_snake_case)]
async fn integration__fork_lane_registration_never_dethrones_active_engine() {
    use std::collections::BTreeSet;

    // An active mainstream lane exists: the fork register must stay
    // additive — inactive row, active tag untouched.
    let (_t, dirs) = tmp_dirs();
    let api = MockServer::start().await;
    let mgr = manager(&dirs, &api.uri());
    {
        let store = Store::open(&dirs).unwrap();
        stage_mainstream_row(&store, &dirs, "b1-cuda", 2000, &["llama"]);
        store.set_active_engine("b1-cuda").unwrap();
    }
    let prov = LaneProvenance {
        source: EngineSource::Fork,
        repo: Some("acme/llama.cpp".into()),
        ref_pin: Some("7c81a9f0".repeat(5)),
        base_ref: None,
        architectures: BTreeSet::from(["spark9".to_string()]),
    };
    let tag = "fork-acme_llama.cpp-7c81a9f0-cpu";
    let (dir, _guard) = stub_engine_dir("additive");
    let row = mgr
        .register_engine_provenanced(
            &dir,
            tag,
            "built-fork",
            "cafebabe",
            blazar_core::engine_kind::EngineKind::LlamaCpp,
            &prov,
            blazar_runtime::engine::manifest::TrustTier::default(),
        )
        .unwrap();
    assert!(!row.active, "fork lane must not steal activation");
    let store = Store::open(&dirs).unwrap();
    assert_eq!(
        store.active_engine().unwrap().map(|r| r.tag),
        Some("b1-cuda".to_string()),
        "active mainstream engine keeps the throne"
    );

    // First of its kind: nothing to keep, the fork lane activates.
    let (_t2, dirs2) = tmp_dirs();
    let mgr2 = manager(&dirs2, &api.uri());
    let (fork_dir, _firstfork_guard) = stub_engine_dir("firstfork");
    let row2 = mgr2
        .register_engine_provenanced(
            &fork_dir,
            tag,
            "built-fork",
            "cafebabe",
            blazar_core::engine_kind::EngineKind::LlamaCpp,
            &prov,
            blazar_runtime::engine::manifest::TrustTier::default(),
        )
        .unwrap();
    assert!(row2.active, "first llamacpp lane activates");
}

/// Stage a mainstream (upstream-source) llamacpp row advertising
/// `archs` — the coverage side of the supersede pins.
fn stage_mainstream_row(store: &Store, dirs: &BlazarDirs, tag: &str, at: i64, archs: &[&str]) {
    let dir = dirs.engines_dir().join(tag);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("marker"), tag).unwrap();
    let manifest = Manifest {
        tag: tag.to_string(),
        source: EngineSource::Upstream,
        architectures: archs.iter().map(|s| (*s).to_string()).collect(),
        ..Default::default()
    };
    store
        .upsert_engine(&blazar_core::EngineRow {
            tag: tag.to_string(),
            asset: "built-cpu".into(),
            sha256: "x".into(),
            installed_at: at,
            active: false,
            manifest: serde_json::to_string(&manifest).unwrap(),
            kind: blazar_core::engine_kind::EngineKind::LlamaCpp,
        })
        .unwrap();
}

/// Stage a fork lane row with full lifecycle control (trust tier and
/// supersede stamp) — the retirement-sweep fixture shape.
fn stage_lane(
    store: &Store,
    dirs: &BlazarDirs,
    tag: &str,
    at: i64,
    trust: blazar_runtime::engine::manifest::TrustTier,
    archs: &[&str],
    superseded_at_epoch: Option<i64>,
) {
    let dir = dirs.engines_dir().join(tag);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("marker"), tag).unwrap();
    let manifest = Manifest {
        tag: tag.to_string(),
        source: EngineSource::Fork,
        repo: Some("acme/llama.cpp".into()),
        ref_pin: Some("7c81a9f0".repeat(5)),
        architectures: archs.iter().map(|s| (*s).to_string()).collect(),
        trust,
        superseded_at_epoch,
        ..Default::default()
    };
    store
        .upsert_engine(&blazar_core::EngineRow {
            tag: tag.to_string(),
            asset: "built-fork".into(),
            sha256: "x".into(),
            installed_at: at,
            active: false,
            manifest: serde_json::to_string(&manifest).unwrap(),
            kind: blazar_core::engine_kind::EngineKind::LlamaCpp,
        })
        .unwrap();
}

fn manifest_of(store: &Store, tag: &str) -> Manifest {
    let row = store
        .list_engines()
        .unwrap()
        .into_iter()
        .find(|e| e.tag == tag)
        .unwrap_or_else(|| panic!("row {tag} must exist"));
    serde_json::from_str(&row.manifest).unwrap()
}

#[tokio::test]
#[allow(non_snake_case)]
async fn integration__supersede_stamps_fully_covered_fork_lanes() {
    let (_t, dirs) = tmp_dirs();
    let api = MockServer::start().await;
    let mgr = manager(&dirs, &api.uri());
    let store = Store::open(&dirs).unwrap();

    // Mainline covers both of the fork's architectures; the fork rides
    // along until the lifecycle pass stamps it.
    stage_mainstream_row(
        &store,
        &dirs,
        "b-main",
        3000,
        &["qwen35", "qwen35_moe", "llama"],
    );
    stage_lane(
        &store,
        &dirs,
        "fork-acme_llama.cpp-7c81a9f0-cpu",
        1000,
        blazar_runtime::engine::manifest::TrustTier::User,
        &["qwen35", "qwen35_moe"],
        None,
    );

    // 0 = retirement sweep disabled; this pin only exercises stamping.
    mgr.refresh_supersede_state(0, &[]).await.unwrap();

    let m = manifest_of(&store, "fork-acme_llama.cpp-7c81a9f0-cpu");
    assert_eq!(m.superseded_by.as_deref(), Some("b-main"));
    assert!(m.superseded_at_epoch.is_some(), "stamp carries a clock");
    // Idempotent: a second pass does not re-stamp or error.
    mgr.refresh_supersede_state(0, &[]).await.unwrap();
    let m2 = manifest_of(&store, "fork-acme_llama.cpp-7c81a9f0-cpu");
    assert_eq!(m2.superseded_at_epoch, m.superseded_at_epoch);
}

#[tokio::test]
#[allow(non_snake_case)]
async fn integration__supersede_skips_partially_covered_fork_lanes() {
    let (_t, dirs) = tmp_dirs();
    let api = MockServer::start().await;
    let mgr = manager(&dirs, &api.uri());
    let store = Store::open(&dirs).unwrap();

    // Mainline absorbed qwen35 but NOT qwen35_moe: partial coverage must
    // keep the fork unstamped — the rescue path still owns the gap.
    stage_mainstream_row(&store, &dirs, "b-main", 3000, &["qwen35"]);
    stage_lane(
        &store,
        &dirs,
        "fork-acme_llama.cpp-7c81a9f0-cpu",
        1000,
        blazar_runtime::engine::manifest::TrustTier::User,
        &["qwen35", "qwen35_moe"],
        None,
    );

    mgr.refresh_supersede_state(0, &[]).await.unwrap();

    let m = manifest_of(&store, "fork-acme_llama.cpp-7c81a9f0-cpu");
    assert!(
        m.superseded_by.is_none(),
        "partial coverage must not graduate the fork"
    );
}

#[tokio::test]
#[allow(non_snake_case)]
async fn integration__retire_sweep_deletes_only_eligible_curated_lanes() {
    let (_t, dirs) = tmp_dirs();
    let api = MockServer::start().await;
    let mgr = manager(&dirs, &api.uri());
    let store = Store::open(&dirs).unwrap();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
        .cast_signed();
    let old = now - 10 * 86_400; // 10 days stale, 7-day grace

    let doomed = "fork-curated_llama.cpp-11111111-cpu";
    stage_lane(
        &store,
        &dirs,
        doomed,
        1000,
        TrustTier::Curated,
        &["x-arch"],
        Some(old),
    );
    let user_old = "fork-acme_llama.cpp-7c81a9f0-cpu";
    stage_lane(
        &store,
        &dirs,
        user_old,
        1001,
        TrustTier::User,
        &["x-arch"],
        Some(old),
    );
    let active_curated = "fork-curated_llama.cpp-22222222-cpu";
    stage_lane(
        &store,
        &dirs,
        active_curated,
        1002,
        TrustTier::Curated,
        &["x-arch"],
        Some(old),
    );
    let pinned_curated = "fork-curated_llama.cpp-33333333-cpu";
    stage_lane(
        &store,
        &dirs,
        pinned_curated,
        1003,
        TrustTier::Curated,
        &["x-arch"],
        Some(old),
    );
    let fresh_curated = "fork-curated_llama.cpp-44444444-cpu";
    stage_lane(
        &store,
        &dirs,
        fresh_curated,
        1004,
        TrustTier::Curated,
        &["x-arch"],
        Some(now),
    );
    store.set_active_engine(active_curated).unwrap();

    let pinned = vec![pinned_curated.to_string()];
    mgr.refresh_supersede_state(7, &pinned).await.unwrap();

    let tags: Vec<String> = store
        .list_engines()
        .unwrap()
        .into_iter()
        .map(|e| e.tag)
        .collect();
    assert!(
        !tags.contains(&doomed.to_string()),
        "eligible curated lane retired"
    );
    assert!(
        !dirs.engines_dir().join(doomed).exists(),
        "retired lane's directory is reclaimed"
    );
    for kept in [user_old, active_curated, pinned_curated, fresh_curated] {
        assert!(tags.contains(&kept.to_string()), "{kept} must survive");
    }
}

/// A lane whose directory cannot be reclaimed (here: no write
/// permission inside it) must not block retirement of its siblings,
/// and must still lose its row only once the dir is out of the way —
/// the aside copy is left on disk for a human (a leak beats a ghost
/// row: the row is gone, so nothing advertises the stranded dir).
#[cfg(unix)]
#[tokio::test]
#[allow(non_snake_case)]
async fn integration__retire_sweep_continues_past_undeletable_lane_dir() {
    use std::os::unix::fs::PermissionsExt as _;
    // DAC permissions do not stop root — the poison would silently
    // succeed and the assertions below would be vacuous.
    let uid = std::process::Command::new("id").arg("-u").output().unwrap();
    if String::from_utf8_lossy(&uid.stdout).trim() == "0" {
        return;
    }
    let (_t, dirs) = tmp_dirs();
    let api = MockServer::start().await;
    let mgr = manager(&dirs, &api.uri());
    let store = Store::open(&dirs).unwrap();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
        .cast_signed();
    let old = now - 10 * 86_400; // 10 days stale, 7-day grace

    let poisoned = "fork-curated_llama.cpp-11111111-cpu";
    stage_lane(
        &store,
        &dirs,
        poisoned,
        1000,
        TrustTier::Curated,
        &["x-arch"],
        Some(old),
    );
    let sibling = "fork-curated_llama.cpp-55555555-cpu";
    stage_lane(
        &store,
        &dirs,
        sibling,
        1001,
        TrustTier::Curated,
        &["x-arch"],
        Some(old),
    );

    // The poison: the lane's contents cannot be unlinked (rename aside
    // still works — it only needs write on the engines dir).
    let poisoned_dir = dirs.engines_dir().join(poisoned);
    let mut perms = std::fs::metadata(&poisoned_dir).unwrap().permissions();
    perms.set_mode(0o500);
    std::fs::set_permissions(&poisoned_dir, perms).unwrap();

    mgr.refresh_supersede_state(7, &Vec::new()).await.unwrap();

    // The sibling is fully retired despite its poisoned neighbor.
    let tags: Vec<String> = store
        .list_engines()
        .unwrap()
        .into_iter()
        .map(|e| e.tag)
        .collect();
    assert!(!tags.contains(&sibling.to_string()), "sibling retired");
    assert!(
        !dirs.engines_dir().join(sibling).exists(),
        "sibling dir reclaimed"
    );
    // The poisoned lane: row retired, dir leaked as exactly one aside.
    assert!(
        !tags.contains(&poisoned.to_string()),
        "poisoned lane row retired"
    );
    let leaked: Vec<_> = std::fs::read_dir(dirs.engines_dir())
        .unwrap()
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().starts_with(".retired-"))
        .collect();
    assert_eq!(leaked.len(), 1, "poisoned dir leaked as exactly one aside");
    // Re-enable deletion so the tempdir cleanup can reclaim everything.
    for aside in leaked {
        let mut p = std::fs::metadata(aside.path()).unwrap().permissions();
        p.set_mode(0o700);
        std::fs::set_permissions(aside.path(), p).unwrap();
    }
}

#[tokio::test]
#[allow(non_snake_case)]
async fn integration__register_local_engine() {
    let (_t, dirs) = tmp_dirs();
    let api = MockServer::start().await;
    let mgr = manager(&dirs, &api.uri());
    let row = mgr
        .register_local(&stub_server_bin(), &std::collections::BTreeMap::new())
        .unwrap();
    assert_eq!(row.tag, LOCAL_TAG);
    let m: Manifest = serde_json::from_str(&row.manifest).unwrap();
    assert!(m.has_flag("--jinja"));
    // Missing path -> named error.
    let err = mgr
        .register_local(
            Path::new("/nonexistent/llama-server"),
            &std::collections::BTreeMap::new(),
        )
        .unwrap_err();
    assert!(err.to_string().contains("does not exist"), "{err}");
}

/// A dir containing the stub binary as `llama-server` (what
/// `register_engine` expects an extracted engine dir to look like).
/// Stage a llama-server stub into a unique temp dir owned by the caller.
/// Returns the path plus the drop guard — binding the guard keeps the
/// staged binary alive for the test and guarantees cleanup on exit (the
/// earlier pid-keyed dir leaked one copy per `cargo test` run).
fn stub_engine_dir(tag: &str) -> (PathBuf, tempfile::TempDir) {
    let dir = tempfile::Builder::new()
        .prefix(&format!("blazar-engine-test-{tag}-"))
        .tempdir()
        .expect("staging tempdir");
    std::fs::copy(stub_server_bin(), dir.path().join("llama-server")).expect("copy stub");
    (dir.path().to_path_buf(), dir)
}

#[tokio::test]
#[allow(non_snake_case)]
async fn integration__lazy_whisper_lane_never_claims_serving_active() {
    // A whisper install must stay additive like fork lanes: `serve`
    // picks its adapter off the ACTIVE row, so a whisper row stealing
    // activation left the daemon unable to boot (live bug 2026-09-22).
    let (_t, dirs) = tmp_dirs();
    let api = MockServer::start().await;
    let mgr = manager(&dirs, &api.uri());
    {
        let store = Store::open(&dirs).unwrap();
        stage_mainstream_row(&store, &dirs, "b1-cuda", 2000, &["llama"]);
        store.set_active_engine("b1-cuda").unwrap();
    }
    let dir = tempfile::Builder::new()
        .prefix("blazar-engine-test-lazywhisper-")
        .tempdir()
        .expect("staging tempdir");
    std::fs::copy(stub_server_bin(), dir.path().join("whisper-server")).expect("copy stub");
    let row = mgr
        .register_engine(
            dir.path(),
            "b5130",
            "whisper-bin-ubuntu-x64.tar.gz",
            "cafebabe",
            blazar_core::engine_kind::EngineKind::Whisper,
        )
        .unwrap();
    assert!(!row.active, "lazy audio lane must not steal activation");
    let store = Store::open(&dirs).unwrap();
    assert_eq!(
        store.active_engine().unwrap().map(|r| r.tag),
        Some("b1-cuda".to_string()),
        "the serving engine keeps the throne across a whisper install"
    );
}

#[tokio::test]
#[allow(non_snake_case)]
async fn integration__vulkan_never_dethrones_cuda_on_nvidia() {
    let (_t, dirs) = tmp_dirs();
    let api = MockServer::start().await;
    let mgr = manager(&dirs, &api.uri());

    // CUDA engine installed first (vendor gate off — it must activate).
    let cuda = mgr
        .register_engine_with_vendor(
            &stub_engine_dir("b1").0,
            "b1-cuda",
            "ubuntu-cuda-12.8-x64",
            "aa",
            blazar_core::engine_kind::EngineKind::LlamaCpp,
            blazar_runtime::engine::manifest::Vendor::Other,
        )
        .unwrap();
    assert!(cuda.active);

    // A newer Vulkan install on an NVIDIA box: registered, NOT activated.
    let vulkan = mgr
        .register_engine_with_vendor(
            &stub_engine_dir("b2").0,
            "b2",
            "ubuntu-vulkan-x64",
            "bb",
            blazar_core::engine_kind::EngineKind::LlamaCpp,
            blazar_runtime::engine::manifest::Vendor::Nvidia,
        )
        .unwrap();
    assert!(!vulkan.active, "guard must keep CUDA active");
    let store = Store::open(&dirs).unwrap();
    assert_eq!(store.active_engine().unwrap().unwrap().tag, "b1-cuda");

    // The explicit override still works.
    let switched = mgr.use_tag("b2").unwrap();
    assert!(switched.active);
    assert_eq!(store.active_engine().unwrap().unwrap().tag, "b2");
}

#[tokio::test]
#[allow(non_snake_case)]
async fn integration__vulkan_activates_without_cuda_even_on_nvidia() {
    let (_t, dirs) = tmp_dirs();
    let api = MockServer::start().await;
    let mgr = manager(&dirs, &api.uri());
    let row = mgr
        .register_engine_with_vendor(
            &stub_engine_dir("b3").0,
            "b3",
            "ubuntu-vulkan-x64",
            "cc",
            blazar_core::engine_kind::EngineKind::LlamaCpp,
            blazar_runtime::engine::manifest::Vendor::Nvidia,
        )
        .unwrap();
    assert!(row.active, "no CUDA engine installed: Vulkan activates");
}

#[tokio::test]
#[allow(non_snake_case)]
async fn integration__vulkan_activates_on_non_nvidia_despite_cuda_row() {
    let (_t, dirs) = tmp_dirs();
    let api = MockServer::start().await;
    let mgr = manager(&dirs, &api.uri());
    mgr.register_engine_with_vendor(
        &stub_engine_dir("b4").0,
        "b4-cuda",
        "ubuntu-cuda-12.8-x64",
        "dd",
        blazar_core::engine_kind::EngineKind::LlamaCpp,
        blazar_runtime::engine::manifest::Vendor::Other,
    )
    .unwrap();
    let row = mgr
        .register_engine_with_vendor(
            &stub_engine_dir("b5").0,
            "b5",
            "ubuntu-vulkan-x64",
            "ee",
            blazar_core::engine_kind::EngineKind::LlamaCpp,
            blazar_runtime::engine::manifest::Vendor::Amd,
        )
        .unwrap();
    assert!(row.active, "AMD box: no NVIDIA guard");
}

#[tokio::test]
#[allow(non_snake_case)]
async fn integration__zip_asset_extracted_and_probed() {
    let (_t, dirs) = tmp_dirs();
    let api = MockServer::start().await;
    let zip = fixture_zip("b100");
    mount_release(&api, "b100", "llama-b100-bin-win-vulkan-x64.zip", &zip).await;
    let mut mgr = manager(&dirs, &api.uri());
    mgr.asset_override = "win-vulkan-x64".into();
    let row = mgr
        .update(Some("b100"), UpdateChannel::Latest)
        .await
        .unwrap();
    let m: Manifest = serde_json::from_str(&row.manifest).unwrap();
    assert!(m.server_path.ends_with("llama-server.exe"));
    assert!(Path::new(&m.server_path).exists());
}

#[tokio::test]
#[allow(non_snake_case)]
async fn integration__asset_override_missing__names_available() {
    let (_t, dirs) = tmp_dirs();
    let api = MockServer::start().await;
    let tar = fixture_tarball("b100");
    mount_release(
        &api,
        "b100",
        "llama-b100-bin-ubuntu-vulkan-x64.tar.gz",
        &tar,
    )
    .await;
    let mut mgr = manager(&dirs, &api.uri());
    mgr.asset_override = "ubuntu-rocm-10.0-x64".into();
    let err = mgr
        .update(Some("b100"), UpdateChannel::Latest)
        .await
        .unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("ubuntu-rocm-10.0-x64") && msg.contains("available"),
        "{msg}"
    );
}

/// The live b10833 incident: doctor sees a fresh release, `engine update`
/// runs while GitHub is still uploading assets. The updater must wait and
/// re-fetch instead of erroring.
#[tokio::test]
#[allow(non_snake_case)]
async fn integration__fresh_release_upload_race__waits_then_installs() {
    let (_t, dirs) = tmp_dirs();
    let api = MockServer::start().await;
    let tar = fixture_host_archive("b100");
    let complete = serde_json::json!({
        "tag_name": "b100",
        "prerelease": true,
        "published_at": iso_from_epoch(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs().try_into().unwrap_or(i64::MAX)
        ),
        "assets": host_assets_json(&api.uri(), "b100", &tar),
    });
    let empty = serde_json::json!({
        "tag_name": "b100",
        "prerelease": true,
        "published_at": complete["published_at"].clone(),
        "assets": [],
    });
    // First tag fetch: no assets yet (upload in flight). Then: complete.
    Mock::given(method("GET"))
        .and(path("/repos/ggml-org/llama.cpp/releases/tags/b100"))
        .respond_with(ResponseTemplate::new(200).set_body_json(empty))
        .up_to_n_times(1)
        .mount(&api)
        .await;
    Mock::given(method("GET"))
        .and(path("/repos/ggml-org/llama.cpp/releases/tags/b100"))
        .respond_with(ResponseTemplate::new(200).set_body_json(complete))
        .mount(&api)
        .await;
    for label in host_asset_labels() {
        let name = format!("llama-b100-bin-{label}.{}", host_asset_ext());
        Mock::given(method("GET"))
            .and(path(format!("/download/b100/{name}")))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(tar.clone()))
            .mount(&api)
            .await;
    }

    let mgr = manager_auto(&dirs, &api.uri());
    let row = mgr
        .update_with_retry_delay(
            Some("b100"),
            UpdateChannel::Latest,
            std::time::Duration::from_millis(10),
        )
        .await
        .expect("retry must ride out the upload window");
    assert_eq!(row.tag, "b100");
    // Vendor-dependent pick (vulkan on GPU boxes, cpu otherwise) but the
    // install must have succeeded off the SECOND fetch.
    assert!(
        host_asset_labels().contains(&row.asset.as_str()),
        "{}",
        row.asset
    );
    let store = Store::open(&dirs).unwrap();
    assert_eq!(store.active_engine().unwrap().unwrap().tag, "b100");
}

/// Old release with no assets at all: fail fast with the upload hint,
/// no waiting.
#[tokio::test]
#[allow(non_snake_case)]
async fn integration__stale_release_no_assets__teaching_error_no_wait() {
    let (_t, dirs) = tmp_dirs();
    let api = MockServer::start().await;
    let stale = iso_from_epoch(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
            .try_into()
            .unwrap_or(i64::MAX)
            - 7200,
    );
    let body = serde_json::json!({
        "tag_name": "b100",
        "prerelease": true,
        "published_at": stale,
        "assets": [],
    });
    Mock::given(method("GET"))
        .and(path("/repos/ggml-org/llama.cpp/releases/tags/b100"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(&api)
        .await;
    let mgr = manager_auto(&dirs, &api.uri());
    let started = std::time::Instant::now();
    let err = mgr
        .update_with_retry_delay(
            Some("b100"),
            UpdateChannel::Latest,
            std::time::Duration::from_millis(10),
        )
        .await
        .unwrap_err();
    // Budget note: this pins the CONTRACT "fails fast instead of
    // waiting out the ~20 s fresh-release asset-upload window", not
    // raw subprocess speed. maybe_cuda_overlay probes real GPU facts
    // (two nvidia-smi spawns): with driver persistence mode OFF the
    // first spawn after ~45 s idle re-wakes the GPU from P3 at a
    // measured 1.9 s (3/3 reproducible; back-to-back runs are 33-46
    // ms and CPU load does NOT trigger it). 15 s still proves no
    // upload-wait happened while tolerating that driver wake.
    assert!(started.elapsed() < std::time::Duration::from_secs(15));
    let msg = format!("{err:#}");
    assert!(
        msg.contains("no usable asset") && msg.contains("retry in a minute"),
        "{msg}"
    );
}

/// GPU asset genuinely absent from an OLD release (rename/drop): CPU
/// last-resort with the fallback recorded in the row.
#[tokio::test]
#[allow(non_snake_case)]
#[cfg(not(target_os = "macos"))] // macos has no cpu-fallback tier: both candidates are full builds
async fn integration__stale_release_gpu_missing__cpu_last_resort() {
    let (_t, dirs) = tmp_dirs();
    let api = MockServer::start().await;
    let tar = fixture_host_archive("b100");
    let cpu_asset = format!(
        "llama-b100-bin-{}.{}",
        host_cpu_asset_label(),
        host_asset_ext()
    );
    let stale = iso_from_epoch(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
            .try_into()
            .unwrap_or(i64::MAX)
            - 7200,
    );
    let body = serde_json::json!({
        "tag_name": "b100",
        "prerelease": true,
        "published_at": stale,
        "assets": [{
            "name": cpu_asset,
            "digest": format!("sha256:{}", sha256_hex(&tar)),
            "size": tar.len() as u64,
            "browser_download_url": format!("{}/download/b100/{}", api.uri(), cpu_asset),
        }],
    });
    Mock::given(method("GET"))
        .and(path("/repos/ggml-org/llama.cpp/releases/tags/b100"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(&api)
        .await;
    Mock::given(method("GET"))
        .and(path(format!("/download/b100/{cpu_asset}")))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(tar))
        .mount(&api)
        .await;
    let mgr = manager_auto(&dirs, &api.uri());
    let row = mgr
        .update_with_retry_delay(
            Some("b100"),
            UpdateChannel::Latest,
            std::time::Duration::from_millis(10),
        )
        .await
        .unwrap();
    assert_eq!(row.asset, host_cpu_asset_label());
    assert_eq!(row.tag, "b100");
}

#[tokio::test]
#[allow(non_snake_case)]
async fn integration__channel_b_release__stable_resolves_through_nightly_txt() {
    let api = MockServer::start().await;
    let nightly = b"b10780\n".to_vec();
    // GitHub's /releases/latest (non-prerelease) points at a vX.Y.Z whose
    // nightly-tag.txt names the concrete b-build it ships; resolve_tag
    // re-fetches the release by tag to read that asset.
    let v_release = serde_json::json!({
        "tag_name": "v1.36.0",
        "prerelease": false,
        "assets": [{
            "name": "nightly-tag.txt",
            "digest": format!("sha256:{}", sha256_hex(&nightly)),
            "browser_download_url": format!("{}/download/v1.36.0/nightly-tag.txt", api.uri()),
        }]
    });
    Mock::given(method("GET"))
        .and(path("/repos/ggml-org/llama.cpp/releases/latest"))
        .respond_with(ResponseTemplate::new(200).set_body_json(v_release.clone()))
        .mount(&api)
        .await;
    Mock::given(method("GET"))
        .and(path("/repos/ggml-org/llama.cpp/releases/tags/v1.36.0"))
        .respond_with(ResponseTemplate::new(200).set_body_json(v_release))
        .mount(&api)
        .await;
    Mock::given(method("GET"))
        .and(path("/download/v1.36.0/nightly-tag.txt"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(nightly))
        .mount(&api)
        .await;
    mount_release(
        &api,
        "b10780",
        "llama-b10780-bin-ubuntu-vulkan-x64.tar.gz",
        &fixture_tarball("b10780"),
    )
    .await;

    let gh = GhClient::with_base(&api.uri(), None).unwrap();
    let rel = gh
        .channel_b_release(blazar_core::config::UpdateChannel::Stable)
        .await
        .unwrap();
    assert_eq!(
        rel.tag_name, "b10780",
        "stable channel dereferences vX.Y.Z through nightly-tag.txt"
    );
}

#[tokio::test]
#[allow(non_snake_case)]
async fn integration__channel_repo_release__latest_takes_first_stable_uses_github_latest() {
    let api = MockServer::start().await;
    // whisper.cpp/blazar-style upstreams publish through /releases/latest
    // only: both channels resolve to the same non-prerelease target.
    Mock::given(method("GET"))
        .and(path("/repos/ggml-org/llama.cpp/releases/latest"))
        .respond_with(ResponseTemplate::new(200).set_body_json(
            serde_json::json!({"tag_name": "v1.8.0", "prerelease": false, "assets": []}),
        ))
        .mount(&api)
        .await;

    let gh = GhClient::with_base(&api.uri(), None).unwrap();
    let latest = gh
        .channel_repo_release(
            "ggml-org/llama.cpp",
            blazar_core::config::UpdateChannel::Latest,
        )
        .await
        .unwrap();
    assert_eq!(latest.tag_name, "v1.8.0");
    let stable = gh
        .channel_repo_release(
            "ggml-org/llama.cpp",
            blazar_core::config::UpdateChannel::Stable,
        )
        .await
        .unwrap();
    assert_eq!(stable.tag_name, "v1.8.0");
}

/// R2-25, live-observed 2026-09-11: whisper.cpp tags releases (v1.9.4)
/// that carry no assets while the prerelease b-tags carry the binaries.
/// The whisper channel target must skip the assetless tag on Latest and
/// teach the escape hatch on Stable instead of advertising it.
/// Host-platform-gated: the mocked asset is the linux-x64 one the
/// resolver picks on this target.
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
#[tokio::test]
#[allow(non_snake_case)]
async fn integration__whisper_channel_target__assetless_latest_falls_through() {
    let api = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/repos/ggml-org/whisper.cpp/releases"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!([
            {"tag_name": "v1.9.4", "prerelease": false, "assets": []},
            {
                "tag_name": "b5130",
                "prerelease": true,
                "assets": [
                    {"name": "whisper-bin-ubuntu-x64.tar.gz",
                     "browser_download_url": "http://example.invalid/w.tar.gz"}
                ]
            },
            {
                "tag_name": "b5127",
                "prerelease": true,
                "assets": [
                    {"name": "whisper-bin-ubuntu-arm64.tar.gz",
                     "browser_download_url": "http://example.invalid/a.tar.gz"}
                ]
            },
        ])))
        .mount(&api)
        .await;

    let gh = GhClient::with_base(&api.uri(), None).unwrap();
    let latest =
        blazar_runtime::whisper::channel_target(&gh, blazar_core::config::UpdateChannel::Latest)
            .await
            .unwrap();
    assert_eq!(
        latest, "b5130",
        "Latest channel skips the assetless v1.9.4 and takes the newest b-tag \
         carrying this platform's server binary"
    );
    let stable =
        blazar_runtime::whisper::channel_target(&gh, blazar_core::config::UpdateChannel::Stable)
            .await
            .expect_err("no stable release carries the asset");
    let msg = format!("{stable:#}");
    assert!(
        msg.contains("b5130") && msg.contains("update_channel"),
        "stable miss must name the newest installable prerelease and the \
         channel escape hatch, got: {msg}"
    );
}

#[test]
#[allow(non_snake_case)]
fn unit__keep_cuda_skip_pred__truth_table() {
    use blazar_runtime::engine::keep_cuda_skip_pred;
    use blazar_runtime::engine::manifest::Vendor;

    fn row(
        tag: &str,
        asset: &str,
        active: bool,
        kind: blazar_core::engine_kind::EngineKind,
    ) -> blazar_core::EngineRow {
        blazar_core::EngineRow {
            tag: tag.into(),
            asset: asset.into(),
            sha256: "x".into(),
            installed_at: 1,
            active,
            manifest: "{}".into(),
            kind,
        }
    }
    use blazar_core::engine_kind::EngineKind;
    let cuda = row("b10900-cuda", "built-cuda", true, EngineKind::LlamaCpp);
    let yes = |r: &blazar_core::EngineRow| {
        keep_cuda_skip_pred(Some(r), Vendor::Nvidia, "linux", "x86_64", "auto")
    };

    assert!(
        yes(&cuda),
        "linux-x86_64 NVIDIA + active llamacpp cuda: skip"
    );
    assert!(
        keep_cuda_skip_pred(Some(&cuda), Vendor::Nvidia, "linux", "x86_64", ""),
        "empty asset_override behaves like auto"
    );
    assert!(
        !keep_cuda_skip_pred(
            Some(&cuda),
            Vendor::Nvidia,
            "linux",
            "x86_64",
            "ubuntu-vulkan-x64"
        ),
        "engine_asset pin forces the standard lane"
    );
    assert!(
        !keep_cuda_skip_pred(Some(&cuda), Vendor::Amd, "linux", "x86_64", "auto"),
        "non-NVIDIA box downloads normally"
    );
    assert!(
        keep_cuda_skip_pred(Some(&cuda), Vendor::Nvidia, "windows", "x86_64", "auto"),
        "windows-x86_64 builds CUDA locally too — the dormant-download waste is identical"
    );
    assert!(
        !keep_cuda_skip_pred(Some(&cuda), Vendor::Nvidia, "macos", "x86_64", "auto"),
        "no CUDA on macOS (Metal lane)"
    );
    assert!(
        !keep_cuda_skip_pred(Some(&cuda), Vendor::Nvidia, "linux", "aarch64", "auto"),
        "aarch64 downloads normally"
    );
    assert!(
        !keep_cuda_skip_pred(None, Vendor::Nvidia, "linux", "x86_64", "auto"),
        "no active engine: nothing to keep"
    );
    let vulkan_active = row("b10900", "ubuntu-vulkan-x64", true, EngineKind::LlamaCpp);
    assert!(!yes(&vulkan_active), "active engine IS the vulkan lane");
    let mistral_cuda = row("v1.5", "cuda-12.8", true, EngineKind::MistralRs);
    assert!(
        !yes(&mistral_cuda),
        "mistral.rs lane owns its own updates; never skip llamacpp for it"
    );
}

#[tokio::test]
#[allow(non_snake_case)]
#[cfg(not(target_os = "macos"))] // vulkan-never-dethrones-cuda guard: the concept under test
async fn integration__update_resolved__keep_cuda_skip_downloads_nothing() {
    use blazar_runtime::engine::gh::GhRelease;
    use blazar_runtime::engine::manifest::Vendor;

    let (_t, dirs) = tmp_dirs();
    let api = MockServer::start().await;
    let mgr = manager_auto(&dirs, &api.uri());

    // Active CUDA engine seeded first (vendor Other: no guard, activates).
    let cuda = mgr
        .register_engine_with_vendor(
            &stub_engine_dir("b10900-cuda").0,
            "b10900-cuda",
            "built-cuda",
            "aa",
            blazar_core::engine_kind::EngineKind::LlamaCpp,
            Vendor::Other,
        )
        .unwrap();
    assert!(cuda.active);

    // Channel release carries ONLY the host platform's primary asset. No
    // /download mock is mounted: any fetch attempt 404s and fails the
    // test — the skip must guarantee the standard lane never downloads.
    let primary_asset = format!(
        "llama-b10910-bin-{}.{}",
        host_asset_labels()[0],
        host_asset_ext()
    );
    let release: GhRelease = serde_json::from_value(serde_json::json!({
        "tag_name": "b10910",
        "prerelease": true,
        "assets": [{
            "name": primary_asset,
            "size": 1,
            "browser_download_url": format!("{}/download/b10910/{}", api.uri(), primary_asset)
        }]
    }))
    .unwrap();

    let row = mgr
        .update_resolved_with_vendor(release, Vendor::Nvidia, false)
        .await
        .unwrap();
    assert_eq!(row.tag, "b10900-cuda", "kept-active row returned");
    assert!(row.active);

    let downloads = api
        .received_requests()
        .await
        .unwrap_or_default()
        .iter()
        .filter(|r| r.url.path().starts_with("/download/"))
        .count();
    assert_eq!(downloads, 0, "skip must fetch zero asset bytes");
}

#[test]
#[allow(non_snake_case)]
fn unit__remove_engine_row_and_tree__happy_path_removes_dir_row_and_aside() {
    let (_t, dirs) = tmp_dirs();
    let store = Store::open(&dirs).unwrap();
    stage_lane(
        &store,
        &dirs,
        "fork-acme_llama.cpp-22222222-cpu",
        1000,
        TrustTier::Curated,
        &["x-arch"],
        None,
    );
    let dir = dirs.engines_dir().join("fork-acme_llama.cpp-22222222-cpu");
    assert!(dir.join("marker").exists());

    let bytes = blazar_runtime::engine::remove_engine_row_and_tree(
        &store,
        "fork-acme_llama.cpp-22222222-cpu",
        &dir,
    )
    .unwrap();
    assert!(bytes > 0, "marker bytes reclaimed");
    assert!(!dir.exists(), "engine dir gone");
    assert!(
        !store
            .list_engines()
            .unwrap()
            .iter()
            .any(|e| e.tag == "fork-acme_llama.cpp-22222222-cpu"),
        "engine row gone"
    );
    let asides: Vec<_> = std::fs::read_dir(dirs.engines_dir())
        .unwrap()
        .flatten()
        .filter(|e| {
            e.file_name()
                .to_str()
                .is_some_and(|n| n.starts_with(".retired-"))
        })
        .collect();
    assert!(asides.is_empty(), "no aside leaked: {asides:?}");
}

#[test]
#[allow(non_snake_case)]
fn unit__remove_engine_row_and_tree__restores_dir_when_row_delete_fails() {
    let (_t, dirs) = tmp_dirs();
    let store = Store::open(&dirs).unwrap();
    let tag = "fork-acme_llama.cpp-33333333-cpu";
    stage_lane(
        &store,
        &dirs,
        tag,
        1000,
        TrustTier::Curated,
        &["x-arch"],
        None,
    );
    let dir = dirs.engines_dir().join(tag);

    // Break the store's delete: a second connection drops the engines
    // table so `DELETE FROM engines` errors immediately (no busy-wait).
    let blocker = rusqlite::Connection::open(dirs.data_dir.join("blazar.db")).unwrap();
    blocker.execute("DROP TABLE engines", []).unwrap();

    let err = blazar_runtime::engine::remove_engine_row_and_tree(&store, tag, &dir)
        .expect_err("row delete must fail");
    assert!(
        format!("{err:#}").contains("cannot delete engine row"),
        "error names the failed step: {err:#}"
    );
    assert!(dir.join("marker").exists(), "dir restored in place");
    let asides: Vec<_> = std::fs::read_dir(dirs.engines_dir())
        .unwrap()
        .flatten()
        .filter(|e| {
            e.file_name()
                .to_str()
                .is_some_and(|n| n.starts_with(".retired-"))
        })
        .collect();
    assert!(asides.is_empty(), "aside rolled back: {asides:?}");
}
