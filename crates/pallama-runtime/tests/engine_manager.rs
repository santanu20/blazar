//! Engine manager integration tests: full install -> manifest -> activate
//! -> rollback cycle against a wiremock GitHub API, with a REAL tar.gz
//! archive containing the compiled stub-llama-server (exercises the
//! genuine probe path: --version / --list-devices / --help).

use sha2::Digest as _;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use pallama_core::store::Store;
use pallama_core::PallamaDirs;
use pallama_runtime::engine::gh::GhClient;
use pallama_runtime::engine::manifest::Manifest;
use pallama_runtime::engine::{EngineManager, KEEP_TAGS, LOCAL_TAG};
use pallama_runtime::EventBus;
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

fn manager(dirs: &PallamaDirs, api_uri: &str) -> EngineManager {
    let gh = GhClient::with_base(api_uri, None).unwrap();
    EngineManager {
        dirs: dirs.clone(),
        gh,
        bus: EventBus::default(),
        asset_override: "ubuntu-vulkan-x64".into(),
    }
}

/// Auto asset selection (no override): exercises the resolve/retry path.
fn manager_auto(dirs: &PallamaDirs, api_uri: &str) -> EngineManager {
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

fn tmp_dirs() -> (tempfile::TempDir, PallamaDirs) {
    let tmp = tempfile::tempdir().unwrap();
    let dirs = PallamaDirs {
        config_dir: tmp.path().join("cfg"),
        data_dir: tmp.path().join("data"),
    };
    dirs.ensure().unwrap();
    (tmp, dirs)
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
    let row = mgr.update(Some("b100")).await.unwrap();
    assert_eq!(row.tag, "b100");
    assert_eq!(row.sha256, sha256_hex(&t1));
    let store = Store::open(&dirs).unwrap();
    assert_eq!(store.active_engine().unwrap().unwrap().tag, "b100");

    // Manifest is a real probe of the stub: version/build/devices/flags.
    let m: Manifest = serde_json::from_str(&row.manifest).unwrap();
    assert_eq!(m.build_number, 9999, "stub STUB_BUILD default");
    assert_eq!(m.devices.len(), 1, "stub --list-devices GPU fixture");
    assert!(m.has_flag("--ctx-size"));
    assert!(m.has_flag("--jinja"));
    assert!(m.has_flag("--spec-draft-model"));
    assert!(m.spec_types.contains(&"draft-simple".to_string()));
    // Extraction landed the binary under engines/<tag>/.
    assert!(Path::new(&m.server_path).exists());

    // Install b200 -> becomes active; rollback returns to b100.
    mgr.update(Some("b200")).await.unwrap();
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
    let err = mgr.update(Some("b100")).await.unwrap_err();
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
    let row = mgr.update(Some("v0.4.0")).await.unwrap();
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
async fn integration__prune_keeps_last_three_and_local() {
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
            .upsert_engine(&pallama_core::EngineRow {
                tag: tag.to_string(),
                asset: "x".into(),
                sha256: "x".into(),
                installed_at: 1000 + i64::try_from(i).unwrap_or(0),
                active: false,
                manifest: "{}".into(),
            })
            .unwrap();
    }
    // b1 also registered as local-path pseudo tag stays even though oldest.
    store
        .upsert_engine(&pallama_core::EngineRow {
            tag: LOCAL_TAG.to_string(),
            asset: "local".into(),
            sha256: "local".into(),
            installed_at: 1,
            active: false,
            manifest: "{}".into(),
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
    // Newest KEEP_TAGS (b3,b4,b5) + local + active b2 survive; b1 pruned.
    for kept in ["b2", "b3", "b4", "b5", LOCAL_TAG] {
        assert!(
            remaining.iter().any(|t| t == kept),
            "missing {kept} in {remaining:?}"
        );
    }
    assert!(
        !remaining.iter().any(|t| t == "b1"),
        "b1 should be pruned: {remaining:?}"
    );
    assert!(!dirs.engines_dir().join("b1").exists());
    // Pruned count matches KEEP_TAGS policy.
    assert_eq!(remaining.len(), KEEP_TAGS + 2);
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

#[tokio::test]
#[allow(non_snake_case)]
async fn integration__zip_asset_extracted_and_probed() {
    let (_t, dirs) = tmp_dirs();
    let api = MockServer::start().await;
    let zip = fixture_zip("b100");
    mount_release(&api, "b100", "llama-b100-bin-win-vulkan-x64.zip", &zip).await;
    let mut mgr = manager(&dirs, &api.uri());
    mgr.asset_override = "win-vulkan-x64".into();
    let row = mgr.update(Some("b100")).await.unwrap();
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
    let err = mgr.update(Some("b100")).await.unwrap_err();
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
    let tar = fixture_tarball("b100");
    let complete = serde_json::json!({
        "tag_name": "b100",
        "prerelease": true,
        "published_at": iso_from_epoch(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs().try_into().unwrap_or(i64::MAX)
        ),
        "assets": [
            {
                "name": "llama-b100-bin-ubuntu-vulkan-x64.tar.gz",
                "digest": format!("sha256:{}", sha256_hex(&tar)),
                "size": tar.len() as u64,
                "browser_download_url": format!("{}/download/b100/llama-b100-bin-ubuntu-vulkan-x64.tar.gz", api.uri()),
            },
            {
                "name": "llama-b100-bin-ubuntu-x64.tar.gz",
                "digest": format!("sha256:{}", sha256_hex(&tar)),
                "size": tar.len() as u64,
                "browser_download_url": format!("{}/download/b100/llama-b100-bin-ubuntu-x64.tar.gz", api.uri()),
            },
        ],
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
    Mock::given(method("GET"))
        .and(path(
            "/download/b100/llama-b100-bin-ubuntu-vulkan-x64.tar.gz",
        ))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(tar.clone()))
        .mount(&api)
        .await;
    Mock::given(method("GET"))
        .and(path("/download/b100/llama-b100-bin-ubuntu-x64.tar.gz"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(tar))
        .mount(&api)
        .await;

    let mgr = manager_auto(&dirs, &api.uri());
    let row = mgr
        .update_with_retry_delay(Some("b100"), std::time::Duration::from_millis(10))
        .await
        .expect("retry must ride out the upload window");
    assert_eq!(row.tag, "b100");
    // Vendor-dependent pick (vulkan on GPU boxes, cpu otherwise) but the
    // install must have succeeded off the SECOND fetch.
    assert!(
        row.asset == "ubuntu-vulkan-x64" || row.asset == "ubuntu-x64",
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
        .update_with_retry_delay(Some("b100"), std::time::Duration::from_millis(10))
        .await
        .unwrap_err();
    assert!(started.elapsed() < std::time::Duration::from_secs(2));
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
async fn integration__stale_release_gpu_missing__cpu_last_resort() {
    let (_t, dirs) = tmp_dirs();
    let api = MockServer::start().await;
    let tar = fixture_tarball("b100");
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
            "name": "llama-b100-bin-ubuntu-x64.tar.gz",
            "digest": format!("sha256:{}", sha256_hex(&tar)),
            "size": tar.len() as u64,
            "browser_download_url": format!("{}/download/b100/llama-b100-bin-ubuntu-x64.tar.gz", api.uri()),
        }],
    });
    Mock::given(method("GET"))
        .and(path("/repos/ggml-org/llama.cpp/releases/tags/b100"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(&api)
        .await;
    Mock::given(method("GET"))
        .and(path("/download/b100/llama-b100-bin-ubuntu-x64.tar.gz"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(tar))
        .mount(&api)
        .await;
    let mgr = manager_auto(&dirs, &api.uri());
    let row = mgr
        .update_with_retry_delay(Some("b100"), std::time::Duration::from_millis(10))
        .await
        .unwrap();
    assert_eq!(row.asset, "ubuntu-x64");
    assert_eq!(row.tag, "b100");
}
