// Alias-never-destroys-source is pinned via inode identity of hardlink
// twins — a unix filesystem concept.
#![cfg(unix)]
//! cp/rm alias semantics — an alias must NEVER be able to destroy its
//! source. Regression for the 2026-09-05 incident: `copy_model` created the
//! alias row pointing at the SOURCE file's path, so `rm <alias>` deleted
//! the original 5.3 GiB model.

#![allow(non_snake_case)] // unit__scenario__expected naming convention

use std::os::unix::fs::MetadataExt as _;

use blazar_core::store::Store;
use blazar_core::{BlazarDirs, ModelRow};
use blazar_runtime::models::{copy_model, remove_model};

fn row(name: &str, path: &str) -> ModelRow {
    ModelRow {
        name: name.into(),
        repo: "o/m1".into(),
        quant: "Q4_K_M".into(),
        path: path.into(),
        bytes: 42,
        sha256: None,
        mmproj_path: None,
        components: vec![],
        shards: 1,
        arch: Some("qwen3".into()),
        params: Some(0.5),
        ctx_train: Some(40_960),
        pulled_at: 1,
    }
}

fn setup(tag: &str) -> (tempfile::TempDir, BlazarDirs) {
    let tmp = tempfile::tempdir().unwrap();
    let dirs = BlazarDirs {
        config_dir: tmp.path().join("c"),
        data_dir: tmp.path().join("d"),
    };
    dirs.ensure().unwrap();
    let gguf = dirs.models_dir().join(format!("{tag}-Q4_K_M.gguf"));
    std::fs::write(&gguf, b"gguf-bytes-for-alias-safety").unwrap();
    let store = Store::open(&dirs).unwrap();
    store
        .upsert_model(&row(tag, &gguf.display().to_string()))
        .unwrap();
    (tmp, dirs)
}

#[test]
fn unit__copy_then_rm_alias__source_file_survives() {
    let (_tmp, dirs) = setup("m1");
    let store = Store::open(&dirs).unwrap();
    let src_path = store.get_model("m1").unwrap().unwrap().path;

    copy_model(&dirs, "m1", "alias1").unwrap();
    let alias_path = store.get_model("alias1").unwrap().unwrap().path;
    assert_ne!(
        alias_path, src_path,
        "alias row must never point at the source file's path"
    );
    // Same inode (a true zero-byte hardlink alias).
    assert_eq!(
        std::fs::metadata(&src_path).unwrap().ino(),
        std::fs::metadata(&alias_path).unwrap().ino(),
        "alias should be a hardlink (same inode), not a copy"
    );

    remove_model(&dirs, "alias1").unwrap();
    assert!(
        std::path::Path::new(&src_path).exists(),
        "rm on the alias MUST NOT delete the source file"
    );
    assert!(
        store.get_model("m1").unwrap().is_some(),
        "source row survives"
    );
    assert!(
        store.get_model("alias1").unwrap().is_none(),
        "alias row removed"
    );
}

#[test]
fn unit__rm_alias__shared_mmproj_survives() {
    // The twice-lived incident: alias rows share the source's mmproj;
    // rm on the alias must not delete a file the source still needs.
    let (_tmp, dirs) = setup("m4");
    let store = Store::open(&dirs).unwrap();
    let mm = dirs.models_dir().join("mmproj-F16.gguf");
    std::fs::write(&mm, b"projector-bytes").unwrap();
    let mut src = store.get_model("m4").unwrap().unwrap();
    src.mmproj_path = Some(mm.display().to_string());
    store.upsert_model(&src).unwrap();

    copy_model(&dirs, "m4", "alias4").unwrap();
    let alias_mm = store.get_model("alias4").unwrap().unwrap().mmproj_path;
    assert_eq!(
        alias_mm.as_deref(),
        Some(mm.display().to_string().as_str()),
        "alias shares the source projector"
    );

    remove_model(&dirs, "alias4").unwrap();
    assert!(mm.exists(), "shared mmproj must survive rm of the alias");
    assert!(store.get_model("m4").unwrap().is_some());

    // Deleting the SOURCE (last referencer) may reclaim the projector.
    remove_model(&dirs, "m4").unwrap();
    assert!(!mm.exists(), "last reference gone -> projector reclaimed");
}

#[test]
fn unit__copy_alias__distinct_path_named_after_destination() {
    let (_tmp, dirs) = setup("m2");
    copy_model(&dirs, "m2", "my-alias").unwrap();
    let alias_path = Store::open(&dirs)
        .unwrap()
        .get_model("my-alias")
        .unwrap()
        .unwrap()
        .path;
    assert!(
        alias_path.contains("my-alias__alias__"),
        "alias file name should derive from the destination: {alias_path}"
    );
}

#[test]
fn unit__copy_refuses_existing_destination() {
    let (_tmp, dirs) = setup("m3");
    let err = copy_model(&dirs, "m3", "m3").unwrap_err().to_string();
    assert!(err.contains("already exists"), "{err}");
}
