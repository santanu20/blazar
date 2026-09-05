//! Model store operations: remove with running-instance guard, listing.
//! The `run/<name>.pid` marker is the same file the supervisor (step E)
//! writes; sharing the contract here means rm refuses while running from
//! day one (409 semantics) without coupling to the supervisor.

use std::path::PathBuf;

use anyhow::{anyhow, Result};

use pallama_core::store::Store;
use pallama_core::PallamaDirs;

/// True when an instance marker exists for `name` (process may be loading
/// or serving; the supervisor owns the marker's lifecycle).
#[must_use] 
pub fn instance_running(dirs: &PallamaDirs, name: &str) -> bool {
    dirs.run_dir().join(format!("{name}.pid")).exists()
}

/// Delete a model: all shard files (by gguf-split naming convention), the
/// mmproj, and the store row. Refuses (409-style error) while an instance
/// of the model is running.
#[allow(clippy::case_sensitive_file_extension_comparisons)] // operand pre-lowercased
pub fn remove_model(dirs: &PallamaDirs, name: &str) -> Result<()> {
    if instance_running(dirs, name) {
        return Err(anyhow!(
            "model {name} is currently running; stop it first (`pallama stop {name}` or wait for eviction)"
        ));
    }
    let store = Store::open(dirs)?;
    let row = store
        .get_model(name)?
        .ok_or_else(|| anyhow!("no such model: {name}"))?;

    let mut files: Vec<PathBuf> = vec![PathBuf::from(&row.path)];
    // Shards share the stored first-shard filename convention
    // (`base-NNNNN-of-MMMMM.gguf`); collect every part of the set.
    if let (Some(dir), Some(leaf)) = (PathBuf::from(&row.path).parent(), PathBuf::from(&row.path).file_name()) {
        let leaf = leaf.to_string_lossy().to_string();
        let base = crate::hf::parse_shard_marker_pub(&leaf).map_or_else(|| leaf.trim_end_matches(".gguf").to_string(), |(_, _, base)| base);
        if let Ok(entries) = std::fs::read_dir(dir) {
            for e in entries.flatten() {
                let fname = e.file_name().to_string_lossy().to_string();
                if fname.starts_with(&base)
                    && fname.contains("-of-")
                    && fname.ends_with(".gguf")
                {
                    files.push(e.path());
                }
            }
        }
    }
    if let Some(mm) = &row.mmproj_path {
        files.push(PathBuf::from(mm));
    }

    for f in files {
        if f.exists() {
            std::fs::remove_file(&f)
                .map_err(|e| anyhow!("delete {}: {e}", f.display()))?;
        }
    }
    store.delete_model(name)?;
    Ok(())
}

/// Alias a model under a new name (`pallama cp`): zero-byte hardlink of the
/// GGUF (both files live in the same models dir, so linking always works)
/// plus a new store row. No blob ceremony, no byte copies.
pub fn copy_model(dirs: &PallamaDirs, src: &str, dst: &str) -> Result<()> {
    if instance_running(dirs, src) {
        return Err(anyhow!("model {src} is currently running; copy after it unloads"));
    }
    let store = Store::open(dirs)?;
    let row = store
        .get_model(src)?
        .ok_or_else(|| anyhow!("no such model: {src}"))?;
    if store.get_model(dst)?.is_some() {
        return Err(anyhow!("model {dst} already exists"));
    }
    let src_path = PathBuf::from(&row.path);
    let leaf = src_path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    // The alias hardlink MUST live at its own path. Reusing the source's
    // leaf name made the alias row point at the ORIGINAL file — rm on the
    // alias then deleted the source (live data-loss incident 2026-09-05).
    let dst_path = dirs.models_dir().join(format!("{dst}__alias__{leaf}"));
    if dst_path == src_path {
        return Err(anyhow!("alias path collides with the source file; refusing"));
    }
    if !dst_path.exists() {
        std::fs::hard_link(&src_path, &dst_path)
            .map_err(|e| anyhow!("hardlink {} -> {}: {e}", dst_path.display(), src_path.display()))?;
    }
    store.upsert_model(&pallama_core::ModelRow {
        name: dst.to_string(),
        repo: row.repo.clone(),
        quant: row.quant.clone(),
        path: dst_path.display().to_string(),
        bytes: row.bytes,
        sha256: row.sha256.clone(),
        mmproj_path: row.mmproj_path.clone(),
        shards: row.shards,
        arch: row.arch.clone(),
        params: row.params,
        ctx_train: row.ctx_train,
        pulled_at: row.pulled_at,
    })?;
    Ok(())
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    fn setup() -> (tempfile::TempDir, PallamaDirs) {
        let tmp = tempfile::tempdir().unwrap();
        let dirs = PallamaDirs {
            config_dir: tmp.path().join("cfg"),
            data_dir: tmp.path().join("data"),
        };
        dirs.ensure().unwrap();
        (tmp, dirs)
    }

    #[test]
    fn unit__remove_model__deletes_files_and_row() {
        let (_t, dirs) = setup();
        let d = dirs.models_dir();
        std::fs::write(d.join("m-q4_k_m-00001-of-00002.gguf"), b"a").unwrap();
        std::fs::write(d.join("m-q4_k_m-00002-of-00002.gguf"), b"b").unwrap();
        std::fs::write(d.join("mmproj-m.gguf"), b"p").unwrap();
        let store = Store::open(&dirs).unwrap();
        store
            .upsert_model(&pallama_core::ModelRow {
                name: "m".into(),
                repo: "o/m".into(),
                quant: "Q4_K_M".into(),
                path: d.join("m-q4_k_m-00001-of-00002.gguf").display().to_string(),
                bytes: 2,
                sha256: None,
                mmproj_path: Some(d.join("mmproj-m.gguf").display().to_string()),
                shards: 2,
                arch: None,
                params: None,
                ctx_train: None,
                pulled_at: 1,
            })
            .unwrap();

        remove_model(&dirs, "m").unwrap();
        assert!(store.get_model("m").unwrap().is_none());
        assert!(!d.join("m-q4_k_m-00001-of-00002.gguf").exists());
        assert!(!d.join("m-q4_k_m-00002-of-00002.gguf").exists());
        assert!(!d.join("mmproj-m.gguf").exists());
    }

    #[test]
    fn unit__remove_model__refuses_while_running() {
        let (_t, dirs) = setup();
        std::fs::write(dirs.run_dir().join("m.pid"), "123").unwrap();
        let err = remove_model(&dirs, "m").unwrap_err();
        assert!(err.to_string().contains("running"), "{err}");
    }

    #[test]
    fn unit__remove_model__unknown_model__named_error() {
        let (_t, dirs) = setup();
        let err = remove_model(&dirs, "nope").unwrap_err();
        assert!(err.to_string().contains("no such model"), "{err}");
    }

    #[test]
    fn unit__copy_model__hardlink_alias_zero_byte_copy() {
        let (_t, dirs) = setup();
        let gguf = dirs.models_dir().join("m-q4_k_m.gguf");
        std::fs::write(&gguf, b"gguf-bytes").unwrap();
        let s = pallama_core::Store::open(&dirs).unwrap();
        s.upsert_model(&pallama_core::ModelRow {
            name: "m".into(),
            repo: "r".into(),
            quant: "Q4_K_M".into(),
            path: gguf.display().to_string(),
            bytes: 10,
            sha256: None,
            mmproj_path: None,
            shards: 1,
            arch: None,
            params: None,
            ctx_train: None,
            pulled_at: 1,
        })
        .unwrap();
        copy_model(&dirs, "m", "m-alias").unwrap();
        let alias = s.get_model("m-alias").unwrap().unwrap();
        // The alias row must point at its OWN path (same-leaf naming made
        // rm-on-alias delete the source — live incident 2026-09-05).
        assert_ne!(alias.path, gguf.display().to_string());
        assert!(alias.path.contains("m-alias__alias__"), "{}", alias.path);
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt as _;
            let i1 = std::fs::metadata(&gguf).unwrap().ino();
            let i2 = std::fs::metadata(&alias.path).unwrap().ino();
            assert_eq!(i1, i2, "alias must be a hardlink, not a copy");
        }
        assert!(copy_model(&dirs, "m", "m-alias").is_err(), "duplicate alias refused");
    }
}
