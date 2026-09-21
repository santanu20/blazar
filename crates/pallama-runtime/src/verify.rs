//! Post-pull integrity verification for `pallama pull --verify`.
//!
//! The store row's `sha256` means different things per pull lane:
//!
//! * registry (OCI) rows: hex content digest of the GGUF blob,
//! * HF GGUF rows: the LFS sha256 when the API provides one,
//! * safetensors dir rows: a revision-identity hash, NOT a file digest.
//!
//! Only the first two can be re-hashed and compared, so dir rows are
//! refused loudly instead of silently "failing" a digest that was never a
//! content hash. Rows with no digest at all (`sha256 = NULL`, e.g. some
//! HF repos omit LFS metadata) are reported as unverifiable rather than
//! treated as corrupt.

use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Result};
use sha2::{Digest, Sha256};

use pallama_core::{ModelRow, PallamaDirs, Store};

/// Streaming hash buffer: big enough to amortize syscalls on multi-GB
/// weights, small enough to stay cache-friendly (same size class the
/// download lanes use).
const HASH_BUF_BYTES: usize = 8 * 1024 * 1024;

/// Outcome of verifying one model row. `Verified`/`Mismatch` carry the
/// recomputed digest so the CLI can print it; the refusal variants name
/// the exact row and path so the user knows what to re-pull.
#[derive(Debug, Clone, PartialEq)]
pub enum VerifyReport {
    Verified {
        name: String,
        bytes: u64,
        sha256: String,
    },
    Mismatch {
        name: String,
        path: PathBuf,
        expected: String,
        actual: String,
        bytes: u64,
    },
    NoDigest {
        name: String,
        path: PathBuf,
    },
    DirRowUnsupported {
        name: String,
        path: PathBuf,
    },
    NotFound {
        name: String,
    },
}

/// Verify the on-disk file behind store row `name` against its recorded
/// sha256. The heavy hash runs on the blocking pool so a multi-GB digest
/// never stalls the async runtime.
pub async fn verify_model(dirs: &PallamaDirs, name: &str) -> Result<VerifyReport> {
    // Read-only intent: never materialize a store (and a data dir) just to
    // answer "is it there" — a missing DB means the model cannot exist.
    if !dirs.db_file().is_file() {
        return Ok(VerifyReport::NotFound {
            name: name.to_string(),
        });
    }
    let store = Store::open(dirs)?;
    let resolved = store.resolve_model_name(name);
    let row = store
        .get_model(&resolved)?
        .ok_or_else(|| anyhow!("model {resolved:?} not found in the local store"))?;
    verify_row(row).await
}

async fn verify_row(row: ModelRow) -> Result<VerifyReport> {
    let name = row.name.clone();
    let path = PathBuf::from(&row.path);

    if path.is_dir() {
        return Ok(VerifyReport::DirRowUnsupported { name, path });
    }
    let Some(expected) = row.sha256 else {
        return Ok(VerifyReport::NoDigest { name, path });
    };

    let hash_path = path.clone();
    let (bytes, actual) = tokio::task::spawn_blocking(move || sha256_file(&hash_path))
        .await
        .map_err(|e| {
            anyhow!(
                "sha256 worker panicked while hashing {}: {e}",
                path.display()
            )
        })?
        .map_err(|e| anyhow!("hash {}: {e}", path.display()))?;

    if actual == expected {
        Ok(VerifyReport::Verified {
            name,
            bytes,
            sha256: actual,
        })
    } else {
        Ok(VerifyReport::Mismatch {
            name,
            path,
            expected,
            actual,
            bytes,
        })
    }
}

/// Stream a file through sha256; returns (file size hashed, hex digest).
fn sha256_file(path: &Path) -> std::io::Result<(u64, String)> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; HASH_BUF_BYTES];
    let mut bytes = 0u64;
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        bytes += n as u64;
    }
    Ok((bytes, format!("{:x}", hasher.finalize())))
}

// Test names use the project's `unit__area__behavior` convention.
#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    fn sandbox() -> (tempfile::TempDir, PallamaDirs) {
        let tmp = tempfile::tempdir().unwrap();
        let dirs = PallamaDirs {
            config_dir: tmp.path().join("cfg"),
            data_dir: tmp.path().join("data"),
        };
        (tmp, dirs)
    }

    fn row_for(path: &std::path::Path, sha256: Option<String>) -> ModelRow {
        ModelRow {
            name: "m".into(),
            repo: "o/m".into(),
            quant: "Q4_K_M".into(),
            path: path.display().to_string(),
            bytes: 4,
            sha256,
            mmproj_path: None,
            shards: 1,
            arch: None,
            params: None,
            ctx_train: None,
            pulled_at: 1,
        }
    }

    #[tokio::test]
    async fn unit__verify__matching_digest_reports_verified() {
        let (tmp, dirs) = sandbox();
        let f = tmp.path().join("m.gguf");
        let payload = b"gguf-bytes";
        std::fs::write(&f, payload).unwrap();
        let mut hex = Sha256::new();
        hex.update(payload);
        let digest = format!("{:x}", hex.finalize());

        let store = Store::open(&dirs).unwrap();
        store.upsert_model(&row_for(&f, Some(digest))).unwrap();

        let report = verify_model(&dirs, "m").await.unwrap();
        assert_eq!(
            report,
            VerifyReport::Verified {
                name: "m".into(),
                bytes: payload.len() as u64,
                sha256: store.get_model("m").unwrap().unwrap().sha256.unwrap(),
            }
        );
    }

    #[tokio::test]
    async fn unit__verify__tampered_file_reports_mismatch() {
        let (tmp, dirs) = sandbox();
        let f = tmp.path().join("m.gguf");
        std::fs::write(&f, b"tampered").unwrap();

        let store = Store::open(&dirs).unwrap();
        store
            .upsert_model(&row_for(&f, Some("00".repeat(32))))
            .unwrap();

        let report = verify_model(&dirs, "m").await.unwrap();
        let VerifyReport::Mismatch { bytes, actual, .. } = report else {
            panic!("expected Mismatch, got {report:?}");
        };
        assert_eq!(bytes, 8);
        assert_eq!(actual.len(), 64, "hex sha256, got {actual}");
    }

    #[tokio::test]
    async fn unit__verify__dir_row_refused_not_hashed() {
        let (tmp, dirs) = sandbox();
        let d = tmp.path().join("model.d");
        std::fs::create_dir_all(&d).unwrap();

        let store = Store::open(&dirs).unwrap();
        store
            .upsert_model(&row_for(&d, Some("00".repeat(32))))
            .unwrap();

        let report = verify_model(&dirs, "m").await.unwrap();
        assert_eq!(
            report,
            VerifyReport::DirRowUnsupported {
                name: "m".into(),
                path: d,
            }
        );
    }

    #[tokio::test]
    async fn unit__verify__null_digest_reports_no_digest() {
        let (tmp, dirs) = sandbox();
        let f = tmp.path().join("m.gguf");
        std::fs::write(&f, b"gguf").unwrap();

        let store = Store::open(&dirs).unwrap();
        store.upsert_model(&row_for(&f, None)).unwrap();

        let report = verify_model(&dirs, "m").await.unwrap();
        assert_eq!(
            report,
            VerifyReport::NoDigest {
                name: "m".into(),
                path: f,
            }
        );
    }

    #[tokio::test]
    async fn unit__verify__absent_model_without_db_reports_not_found() {
        let (_tmp, dirs) = sandbox();
        // No store ever opened -> no DB file: must NOT create one.
        let report = verify_model(&dirs, "ghost").await.unwrap();
        assert_eq!(
            report,
            VerifyReport::NotFound {
                name: "ghost".into()
            }
        );
        assert!(!dirs.db_file().is_file(), "verify must not create a store");
    }

    #[tokio::test]
    async fn unit__verify__row_without_file_is_loud_error() {
        let (tmp, dirs) = sandbox();
        let f = tmp.path().join("vanished.gguf");

        let store = Store::open(&dirs).unwrap();
        store
            .upsert_model(&row_for(&f, Some("00".repeat(32))))
            .unwrap();

        let err = verify_model(&dirs, "m").await.unwrap_err().to_string();
        assert!(
            err.contains("vanished.gguf"),
            "error must name the missing file: {err}"
        );
    }
}
