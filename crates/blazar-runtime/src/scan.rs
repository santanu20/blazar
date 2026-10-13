//! Cross-runtime model discovery: what this machine already has on disk
//! from OTHER runtimes — an ollama store, the Hugging Face hub cache, or
//! an LM Studio install — and the blazar command that adopts each entry
//! without a re-download. Discovery is read-only and cheap: sizes come
//! from manifests or `fs::metadata`, nothing is hashed or opened for
//! write; the actual reuse at pull/import time is still gated by the
//! byte-exact verification in the pull lanes (`crate::hf::reuse_byte_exact`).

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use crate::hf::hf_hub_root;
use crate::registry::{
    OLLAMA_MODEL_LAYER, OLLAMA_PROJECTOR_LAYER, RegistryConfig, RegistryManifest, ollama_blob_path,
    ollama_store_roots,
};

/// Which runtime's on-disk store a reusable model was found in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ReuseSource {
    Ollama,
    HfHub,
    LmStudio,
}

impl std::fmt::Display for ReuseSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            ReuseSource::Ollama => "ollama",
            ReuseSource::HfHub => "hf-hub",
            ReuseSource::LmStudio => "lm-studio",
        })
    }
}

/// One model already on disk that blazar can adopt with zero download.
#[derive(Debug, Clone)]
pub struct ReusableModel {
    pub source: ReuseSource,
    /// What the user types after the suggestion command: `qwen3:0.6b`,
    /// `org/repo`, or a file path for loose imports.
    pub id: String,
    /// Quant label when the source knows it (ollama config blob);
    /// `None` shows as `-`.
    pub quant: Option<String>,
    pub bytes: u64,
    /// Ready-to-run command that adopts this model (`blazar pull …` /
    /// `blazar import …`).
    pub suggestion: String,
}

/// Scan every probed runtime store on this machine. Absent stores are
/// skipped silently (probing is read-only); output is sorted by source
/// then id so listing output is stable across runs.
#[must_use]
pub fn scan_reusable_models() -> Vec<ReusableModel> {
    let mut rows = Vec::new();
    for root in ollama_store_roots().into_iter().filter(|r| r.is_dir()) {
        rows.extend(scan_ollama_store_at(&root));
    }
    if let Some(root) = hf_hub_root().filter(|r| r.is_dir()) {
        rows.extend(scan_hf_hub_at(&root));
    }
    for root in lm_studio_roots().into_iter().filter(|r| r.is_dir()) {
        rows.extend(scan_lm_studio_at(&root));
    }
    rows.sort_by(|a, b| (a.source, &a.id).cmp(&(b.source, &b.id)));
    rows
}

/// Walk an ollama store's manifest tree
/// (`manifests/registry.ollama.ai/<ns>/<model>/<tag>`) and list one row
/// per (model, tag). Sizes come from the manifest descriptors — no blob
/// hashing at scan time; a later `blazar pull` verifies bytes before
/// linking. Manifests without a model layer, or unreadable/corrupt ones,
/// are skipped: a listing must never crash on a weird store.
pub(crate) fn scan_ollama_store_at(root: &Path) -> Vec<ReusableModel> {
    let registry = root.join("manifests").join("registry.ollama.ai");
    let mut rows = Vec::new();
    for ns in sub_dirs(&registry) {
        let ns_name = ns.file_name().to_string_lossy().to_string();
        for model in sub_dirs(&ns.path()) {
            let model_name = model.file_name().to_string_lossy().to_string();
            let Some(tags) = read_dir_entries(&model.path()) else {
                continue;
            };
            for tag in tags {
                if !tag.file_type().is_ok_and(|t| t.is_file()) {
                    continue;
                }
                let tag_name = tag.file_name().to_string_lossy().to_string();
                let Some((bytes, quant)) = ollama_manifest_facts(root, &tag.path()) else {
                    continue;
                };
                // `library/` is ollama's default namespace and stays
                // implicit everywhere users type targets.
                let repo = if ns_name == "library" {
                    model_name.clone()
                } else {
                    format!("{ns_name}/{model_name}")
                };
                let id = format!("{repo}:{tag_name}");
                rows.push(ReusableModel {
                    source: ReuseSource::Ollama,
                    suggestion: format!("blazar pull {id}"),
                    id,
                    quant,
                    bytes,
                });
            }
        }
    }
    rows
}

/// (model+projector layer bytes, best-effort quant) from one manifest
/// file; `None` when it is not a parseable manifest with a model layer.
fn ollama_manifest_facts(root: &Path, manifest_path: &Path) -> Option<(u64, Option<String>)> {
    let raw = std::fs::read(manifest_path).ok()?;
    let manifest: RegistryManifest = serde_json::from_slice(&raw).ok()?;
    let model = manifest
        .layers
        .iter()
        .find(|l| l.media_type == OLLAMA_MODEL_LAYER)?;
    let mut bytes = model.size;
    if let Some(proj) = manifest
        .layers
        .iter()
        .find(|l| l.media_type == OLLAMA_PROJECTOR_LAYER)
    {
        bytes += proj.size;
    }
    let quant = manifest
        .config
        .as_ref()
        .and_then(|cfg| std::fs::read(ollama_blob_path(root, &cfg.digest)).ok())
        .and_then(|b| serde_json::from_slice::<RegistryConfig>(&b).ok())
        .and_then(|c| c.file_type);
    Some((bytes, quant))
}

/// List every `models--<org>--<name>` repo in a hub cache that holds at
/// least one GGUF or safetensors file. One row per repo (the suggestion
/// is repo-level); sizes are summed over the repo's snapshot files,
/// deduplicated by blob target so two revisions symlinked at the same
/// blob count once. Non-model repos (ONNX exports, configs-only) yield
/// no row.
pub(crate) fn scan_hf_hub_at(root: &Path) -> Vec<ReusableModel> {
    let Some(repos) = read_dir_entries(root) else {
        return Vec::new();
    };
    let mut rows = Vec::new();
    for repo in repos {
        if !repo.file_type().is_ok_and(|t| t.is_dir()) {
            continue;
        }
        let dir_name = repo.file_name().to_string_lossy().to_string();
        let Some(repo_id) = dir_name.strip_prefix("models--") else {
            continue;
        };
        // huggingface_hub encodes `org/name` as `org--name`; org names
        // cannot contain `--`, so the FIRST separator is the split.
        let Some((org, name)) = repo_id.split_once("--") else {
            continue;
        };
        let mut files = Vec::new();
        collect_model_files(&repo.path().join("snapshots"), &mut files);
        let mut seen = HashSet::new();
        let mut bytes = 0u64;
        for file in files {
            // Symlinked snapshot files resolve to their blob; canonical
            // targets dedupe cross-revision copies of the same content.
            let Ok(target) = file.canonicalize() else {
                continue;
            };
            let Ok(meta) = std::fs::metadata(&file) else {
                continue;
            };
            if seen.insert(target) {
                bytes += meta.len();
            }
        }
        if bytes == 0 {
            continue;
        }
        let id = format!("{org}/{name}");
        rows.push(ReusableModel {
            source: ReuseSource::HfHub,
            suggestion: format!("blazar pull {id}"),
            id,
            quant: None,
            bytes,
        });
    }
    rows
}

/// Recursive *.gguf / *.safetensors collector for a snapshots tree
/// (repos keep model weights at the root or in subdirs like `vae/`).
fn collect_model_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Some(entries) = read_dir_entries(dir) else {
        return;
    };
    for entry in entries {
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if file_type.is_dir() {
            collect_model_files(&entry.path(), out);
        } else if entry.file_name().to_string_lossy().ends_with(".gguf")
            || entry
                .file_name()
                .to_string_lossy()
                .ends_with(".safetensors")
        {
            out.push(entry.path());
        }
    }
}

/// Candidate LM Studio models directories. The modern default is
/// `~/.lmstudio/models` (0.3+); older builds used per-OS cache paths.
/// The location is configurable in-app (no env var), so probing is
/// best-effort by design — absent roots are a no-op.
fn lm_studio_roots() -> Vec<PathBuf> {
    let Some(home) = dirs::home_dir() else {
        return Vec::new();
    };
    vec![
        home.join(".lmstudio").join("models"),
        home.join(".cache").join("lm-studio").join("models"),
        home.join("Library")
            .join("Application Support")
            .join("LM Studio")
            .join("models"),
    ]
}

/// Loose GGUFs under an LM Studio models tree (`<publisher>/<model>/…`),
/// one row per file — each is adoptable via `blazar import <path>`.
pub(crate) fn scan_lm_studio_at(root: &Path) -> Vec<ReusableModel> {
    let mut files = Vec::new();
    collect_gguf_files(root, &mut files);
    files
        .into_iter()
        .filter_map(|file| {
            let bytes = std::fs::metadata(&file).ok()?.len();
            let id = file.file_name()?.to_string_lossy().to_string();
            Some(ReusableModel {
                source: ReuseSource::LmStudio,
                suggestion: format!("blazar import {}", file.display()),
                id,
                quant: None,
                bytes,
            })
        })
        .collect()
}

/// LM Studio trees hold GGUFs only (safetensors there ship as `.jml`
/// bundles blazar has no lane for yet), hence a GGUF-only walker.
fn collect_gguf_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Some(entries) = read_dir_entries(dir) else {
        return;
    };
    for entry in entries {
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if file_type.is_dir() {
            collect_gguf_files(&entry.path(), out);
        } else if entry.file_name().to_string_lossy().ends_with(".gguf") {
            out.push(entry.path());
        }
    }
}

fn read_dir_entries(dir: &Path) -> Option<Vec<std::fs::DirEntry>> {
    let entries = std::fs::read_dir(dir).ok()?;
    Some(entries.filter_map(std::result::Result::ok).collect())
}

fn sub_dirs(dir: &Path) -> Vec<std::fs::DirEntry> {
    read_dir_entries(dir)
        .unwrap_or_default()
        .into_iter()
        .filter(|e| e.file_type().is_ok_and(|t| t.is_dir()))
        .collect()
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    fn sha256_hex(bytes: &[u8]) -> String {
        let digest = <sha2::Sha256 as sha2::Digest>::digest(bytes);
        digest.iter().fold(String::new(), |mut out, b| {
            use std::fmt::Write as _;
            let _ = write!(out, "{b:02x}");
            out
        })
    }

    /// Minimal honest ollama store: manifest with config + model layer,
    /// blobs on disk for both. Returns the store root.
    fn ollama_fixture(tmp: &Path, model_bytes: &[u8]) -> PathBuf {
        let root = tmp.join("ollama-store");
        let config = br#"{"file_type":"Q4_K_M"}"#;
        let config_digest = format!("sha256:{}", sha256_hex(config));
        let model_digest = format!("sha256:{}", sha256_hex(model_bytes));
        let manifest = format!(
            r#"{{"schemaVersion":2,"config":{{"mediaType":"application/vnd.docker.container.image.v1+json","digest":"{config_digest}","size":{}}},"layers":[{{"mediaType":"application/vnd.ollama.image.model","digest":"{model_digest}","size":{}}}]}}"#,
            config.len(),
            model_bytes.len()
        );
        // Real ollama layout: the TAG is a manifest file directly under
        // the model dir (…/library/qwen3/0.6b is a file, not a dir).
        let dir = root
            .join("manifests")
            .join("registry.ollama.ai")
            .join("library")
            .join("qwen3");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("0.6b"), manifest).unwrap();
        std::fs::create_dir_all(root.join("blobs")).unwrap();
        std::fs::write(
            root.join("blobs").join(config_digest.replace(':', "-")),
            config,
        )
        .unwrap();
        std::fs::write(
            root.join("blobs").join(model_digest.replace(':', "-")),
            model_bytes,
        )
        .unwrap();
        root
    }

    #[test]
    fn unit__scan_ollama_store_at__manifest_tree_yields_pull_rows() {
        let tmp = tempfile::tempdir().unwrap();
        let root = ollama_fixture(tmp.path(), b"model-bytes");
        let rows = scan_ollama_store_at(&root);
        assert_eq!(rows.len(), 1, "{rows:?}");
        assert_eq!(rows[0].id, "qwen3:0.6b");
        assert_eq!(rows[0].source, ReuseSource::Ollama);
        assert_eq!(rows[0].quant.as_deref(), Some("Q4_K_M"));
        assert_eq!(rows[0].bytes, b"model-bytes".len() as u64);
        assert_eq!(rows[0].suggestion, "blazar pull qwen3:0.6b");
    }

    #[test]
    fn unit__scan_ollama_store_at__library_namespace_stays_implicit() {
        let tmp = tempfile::tempdir().unwrap();
        let root = ollama_fixture(tmp.path(), b"m");
        // A second manifest under a user namespace: id keeps the ns.
        let dir = root
            .join("manifests")
            .join("registry.ollama.ai")
            .join("jdoe")
            .join("mistral");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("7b"), br#"{"schemaVersion":2,"layers":[]}"#).unwrap();
        let rows = scan_ollama_store_at(&root);
        let ids: Vec<&str> = rows.iter().map(|r| r.id.as_str()).collect();
        // The jdoe manifest has no model layer -> skipped; only library
        // survives, proving the ns-drop logic on the surviving row.
        assert_eq!(ids, vec!["qwen3:0.6b"]);
    }

    #[test]
    fn unit__scan_ollama_store_at__corrupt_manifests_skipped() {
        let tmp = tempfile::tempdir().unwrap();
        let root = ollama_fixture(tmp.path(), b"m");
        // Corrupt the manifest body: scan must skip, not crash.
        std::fs::write(
            root.join("manifests/registry.ollama.ai/library/qwen3/0.6b"),
            b"not json",
        )
        .unwrap();
        assert!(scan_ollama_store_at(&root).is_empty());
    }

    #[test]
    fn unit__scan_ollama_store_at__empty_or_absent_root_is_noop() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(scan_ollama_store_at(&tmp.path().join("nothing")).is_empty());
    }

    #[test]
    fn unit__scan_hf_hub_at__gguf_repo_listed_non_model_repo_ignored() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("hub");
        let gguf = root
            .join("models--Qwen--Qwen2.5-0.5B-Instruct")
            .join("snapshots")
            .join("rev1");
        std::fs::create_dir_all(&gguf).unwrap();
        std::fs::write(gguf.join("model.q4_k_m.gguf"), vec![7u8; 1024]).unwrap();
        let onnx = root
            .join("models--welcomyou--doclayout-yolo-onnx")
            .join("snapshots")
            .join("rev1");
        std::fs::create_dir_all(&onnx).unwrap();
        std::fs::write(onnx.join("model.onnx"), vec![1u8; 4096]).unwrap();

        let rows = scan_hf_hub_at(&root);
        assert_eq!(rows.len(), 1, "{rows:?}");
        assert_eq!(rows[0].id, "Qwen/Qwen2.5-0.5B-Instruct");
        assert_eq!(rows[0].source, ReuseSource::HfHub);
        assert_eq!(rows[0].bytes, 1024);
        assert_eq!(rows[0].suggestion, "blazar pull Qwen/Qwen2.5-0.5B-Instruct");
    }

    #[test]
    fn unit__scan_hf_hub_at__subdir_weights_and_nested_names_counted() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("hub");
        let repo = root
            .join("models--black-forest-labs--FLUX.1-schnell")
            .join("snapshots")
            .join("rev1");
        std::fs::create_dir_all(repo.join("vae")).unwrap();
        std::fs::write(repo.join("vae").join("ae.safetensors"), vec![3u8; 512]).unwrap();
        std::fs::write(repo.join("model.safetensors"), vec![9u8; 256]).unwrap();
        // Repo names may contain dashes (FLUX.1-schnell): the FIRST `--`
        // splits org from name.
        let rows = scan_hf_hub_at(&root);
        assert_eq!(rows.len(), 1, "{rows:?}");
        assert_eq!(rows[0].id, "black-forest-labs/FLUX.1-schnell");
        assert_eq!(rows[0].bytes, 512 + 256);
    }

    #[cfg(target_family = "unix")]
    #[test]
    fn unit__scan_hf_hub_at__same_blob_in_two_revisions_counted_once() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("hub");
        let repo_dir = root.join("models--org--name").join("snapshots");
        let blobs = root.join("blobs");
        std::fs::create_dir_all(repo_dir.join("rev1")).unwrap();
        std::fs::create_dir_all(repo_dir.join("rev2")).unwrap();
        std::fs::create_dir_all(&blobs).unwrap();
        let blob = blobs.join("aaa");
        std::fs::write(&blob, vec![5u8; 2048]).unwrap();
        std::os::unix::fs::symlink(&blob, repo_dir.join("rev1").join("model.gguf")).unwrap();
        std::os::unix::fs::symlink(&blob, repo_dir.join("rev2").join("model.gguf")).unwrap();

        let rows = scan_hf_hub_at(&root);
        assert_eq!(rows.len(), 1, "{rows:?}");
        assert_eq!(rows[0].bytes, 2048, "one blob must count once across revs");
    }

    #[test]
    fn unit__scan_lm_studio_at__nested_ggufs_listed_with_import_suggestion() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("models");
        let nested = root.join("lmstudio-community").join("qwen").join("1.5b");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(nested.join("Qwen2.5-1.5B-Q4_K_M.gguf"), vec![1u8; 128]).unwrap();
        std::fs::write(nested.join("config.json"), b"{}").unwrap();

        let rows = scan_lm_studio_at(&root);
        assert_eq!(rows.len(), 1, "{rows:?}");
        assert_eq!(rows[0].id, "Qwen2.5-1.5B-Q4_K_M.gguf");
        assert_eq!(rows[0].source, ReuseSource::LmStudio);
        assert_eq!(rows[0].bytes, 128);
        assert_eq!(
            rows[0].suggestion,
            format!(
                "blazar import {}",
                nested.join("Qwen2.5-1.5B-Q4_K_M.gguf").display()
            )
        );
    }

    #[test]
    fn unit__scan_lm_studio_at__empty_root_is_noop() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(scan_lm_studio_at(&tmp.path().join("nothing")).is_empty());
    }
}
