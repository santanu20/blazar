//! KV-checkpoint identity manifests (#20): every session save writes a
//! sibling `<file>.identity.json` recording the exact runtime shape the
//! KV bytes were produced under (engine build, model bytes, ctx, cache
//! type, KV unification, slots). Restores verify before injecting — KV
//! state from a different shape is silent garbage, not a warm start.
//!
//! Unknown-vs-unknown never fails a restore: a missing `model_sha`
//! (pulled before hashing existed) is "unverifiable", not "mismatched" —
//! only fields present on BOTH sides are compared.

use std::path::Path;
use std::path::PathBuf;

use serde::Deserialize;
use serde::Serialize;

use crate::config::Config;
use crate::dirs::BlazarDirs;
use crate::store::Store;

/// The runtime shape under which a KV checkpoint was written.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionIdentity {
    pub blazar_version: String,
    pub engine_tag: String,
    pub engine_sha: String,
    #[serde(default)]
    pub model_sha: Option<String>,
    pub ctx: u32,
    pub cache_type: String,
    #[serde(default)]
    pub kv_unified: Option<bool>,
    pub slots: u32,
    pub saved_at_unix: u64,
}

/// Sibling manifest path: `<session-file>.identity.json` (append — the
/// session name may itself contain dots).
#[must_use]
pub fn manifest_path(session_file: &Path) -> PathBuf {
    let mut p = session_file.as_os_str().to_os_string();
    p.push(".identity.json");
    PathBuf::from(p)
}

/// Snapshot the current runtime shape for `model`. `None` = cannot
/// determine (store failure / no active engine): callers warn and skip
/// the manifest rather than failing the save.
#[must_use]
pub fn build(dirs: &BlazarDirs, config: &Config, model: &str) -> Option<SessionIdentity> {
    let store = Store::open(dirs).ok()?;
    let engine = store.active_engine().ok()??;
    let model_sha = store.get_model(model).ok().flatten().and_then(|m| m.sha256);
    Some(SessionIdentity {
        blazar_version: env!("CARGO_PKG_VERSION").to_string(),
        engine_tag: engine.tag,
        engine_sha: engine.sha256,
        model_sha,
        ctx: config.effective_ctx(model),
        cache_type: config.effective_cache_type(model).to_string(),
        kv_unified: config.effective_kv_unified(model),
        // F121: record the overlay-effective slot count, not the raw
        // global — a per-model `slots` override changes what actually
        // runs, and the restore check compares against this value.
        // (0 = upstream auto: recorded as configured; the profile
        // compiler resolves auto downstream.)
        slots: config
            .overlay_for(model)
            .slots
            .filter(|&s| s != 0)
            .unwrap_or(config.slots),
        saved_at_unix: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs()),
    })
}

/// Write the manifest beside a just-saved checkpoint. Best-effort at
/// call sites: an IO failure logs a warning, never fails the save.
pub fn write_manifest(session_file: &Path, id: &SessionIdentity) -> std::io::Result<()> {
    let bytes = serde_json::to_vec_pretty(id).map_err(|e| std::io::Error::other(e.to_string()))?;
    std::fs::write(manifest_path(session_file), bytes)
}

/// Read the manifest for a checkpoint file; `None` = missing or
/// unreadable (callers treat as unverifiable and proceed with a warn).
#[must_use]
pub fn read_manifest(session_file: &Path) -> Option<SessionIdentity> {
    let raw = std::fs::read(manifest_path(session_file)).ok()?;
    serde_json::from_slice(&raw).ok()
}

/// Field-by-field comparison. Empty vec = safe to restore. Fields
/// absent (`None`) on either side are skipped — unverifiable is not a
/// mismatch.
#[must_use]
pub fn verify(saved: &SessionIdentity, current: &SessionIdentity) -> Vec<String> {
    let mut diffs = Vec::new();
    let mut push = |field: &str, was: String, now: String| {
        diffs.push(format!("{field}: saved {was}, now {now}"));
    };
    if saved.blazar_version != current.blazar_version {
        push(
            "blazar_version",
            saved.blazar_version.clone(),
            current.blazar_version.clone(),
        );
    }
    if saved.engine_tag != current.engine_tag {
        push(
            "engine_tag",
            saved.engine_tag.clone(),
            current.engine_tag.clone(),
        );
    }
    if saved.engine_sha != current.engine_sha {
        push(
            "engine_sha",
            saved.engine_sha.clone(),
            current.engine_sha.clone(),
        );
    }
    if let (Some(s), Some(c)) = (&saved.model_sha, &current.model_sha) {
        if s != c {
            push("model_sha", s.clone(), c.clone());
        }
    }
    if saved.ctx != current.ctx {
        push("ctx", saved.ctx.to_string(), current.ctx.to_string());
    }
    if saved.cache_type != current.cache_type {
        push(
            "cache_type",
            saved.cache_type.clone(),
            current.cache_type.clone(),
        );
    }
    if let (Some(s), Some(c)) = (saved.kv_unified, current.kv_unified) {
        if s != c {
            push("kv_unified", s.to_string(), c.to_string());
        }
    }
    if saved.slots != current.slots {
        push("slots", saved.slots.to_string(), current.slots.to_string());
    }
    diffs
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    fn ident() -> SessionIdentity {
        SessionIdentity {
            blazar_version: "0.4.0".into(),
            engine_tag: "b4091".into(),
            engine_sha: "abc".into(),
            model_sha: Some("def".into()),
            ctx: 8192,
            cache_type: "q8_0".into(),
            kv_unified: Some(true),
            slots: 2,
            saved_at_unix: 1_700_000_000,
        }
    }

    #[test]
    fn unit__session_identity__manifest_roundtrip_and_path() {
        let dir = tempfile::tempdir().expect("tempdir");
        let ckpt = dir.path().join("conv.v2");
        let mp = manifest_path(&ckpt);
        assert!(
            mp.to_string_lossy().ends_with("conv.v2.identity.json"),
            "dots in the name survive: {}",
            mp.display()
        );
        assert!(!mp.exists());
        assert!(read_manifest(&ckpt).is_none(), "missing = None");
        write_manifest(&ckpt, &ident()).expect("write");
        let back = read_manifest(&ckpt).expect("read back");
        assert_eq!(back, ident());
    }

    #[test]
    fn unit__session_identity__verify_match_and_tamper_each_field() {
        let base = ident();
        assert!(
            verify(&base, &ident()).is_empty(),
            "identical shapes verify clean"
        );
        let mut t = ident();
        t.blazar_version = "0.5.0".into();
        assert!(verify(&base, &t)
            .iter()
            .any(|d| d.contains("blazar_version")));
        let mut t = ident();
        t.engine_sha = "zzz".into();
        assert!(verify(&base, &t).iter().any(|d| d.contains("engine_sha")));
        let mut t = ident();
        t.ctx = 16_384;
        assert!(verify(&base, &t)
            .iter()
            .any(|d| d.contains("ctx: saved 8192")));
        let mut t = ident();
        t.cache_type = "q4_0".into();
        assert!(verify(&base, &t).iter().any(|d| d.contains("cache_type")));
        let mut t = ident();
        t.kv_unified = Some(false);
        assert!(verify(&base, &t).iter().any(|d| d.contains("kv_unified")));
        let mut t = ident();
        t.slots = 4;
        assert!(verify(&base, &t).iter().any(|d| d.contains("slots")));
    }

    #[test]
    fn unit__session_identity__unknown_fields_skip_not_mismatch() {
        let mut saved = ident();
        saved.model_sha = None; // pre-hashing pull
        saved.kv_unified = None; // pre-knob manifest
        let current = ident();
        assert!(
            verify(&saved, &current).is_empty(),
            "null-vs-value is unverifiable, not a mismatch"
        );
        // Both set but EQUAL unknowns stay skipped; only real diffs fire.
        let mut saved = ident();
        saved.model_sha = Some("same".into());
        let mut current = ident();
        current.model_sha = Some("same".into());
        assert!(verify(&saved, &current).is_empty());
        current.model_sha = Some("other".into());
        assert!(verify(&saved, &current)
            .iter()
            .any(|d| d.contains("model_sha")));
    }

    #[test]
    fn integration__session_identity__build_reads_store_and_config() {
        let dir = tempfile::tempdir().expect("tempdir");
        let dirs = BlazarDirs {
            config_dir: dir.path().join("cfg"),
            data_dir: dir.path().join("data"),
        };
        let store = Store::open(&dirs).expect("store");
        store
            .upsert_engine(&crate::store::EngineRow {
                tag: "b4091".into(),
                asset: "llama-server".into(),
                sha256: "e3b0".into(),
                installed_at: 1,
                active: true,
                manifest: "{}".into(),
                kind: crate::engine_kind::EngineKind::default(),
            })
            .expect("engine");
        store
            .upsert_model(&crate::store::ModelRow {
                name: "m1".into(),
                repo: "r".into(),
                quant: "q4".into(),
                path: dir.path().join("m.gguf").to_string_lossy().into_owned(),
                bytes: 1,
                sha256: Some("cafe".into()),
                mmproj_path: None,
                vae_path: None,
                llm_path: None,
                llm_vision_path: None,
                shards: 1,
                arch: None,
                params: None,
                ctx_train: None,
                pulled_at: 1,
            })
            .expect("model");
        let cfg = Config {
            default_ctx: 4096,
            ..Config::default()
        };
        let id = build(&dirs, &cfg, "m1").expect("identity builds");
        assert_eq!(id.engine_tag, "b4091");
        assert_eq!(id.engine_sha, "e3b0");
        assert_eq!(id.model_sha.as_deref(), Some("cafe"));
        assert_eq!(id.ctx, 4096);
        assert_eq!(id.blazar_version, env!("CARGO_PKG_VERSION"));
    }
}
