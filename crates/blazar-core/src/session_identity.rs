//! KV-checkpoint identity manifests (#20): every session save writes a
//! sibling `<file>.identity.json` recording the exact runtime shape the
//! KV bytes were produced under (engine kind and build, model bytes,
//! ctx, cache type, KV unification, slots). Restores verify before
//! injecting - KV state from a different shape is silent garbage, not a
//! warm start.
//!
//! Two verification postures live here. [`verify`] compares only
//! fields present on both sides (a pre-hashing manifest is
//! "unverifiable", not "mismatched") - suitable for informational
//! diffs. [`verify_for_restore`] is the strict gate actual KV
//! restoration must pass: unknown identity REJECTS, because a cold
//! prefill costs latency while injected incompatible state costs
//! correctness.

use std::path::Path;
use std::path::PathBuf;

use serde::Deserialize;
use serde::Serialize;

use crate::config::Config;
use crate::dirs::BlazarDirs;
use crate::store::Store;

/// One attached `LoRA` adapter, as bound into a session identity.
/// `sha256` is the digest captured at `lora add` time; `None` marks a
/// legacy row (pre-v23 store) or an unhashable file - the strict
/// restore gate refuses to certify those.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LoraBinding {
    pub path: String,
    pub sha256: Option<String>,
}

/// The runtime shape under which a KV checkpoint was written.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionIdentity {
    pub blazar_version: String,
    /// Engine kind (e.g. `llama.cpp`, `sglang`). Empty on legacy
    /// manifests written before this dimension existed - the strict
    /// restore verifier rejects those as unverifiable (cold start),
    /// while the informational compare skips the field.
    #[serde(default)]
    pub engine_kind: String,
    pub engine_tag: String,
    pub engine_sha: String,
    #[serde(default)]
    pub model_sha: Option<String>,
    /// Adapter set attached to the model at save time, sorted by path.
    /// Empty on legacy manifests - the strict restore verifier then
    /// rejects any restore against a model that has adapters NOW
    /// (old-empty vs new-nonempty is a set diff, never a pass).
    #[serde(default)]
    pub adapters: Vec<LoraBinding>,
    pub ctx: u32,
    pub cache_type: String,
    #[serde(default)]
    pub kv_unified: Option<bool>,
    pub slots: u32,
    pub saved_at_unix: u64,
}

/// Adapter rows for `model`, sorted by path so the identity compare is
/// order-insensitive. Read failures degrade to an empty set: an
/// unverifiable adapter set then rejects at the strict gate rather
/// than here (bank paths warn and skip, they never panic).
fn adapter_bindings(store: &Store, model: &str) -> Vec<LoraBinding> {
    let mut bindings = store
        .list_loras(Some(model))
        .map(|rows| {
            rows.into_iter()
                .map(|r| LoraBinding {
                    path: r.path,
                    sha256: r.sha256,
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    bindings.sort_by(|a, b| a.path.cmp(&b.path));
    bindings
}

/// Sibling manifest path: `<session-file>.identity.json` (append - the
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
    let model_sha =
        crate::store::row_or_warn("models.get", store.get_model(model))?.and_then(|m| m.sha256);
    let adapters = adapter_bindings(&store, model);
    Some(SessionIdentity {
        blazar_version: env!("CARGO_PKG_VERSION").to_string(),
        engine_kind: engine.kind.as_str().to_string(),
        engine_tag: engine.tag,
        engine_sha: engine.sha256,
        model_sha,
        adapters,
        ctx: config.effective_ctx(model),
        // Symmetric grades record the legacy single value (old manifests
        // verify unchanged); a differential K/V pair composes "k/v" - the
        // string compare then refuses cross-quant restores in every
        // old-vs-new direction without a schema change.
        cache_type: {
            let (k, v) = config.effective_cache_type_kv(model);
            if k == v { k } else { format!("{k}/{v}") }
        },
        kv_unified: config.effective_kv_unified(model),
        // F121: record the overlay-effective slot count, not the raw
        // global - a per-model `slots` override changes what actually
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

/// Snapshot the runtime shape of one ACTUAL serving instance. Unlike
/// [`build`] (which snapshots the globally active engine and is kept
/// for tests/compat), this binds to the engine that will really serve
/// `model_name` under per-model routing, and takes the effective ctx
/// and slot count from the caller who knows the instance. `None` =
/// exact identity cannot be established (unknown model, unhashed
/// weights, engine row missing/kind-mismatched/unpinned sha) - callers
/// treat that as "do not bank / do not restore", never as "close
/// enough".
#[must_use]
pub fn build_for_instance(
    dirs: &BlazarDirs,
    config: &Config,
    model_name: &str,
    engine_tag: &str,
    engine_kind: crate::engine_kind::EngineKind,
    ctx: u32,
    slots: u32,
) -> Option<SessionIdentity> {
    let store = Store::open(dirs).ok()?;
    // An unhashed model cannot establish exact checkpoint identity.
    let model_sha = store.get_model(model_name).ok()??.sha256?;
    let engine = store
        .list_engines()
        .ok()?
        .into_iter()
        .find(|e| e.tag == engine_tag)?;
    if engine.kind != engine_kind || engine.sha256.is_empty() {
        return None;
    }
    let (cache_k, cache_v) = config.effective_cache_type_kv(model_name);
    let adapters = adapter_bindings(&store, model_name);
    Some(SessionIdentity {
        blazar_version: env!("CARGO_PKG_VERSION").to_string(),
        engine_kind: engine_kind.as_str().to_string(),
        engine_tag: engine.tag,
        engine_sha: engine.sha256,
        model_sha: Some(model_sha),
        adapters,
        ctx,
        cache_type: if cache_k == cache_v {
            cache_k
        } else {
            format!("{cache_k}/{cache_v}")
        },
        kv_unified: config.effective_kv_unified(model_name),
        slots,
        saved_at_unix: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs()),
    })
}

/// Strict restore gate: [`verify`] plus the rejection of any identity
/// dimension that is merely unknown. A checkpoint whose model digest or
/// engine kind cannot be established is refused - restoring it would
/// inject state of unknown provenance, and a cold prefill is the
/// correct fallback instead.
#[must_use]
pub fn verify_for_restore(saved: &SessionIdentity, current: &SessionIdentity) -> Vec<String> {
    let mut diffs = verify(saved, current);
    if saved.model_sha.is_none() || current.model_sha.is_none() {
        diffs.push("model_sha: exact identity unavailable".into());
    }
    if saved.engine_kind.is_empty() || current.engine_kind.is_empty() {
        diffs.push("engine_kind: exact identity unavailable".into());
    }
    // An adapter set with an unknown digest (legacy row / unhashable
    // file) on EITHER side cannot certify what the child actually
    // loaded - refuse rather than restore foreign state. Both-empty is
    // the plain no-adapter case and passes.
    let sha_unknown = |v: &[LoraBinding]| !v.is_empty() && v.iter().any(|b| b.sha256.is_none());
    if sha_unknown(&saved.adapters) || sha_unknown(&current.adapters) {
        diffs.push("adapters: exact identity unavailable".into());
    }
    diffs
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
/// absent (`None`) on either side are skipped - unverifiable is not a
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
    if let (Some(s), Some(c)) = (&saved.model_sha, &current.model_sha)
        && s != c
    {
        push("model_sha", s.clone(), c.clone());
    }
    // Adapter-set compare, order-insensitive on (path, sha): a swapped
    // file under an unchanged path shows as a digest diff, an
    // added/removed adapter as a set diff. Legacy empty-vs-nonempty is
    // a diff in every direction (fail-closed). Builders sort by path;
    // sorting again keeps hand-edited manifests honest too.
    let sorted_adapters = |v: &Vec<LoraBinding>| {
        let mut s = v.clone();
        s.sort_by(|a, b| a.path.cmp(&b.path));
        s
    };
    if sorted_adapters(&saved.adapters) != sorted_adapters(&current.adapters) {
        let render = |v: &[LoraBinding]| {
            v.iter()
                .map(|b| match &b.sha256 {
                    Some(sha) => format!("{}#{}", b.path, &sha[..sha.len().min(12)]),
                    None => format!("{}#?", b.path),
                })
                .collect::<Vec<_>>()
                .join(", ")
        };
        push(
            "adapters",
            render(&saved.adapters),
            render(&current.adapters),
        );
    }
    // Legacy manifests carry no engine kind: unknown-vs-unknown skips
    // here, but the strict restore verifier below rejects it - the
    // kind dimension (e.g. llama.cpp vs sglang) changes the KV byte
    // format entirely.
    if !saved.engine_kind.is_empty()
        && !current.engine_kind.is_empty()
        && saved.engine_kind != current.engine_kind
    {
        push(
            "engine_kind",
            saved.engine_kind.clone(),
            current.engine_kind.clone(),
        );
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
    if let (Some(s), Some(c)) = (saved.kv_unified, current.kv_unified)
        && s != c
    {
        push("kv_unified", s.to_string(), c.to_string());
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
            engine_kind: "llama.cpp".into(),
            engine_tag: "b4091".into(),
            engine_sha: "abc".into(),
            model_sha: Some("def".into()),
            adapters: Vec::new(),
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
        assert!(
            verify(&base, &t)
                .iter()
                .any(|d| d.contains("blazar_version"))
        );
        let mut t = ident();
        t.engine_sha = "zzz".into();
        assert!(verify(&base, &t).iter().any(|d| d.contains("engine_sha")));
        let mut t = ident();
        t.ctx = 16_384;
        assert!(
            verify(&base, &t)
                .iter()
                .any(|d| d.contains("ctx: saved 8192"))
        );
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
        assert_eq!(verify(&saved, &current).len(), 0);
        current.model_sha = Some("other".into());
        assert!(
            verify(&saved, &current)
                .iter()
                .any(|d| d.contains("model_sha"))
        );
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
                components: vec![],
                shards: 1,
                arch: None,
                params: None,
                ctx_train: None,
                pulled_at: 1,
                last_used_at: 1,

                license: None,
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

    #[test]
    fn integration__session_identity__differential_pair_composes_and_refuses_cross_quant() {
        // A differential K/V pair composes "k/v" in the manifest: old
        // manifests (single grade) verify unchanged, and the string
        // compare refuses cross-quant restores in BOTH directions
        // without a schema change.
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
        let cfg = Config {
            cache_type_k: "q8_0".into(),
            cache_type_v: "q4_0".into(),
            ..Config::default()
        };
        let id = build(&dirs, &cfg, "m1").expect("identity builds");
        assert_eq!(id.cache_type, "q8_0/q4_0");

        // Symmetric grade records the legacy single value.
        let cfg_sym = Config {
            cache_type: "q5_0".into(),
            ..Config::default()
        };
        let id_sym = build(&dirs, &cfg_sym, "m1").expect("identity builds");
        assert_eq!(id_sym.cache_type, "q5_0");

        // Cross-quant restore is refused in both old-vs-new directions.
        let mut saved = ident();
        saved.cache_type = "q8_0".into();
        let mut current = ident();
        current.cache_type = "q8_0/q4_0".into();
        let diffs = verify(&saved, &current);
        assert!(diffs.iter().any(|d| d.contains("cache_type")), "{diffs:?}");
        let diffs = verify(&current, &saved);
        assert!(diffs.iter().any(|d| d.contains("cache_type")), "{diffs:?}");
        // Same composed value on both sides restores cleanly.
        let mut twin = ident();
        twin.cache_type = "q8_0/q4_0".into();
        assert_eq!(verify(&current, &twin).len(), 0);
    }

    /// Fixture pair builder for the strict-gate tests.
    fn pair() -> (SessionIdentity, SessionIdentity) {
        (ident(), ident())
    }

    #[test]
    fn unit__verify_for_restore__rejects_cross_engine_kind() {
        let (saved, mut current) = pair();
        current.engine_kind = "sglang".into();
        let diffs = verify_for_restore(&saved, &current);
        assert!(diffs.iter().any(|d| d.contains("engine_kind")));
    }

    #[test]
    fn unit__verify_for_restore__rejects_engine_sha_and_model_sha_drift() {
        let (saved, mut current) = pair();
        current.engine_sha = "zzz".into();
        assert!(
            verify_for_restore(&saved, &current)
                .iter()
                .any(|d| d.contains("engine_sha"))
        );
        let (saved, mut current) = pair();
        current.model_sha = Some("other".into());
        assert!(
            verify_for_restore(&saved, &current)
                .iter()
                .any(|d| d.contains("model_sha: saved"))
        );
    }

    #[test]
    fn unit__verify_for_restore__rejects_unknown_identity_even_when_equal() {
        // Equal unknowns skip the informational compare - but the
        // strict restore gate rejects them: unverifiable is not
        // restoreable.
        let (mut saved, mut current) = pair();
        saved.model_sha = None;
        current.model_sha = None;
        assert!(verify(&saved, &current).is_empty());
        let diffs = verify_for_restore(&saved, &current);
        assert!(
            diffs
                .iter()
                .any(|d| d.contains("model_sha: exact identity unavailable"))
        );
        // Legacy manifests (no engine_kind) cold-start too.
        let (mut saved, mut current) = pair();
        saved.engine_kind = String::new();
        current.engine_kind = String::new();
        assert!(
            verify_for_restore(&saved, &current)
                .iter()
                .any(|d| d.contains("engine_kind: exact identity unavailable"))
        );
    }

    #[test]
    fn integration__build_for_instance__binds_named_engine_and_canonical_model() {
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
                components: vec![],
                shards: 1,
                arch: None,
                params: None,
                ctx_train: None,
                pulled_at: 1,
                last_used_at: 1,

                license: None,
            })
            .expect("model");
        let cfg = Config::default();
        let kind = crate::engine_kind::EngineKind::default();
        let id = build_for_instance(&dirs, &cfg, "m1", "b4091", kind, 4096, 7).expect("identity");
        assert_eq!(id.engine_kind, kind.as_str());
        assert_eq!(id.engine_tag, "b4091");
        assert_eq!(id.model_sha.as_deref(), Some("cafe"));
        assert_eq!(id.ctx, 4096);
        assert_eq!(id.slots, 7);
        // Wrong tag / wrong kind / suffixed (non-canonical) model /
        // unhashed model: exact identity is unavailable, never guessed.
        assert!(build_for_instance(&dirs, &cfg, "m1", "nope", kind, 8, 1).is_none());
        assert!(
            build_for_instance(
                &dirs,
                &cfg,
                "m1",
                "b4091",
                crate::engine_kind::EngineKind::Sglang,
                8,
                1
            )
            .is_none()
        );
        assert!(build_for_instance(&dirs, &cfg, "m1#1", "b4091", kind, 8, 1).is_none());
        store
            .upsert_model(&crate::store::ModelRow {
                name: "raw".into(),
                repo: "r".into(),
                quant: "q4".into(),
                path: dir.path().join("raw.gguf").to_string_lossy().into_owned(),
                bytes: 1,
                sha256: None,
                mmproj_path: None,
                components: vec![],
                shards: 1,
                arch: None,
                params: None,
                ctx_train: None,
                pulled_at: 1,
                last_used_at: 1,

                license: None,
            })
            .expect("raw model");
        assert!(build_for_instance(&dirs, &cfg, "raw", "b4091", kind, 8, 1).is_none());
    }

    fn binding(path: &str, sha: Option<&str>) -> LoraBinding {
        LoraBinding {
            path: path.into(),
            sha256: sha.map(String::from),
        }
    }

    #[test]
    fn unit__verify__adapter_set_order_insensitive_and_fail_closed_on_any_diff() {
        let base = ident();
        // Both-empty: the plain no-adapter case, no diff.
        let saved = base.clone();
        let current = base.clone();
        assert!(
            !verify(&saved, &current)
                .iter()
                .any(|d| d.starts_with("adapters"))
        );
        // Same set, different store order: builders sort, and the
        // compare re-sorts defensively - order is not identity.
        let mut saved = base.clone();
        saved.adapters = vec![
            binding("/a.gguf", Some("111")),
            binding("/b.gguf", Some("222")),
        ];
        let mut current = base.clone();
        current.adapters = vec![
            binding("/b.gguf", Some("222")),
            binding("/a.gguf", Some("111")),
        ];
        assert!(
            !verify(&saved, &current)
                .iter()
                .any(|d| d.starts_with("adapters"))
        );
        // Swapped file under an unchanged path: sha diff, named with
        // the path and both digest prefixes.
        let mut current = base.clone();
        current.adapters = vec![
            binding("/a.gguf", Some("333")),
            binding("/b.gguf", Some("222")),
        ];
        let diffs = verify(&saved, &current);
        let diff = diffs
            .iter()
            .find(|d| d.starts_with("adapters"))
            .expect("sha swap must diff");
        assert!(
            diff.contains("/a.gguf#111") && diff.contains("/a.gguf#333"),
            "{diff}"
        );
        // Added / removed adapter: set diff in every direction,
        // including legacy empty-vs-nonempty.
        let mut current = base.clone();
        current.adapters = vec![binding("/a.gguf", Some("111"))];
        assert!(
            verify(&saved, &current)
                .iter()
                .any(|d| d.starts_with("adapters"))
        );
        assert!(
            verify(&current, &saved)
                .iter()
                .any(|d| d.starts_with("adapters"))
        );
    }

    #[test]
    fn unit__verify_for_restore__rejects_unknown_adapter_sha_both_sides() {
        let base = ident();
        // Exact known set on both sides: restores.
        let mut saved = base.clone();
        saved.adapters = vec![binding("/a.gguf", Some("111"))];
        let mut current = saved.clone();
        current.saved_at_unix += 1;
        assert!(
            !verify_for_restore(&saved, &current)
                .iter()
                .any(|d| d.starts_with("adapters"))
        );
        // Unknown sha on the SAVED side (legacy row): unverifiable.
        let mut saved_legacy = base.clone();
        saved_legacy.adapters = vec![binding("/a.gguf", None)];
        let diffs = verify_for_restore(&saved_legacy, &current);
        assert!(
            diffs
                .iter()
                .any(|d| d.contains("adapters: exact identity unavailable")),
            "{diffs:?}"
        );
        // Unknown sha on the CURRENT side: equally unverifiable.
        let mut current_legacy = base.clone();
        current_legacy.adapters = vec![binding("/a.gguf", None)];
        let diffs = verify_for_restore(&saved, &current_legacy);
        assert!(
            diffs
                .iter()
                .any(|d| d.contains("adapters: exact identity unavailable")),
            "{diffs:?}"
        );
        // Both-empty stays the clean no-adapter restore.
        let diffs = verify_for_restore(&base, &base.clone());
        assert!(!diffs.iter().any(|d| d.starts_with("adapters")));
    }

    #[test]
    fn integration__build_for_instance__binds_sorted_adapter_set() {
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
                components: vec![],
                shards: 1,
                arch: None,
                params: None,
                ctx_train: None,
                pulled_at: 1,
                last_used_at: 1,

                license: None,
            })
            .expect("model");
        // Insert OUT of path order: the identity must come back sorted
        // so the restore compare is order-insensitive by construction.
        store
            .add_lora("m1", "/adapters/zwei.gguf", 1.0, Some("ccc"))
            .unwrap();
        store
            .add_lora("m1", "/adapters/eins.gguf", 0.5, Some("ddd"))
            .unwrap();
        store
            .add_lora("other", "/adapters/fremd.gguf", 1.0, None)
            .unwrap();
        let cfg = Config::default();
        let kind = crate::engine_kind::EngineKind::default();
        let id = build_for_instance(&dirs, &cfg, "m1", "b4091", kind, 4096, 7).expect("identity");
        assert_eq!(
            id.adapters,
            vec![
                binding("/adapters/eins.gguf", Some("ddd")),
                binding("/adapters/zwei.gguf", Some("ccc")),
            ],
            "adapters must bind the model's set, path-sorted, other models' rows excluded"
        );
    }
}
