//! Model store operations: remove with running-instance guard, listing,
//! and the boot-time models-dir reconcile.

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Result};

use blazar_core::store::Store;
use blazar_core::{BlazarDirs, ModelRow};

/// True when an instance marker exists for `name` (process may be loading
/// or serving; the supervisor owns the marker's lifecycle).
#[must_use]
pub fn instance_running(dirs: &BlazarDirs, name: &str) -> bool {
    // Replicas spawn per-key pidfiles (`model#N.pid`); any live pidfile —
    // plain or replica — counts as running.
    let run = dirs.run_dir();
    if run.join(format!("{name}.pid")).exists() {
        return true;
    }
    let Ok(entries) = std::fs::read_dir(&run) else {
        return false;
    };
    let prefix = format!("{name}#");
    entries.filter_map(std::result::Result::ok).any(|e| {
        let f = e.file_name().to_string_lossy().into_owned();
        f.starts_with(&prefix) && f.to_lowercase().ends_with(".pid")
    })
}

/// Delete a model: all shard files (by gguf-split naming convention), the
/// mmproj, and the store row. Refuses (409-style error) while an instance
/// of the model is running.
#[allow(clippy::case_sensitive_file_extension_comparisons)] // operand pre-lowercased
pub fn remove_model(dirs: &BlazarDirs, name: &str) -> Result<()> {
    if instance_running(dirs, name) {
        return Err(anyhow!(
            "model {name} is currently running; stop it first (`blazar stop {name}` or wait for eviction)"
        ));
    }
    let store = Store::open(dirs)?;
    let row = store
        .get_model(name)?
        .ok_or_else(|| anyhow!("no such model: {name}"))?;

    // Safetensors dir rows (sglang lane): the recorded path IS the model
    // directory — remove it wholesale. The shared-asset guard applies:
    // another row pointing at the same dir keeps it alive.
    let row_path = PathBuf::from(&row.path);
    if row_path.is_dir() {
        let shared = store
            .list_models()?
            .into_iter()
            .any(|m| m.name != row.name && m.path == row.path);
        if !shared {
            std::fs::remove_dir_all(&row_path)
                .map_err(|e| anyhow!("delete {}: {e}", row_path.display()))?;
        }
        store.delete_model(name)?;
        return Ok(());
    }

    let mut files: Vec<PathBuf> = vec![PathBuf::from(&row.path)];
    // Shards share the stored first-shard filename convention
    // (`base-NNNNN-of-MMMMM.gguf`); collect every part of the set.
    if let (Some(dir), Some(leaf)) = (
        PathBuf::from(&row.path).parent(),
        PathBuf::from(&row.path).file_name(),
    ) {
        let leaf = leaf.to_string_lossy().to_string();
        let base = crate::hf::parse_shard_marker_pub(&leaf).map_or_else(
            || leaf.trim_end_matches(".gguf").to_string(),
            |(_, _, base)| base,
        );
        if let Ok(entries) = std::fs::read_dir(dir) {
            for e in entries.flatten() {
                let fname = e.file_name().to_string_lossy().to_string();
                if fname.starts_with(&base) && fname.contains("-of-") && fname.ends_with(".gguf") {
                    files.push(e.path());
                }
            }
        }
    }
    if let Some(mm) = &row.mmproj_path {
        files.push(PathBuf::from(mm));
    }
    // Diffusion component set (sdcpp lane): VAE / text-encoder / vision
    // files ride the row exactly like the mmproj sidecar does.
    for component in [&row.vae_path, &row.llm_path, &row.llm_vision_path]
        .into_iter()
        .flatten()
    {
        files.push(PathBuf::from(component));
    }

    // Shared-asset guard: aliases reference the SAME mmproj (and are
    // hardlinks of the same weights). Deleting this row must never
    // destroy a file another row still points at — the incident class
    // that ate the real 875 MiB projector via `rm <alias>`, twice.
    let other_paths: Vec<String> = store
        .list_models()?
        .into_iter()
        .filter(|m| m.name != row.name)
        .flat_map(|m| {
            let mut v = vec![m.path];
            if let Some(mm) = m.mmproj_path {
                v.push(mm);
            }
            v.extend(
                [m.vae_path, m.llm_path, m.llm_vision_path]
                    .into_iter()
                    .flatten(),
            );
            v
        })
        .collect();
    files.retain(|f| {
        !other_paths
            .iter()
            .any(|p| std::path::Path::new(p.as_str()) == f.as_path())
    });

    for f in files {
        if f.exists() {
            std::fs::remove_file(&f).map_err(|e| anyhow!("delete {}: {e}", f.display()))?;
        }
    }
    store.delete_model(name)?;
    Ok(())
}

/// Alias a model under a new name (`blazar cp`): zero-byte hardlink of the
/// GGUF (both files live in the same models dir, so linking always works)
/// plus a new store row. No blob ceremony, no byte copies.
/// User-supplied model names become filename components (alias files,
/// pidfiles, import destinations). Refuse anything that could escape the
/// models dir or corrupt those paths (F98).
pub fn ensure_portable_name(name: &str) -> Result<()> {
    if name.is_empty() || name == "." || name == ".." {
        return Err(anyhow!("model name must not be empty, '.', or '..'"));
    }
    if name.len() > 128 {
        return Err(anyhow!("model name longer than 128 chars: {}", &name[..64]));
    }
    if name
        .chars()
        .any(|c| c == '/' || c == '\\' || c.is_control())
    {
        return Err(anyhow!(
            "model name must not contain '/', '\\\\', or control characters: {name}"
        ));
    }
    Ok(())
}

/// A model adopted from disk by the boot-time reconcile.
#[derive(Debug)]
pub struct AdoptedModel {
    pub name: String,
    /// "gguf" or "safetensors" — the lane the row serves on.
    pub format: &'static str,
    pub bytes: i64,
}

/// Outcome of a boot-time models-dir reconcile. Never an error state: a
/// store with nothing to adopt is the healthy common case.
#[derive(Debug)]
pub struct ReconcileReport {
    pub adopted: Vec<AdoptedModel>,
    /// Models whose projector sidecar was re-linked on a later boot
    /// (row adopted before the sidecar existed on disk, or before
    /// reconcile could match it).
    pub relinked: Vec<String>,
    /// (file/dir, reason) for candidates that looked like models but
    /// were refused — surfaced as boot warnings, never fatal.
    pub skipped: Vec<(String, String)>,
}

/// Adopt store-external model files living in the models dir: GGUFs (and
/// GGUF shard sets) whose path no row owns, plus safetensors dirs with a
/// readable `config.json`. Adds rows ONLY — never moves, renames, or
/// deletes files, so an uninstall that kept model files stays honest
/// across reinstalls. Row dialects mirror `blazar import` (GGUF) and the
/// pull lane (safetensors); `repo` carries an `adopted:<path>` marker so
/// adopted rows are identifiable in `blazar list`.
#[must_use]
// One cohesive adoption pass (dirs -> shard sets -> singles -> sidecar
// backfill); extracted helpers already carry the real logic.
#[allow(clippy::too_many_lines)]
pub fn reconcile_models(dirs: &BlazarDirs, store: &Store) -> ReconcileReport {
    let mut report = ReconcileReport {
        adopted: Vec::new(),
        relinked: Vec::new(),
        skipped: Vec::new(),
    };
    let models_dir = dirs.models_dir();
    let Ok(entries) = std::fs::read_dir(&models_dir) else {
        return report; // no dir yet = nothing to adopt
    };

    // Ownership = canonical paths of existing rows plus their inodes, so
    // hardlink twins of owned files (aliases) are not double-adopted.
    // Linked projectors count as owned too: a surviving row's mmproj must
    // never be re-attached to a different model by reconcile.
    let (owned, owned_inodes) = ownership_set(store);

    // Deterministic scan order: boot output must be stable across runs.
    let mut names: Vec<String> = entries
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();

    // GGUF shard sets (`base-00001-of-00002.gguf`) adopt as ONE model.
    // `mmproj*` GGUFs pulled beside a model are vision projector sidecars:
    // pull-convention names carry the repo prefix, which is the only
    // deterministic way to re-link one after a store rebuild.
    let mut sidecars: Vec<String> = Vec::new();
    let mut shard_sets: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut singles: Vec<String> = Vec::new();
    for name in &names {
        if !name.to_lowercase().ends_with(".gguf") {
            continue;
        }
        let path = models_dir.join(name);
        if owned.contains(&path.canonicalize().unwrap_or_else(|_| path.clone())) {
            continue;
        }
        if name.to_lowercase().contains("mmproj") {
            sidecars.push(name.clone());
            continue; // sidecars are never adopted as servable models
        }
        match crate::hf::parse_shard_marker_pub(name) {
            Some((_, _, base)) => shard_sets.entry(base).or_default().push(name.clone()),
            None => singles.push(name.clone()),
        }
    }
    let mut attached_sidecars: HashSet<String> = HashSet::new();
    #[cfg_attr(not(unix), allow(unused_mut))] // inode twins only tracked on unix
    let mut adopted_inodes: HashSet<(u64, u64)> = HashSet::new();

    // Safetensors dirs adopt BEFORE any GGUF: a dir's name minus `.d` is
    // the model's own canonical repo name, while a GGUF's embedded
    // general.name is a candidate ladder rung that can collide with it
    // (quant siblings all embed the base name). Dirs-first lets the dir
    // claim its name and the GGUF ladder fall through honestly.
    for name in &names {
        let dir = models_dir.join(name);
        if !dir.is_dir() {
            continue;
        }
        let looks_pulled = name.to_lowercase().ends_with(".d");
        if !dir.join("config.json").exists() {
            continue; // not an HF model dir — never ours to judge
        }
        if !root_safetensors(&dir).is_empty() || looks_pulled {
            adopt_dir(&models_dir, store, name, &owned, &mut report);
        }
    }

    for leaves in shard_sets.values() {
        let mut ctx = AdoptCtx {
            owned: &owned,
            owned_inodes: &owned_inodes,
            sidecars: &sidecars,
            attached_sidecars: &mut attached_sidecars,
            adopted_inodes: &mut adopted_inodes,
        };
        adopt_gguf(&models_dir, store, leaves, &mut ctx, &mut report);
    }
    for leaf in &singles {
        let mut ctx = AdoptCtx {
            owned: &owned,
            owned_inodes: &owned_inodes,
            sidecars: &sidecars,
            attached_sidecars: &mut attached_sidecars,
            adopted_inodes: &mut adopted_inodes,
        };
        adopt_gguf(
            &models_dir,
            store,
            std::slice::from_ref(leaf),
            &mut ctx,
            &mut report,
        );
    }

    // Backfill: rows adopted before their projector sidecar existed (or
    // before prefix matching landed) converge on the next boot. Only
    // reconcile-adopted rows are healed — pull/import rows carry their
    // own linkage contract.
    let current: Vec<ModelRow> = store.list_models().unwrap_or_default();
    // One working set across iterations: two adopted rows sharing a repo
    // prefix (quant siblings) must not both claim the single sidecar.
    let mut consumed: HashSet<String> = current
        .iter()
        .filter_map(|r| {
            r.mmproj_path
                .as_ref()
                .and_then(|p| Path::new(p).file_name())
                .map(|f| f.to_string_lossy().into_owned())
        })
        .chain(attached_sidecars.iter().cloned())
        .collect();
    for mut row in current {
        if row.mmproj_path.is_some() || !row.repo.starts_with("adopted:") {
            continue;
        }
        let Some(leaf) = Path::new(&row.path)
            .file_name()
            .map(|f| f.to_string_lossy().into_owned())
        else {
            continue;
        };
        if let Some(hit) = match_sidecar_for(&leaf, &sidecars, &mut consumed) {
            row.mmproj_path = Some(models_dir.join(&hit).display().to_string());
            if store.upsert_model(&row).is_ok() {
                report.relinked.push(row.name.clone());
            }
        }
    }
    report
}

/// Canonical paths + inodes of everything existing rows own (model files
/// and linked projectors). Hardlink twins of these are silently skipped
/// during adoption; a row's projector must never be re-attached to a
/// different model by reconcile.
fn ownership_set(store: &Store) -> (HashSet<PathBuf>, HashSet<(u64, u64)>) {
    let mut owned: HashSet<PathBuf> = HashSet::new();
    #[cfg_attr(not(unix), allow(unused_mut))] // inode twins only tracked on unix
    let mut owned_inodes: HashSet<(u64, u64)> = HashSet::new();
    for row in store.list_models().unwrap_or_default() {
        for p in [
            Some(PathBuf::from(&row.path)),
            row.mmproj_path.map(PathBuf::from),
        ]
        .into_iter()
        .flatten()
        {
            if let Ok(c) = p.canonicalize() {
                owned.insert(c);
            }
            #[cfg(unix)]
            if let Ok(md) = std::fs::metadata(&p) {
                use std::os::unix::fs::MetadataExt as _;
                owned_inodes.insert((md.dev(), md.ino()));
            }
        }
    }
    (owned, owned_inodes)
}

/// Root-level `.safetensors` weights of an HF model dir (the pull lane's
/// selection rule: shards live at the root, config.json is mandatory).
fn root_safetensors(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut files: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.is_file()
                && p.file_name()
                    .is_some_and(|f| f.to_string_lossy().to_lowercase().ends_with(".safetensors"))
        })
        .collect();
    files.sort();
    files
}

/// Shared adoption working state: what the store already owns (paths +
/// inodes), the sidecar pool, and the cross-adoption bookkeeping.
#[cfg_attr(not(unix), allow(dead_code))] // inode sets are only consulted on unix
struct AdoptCtx<'a> {
    owned: &'a HashSet<PathBuf>,
    owned_inodes: &'a HashSet<(u64, u64)>,
    sidecars: &'a [String],
    attached_sidecars: &'a mut HashSet<String>,
    adopted_inodes: &'a mut HashSet<(u64, u64)>,
}

/// Adopt one GGUF (or a whole shard set) at its EXISTING location.
/// Mirrors `blazar import`'s derivation exactly: quant from the filename
/// tail-token, name from GGUF metadata or the file stem.
fn adopt_gguf(
    models_dir: &Path,
    store: &Store,
    leaves: &[String],
    ctx: &mut AdoptCtx<'_>,
    report: &mut ReconcileReport,
) {
    let first = &leaves[0];
    let path = models_dir.join(first);
    if ctx
        .owned
        .contains(&path.canonicalize().unwrap_or_else(|_| path.clone()))
    {
        return; // a row already owns this file
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt as _;
        if let Ok(md) = std::fs::metadata(&path) {
            let ino = (md.dev(), md.ino());
            if ctx.owned_inodes.contains(&ino) || ctx.adopted_inodes.contains(&ino) {
                return; // hardlink twin of an owned/adopted file
            }
            ctx.adopted_inodes.insert(ino);
        }
    }
    let label = if leaves.len() > 1 {
        format!("{} (+{} shards)", first, leaves.len() - 1)
    } else {
        first.clone()
    };
    let meta = match blazar_core::read_metadata_file(&path) {
        Ok(m) => m,
        Err(e) => {
            // 0-KV GGUFs are diffusion/model-component files (image-repo
            // DiT/VAE/encoder splits). Adopting them as text models would
            // mint rows no engine can serve; the honest outcome is a skip
            // whose reason names the lane they actually belong to.
            let reason = if e.to_string().contains("missing general.architecture") {
                "diffusion/model-component GGUF (no architecture metadata) — pull it through a diffusion family instead: blazar pull <qwen-image-repo>:QUANT (sdcpp lane)".to_string()
            } else {
                format!("not a readable GGUF: {e}")
            };
            report.skipped.push((label, reason));
            return;
        }
    };
    let (quant, candidates) = gguf_derivations(first, &meta);
    // First free-and-portable candidate wins; quant siblings share
    // embedded metadata names (base + fp16 both say "Qwen2.5 0.5B
    // Instruct") and must not shadow each other.
    let Some(model_name) = candidates
        .iter()
        .find(|c| portable_model_name(c).is_ok() && store.get_model(c).ok().flatten().is_none())
        .cloned()
    else {
        report.skipped.push((
            label,
            format!(
                "every derived name is taken or unusable: {}",
                candidates.join(", ")
            ),
        ));
        return;
    };
    let bytes: u64 = leaves
        .iter()
        .filter_map(|l| std::fs::metadata(models_dir.join(l)).ok())
        .map(|m| m.len())
        .sum();
    // Projector re-link: a sidecar from the SAME pull repo (identical
    // `owner--repo--` prefix) belongs to this model with the same
    // certainty pull had when it stored both files. Bare sidecars carry
    // no repo signal and are never guessed onto a bare model.
    let mmproj_path = match_sidecar_for(first, ctx.sidecars, ctx.attached_sidecars)
        .map(|f| models_dir.join(f).display().to_string());
    let row = ModelRow {
        name: model_name.clone(),
        repo: format!("adopted:{}", path.display()),
        quant: quant.clone(),
        path: path.display().to_string(),
        bytes: i64::try_from(bytes).unwrap_or(i64::MAX),
        sha256: None,
        // Re-linked deterministically from the pull-convention prefix
        // when possible; bare projectors are never guessed.
        mmproj_path,
        // Reconcile adopts parsed GGUFs only; diffusion component sets
        // (0-metadata files) never reach here — they arrive via pull.
        vae_path: None,
        llm_path: None,
        llm_vision_path: None,
        shards: i64::try_from(leaves.len()).unwrap_or(i64::MAX),
        arch: Some(meta.architecture.clone()),
        params: Some(crate::hf::est_params(bytes, &quant)),
        ctx_train: meta.context_length.and_then(|c| i64::try_from(c).ok()),
        pulled_at: now_secs(),
    };
    match store.upsert_model(&row) {
        Ok(()) => report.adopted.push(AdoptedModel {
            name: model_name,
            format: "gguf",
            bytes: row.bytes,
        }),
        Err(e) => report
            .skipped
            .push((label, format!("store refused row: {e}"))),
    }
}

/// Adopt a safetensors model dir (`<name>.d`) at its existing location,
/// mirroring the pull lane's row dialect.
/// Pick the projector sidecar belonging to a pull-convention GGUF.
///
/// Pull stores model and projector under the same `owner--repo--` prefix;
/// that shared prefix is the only disk-surviving proof of the pairing, so
/// it is the only one reconcile trusts. Bare names (`mmproj-F16.gguf`)
/// carry no repo signal and stay unattached (import `--mmproj` is the
/// manual path for those).
fn match_sidecar_for(
    leaf: &str,
    sidecars: &[String],
    attached: &mut HashSet<String>,
) -> Option<String> {
    let parts: Vec<&str> = leaf.split("--").collect();
    if parts.len() < 3 {
        return None; // bare or non-pull name — no repo prefix to match
    }
    let prefix = format!("{}--{}--", parts[0], parts[1]);
    let hit = sidecars
        .iter()
        .find(|s| s.starts_with(&prefix) && !attached.contains(*s))?;
    attached.insert(hit.clone());
    Some(hit.clone())
}

fn adopt_dir(
    models_dir: &Path,
    store: &Store,
    dir_name: &str,
    owned: &HashSet<PathBuf>,
    report: &mut ReconcileReport,
) {
    let dir = models_dir.join(dir_name);
    let weights = root_safetensors(&dir);
    let model_name = dir_name.strip_suffix(".d").unwrap_or(dir_name);
    if weights.is_empty() {
        report.skipped.push((
            dir_name.to_string(),
            "has config.json but no root .safetensors weights".to_string(),
        ));
        return;
    }
    // A row already pointing at THIS dir is the idempotent second boot —
    // not a collision, and not worth a warning on every start.
    let canon = dir.canonicalize().unwrap_or_else(|_| dir.clone());
    if owned.contains(&canon) {
        return;
    }
    let meta = match blazar_core::hfmeta::read_hf_config(&dir) {
        Ok(m) => m,
        Err(e) => {
            report
                .skipped
                .push((dir_name.to_string(), format!("config.json unusable: {e}")));
            return;
        }
    };
    if let Err(reason) = portable_model_name(model_name) {
        report.skipped.push((dir_name.to_string(), reason));
        return;
    }
    if store.get_model(model_name).ok().flatten().is_some() {
        report.skipped.push((
            dir_name.to_string(),
            format!("name `{model_name}` already taken by another row"),
        ));
        return;
    }
    let bytes: u64 = weights
        .iter()
        .filter_map(|w| std::fs::metadata(w).ok())
        .map(|m| m.len())
        .sum();
    let quant = crate::hf::hf_quant_label(&meta);
    let row = ModelRow {
        name: model_name.to_string(),
        repo: format!("adopted:{}", dir.display()),
        quant: quant.clone(),
        path: dir.display().to_string(),
        bytes: i64::try_from(bytes).unwrap_or(i64::MAX),
        sha256: None,
        mmproj_path: None,
        vae_path: None,
        llm_path: None,
        llm_vision_path: None,
        shards: i64::try_from(weights.len()).unwrap_or(i64::MAX),
        arch: (!meta.architecture.is_empty()).then(|| meta.architecture.clone()),
        params: Some(crate::hf::est_params(bytes, &quant)),
        ctx_train: meta.ctx_train.and_then(|c| i64::try_from(c).ok()),
        pulled_at: now_secs(),
    };
    match store.upsert_model(&row) {
        Ok(()) => report.adopted.push(AdoptedModel {
            name: model_name.to_string(),
            format: "safetensors",
            bytes: row.bytes,
        }),
        Err(e) => report
            .skipped
            .push((dir_name.to_string(), format!("store refused row: {e}"))),
    }
}

/// GGUF quant-name family shape (`q4_k_m`, `iq4_xs`, `q8_0`, `f16`,
/// `fp8`, ...). Non-quant tail tokens (`0.5b`) must not be mistaken for
/// a quant suffix when stripping the name stem.
fn is_quant_token(token: &str) -> bool {
    let t = token.to_lowercase();
    let q_prefixed = |p: &str| {
        t.strip_prefix(p)
            .is_some_and(|rest| rest.starts_with(|c: char| c.is_ascii_digit()))
    };
    q_prefixed("q")
        || q_prefixed("iq")
        || matches!(
            t.as_str(),
            "f16" | "f32" | "bf16" | "fp4" | "fp6" | "fp8" | "fp16"
        )
}

/// Derive (quant label, candidate names) from a GGUF's on-disk filename
/// and metadata. Candidate names, most-faithful first: (1) pull
/// downloads name files `owner--repo--leaf.gguf` — the repo segment
/// reproduces the exact pre-uninstall name; (2) GGUF `general.name`;
/// (3) stem minus a quant-shaped tail; (4) full stem. All keep dots
/// (`qwen2.5-0.5b`) — these files were written by pull, whose registry
/// names carry them; only interactive import rewrites dots. The quant
/// tail-token comes from the base name, not the shard leaf
/// (`base-00001-of-00002` would otherwise derive quant `00002`).
fn gguf_derivations(first_leaf: &str, meta: &blazar_core::GgufMeta) -> (String, Vec<String>) {
    let quant_source = crate::hf::parse_shard_marker_pub(first_leaf)
        .map_or_else(
            || first_leaf.trim_end_matches(".gguf").to_string(),
            |(_, _, base)| base,
        )
        .to_lowercase();
    let quant = quant_source
        .rsplit('-')
        .next()
        .unwrap_or("adopted")
        .to_uppercase();
    let mut candidates: Vec<String> = Vec::new();
    let segments: Vec<&str> = quant_source.split("--").collect();
    if segments.len() >= 3 {
        let repo = segments[1].to_lowercase();
        let repo = repo.strip_suffix("-gguf").unwrap_or(&repo);
        candidates.push(repo.replace(' ', "-"));
    }
    if let Some(name) = meta
        .name
        .as_ref()
        .map(|n| n.to_lowercase().replace(' ', "-"))
    {
        candidates.push(name);
    }
    let tail = quant_source
        .rsplit('-')
        .next()
        .unwrap_or_default()
        .to_string();
    if is_quant_token(&tail) {
        if let Some(base) = quant_source.strip_suffix(&format!("-{tail}")) {
            candidates.push(base.to_string());
        }
    }
    candidates.push(quant_source.clone());
    (quant, candidates)
}

/// Name gate shared by both arms: portable-name rules plus the store's
/// reserved `#` (replica keys).
fn portable_model_name(name: &str) -> Result<(), String> {
    if name.contains('#') {
        return Err(format!("derived name `{name}` contains reserved '#'"));
    }
    ensure_portable_name(name).map_err(|e| e.to_string())
}

fn now_secs() -> i64 {
    i64::try_from(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
    )
    .unwrap_or(i64::MAX)
}

pub fn copy_model(dirs: &BlazarDirs, src: &str, dst: &str) -> Result<()> {
    ensure_portable_name(dst)?;
    if instance_running(dirs, src) {
        return Err(anyhow!(
            "model {src} is currently running; copy after it unloads"
        ));
    }
    let store = Store::open(dirs)?;
    let row = store
        .get_model(src)?
        .ok_or_else(|| anyhow!("no such model: {src}"))?;
    if store.get_model(dst)?.is_some() {
        return Err(anyhow!("model {dst} already exists"));
    }
    let src_path = PathBuf::from(&row.path);
    // Dir rows (safetensors/sglang lane) cannot be hardlink-aliased;
    // teach the pull-again path instead of failing deep in link(2).
    if src_path.is_dir() {
        return Err(anyhow!(
            "model {src} is a safetensors directory (sglang lane) — directories cannot be aliased; pull the repo again under the new name"
        ));
    }
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
        return Err(anyhow!(
            "alias path collides with the source file; refusing"
        ));
    }
    if !dst_path.exists() {
        std::fs::hard_link(&src_path, &dst_path).map_err(|e| {
            anyhow!(
                "hardlink {} -> {}: {e}",
                dst_path.display(),
                src_path.display()
            )
        })?;
    }
    store.upsert_model(&blazar_core::ModelRow {
        name: dst.to_string(),
        repo: row.repo.clone(),
        quant: row.quant.clone(),
        path: dst_path.display().to_string(),
        bytes: row.bytes,
        sha256: row.sha256.clone(),
        mmproj_path: row.mmproj_path.clone(),
        // Component-set assets are shared references (read-only weights,
        // hardlink-friendly), not per-row copies: the duplicate points at
        // the same VAE/TE files.
        vae_path: row.vae_path.clone(),
        llm_path: row.llm_path.clone(),
        llm_vision_path: row.llm_vision_path.clone(),
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

    fn setup() -> (tempfile::TempDir, BlazarDirs) {
        let tmp = tempfile::tempdir().unwrap();
        let dirs = BlazarDirs {
            config_dir: tmp.path().join("cfg"),
            data_dir: tmp.path().join("data"),
        };
        dirs.ensure().unwrap();
        (tmp, dirs)
    }

    #[test]
    fn unit__instance_running__detects_replica_pidfiles() {
        let (_t, dirs) = setup();
        let run = dirs.run_dir();
        assert!(!instance_running(&dirs, "m"));
        std::fs::write(run.join("other.pid"), b"1").unwrap();
        assert!(!instance_running(&dirs, "m"));
        std::fs::write(run.join("m#2.pid"), b"99").unwrap();
        assert!(instance_running(&dirs, "m"));
        std::fs::remove_file(run.join("m#2.pid")).unwrap();
        std::fs::write(run.join("m.pid"), b"1").unwrap();
        assert!(instance_running(&dirs, "m"));
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
            .upsert_model(&blazar_core::ModelRow {
                name: "m".into(),
                repo: "o/m".into(),
                quant: "Q4_K_M".into(),
                path: d.join("m-q4_k_m-00001-of-00002.gguf").display().to_string(),
                bytes: 2,
                sha256: None,
                mmproj_path: Some(d.join("mmproj-m.gguf").display().to_string()),
                vae_path: None,
                llm_path: None,
                llm_vision_path: None,
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
    fn unit__remove_model__component_set_deleted_unless_shared() {
        let (_t, dirs) = setup();
        let d = dirs.models_dir();
        std::fs::write(d.join("dit.gguf"), b"dit").unwrap();
        std::fs::write(d.join("alias-dit.gguf"), b"dit2").unwrap();
        std::fs::write(d.join("vae.safetensors"), b"vae").unwrap();
        std::fs::write(d.join("te.gguf"), b"te").unwrap();
        std::fs::write(d.join("vis.gguf"), b"vis").unwrap();
        let store = Store::open(&dirs).unwrap();
        let row = |name: &str, path: &str, vae: Option<String>| blazar_core::ModelRow {
            name: name.into(),
            repo: "o/qwen-image".into(),
            quant: "Q4_K_M".into(),
            path: path.into(),
            bytes: 3,
            sha256: None,
            mmproj_path: None,
            vae_path: vae,
            llm_path: Some(d.join("te.gguf").display().to_string()),
            llm_vision_path: Some(d.join("vis.gguf").display().to_string()),
            shards: 1,
            arch: None,
            params: None,
            ctx_train: None,
            pulled_at: 1,
        };
        store
            .upsert_model(&row(
                "m",
                &d.join("dit.gguf").display().to_string(),
                Some(d.join("vae.safetensors").display().to_string()),
            ))
            .unwrap();
        // The alias shares ONLY the VAE; its own TE/vision slots are empty.
        store
            .upsert_model(&blazar_core::ModelRow {
                name: "alias".into(),
                repo: "o/qwen-image".into(),
                quant: "Q4_K_M".into(),
                path: d.join("alias-dit.gguf").display().to_string(),
                bytes: 4,
                sha256: None,
                mmproj_path: None,
                vae_path: Some(d.join("vae.safetensors").display().to_string()),
                llm_path: None,
                llm_vision_path: None,
                shards: 1,
                arch: None,
                params: None,
                ctx_train: None,
                pulled_at: 1,
            })
            .unwrap();

        remove_model(&dirs, "m").unwrap();
        assert!(!d.join("dit.gguf").exists(), "DiT goes with its row");
        assert!(!d.join("te.gguf").exists(), "unshared TE goes");
        assert!(!d.join("vis.gguf").exists(), "unshared vision goes");
        assert!(
            d.join("vae.safetensors").exists(),
            "VAE still referenced by `alias` must survive"
        );
        remove_model(&dirs, "alias").unwrap();
        assert!(
            !d.join("vae.safetensors").exists(),
            "last ref owns the delete"
        );
    }

    #[test]
    fn unit__remove_model__deletes_safetensors_dir_and_row() {
        let (_t, dirs) = setup();
        let d = dirs.models_dir().join("m.d");
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("config.json"), b"{}").unwrap();
        std::fs::write(d.join("model.safetensors"), b"weights").unwrap();
        let store = Store::open(&dirs).unwrap();
        store
            .upsert_model(&blazar_core::ModelRow {
                name: "m".into(),
                repo: "o/m".into(),
                quant: "BF16".into(),
                path: d.display().to_string(),
                bytes: 9,
                sha256: None,
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
            .unwrap();

        remove_model(&dirs, "m").unwrap();
        assert!(store.get_model("m").unwrap().is_none());
        assert!(!d.exists(), "dir row removal deletes the whole dir");
    }

    #[test]
    fn unit__remove_model__shared_dir_survives_other_rows() {
        let (_t, dirs) = setup();
        let d = dirs.models_dir().join("m.d");
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("config.json"), b"{}").unwrap();
        let store = Store::open(&dirs).unwrap();
        for name in ["m", "alias"] {
            store
                .upsert_model(&blazar_core::ModelRow {
                    name: name.into(),
                    repo: "o/m".into(),
                    quant: "BF16".into(),
                    path: d.display().to_string(),
                    bytes: 2,
                    sha256: None,
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
                .unwrap();
        }
        remove_model(&dirs, "m").unwrap();
        assert!(d.exists(), "dir still referenced by `alias` must survive");
        assert!(store.get_model("alias").unwrap().is_some());
    }

    #[test]
    fn unit__copy_model__dir_row_refused_with_teaching() {
        let (_t, dirs) = setup();
        let d = dirs.models_dir().join("m.d");
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("config.json"), b"{}").unwrap();
        let s = Store::open(&dirs).unwrap();
        s.upsert_model(&blazar_core::ModelRow {
            name: "m".into(),
            repo: "r".into(),
            quant: "BF16".into(),
            path: d.display().to_string(),
            bytes: 2,
            sha256: None,
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
        .unwrap();
        let err = copy_model(&dirs, "m", "m-alias").unwrap_err();
        assert!(err.to_string().contains("cannot be aliased"), "{err}");
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
        let s = blazar_core::Store::open(&dirs).unwrap();
        s.upsert_model(&blazar_core::ModelRow {
            name: "m".into(),
            repo: "r".into(),
            quant: "Q4_K_M".into(),
            path: gguf.display().to_string(),
            bytes: 10,
            sha256: None,
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
        assert!(
            copy_model(&dirs, "m", "m-alias").is_err(),
            "duplicate alias refused"
        );
    }

    // ---- boot-time reconcile (reconcile_models) ----

    /// Minimal valid GGUF v3 with a `general.architecture` string kv and
    /// an optional `general.name` string kv — the same byte dialect as
    /// the gguf health tests in hf.rs.
    fn write_gguf(path: &std::path::Path, arch: &str, name: Option<&str>) {
        fn str_kv(b: &mut Vec<u8>, k: &str, v: &str) {
            b.extend_from_slice(&(k.len() as u64).to_le_bytes());
            b.extend_from_slice(k.as_bytes());
            b.extend_from_slice(&8u32.to_le_bytes()); // GGUF type: string
            b.extend_from_slice(&(v.len() as u64).to_le_bytes());
            b.extend_from_slice(v.as_bytes());
        }
        let mut b: Vec<u8> = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes()); // version
        b.extend_from_slice(&0u64.to_le_bytes()); // tensor count
        let kv = 1 + usize::from(name.is_some());
        b.extend_from_slice(&(kv as u64).to_le_bytes());
        str_kv(&mut b, "general.architecture", arch);
        if let Some(n) = name {
            str_kv(&mut b, "general.name", n);
        }
        std::fs::write(path, b).unwrap();
    }

    fn row(path: &std::path::Path, name: &str) -> blazar_core::ModelRow {
        blazar_core::ModelRow {
            name: name.into(),
            repo: "o/m".into(),
            quant: "Q4_K_M".into(),
            path: path.display().to_string(),
            bytes: 1,
            sha256: None,
            mmproj_path: None,
            vae_path: None,
            llm_path: None,
            llm_vision_path: None,
            shards: 1,
            arch: None,
            params: None,
            ctx_train: None,
            pulled_at: 1,
        }
    }

    #[test]
    fn unit__reconcile__adopts_orphan_gguf_with_import_dialect() {
        let (_t, dirs) = setup();
        let f = dirs.models_dir().join("keepme-0.5b-q4_k_m.gguf");
        write_gguf(&f, "qwen3", Some("KeepMe 0.5B"));
        let store = Store::open(&dirs).unwrap();

        let r = reconcile_models(&dirs, &store);
        assert_eq!(r.adopted.len(), 1, "{:?}", r.skipped);
        assert_eq!(r.adopted[0].name, "keepme-0.5b"); // name kv, dots kept (pull dialect)
        assert_eq!(r.adopted[0].format, "gguf");
        let m = store.get_model("keepme-0.5b").unwrap().unwrap();
        assert_eq!(m.quant, "Q4_K_M", "quant from filename tail token");
        assert_eq!(m.arch.as_deref(), Some("qwen3"));
        assert_eq!(m.shards, 1);
        assert!(m.repo.starts_with("adopted:"), "{}", m.repo);
        assert_eq!(
            m.path,
            f.display().to_string(),
            "adopted at its EXISTING location"
        );
        assert!(m.mmproj_path.is_none(), "no mmproj guessing");

        // Second boot: no churn — the file is owned now.
        let r2 = reconcile_models(&dirs, &store);
        assert!(
            r2.adopted.is_empty() && r2.skipped.is_empty(),
            "{:?} {:?}",
            r2.adopted,
            r2.skipped
        );
    }

    #[test]
    fn unit__reconcile__shard_set_adopts_as_one_model() {
        let (_t, dirs) = setup();
        let d = dirs.models_dir();
        write_gguf(
            &d.join("big-00001-of-00002.gguf"),
            "llama",
            Some("Big Model"),
        );
        write_gguf(&d.join("big-00002-of-00002.gguf"), "llama", None);
        let store = Store::open(&dirs).unwrap();

        let r = reconcile_models(&dirs, &store);
        assert_eq!(r.adopted.len(), 1, "{:?}", r.skipped);
        let m = store.get_model("big-model").unwrap().unwrap();
        assert_eq!(m.shards, 2, "shard set adopts as one row");
        assert_eq!(
            m.path,
            d.join("big-00001-of-00002.gguf").display().to_string()
        );
    }

    #[test]
    fn unit__reconcile__adopts_safetensors_dir_with_pull_dialect() {
        let (_t, dirs) = setup();
        let d = dirs.models_dir().join("qwen-instruct.d");
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(
            d.join("config.json"),
            br#"{"architectures":["Qwen2ForCausalLM"],"torch_dtype":"bfloat16","max_position_embeddings":32768}"#,
        )
        .unwrap();
        std::fs::write(d.join("model-00001-of-00002.safetensors"), b"aaaa").unwrap();
        std::fs::write(d.join("model-00002-of-00002.safetensors"), b"bb").unwrap();
        let store = Store::open(&dirs).unwrap();

        let r = reconcile_models(&dirs, &store);
        assert_eq!(r.adopted.len(), 1, "{:?}", r.skipped);
        assert_eq!(r.adopted[0].name, "qwen-instruct"); // .d suffix stripped
        assert_eq!(r.adopted[0].format, "safetensors");
        let m = store.get_model("qwen-instruct").unwrap().unwrap();
        assert_eq!(m.quant, "BF16");
        assert_eq!(m.bytes, 6, "bytes = sum of root safetensors");
        assert_eq!(m.shards, 2);
        assert_eq!(m.arch.as_deref(), Some("Qwen2ForCausalLM"));
        assert_eq!(m.ctx_train, Some(32768));
        assert_eq!(m.path, d.display().to_string());

        // Second boot: fully silent — owned dirs are not collisions and
        // must not warn on every start.
        let r2 = reconcile_models(&dirs, &store);
        assert!(r2.adopted.is_empty(), "{:?}", r2.adopted);
        assert!(r2.skipped.is_empty(), "{:?}", r2.skipped);
    }

    #[test]
    fn unit__reconcile__dir_wins_canonical_name_over_gguf_meta_ladder() {
        // Real-world shape from the 09-17 reinstall: the fp16 GGUF's
        // embedded general.name ("Qwen2.5 0.5B Instruct") collides with
        // the safetensors dir's own repo name. Dirs adopt first so the
        // dir claims qwen2.5-0.5b-instruct and the GGUF ladder falls
        // through to its full stem — no shadowing, no warn, both rows.
        let (_t, dirs) = setup();
        let d = dirs.models_dir();
        let sd = d.join("qwen2.5-0.5b-instruct.d");
        std::fs::create_dir_all(&sd).unwrap();
        std::fs::write(
            sd.join("config.json"),
            br#"{"architectures":["Qwen2ForCausalLM"],"torch_dtype":"bfloat16"}"#,
        )
        .unwrap();
        std::fs::write(sd.join("model.safetensors"), b"aaaa").unwrap();
        write_gguf(
            &d.join("qwen2.5-0.5b-instruct-fp16.gguf"),
            "qwen3",
            Some("Qwen2.5 0.5B Instruct"),
        );
        let store = Store::open(&dirs).unwrap();

        let r = reconcile_models(&dirs, &store);
        assert_eq!(r.adopted.len(), 2, "{:?}", r.skipped);
        assert!(r.skipped.is_empty(), "{:?}", r.skipped);
        let dir_row = store.get_model("qwen2.5-0.5b-instruct").unwrap().unwrap();
        assert_eq!(dir_row.path, sd.display().to_string());
        let gguf_row = store
            .get_model("qwen2.5-0.5b-instruct-fp16")
            .unwrap()
            .unwrap();
        assert!(gguf_row.path.to_ascii_lowercase().ends_with(".gguf"));
    }

    #[test]
    // Twin detection is inode-based and unix-only by design (std exposes
    // no file-index surface on Windows); there a link twin is adoptable.
    #[cfg(unix)]
    fn unit__reconcile__owned_and_twins_left_alone() {
        let (_t, dirs) = setup();
        let d = dirs.models_dir();
        let owned_file = d.join("m-q4_k_m.gguf");
        write_gguf(&owned_file, "qwen3", Some("m"));
        let store = Store::open(&dirs).unwrap();
        store.upsert_model(&row(&owned_file, "m")).unwrap();
        // Hardlink twin of the owned file under a different name.
        std::fs::hard_link(&owned_file, d.join("twin-copy.gguf")).unwrap();
        // Random non-model file stays silent.
        std::fs::write(d.join("mmproj-15b.gguf"), b"sidecar").unwrap();
        std::fs::write(d.join("model.part"), b"partial").unwrap();

        let r = reconcile_models(&dirs, &store);
        assert!(r.adopted.is_empty(), "{:?}", r.adopted);
        // mmproj.part/model.part: silent skips (non-gguf / non-model files
        // are not ours to judge); twin is silent too.
        assert!(
            r.skipped
                .iter()
                .all(|(f, _)| !f.contains("twin") && !f.contains(".part")),
            "{:?}",
            r.skipped
        );
    }

    #[test]
    fn unit__reconcile__clip_sidecar_and_corrupt_skipped() {
        let (_t, dirs) = setup();
        let d = dirs.models_dir();
        write_gguf(&d.join("m-mmproj.gguf"), "clip", None); // projector, not a model
        std::fs::write(d.join("broken-q4.gguf"), b"garbage not gguf").unwrap();
        let store = Store::open(&dirs).unwrap();

        let r = reconcile_models(&dirs, &store);
        assert!(r.adopted.is_empty(), "{:?}", r.adopted);
        assert!(
            r.skipped
                .iter()
                .any(|(f, why)| f == "broken-q4.gguf" && why.contains("not a readable GGUF")),
            "{:?}",
            r.skipped
        );
        assert!(
            r.skipped.iter().all(|(f, _)| !f.contains("mmproj")),
            "clip sidecars are silent (not model candidates): {:?}",
            r.skipped
        );
    }

    #[test]
    fn unit__reconcile__kvless_component_gguf_skips_with_lane_teaching() {
        let (_t, dirs) = setup();
        let d = dirs.models_dir();
        // 0-metadata-KV GGUF: the diffusion DiT shape (0 KVs, N tensors).
        let mut gguf = b"GGUF".to_vec();
        gguf.extend_from_slice(&3u32.to_le_bytes());
        gguf.extend_from_slice(&297u64.to_le_bytes());
        gguf.extend_from_slice(&0u64.to_le_bytes());
        std::fs::write(d.join("dit-q4_k_m.gguf"), &gguf).unwrap();
        let store = Store::open(&dirs).unwrap();

        let r = reconcile_models(&dirs, &store);
        assert!(
            r.adopted.is_empty(),
            "component GGUF must not adopt: {:?}",
            r.adopted
        );
        assert!(
            r.skipped.iter().any(|(f, why)| f == "dit-q4_k_m.gguf"
                && why.contains("diffusion/model-component")
                && why.contains("sdcpp")),
            "{:?}",
            r.skipped
        );
    }

    #[test]
    fn unit__reconcile__taken_name_falls_through_candidates_never_shadows() {
        let (_t, dirs) = setup();
        let d = dirs.models_dir();
        // Row `m` exists pointing at ANOTHER path; the orphan's GGUF
        // metadata derives the same name. The ladder must NOT shadow the
        // existing row — it falls through to the stem-derived name, and
        // only skips when EVERY candidate is taken.
        let elsewhere = dirs.data_dir.join("elsewhere.gguf");
        write_gguf(&elsewhere, "qwen3", None);
        let store = Store::open(&dirs).unwrap();
        store.upsert_model(&row(&elsewhere, "m")).unwrap();
        write_gguf(&d.join("orphan-q8_0.gguf"), "llama", Some("m"));

        let r = reconcile_models(&dirs, &store);
        assert_eq!(r.adopted.len(), 1, "{:?}", r.skipped);
        assert_eq!(r.adopted[0].name, "orphan", "stem minus quant tail");
        // The pre-existing row is untouched.
        assert_eq!(
            store.get_model("m").unwrap().unwrap().path,
            elsewhere.display().to_string()
        );
    }

    #[test]
    fn unit__reconcile__total_name_saturation_skips_honestly() {
        let (_t, dirs) = setup();
        let d = dirs.models_dir();
        // Row `dup` lives elsewhere; the orphan's stem AND metadata both
        // derive exactly `dup` — every candidate is taken, so it must
        // skip with an honest reason instead of shadowing.
        let elsewhere = dirs.data_dir.join("elsewhere.gguf");
        write_gguf(&elsewhere, "qwen3", None);
        let store = Store::open(&dirs).unwrap();
        store.upsert_model(&row(&elsewhere, "dup")).unwrap();
        write_gguf(&d.join("dup.gguf"), "llama", Some("dup"));

        let r = reconcile_models(&dirs, &store);
        assert!(r.adopted.is_empty(), "{:?}", r.adopted);
        assert!(
            r.skipped
                .iter()
                .any(|(f, why)| f == "dup.gguf" && why.contains("every derived name is taken")),
            "{:?}",
            r.skipped
        );
        // File survives untouched — adoption never deletes.
        assert!(d.join("dup.gguf").exists());
    }

    #[test]
    fn unit__reconcile__clean_store_is_noop() {
        let (_t, dirs) = setup();
        std::fs::write(dirs.models_dir().join("readme.txt"), b"hi").unwrap();
        let store = Store::open(&dirs).unwrap();
        let r = reconcile_models(&dirs, &store);
        assert!(
            r.adopted.is_empty() && r.skipped.is_empty(),
            "{:?} {:?}",
            r.adopted,
            r.skipped
        );
    }

    #[test]
    fn unit__reconcile__pull_convention_recovers_repo_name() {
        let (_t, dirs) = setup();
        let d = dirs.models_dir();
        // Pull-lane on-disk naming: owner--repo--leaf.gguf. The repo
        // segment (minus -gguf suffix) is the pre-uninstall model name —
        // it must win over GGUF-internal general.name.
        write_gguf(
            &d.join("unsloth--Qwen3.5-9B-MTP-GGUF--Qwen3.5-9B-Q4_K_M.gguf"),
            "qwen3",
            Some("Qwen3.5 9B"),
        );
        let store = Store::open(&dirs).unwrap();
        let r = reconcile_models(&dirs, &store);
        assert_eq!(r.adopted.len(), 1, "{:?}", r.skipped);
        assert_eq!(
            r.adopted[0].name, "qwen3.5-9b-mtp",
            "repo segment, -gguf stripped"
        );
        let m = store.get_model("qwen3.5-9b-mtp").unwrap().unwrap();
        assert_eq!(m.quant, "Q4_K_M", "quant from leaf tail token");
    }

    #[test]
    fn unit__reconcile__pull_convention_sidecar_relinked() {
        let (_t, dirs) = setup();
        let d = dirs.models_dir();
        let weights = "unsloth--Qwen3.5-9B-MTP-GGUF--Qwen3.5-9B-Q4_K_M.gguf";
        let projector = "unsloth--Qwen3.5-9B-MTP-GGUF--mmproj-F16.gguf";
        write_gguf(&d.join(weights), "qwen3", Some("Qwen3.5 9B"));
        write_gguf(&d.join(projector), "clip", None);
        let store = Store::open(&dirs).unwrap();

        let r = reconcile_models(&dirs, &store);
        assert_eq!(r.adopted.len(), 1, "{:?}", r.skipped);
        assert!(
            r.skipped.is_empty(),
            "sidecar must not warn: {:?}",
            r.skipped
        );
        let m = store.get_model("qwen3.5-9b-mtp").unwrap().unwrap();
        assert_eq!(
            m.mmproj_path.as_deref(),
            Some(d.join(projector).display().to_string().as_str()),
            "same-repo sidecar re-linked by pull prefix"
        );
        // The projector itself is never adopted as a servable model.
        assert!(store.get_model("mmproj-f16").unwrap().is_none());
        // Second boot: no churn, linkage stable.
        let r2 = reconcile_models(&dirs, &store);
        assert!(r2.adopted.is_empty() && r2.skipped.is_empty(), "{r2:?}");
    }

    #[test]
    fn unit__reconcile__owned_projector_never_reassigned() {
        let (_t, dirs) = setup();
        let d = dirs.models_dir();
        let projector = "unsloth--Qwen3.5-9B-MTP-GGUF--mmproj-F16.gguf";
        write_gguf(
            &d.join("unsloth--Qwen3.5-9B-MTP-GGUF--Qwen3.5-9B-Q4_K_M.gguf"),
            "qwen3",
            Some("Qwen3.5 9B"),
        );
        write_gguf(&d.join(projector), "clip", None);
        // A surviving row (different model, elsewhere) already owns the
        // projector: reconcile must not attach it to the adopted model.
        let other = d.join("other-q8_0.gguf");
        write_gguf(&other, "llama", Some("other"));
        let store = Store::open(&dirs).unwrap();
        let mut orow = row(&other, "other");
        orow.mmproj_path = Some(d.join(projector).display().to_string());
        store.upsert_model(&orow).unwrap();

        let r = reconcile_models(&dirs, &store);
        assert_eq!(r.adopted.len(), 1, "{:?}", r.skipped);
        let m = store.get_model("qwen3.5-9b-mtp").unwrap().unwrap();
        assert!(
            m.mmproj_path.is_none(),
            "projector owned by another row must not be re-attached"
        );
    }

    #[test]
    fn unit__reconcile__bare_sidecar_never_guessed() {
        let (_t, dirs) = setup();
        let d = dirs.models_dir();
        write_gguf(
            &d.join("Qwen3.5-9B-Q4_K_M.gguf"),
            "qwen3",
            Some("Qwen3.5 9B"),
        );
        write_gguf(&d.join("mmproj-F16.gguf"), "clip", None);
        let store = Store::open(&dirs).unwrap();

        let r = reconcile_models(&dirs, &store);
        assert_eq!(r.adopted.len(), 1, "{:?}", r.skipped);
        let m = store.get_model("qwen3.5-9b").unwrap().unwrap();
        assert!(
            m.mmproj_path.is_none(),
            "bare sidecar has no repo signal — never guessed"
        );
    }

    #[test]
    fn unit__reconcile__backfill_relinks_older_adopted_row() {
        let (_t, dirs) = setup();
        let d = dirs.models_dir();
        let weights = "unsloth--Qwen3.5-9B-MTP-GGUF--Qwen3.5-9B-Q4_K_M.gguf";
        let projector = "unsloth--Qwen3.5-9B-MTP-GGUF--mmproj-F16.gguf";
        write_gguf(&d.join(weights), "qwen3", Some("Qwen3.5 9B"));
        write_gguf(&d.join(projector), "clip", None);
        // Simulate the two-boot reality: the row was adopted by an older
        // build (no sidecar matching), projector link missing.
        let store = Store::open(&dirs).unwrap();
        let mut old = row(&d.join(weights), "qwen3.5-9b-mtp");
        old.repo = format!("adopted:{}", d.join(weights).display());
        store.upsert_model(&old).unwrap();

        let r = reconcile_models(&dirs, &store);
        assert!(r.adopted.is_empty(), "{:?}", r.adopted);
        assert_eq!(r.relinked, ["qwen3.5-9b-mtp"]);
        let m = store.get_model("qwen3.5-9b-mtp").unwrap().unwrap();
        assert_eq!(
            m.mmproj_path.as_deref(),
            Some(d.join(projector).display().to_string().as_str())
        );
        // Third boot: fully stable.
        let r3 = reconcile_models(&dirs, &store);
        assert!(r3.adopted.is_empty() && r3.relinked.is_empty(), "{r3:?}");
    }

    #[test]
    fn unit__reconcile__backfill_never_touches_pull_rows() {
        let (_t, dirs) = setup();
        let d = dirs.models_dir();
        let weights = "unsloth--Qwen3.5-9B-MTP-GGUF--Qwen3.5-9B-Q4_K_M.gguf";
        let projector = "unsloth--Qwen3.5-9B-MTP-GGUF--mmproj-F16.gguf";
        write_gguf(&d.join(weights), "qwen3", Some("Qwen3.5 9B"));
        write_gguf(&d.join(projector), "clip", None);
        // Pull/import rows own their linkage contract: reconcile must not
        // edit them even when a matching sidecar sits unused on disk.
        let store = Store::open(&dirs).unwrap();
        store
            .upsert_model(&row(&d.join(weights), "qwen3.5-9b-mtp"))
            .unwrap();

        let r = reconcile_models(&dirs, &store);
        assert!(r.adopted.is_empty() && r.relinked.is_empty(), "{r:?}");
        let m = store.get_model("qwen3.5-9b-mtp").unwrap().unwrap();
        assert!(m.mmproj_path.is_none());
    }

    #[test]
    fn unit__reconcile__gguf_meta_name_used_when_not_pull_convention() {
        let (_t, dirs) = setup();
        let d = dirs.models_dir();
        write_gguf(&d.join("plain-q4_k_m.gguf"), "llama", Some("My Model 8B"));
        let store = Store::open(&dirs).unwrap();
        let _report = reconcile_models(&dirs, &store);
        assert!(
            store.get_model("my-model-8b").unwrap().is_some(),
            "meta.name dialect, dots kept"
        );
    }

    #[test]
    fn unit__reconcile__shared_meta_name_falls_to_stem_candidates() {
        let (_t, dirs) = setup();
        let d = dirs.models_dir();
        // Real-world class: base + fp16 GGUF embed the SAME general.name,
        // and a safetensors row already holds that name. Each file must
        // still adopt under its own stem-derived name.
        let st = dirs.models_dir().join("qwen2.5-0.5b-instruct.d");
        std::fs::create_dir_all(&st).unwrap();
        std::fs::write(
            st.join("config.json"),
            br#"{"architectures":["Qwen2ForCausalLM"],"torch_dtype":"bfloat16"}"#,
        )
        .unwrap();
        std::fs::write(st.join("model.safetensors"), b"aaaa").unwrap();
        let store = Store::open(&dirs).unwrap();
        let _report = reconcile_models(&dirs, &store);
        assert!(store.get_model("qwen2.5-0.5b-instruct").unwrap().is_some());

        write_gguf(
            &d.join("qwen2.5-0.5b.gguf"),
            "qwen3",
            Some("Qwen2.5 0.5B Instruct"),
        );
        write_gguf(
            &d.join("qwen2.5-0.5b-instruct-fp16.gguf"),
            "qwen3",
            Some("Qwen2.5 0.5B Instruct"),
        );
        let r = reconcile_models(&dirs, &store);
        assert_eq!(r.adopted.len(), 2, "{:?}", r.skipped);
        // Base file: meta + stem-minus-tail both taken/invalid (`0.5b` is
        // not a quant token) → full stem.
        assert!(
            store.get_model("qwen2.5-0.5b").unwrap().is_some(),
            "{:?}",
            r.adopted
        );
        // fp16 file: meta taken → stem-minus-quant-tail ALSO taken →
        // full stem with quant.
        assert!(
            store
                .get_model("qwen2.5-0.5b-instruct-fp16")
                .unwrap()
                .is_some(),
            "{:?}",
            r.adopted
        );
    }
}
