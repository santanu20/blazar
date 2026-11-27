//! KV lifecycle inventory: what session-bank state exists on disk,
//! which model owns it, and what nothing owns.
//!
//! The scan mirrors the layouts the writer owns, so `kv status` /
//! `kv inspect` / `kv gc --dry-run` report the same truth the
//! supervisor's bank paths act on:
//!
//! * current banks: `<data>/sessions/<path_safe(model)>/_auto-<ctx>`
//!   (slot 0) and `_auto-<ctx>-s<N>` (slot N), with
//!   `<bank>.identity.json` manifests beside them;
//! * legacy banks: `<data>/sessions/<model>-*/` directories (the
//!   pre-path_safe layout `discover_newest_bank` still reads);
//! * mlx stage: `<data>/kv/mlx/<path_safe(model)>/` (probed by the
//!   caller - [`super::mlx_preload`] owns readiness).
//!
//! Everything here is read-only. `gc` stays a report in this release:
//! the scan names orphans and reclaimable bytes, it never deletes.

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use blazar_core::profile;

/// One bank file on disk, with the ctx/slot parsed back out of the
/// `_auto-<ctx>[-s<N>]` name. `None` fields = a name the parser does
/// not recognize (manual or future layout) - reported as-is, never
/// guessed at.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BankFile {
    pub name: String,
    pub bytes: u64,
    pub modified: SystemTime,
    pub ctx: Option<u32>,
    pub slot: Option<u32>,
}

/// Parse `_auto-<ctx>` / `_auto-<ctx>-s<N>` back into (ctx, slot).
/// Slot 0 is the unprefixed legacy form (see the supervisor's
/// `bank_slot_file`).
fn parse_bank_name(name: &str) -> Option<(u32, u32)> {
    let rest = name.strip_prefix("_auto-")?;
    if let Some((ctx, slot)) = rest.split_once("-s") {
        return Some((ctx.parse().ok()?, slot.parse().ok()?));
    }
    let ctx = rest.parse().ok()?;
    Some((ctx, 0))
}

/// Session-bank state for one live model, current + legacy layouts.
#[derive(Debug, Clone, Default)]
pub struct ModelKvInventory {
    pub model: String,
    /// Current-layout directory (`sessions/<path_safe(model)>`),
    /// present only when it exists on disk.
    pub session_dir: Option<PathBuf>,
    /// Banks in the current layout, newest mtime first.
    pub banks: Vec<BankFile>,
    /// Banks under legacy `sessions/<model>-*/` dirs, newest first.
    pub legacy_banks: Vec<BankFile>,
    /// Identity manifests found beside banks (parsed by the caller -
    /// [`blazar_core::session_identity::read_manifest`]).
    pub identity_manifests: Vec<PathBuf>,
    /// mlx stage present and ready (`kv/mlx/<path_safe(model)>`,
    /// both stage files present). `None` = no stage directory.
    pub mlx_stage_ready: Option<bool>,
}

impl ModelKvInventory {
    /// Total bytes across both layouts - the number `kv status`
    /// prints and `kv gc` accounts against live models.
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.banks
            .iter()
            .chain(&self.legacy_banks)
            .map(|b| b.bytes)
            .sum()
    }
}

/// A session-directory entry no live model owns, with the reason and
/// the reclaimable bytes (recursive directory size).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrphanKv {
    pub path: PathBuf,
    pub bytes: u64,
    pub reason: String,
}

/// List one directory's files as bank rows, skipping dotfiles and
/// JSON (identity manifests are collected separately). One unreadable
/// entry never aborts the scan.
fn bank_files(dir: &Path) -> (Vec<BankFile>, Vec<PathBuf>) {
    let mut banks = Vec::new();
    let mut manifests = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return (banks, manifests);
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with('.') {
            continue;
        }
        if name.ends_with(".identity.json") {
            manifests.push(entry.path());
            continue;
        }
        if Path::new(&name)
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("json"))
        {
            continue;
        }
        let Ok(md) = entry.metadata() else {
            continue;
        };
        let Ok(modified) = md.modified() else {
            continue;
        };
        let (ctx, slot) = parse_bank_name(&name).map_or((None, None), |(c, s)| (Some(c), Some(s)));
        banks.push(BankFile {
            name,
            bytes: md.len(),
            modified,
            ctx,
            slot,
        });
    }
    banks.sort_by_key(|a| std::cmp::Reverse(a.modified));
    (banks, manifests)
}

/// Recursive byte size of a path (files only; symlink metadata, never
/// followed). Unreadable entries count zero - the report errs toward
/// understating reclaimable space.
fn dir_size(path: &Path) -> u64 {
    let Ok(md) = std::fs::symlink_metadata(path) else {
        return 0;
    };
    if md.is_file() {
        return md.len();
    }
    if !md.is_dir() {
        return 0;
    }
    let Ok(entries) = std::fs::read_dir(path) else {
        return 0;
    };
    entries.flatten().map(|e| dir_size(&e.path())).sum()
}

/// Scan session-bank state for every live model. Pure read: absent
/// directories simply yield empty inventories, and the sessions root
/// itself may be missing on a fresh install.
#[must_use]
pub fn scan(data_dir: &Path, models: &[String]) -> Vec<ModelKvInventory> {
    let sessions = data_dir.join("sessions");
    models
        .iter()
        .map(|model| {
            // Current layout: one dir named exactly path_safe(model).
            let current = sessions.join(profile::path_safe(model));
            let (mut banks, identity_manifests) = bank_files(&current);
            let legacy_banks = legacy_banks_for(&sessions, model, &current);
            banks.sort_by_key(|a| std::cmp::Reverse(a.modified));
            ModelKvInventory {
                model: model.clone(),
                // Dir exists (even empty) = still report it: an empty
                // current dir with legacy banks is the pre-migration
                // shape users ask `kv status` about.
                session_dir: std::fs::read_dir(&current).is_ok().then_some(current),
                banks,
                legacy_banks,
                identity_manifests,
                mlx_stage_ready: super::mlx_preload::stage_dir(data_dir, model)
                    .exists()
                    .then(|| {
                        super::mlx_preload::stage_ready(&super::mlx_preload::stage_dir(
                            data_dir, model,
                        ))
                    }),
            }
        })
        .collect()
}

/// Banks under legacy `sessions/<model>-*/` directories - the prefix
/// scan `discover_newest_bank` reads. The prefix match is the legacy
/// contract itself (a model named `qwen` also claims `qwen2-...`
/// dirs); keeping it identical here means status and the doctor row
/// can never disagree about what legacy state exists. The model's
/// CURRENT dir is excluded: `path_safe(model)` is itself
/// `<name>-<hash>` shaped, so it would always prefix-match and every
/// current bank would double-count as legacy.
fn legacy_banks_for(sessions: &Path, model: &str, current_dir: &Path) -> Vec<BankFile> {
    let prefix = format!("{model}-");
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(sessions) else {
        return out;
    };
    for dir in entries.flatten() {
        if !dir.file_name().to_string_lossy().starts_with(&prefix) {
            continue;
        }
        let path = dir.path();
        if path == current_dir {
            continue; // current layout, already scanned above
        }
        let (banks, _) = bank_files(&path);
        out.extend(banks);
    }
    out.sort_by_key(|a| std::cmp::Reverse(a.modified));
    out
}

/// Session-directory entries no live model owns:
///
/// * directories matching neither a current `path_safe(model)` name
///   nor any model's legacy `<model>-*` prefix;
/// * directories that DO match a live model but hold no banks, no
///   legacy banks, and no identity manifests (debris after a bank
///   sweep - the model row keeps the dir alive, but there is nothing
///   in it);
/// * identity manifests inside a live model's current dir whose bank
///   file is gone (a save interrupted after the manifest write).
///
/// Report-only: the caller never deletes on this scan's authority.
#[must_use]
pub fn orphan_scan(data_dir: &Path, models: &[String]) -> Vec<OrphanKv> {
    let sessions = data_dir.join("sessions");
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(&sessions) else {
        return out;
    };
    for dir in entries.flatten() {
        let name = dir.file_name().to_string_lossy().to_string();
        if name.starts_with('.') {
            continue;
        }
        let path = dir.path();
        // Live check: exact current-layout dir, or legacy prefix.
        let owner_current = models.iter().find(|m| profile::path_safe(m) == name);
        let owner_legacy = models
            .iter()
            .find(|m| name.starts_with(&format!("{}-", m.as_str())));
        if let Some(model) = owner_current {
            // Current dir of a live model: only orphaned identity
            // manifests inside are reclaim candidates.
            for orphan in orphaned_manifests(&path, model) {
                out.push(orphan);
            }
            // Empty current dir + no legacy banks = debris (nothing
            // references it until the next bank save reuses it).
            let (banks, manifests) = bank_files(&path);
            if banks.is_empty()
                && manifests.is_empty()
                && legacy_banks_for(&sessions, model, &path).is_empty()
            {
                out.push(OrphanKv {
                    path,
                    bytes: 0,
                    reason: format!("empty session directory (live model {model:?} has no banks)"),
                });
            }
            continue;
        }
        if owner_legacy.is_some() {
            continue; // legacy banks of a live model stay
        }
        out.push(OrphanKv {
            bytes: dir_size(&path),
            path,
            reason: "no live model owns this session directory".to_string(),
        });
    }
    out.sort_by(|a, b| a.path.cmp(&b.path));
    out
}

/// Identity manifests whose bank file is missing: `<bank>.identity.json`
/// without `<bank>` on disk.
fn orphaned_manifests(current_dir: &Path, model: &str) -> Vec<OrphanKv> {
    let (_, manifests) = bank_files(current_dir);
    manifests
        .into_iter()
        .filter(|m| {
            // `<bank>.identity.json` -> `<bank>`: strip the appended
            // suffix (session names may contain dots, so the match is
            // anchored at the end, never split on extension).
            let name = m.file_name().unwrap_or_default().to_string_lossy();
            let bank = m.with_file_name(name.trim_end_matches(".identity.json"));
            !bank.is_file()
        })
        .map(|m| OrphanKv {
            bytes: m.metadata().map_or(0, |md| md.len()),
            path: m,
            reason: format!(
                "identity manifest without its bank (interrupted save; model {model:?})"
            ),
        })
        .collect()
}

#[cfg(test)]
#[allow(non_snake_case)] // suite convention: unit__scenario__expected
mod tests {
    use super::*;

    fn write(path: &Path, bytes: &[u8]) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, bytes).unwrap();
    }

    fn touch_past(path: &Path) {
        // Any mtime older than "now" suffices for ordering asserts.
        let past = SystemTime::now() - std::time::Duration::from_secs(3600);
        let f = std::fs::File::options().append(true).open(path).unwrap();
        f.set_modified(past).unwrap();
    }

    /// `parse_bank_name` round-trips both current forms and rejects
    /// non-bank names (reported as-is downstream, never guessed).
    #[test]
    fn unit__parse_bank_name__ctx_and_slot_forms() {
        assert_eq!(parse_bank_name("_auto-8192"), Some((8192, 0)));
        assert_eq!(parse_bank_name("_auto-8192-s3"), Some((8192, 3)));
        assert_eq!(parse_bank_name("manual.bank"), None);
        assert_eq!(parse_bank_name("_auto-abc"), None);
    }

    /// scan reports current + legacy banks per model, parses ctx/slot,
    /// collects identity manifests, and probes the mlx stage.
    #[test]
    fn unit__scan__current_legacy_and_mlx_stage() {
        let root = tempfile::tempdir().unwrap();
        let data = root.path();
        // path_safe(model) is itself `<name>-<fnv>` shaped; the current
        // dir name must be derived, never hand-written.
        let cur = profile::path_safe("qwen-a");
        // Current layout: one slot-0 bank, one slot-2 bank, one
        // identity manifest beside the slot-0 bank.
        write(&data.join(format!("sessions/{cur}/_auto-4096")), b"bank");
        write(
            &data.join(format!("sessions/{cur}/_auto-4096-s2")),
            b"bank2",
        );
        write(
            &data.join(format!("sessions/{cur}/_auto-4096.identity.json")),
            b"{}",
        );
        // Legacy layout: `qwen-a-...` dir (claimed by prefix scan, but
        // distinct from the current dir which is also prefix-shaped).
        write(&data.join("sessions/qwen-a-20260101/_auto-2048"), b"old");
        // mlx stage: ready shape = cache + tokens files.
        let stage = super::super::mlx_preload::stage_dir(data, "qwen-a");
        write(&stage.join(super::super::mlx_preload::CACHE_NAME), b"c");
        write(&stage.join(super::super::mlx_preload::TOKENS_NAME), b"t");

        let models = vec!["qwen-a".to_string()];
        let inv = scan(data, &models);
        assert_eq!(inv.len(), 1);
        let inv = &inv[0];
        assert_eq!(
            inv.session_dir.as_deref(),
            Some(data.join(format!("sessions/{cur}")).as_path())
        );
        assert_eq!(inv.banks.len(), 2, "{:?}", inv.banks);
        assert!(
            inv.banks
                .iter()
                .any(|b| b.ctx == Some(4096) && b.slot == Some(0))
        );
        assert!(
            inv.banks
                .iter()
                .any(|b| b.ctx == Some(4096) && b.slot == Some(2))
        );
        // The current dir is prefix-shaped too; it must NOT surface
        // again under legacy_banks (double-count pin).
        assert_eq!(inv.legacy_banks.len(), 1, "only the 20260101 dir");
        assert_eq!(inv.legacy_banks[0].ctx, Some(2048));
        assert_eq!(inv.identity_manifests.len(), 1);
        assert_eq!(inv.total_bytes(), 12); // 4+5+3 = "bank" + "bank2" + "old"
        assert_eq!(inv.mlx_stage_ready, Some(true));
    }

    /// A model with no state on disk reports an absent session dir
    /// (None), not an empty-dir phantom.
    #[test]
    fn unit__scan__absent_state_is_none_not_phantom() {
        let root = tempfile::tempdir().unwrap();
        let inv = scan(root.path(), &["ghost".to_string()]);
        assert_eq!(inv[0].session_dir, None);
        assert_eq!(inv[0].banks, [] as [crate::kv::inventory::BankFile; 0]);
        assert_eq!(inv[0].mlx_stage_ready, None);
        assert_eq!(inv[0].total_bytes(), 0);
    }

    /// `orphan_scan` names unowned dirs, empty live dirs, and identity
    /// manifests without banks - and keeps legacy dirs of live models.
    #[test]
    fn unit__orphan_scan__classes_and_live_model_exemptions() {
        let root = tempfile::tempdir().unwrap();
        let data = root.path();
        let live_cur = profile::path_safe("live-m");
        let kept_cur = profile::path_safe("kept-m");
        // Unowned dir: matches no model.
        write(&data.join("sessions/deadbeef/_auto-4096"), b"x");
        // Live model current dir, EMPTY (no banks): debris.
        std::fs::create_dir_all(data.join(format!("sessions/{live_cur}"))).unwrap();
        // Live model with a real bank: untouched.
        write(&data.join(format!("sessions/{kept_cur}/_auto-4096")), b"y");
        // Legacy dir of a live model: kept.
        write(&data.join("sessions/kept-m-legacy/_auto-1024"), b"z");
        // Orphaned identity manifest: manifest present, bank missing.
        write(
            &data.join(format!("sessions/{kept_cur}/_auto-999.identity.json")),
            b"{}",
        );

        let models = vec!["live-m".to_string(), "kept-m".to_string()];
        // orphan_scan's own ordering: sorted by full path - deadbeef,
        // kept-m's manifest, live-m's empty dir.
        let orphans = orphan_scan(data, &models);
        assert_eq!(orphans.len(), 3, "{orphans:?}");

        // deadbeef: unowned, bytes counted.
        assert_eq!(orphans[0].path, data.join("sessions/deadbeef"));
        assert_eq!(orphans[0].bytes, 1);
        assert!(orphans[0].reason.contains("no live model owns"));

        // orphaned manifest: named with its model.
        assert!(
            orphans[1]
                .path
                .to_string_lossy()
                .ends_with("_auto-999.identity.json")
        );
        assert!(orphans[1].reason.contains("without its bank"));

        // empty live dir: named as debris, zero bytes.
        assert!(orphans[2].path.to_string_lossy().contains(&live_cur));
        assert!(orphans[2].reason.contains("empty session directory"));
    }

    /// `dir_size` counts recursively, does not follow the sessions root
    /// when absent, and returns 0 for missing paths.
    #[test]
    fn unit__dir_size__recursive_and_missing() {
        let root = tempfile::tempdir().unwrap();
        let d = root.path().join("a");
        write(&d.join("one"), b"12345");
        write(&d.join("sub/two"), b"123");
        assert_eq!(dir_size(&d), 8);
        assert_eq!(dir_size(&root.path().join("missing")), 0);
    }

    /// Legacy prefix semantics mirror `discover_newest_bank`'s claim
    /// rule (a model owns `<model>-*` dirs) - pinned so the inventory
    /// and the doctor row can never disagree. The CURRENT dir
    /// (`path_safe` always prefix-matches the legacy rule) is excluded.
    #[test]
    fn unit__legacy_banks_for__prefix_claim_matches_doctor_rule() {
        let root = tempfile::tempdir().unwrap();
        let sessions = root.path().join("sessions");
        write(&sessions.join("qwen-2026/_auto-512"), b"b");
        write(&sessions.join("other-2026/_auto-512"), b"b");
        let current = sessions.join(blazar_core::profile::path_safe("qwen"));
        let mine = legacy_banks_for(&sessions, "qwen", &current);
        assert_eq!(mine.len(), 1);
        assert_eq!(mine[0].ctx, Some(512));
        touch_past(&sessions.join("qwen-2026/_auto-512"));
        // Ordering: newest first after re-scan.
        write(&sessions.join("qwen-2026/_auto-1024"), b"bb");
        let mine = legacy_banks_for(&sessions, "qwen", &current);
        assert_eq!(mine[0].ctx, Some(1024));
    }
}
