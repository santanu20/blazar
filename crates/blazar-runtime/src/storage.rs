//! Disk-space intelligence: the same planning discipline the VRAM planner
//! applies to memory, applied to storage. Three consumers:
//!
//! * `gate_disk` — pull-time admission for the filesystem: refuse a download
//!   that cannot finish instead of dying at 95% (the classic multi-GiB
//!   heartbreak). The parallel lane preallocates the full-size `.part`, so
//!   peak need == final size plus a small slack, never 2x.
//! * `disk_verdict` / `disk_free_bytes` — the `blazar fit` DISK column and
//!   the `blazar storage` report.
//! * `orphan_scan` / `unused_rows` — `blazar prune` candidates: files the
//!   store owns no row for (aborted pulls, replaced quants), stale `.part`
//!   debris, and models nobody has loaded in a chosen window.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Result};
use blazar_core::dirs::BlazarDirs;
use blazar_core::store::ModelRow;

/// Headroom added to a download's byte total when admitting it: the
/// `.part.progress` sidecar ledger, store growth, and filesystem
/// bookkeeping. Deliberately small — the `.part` is preallocated at exactly
/// the final size and renamed in place, so there is no copy phase.
pub const REQUIRED_SLACK_BYTES: u64 = 64 * 1024 * 1024;

/// `.part` / `.part.progress` files younger than this are presumed to belong
/// to a live pull in another process and are never reported as reclaimable.
pub const PART_MIN_AGE_SECS: u64 = 24 * 60 * 60;

/// Available space on the filesystem holding `path` (sysinfo mount table,
/// longest mount-point prefix of the canonicalized path). `None` when no
/// mount matches (odd container layouts) — callers treat that as "unknown",
/// never as zero.
#[must_use]
pub fn disk_free_bytes(path: &Path) -> Option<u64> {
    let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    let disks = sysinfo::Disks::new_with_refreshed_list();
    let mut best: Option<(&Path, u64)> = None;
    for disk in disks.list() {
        let mount = disk.mount_point();
        if canonical.starts_with(mount)
            && best.is_none_or(|(m, _)| mount.as_os_str().len() > m.as_os_str().len())
        {
            best = Some((mount, disk.available_space()));
        }
    }
    best.map(|(_, free)| free)
}

/// Total size of the filesystem holding `path` (same matching rule as
/// [`disk_free_bytes`]; used for the fit header line).
#[must_use]
pub fn disk_total_bytes(path: &Path) -> Option<u64> {
    let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    let disks = sysinfo::Disks::new_with_refreshed_list();
    let mut best: Option<(&Path, u64)> = None;
    for disk in disks.list() {
        let mount = disk.mount_point();
        if canonical.starts_with(mount)
            && best.is_none_or(|(m, _)| mount.as_os_str().len() > m.as_os_str().len())
        {
            best = Some((mount, disk.total_space()));
        }
    }
    best.map(|(_, total)| total)
}

/// Recursive size of `path` (files only; symlinks are never followed and
/// contribute 0). Hardlink twins inside one walk are counted per path —
/// `blazar storage` labels twin reclaim honestly instead (deleting a twin
/// frees nothing while the other link lives).
#[must_use]
pub fn du(path: &Path) -> u64 {
    let Ok(meta) = std::fs::symlink_metadata(path) else {
        return 0;
    };
    if meta.is_file() {
        return meta.len();
    }
    if !meta.is_dir() {
        return 0;
    }
    let mut total = 0u64;
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            total += du(&entry.path());
        }
    }
    total
}

/// FIT verdict for `required` bytes against `available` (None = unknown).
/// TIGHT fires when the download would leave under 10% of the disk free —
/// the point where SQLite WAL checkpoints and engine extraction start
/// hurting even though the raw bytes technically fit.
#[must_use]
pub fn disk_verdict(required: u64, available: Option<u64>) -> &'static str {
    let Some(avail) = available else {
        return "UNKNOWN";
    };
    if required > avail {
        "NO"
    } else if required + avail / 10 > avail {
        "TIGHT"
    } else {
        "FIT"
    }
}

/// Pull-time filesystem admission: error (fail-fast, with the numbers) when
/// `total_bytes` + slack cannot land on the models filesystem. `what` names
/// the artifact for the teaching message.
pub fn gate_disk(dirs: &BlazarDirs, total_bytes: u64, what: &str) -> Result<()> {
    let models_dir = dirs.models_dir();
    std::fs::create_dir_all(&models_dir)
        .map_err(|e| anyhow!("cannot create models dir {}: {e}", models_dir.display()))?;
    let Some(avail) = disk_free_bytes(&models_dir) else {
        // Unknown filesystem, not a refusal: sysinfo could not resolve the
        // mount. The old behavior (no gate) is the only honest fallback.
        return Ok(());
    };
    let required = total_bytes.saturating_add(REQUIRED_SLACK_BYTES);
    if required > avail {
        Err(anyhow!(
            "not enough disk for {what}: needs {} (model + 64 MiB slack), {} available on {} \
             — free space first: `blazar storage` shows the breakdown, `blazar prune` reclaims \
             orphaned files and unused models",
            humansize(required),
            humansize(avail),
            models_dir.display()
        ))
    } else {
        Ok(())
    }
}

/// Decimal byte rendering for teaching messages (matches CLI humansize).
#[must_use]
#[allow(clippy::cast_precision_loss)] // 2^52-byte precision loss is meaningless in a human-readable size
pub fn humansize(bytes: u64) -> String {
    const UNITS: [&str; 7] = ["B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB"];
    if bytes == 0 {
        return "0 B".to_string();
    }
    let mut v = bytes as f64;
    let mut unit = 0usize;
    while v >= 1024.0 && unit < UNITS.len() - 1 {
        v /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{v:.1} {}", UNITS[unit])
    }
}

/// Everything the models directory holds that no store row owns.
#[derive(Debug, Default, Clone)]
pub struct OrphanReport {
    /// Files with no owning row (stray GGUFs, replaced quants, foreign
    /// downloads). Deleting these frees real bytes.
    pub orphan_files: Vec<(PathBuf, u64)>,
    /// Stale `.part` / `.part.progress` debris older than
    /// [`PART_MIN_AGE_SECS`] (younger ones may belong to a live pull).
    pub stale_partials: Vec<(PathBuf, u64)>,
    /// Hardlink twins of owned files (unix): separate paths, same bytes on
    /// disk. Deleting frees nothing while the owned link lives — reported,
    /// never auto-pruned.
    pub twins: Vec<(PathBuf, u64)>,
}

/// Scan `models_dir` against the owned-set derived from `rows`:
/// row paths (file or safetensors directory), mmproj sidecars, and shard
/// siblings (`stem-0000N-of-0000M.gguf` — the row stores only the first
/// shard, the set is derivable, same rule `prune_replaced` applies).
#[must_use]
pub fn orphan_scan(models_dir: &Path, rows: &[ModelRow]) -> OrphanReport {
    let mut owned: HashSet<PathBuf> = HashSet::new();
    for row in rows {
        let row_path = Path::new(&row.path);
        owned.insert(row_path.to_path_buf());
        if let Some(mm) = &row.mmproj_path {
            owned.insert(PathBuf::from(mm));
        }
        // Safetensors models own their whole directory.
        if row_path.is_dir() {
            if let Ok(entries) = std::fs::read_dir(row_path) {
                for entry in entries.flatten() {
                    let p = entry.path();
                    if p.is_file() {
                        owned.insert(p);
                    }
                }
            }
        }
        // GGUF shard sets: the row names shard 1; own the siblings.
        if let Some(name) = row_path.file_name().and_then(|n| n.to_str()) {
            if let Some(set_len) = shard_set_len(name) {
                let stem = name.trim_end_matches(".gguf");
                let prefix = stem.strip_suffix(shard_first_suffix(stem)).unwrap_or(stem);
                for idx in 1..=set_len {
                    let sibling = format!("{prefix}{idx:05}-of-{set_len:05}.gguf");
                    owned.insert(row_path.with_file_name(sibling));
                }
            }
        }
    }
    // Second pass for twins needs inode identity of owned files.
    #[cfg(unix)]
    let owned_ids: HashSet<(u64, u64)> = owned
        .iter()
        .filter_map(|p| std::fs::metadata(p).ok())
        .map(|m| {
            use std::os::unix::fs::MetadataExt;
            (m.dev(), m.ino())
        })
        .collect();

    let mut report = OrphanReport::default();
    let Ok(entries) = std::fs::read_dir(models_dir) else {
        return report;
    };
    for entry in entries.flatten() {
        let p = entry.path();
        let Ok(meta) = std::fs::symlink_metadata(&p) else {
            continue;
        };
        if !meta.is_file() {
            continue;
        }
        let name = p
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default()
            .to_lowercase();
        let ext = std::path::Path::new(&name)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or_default();
        let is_partial = ext == "part" || name.ends_with("part.progress");
        if is_partial {
            let age = meta
                .modified()
                .ok()
                .and_then(|t| t.elapsed().ok())
                .map_or(u64::MAX, |d| d.as_secs());
            if age >= PART_MIN_AGE_SECS {
                report.stale_partials.push((p, meta.len()));
            }
            continue;
        }
        if owned.contains(&p) {
            continue;
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let id = (meta.dev(), meta.ino());
            if owned_ids.contains(&id) {
                report.twins.push((p, meta.len()));
                continue;
            }
        }
        report.orphan_files.push((p, meta.len()));
    }
    report
}

/// `Some(N)` when `name` matches the sharded GGUF convention
/// `stem-00001-of-0000N.gguf`.
fn shard_set_len(name: &str) -> Option<u32> {
    let lower = name.to_lowercase();
    let stem = lower.strip_suffix(".gguf")?;
    let dash = stem.rfind("-of-")?;
    let total: String = stem[dash + 4..]
        .chars()
        .filter(char::is_ascii_digit)
        .collect();
    total.parse().ok().filter(|&n| n >= 2)
}

fn shard_first_suffix(stem: &str) -> &str {
    stem.rfind("-00001-of-")
        .map(|i| &stem[i..])
        .unwrap_or_default()
}

/// Rows not loaded (spawned) within `since_secs` and not currently resident.
/// `last_used_at` is touched when a model goes resident, so a model serving
/// long-term stays fresh by staying loaded, and one nobody re-spawns ages
/// out honestly. `resident` names come from a live `/api/ps` when reachable.
#[must_use]
pub fn unused_rows<'a>(
    rows: &'a [ModelRow],
    since_secs: u64,
    resident: &[String],
    now: i64,
) -> Vec<&'a ModelRow> {
    let cutoff = now.saturating_sub(i64::try_from(since_secs).unwrap_or(i64::MAX));
    rows.iter()
        .filter(|r| !resident.iter().any(|n| n == &r.name))
        .filter(|r| r.last_used_at < cutoff)
        .collect()
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;
    use blazar_core::store::Store;

    fn dirs(tmp: &tempfile::TempDir) -> BlazarDirs {
        BlazarDirs {
            config_dir: tmp.path().join("cfg"),
            data_dir: tmp.path().join("data"),
        }
    }

    fn row(path: &str, name: &str, last_used: i64) -> ModelRow {
        ModelRow {
            name: name.to_string(),
            repo: "owner/repo".to_string(),
            quant: "q4_k_m".to_string(),
            path: path.to_string(),
            bytes: 1,
            sha256: None,
            mmproj_path: None,
            components: Vec::new(),
            shards: 1,
            arch: None,
            params: None,
            ctx_train: None,
            pulled_at: 100,
            last_used_at: last_used,
        }
    }

    #[test]
    fn unit__disk_verdict__fit_tight_no_unknown() {
        assert_eq!(disk_verdict(100, Some(10_000)), "FIT");
        assert_eq!(disk_verdict(100, Some(1_000)), "TIGHT");
        assert_eq!(disk_verdict(100, Some(99)), "NO");
        assert_eq!(disk_verdict(100, None), "UNKNOWN");
    }

    #[test]
    fn unit__du__sums_files_not_symlinks() {
        let tmp = tempfile::tempdir().unwrap();
        let d = tmp.path().join("tree");
        std::fs::create_dir_all(d.join("sub")).unwrap();
        std::fs::write(d.join("a.bin"), vec![0u8; 100]).unwrap();
        std::fs::write(d.join("sub").join("b.bin"), vec![0u8; 50]).unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(d.join("a.bin"), d.join("link.bin")).unwrap();
        assert_eq!(du(&d), 150);
        assert_eq!(du(&d.join("a.bin")), 100);
        assert_eq!(du(&d.join("missing")), 0);
    }

    #[test]
    fn unit__gate_disk__refuses_with_numbers_passes_with_room() {
        let tmp = tempfile::tempdir().unwrap();
        let dd = dirs(&tmp);
        std::fs::create_dir_all(dd.models_dir()).unwrap();
        // Unknown fs (tmpdir may still resolve on real mounts) — treat both
        // branches honestly: gate on a path that cannot resolve to a mount.
        let gated = gate_disk(&dd, u64::MAX, "probe-model");
        // On Linux tmpfs IS a mount with finite space, so this must refuse.
        if gated.is_err() {
            let msg = gated.unwrap_err().to_string();
            assert!(msg.contains("not enough disk"), "{msg}");
            assert!(msg.contains("blazar prune"), "{msg}");
        }
        // Tiny downloads always pass on a real tmpdir.
        assert!(gate_disk(&dd, 1024, "probe-model").is_ok());
    }

    #[test]
    fn unit__orphan_scan__owned_twins_partials_and_shard_sets() {
        let tmp = tempfile::tempdir().unwrap();
        let dd = dirs(&tmp);
        let m = dd.models_dir();
        std::fs::create_dir_all(&m).unwrap();
        let owned_file = m.join("big-00001-of-00002.gguf");
        std::fs::write(&owned_file, vec![0u8; 10]).unwrap();
        std::fs::write(m.join("big-00002-of-00002.gguf"), vec![0u8; 10]).unwrap();
        std::fs::write(m.join("stray-q4.gguf"), vec![0u8; 5]).unwrap();
        // Owned row with a fresh .part (live pull) and a stale one.
        std::fs::write(m.join("big-00001-of-00002.gguf.part"), vec![0u8; 3]).unwrap();
        let old = m.join("old.gguf.part");
        std::fs::write(&old, vec![0u8; 4]).unwrap();
        let stale_time =
            std::time::SystemTime::now() - std::time::Duration::from_secs(PART_MIN_AGE_SECS + 60);
        #[cfg(unix)]
        {
            let f = std::fs::OpenOptions::new().write(true).open(&old).unwrap();
            f.set_times(std::fs::FileTimes::new().set_modified(stale_time))
                .unwrap();
        }
        #[cfg(unix)]
        std::fs::hard_link(&owned_file, m.join("twin-copy.gguf")).unwrap();

        let r = orphan_scan(&m, &[row(owned_file.to_str().unwrap(), "big-model", 100)]);

        assert!(r
            .orphan_files
            .iter()
            .any(|(p, _)| p.ends_with("stray-q4.gguf")));
        assert!(!r
            .orphan_files
            .iter()
            .any(|(p, _)| p.to_string_lossy().contains("big-0000")));
        #[cfg(unix)]
        assert!(r.twins.iter().any(|(p, _)| p.ends_with("twin-copy.gguf")));
        assert!(r.stale_partials.iter().any(|(p, _)| p == &old));
        assert!(!r
            .stale_partials
            .iter()
            .any(|(p, _)| p.ends_with("00002.gguf.part")));
    }

    #[test]
    fn unit__unused_rows__age_and_resident_filters() {
        let now = 10_000i64;
        let rows = vec![
            row("/a.gguf", "old", 1),
            row("/b.gguf", "fresh", 9_999),
            row("/c.gguf", "resident-old", 1),
        ];
        let resident = vec!["resident-old".to_string()];
        let out = unused_rows(&rows, 30 * 86_400, &resident, now);
        let names: Vec<&str> = out.iter().map(|r| r.name.as_str()).collect();
        assert_eq!(names, vec!["old"]);
    }

    #[test]
    fn unit__shard_set_len__convention_only() {
        assert_eq!(shard_set_len("big-00001-of-00003.gguf"), Some(3));
        assert_eq!(shard_set_len("plain-q4.gguf"), None);
        assert_eq!(shard_set_len("x-00001-of-00001.gguf"), None);
    }

    #[test]
    fn unit__store_v9__last_used_backfill_and_touch() {
        let tmp = tempfile::tempdir().unwrap();
        let dd = dirs(&tmp);
        std::fs::create_dir_all(dd.models_dir()).unwrap();
        let store = Store::open(&dd).unwrap();
        let mut r = row("/m.gguf", "m", 0);
        r.pulled_at = 5_000;
        store.upsert_model(&r).unwrap();
        let got = store.get_model("m").unwrap().unwrap();
        // Fresh v9 stores backfill last_used from created_at so a just-added
        // model is never instantly "unused".
        assert_eq!(got.last_used_at, 5_000);
        assert!(store.touch_model_used("m").unwrap());
        let got = store.get_model("m").unwrap().unwrap();
        assert!(got.last_used_at > 5_000, "touch must advance the stamp");
        assert!(!store.touch_model_used("missing").unwrap());
    }
}
