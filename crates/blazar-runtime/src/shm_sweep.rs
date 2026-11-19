//! Crash-leak hygiene for `vLLM`'s offload mmaps in `/dev/shm`.
//!
//! The `OffloadingConnector`'s CPU tier lives in shared memory
//! (`vllm_offload_*.mmap`, sized `cpu_bytes_gib`). A child that dies
//! mid-flight - OOM, engine crash - leaves the file behind with no
//! owner: leaks were observed live at 2 GiB per crash, three deep
//! after one debugging session, and a full `/dev/shm` kills every
//! later vllm spawn with "Insufficient space" before any of the
//! supervisor's teaching can fire. The sweep deletes only files no
//! live process still holds open.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

const SHM_DIR: &str = "/dev/shm";
const VLLM_OFFLOAD_PREFIX: &str = "vllm_offload_";

/// Offload mmaps under `/dev/shm` that at least one live process still
/// holds open, discovered through `/proc/*/fd` readlinks. A file held
/// by ANY process is never swept - including a sibling vllm child on a
/// different lane.
fn collect_in_use() -> HashSet<PathBuf> {
    let mut in_use = HashSet::new();
    let Ok(procs) = std::fs::read_dir("/proc") else {
        return in_use;
    };
    for proc_entry in procs.flatten() {
        let Ok(fds) = std::fs::read_dir(proc_entry.path().join("fd")) else {
            continue;
        };
        for fd in fds.flatten() {
            let Ok(link) = fd.path().read_link() else {
                continue;
            };
            if link.to_string_lossy().starts_with(SHM_DIR) {
                in_use.insert(link);
            }
        }
    }
    in_use
}

/// Delete `vllm_offload_*` entries in `dir` that are not in `in_use`.
/// Split from discovery so tests can drive it with a fake directory
/// and a fake in-use set - the real `/proc` walk is not hermetic.
fn delete_unowned(dir: &Path, in_use: &HashSet<PathBuf>) -> usize {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    let mut freed = 0usize;
    for entry in entries.flatten() {
        if !entry
            .file_name()
            .to_string_lossy()
            .starts_with(VLLM_OFFLOAD_PREFIX)
        {
            continue;
        }
        let path = entry.path();
        if in_use.contains(&path) {
            continue;
        }
        match std::fs::remove_file(&path) {
            Ok(()) => freed += 1,
            Err(e) => tracing::warn!(
                target: "blazar::shm",
                "could not remove orphaned vllm offload mmap {}: {e}",
                path.display()
            ),
        }
    }
    freed
}

/// One sweep pass; returns how many orphaned mmaps were removed. On
/// platforms without `/dev/shm` this is a no-op. Called before vllm
/// spawns (the acquire path the leak would break) and at supervisor
/// shutdown (children reaped, nothing legitimate remains).
pub fn sweep_orphaned_offload_mmaps() -> usize {
    let dir = Path::new(SHM_DIR);
    if !dir.is_dir() {
        return 0;
    }
    let freed = delete_unowned(dir, &collect_in_use());
    if freed > 0 {
        tracing::info!(
            target: "blazar::shm",
            freed,
            "removed orphaned vllm offload mmap(s) from /dev/shm"
        );
    }
    freed
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The sweep deletes only unowned vllm offload files: an in-use
    /// mmap and a foreign file both survive, the orphan goes.
    #[test]
    #[allow(non_snake_case)]
    fn unit__delete_unowned__removes_only_orphaned_offload_mmaps() {
        let dir = tempfile::tempdir().unwrap();
        let orphan = dir.path().join("vllm_offload_deadbeef.mmap");
        let live = dir.path().join("vllm_offload_alive.mmap");
        let foreign = dir.path().join("other_engine.mmap");
        for p in [&orphan, &live, &foreign] {
            std::fs::write(p, b"x").unwrap();
        }
        let in_use: HashSet<PathBuf> = [live.clone()].into_iter().collect();
        let freed = delete_unowned(dir.path(), &in_use);
        assert_eq!(freed, 1);
        assert!(!orphan.exists());
        assert!(live.exists(), "an in-use mmap must never be swept");
        assert!(
            foreign.exists(),
            "files outside the vllm prefix are not ours"
        );
    }

    /// No vllm files at all -> the pass is quiet and reports zero.
    #[test]
    #[allow(non_snake_case)]
    fn unit__delete_unowned__empty_directory_is_a_quiet_no_op() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(delete_unowned(dir.path(), &HashSet::new()), 0);
        assert_eq!(
            delete_unowned(Path::new("/nonexistent"), &HashSet::new()),
            0
        );
    }
}
