//! Containment-guarded directory removal for Blazar-managed trees.
//!
//! Every destructive sweep (engine retention, orphan-dir reclamation,
//! rollback-aside cleanup, piper/whisper lane pruning, snapshot pruning)
//! removes a path derived from the data root (`engines/<tag>`,
//! `whisper/bin/<tag>`, ...). When a user symlinks one of those managed
//! roots elsewhere — a bigger disk, a shared store — a naive
//! `remove_dir_all` follows the link and deletes content the data root
//! does not own (live incident 2026-09-26: a scratch XDG data dir with
//! `engines -> ~/.local/share/blazar/engines` reclaimed five real engine
//! installs through the link). This module makes that escape impossible:
//! deletion only proceeds on a path that canonically sits inside the
//! anchor root, and a symlink at the final component is unlinked as a
//! link — its target is never descended into.

use std::path::Path;

/// Outcome of a guarded removal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuardedRemoval {
    /// The directory tree (or final-component link) was removed.
    Removed,
    /// Nothing existed at the path.
    Absent,
    /// The path resolves outside the anchor root — deliberately left
    /// untouched. Hygiene sweeps skip it; retention callers keep the
    /// store row so dir and row never desync. Wherever that symlink
    /// points is the user's ground, not the data root's.
    Escaped,
}

/// Remove `target` only when it canonically sits inside `anchor`.
///
/// The containment check runs on canonicalized paths, so a symlink
/// anywhere along the target path that redirects outside the anchor —
/// including a symlinked managed root itself — yields
/// [`GuardedRemoval::Escaped`] instead of a deletion on foreign ground.
/// A symlink at the final component is removed as a link
/// (`remove_file`, falling back to `remove_dir` for Windows directory
/// reparse points); its target is never touched.
///
/// IO failures propagate for the caller's existing posture: fatal for
/// explicit prunes, logged-and-skipped for fail-open sweeps. A missing
/// anchor canonicalizes to an error, which callers treat as "cannot
/// prove containment" — same [`GuardedRemoval::Escaped`] refusal.
pub fn remove_dir_within(anchor: &Path, target: &Path) -> std::io::Result<GuardedRemoval> {
    let meta = match std::fs::symlink_metadata(target) {
        Ok(m) => m,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Ok(GuardedRemoval::Absent);
        }
        Err(e) => return Err(e),
    };
    if meta.file_type().is_symlink() {
        // Unlink the link itself. `remove_dir` on a Windows directory
        // symlink removes the reparse point, not its target — try it
        // when the plain unlink is refused.
        return std::fs::remove_file(target)
            .or_else(|_| std::fs::remove_dir(target))
            .map(|()| GuardedRemoval::Removed);
    }
    let (Ok(anchor_canon), Ok(target_canon)) =
        (std::fs::canonicalize(anchor), std::fs::canonicalize(target))
    else {
        return Ok(GuardedRemoval::Escaped);
    };
    // `Path::starts_with` compares components, so a sibling-prefix dir
    // (`engines-bak` next to `engines`) never passes as contained.
    if !target_canon.starts_with(&anchor_canon) {
        tracing::warn!(
            "refusing to remove {} — resolves to {}, outside the data root {} \
             (symlinked managed dir? remove it by hand if that is intended)",
            target.display(),
            target_canon.display(),
            anchor_canon.display()
        );
        return Ok(GuardedRemoval::Escaped);
    }
    std::fs::remove_dir_all(target).map(|()| GuardedRemoval::Removed)
}

/// Non-destructive containment probe: does `target` canonically sit
/// inside `anchor`? Missing paths and unusable anchors read as
/// "cannot prove containment" (false), so callers fail safe.
#[must_use]
pub fn path_is_within(anchor: &Path, target: &Path) -> bool {
    match (std::fs::canonicalize(anchor), std::fs::canonicalize(target)) {
        (Ok(a), Ok(t)) => t.starts_with(&a),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)] // suite convention: unit__scenario__expected (§6b)
    use super::*;
    use std::path::PathBuf;

    fn tree(root: &Path) {
        std::fs::create_dir_all(root.join("a/b")).unwrap();
        std::fs::write(root.join("a/b/f.txt"), b"x").unwrap();
    }

    #[test]
    fn unit__remove_dir_within__contained_tree_removed() {
        let tmp = tempfile::tempdir().unwrap();
        let anchor = tmp.path().join("data");
        let dir = anchor.join("engines/tag-x");
        tree(&dir);
        assert_eq!(
            remove_dir_within(&anchor, &dir).unwrap(),
            GuardedRemoval::Removed
        );
        assert!(!dir.exists());
    }

    #[test]
    fn unit__remove_dir_within__absent_path_reports_absent() {
        let tmp = tempfile::tempdir().unwrap();
        assert_eq!(
            remove_dir_within(tmp.path(), &tmp.path().join("nope")).unwrap(),
            GuardedRemoval::Absent
        );
    }

    #[cfg(unix)]
    #[test]
    fn unit__remove_dir_within__final_symlink_unlinks_link_only() {
        let tmp = tempfile::tempdir().unwrap();
        let anchor = tmp.path().join("data");
        let foreign = tmp.path().join("foreign");
        tree(&foreign);
        let link = anchor.join("engines/tag-x");
        std::fs::create_dir_all(anchor.join("engines")).unwrap();
        std::os::unix::fs::symlink(&foreign, &link).unwrap();
        assert_eq!(
            remove_dir_within(&anchor, &link).unwrap(),
            GuardedRemoval::Removed
        );
        assert!(!link.exists(), "the link itself must be gone");
        assert!(foreign.join("a/b/f.txt").is_file(), "target survives");
    }

    #[cfg(unix)]
    #[test]
    fn unit__remove_dir_within__symlinked_root_escapes_anchor() {
        // The incident shape: managed root is a symlink to a directory
        // outside the data root; a child dir of it must never be removed.
        let tmp = tempfile::tempdir().unwrap();
        let anchor = tmp.path().join("data");
        std::fs::create_dir_all(&anchor).unwrap();
        let foreign_root = tmp.path().join("real-store/engines");
        tree(&foreign_root.join("b11193-cuda"));
        std::fs::create_dir_all(foreign_root.parent().unwrap()).unwrap();
        std::os::unix::fs::symlink(&foreign_root, anchor.join("engines")).unwrap();
        let victim = anchor.join("engines/b11193-cuda");
        assert_eq!(
            remove_dir_within(&anchor, &victim).unwrap(),
            GuardedRemoval::Escaped
        );
        assert!(
            foreign_root.join("b11193-cuda/a/b/f.txt").is_file(),
            "content behind the symlinked root survives"
        );
        assert!(victim.exists(), "path through the link still resolves");
    }

    #[cfg(unix)]
    #[test]
    fn unit__remove_dir_within__sibling_prefix_is_not_contained() {
        let tmp = tempfile::tempdir().unwrap();
        let anchor = tmp.path().join("data/engines");
        let sibling = tmp.path().join("data/engines-bak");
        tree(&sibling);
        assert_eq!(
            remove_dir_within(&anchor, &sibling).unwrap(),
            GuardedRemoval::Escaped
        );
        assert!(sibling.join("a/b/f.txt").is_file());
    }

    #[test]
    fn unit__remove_dir_within__anchor_equals_target_boundary() {
        let tmp = tempfile::tempdir().unwrap();
        let dir: PathBuf = tmp.path().to_path_buf();
        tree(&dir);
        // Removing the anchor itself is in-bounds by definition.
        assert_eq!(
            remove_dir_within(&dir, &dir).unwrap(),
            GuardedRemoval::Removed
        );
        assert!(!dir.exists());
    }
}
