//! Post-install verification for the pip-venv engine lanes.
//!
//! GH-asset lanes carry a publisher digest and verify their downloads
//! in flight; the pip lanes historically stamped `unverified` on their
//! store rows. PEP 376 gives them a better source of truth: every venv
//! ships `*.dist-info/RECORD` manifests listing each installed file
//! with the sha256 pip itself computed at write time. Re-hashing those
//! files proves the venv on disk is byte-for-byte the one the
//! installer produced - a truncated or corrupted install becomes a
//! hard failure at install time instead of a latent crash at serve
//! time. The lane digest `pip:<hex>` (sha256 over the venv's full
//! normalized RECORD set) lands on the store row so `engine verify`
//! can re-check it later.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use base64::Engine as _;
use sha2::{Digest, Sha256};

/// Row-digest prefix for lanes verified through their venv RECORD set
/// (the `sha256:<hex>` form stays reserved for GH publisher digests).
pub(crate) const PIP_DIGEST_PREFIX: &str = "pip:";

/// Verify every RECORD-listed file of the venv under `engine_dir` and
/// return the lane digest `pip:<hex>`: sha256 over the venv's full
/// normalized RECORD set (sorted `path,hash` lines, hashless entries
/// included). A listed file that is missing, unreachable or re-hashes
/// differently is a hard error naming the file and both digests -
/// callers roll the install back.
pub(crate) fn verify_venv_record(engine_dir: &Path) -> Result<String> {
    let venv = engine_dir.join("venv");
    if !venv.is_dir() {
        bail!("engine dir {} has no venv/ to verify", engine_dir.display());
    }
    let mut records = Vec::new();
    collect_record_files(&venv, &mut records)?;
    if records.is_empty() {
        bail!(
            "venv {} contains no RECORD manifest - cannot verify",
            venv.display()
        );
    }
    let mut lines: Vec<String> = Vec::new();
    for record in &records {
        verify_one_record(&venv, record, &mut lines)?;
    }
    lines.sort_unstable();
    let mut hasher = Sha256::new();
    for line in &lines {
        hasher.update(line.as_bytes());
        hasher.update(b"\n");
    }
    Ok(format!(
        "{PIP_DIGEST_PREFIX}{}",
        hex::encode(hasher.finalize())
    ))
}

/// Depth-first walk collecting every `<pkg>.dist-info/RECORD` under
/// the venv. Symlinked directories are not followed - the venv pip
/// builds contains none, and a hostile one must not steer the walk
/// outside the engine dir.
fn collect_record_files(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    let entries =
        std::fs::read_dir(dir).with_context(|| format!("walk venv dir {}", dir.display()))?;
    for entry in entries {
        let entry = entry.with_context(|| format!("walk venv dir {}", dir.display()))?;
        let is_dir = entry
            .file_type()
            .with_context(|| format!("stat venv entry {}", entry.path().display()))?
            .is_dir();
        if is_dir {
            collect_record_files(&entry.path(), out)?;
        } else if entry.file_name() == "RECORD" {
            // Only distributions installed directly into a site-packages
            // root own their RECORD. Nested ones - setuptools vendors
            // wheels under `<pkg>/_vendor/*.dist-info` and drops their
            // scripts - describe a wheel this venv never installed, so
            // their file lists lie about the venv on purpose.
            let record_path = entry.path();
            let dist_info = record_path.parent();
            let parent_is_dist_info = dist_info
                .and_then(|p| p.file_name())
                .is_some_and(|n| n.to_string_lossy().ends_with(".dist-info"));
            let sits_in_site_packages = dist_info
                .and_then(|p| p.parent())
                .and_then(|p| p.file_name())
                .is_some_and(|n| n == "site-packages");
            if parent_is_dist_info && sits_in_site_packages {
                out.push(entry.path());
            }
        }
    }
    Ok(())
}

/// Re-hash every entry of one RECORD and append the normalized
/// `path,hash` lines (hashless entries as `path,`) to `lines`.
fn verify_one_record(venv: &Path, record: &Path, lines: &mut Vec<String>) -> Result<()> {
    let text = std::fs::read_to_string(record)
        .with_context(|| format!("read RECORD {}", record.display()))?;
    // RECORD paths are relative to the dist-info's parent (the
    // site-packages root); script entries legitimately carry `../..`
    // segments to reach venv/bin.
    let base = record
        .parent()
        .and_then(|dist| dist.parent())
        .ok_or_else(|| anyhow!("RECORD {} has no site-packages base", record.display()))?;
    for (idx, raw) in text.lines().enumerate() {
        let raw = raw.trim_end_matches('\r');
        if raw.is_empty() {
            continue;
        }
        let (path, hash) = parse_record_line(raw).ok_or_else(|| {
            anyhow!(
                "malformed RECORD line {} in {}: {raw:?}",
                idx + 1,
                record.display()
            )
        })?;
        if hash.is_empty() {
            // Hashless entries (the RECORD itself, occasionally a .pyc
            // post-compiled) still fingerprint the manifest.
            lines.push(format!("{path},"));
            continue;
        }
        let expected = decode_record_hash(&hash)
            .with_context(|| format!("bad digest field in {}: {raw:?}", record.display()))?;
        let file = base.join(&path);
        // Read before the containment guard so a missing in-venv file
        // reports "lacks it" (path_is_within canonicalizes, which needs
        // the file to exist); a file that exists but resolves outside
        // the venv is then refused without being hashed.
        let bytes = std::fs::read(&file).with_context(|| {
            format!(
                "RECORD {} lists {path} but the venv lacks it",
                record.display()
            )
        })?;
        if !blazar_core::fs_safety::path_is_within(venv, &file) {
            bail!(
                "RECORD {} lists {path} outside the venv - refusing to hash it",
                record.display()
            );
        }
        let actual = Sha256::digest(&bytes);
        if actual.as_slice() != expected.as_slice() {
            bail!(
                "venv file {path} drifted from its RECORD: expected sha256 {}, got {}",
                hex::encode(&expected),
                hex::encode(actual)
            );
        }
        lines.push(format!("{path},{hash}"));
    }
    Ok(())
}

/// One RECORD line: `path,hash[,size]` with RFC 4180 quoting - paths
/// may contain commas and doubled quotes; the hash field is empty for
/// hashless entries. The trailing size field is not part of the
/// fingerprint, so it is dropped here. Returns `None` for lines that
/// are not a well-formed two-field record.
fn parse_record_line(raw: &str) -> Option<(String, String)> {
    let mut path = String::new();
    let mut hash = String::new();
    let mut field = String::new();
    // 0 = the path field, 1 = the hash field, 2+ = size and any tail
    // (kept out of the fingerprint).
    let mut field_idx = 0usize;
    let mut in_quotes = false;
    let mut chars = raw.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '"' if in_quotes => {
                // Doubled quote = a literal quote inside the quoted
                // field; a lone one closes it.
                if chars.peek() == Some(&'"') {
                    chars.next();
                    field.push('"');
                } else {
                    in_quotes = false;
                }
            }
            '"' => {
                // Opening quotes only start a field (RFC 4180).
                if !field.is_empty() {
                    return None;
                }
                in_quotes = true;
            }
            ',' if !in_quotes => match field_idx {
                0 => {
                    path = std::mem::take(&mut field);
                    field_idx = 1;
                }
                1 => {
                    hash = std::mem::take(&mut field);
                    field_idx = 2;
                }
                _ => {}
            },
            _ => field.push(c),
        }
    }
    if in_quotes {
        return None; // unterminated quoted field
    }
    if field_idx == 0 {
        return None; // no hash field at all
    }
    if field_idx == 1 {
        hash = field;
    }
    Some((path, hash))
}

/// RECORD hash fields are `sha256=<urlsafe base64, no padding>` (pip's
/// only digest algorithm; anything else is a format change upstream
/// and fails loudly rather than being skipped silently).
fn decode_record_hash(hash: &str) -> Result<Vec<u8>> {
    let encoded = hash
        .strip_prefix("sha256=")
        .ok_or_else(|| anyhow!("unsupported RECORD digest {hash:?}"))?;
    let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(encoded)
        .with_context(|| format!("decode RECORD digest {hash:?}"))?;
    if bytes.len() != 32 {
        bail!("RECORD digest {hash:?} is not a sha256");
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)] // suite convention: unit__scenario__expected
    use std::fmt::Write as _;

    use super::*;

    fn b64(bytes: &[u8]) -> String {
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(Sha256::digest(bytes))
    }

    /// A venv with `entries` = (RECORD path, on-disk bytes or None for
    /// hashless-only) under one dist-info; the `TempDir` handle keeps the
    /// tree alive for the caller's assertions.
    fn venv_with(entries: &[(&str, Option<&[u8]>)]) -> (tempfile::TempDir, PathBuf) {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("eng");
        let site = dir.join("venv/lib/python3.12/site-packages");
        let dist = site.join("stub-1.0.dist-info");
        std::fs::create_dir_all(&dist).unwrap();
        let mut record = String::new();
        for (path, bytes) in entries {
            if let Some(bytes) = bytes {
                let file = site.join(path);
                std::fs::create_dir_all(file.parent().unwrap()).unwrap();
                std::fs::write(&file, bytes).unwrap();
                let _ = writeln!(record, "{path},sha256={},{}", b64(bytes), bytes.len());
            } else {
                let _ = writeln!(record, "{path},,");
            }
        }
        record.push_str("stub-1.0.dist-info/RECORD,,\n");
        std::fs::write(dist.join("RECORD"), record).unwrap();
        (tmp, dir)
    }

    #[test]
    fn unit__parse_record_line__plain_two_fields() {
        let (path, hash) = parse_record_line("pkg/a.py,sha256=abc,12").unwrap();
        assert_eq!(path, "pkg/a.py");
        assert_eq!(hash, "sha256=abc");
    }

    #[test]
    fn unit__parse_record_line__quoted_path_with_comma_and_quote() {
        let raw = "\"weird, \"\"quoted\"\".py\",sha256=abc,12";
        let (path, hash) = parse_record_line(raw).unwrap();
        assert_eq!(path, "weird, \"quoted\".py");
        assert_eq!(hash, "sha256=abc");
    }

    #[test]
    fn unit__parse_record_line__hashless_and_malformed() {
        let (path, hash) = parse_record_line("stub/RECORD,,").unwrap();
        assert_eq!(path, "stub/RECORD");
        assert_eq!(hash, "");
        assert!(parse_record_line("no-second-field").is_none());
        assert!(parse_record_line("\"unterminated,sha256=abc").is_none());
    }

    #[test]
    fn unit__verify_venv_record__happy_path_stable_digest() {
        let (_tmp, dir) = venv_with(&[
            ("pkg/a.py", Some(b"alpha\n".as_slice())),
            ("pkg/sub/b.py", Some(b"beta\n".as_slice())),
        ]);
        let digest = verify_venv_record(&dir).unwrap();
        assert!(digest.starts_with("pip:"));
        assert_eq!(digest.len(), "pip:".len() + 64);
        // Re-verify yields the identical digest.
        assert_eq!(verify_venv_record(&dir).unwrap(), digest);
    }

    #[test]
    fn unit__verify_venv_record__digest_independent_of_walk_order() {
        // Two venvs with the same (path, hash) set must agree no matter
        // which RECORD or file the walk reaches first.
        let (_tmp_a, dir_a) = venv_with(&[("pkg/a.py", Some(b"alpha\n".as_slice()))]);
        let (_tmp_b, dir_b) = venv_with(&[("pkg/a.py", Some(b"alpha\n".as_slice()))]);
        assert_eq!(
            verify_venv_record(&dir_a).unwrap(),
            verify_venv_record(&dir_b).unwrap()
        );
    }

    #[test]
    fn unit__verify_venv_record__hashless_lines_fingerprint_the_manifest() {
        let (_tmp_a, dir_a) = venv_with(&[
            ("pkg/a.py", Some(b"alpha\n".as_slice())),
            ("pkg/extra.txt", None),
        ]);
        let (_tmp_b, dir_b) = venv_with(&[("pkg/a.py", Some(b"alpha\n".as_slice()))]);
        // The hashless entry exists on disk in neither case, but only
        // one manifest lists it - the digests must differ.
        assert_ne!(
            verify_venv_record(&dir_a).unwrap(),
            verify_venv_record(&dir_b).unwrap()
        );
    }

    #[test]
    fn unit__verify_venv_record__tampered_file_names_path_and_digests() {
        let (_tmp, dir) = venv_with(&[("pkg/a.py", Some(b"alpha\n".as_slice()))]);
        std::fs::write(
            dir.join("venv/lib/python3.12/site-packages/pkg/a.py"),
            b"tampered\n",
        )
        .unwrap();
        let err = verify_venv_record(&dir).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("pkg/a.py"), "names the file: {msg}");
        assert!(msg.contains("drifted"), "names the drift: {msg}");
    }

    #[test]
    fn unit__verify_venv_record__missing_recorded_file_fails() {
        let (_tmp, dir) = venv_with(&[("pkg/a.py", Some(b"alpha\n".as_slice()))]);
        std::fs::remove_file(dir.join("venv/lib/python3.12/site-packages/pkg/a.py")).unwrap();
        let err = verify_venv_record(&dir).unwrap_err();
        assert!(
            format!("{err:#}").contains("pkg/a.py"),
            "names the missing file: {err:#}"
        );
    }

    #[test]
    fn unit__verify_venv_record__nested_vendor_record_is_ignored() {
        // setuptools vendors wheels under `<pkg>/_vendor/*.dist-info`;
        // vendoring copies the Python files but drops the wheel's scripts,
        // so those nested RECORDs list files this venv never had. Only
        // distributions installed directly into site-packages own their
        // RECORD - the nested ones must not fail the walk.
        let (_tmp, dir) = venv_with(&[("pkg/a.py", Some(b"alpha\n".as_slice()))]);
        let vendor_dist = dir
            .join("venv/lib/python3.12/site-packages/setuptools/_vendor")
            .join("vendored-1.0.dist-info");
        std::fs::create_dir_all(&vendor_dist).unwrap();
        std::fs::write(
            vendor_dist.join("RECORD"),
            format!("bin/tool,sha256={},5\n", b64(b"dropped-script")),
        )
        .unwrap();
        let digest = verify_venv_record(&dir).unwrap();
        // And the nested RECORD's phantom file stayed out of the fingerprint.
        let (_tmp2, dir2) = venv_with(&[("pkg/a.py", Some(b"alpha\n".as_slice()))]);
        assert_eq!(digest, verify_venv_record(&dir2).unwrap());
    }

    #[test]
    fn unit__verify_venv_record__engine_dir_without_venv_is_an_error() {
        let tmp = tempfile::tempdir().unwrap();
        let err = verify_venv_record(tmp.path()).unwrap_err();
        assert!(
            format!("{err:#}").contains("no venv"),
            "honest error, not a silent pass: {err:#}"
        );
    }

    #[test]
    fn unit__verify_venv_record__empty_venv_is_an_error() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("eng");
        std::fs::create_dir_all(dir.join("venv")).unwrap();
        let err = verify_venv_record(&dir).unwrap_err();
        assert!(
            format!("{err:#}").contains("no RECORD"),
            "a venv without manifests cannot be verified: {err:#}"
        );
    }

    #[test]
    fn unit__verify_venv_record__path_escaping_the_venv_is_refused() {
        let (_tmp, dir) = venv_with(&[]);
        let dist = dir.join("venv/lib/python3.12/site-packages/stub-1.0.dist-info");
        // Four `..` from site-packages lands OUTSIDE the venv (three is
        // the legitimate venv/bin hop); the target exists and matches
        // its digest so the failure is the containment guard alone.
        std::fs::create_dir_all(dir.join("bin")).unwrap();
        std::fs::write(dir.join("bin/python"), b"xxxxxxx").unwrap();
        let escaping = format!(
            "../../../../bin/python,sha256={},7\nstub-1.0.dist-info/RECORD,,\n",
            b64(b"xxxxxxx")
        );
        std::fs::write(dist.join("RECORD"), escaping).unwrap();
        let err = verify_venv_record(&dir).unwrap_err();
        assert!(
            format!("{err:#}").contains("outside the venv"),
            "containment guard fires: {err:#}"
        );
    }
}
