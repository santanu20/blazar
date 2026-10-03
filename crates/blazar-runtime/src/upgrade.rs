//! `blazar upgrade` — self-update from GitHub Releases.
//!
//! Mirrors `scripts/install.sh` exactly: same asset naming
//! (`blazar-{tag}-{triple}.tar.gz|.zip`, flat root), same digest source
//! (the release API's `digest: sha256:…` field, verified by
//! `GhClient::download_asset_bytes` before anything touches disk), same
//! gnu-before-musl platform preference. `BLAZAR_INSTALL_BASE_URL` and
//! `BLAZAR_VERSION` behave as in the installer, which is what the tests
//! exercise against a fake release server.

use anyhow::{Context, Result, anyhow, bail};
use blazar_core::config::UpdateChannel;

use crate::engine::gh::{GhAsset, GhClient};

/// "v1.2.3"/"1.2.3" -> (1,2,3); anything else -> None (exotic tags never
/// produce a false downgrade claim).
fn semver_triple(tag: &str) -> Option<(u64, u64, u64)> {
    let t = tag.strip_prefix('v').unwrap_or(tag);
    let mut parts = t.split('.');
    let triple = (
        parts.next()?.parse().ok()?,
        parts.next()?.parse().ok()?,
        parts.next()?.parse().ok()?,
    );
    parts.next().is_none().then_some(triple)
}

/// "" when the target is not older than `current`; otherwise a downgrade
/// callout so a typo'd pin never silently rolls the binary back.
fn downgrade_note(tag: &str, current: &str) -> String {
    let older = semver_triple(tag)
        .zip(semver_triple(current))
        .is_some_and(|(target, running)| target < running);
    if older {
        format!(" — downgrade from {current} (rollback path)")
    } else {
        String::new()
    }
}

/// Resolve + verify + replace in one call. Returns the human summary.
///
/// `dry_run` resolves and downloads (verifying the digest) but replaces
/// nothing — it proves the whole chain except the final rename.
pub async fn run(
    client: &GhClient,
    repo: &str,
    version: Option<&str>,
    channel: UpdateChannel,
    dry_run: bool,
    current: &str,
) -> String {
    match run_inner(client, repo, version, channel, dry_run, current).await {
        Ok(summary) => summary,
        Err(e) => format!("upgrade failed: {e:#}"),
    }
}

async fn run_inner(
    client: &GhClient,
    repo: &str,
    version: Option<&str>,
    channel: UpdateChannel,
    dry_run: bool,
    current: &str,
) -> Result<String> {
    let plan = resolve(client, repo, version, channel).await?;
    let bytes = client.download_asset_bytes(&plan.asset).await?;
    let binary = extract_binary(&plan.asset.name, &bytes)?;
    let note = downgrade_note(&plan.tag, current);
    if dry_run {
        return Ok(format!(
            "dry-run ok: {} -> asset {} verified ({} bytes extracted){}; rerun without --dry-run to install",
            plan.tag,
            plan.asset.name,
            binary.len(),
            note
        ));
    }
    let exe = replace_current_exe(&binary)?;
    Ok(format!(
        "upgraded to {} ({}){}; daemon note: the CLI restarts a running daemon onto the new binary",
        plan.tag,
        exe.display(),
        note
    ))
}

pub struct UpgradePlan {
    pub tag: String,
    pub asset: GhAsset,
}

/// Asset names for this platform in preference order (gnu before musl so
#[must_use]
pub fn preferred_assets(tag: &str) -> Vec<String> {
    let arch = std::env::consts::ARCH;
    match std::env::consts::OS {
        "linux" => vec![
            format!("blazar-{tag}-{arch}-unknown-linux-gnu.tar.gz"),
            format!("blazar-{tag}-{arch}-unknown-linux-musl.tar.gz"),
        ],
        "macos" => vec![format!("blazar-{tag}-{arch}-apple-darwin.tar.gz")],
        "windows" => vec![format!("blazar-{tag}-{arch}-pc-windows-msvc.zip")],
        _ => Vec::new(),
    }
}

pub async fn resolve(
    client: &GhClient,
    repo: &str,
    version: Option<&str>,
    channel: UpdateChannel,
) -> Result<UpgradePlan> {
    // Explicit --version overrides the channel; otherwise the channel
    // picks the target (stable = GitHub's releases/latest, latest =
    // newest release including prereleases).
    let release = match version {
        Some(v) => client.release_by(repo, Some(v)).await?,
        None => client.channel_repo_release(repo, channel).await?,
    };
    let candidates = preferred_assets(&release.tag_name);
    if candidates.is_empty() {
        bail!("unsupported platform for self-update");
    }
    if let Some(asset) = release.assets.iter().find(|a| candidates.contains(&a.name)) {
        return Ok(UpgradePlan {
            tag: release.tag_name,
            asset: asset.clone(),
        });
    }
    Err(anyhow!(
        "no blazar asset for this platform in {} (wanted one of {candidates:?}; available: {:?})",
        release.tag_name,
        release.assets.iter().map(|a| &a.name).collect::<Vec<_>>()
    ))
}

/// Extract the `blazar`/`blazar.exe` binary from a release archive.
pub fn extract_binary(asset_name: &str, bytes: &[u8]) -> Result<Vec<u8>> {
    use std::io::Read as _;
    if asset_name.ends_with(".tar.gz") {
        let decoder = flate2::read::GzDecoder::new(bytes);
        let mut archive = tar::Archive::new(decoder);
        for entry in archive.entries().context("read tar entries")? {
            let mut entry = entry.context("tar entry")?;
            let is_binary = entry
                .path()
                .ok()
                .and_then(|p| p.file_name().map(|f| f == "blazar"))
                .unwrap_or(false);
            if is_binary {
                let mut out = Vec::new();
                entry
                    .read_to_end(&mut out)
                    .context("read binary from tar")?;
                return Ok(out);
            }
        }
        bail!("no 'blazar' binary at the archive root");
    }
    let mut zip = zip::ZipArchive::new(std::io::Cursor::new(bytes.to_vec())).context("open zip")?;
    for i in 0..zip.len() {
        let mut file = zip.by_index(i).context("zip entry")?;
        if file.name() == "blazar.exe" {
            let mut out = Vec::new();
            file.read_to_end(&mut out).context("read binary from zip")?;
            return Ok(out);
        }
    }
    bail!("no 'blazar.exe' at the archive root");
}

/// Atomic self-replace. Unix: sibling temp + rename over the running exe
/// (the old inode stays alive for the current process; a running daemon
/// picks the new binary up on next start). Windows cannot rename over a
/// running exe — the new binary is parked next to it with instructions.
pub fn replace_current_exe(binary: &[u8]) -> Result<std::path::PathBuf> {
    let exe = std::env::current_exe().context("resolve current exe path")?;
    // Append instead of with_extension: a versioned binary name
    // (blazar-v2.1) would have its ".1" swapped for the suffix (F106).
    let mut staged_name = std::ffi::OsString::from(exe.as_os_str());
    staged_name.push(".upgrade-new");
    let staged = std::path::PathBuf::from(staged_name);
    std::fs::write(&staged, binary).with_context(|| format!("write {}", staged.display()))?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        std::fs::set_permissions(&staged, std::fs::Permissions::from_mode(0o755))
            .context("chmod 755 staged binary")?;
        std::fs::rename(&staged, &exe).with_context(|| format!("replace {}", exe.display()))?;
        Ok(exe)
    }

    #[cfg(windows)]
    {
        let _ = &exe;
        bail!(
            "Windows cannot replace a running exe; staged binary at {} — stop blazar, then move it over {}",
            staged.display(),
            exe.display()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::{downgrade_note, semver_triple};

    #[test]
    #[allow(non_snake_case)] // suite convention: unit__scenario__expected
    fn unit__semver_triple__v_prefixed_bare_and_rejected_shapes() {
        assert_eq!(semver_triple("v0.14.0"), Some((0, 14, 0)));
        assert_eq!(semver_triple("0.14.0"), Some((0, 14, 0)));
        assert_eq!(semver_triple("v1.2.3"), Some((1, 2, 3)));
        // Exotic tags must never yield a false downgrade claim.
        assert_eq!(semver_triple("b10941"), None);
        assert_eq!(semver_triple("latest"), None);
        assert_eq!(semver_triple("v1.2.3-rc.1"), None);
        assert_eq!(semver_triple(""), None);
    }

    #[test]
    #[allow(non_snake_case)] // suite convention: unit__scenario__expected
    fn unit__downgrade_note__older_target_flags_rollback_only() {
        assert_eq!(
            downgrade_note("v0.13.0", "0.14.0"),
            " — downgrade from 0.14.0 (rollback path)"
        );
        // Newer, equal, and unparseable targets stay silent.
        assert_eq!(downgrade_note("v0.15.0", "0.14.0"), "");
        assert_eq!(downgrade_note("v0.14.0", "0.14.0"), "");
        assert_eq!(downgrade_note("b10941", "0.14.0"), "");
        assert_eq!(downgrade_note("v0.13.0", "b10941"), "");
    }
}
