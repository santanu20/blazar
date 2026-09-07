//! `pallama upgrade` — self-update from GitHub Releases.
//!
//! Mirrors `scripts/install.sh` exactly: same asset naming
//! (`pallama-{tag}-{triple}.tar.gz|.zip`, flat root), same digest source
//! (the release API's `digest: sha256:…` field, verified by
//! `GhClient::download_asset_bytes` before anything touches disk), same
//! gnu-before-musl platform preference. `PALLAMA_INSTALL_BASE_URL` and
//! `PALLAMA_VERSION` behave as in the installer, which is what the tests
//! exercise against a fake release server.

use anyhow::{anyhow, bail, Context, Result};

use crate::engine::gh::{GhAsset, GhClient};

/// Resolve + verify + replace in one call. Returns the human summary.
///
/// `dry_run` resolves and downloads (verifying the digest) but replaces
/// nothing — it proves the whole chain except the final rename.
pub async fn run(client: &GhClient, repo: &str, version: Option<&str>, dry_run: bool) -> String {
    match run_inner(client, repo, version, dry_run).await {
        Ok(summary) => summary,
        Err(e) => format!("upgrade failed: {e:#}"),
    }
}

async fn run_inner(
    client: &GhClient,
    repo: &str,
    version: Option<&str>,
    dry_run: bool,
) -> Result<String> {
    let plan = resolve(client, repo, version).await?;
    let bytes = client.download_asset_bytes(&plan.asset).await?;
    let binary = extract_binary(&plan.asset.name, &bytes)?;
    if dry_run {
        return Ok(format!(
            "dry-run ok: {} -> asset {} verified ({} bytes extracted); rerun without --dry-run to install",
            plan.tag,
            plan.asset.name,
            binary.len()
        ));
    }
    let exe = replace_current_exe(&binary)?;
    Ok(format!(
        "upgraded to {} ({}); daemon note: a running daemon keeps the old binary until 'pallama stop' + any command",
        plan.tag,
        exe.display()
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
            format!("pallama-{tag}-{arch}-unknown-linux-gnu.tar.gz"),
            format!("pallama-{tag}-{arch}-unknown-linux-musl.tar.gz"),
        ],
        "macos" => vec![format!("pallama-{tag}-{arch}-apple-darwin.tar.gz")],
        "windows" => vec![format!("pallama-{tag}-{arch}-pc-windows-msvc.zip")],
        _ => Vec::new(),
    }
}

pub async fn resolve(client: &GhClient, repo: &str, version: Option<&str>) -> Result<UpgradePlan> {
    let release = client.release_by(repo, version).await?;
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
        "no pallama asset for this platform in {} (wanted one of {candidates:?}; available: {:?})",
        release.tag_name,
        release.assets.iter().map(|a| &a.name).collect::<Vec<_>>()
    ))
}

/// Extract the `pallama`/`pallama.exe` binary from a release archive.
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
                .and_then(|p| p.file_name().map(|f| f == "pallama"))
                .unwrap_or(false);
            if is_binary {
                let mut out = Vec::new();
                entry
                    .read_to_end(&mut out)
                    .context("read binary from tar")?;
                return Ok(out);
            }
        }
        bail!("no 'pallama' binary at the archive root");
    }
    let mut zip = zip::ZipArchive::new(std::io::Cursor::new(bytes.to_vec())).context("open zip")?;
    for i in 0..zip.len() {
        let mut file = zip.by_index(i).context("zip entry")?;
        if file.name() == "pallama.exe" {
            let mut out = Vec::new();
            file.read_to_end(&mut out).context("read binary from zip")?;
            return Ok(out);
        }
    }
    bail!("no 'pallama.exe' at the archive root");
}

/// Atomic self-replace. Unix: sibling temp + rename over the running exe
/// (the old inode stays alive for the current process; a running daemon
/// picks the new binary up on next start). Windows cannot rename over a
/// running exe — the new binary is parked next to it with instructions.
pub fn replace_current_exe(binary: &[u8]) -> Result<std::path::PathBuf> {
    let exe = std::env::current_exe().context("resolve current exe path")?;
    let staged = exe.with_extension("upgrade-new");
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
            "Windows cannot replace a running exe; staged binary at {} — stop pallama, then move it over {}",
            staged.display(),
            exe.display()
        );
    }
}
