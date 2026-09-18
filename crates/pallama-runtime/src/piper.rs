//! Piper TTS lane: offline text-to-speech through the piper CLI
//! (rhasspy/piper releases), voices from `rhasspy/piper-voices` on HF.
//!
//! Unlike whisper-server, piper is a one-shot binary (stdin text, WAV on
//! stdout via `--output_file -`), so this lane spawns per synthesis
//! instead of holding a child: no runtime state, nothing to tear down.
//! The install/pin/prune rails mirror the whisper lane exactly.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use tokio::io::AsyncWriteExt as _;

use crate::engine::gh::{GhClient, GhRelease};
use pallama_core::PallamaDirs;

pub const PIPER_REPO: &str = "rhasspy/piper";
pub const VOICES_REPO: &str = "rhasspy/piper-voices";

/// Release asset for the running platform. Upstream ships six; anything
/// else (freebsd, ...) has no binary lane.
#[must_use]
pub fn asset_name(os: &str, arch: &str) -> Option<&'static str> {
    match (os, arch) {
        ("linux", "x86_64") => Some("piper_linux_x86_64.tar.gz"),
        ("linux", "aarch64") => Some("piper_linux_aarch64.tar.gz"),
        ("linux", "arm") => Some("piper_linux_armv7l.tar.gz"),
        ("macos", "x86_64") => Some("piper_macos_x64.tar.gz"),
        ("macos", "aarch64") => Some("piper_macos_aarch64.tar.gz"),
        ("windows", "x86_64") => Some("piper_windows_amd64.zip"),
        _ => None,
    }
}

fn bin_root(dirs: &PallamaDirs) -> PathBuf {
    dirs.data_dir.join("piper")
}

#[must_use]
pub fn voices_dir(dirs: &PallamaDirs) -> PathBuf {
    dirs.data_dir.join("voices")
}

fn pin_path(dirs: &PallamaDirs) -> PathBuf {
    bin_root(dirs).join("pin")
}

fn valid_tag(tag: &str) -> bool {
    !tag.is_empty()
        && !tag.contains(std::path::MAIN_SEPARATOR)
        && !tag.contains('/')
        && !tag.contains('\\')
        && !tag.contains("..")
}

/// Sort key for a piper tag: the date shape `YYYY.MM.DD-N` (upstream
/// has never shipped another form). Non-conforming tags order after
/// date tags, alphabetically — newest installed still wins sanely.
type TagKey = (u64, u64, u64, u64);

fn tag_key(tag: &str) -> Option<TagKey> {
    let mut parts = [0u64; 4];
    let mut saw = 0usize;
    for piece in tag.split(['.', '-']) {
        if saw == 4 {
            return None;
        }
        match piece.parse::<u64>() {
            Ok(n) => parts[saw] = n,
            Err(_) => return None,
        }
        saw += 1;
    }
    (saw > 0).then_some((parts[0], parts[1], parts[2], parts[3]))
}

type TagDirEntry = (Option<TagKey>, String, PathBuf);

fn sorted_tag_dirs(dirs: &PallamaDirs) -> Vec<PathBuf> {
    let mut dirs: Vec<TagDirEntry> = std::fs::read_dir(bin_root(dirs))
        .map(|rd| {
            rd.flatten()
                .filter(|e| e.path().is_dir())
                .filter_map(|e| {
                    let name = e.file_name().to_string_lossy().into_owned();
                    if name == "pin" {
                        return None;
                    }
                    Some((tag_key(&name), name.clone(), e.path()))
                })
                .collect()
        })
        .unwrap_or_default();
    // Newest first: date-keyed tags by key, shapeless tags after them
    // (alphabetical) so a stray dir never shadows a real release.
    dirs.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.cmp(&a.1)));
    dirs.into_iter().map(|(_, _, p)| p).collect()
}

#[must_use]
pub fn pinned_tag(dirs: &PallamaDirs) -> Option<String> {
    let raw = std::fs::read_to_string(pin_path(dirs)).ok()?;
    let t = raw.trim().to_string();
    (!t.is_empty()).then_some(t)
}

#[must_use]
pub fn installed_tags(dirs: &PallamaDirs) -> Vec<String> {
    sorted_tag_dirs(dirs)
        .into_iter()
        .filter_map(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
        .collect()
}

fn piper_bin_in(dir: &Path) -> Option<PathBuf> {
    let name = if cfg!(windows) { "piper.exe" } else { "piper" };
    crate::whisper::walk_for_file(dir, name)
}

/// Newest release that actually ships `asset` (GitHub lists
/// newest-first). Piper tags carry their assets on every release, but
/// the guard keeps the lane honest if that ever changes.
fn newest_with_asset<'a>(releases: &'a [GhRelease], asset: &str) -> Option<&'a GhRelease> {
    releases
        .iter()
        .find(|r| r.assets.iter().any(|a| a.name == asset))
}

async fn release_for_channel(gh: &GhClient) -> Result<GhRelease> {
    let asset_name = asset_name(std::env::consts::OS, std::env::consts::ARCH).ok_or_else(|| {
        anyhow!(
            "piper releases ship no {}/{} binary — see \
                 https://github.com/rhasspy/piper for source builds",
            std::env::consts::OS,
            std::env::consts::ARCH
        )
    })?;
    let releases = gh.list_releases_repo(PIPER_REPO).await?;
    newest_with_asset(&releases, asset_name)
        .cloned()
        .ok_or_else(|| {
            anyhow!(
                "no {PIPER_REPO} release ships the {asset_name} asset \
                 ({} checked) — upstream may have renamed assets",
                releases.len()
            )
        })
}

/// Download + extract a piper release. `Some(tag)` installs that release
/// (`pin = true` also pins it); `None` installs the newest and returns to
/// tracking (clears any pin). Old tags prune to `KEEP_TAGS` (pinned
/// always kept). Returns the installed tag.
pub async fn install(
    gh: &GhClient,
    dirs: &PallamaDirs,
    tag: Option<&str>,
    pin: bool,
) -> Result<String> {
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    let Some(asset_name) = asset_name(os, arch) else {
        return Err(anyhow!(
            "piper releases ship no {os}/{arch} binary — see \
             https://github.com/rhasspy/piper for source builds"
        ));
    };
    let release = match tag {
        Some(t) => gh.release_by(PIPER_REPO, Some(t)).await?,
        None => release_for_channel(gh).await?,
    };
    let asset = release
        .assets
        .iter()
        .find(|a| a.name == asset_name)
        .ok_or_else(|| anyhow!("release {} has no asset {asset_name}", release.tag_name))?;
    let bytes = gh.download_asset_bytes(asset).await?;
    let dir = bin_root(dirs).join(&release.tag_name);
    // Replace, don't merge — a re-install over an existing tag dir must
    // not leave stale binaries from the old extract behind.
    if dir.exists() {
        std::fs::remove_dir_all(&dir).with_context(|| format!("replace {}", dir.display()))?;
    }
    std::fs::create_dir_all(&dir).with_context(|| format!("mkdir {}", dir.display()))?;
    crate::engine::extract_archive(&bytes, &dir, &asset.name)?;
    piper_bin_in(&dir).ok_or_else(|| {
        anyhow!(
            "extracted {} but no piper binary found under {}",
            asset.name,
            dir.display()
        )
    })?;
    match (tag, pin) {
        (Some(_), true) => {
            std::fs::write(pin_path(dirs), format!("{}\n", release.tag_name))
                .with_context(|| format!("write pin {}", pin_path(dirs).display()))?;
        }
        (Some(_), false) => {}
        (None, _) => {
            let _ = std::fs::remove_file(pin_path(dirs));
        }
    }
    prune(dirs)?;
    Ok(release.tag_name)
}

/// Pin management, mirroring the whisper lane: pin an installed tag or
/// `None` to clear; prune re-runs so a formerly-protected old tag drops.
pub fn set_pin(dirs: &PallamaDirs, tag: Option<&str>) -> Result<()> {
    match tag {
        Some(t) => {
            let t = t.trim();
            if !valid_tag(t) {
                anyhow::bail!("invalid tag {t:?}: must be a plain tag name (no path separators)");
            }
            if !installed_tags(dirs).iter().any(|installed| installed == t) {
                anyhow::bail!(
                    "tag {t} is not installed (installed: {}) — run: pallama tts --install --tag {t}",
                    installed_tags(dirs).join(", ")
                );
            }
            std::fs::write(pin_path(dirs), format!("{t}\n"))
                .with_context(|| format!("write pin {}", pin_path(dirs).display()))?;
        }
        None => {
            let _ = std::fs::remove_file(pin_path(dirs));
        }
    }
    prune(dirs)?;
    Ok(())
}

fn prune(dirs: &PallamaDirs) -> Result<()> {
    let pin = pinned_tag(dirs);
    for dir in sorted_tag_dirs(dirs)
        .into_iter()
        .skip(crate::engine::KEEP_TAGS)
    {
        let name = dir
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        if Some(&name) == pin.as_ref() {
            continue;
        }
        std::fs::remove_dir_all(&dir)
            .with_context(|| format!("prune piper dir {}", dir.display()))?;
        tracing::info!("pruned old piper {name}");
    }
    Ok(())
}

/// Active piper binary plus its directory. The directory is load-bearing
/// twice on Linux: `LD_LIBRARY_PATH` (the binary links sibling
/// libonnxruntime/libespeak-ng) and the bundled `espeak-ng-data` passed
/// as `--espeak_data` (piper resolves it relative to cwd otherwise).
#[must_use]
pub fn server_bin(dirs: &PallamaDirs) -> Option<(PathBuf, PathBuf)> {
    if let Some(tag) = pinned_tag(dirs) {
        let dir = bin_root(dirs).join(&tag);
        if let Some((bin, lib)) = bin_and_lib_in(&dir) {
            return Some((bin, lib));
        }
        tracing::warn!("piper pin {tag} has no binary; using newest installed tag");
    }
    for tag_dir in sorted_tag_dirs(dirs) {
        if let Some((bin, lib)) = bin_and_lib_in(&tag_dir) {
            return Some((bin, lib));
        }
    }
    None
}

/// Binary plus its own directory (libs + espeak-ng-data live beside it).
fn bin_and_lib_in(dir: &Path) -> Option<(PathBuf, PathBuf)> {
    let bin = piper_bin_in(dir)?;
    let lib = bin
        .parent()
        .map_or_else(|| dir.to_path_buf(), Path::to_path_buf);
    Some((bin, lib))
}

/// HF path stem for a piper voice id (`<locale>-<name>-<quality>`, e.g.
/// `en_US-amy-medium` → `en/en_US/amy/medium/en_US-amy-medium`).
/// `None` on names that don't carry the three-part shape.
#[must_use]
pub fn voice_repo_path(voice: &str) -> Option<String> {
    let (locale, name, quality) = parse_voice_id(voice)?;
    let lang = locale.split('_').next()?;
    Some(format!("{lang}/{locale}/{name}/{quality}/{voice}"))
}

/// Voice id → (locale, name, quality). Piper names use `_`, never `-`
/// (en_GB-northern_english_male-medium), so an rsplitn(3, '-') is
/// unambiguous.
fn parse_voice_id(voice: &str) -> Option<(String, String, String)> {
    let mut parts = voice.rsplitn(3, '-');
    let quality = parts.next()?.to_string();
    let name = parts.next()?.to_string();
    let locale = parts.next()?.to_string();
    if locale.is_empty() || name.is_empty() || quality.is_empty() || !locale.contains('_') {
        return None;
    }
    Some((locale, name, quality))
}

/// Installed voices: directories under `data/voices` carrying both the
/// `.onnx` and `.onnx.json` halves.
#[must_use]
pub fn list_voices(dirs: &PallamaDirs) -> Vec<String> {
    let mut out: Vec<String> = std::fs::read_dir(voices_dir(dirs))
        .map(|rd| {
            rd.flatten()
                .filter(|e| e.path().is_dir())
                .filter(|e| voice_file(dirs, &e.file_name().to_string_lossy()).is_some())
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_default();
    out.sort();
    out
}

/// Both halves of an installed voice, if present.
#[must_use]
pub fn voice_file(dirs: &PallamaDirs, voice: &str) -> Option<(PathBuf, PathBuf)> {
    let dir = voices_dir(dirs).join(voice);
    let onnx = dir.join(format!("{voice}.onnx"));
    let json = dir.join(format!("{voice}.onnx.json"));
    (onnx.is_file() && json.is_file()).then_some((onnx, json))
}

fn valid_voice(voice: &str) -> bool {
    !voice.is_empty()
        && voice
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

/// Download a voice (`.onnx` + `.onnx.json`) from
/// `rhasspy/piper-voices`. Fails fast on a non-onnx payload instead of
/// installing a corrupt voice.
pub async fn pull_voice(
    hf: &crate::hf::HfClient,
    dirs: &PallamaDirs,
    voice: &str,
    on_progress: impl FnMut(u64, u64),
) -> Result<PathBuf> {
    if !valid_voice(voice) {
        return Err(anyhow!(
            "invalid voice {voice:?}: expected a piper voice id like en_US-amy-medium"
        ));
    }
    let Some(stem) = voice_repo_path(voice) else {
        return Err(anyhow!(
            "invalid voice {voice:?}: piper voice ids are <locale>-<name>-<quality>, \
             e.g. en_US-amy-medium"
        ));
    };
    let dir = voices_dir(dirs).join(voice);
    std::fs::create_dir_all(&dir)?;
    let dest = dir.join(format!("{voice}.onnx"));
    let mut on_progress = on_progress;
    {
        let plan = crate::hf::FilePlan {
            filename: format!("{stem}.onnx"),
            bytes: 0,
            sha256: None,
        };
        hf.download_file(VOICES_REPO, &plan, &dest, &mut on_progress)
            .await
            .with_context(|| format!("download {VOICES_REPO}/{voice}.onnx"))?;
    }
    {
        let json_dest = dir.join(format!("{voice}.onnx.json"));
        let plan = crate::hf::FilePlan {
            filename: format!("{stem}.onnx.json"),
            bytes: 0,
            sha256: None,
        };
        let mut json_progress = |_, _| {};
        hf.download_file(VOICES_REPO, &plan, &json_dest, &mut json_progress)
            .await
            .with_context(|| format!("download {VOICES_REPO}/{voice}.onnx.json"))?;
    }
    // Sanity floor: a real voice is tens of MB; a truncated error page
    // is not. The config half must parse as JSON.
    let onnx_len = std::fs::metadata(&dest).map_or(0, |m| m.len());
    if onnx_len < 1_000_000 {
        let _ = std::fs::remove_dir_all(&dir);
        return Err(anyhow!(
            "{VOICES_REPO}/{voice}: payload is not a piper voice ({onnx_len} bytes, expected a \
             multi-MB onnx) — removed"
        ));
    }
    let json_raw = std::fs::read_to_string(dir.join(format!("{voice}.onnx.json")))
        .with_context(|| format!("read {voice} config"))?;
    if serde_json::from_str::<serde_json::Value>(&json_raw).is_err() {
        let _ = std::fs::remove_dir_all(&dir);
        return Err(anyhow!(
            "{VOICES_REPO}/{voice}: config half is not JSON — removed"
        ));
    }
    Ok(dest)
}

/// Upper bound on synthesis input: piper is linear-time; a novel would
/// hold the request for minutes and balloon the WAV into RAM.
pub const MAX_INPUT_CHARS: usize = 10_000;

/// Synthesize `text` to a full WAV file (RIFF bytes) with the installed
/// binary and voice. `speed` follows the `OpenAI` contract (1.0 = native,
/// 2.0 = twice as fast) and maps to piper's inverse `--length_scale`.
/// Spawn-per-call: piper is a one-shot binary (no server, nothing to
/// pool), cold start ~1 s.
pub async fn synthesize(
    dirs: &PallamaDirs,
    voice: &str,
    text: &str,
    speed: Option<f64>,
    timeout: std::time::Duration,
) -> Result<Vec<u8>> {
    if text.trim().is_empty() {
        return Err(anyhow!("input text is empty"));
    }
    if text.chars().count() > MAX_INPUT_CHARS {
        return Err(anyhow!(
            "input text is {} chars (max {MAX_INPUT_CHARS}) — split long documents",
            text.chars().count()
        ));
    }
    let Some(speed) = speed else {
        return synthesize_inner(dirs, voice, text, None, timeout).await;
    };
    if !(0.25..=4.0).contains(&speed) {
        return Err(anyhow!(
            "speed {speed} out of range (0.25..=4.0, OpenAI contract)"
        ));
    }
    synthesize_inner(dirs, voice, text, Some(1.0 / speed), timeout).await
}

async fn synthesize_inner(
    dirs: &PallamaDirs,
    voice: &str,
    text: &str,
    length_scale: Option<f64>,
    timeout: std::time::Duration,
) -> Result<Vec<u8>> {
    let Some((bin, lib_dir)) = server_bin(dirs) else {
        return Err(anyhow!(
            "piper is not installed — run: pallama tts --install"
        ));
    };
    let Some((onnx, _json)) = voice_file(dirs, voice) else {
        let installed = list_voices(dirs);
        return Err(anyhow!(
            "voice {voice} is not pulled{} — run: pallama tts --pull {voice}",
            if installed.is_empty() {
                String::new()
            } else {
                format!(" (installed: {})", installed.join(", "))
            }
        ));
    };
    let espeak_data = lib_dir.join("espeak-ng-data");
    let mut cmd = tokio::process::Command::new(&bin);
    cmd.arg("-q")
        .arg("--model")
        .arg(&onnx)
        .arg("--output_file")
        .arg("-")
        .arg("--espeak_data")
        .arg(&espeak_data)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    if let Some(scale) = length_scale {
        cmd.arg("--length_scale").arg(format!("{scale:.4}"));
    }
    if cfg!(unix) {
        // The tarball binary links sibling .so files in place.
        cmd.env("LD_LIBRARY_PATH", &lib_dir);
    }
    // Timeout safety: on expiry the dropped child is killed instead of
    // orphaning a piper that keeps synthesizing a novel.
    cmd.kill_on_drop(true);
    let mut child = cmd
        .spawn()
        .with_context(|| format!("spawn {}", bin.display()))?;
    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(text.as_bytes()).await;
        let _ = stdin.shutdown().await;
    }
    let out = tokio::time::timeout(timeout, child.wait_with_output()).await;
    match out {
        Ok(Ok(output)) if output.status.success() => {
            if !output.stdout.starts_with(b"RIFF") {
                let tail = String::from_utf8_lossy(&output.stderr);
                let tail = tail.lines().last().unwrap_or("").trim();
                return Err(anyhow!(
                    "piper produced no WAV ({} bytes stdout{})",
                    output.stdout.len(),
                    if tail.is_empty() {
                        String::new()
                    } else {
                        format!("; stderr: {tail}")
                    }
                ));
            }
            Ok(output.stdout)
        }
        Ok(Ok(output)) => {
            let tail = String::from_utf8_lossy(&output.stderr);
            let tail = tail.lines().last().unwrap_or("").trim();
            Err(anyhow!(
                "piper exited {}{}",
                output.status,
                if tail.is_empty() {
                    String::new()
                } else {
                    format!(": {tail}")
                }
            ))
        }
        Ok(Err(e)) => Err(anyhow!("piper wait failed: {e:#}")),
        Err(_) => {
            // kill_on_drop already reaped the child when the future was
            // dropped by the timeout.
            Err(anyhow!(
                "piper synthesis timed out after {} s (input {} chars)",
                timeout.as_secs(),
                text.chars().count()
            ))
        }
    }
}

#[cfg(test)]
#[allow(non_snake_case)] // scenario-style test names match the repo convention
mod tests {
    use super::*;

    #[test]
    fn unit__asset_name__six_platforms() {
        assert_eq!(
            asset_name("linux", "x86_64"),
            Some("piper_linux_x86_64.tar.gz")
        );
        assert_eq!(
            asset_name("linux", "aarch64"),
            Some("piper_linux_aarch64.tar.gz")
        );
        assert_eq!(
            asset_name("linux", "arm"),
            Some("piper_linux_armv7l.tar.gz")
        );
        assert_eq!(
            asset_name("macos", "x86_64"),
            Some("piper_macos_x64.tar.gz")
        );
        assert_eq!(
            asset_name("macos", "aarch64"),
            Some("piper_macos_aarch64.tar.gz")
        );
        assert_eq!(
            asset_name("windows", "x86_64"),
            Some("piper_windows_amd64.zip")
        );
        assert_eq!(asset_name("freebsd", "x86_64"), None);
    }

    #[test]
    fn unit__voice_repo_path__three_part_shape() {
        assert_eq!(
            voice_repo_path("en_US-amy-medium").as_deref(),
            Some("en/en_US/amy/medium/en_US-amy-medium")
        );
        // Underscored names survive; locale keeps its region.
        assert_eq!(
            voice_repo_path("en_GB-northern_english_male-medium").as_deref(),
            Some("en/en_GB/northern_english_male/medium/en_GB-northern_english_male-medium")
        );
        assert_eq!(
            voice_repo_path("pt_BR-faber-medium").as_deref(),
            Some("pt/pt_BR/faber/medium/pt_BR-faber-medium")
        );
        // Malformed: no quality / no locale region / too few parts.
        assert_eq!(voice_repo_path("en_US-amy"), None);
        assert_eq!(voice_repo_path("amy-medium"), None);
        assert_eq!(voice_repo_path(""), None);
    }

    #[test]
    fn unit__tag_key__date_shape_orders_newest_first() {
        assert_eq!(tag_key("2023.11.14-2"), Some((2023, 11, 14, 2)));
        assert_eq!(tag_key("2024.1.2"), Some((2024, 1, 2, 0)));
        assert!(tag_key("2023.11.14-2") > tag_key("2023.11.14-1"));
        assert!(tag_key("nightly").is_none());
    }

    #[test]
    fn unit__synthesize__validates_before_spawning() {
        let dirs = PallamaDirs {
            config_dir: std::env::temp_dir().join("pallama-piper-test-cfg"),
            data_dir: std::env::temp_dir().join("pallama-piper-test-data"),
        };
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let e = rt
            .block_on(synthesize(
                &dirs,
                "en_US-amy-medium",
                "  ",
                None,
                std::time::Duration::from_secs(5),
            ))
            .unwrap_err();
        assert!(e.to_string().contains("empty"), "{e}");
        let e = rt
            .block_on(synthesize(
                &dirs,
                "en_US-amy-medium",
                "hi",
                Some(9.0),
                std::time::Duration::from_secs(5),
            ))
            .unwrap_err();
        assert!(e.to_string().contains("0.25..=4.0"), "{e}");
        let long = "x".repeat(MAX_INPUT_CHARS + 1);
        let e = rt
            .block_on(synthesize(
                &dirs,
                "en_US-amy-medium",
                &long,
                None,
                std::time::Duration::from_secs(5),
            ))
            .unwrap_err();
        assert!(e.to_string().contains("split long documents"), "{e}");
        // Valid shape but nothing installed: the teaching error names
        // the install step, not a bare spawn failure.
        let e = rt
            .block_on(synthesize(
                &dirs,
                "en_US-amy-medium",
                "hi",
                None,
                std::time::Duration::from_secs(5),
            ))
            .unwrap_err();
        assert!(e.to_string().contains("pallama tts --install"), "{e}");
    }
}
