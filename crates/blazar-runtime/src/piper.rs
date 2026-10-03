//! Piper TTS lane: offline text-to-speech through the piper CLI
//! (rhasspy/piper releases), voices from `rhasspy/piper-voices` on HF.
//!
//! Unlike whisper-server, piper is a one-shot binary (stdin text, WAV on
//! stdout via `--output_file -`), so this lane spawns per synthesis
//! instead of holding a child: no runtime state, nothing to tear down.
//! Installs ride the engines lane (`blazar engine install --kind piper`
//! / `blazar tts --install`) with rows, manifests, prune, and rollback
//! like every other kind; a legacy `data/piper` tree predating that
//! lane stays a read-only serving source until adopted.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use tokio::io::AsyncWriteExt as _;

use blazar_core::BlazarDirs;

pub use crate::engine::gh::PIPER_REPO;
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

fn bin_root(dirs: &BlazarDirs) -> PathBuf {
    dirs.data_dir.join("piper")
}

#[must_use]
pub fn voices_dir(dirs: &BlazarDirs) -> PathBuf {
    dirs.data_dir.join("voices")
}

fn pin_path(dirs: &BlazarDirs) -> PathBuf {
    bin_root(dirs).join("pin")
}

/// Pin-file location from the bare data dir — engine-removal
/// reconciliation (`engine rm` of a pinned tag) runs from the removal
/// site, which owns the data dir but not a full `BlazarDirs`.
#[must_use]
pub fn pin_path_in(data_dir: &Path) -> PathBuf {
    data_dir.join("piper").join("pin")
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
/// `pub` mirrors `gh::btag_number`: the CLI currency lane ranks tags
/// with it.
pub type TagKey = (u64, u64, u64, u64);

#[must_use]
pub fn tag_key(tag: &str) -> Option<TagKey> {
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

/// Installed tag dirs, newest first — the LEGACY `data/piper` tree
/// only. `pub(crate)`: the engines lane's legacy-tree adoption walks
/// these same dirs, newest first (mirror of the whisper lane).
pub(crate) fn sorted_tag_dirs(dirs: &BlazarDirs) -> Vec<PathBuf> {
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
pub fn pinned_tag(dirs: &BlazarDirs) -> Option<String> {
    let raw = std::fs::read_to_string(pin_path(dirs)).ok()?;
    let t = raw.trim().to_string();
    (!t.is_empty()).then_some(t)
}

#[must_use]
pub fn installed_tags(dirs: &BlazarDirs) -> Vec<String> {
    sorted_tag_dirs(dirs)
        .into_iter()
        .filter_map(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
        .collect()
}

fn piper_bin_in(dir: &Path) -> Option<PathBuf> {
    let name = if cfg!(windows) { "piper.exe" } else { "piper" };
    crate::whisper::walk_for_file(dir, name)
}

/// Pin management, mirroring the whisper lane: pin an installed tag or
/// `None` to clear; prune re-runs so a formerly-protected old tag drops.
/// The tag may live in either lane — engines rows or the legacy tree —
/// because the user has ONE `--pin` flag and cannot be expected to know
/// which channel holds the tag.
pub fn set_pin(dirs: &BlazarDirs, tag: Option<&str>) -> Result<()> {
    match tag {
        Some(t) => {
            let t = t.trim();
            if !valid_tag(t) {
                anyhow::bail!("invalid tag {t:?}: must be a plain tag name (no path separators)");
            }
            let mut known = installed_tags(dirs);
            known.extend(piper_engine_tags(dirs)?);
            known.sort();
            known.dedup();
            if !known.iter().any(|installed| installed == t) {
                anyhow::bail!(
                    "tag {t} is not installed (installed: {}) — run: blazar tts --install --tag {t}",
                    known.join(", ")
                );
            }
            // The engines lane never creates the legacy pin dir — a
            // fresh install has no `data/piper` tree at all — so the
            // pin site mkdirs its own home before the write.
            std::fs::create_dir_all(bin_root(dirs))
                .with_context(|| format!("mkdir {}", bin_root(dirs).display()))?;
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

/// Piper-kind engine row tags (the engines lane's install set), for
/// `--pin` validation. Unlike the lane lookups this propagates a store
/// failure: refusing a valid pin because the store could not be read
/// would be a silent no-op of the user's explicit intent.
fn piper_engine_tags(dirs: &BlazarDirs) -> Result<Vec<String>> {
    let store =
        blazar_core::Store::open(dirs).with_context(|| "open store to list piper engine tags")?;
    let tags = store
        .list_engines()?
        .into_iter()
        .filter(|r| r.kind == blazar_core::engine_kind::EngineKind::Piper)
        .map(|r| r.tag)
        .collect();
    Ok(tags)
}

fn prune(dirs: &BlazarDirs) -> Result<()> {
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
        match blazar_core::fs_safety::remove_dir_within(&dirs.data_dir, &dir) {
            Ok(
                blazar_core::fs_safety::GuardedRemoval::Removed
                | blazar_core::fs_safety::GuardedRemoval::Absent,
            ) => {}
            Ok(blazar_core::fs_safety::GuardedRemoval::Escaped) => {
                // Not this store's dir — skip, keep the pass moving.
                continue;
            }
            Err(e) => return Err(anyhow!("prune piper dir {}: {e}", dir.display())),
        }
        tracing::info!("pruned old piper {name}");
    }
    Ok(())
}

/// Active piper binary plus its directory. The directory is load-bearing
/// twice on Linux: `LD_LIBRARY_PATH` (the binary links sibling
/// libonnxruntime/libespeak-ng) and the bundled `espeak-ng-data` passed
/// as `--espeak_data` (piper resolves it relative to cwd otherwise).
///
/// Resolution mirrors the whisper lane: a pin is lane-agnostic (honored
/// against the engines table first, then the legacy tree), then the
/// engines lane's row, then the legacy `data/piper` tree newest-first —
/// both stay working installs (the engines lane adopts legacy trees at
/// serve preflight, doctor, and `engine prune`).
#[must_use]
pub fn server_bin(dirs: &BlazarDirs) -> Option<(PathBuf, PathBuf)> {
    if let Some(tag) = pinned_tag(dirs) {
        if let Some(row) = engines_lane_row(dirs)
            && row.tag == tag
            && let Some(bin) = engines_lane_server_bin(dirs, &row)
        {
            let lib = bin.parent()?.to_path_buf();
            return Some((bin, lib));
        }
        let dir = bin_root(dirs).join(&tag);
        if let Some((bin, lib)) = bin_and_lib_in(&dir) {
            return Some((bin, lib));
        }
        // Dangling pin (no row, no dir): fall through to the lanes' pick.
        tracing::warn!("piper pin {tag} has no binary; using newest installed tag");
    }
    if let Some(hit) = engines_lane_bin(dirs) {
        return Some(hit);
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

/// The engines-table lane's piper row: a row matching the pin (see
/// [`server_bin`] — pins are lane-agnostic) first, else the active row
/// when one is flagged, else the newest installed. `None` when no piper
/// row exists (the legacy tree decides) or the store cannot be read
/// (warn, never mask).
fn engines_lane_row(dirs: &BlazarDirs) -> Option<blazar_core::store::EngineRow> {
    let store = match blazar_core::Store::open(dirs) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!("piper engines-lane lookup could not open the store: {e:#}");
            return None;
        }
    };
    let rows = match store.list_engines() {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!("piper engines-lane listing failed: {e}");
            return None;
        }
    };
    if let Some(tag) = pinned_tag(dirs)
        && let Some(row) = rows
            .iter()
            .find(|r| r.kind == blazar_core::engine_kind::EngineKind::Piper && r.tag == tag)
    {
        return Some(row.clone());
    }
    // `list_engines` orders newest-installed first; the active row (if
    // any) still wins so a flagged lane is honored. `installed_at`
    // breaks ties explicitly — max_by_key on the active flag alone
    // returns the LAST row among equal keys, i.e. the OLDEST.
    rows.into_iter()
        .filter(|r| r.kind == blazar_core::engine_kind::EngineKind::Piper)
        .max_by_key(|r| (i64::from(r.active), r.installed_at))
}

/// The engines-table lane: resolve the piper row's binary the same way
/// register did. `None` when no piper row exists (legacy tree decides)
/// or the store cannot be read (warn, never mask).
fn engines_lane_bin(dirs: &BlazarDirs) -> Option<(PathBuf, PathBuf)> {
    let row = engines_lane_row(dirs)?;
    let bin = engines_lane_server_bin(dirs, &row)?;
    let lib = bin.parent()?.to_path_buf();
    Some((bin, lib))
}

/// Row's binary: the install-time probed path when it still exists,
/// otherwise re-derived from the tag dir (a relocated data dir must not
/// brick the lane — same recovery register performs).
fn engines_lane_server_bin(
    dirs: &BlazarDirs,
    row: &blazar_core::store::EngineRow,
) -> Option<PathBuf> {
    if let Ok(mut m) = serde_json::from_str::<crate::engine::manifest::Manifest>(&row.manifest) {
        m.anchor_server_path(&dirs.data_dir);
        let probed = PathBuf::from(&m.server_path);
        if probed.is_file() {
            return Some(probed);
        }
    }
    crate::engine::find_engine_binary(&dirs.engines_dir().join(&row.tag), &["piper", "piper.exe"])
        .ok()
}

/// The tag currency verdicts must compare against upstream: a mirror of
/// [`server_bin`]'s pick (pin first — either lane — then the engines
/// row, then the legacy tree). Update hints keyed on a bare lane listing
/// would warn about a version the serving lane can never pick.
#[must_use]
pub fn installed_tag(dirs: &BlazarDirs) -> Option<String> {
    if let Some(tag) = pinned_tag(dirs) {
        let row_hit = engines_lane_row(dirs)
            .filter(|r| r.tag == tag)
            .is_some_and(|r| engines_lane_server_bin(dirs, &r).is_some());
        let legacy_hit = bin_and_lib_in(&bin_root(dirs).join(&tag)).is_some();
        if row_hit || legacy_hit {
            return Some(tag);
        }
        // Dangling pin (no row with a binary, no dir): fall through.
    }
    if let Some(row) = engines_lane_row(dirs)
        && engines_lane_server_bin(dirs, &row).is_some()
    {
        return Some(row.tag);
    }
    sorted_tag_dirs(dirs)
        .into_iter()
        .next()
        .and_then(|d| d.file_name().map(|n| n.to_string_lossy().into_owned()))
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
pub fn list_voices(dirs: &BlazarDirs) -> Vec<String> {
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
pub fn voice_file(dirs: &BlazarDirs, voice: &str) -> Option<(PathBuf, PathBuf)> {
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

/// One voice in the upstream index (`rhasspy/piper-voices`), with its
/// on-disk byte size — the searchable remote catalog behind
/// `tts --search`.
#[derive(Debug, Clone)]
pub struct RemoteVoice {
    pub id: String,
    pub quality: String,
    pub bytes: u64,
}

/// Voice id + quality from a tree path like
/// `en/en_US/amy/medium/en_US-amy-medium.onnx`. `None` for the `.json`
/// config halves and any stem that lacks the pullable
/// `<locale>-<name>-<quality>` shape — every listed voice must resolve
/// through [`voice_repo_path`] later or the row is a dead end.
#[must_use]
pub fn voice_from_tree_path(path: &str) -> Option<(String, String)> {
    let base = path.rsplit('/').next()?;
    let id = base.strip_suffix(".onnx")?;
    let quality = id.rsplit('-').next()?.to_string();
    parse_voice_id(id)?;
    Some((id.to_string(), quality))
}

/// Language directories to walk for a query: a language-prefix query
/// (`en`) lists that language tree only; anything else (`amy`,
/// `en_GB-northern`) walks every language. Pure so the routing is
/// pinnable without a network.
#[must_use]
fn voice_lang_scope(root: &[crate::hf::HfTreeEntry], query: &str) -> Vec<String> {
    let langs: Vec<String> = root
        .iter()
        .filter(|e| e.is_dir())
        .map(|e| e.path.clone())
        .collect();
    let scoped: Vec<String> = langs
        .iter()
        .filter(|l| l.starts_with(query))
        .cloned()
        .collect();
    if scoped.is_empty() { langs } else { scoped }
}

/// Search `rhasspy/piper-voices` for voices whose id contains `query`
/// (case-insensitive). The Hub siblings expansion truncates this repo
/// (live: 3301 of thousands of files), so the tree API with cursor
/// pagination is the only complete index.
pub async fn search_voices(hf: &crate::hf::HfClient, query: &str) -> Result<Vec<RemoteVoice>> {
    let root = hf.list_tree(VOICES_REPO, "", false).await?;
    let needle = query.to_lowercase();
    // One language subtree per request, all in flight at once — each is
    // an independent paginated walk (futures is already a workspace dep).
    // The async block owns its `lang` because `list_tree` borrows the
    // path for the whole paginated future.
    let walks = futures::future::join_all(
        voice_lang_scope(&root, &needle)
            .into_iter()
            .map(|lang| async move { hf.list_tree(VOICES_REPO, &lang, true).await }),
    )
    .await;
    let mut out: Vec<RemoteVoice> = Vec::new();
    for page in walks {
        for entry in page?.into_iter().filter(crate::hf::HfTreeEntry::is_file) {
            if let Some((id, quality)) = voice_from_tree_path(&entry.path)
                && id.to_lowercase().contains(&needle)
            {
                out.push(RemoteVoice {
                    id,
                    quality,
                    bytes: entry.size.unwrap_or(0),
                });
            }
        }
    }
    out.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(out)
}

/// Download a voice (`.onnx` + `.onnx.json`) from
/// `rhasspy/piper-voices`. Fails fast on a non-onnx payload instead of
/// installing a corrupt voice.
pub async fn pull_voice(
    hf: &crate::hf::HfClient,
    dirs: &BlazarDirs,
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
        let _ = blazar_core::fs_safety::remove_dir_within(&dirs.data_dir, &dir);
        return Err(anyhow!(
            "{VOICES_REPO}/{voice}: payload is not a piper voice ({onnx_len} bytes, expected a \
             multi-MB onnx) — removed"
        ));
    }
    let json_raw = std::fs::read_to_string(dir.join(format!("{voice}.onnx.json")))
        .with_context(|| format!("read {voice} config"))?;
    if serde_json::from_str::<serde_json::Value>(&json_raw).is_err() {
        let _ = blazar_core::fs_safety::remove_dir_within(&dirs.data_dir, &dir);
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
    dirs: &BlazarDirs,
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
    dirs: &BlazarDirs,
    voice: &str,
    text: &str,
    length_scale: Option<f64>,
    timeout: std::time::Duration,
) -> Result<Vec<u8>> {
    let Some((bin, lib_dir)) = server_bin(dirs) else {
        return Err(anyhow!(
            "piper is not installed — run: blazar tts --install"
        ));
    };
    let Some((onnx, _json)) = voice_file(dirs, voice) else {
        let installed = list_voices(dirs);
        return Err(anyhow!(
            "voice {voice} is not pulled{} — run: blazar tts --pull {voice}",
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

    /// Tree-API entry for scope tests (kind: "file" | "directory").
    fn tree_entry(path: &str, kind: &str) -> crate::hf::HfTreeEntry {
        crate::hf::HfTreeEntry {
            path: path.to_string(),
            kind: kind.to_string(),
            size: None,
        }
    }

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
    fn unit__voice_from_tree_path__onnx_halves_with_pullable_shape() {
        let (id, quality) =
            voice_from_tree_path("en/en_US/amy/medium/en_US-amy-medium.onnx").unwrap();
        assert_eq!(id, "en_US-amy-medium");
        assert_eq!(quality, "medium");
        // Config halves never surface as voices.
        assert!(voice_from_tree_path("en/en_US/amy/medium/en_US-amy-medium.onnx.json").is_none());
        // A stem without the <locale>-<name>-<quality> shape cannot round-trip
        // through voice_repo_path later — a listed dead end.
        assert!(voice_from_tree_path("en/stray/voice.onnx").is_none());
    }

    #[test]
    fn unit__voice_lang_scope__language_prefix_scopes_else_walks_all() {
        let root = vec![
            tree_entry("de", "directory"),
            tree_entry("en", "directory"),
            tree_entry("eo", "directory"),
            tree_entry("es", "directory"),
            tree_entry("README.md", "file"),
        ];
        // A language prefix narrows the walk to matching roots.
        assert_eq!(voice_lang_scope(&root, "en"), vec!["en"]);
        assert_eq!(voice_lang_scope(&root, "e"), vec!["en", "eo", "es"]);
        // Name/quality queries match no language root: walk everything.
        let mut all = voice_lang_scope(&root, "amy");
        all.sort();
        assert_eq!(all, vec!["de", "en", "eo", "es"]);
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
        let dirs = BlazarDirs {
            config_dir: std::env::temp_dir().join("blazar-piper-test-cfg"),
            data_dir: std::env::temp_dir().join("blazar-piper-test-data"),
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
        assert!(e.to_string().contains("blazar tts --install"), "{e}");
    }
}
