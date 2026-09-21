//! Hugging Face model registry client + pull orchestration.
//!
//! Security invariants (ollama complaint #7 / CVE-2025-51471 class):
//! - outbound HTTPS only to HF hosts via a redirect-host allowlist
//! - the HF token is attached ONLY to first-party `huggingface.co` requests
//!   and is never logged, stored, or forwarded to CDN hosts
//! - LFS sha256 (when the API provides it) is verified; mismatch deletes
//!   the partial file and fails the pull
//!
//! Store layout stays plain files (`models/*.gguf`) so any tool can use
//! them directly — no blob storage, ever (complaint #4).

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use blazar_core::gguf;
use blazar_core::store::{ModelRow, Store};
use blazar_core::BlazarDirs;

use crate::events::BlazarEvent;

use crate::events::EventBus;

pub const HF_API_BASE: &str = "https://huggingface.co";
pub const DEFAULT_QUANT: &str = "Q4_K_M";

/// Redirect-host allowlist. Anything else is refused mid-redirect.
const EXTRA_DOWNLOAD_HOSTS: &[&str] = &["cdn-lfs.huggingface.co", "cas-bridge.xethub.hf.co"];

#[must_use]
pub fn is_allowed_download_host(host: &str) -> bool {
    host == "huggingface.co"
        || host.ends_with(".huggingface.co")
        || host == "hf.co"
        || host.ends_with(".hf.co")
        || EXTRA_DOWNLOAD_HOSTS.contains(&host)
}

/// Extra-host match with wildcard support: an exact pattern matches only
/// itself; a `*.suffix` pattern admits any subdomain of `suffix` (the
/// apex itself is NOT matched — presigned-CDN families live on
/// subdomains, e.g. `*.r2.cloudflarestorage.com`).
#[must_use]
pub(crate) fn extra_host_matches(host: &str, pattern: &str) -> bool {
    if pattern == host {
        return true;
    }
    pattern.strip_prefix("*.").is_some_and(|suffix| {
        host.strip_suffix(suffix)
            .is_some_and(|prefix| prefix.ends_with('.'))
    })
}

// ---------------------------------------------------------------------------
// Naming (complaint #6: registry name = actual repo name, never an alias)
// ---------------------------------------------------------------------------

/// `Qwen/Qwen2.5-0.5B-Instruct-GGUF` -> `qwen2.5-0.5b-instruct`.
#[must_use]
pub fn registry_name(repo: &str) -> String {
    let tail = repo.rsplit('/').next().unwrap_or(repo);
    let lowered = tail.to_lowercase();
    lowered
        .strip_suffix("-gguf")
        .unwrap_or(&lowered)
        .to_string()
}

/// Row name for a pull: the repo tail, or — when the quant slot carries
/// an exact `.gguf` filename (drafter-in-same-repo pulls) — that file's
/// stem, so `dflash-Qwen3-8B-Q8_0.gguf` and `Qwen3-8B-Q8_0.gguf` from
/// one repo land as distinct rows (`dflash-qwen3-8b-q8_0` vs
/// `qwen3-8b`).
#[must_use]
#[allow(clippy::case_sensitive_file_extension_comparisons)] // operand pre-lowercased
pub fn draft_aware_name(repo: &str, quant_slot: &str) -> String {
    let lowered = quant_slot.to_lowercase();
    if lowered.ends_with(".gguf") {
        return lowered.trim_end_matches(".gguf").to_string();
    }
    registry_name(repo)
}

#[derive(Debug, Clone, PartialEq)]
pub struct PullTarget {
    pub repo: String,
    pub quant: String,
}

/// Accepts `owner/repo:QUANT`, `owner/repo` (default quant), or a catalog
/// short name (resolved via the embedded catalog).
pub fn parse_pull_target(input: &str) -> Result<PullTarget> {
    let input = input.trim();
    if input.contains('/') {
        let (repo, quant) = match input.split_once(':') {
            Some((r, q)) => (r, q.to_uppercase()),
            None => (input, DEFAULT_QUANT.to_string()),
        };
        if repo.is_empty() || quant.is_empty() {
            return Err(anyhow!(
                "invalid pull target {input:?}: empty repo or quant"
            ));
        }
        return Ok(PullTarget {
            repo: repo.to_string(),
            quant,
        });
    }
    let entry = blazar_core::resolve(input).map_err(|e| anyhow!("{e}"))?;
    Ok(PullTarget {
        repo: entry.repo.clone(),
        quant: DEFAULT_QUANT.to_string(),
    })
}

// ---------------------------------------------------------------------------
// HF API shapes (verified against a live `?blobs=true` response; fixture:
// crates/blazar-runtime/tests/fixtures/qwen2.5-0.5b-blobs.json)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct HfLfs {
    pub sha256: String,
    #[serde(default)]
    pub size: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HfSibling {
    pub rfilename: String,
    #[serde(default)]
    pub size: Option<u64>,
    #[serde(default)]
    pub lfs: Option<HfLfs>,
}

/// `expand[]=gguf` aggregate over a repo's GGUF files. `total` is the
/// PARAMETER COUNT (not bytes — never render it as a size); the actual
/// on-disk bytes live in `totalFileSize` (sum over every `.gguf`, all
/// quants). `architecture`/`context_length` are read from the first
/// GGUF's header metadata and may be absent when the Hub could not
/// parse a file (live: leejet/* image GGUFs carry no architecture).
#[derive(Debug, Clone, Deserialize)]
pub struct HfGgufInfo {
    #[serde(default)]
    pub total: Option<u64>,
    #[serde(default, rename = "totalFileSize")]
    pub total_file_size: Option<u64>,
    #[serde(default)]
    pub architecture: Option<String>,
    #[serde(default)]
    pub context_length: Option<u64>,
}

/// `expand[]=safetensors` summary — the Hub's per-repo parameter
/// histogram. `parameters` maps safetensors dtype keys to tensor-element
/// counts (MLX/GPTQ quants pack 8 weights per U32/I32, so counts are
/// ELEMENTS, not on-disk cells); `total` is the element sum.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct HfSafetensorsInfo {
    #[serde(default)]
    pub parameters: Option<std::collections::BTreeMap<String, u64>>,
    #[serde(default)]
    pub total: Option<u64>,
}

impl HfSafetensorsInfo {
    /// On-disk byte estimate = Σ dtype-width × element count — exactly
    /// what `pull` downloads (a U32-packed MLX 4-bit cell is still 4
    /// bytes on disk). `None` when the histogram is empty or carries an
    /// unknown dtype (never guess a size).
    #[must_use]
    pub fn byte_estimate(&self) -> Option<u64> {
        let params = self.parameters.as_ref()?;
        if params.is_empty() {
            return None;
        }
        let mut total = 0u64;
        for (dtype, count) in params {
            total = total.checked_add(count.checked_mul(dtype_width_bytes(dtype)?)?)?;
        }
        Some(total)
    }
}

/// On-disk bytes per safetensors element for the Hub's dtype keys.
#[must_use]
fn dtype_width_bytes(dtype: &str) -> Option<u64> {
    Some(match dtype {
        "F64" | "I64" | "U64" => 8,
        "F32" | "I32" | "U32" => 4,
        "BF16" | "F16" | "I16" | "U16" => 2,
        "I8" | "U8" | "BOOL" | "F8_E4M3" | "F8_E5M2" => 1,
        _ => return None,
    })
}

/// `expand[]=config` summary — enough to label ARCH without pulling
/// `config.json` per repo (`max_position_embeddings` is NOT part of the
/// search expansion, so context stays a GGUF-only column). Diffusers
/// repos have no root `config.json`; their pipeline class arrives under
/// `diffusers._class_name` (live: `QwenImage21Pipeline`) and is the only
/// arch signal the search response carries for them.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct HfConfigSummary {
    #[serde(default)]
    pub architectures: Option<Vec<String>>,
    #[serde(default)]
    pub model_type: Option<String>,
    #[serde(default)]
    pub diffusers: Option<HfDiffusersSummary>,
}

/// The `config.diffusers` sub-object of the search expansion.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct HfDiffusersSummary {
    #[serde(default, rename = "_class_name")]
    pub class_name: Option<String>,
}

impl HfConfigSummary {
    /// ARCH ladder for one repo: `model_type` > first `architectures`
    /// entry (lowercased) > diffusers pipeline class as a snake-case
    /// token (`QwenImage21Pipeline` → `qwen_image21`, `FluxPipeline`
    /// → `flux`). `None` when the Hub carried none of the three.
    #[must_use]
    pub fn arch_hint(&self) -> Option<String> {
        if let Some(mt) = self.model_type.as_ref().filter(|m| !m.is_empty()) {
            return Some(mt.clone());
        }
        if let Some(first) = self.architectures.as_ref().and_then(|a| a.first()) {
            return Some(first.to_ascii_lowercase());
        }
        self.diffusers
            .as_ref()
            .and_then(|d| d.class_name.as_deref())
            .and_then(pipeline_class_to_arch)
    }
}

/// `QwenImage21Pipeline` → `qwen_image21`: strip the trailing
/// `Pipeline`, split camel-case words, lowercase, join with `_`.
/// Digits stay glued to the preceding word (`Image21` → `image21`).
fn pipeline_class_to_arch(class: &str) -> Option<String> {
    let stem = class.strip_suffix("Pipeline").unwrap_or(class);
    if stem.is_empty() {
        return None;
    }
    let mut out = String::with_capacity(stem.len() + 4);
    for (i, ch) in stem.chars().enumerate() {
        if i > 0 && ch.is_uppercase() {
            out.push('_');
        }
        out.push(ch.to_ascii_lowercase());
    }
    Some(out)
}

#[derive(Debug, Clone, Deserialize)]
pub struct HfModelInfo {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub siblings: Vec<HfSibling>,
    #[serde(default)]
    pub gguf: Option<HfGgufInfo>,
}

/// One file to download (a model shard or an mmproj).
#[derive(Debug, Clone)]
pub struct FilePlan {
    pub filename: String,
    pub bytes: u64,
    pub sha256: Option<String>,
}

/// Result of file selection for a pull.
#[derive(Debug, Clone)]
pub struct SelectedFiles {
    /// Shards in order; single-element vec for unsplit models.
    pub shards: Vec<FilePlan>,
    /// Parsed quant actually chosen (may differ from requested on fallback).
    pub quant: String,
    pub quant_fallback: bool,
    pub mmproj: Option<FilePlan>,
}

/// Pick model files from repo siblings: quant match (case-insensitive),
/// shard sets (`-00001-of-0000N`), and an optional mmproj projector.
/// Falls back to the smallest single GGUF (flagged) when the requested
/// quant is absent — never silently: the fallback is recorded in the row.
#[allow(clippy::case_sensitive_file_extension_comparisons)] // operand pre-lowercased
pub fn select_files(siblings: &[HfSibling], wanted_quant: &str) -> Result<SelectedFiles> {
    let wanted = wanted_quant.to_lowercase();
    let ggufs: Vec<&HfSibling> = siblings
        .iter()
        .filter(|s| s.rfilename.to_lowercase().ends_with(".gguf")) // pre-lowercased above
        .collect();
    if ggufs.is_empty() {
        return Err(anyhow!("repo has no .gguf files"));
    }

    // Group: unsplit vs sharded, keyed by quant-bearing base name.
    // Shard marker: `-00001-of-00002.gguf` (gguf-split convention).
    let mut sharded: std::collections::BTreeMap<(String, u32), Vec<(u32, &HfSibling)>> =
        std::collections::BTreeMap::new();
    let mut singles: Vec<&HfSibling> = Vec::new();
    for s in &ggufs {
        match parse_shard_marker(&s.rfilename) {
            Some((idx, count, base)) => {
                sharded.entry((base, count)).or_default().push((idx, s));
            }
            None => singles.push(s),
        }
    }

    // Same grammar as the search table (`quant_token_of`): shard bases
    // and full filenames both parse, lowercase to match `wanted`.
    let quant_of = |fname: &str| -> Option<String> { quant_token_of(fname) };

    // Exact-filename request (the quant slot carries a `.gguf` leaf,
    // e.g. `owner/repo:dflash-Model-Q8_0.gguf`): unambiguous single-file
    // selection — how drafter artifacts that share a quant token with
    // their target get pulled.
    if wanted.ends_with(".gguf") {
        return exact_filename_selection(siblings, &wanted);
    }

    // Try requested quant among singles first, then shard sets. When
    // several singles share the quant token (a repo hosting both a model
    // and its drafter, or mmproj-F16 beside model-F16), the LARGEST file
    // is the model — deterministic, no filename-prefix heuristics — and
    // the skipped same-quant siblings are named in a warning.
    let mut quant_hits: Vec<&HfSibling> = singles
        .iter()
        .copied()
        .filter(|s| quant_of(&s.rfilename).as_deref() == Some(wanted.as_str()))
        .collect();
    if quant_hits.len() > 1 {
        quant_hits.sort_by_key(|s| std::cmp::Reverse(plan(s).bytes));
        let skipped: Vec<&str> = quant_hits[1..]
            .iter()
            .map(|s| s.rfilename.as_str())
            .collect();
        tracing::warn!(
            "quant {wanted:?} matches {} files; taking {} (largest), skipping {}",
            quant_hits.len(),
            quant_hits[0].rfilename,
            skipped.join(", ")
        );
    }
    if let Some(chosen) = quant_hits.first() {
        return Ok(finish(
            vec![plan(chosen)],
            wanted_quant.to_string(),
            false,
            siblings,
        ));
    }
    for ((base, _count), group) in &sharded {
        if quant_of(base).as_deref() == Some(wanted.as_str()) {
            let mut ordered: Vec<&HfSibling> = group.iter().map(|(_, s)| *s).collect();
            ordered.sort_by_key(|s| parse_shard_marker(&s.rfilename).map_or(0, |m| m.0));
            let plans: Vec<FilePlan> = ordered.iter().map(|s| plan(s)).collect();
            return Ok(finish(plans, wanted_quant.to_string(), false, siblings));
        }
    }

    // Fallback: smallest single GGUF (deterministic, most likely to run).
    if let Some(smallest) = singles.iter().min_by_key(|s| plan(s).bytes) {
        let q = quant_of(&smallest.rfilename).unwrap_or_else(|| "unknown".into());
        return Ok(finish(
            vec![plan(smallest)],
            q.to_uppercase(),
            true,
            siblings,
        ));
    }
    // No singles at all: pick the smallest shard set by total size.
    let (_, group) = sharded
        .iter()
        .min_by_key(|(_, g)| g.iter().map(|(_, s)| plan(s).bytes).sum::<u64>())
        .ok_or_else(|| anyhow!("repo has GGUFs but no complete file set"))?;
    let mut ordered: Vec<&HfSibling> = group.iter().map(|(_, s)| *s).collect();
    ordered.sort_by_key(|s| parse_shard_marker(&s.rfilename).map_or(0, |m| m.0));
    let plans: Vec<FilePlan> = ordered.iter().map(|s| plan(s)).collect();
    let q = quant_of(&ordered[0].rfilename).unwrap_or_else(|| "unknown".into());
    Ok(finish(plans, q.to_uppercase(), true, siblings))
}

/// Exact-`.gguf`-filename selection for the quant slot. A filename-shaped
/// slot NEVER falls back to a quant guess: no match is a named error.
fn exact_filename_selection(siblings: &[HfSibling], wanted: &str) -> Result<SelectedFiles> {
    let hit = siblings.iter().find(|s| {
        let lower = s.rfilename.to_lowercase();
        lower.ends_with(&format!("/{wanted}")) || lower == wanted
    });
    let Some(hit) = hit else {
        return Err(anyhow!("repo has no file matching {wanted:?}"));
    };
    // The slot is a selector, not a display quant: the row shows the
    // trailing quant-looking token of the actual filename ("…-Q8_0.gguf"
    // -> "Q8_0") so `blazar list` and est_params see a real quant
    // instead of the whole uppercased filename.
    let stem = wanted.trim_end_matches(".gguf").to_lowercase();
    let display_quant = stem.rsplit('-').next().unwrap_or(&stem).to_uppercase();
    Ok(finish(vec![plan(hit)], display_quant, false, siblings))
}

/// `base-q4_k_m-00001-of-00002.gguf` -> `(1, 2, "base-q4_k_m")`.
/// Accepts only the gguf-split 5-digit `-of-` tail with idx in 1..=count.
fn parse_shard_marker(fname: &str) -> Option<(u32, u32, String)> {
    parse_shard_marker_pub(fname)
}

/// Public wrapper (shard-set file enumeration is shared with model removal).
#[must_use]
pub fn parse_shard_marker_pub(fname: &str) -> Option<(u32, u32, String)> {
    let lower = fname.to_lowercase();
    let stem = lower.strip_suffix(".gguf")?;
    let of_pos = stem.rfind("-of-")?;
    let count: u32 = stem[of_pos + 4..].parse().ok()?;
    let before = &stem[..of_pos];
    let idx_start = before.rfind('-')? + 1;
    let idx: u32 = before[idx_start..].parse().ok()?;
    if idx < 1 || idx > count || count < 2 {
        return None;
    }
    let base = before[..idx_start - 1].to_string();
    Some((idx, count, base))
}

fn plan(s: &HfSibling) -> FilePlan {
    FilePlan {
        filename: s.rfilename.clone(),
        bytes: s.lfs.as_ref().and_then(|l| l.size).or(s.size).unwrap_or(0),
        sha256: s.lfs.as_ref().map(|l| l.sha256.clone()),
    }
}

#[allow(clippy::case_sensitive_file_extension_comparisons)] // operand pre-lowercased
fn finish(
    shards: Vec<FilePlan>,
    quant: String,
    quant_fallback: bool,
    siblings: &[HfSibling],
) -> SelectedFiles {
    let mmproj = siblings
        .iter()
        .filter(|s| {
            let lower = s.rfilename.to_lowercase();
            lower.starts_with("mmproj") && lower.ends_with(".gguf")
        })
        .min_by_key(|s| plan(s).bytes)
        .map(plan);
    SelectedFiles {
        shards,
        quant,
        quant_fallback,
        mmproj,
    }
}

/// Root-level non-weight files of an HF model repo that the
/// safetensors lane downloads alongside the shards.
const HF_AUX_FILES: &[&str] = &[
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "vocab.txt",
    "preprocessor_config.json",
    "processor_config.json",
];

/// Selection for the safetensors (sglang) lane.
#[derive(Debug, Clone)]
pub struct SafetensorsSelection {
    /// Aux files (config/tokenizer/index) first, then shards in filename
    /// order — a partial pull keeps the small metadata early.
    pub files: Vec<FilePlan>,
    pub shard_count: usize,
}

/// Pick the safetensors file set from repo siblings: every ROOT-level
/// `*.safetensors` shard plus the loader/config/tokenizer allowlist.
/// Nested trees (`original/`, `onnx/`, `consolidated/`) and foreign
/// formats (`.bin`, `.pth`) are skipped — the sglang lane loads the
/// canonical HF layout only. `config.json` is mandatory (H1: a dir
/// without it is unloadable; fail at pull time with a teaching error).
#[allow(clippy::case_sensitive_file_extension_comparisons)] // operand pre-lowercased
pub fn select_safetensors_files(siblings: &[HfSibling]) -> Result<SafetensorsSelection> {
    let mut aux = Vec::new();
    let mut shards = Vec::new();
    for s in siblings.iter().filter(|s| !s.rfilename.contains('/')) {
        let lower = s.rfilename.to_lowercase();
        if lower.ends_with(".safetensors") {
            shards.push(plan(s));
        } else if lower.ends_with(".jinja") || HF_AUX_FILES.contains(&lower.as_str()) {
            aux.push(plan(s));
        }
    }
    if shards.is_empty() {
        return Err(anyhow!("repo has no root-level .safetensors shards"));
    }
    if !aux
        .iter()
        .any(|a| a.filename.eq_ignore_ascii_case("config.json"))
    {
        return Err(anyhow!(
            "repo has safetensors shards but no config.json — not a loadable HF model repo"
        ));
    }
    aux.sort_by(|a, b| a.filename.cmp(&b.filename));
    shards.sort_by(|a, b| a.filename.cmp(&b.filename));
    let shard_count = shards.len();
    aux.extend(shards);
    Ok(SafetensorsSelection {
        files: aux,
        shard_count,
    })
}

/// Stable identity of a repo revision for dir rows: sha256 over the
/// sorted `filename:bytes:sha` lines of the full selection.
fn revision_digest(files: &[FilePlan]) -> String {
    use sha2::Digest;
    let mut h = sha2::Sha256::new();
    let mut lines: Vec<String> = files
        .iter()
        .map(|f| {
            format!(
                "{}:{}:{}",
                f.filename,
                f.bytes,
                f.sha256.as_deref().unwrap_or("-")
            )
        })
        .collect();
    lines.sort();
    for l in lines {
        h.update(l.as_bytes());
        h.update(b"\n");
    }
    format!("{:x}", h.finalize())
}

/// Is the pulled dir complete? Every planned file present with the
/// exact listed size (cheap metadata check — downloads already
/// sha-verified each file).
fn safetensors_dir_intact(dir: &Path, sel: &SafetensorsSelection) -> bool {
    sel.files.iter().all(|f| {
        std::fs::metadata(dir.join(&f.filename)).is_ok_and(|m| {
            // A missing sibling size (0) can only be confirmed by
            // re-download; treat as not intact.
            f.bytes > 0 && m.len() == f.bytes
        })
    })
}

/// Post-download integrity: every shard named by
/// `model.safetensors.index.json` (the repo's own manifest) must be on
/// disk. Missing = the download is incomplete or the repo layout is
/// non-canonical — refuse now instead of at engine spawn.
fn verify_index_coverage(dir: &Path) -> Result<()> {
    let idx = dir.join("model.safetensors.index.json");
    if !idx.is_file() {
        return Ok(());
    }
    let raw = std::fs::read_to_string(&idx).with_context(|| format!("read {}", idx.display()))?;
    let v: serde_json::Value =
        serde_json::from_str(&raw).with_context(|| format!("parse {}", idx.display()))?;
    let Some(map) = v.get("weight_map").and_then(serde_json::Value::as_object) else {
        return Ok(());
    };
    for shard in map.values().filter_map(serde_json::Value::as_str) {
        if !dir.join(shard).is_file() {
            return Err(anyhow!(
                "shard {shard} named by model.safetensors.index.json is missing from the download"
            ));
        }
    }
    Ok(())
}

/// Display quant label for an HF model dir: quantization bits when the
/// repo carries a `quantization_config`, else the weight dtype. Uses
/// the HF-side vocabulary (BIT/BF16/FP16/F32/FP8), not llama quant
/// names — labels stay truthful for `est_params`.
pub(crate) fn hf_quant_label(meta: &blazar_core::hfmeta::HfMeta) -> String {
    if let Some(bits) = meta.quant_bits {
        return format!("{bits}BIT");
    }
    let d = meta.dtype.as_deref().map(str::to_lowercase);
    match d.as_deref() {
        Some(x) if x.starts_with("float8") || x.starts_with("fp8") => "FP8".to_string(),
        Some("bfloat16" | "bf16") => "BF16".to_string(),
        Some("float16" | "f16" | "fp16") => "FP16".to_string(),
        Some("float32" | "f32") => "F32".to_string(),
        Some(x) => x.to_uppercase(),
        None => "SAFETENSORS".to_string(),
    }
}

/// Rough bits-per-weight for a quant label; display-only param estimation.
fn quant_bpw(quant: &str) -> f64 {
    let q = quant.to_lowercase();
    match q.as_str() {
        "q2_k" | "q2_k_s" => 3.35,
        "q3_k_s" => 3.5,
        "q3_k_m" => 3.91,
        "q3_k_l" => 4.27,
        "q4_0" | "q4_1" => 4.55,
        "iq4_xs" => 4.25,
        "q4_k_m" => 4.85,
        "q5_0" | "q5_1" => 5.7,
        "q5_k_s" => 5.54,
        "q5_k_m" => 5.69,
        "q6_k" => 6.59,
        "q8_0" | "fp8" | "8bit" => 8.5,
        "fp16" | "f16" | "bf16" => 16.0,
        "f32" => 32.0,
        // HF-side labels from the safetensors lane (hf_quant_label).
        "q4_k_s" | "4bit" => 4.5,
        _ => 5.0,
    }
}

/// Estimated parameter count from file size + quant (labeled estimate,
/// never used for launch decisions).
#[must_use]
#[allow(clippy::cast_precision_loss)] // display-only estimate
pub fn est_params(bytes: u64, quant: &str) -> f64 {
    let bpw = quant_bpw(quant);
    if bpw <= 0.0 {
        return 0.0;
    }
    (bytes as f64 * 8.0 / bpw) / 1e9
}

// ---------------------------------------------------------------------------
// HTTP client
// ---------------------------------------------------------------------------

pub struct HfClient {
    pub(crate) http: reqwest::Client,
    pub(crate) api_base: reqwest::Url,
    dl_base: reqwest::Url,
    token: Option<String>,
    /// Parallel byte-range connections for large downloads (see
    /// `hf_parallel`). 1 = classic single-stream lane.
    pub(crate) download_connections: u32,
    /// Test-only extra redirect-allowed hosts (wiremock).
    extra_hosts: Vec<String>,
}

impl HfClient {
    pub fn new(token: Option<String>) -> Result<Self> {
        Self::with_bases(HF_API_BASE, HF_API_BASE, token, Vec::new())
    }

    /// Builder: set parallel download connections.
    #[must_use]
    pub fn with_download_connections(mut self, connections: u32) -> Self {
        self.download_connections = connections;
        self
    }

    #[allow(clippy::needless_pass_by_value)] // Vec is stored
    pub fn with_bases(
        api_base: &str,
        dl_base: &str,
        token: Option<String>,
        extra_hosts: Vec<String>,
    ) -> Result<Self> {
        // Empty env token degrades to anonymous (an empty Bearer is an
        // invalid credential, not a missing one).
        let token = token.filter(|t| !t.trim().is_empty());
        let extra = extra_hosts.clone();
        let policy = reqwest::redirect::Policy::custom(move |attempt| {
            let host = attempt.url().host_str().unwrap_or_default().to_string();
            if host.is_empty() {
                return attempt.error("redirect target has no host");
            }
            if is_allowed_download_host(&host) || extra.iter().any(|p| extra_host_matches(&host, p))
            {
                attempt.follow()
            } else {
                attempt.error(format!(
                    "blocked redirect to non-allowlisted host {host:?} (HF hosts only)"
                ))
            }
        });
        let http = reqwest::Client::builder()
            .redirect(policy)
            // Downloads are multi-hundred-MB: no total cap, bounded stalls.
            .connect_timeout(Duration::from_secs(30))
            .read_timeout(Duration::from_mins(2))
            .build()
            .context("build HF http client")?;
        Ok(Self {
            http,
            api_base: reqwest::Url::parse(api_base)?,
            dl_base: reqwest::Url::parse(dl_base)?,
            token,
            download_connections: 8,
            extra_hosts,
        })
    }

    /// Token rides ONLY on first-party huggingface.co requests (test
    /// stand-ins included); CDN hops never see it. Wildcard extra-host
    /// patterns ("*.suffix") never receive the token by construction —
    /// the equality below can't match them; presigned CDN URLs are
    /// already authorized and forwarding credentials there would be
    /// exfiltration.
    pub(crate) fn token_for(&self, url: &reqwest::Url) -> Option<String> {
        let host = url.host_str()?;
        let first_party = host == "huggingface.co" || host.ends_with(".huggingface.co");
        (first_party || self.extra_hosts.iter().any(|h| h == host))
            .then(|| self.token.clone())
            .flatten()
    }

    /// GET /api/models/{repo}?blobs=true
    pub async fn model_info(&self, repo: &str) -> Result<HfModelInfo> {
        let url = self
            .api_base
            .join(&format!("api/models/{repo}?blobs=true"))
            .map_err(|e| anyhow!("bad repo {repo:?}: {e}"))?;
        let mut req = self.http.get(url.clone());
        if let Some(token) = self.token_for(&url) {
            req = req.bearer_auth(token);
        }
        let resp = req.send().await.context("HF models API request failed")?;
        match resp.status() {
            reqwest::StatusCode::OK => {}
            reqwest::StatusCode::UNAUTHORIZED | reqwest::StatusCode::FORBIDDEN => {
                return Err(anyhow!(
                    "repo {repo} is gated or private (status {}). Set HF_TOKEN if you have access",
                    resp.status()
                ));
            }
            reqwest::StatusCode::NOT_FOUND => {
                return Err(anyhow!("repo {repo} not found on Hugging Face"));
            }
            reqwest::StatusCode::TOO_MANY_REQUESTS => {
                let retry = resp
                    .headers()
                    .get("retry-after")
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("a while");
                return Err(anyhow!("HF rate limited; retry after {retry}"));
            }
            other => {
                return Err(anyhow!("HF models API {other} for {repo}"));
            }
        }
        let info: HfModelInfo = resp.json().await.context("decode models API JSON")?;
        Ok(info)
    }

    /// Stream one file to `dest` (via `.part` + atomic rename), resuming
    /// from a previous partial when present. Returns final byte count.
    /// `on_progress` is called with (downloaded, total) after each chunk.
    pub async fn download_file(
        &self,
        repo: &str,
        plan: &FilePlan,
        dest: &Path,
        mut on_progress: impl FnMut(u64, u64),
    ) -> Result<u64> {
        let url = self
            .dl_base
            .join(&format!(
                "{repo}/resolve/main/{}",
                url_encode_path(&plan.filename)
            ))
            .map_err(|e| anyhow!("bad download URL for {}: {e}", plan.filename))?;
        self.download_to(url, plan, dest, &mut on_progress).await
    }

    /// Generic streaming-download core: `.part` resume, Range, sha256
    /// verify, atomic rename. The URL arrives prebuilt by the caller (HF
    /// resolve path, ollama-registry blob path, ...), so this client's
    /// redirect allowlist and token policy apply uniformly.
    pub(crate) async fn download_to(
        &self,
        url: reqwest::Url,
        plan: &FilePlan,
        dest: &Path,
        mut on_progress: impl FnMut(u64, u64),
    ) -> Result<u64> {
        // Parallel byte-range lane first: engages only when the expected
        // size (from pull metadata) can pay for it and the server proves
        // Range support on a probe; every other shape falls through to
        // the classic lane below.
        if self.download_connections > 1
            && (plan.bytes == 0 || plan.bytes >= crate::hf_parallel::MIN_PARALLEL_BYTES)
        {
            let token = self.token_for(&url);
            if let Some(bytes) = crate::hf_parallel::try_parallel(
                &self.http,
                token.as_deref(),
                &url,
                plan,
                dest,
                self.download_connections,
                &mut on_progress,
            )
            .await?
            {
                return Ok(bytes);
            }
        }
        let part = sibling_part_path(dest);
        let (mut have, mut hasher) = seed_partial(&part).await?;

        // A full-length `.part` reaching this lane is a sparse parallel
        // artifact (or a stale upstream size): `bytes=have-` would 416 and
        // appending past the end can only corrupt. Restart from zero —
        // the sha256 gate still owns correctness.
        if have > 0 && plan.bytes > 0 && have >= plan.bytes {
            have = 0;
            hasher = Sha256::new();
            let _ = tokio::fs::remove_file(&part).await;
        }
        let mut req = self.http.get(url.clone());
        if let Some(token) = self.token_for(&url) {
            req = req.bearer_auth(token);
        }
        if have > 0 {
            req = req.header(reqwest::header::RANGE, format!("bytes={have}-"));
        }
        let mut resp = req.send().await.context("download request failed")?;
        let status = resp.status();
        if !status.is_success() && status.as_u16() != 206 {
            return Err(anyhow!(
                "download {} failed: {} {}",
                plan.filename,
                status.as_str(),
                status.canonical_reason().unwrap_or("")
            ));
        }
        // Server ignored the Range: restart from zero.
        if have > 0 && status.as_u16() != 206 {
            have = 0;
            hasher = Sha256::new();
        }
        let total = if status.as_u16() == 206 {
            have + resp.content_length().unwrap_or(0)
        } else {
            resp.content_length().unwrap_or(plan.bytes)
        };

        let mut file = if status.as_u16() == 206 {
            tokio::fs::OpenOptions::new()
                .append(true)
                .open(&part)
                .await?
        } else {
            tokio::fs::File::create(&part).await?
        };
        let mut downloaded = have;
        on_progress(downloaded, total);
        while let Some(chunk) = resp.chunk().await? {
            file.write_all(&chunk).await?;
            hasher.update(&chunk);
            downloaded += chunk.len() as u64;
            on_progress(downloaded, total);
        }
        file.flush().await?;
        drop(file);

        if let Some(expected) = &plan.sha256 {
            let got = format!("{:x}", hasher.finalize());
            if !got.eq_ignore_ascii_case(expected) {
                let _ = tokio::fs::remove_file(&part).await;
                return Err(anyhow!(
                    "sha256 mismatch for {}: expected {expected}, got {got}; partial deleted",
                    plan.filename
                ));
            }
        }
        tokio::fs::rename(&part, dest)
            .await
            .with_context(|| format!("finalize {}", dest.display()))?;
        Ok(downloaded)
    }
}

fn url_encode_path(path: &str) -> String {
    path.split('/')
        .map(|seg| {
            seg.bytes()
                .map(|b| match b {
                    b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                        char::from(b).to_string()
                    }
                    _ => format!("%{b:02X}"),
                })
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("/")
}

/// Tail-resume seed for the classic lane: hash the existing `.part` bytes
/// so the final sha256 covers the whole file, and report its length.
async fn seed_partial(part: &Path) -> Result<(u64, Sha256)> {
    let mut hasher = Sha256::new();
    if !part.exists() {
        return Ok((0, hasher));
    }
    let len = std::fs::metadata(part)
        .map_err(|e| anyhow!("stat {}: {e}", part.display()))?
        .len();
    let existing = tokio::fs::File::open(part).await?;
    let mut reader = tokio::io::BufReader::new(existing);
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = reader.read(&mut buf).await?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok((len, hasher))
}

pub(crate) fn sibling_part_path(dest: &Path) -> PathBuf {
    let mut s = dest.as_os_str().to_os_string();
    s.push(".part");
    PathBuf::from(s)
}

// ---------------------------------------------------------------------------
// Search + fit (no download)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct SearchEntry {
    pub id: String,
    #[serde(default)]
    pub downloads: Option<u64>,
    #[serde(default)]
    pub likes: Option<u64>,
    #[serde(default)]
    pub siblings: Vec<HfSibling>,
    #[serde(default)]
    pub gguf: Option<HfGgufInfo>,
    #[serde(default)]
    pub safetensors: Option<HfSafetensorsInfo>,
    #[serde(default)]
    pub config: Option<HfConfigSummary>,
    /// Hub format tags (`gguf`, `mlx`, `safetensors`, `awq`, `onnx`, …).
    /// MLX repos carry BOTH `mlx` and `safetensors` — callers display with
    /// most-specific-first priority, not first-match.
    #[serde(default)]
    pub tags: Vec<String>,
}

/// Extract the quantization token from a filename or shard base, in
/// lowercase. One grammar shared by the search-table listing
/// (`quant_tokens`) and the actual file selection (`select_files`) so
/// the two can never disagree on what counts as a quant:
///
/// - the leaf (path-free) segment, lowercased, `mmproj*` excluded;
/// - `.gguf` suffix optional (shard base names have none);
/// - a trailing `-NNNNN-of-NNNNN` shard marker is stripped first;
/// - the token is the last `.`/`-`-separated segment, matching either
///   the literal `fp16` or an `iq`/`tq`/`bf`/`q`/`f` prefix followed by
///   a required ASCII digit and `[A-Za-z0-9_]*`;
/// - segments are scanned right-to-left, so the trailing segment stays
///   authoritative (`model.q4` is q4) and a double extension
///   (`Q4_K_M.GGUF.gguf`) still finds the quant past the `gguf` noise
///   segment;
/// - when a whole segment carries a leading release token glued with `_`
///   (live: `qwen_image_2.1_Q5_K_M` — the version dot ends the
///   `-`/`.`-segment at `1_Q5_K_M`), the longest `_`-separated suffix
///   tail that matches the grammar is the token;
/// - when the leaf carries no token at all, the parent folder is tried
///   (`Q4_K_M/model.gguf` layouts name the quant in the folder).
///
/// `q4_k_m`, `iq4_xs`, `f16`, `bf16`, `f32` match; model-size tags
/// (`3b`), dates (`2511`) and finetune words (`heretic`) don't — the
/// required digit after the prefix is what separates them.
fn quant_token_of(fname: &str) -> Option<String> {
    let lower = fname.to_ascii_lowercase();
    let mut path = lower.rsplit('/');
    let leaf = path.next().unwrap_or(&lower);
    if leaf.starts_with("mmproj") {
        return None;
    }
    let stem = strip_shard_tail(leaf.strip_suffix(".gguf").unwrap_or(leaf));
    if let Some(token) = quant_token_in_stem(stem) {
        return Some(token);
    }
    let parent = path.next()?;
    if parent.starts_with("mmproj") {
        return None;
    }
    quant_token_in_stem(parent)
}

/// First quant token in one file stem: `.`/`-` segments right-to-left,
/// each tried whole then as `_`-suffix tails. Right-to-left keeps the
/// trailing segment authoritative; earlier segments only fire when the
/// last one carries no token (double-extension `Q4_K_M.GGUF` shapes).
fn quant_token_in_stem(stem: &str) -> Option<String> {
    for seg in stem.rsplit(['.', '-']) {
        if let Some(token) = quant_prefix_match(seg) {
            return Some(token);
        }
        let parts: Vec<&str> = seg.split('_').collect();
        if let Some(token) =
            (1..parts.len()).find_map(|k| quant_prefix_match(&parts[k..].join("_")))
        {
            return Some(token);
        }
    }
    None
}

/// One grammar atom: `fp16` verbatim, or an `iq`/`tq`/`bf`/`q`/`f`
/// prefix followed by a required ASCII digit and `[A-Za-z0-9_]*`.
fn quant_prefix_match(seg: &str) -> Option<String> {
    if seg == "fp16" {
        return Some(seg.to_string());
    }
    for prefix in ["iq", "tq", "bf", "q", "f"] {
        if let Some(rest) = seg.strip_prefix(prefix) {
            let mut chars = rest.chars();
            if matches!(chars.next(), Some(d) if d.is_ascii_digit())
                && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
            {
                return Some(seg.to_string());
            }
        }
    }
    None
}

/// Strip a trailing `-00001-of-00005` gguf-split marker, if present.
/// `parse_shard_marker_pub` needs the `.gguf` suffix, so this walks the
/// last 14 bytes by hand instead.
fn strip_shard_tail(stem: &str) -> &str {
    const TAIL: usize = "-00000-of-00000".len();
    if stem.len() < TAIL {
        return stem;
    }
    let (head, tail) = stem.split_at(stem.len() - TAIL);
    let b = tail.as_bytes();
    // `-NNNNN-of-NNNNN` layout: digits at 1..=5 and 10..=14.
    let digits = |from: usize, to: usize| (from..to).all(|i| b[i].is_ascii_digit());
    if b[0] == b'-' && &tail[6..9] == "-of" && digits(1, 6) && digits(10, TAIL) {
        head
    } else {
        stem
    }
}

/// Quant names advertised by a repo's GGUF filenames, canonical uppercase,
/// deduped, alphabetically sorted. Powers the search table's QUANTS column
/// so `blazar pull <REPO>[:quant]` can be chosen from the listing itself.
///
/// A filename segment qualifies only if it parses as `(IQ|TQ|BF|F|Q)` +
/// digit + `[A-Za-z0-9_]*` — model-size tags (`3B`), dates (`2511`) and
/// finetune words (`heretic`, `fable`) don't. Shards (`-00002-of-00005`,
/// digit-only tail) and `mmproj-*` projectors are excluded by the same
/// rules.
pub fn quant_tokens(names: impl IntoIterator<Item = impl AsRef<str>>) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for name in names {
        let Some(quant) = quant_token(name.as_ref()) else {
            continue;
        };
        if !out.contains(&quant) {
            out.push(quant);
        }
    }
    out.sort();
    out
}

/// `.gguf`-gated variant of [`quant_token_of`] for the search table:
/// only real GGUF files contribute tokens, canonical uppercase.
fn quant_token(filename: &str) -> Option<String> {
    if !std::path::Path::new(filename)
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
    {
        return None;
    }
    quant_token_of(filename).map(|t| t.to_ascii_uppercase())
}

/// True when a bare search token IS a quant name (`q4`, `Q4_K_M`,
/// `iq4_xs`, `f16`) — the CLI lifts such tokens out of the text query
/// into its client-side quant filter. Strict by construction: `qwen`,
/// `gguf`, `awq`, `int4` fail the grammar and stay query text.
#[must_use]
pub fn is_quant_token(token: &str) -> bool {
    quant_prefix_match(&token.to_ascii_lowercase()).is_some()
}

impl HfClient {
    /// Hub model search across every weight format (complaint #15:
    /// discovery beyond a registry; any community quant is findable).
    /// `format` picks the lane — see [`search_path`].
    ///
    /// With a query, the Hub's `sort=downloads` order is re-ranked for
    /// relevance (`relevance_rank`): the 30-day download count is
    /// bot-farmable (live: a derivative finetune sat at the exact same
    /// 1.5M as the official mirror), so downloads only ever breaks ties.
    /// Empty queries keep the Hub order verbatim (browse-most-popular
    /// has no relevance signal to add).
    pub async fn search(&self, query: &str, format: &str, limit: u32) -> Result<Vec<SearchEntry>> {
        let rerank = !query.trim().is_empty();
        let fetch = if rerank {
            // Overfetch so the local re-rank has candidates beyond the
            // download-sorted head; cap at the Hub's 100/page anyway.
            limit.saturating_mul(4).min(100).max(limit)
        } else {
            limit
        };
        let url = self
            .api_base
            .join(&search_path(query, format, fetch))
            .map_err(|e| anyhow!("bad search URL: {e}"))?;
        let resp = self
            .http
            .get(url)
            .send()
            .await
            .context("HF search request")?;
        if !resp.status().is_success() {
            return Err(anyhow!("HF search returned {}", resp.status()));
        }
        let entries: Vec<SearchEntry> = resp.json().await.context("decode search results")?;
        Ok(if rerank {
            let mut ranked = relevance_rank(&entries, query);
            ranked.truncate(limit as usize);
            ranked
        } else {
            entries
        })
    }
}

/// Re-rank Hub search results by name relevance to the query.
///
/// Ordering is lexicographic over: (1) query-token coverage desc — a
/// token matches when some repo-name segment contains it; (2) unmatched
/// name-segment count ASC — derivative finetunes carry long advertising
/// tails (`-The-Defiant-Fable-Uncensored-Heretic-NEO`) that clean
/// mirrors (`Qwen3.5-9B-GGUF`) never have; (3) likes desc — the
/// community-curated signal, far costlier to farm than downloads;
/// (4) downloads desc — tiebreak only. Name evidence only: no owner
/// lists, no heuristics about who "should" win.
#[must_use]
pub fn relevance_rank(entries: &[SearchEntry], query: &str) -> Vec<SearchEntry> {
    let tokens: Vec<String> = query
        .to_ascii_lowercase()
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(str::to_string)
        .collect();
    if tokens.is_empty() {
        // No tokens, no relevance signal: keep the caller's (Hub) order.
        return entries.to_vec();
    }
    let mut scored: Vec<RankKey> = Vec::with_capacity(entries.len());
    for (idx, e) in entries.iter().enumerate() {
        let (coverage, unmatched) = name_match_stats(&e.id, &tokens);
        scored.push((
            std::cmp::Reverse(coverage),
            unmatched,
            std::cmp::Reverse(e.likes.unwrap_or(0)),
            std::cmp::Reverse(e.downloads.unwrap_or(0)),
            idx,
            e.clone(),
        ));
    }
    scored.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then(a.1.cmp(&b.1))
            .then(a.2.cmp(&b.2))
            .then(a.3.cmp(&b.3))
            .then(a.4.cmp(&b.4))
    });
    scored.into_iter().map(|(_, _, _, _, _, e)| e).collect()
}

/// Re-rank sort key: coverage desc, unmatched asc, likes desc, downloads
/// desc, input index (determinism on identical keys).
type RankKey = (
    std::cmp::Reverse<u32>,
    u32,
    std::cmp::Reverse<u64>,
    std::cmp::Reverse<u64>,
    usize,
    SearchEntry,
);

/// (query tokens matched by some name segment, name segments no token
/// matched). The org prefix (`owner/`) is excluded from the penalty
/// count — every repo has one and it is never a query target.
fn name_match_stats(repo_id: &str, tokens: &[String]) -> (u32, u32) {
    let lower = repo_id.to_ascii_lowercase();
    let segments: Vec<&str> = lower
        .split(['/', '-', '_', '.'])
        .filter(|s| !s.is_empty())
        .collect();
    let name = lower.split_once('/').map_or(lower.as_str(), |(_, n)| n);
    let name_segments: Vec<&str> = segments
        .iter()
        .copied()
        .filter(|s| name.contains(s))
        .collect();
    let coverage = u32::try_from(
        tokens
            .iter()
            .filter(|t| segments.iter().any(|s| s.contains(t.as_str())))
            .count(),
    )
    .unwrap_or(u32::MAX);
    let unmatched = u32::try_from(
        name_segments
            .iter()
            .filter(|s| !tokens.iter().any(|t| s.contains(t.as_str())))
            .count(),
    )
    .unwrap_or(u32::MAX);
    (coverage, unmatched)
}

/// Build the `api/models` query for [`HfClient::search`]. `format` is a
/// Hub tag filter passed through verbatim (`gguf`, `safetensors`, `mlx`,
/// `awq`, `gptq`, `fp8`, `onnx`, … — case-insensitive, trimmed); the
/// sentinels `any`/`all` drop the filter entirely so every format is
/// browsable. An empty `query` omits `search=` (browse-most-popular).
/// `expand[]=tags` feeds the per-row FORMAT column; `expand[]=gguf` and
/// `expand[]=safetensors` + `expand[]=config` fill SIZE/ARCH for GGUF and
/// safetensors rows respectively (each is simply absent on the other
/// format — one code path for every format, zero extra round-trips).
#[must_use]
pub fn search_path(query: &str, format: &str, limit: u32) -> String {
    let mut path = format!(
        "api/models?limit={limit}&sort=downloads&direction=-1\
         &expand[]=gguf&expand[]=safetensors&expand[]=config\
         &expand[]=likes&expand[]=siblings&expand[]=tags"
    );
    if !query.is_empty() {
        path.push_str("&search=");
        path.push_str(&url_encode_path(query));
    }
    let format = format.trim().to_ascii_lowercase();
    if !format.is_empty() && format != "any" && format != "all" {
        path.push_str("&filter=");
        path.push_str(&url_encode_path(&format));
    }
    path
}

/// One row of a fit preview.
#[derive(Debug, Clone, serde::Serialize)]
pub struct FitRow {
    pub quant: String,
    pub file: String,
    pub bytes: u64,
    pub fits_vram: bool,
    pub kv_bytes_at_default_ctx: u64,
    pub recommended_ctx: u32,
    /// ctx that fits when the KV cache runs `q8_0` (halved): shows the
    /// headroom the profile ladder's `q8_0` grade buys on this GPU.
    pub recommended_ctx_q8: u32,
}

/// Rule-6 KV ladder outcome for `bytes` of weights against `vram_bytes`:
/// (fits at default ctx, recommended ctx, ctx with `q8_0` KV). The same
/// f16 → q8 → q4 halving the profile compiler applies, pre-download.
fn kv_ladder(bytes: u64, vram_bytes: u64, default_ctx: u32) -> (bool, u32, u32) {
    let kv_f16 = kv_estimate_f16(default_ctx);
    // F105: mirror the compiler's rule-6 KV ladder (f16 -> q8 -> q4)
    // instead of the degenerate `kv_f16.min(kv_f16 / 2)`, which only
    // ever tested the q8 grade.
    let fits = bytes + kv_f16 <= vram_bytes
        || bytes + kv_f16 / 2 <= vram_bytes
        || bytes + kv_f16 / 4 <= vram_bytes;
    let recommended = if fits {
        default_ctx
    } else {
        // shrink ctx until KV fits alongside the weights
        let mut ctx = default_ctx;
        while ctx > 1024 && bytes + kv_estimate_f16(ctx) > vram_bytes {
            ctx /= 2;
        }
        ctx
    };
    // Same shrink loop with halved KV: what q8_0 KV buys (rule-6 grade).
    let mut ctx_q8 = default_ctx;
    while ctx_q8 > 1024 && bytes + kv_estimate_f16(ctx_q8) / 2 > vram_bytes {
        ctx_q8 /= 2;
    }
    (fits, recommended, ctx_q8)
}

/// Pre-download compatibility preview (complaint #14): given a repo's
/// sibling list, produce per-quant fit rows against the local hardware.
/// Uses the same KV math as the profile compiler's rule 6.
#[must_use]
pub fn fit_rows(siblings: &[HfSibling], vram_bytes: u64, default_ctx: u32) -> Vec<FitRow> {
    let mut rows = Vec::new();
    for s in siblings {
        let lower = s.rfilename.to_lowercase();
        let Some(stem) = lower.strip_suffix(".gguf") else {
            continue;
        };
        if stem.contains("-of-") {
            continue; // shard parts: fit uses the set total via sibling sums
        }
        let bytes = s.lfs.as_ref().and_then(|l| l.size).or(s.size).unwrap_or(0);
        if bytes == 0 {
            continue;
        }
        let quant = stem.rsplit('-').next().unwrap_or("unknown").to_uppercase();
        let (fits, recommended, ctx_q8) = kv_ladder(bytes, vram_bytes, default_ctx);
        rows.push(FitRow {
            quant,
            file: s.rfilename.clone(),
            bytes,
            fits_vram: fits,
            kv_bytes_at_default_ctx: kv_estimate_f16(default_ctx),
            recommended_ctx: recommended,
            recommended_ctx_q8: ctx_q8,
        });
    }
    // Safetensors lane (sglang/mistralrs): one AGGREGATE row — shards are
    // parts of a single quantization state, unlike GGUF where each file
    // is a separate quant choice. Same conservative KV allowance; the
    // arch-true number is measured at serve time.
    if rows.is_empty() {
        let mut total: u64 = 0;
        let mut shards = 0u32;
        for s in siblings {
            if !s.rfilename.to_lowercase().ends_with(".safetensors") {
                continue;
            }
            if let Some(bytes) = s.lfs.as_ref().and_then(|l| l.size).or(s.size) {
                if bytes > 0 {
                    total = total.saturating_add(bytes);
                    shards += 1;
                }
            }
        }
        if total > 0 {
            let (fits, recommended, ctx_q8) = kv_ladder(total, vram_bytes, default_ctx);
            rows.push(FitRow {
                quant: "safetensors".to_string(),
                file: format!("{shards} safetensors shard(s)"),
                bytes: total,
                fits_vram: fits,
                kv_bytes_at_default_ctx: kv_estimate_f16(default_ctx),
                recommended_ctx: recommended,
                recommended_ctx_q8: ctx_q8,
            });
        }
    }
    rows.sort_by_key(|r| std::cmp::Reverse(r.bytes));
    rows
}

fn kv_estimate_f16(ctx: u32) -> u64 {
    // Without per-arch metadata pre-download, report a conservative
    // 1.5 GiB-per-16k KV allowance; the real number comes from GGUF at
    // load time (profile rule 6) and is shown in `blazar show`.
    1_572_864_000u64.saturating_mul(u64::from(ctx)) / 16_384
}

// ---------------------------------------------------------------------------
// Pull orchestration
// ---------------------------------------------------------------------------

pub struct Puller {
    pub dirs: BlazarDirs,
    pub client: HfClient,
    pub bus: EventBus,
    /// Allow a pull to flip the model's format (safetensors dir <->
    /// GGUF file) over an existing row. Guarded by [`flip_guard`] —
    /// see the incident note there.
    pub force: bool,
}

/// Format-flip refusal for pulls: replacing an existing model row with
/// a DIFFERENT format (GGUF file over a safetensors `.d` directory, or
/// the reverse) is almost never what the user asked for — a short-name
/// catalog resolve can silently flip formats (live incident 2026-09-17:
/// `pull qwen2.5-0.5b` replaced a 954 MiB safetensors row with a GGUF
/// under the same canonical name). Refuse without `--force`; a forced
/// flip keeps the superseded files on disk — boot preflight re-adopts
/// them under a derived name.
pub(crate) fn flip_guard(
    name: &str,
    existing: Option<&blazar_core::store::ModelRow>,
    incoming_is_dir: bool,
    force: bool,
) -> Result<()> {
    let Some(row) = existing else {
        return Ok(());
    };
    let existing_is_dir = Path::new(&row.path).is_dir();
    if existing_is_dir == incoming_is_dir || force {
        return Ok(());
    }
    let (have, want) = if existing_is_dir {
        ("a safetensors directory", "a GGUF")
    } else {
        ("a GGUF", "a safetensors directory")
    };
    anyhow::bail!(
        "model {name} is already pulled as {have} ({}); this pull would replace it with \
         {want} — pass --force to flip (the old files stay on disk; boot preflight \
         re-adopts them) or pull the exact repo",
        row.path
    )
}

/// RAII lockfile guard: released (removed) on drop, panic-safe.
#[derive(Debug)]
pub(crate) struct PullLock {
    path: PathBuf,
}

impl PullLock {
    pub(crate) fn acquire(dirs: &BlazarDirs, name: &str) -> Result<Self> {
        std::fs::create_dir_all(dirs.run_dir())?;
        let path = dirs.run_dir().join(format!("pull-{name}.lock"));
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(mut f) => {
                use std::io::Write;
                let _ = writeln!(f, "{}", std::process::id());
                Ok(Self { path })
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                // The lockfile carries the owning pid: a dead owner means
                // the pull crashed (kill -9, power loss) and the lock is
                // stale — steal it immediately instead of blocking for the
                // 6h age fallback. Live owner -> refuse, naming the pid.
                let path_display = path.display().to_string();
                let owner = std::fs::read_to_string(&path)
                    .ok()
                    .and_then(|raw| raw.trim().parse::<u32>().ok());
                let owner_alive = owner.is_some_and(pid_alive);
                if !owner_alive {
                    // Unknown/corrupt lockfile with no parseable pid falls
                    // back to the age heuristic so a corrupt file cannot
                    // wedge pulls forever either.
                    let age = std::fs::metadata(&path)
                        .and_then(|m| m.modified())
                        .ok()
                        .and_then(|t| t.elapsed().ok())
                        .unwrap_or_default();
                    if owner.is_none() && age < Duration::from_hours(6) {
                        return Err(anyhow!(
                            "pull already in progress for {name} (lockfile {path_display}, unreadable owner); delete it if you are sure no pull is running"
                        ));
                    }
                    let _ = std::fs::remove_file(&path);
                    return Self::acquire(dirs, name);
                }
                Err(anyhow!(
                    "pull already in progress for {name} by pid {} (lockfile {path_display})",
                    owner.unwrap_or_default()
                ))
            }
            Err(e) => Err(anyhow!("create lock {}: {e}", path.display())),
        }
    }
}

impl Drop for PullLock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Is a process with this pid alive? Linux: `/proc` — no shell-out.
/// Other unixes: `kill -0` (signal 0 = existence probe, no delivery).
/// Windows: `tasklist` filter. Pid reuse can false-positive; the age
/// fallback and manual delete remain as escape hatches.
fn pid_alive(pid: u32) -> bool {
    #[cfg(target_os = "linux")]
    {
        std::path::Path::new(&format!("/proc/{pid}")).exists()
    }
    #[cfg(all(unix, not(target_os = "linux")))]
    {
        std::process::Command::new("kill")
            .arg("-0")
            .arg(pid.to_string())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
    }
    #[cfg(windows)]
    {
        std::process::Command::new("tasklist")
            .args(["/FI", &format!("PID eq {pid}"), "/NH"])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .output()
            .is_ok_and(|o| String::from_utf8_lossy(&o.stdout).contains(&pid.to_string()))
    }
}

/// What a pull produced: the stored row and whether the model was already
/// present (idempotent fast path — zero bytes moved).
#[derive(Debug, Clone)]
pub struct PullOutcome {
    pub row: ModelRow,
    pub already_present: bool,
}

/// What a re-pull of an already-stored model must do.
#[derive(Debug)]
pub(crate) enum Repull {
    /// Files on disk match repo+quant+sha and are intact — zero bytes.
    Present(Box<ModelRow>),
    /// Model file intact but the mmproj sidecar differs — download (or
    /// drop) ONLY the sidecar instead of the multi-GiB model.
    DeltaMmproj { expected: bool },
    /// Diffusion row with an intact `DiT` but a dead required component
    /// (`VAE` / text encoder) — re-fetch the component set only.
    DeltaComponents,
    /// Full download (any dead leaves were already pruned where safe).
    Full,
}

/// Derive every shard leaf of a sharded download from its recorded launch
/// path (`…-00001-of-0000N.gguf` is the HF naming convention). `None`
/// when the recorded path does not follow the pattern — the caller then
/// keeps the old files and falls back to collision-disambiguation.
fn shard_set(model_path: &str, shards: i64) -> Option<Vec<PathBuf>> {
    let p = Path::new(model_path);
    let dir = p.parent()?;
    let name = p.file_name()?.to_str()?;
    let stem = name.strip_suffix(".gguf")?;
    let total = u32::try_from(shards).ok()?;
    let first = format!("-{:05}-of-{:05}", 1, total);
    let base = stem.strip_suffix(first.as_str())?;
    Some(
        (1..=total)
            .map(|n| dir.join(format!("{base}-{n:05}-of-{total:05}.gguf")))
            .collect(),
    )
}

/// Is the row's main file plausibly intact? Single-shard rows verify the
/// exact recorded length AND a parseable GGUF header; sharded rows can
/// only afford a header check on the launch shard (`bytes` is the SUM
/// across files, not any single file's length).
pub(crate) fn model_file_intact(row: &ModelRow) -> bool {
    let path = Path::new(&row.path);
    if !path.is_file() {
        return false;
    }
    if row.shards <= 1 {
        let Ok(meta) = std::fs::metadata(path) else {
            return false;
        };
        meta.len() == u64::try_from(row.bytes).unwrap_or(u64::MAX) && gguf_container_ok(path)
    } else {
        gguf::read_metadata_file(path).is_ok()
    }
}

/// Container-level GGUF check: header + KV section must parse.
/// `read_metadata_file` additionally requires
/// `general.architecture`, which diffusion component files (`DiT` /
/// encoder splits — 0-KV GGUFs) never carry; for them that exact
/// error still proves the container walked clean (magic, version,
/// every KV entry). Same string-match precedent as the reconcile and
/// `read_model_meta` guards. Any other parse error is real corruption.
fn gguf_container_ok(path: &Path) -> bool {
    match gguf::read_metadata_file(path) {
        Ok(_) => true,
        Err(e) => e.to_string().contains("missing general.architecture"),
    }
}

/// The mmproj sidecar agrees with the selection: expected + present, or
/// absent on both sides.
pub(crate) fn mmproj_matches(row: &ModelRow, expects_mmproj: bool) -> bool {
    match (&row.mmproj_path, expects_mmproj) {
        (Some(p), true) => Path::new(p).is_file(),
        (None, false) => true,
        _ => false,
    }
}

/// Drop a leftover `.part` (+ its parallel-download sidecar) once the
/// final leaf is verified present — a stale partial from an interrupted
/// attempt is pure garbage at that point.
pub(crate) fn sweep_stale_part(final_path: &str) {
    for suffix in [".part", ".part.progress"] {
        let stale = format!("{final_path}{suffix}");
        if let Err(e) = std::fs::remove_file(&stale) {
            if e.kind() != std::io::ErrorKind::NotFound {
                tracing::warn!("could not remove stale partial {stale}: {e}");
            }
        }
    }
}

/// Decide what a re-pull must do given the row already stored under
/// `name` (if any). Shared by the HF and registry lanes.
pub(crate) fn repull_gate(
    existing: Option<&ModelRow>,
    name: &str,
    repo: &str,
    quant: &str,
    expected_sha: Option<&str>,
    expects_mmproj: bool,
) -> Repull {
    let Some(row) = existing else {
        return Repull::Full;
    };
    let same_model = row.repo == repo && row.quant.eq_ignore_ascii_case(quant);
    if !same_model {
        return Repull::Full;
    }
    // Upstream revision check: recorded sha vs the freshly listed one
    // (metadata compare — instant, no re-hash of the local file).
    // Unknown on either side (non-LFS files, legacy rows) trusts the
    // local file rather than forcing a multi-GiB redownload.
    let sha_current = expected_sha.is_none_or(|s| {
        row.sha256
            .as_deref()
            .is_none_or(|r| r.eq_ignore_ascii_case(s))
    });
    if !sha_current {
        tracing::warn!(
            model = %name,
            "upstream revision changed for {}:{} — replacing the local copy",
            row.repo,
            row.quant
        );
        prune_replaced(name, row, &[], "upstream revision changed");
        return Repull::Full;
    }
    if model_file_intact(row) {
        if !mmproj_matches(row, expects_mmproj) {
            return Repull::DeltaMmproj {
                expected: expects_mmproj,
            };
        }
        // Component rows: an intact `DiT` with a dead component file is a
        // component-only repair — never a multi-GiB `DiT` redownload.
        if required_component_missing(row) {
            return Repull::DeltaComponents;
        }
        tracing::info!(
            model = %name,
            "already present ({}, {} shards) — skipping download",
            row.quant,
            row.shards
        );
        Repull::Present(Box::new(row.clone()))
    } else {
        prune_replaced(name, row, &[], "integrity check failed");
        Repull::Full
    }
}

/// A diffusion component row whose REQUIRED sidecar (`VAE` or text
/// encoder) no longer exists on disk. The optional vision encoder is
/// not a repair trigger — its loss only disables image edits.
pub(crate) fn required_component_missing(row: &ModelRow) -> bool {
    [row.vae_path.as_deref(), row.llm_path.as_deref()]
        .into_iter()
        .flatten()
        .any(|p| !Path::new(p).is_file())
}

/// Assemble the store row for a fully-downloaded safetensors dir:
/// metadata from config.json, quant label from `quantization_config` /
/// dtype, byte totals from the selection. Kept beside the GGUF row
/// builders so the row dialect stays in one place.
fn safetensors_model_row(
    name: &str,
    repo: &str,
    dir: &std::path::Path,
    sel: &SafetensorsSelection,
    digest: String,
) -> Result<ModelRow> {
    let meta = blazar_core::hfmeta::read_hf_config(dir)
        .map_err(|e| anyhow!("hf model dir {} unusable: {e}", dir.display()))?;
    let quant = hf_quant_label(&meta);
    let bytes: u64 = sel.files.iter().map(|f| f.bytes).sum();
    Ok(ModelRow {
        name: name.to_string(),
        repo: repo.to_string(),
        quant: quant.clone(),
        path: dir.display().to_string(),
        bytes: i64::try_from(bytes).unwrap_or(i64::MAX),
        sha256: Some(digest),
        mmproj_path: None,
        vae_path: None,
        llm_path: None,
        llm_vision_path: None,
        shards: i64::try_from(sel.shard_count).unwrap_or(i64::MAX),
        arch: (!meta.architecture.is_empty()).then(|| meta.architecture.clone()),
        params: Some(est_params(bytes, &quant)),
        ctx_train: meta.ctx_train.and_then(|c| i64::try_from(c).ok()),
        pulled_at: i64::try_from(SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs())
            .unwrap_or(i64::MAX),
    })
}

impl Puller {
    /// Pull a model into the store. `target` = `owner/repo[:QUANT]` or a
    /// catalog short name. The quant slot may instead carry an exact
    /// `.gguf` filename (`owner/repo:draft-Model-Q8_0.gguf`) — used for
    /// drafter artifacts living in the target's own repo; those rows are
    /// named after the file stem so they never collide with the model row.
    pub async fn pull(&self, target: &str) -> Result<PullOutcome> {
        let parsed = parse_pull_target(target)?;
        let name = draft_aware_name(&parsed.repo, &parsed.quant);
        let _lock = PullLock::acquire(&self.dirs, &name)?;
        self.pull_locked(&parsed, &name).await
    }

    async fn pull_locked(&self, target: &PullTarget, name: &str) -> Result<PullOutcome> {
        let info = self.client.model_info(&target.repo).await?;
        // Lane fork: GGUF files (llamacpp/mistralrs engines) vs a
        // safetensors model directory (sglang engine). A repo with BOTH
        // keeps the GGUF lane — existing behavior unchanged; the log
        // teaches the safetensors half exists.
        let has_gguf = info
            .siblings
            .iter()
            .any(|s| s.rfilename.to_lowercase().ends_with(".gguf"));
        if !has_gguf {
            if info
                .siblings
                .iter()
                .any(|s| s.rfilename.to_lowercase().ends_with(".safetensors"))
            {
                return self.pull_safetensors_locked(target, name, &info).await;
            }
            return Err(anyhow!(
                "repo {} has no .gguf or .safetensors model files",
                target.repo
            ));
        }
        if info
            .siblings
            .iter()
            .any(|s| s.rfilename.to_lowercase().ends_with(".safetensors"))
        {
            tracing::info!(
                model = %name,
                "repo {} also hosts safetensors weights; pulling the GGUF lane (llamacpp/mistralrs)",
                target.repo
            );
        }
        let selected = select_files(&info.siblings, &target.quant)?;

        let store = Store::open(&self.dirs)?;

        // Idempotent re-pull ladder, shared with the registry lane:
        //   Present      — repo+quant+sha match, files intact: no-op.
        //   DeltaMmproj  — model intact, sidecar differs: sidecar-only pull.
        //   Full         — replace (dead/replaced leaves pruned where the
        //                  shard set is derivable, so the fresh download
        //                  lands on the canonical filename instead of the
        //                  `owner--repo--leaf` collision slug).
        let existing = store.get_model(name)?;
        flip_guard(name, existing.as_ref(), false, self.force)?;
        let expected_sha = selected.shards[0].sha256.clone();
        let decision = repull_gate(
            existing.as_ref(),
            name,
            &target.repo,
            &selected.quant,
            expected_sha.as_deref(),
            selected.mmproj.is_some(),
        );
        if let Some(outcome) = self
            .handle_repull_decision(name, target, &info, &selected, existing.as_ref(), decision)
            .await?
        {
            return Ok(outcome);
        }

        let (shard_paths, mmproj_dest) = self
            .download_full_selection(name, target, &selected)
            .await?;

        let (mut row, pull_warning) = build_model_row(
            name,
            target,
            &info,
            &selected,
            &shard_paths,
            mmproj_dest.as_ref(),
        )?;
        let mut pull_warning = pull_warning;
        // Kvless GGUF = a diffusion `DiT` component (no text-model
        // architecture inside). A known family pulls the `VAE`/`TE` set the
        // sdcpp engine needs to boot; an unknown one keeps the
        // existing kvless teaching at spawn time.
        if row.arch.is_none() && shard_paths.len() == 1 {
            self.attach_diffusion_set(target, name, &selected.quant, &mut row, &mut pull_warning)
                .await?;
        }
        if let Some(w) = &pull_warning {
            tracing::warn!(model = %name, "{w}");
        }
        store.upsert_model(&row)?;
        // The replaced quant's files are orphaned otherwise (rows are
        // quant-independent by name; a re-quant swap overwrites the row).
        if let Some(old) = existing.as_ref() {
            let keep_model = Path::new(&row.path);
            let keep_mmproj = row.mmproj_path.as_deref().map(Path::new);
            let keep: Vec<&Path> = keep_mmproj
                .as_ref()
                .map_or_else(|| vec![keep_model], |mm| vec![keep_model, mm]);
            prune_replaced(name, old, &keep, "superseded by a different quant");
        }
        self.bus.publish(BlazarEvent::ModelPulled {
            name: name.to_string(),
            warning: pull_warning.clone(),
        });
        if selected.quant_fallback {
            tracing::warn!(
                "quant {} not found in {}; pulled {} instead",
                target.quant,
                target.repo,
                selected.quant
            );
        }
        Ok(PullOutcome {
            row,
            already_present: false,
        })
    }

    /// Safetensors lane: the repo has no GGUFs, so the model is an HF
    /// transformers repo (sharded `*.safetensors` + config + tokenizer).
    /// Everything lands in a `models/<name>.d/` directory consumed by the
    /// sglang engine; the row's `path` is the directory. Identity = a
    /// digest over the full file listing (name+size+sha), so a re-pull
    /// of the same revision is a no-op and a moved tag replaces cleanly.
    async fn pull_safetensors_locked(
        &self,
        target: &PullTarget,
        name: &str,
        info: &HfModelInfo,
    ) -> Result<PullOutcome> {
        let sel = select_safetensors_files(&info.siblings)?;
        let digest = revision_digest(&sel.files);
        let dir = self.dirs.models_dir().join(format!("{name}.d"));
        let store = Store::open(&self.dirs)?;

        let existing = store.get_model(name)?;
        flip_guard(name, existing.as_ref(), true, self.force)?;
        if let Some(outcome) = self.safetensors_repull_outcome(
            existing.as_ref(),
            &dir,
            &sel,
            &digest,
            &target.repo,
            name,
        ) {
            return Ok(outcome);
        }

        if dir.exists() {
            tracing::info!(
                model = %name,
                "reusing {} — completed files re-verify, files dropped upstream are left in place",
                dir.display()
            );
        }
        std::fs::create_dir_all(&dir)?;

        let total_bytes: u64 = sel.files.iter().map(|f| f.bytes).sum();
        let bar = indicatif::ProgressBar::new(total_bytes);
        bar.set_style(
            indicatif::ProgressStyle::default_bar()
                .template("{msg} {bar:30} {bytes}/{total_bytes} ({eta})")
                .expect("valid template"),
        );
        bar.set_message(format!("pull {name} (safetensors)"));
        let mut last_publish = 0u64;
        let mut progress = |downloaded: u64, total: u64| {
            bar.set_position(downloaded);
            if downloaded.saturating_sub(last_publish) >= 16 << 20 || downloaded == total {
                last_publish = downloaded;
                self.bus.publish(BlazarEvent::PullProgress {
                    name: name.to_string(),
                    downloaded,
                    total,
                });
            }
        };

        for (i, file) in sel.files.iter().enumerate() {
            let before: u64 = sel.files[..i].iter().map(|f| f.bytes).sum();
            let mut progress_one = |d: u64, t: u64| progress(before + d.min(t), total_bytes);
            let dest = dir.join(&file.filename);
            self.client
                .download_file(&target.repo, file, &dest, &mut progress_one)
                .await
                .inspect_err(|e| {
                    self.bus.publish(BlazarEvent::PullFailed {
                        name: name.to_string(),
                        error: e.to_string(),
                    });
                })?;
        }
        bar.finish_and_clear();

        // Integrity: the shard index is the repo's own manifest — every
        // shard it names must be on disk (H1: fail now, not at load).
        verify_index_coverage(&dir)?;

        let row = safetensors_model_row(name, &target.repo, &dir, &sel, digest)?;
        store.upsert_model(&row)?;
        self.bus.publish(BlazarEvent::ModelPulled {
            name: name.to_string(),
            warning: None,
        });
        Ok(PullOutcome {
            row,
            already_present: false,
        })
    }

    /// Idempotence + supersede for the safetensors lane:
    /// `Some(outcome)` = resolved (no-op re-pull; caller returns it),
    /// `None` = proceed with the download — stale files already cleaned.
    fn safetensors_repull_outcome(
        &self,
        existing: Option<&blazar_core::store::ModelRow>,
        dir: &Path,
        sel: &SafetensorsSelection,
        digest: &str,
        repo: &str,
        name: &str,
    ) -> Option<PullOutcome> {
        let row = existing?;
        let same_revision = row.repo == repo
            && row
                .sha256
                .as_deref()
                .is_some_and(|s| s.eq_ignore_ascii_case(digest));
        if same_revision && safetensors_dir_intact(dir, sel) {
            let total = u64::try_from(row.bytes).unwrap_or(0);
            self.bus.publish(BlazarEvent::PullProgress {
                name: name.to_string(),
                downloaded: total,
                total,
            });
            self.bus.publish(BlazarEvent::ModelPulled {
                name: name.to_string(),
                warning: None,
            });
            return Some(PullOutcome {
                row: row.clone(),
                already_present: true,
            });
        }
        // Different revision (or damaged dir): replace. A previous
        // dir row at a different path is removed wholesale; a GGUF
        // row under the same name goes through the shared pruner.
        let old_path = Path::new(&row.path);
        if old_path.is_dir() {
            if old_path != dir {
                if let Err(e) = std::fs::remove_dir_all(old_path) {
                    if e.kind() != std::io::ErrorKind::NotFound {
                        tracing::warn!(
                            model = %name,
                            "could not remove replaced dir {}: {e}",
                            old_path.display()
                        );
                    }
                }
            }
        } else {
            prune_replaced(name, row, &[], "replaced by a safetensors pull");
        }
        None
    }

    /// Full-download phase of [`Self::pull_locked`]: one shared progress bar
    /// across all shards plus the mmproj sidecar.
    async fn download_full_selection(
        &self,
        name: &str,
        target: &PullTarget,
        selected: &SelectedFiles,
    ) -> Result<(Vec<PathBuf>, Option<PathBuf>)> {
        let models_dir = self.dirs.models_dir();
        std::fs::create_dir_all(&models_dir)?;

        let total_bytes: u64 = selected.shards.iter().map(|s| s.bytes).sum::<u64>()
            + selected.mmproj.as_ref().map_or(0, |m| m.bytes);
        let bar = indicatif::ProgressBar::new(total_bytes);
        bar.set_style(
            indicatif::ProgressStyle::default_bar()
                .template("{msg} {bar:30} {bytes}/{total_bytes} ({eta})")
                .expect("valid template"),
        );
        bar.set_message(format!("pull {name}:{}", selected.quant));

        let mut last_publish = 0u64;
        let mut progress = |downloaded: u64, total: u64| {
            bar.set_position(downloaded);
            // Throttle bus traffic: ~every 16 MiB.
            if downloaded.saturating_sub(last_publish) >= 16 << 20 || downloaded == total {
                last_publish = downloaded;
                self.bus.publish(BlazarEvent::PullProgress {
                    name: name.to_string(),
                    downloaded,
                    total,
                });
            }
        };

        let mut shard_paths = Vec::new();
        for (i, shard) in selected.shards.iter().enumerate() {
            let mut progress_one = |d: u64, t: u64| {
                // Aggregate across files for the shared bar.
                let before: u64 = selected.shards[..i].iter().map(|s| s.bytes).sum();
                progress(before + d.min(t), total_bytes);
            };
            let dest = unique_dest(&models_dir, &shard.filename, &target.repo);
            self.client
                .download_file(&target.repo, shard, &dest, &mut progress_one)
                .await
                .inspect_err(|e| {
                    self.bus.publish(BlazarEvent::PullFailed {
                        name: name.to_string(),
                        error: e.to_string(),
                    });
                })?;
            shard_paths.push(dest);
        }
        let mmproj_dest = selected
            .mmproj
            .as_ref()
            .map(|mm| unique_dest(&models_dir, &mm.filename, &target.repo));
        if let (Some(mm), Some(dest)) = (&selected.mmproj, &mmproj_dest) {
            self.client
                .download_file(&target.repo, mm, dest, &mut progress)
                .await
                .inspect_err(|e| {
                    // F104: mmproj failure must reach /api/pull watchers
                    // like a shard failure does, not vanish via `?`.
                    self.bus.publish(BlazarEvent::PullFailed {
                        name: name.to_string(),
                        error: format!("mmproj: {e}"),
                    });
                })?;
        }
        bar.finish_and_clear();
        Ok((shard_paths, mmproj_dest))
    }

    /// Fetch the diffusion component set for a kvless `DiT` pull: `VAE` and
    /// text encoder are required (missing files fail the pull loudly);
    /// the vision encoder for edits is best-effort. Component files are
    /// shared, read-only weights: an intact file at the canonical dest
    /// is re-used byte-for-byte, never re-downloaded.
    ///
    /// Complete a kvless `DiT` pull with its component set: a known
    /// diffusion family fetches the `VAE`/`TE` files (vision optional)
    /// onto the row; an unknown family keeps the file but records the
    /// boot-blocking warning that spawn time will teach verbatim.
    async fn attach_diffusion_set(
        &self,
        target: &PullTarget,
        name: &str,
        quant: &str,
        row: &mut blazar_core::ModelRow,
        pull_warning: &mut Option<String>,
    ) -> Result<()> {
        let Some(family) = crate::diffusion::diffusion_family(&target.repo) else {
            let note = format!(
                "diffusion component GGUF with no known model family (supported: {}); \
                 pulled the `DiT` file only — it cannot boot without its VAE/text encoder",
                crate::diffusion::supported_families().join(", ")
            );
            tracing::warn!(model = %name, "{note}");
            pull_warning.get_or_insert(note);
            return Ok(());
        };
        let (vae, llm, llm_vision) = self.pull_diffusion_components(name, family, quant).await?;
        row.vae_path = Some(vae);
        row.llm_path = Some(llm);
        row.llm_vision_path = llm_vision;
        Ok(())
    }

    async fn pull_diffusion_components(
        &self,
        name: &str,
        family: &crate::diffusion::DiffusionFamily,
        quant: &str,
    ) -> Result<(String, String, Option<String>)> {
        let models_dir = self.dirs.models_dir();
        std::fs::create_dir_all(&models_dir)?;

        let vae_info = self.client.model_info(family.vae.repo).await?;
        let vae_plan =
            crate::diffusion::component_plan(&vae_info, &family.vae, quant).ok_or_else(|| {
                anyhow!(
                    "VAE {} not found in {}",
                    family.vae.repo_path,
                    family.vae.repo
                )
            })?;

        let te_info = self.client.model_info(family.text_encoder.repo).await?;
        let (te_plan, te_note) = if let Some(p) =
            crate::diffusion::component_plan(&te_info, &family.text_encoder, quant)
        {
            (p, None)
        } else {
            let fallback_quant = family.text_encoder_fallback_quant;
            let fallback =
                crate::diffusion::component_plan(&te_info, &family.text_encoder, fallback_quant)
                    .ok_or_else(|| {
                        anyhow!(
                            "text encoder for quant {quant} (and fallback {fallback_quant}) \
                             not found in {}",
                            family.text_encoder.repo
                        )
                    })?;
            let note =
                format!("text encoder quant {quant} unavailable; pulled {fallback_quant} instead");
            (fallback, Some(note))
        };

        // Vision encoder is optional: absent upstream (or a dead listing)
        // downgrades to a warning, never a failed pull.
        let mut vision_plan = None;
        if let Some(src) = &family.vision_encoder {
            vision_plan = match self.client.model_info(src.repo).await {
                Ok(info) => crate::diffusion::component_plan(&info, src, quant),
                Err(e) => {
                    tracing::warn!(model = %name, "vision encoder listing failed ({e}); edits disabled for this set");
                    None
                }
            };
        }

        let mut plans = vec![
            (family.vae.repo, &vae_plan, true),
            (family.text_encoder.repo, &te_plan, true),
        ];
        if let (Some(src), Some(plan)) = (&family.vision_encoder, &vision_plan) {
            plans.push((src.repo, plan, false));
        }
        let total_bytes: u64 = plans.iter().map(|(_, p, _)| p.bytes).sum();
        let bar = indicatif::ProgressBar::new(total_bytes);
        bar.set_style(
            indicatif::ProgressStyle::default_bar()
                .template("{msg} {bar:30} {bytes}/{total_bytes} ({eta})")
                .expect("valid template"),
        );
        bar.set_message(format!("pull {name}: components"));

        let mut done: u64 = 0;
        let mut paths: Vec<(bool, PathBuf)> = Vec::new();
        for (repo, plan, required) in &plans {
            let fetched = self
                .fetch_component_file(name, repo, plan, *required, &bar, &mut done)
                .await?;
            if let Some(dest) = fetched {
                paths.push((*required, dest));
            }
        }
        bar.finish_and_clear();

        let vae = paths[0].1.display().to_string();
        let llm = paths[1].1.display().to_string();
        let llm_vision = paths.get(2).map(|p| p.1.display().to_string());
        if let Some(note) = te_note {
            tracing::warn!(model = %name, "{note}");
        }
        Ok((vae, llm, llm_vision))
    }

    /// Fetch one component file with the shared progress bar. Returns
    /// the destination when the file is on disk (fetched now, or
    /// re-used intact from an earlier pull); `None` for an optional
    /// component that failed (warned, set continues without it).
    async fn fetch_component_file(
        &self,
        name: &str,
        repo: &str,
        plan: &FilePlan,
        required: bool,
        bar: &indicatif::ProgressBar,
        done: &mut u64,
    ) -> Result<Option<PathBuf>> {
        let models_dir = self.dirs.models_dir();
        let leaf = Path::new(&plan.filename).file_name().map_or_else(
            || plan.filename.clone(),
            |f| f.to_string_lossy().into_owned(),
        );
        // Content-first reuse: a byte-exact file may already sit under
        // the bare leaf (hand-placed, or left by an earlier install) —
        // `unique_dest` disambiguates NAME collisions before content is
        // considered, which once re-downloaded a 5 GiB text encoder
        // right next to its own byte-exact copy. Both the bare leaf
        // and the collision-resolved dest are content-checked.
        let bare = models_dir.join(&leaf);
        let dest = unique_dest(&models_dir, &leaf, repo);
        for candidate in [&bare, &dest] {
            if let Some(reused) = reuse_byte_exact(candidate, plan) {
                tracing::info!(model = %name, "component {} already on disk (byte-exact) — reusing", leaf);
                *done += plan.bytes;
                bar.set_position(*done);
                return Ok(Some(reused));
            }
        }
        let before = *done;
        let mut progress = |d: u64, t: u64| {
            bar.set_position(before + d.min(t));
        };
        let fetched = self
            .client
            .download_file(repo, plan, &dest, &mut progress)
            .await
            .inspect_err(|e| {
                self.bus.publish(BlazarEvent::PullFailed {
                    name: name.to_string(),
                    error: format!("component {}: {e}", plan.filename),
                });
            });
        match fetched {
            Ok(bytes) => {
                *done += bytes.max(plan.bytes);
                Ok(Some(dest))
            }
            Err(e) if !required => {
                tracing::warn!(model = %name, "optional component {} failed ({e}) — skipped", plan.filename);
                Ok(None)
            }
            Err(e) => Err(anyhow!(
                "component set incomplete: {}/{} failed: {e}; re-run the pull to resume",
                repo,
                plan.filename
            )),
        }
    }

    /// Act on the gate's decision. `Some(outcome)` = the pull is already
    /// finished (no-op or delta); `None` = proceed with the full download
    /// (any pre-download pruning is done here).
    async fn handle_repull_decision(
        &self,
        name: &str,
        target: &PullTarget,
        info: &HfModelInfo,
        selected: &SelectedFiles,
        existing: Option<&ModelRow>,
        decision: Repull,
    ) -> Result<Option<PullOutcome>> {
        if let Repull::Present(row) = decision {
            let total = u64::try_from(row.bytes).unwrap_or(0);
            self.bus.publish(BlazarEvent::PullProgress {
                name: name.to_string(),
                downloaded: total,
                total,
            });
            self.bus.publish(BlazarEvent::ModelPulled {
                name: name.to_string(),
                warning: None,
            });
            sweep_stale_part(&row.path);
            return Ok(Some(PullOutcome {
                row: *row,
                already_present: true,
            }));
        }
        if let Repull::DeltaComponents = decision {
            // The `DiT` is intact (gate proved it); only the component set
            // needs fetching. Family comes from the row's own repo —
            // the gate only reaches here on a same-repo re-pull.
            if let Some(old) = existing.filter(|r| r.vae_path.is_some() || r.llm_path.is_some()) {
                let family = crate::diffusion::diffusion_family(&old.repo).ok_or_else(|| {
                    anyhow!(
                        "component repair for {} has no known family (supported: {})",
                        old.repo,
                        crate::diffusion::supported_families().join(", ")
                    )
                })?;
                let (vae, llm, llm_vision) = self
                    .pull_diffusion_components(name, family, &old.quant)
                    .await?;
                let mut row = old.clone();
                row.vae_path = Some(vae);
                row.llm_path = Some(llm);
                row.llm_vision_path = llm_vision;
                Store::open(&self.dirs)?.upsert_model(&row)?;
                tracing::info!(model = %name, "component set repaired — DiT untouched");
                self.bus.publish(BlazarEvent::ModelPulled {
                    name: name.to_string(),
                    warning: None,
                });
                return Ok(Some(PullOutcome {
                    row,
                    already_present: false,
                }));
            }
        }
        if let Repull::DeltaMmproj { expected } = decision {
            // Shard paths are only fully known for single-shard rows
            // (multi-shard rows record just the launch leaf); escalate
            // those to a full download.
            if let Some(old) = existing.filter(|r| r.shards <= 1) {
                let (row, pull_warning) = self
                    .delta_mmproj(name, target, info, selected, old, expected)
                    .await?;
                if let Some(w) = &pull_warning {
                    tracing::warn!(model = %name, "{w}");
                }
                Store::open(&self.dirs)?.upsert_model(&row)?;
                self.bus.publish(BlazarEvent::ModelPulled {
                    name: name.to_string(),
                    warning: pull_warning,
                });
                return Ok(Some(PullOutcome {
                    row,
                    already_present: false,
                }));
            }
        }
        if let Some(old) = existing {
            if old.repo != target.repo {
                tracing::warn!(
                    model = %name,
                    "replacing row previously pulled from {} with {}",
                    old.repo,
                    target.repo
                );
            }
            // Same canonical filename as the new selection means the
            // download would collide — prune BEFORE it so the fresh bytes
            // land on the canonical leaf (upstream revision change).
            let new_leaf = selected.shards[0].filename.as_str();
            if Path::new(&old.path)
                .file_name()
                .and_then(|f| f.to_str())
                .is_some_and(|f| f.eq_ignore_ascii_case(new_leaf))
            {
                prune_replaced(name, old, &[], "upstream revision changed");
            }
        }
        Ok(None)
    }

    /// Sidecar-only pull: the model file is intact and only the mmproj
    // projector differs (repo added one, or dropped it). Moves megabytes,
    // not gigabytes.
    async fn delta_mmproj(
        &self,
        name: &str,
        target: &PullTarget,
        info: &HfModelInfo,
        selected: &SelectedFiles,
        old: &ModelRow,
        expected: bool,
    ) -> Result<(ModelRow, Option<String>)> {
        let models_dir = self.dirs.models_dir();
        std::fs::create_dir_all(&models_dir)?;
        let mut mmproj_dest = None;
        if expected {
            let mm = selected.mmproj.as_ref().ok_or_else(|| {
                anyhow!("delta pull requested a projector but the selection has none")
            })?;
            let dest = unique_dest(&models_dir, &mm.filename, &target.repo);
            let bar = indicatif::ProgressBar::new(mm.bytes);
            bar.set_style(
                indicatif::ProgressStyle::default_bar()
                    .template("{msg} {bar:30} {bytes}/{total_bytes} ({eta})")
                    .expect("valid template"),
            );
            bar.set_message(format!("pull {name}:{}", selected.quant));
            let mut last_publish = 0u64;
            let mut progress = |downloaded: u64, total: u64| {
                bar.set_position(downloaded);
                if downloaded.saturating_sub(last_publish) >= 16 << 20 || downloaded == total {
                    last_publish = downloaded;
                    self.bus.publish(BlazarEvent::PullProgress {
                        name: name.to_string(),
                        downloaded,
                        total,
                    });
                }
            };
            self.client
                .download_file(&target.repo, mm, &dest, &mut progress)
                .await
                .inspect_err(|e| {
                    self.bus.publish(BlazarEvent::PullFailed {
                        name: name.to_string(),
                        error: format!("mmproj: {e}"),
                    });
                })?;
            bar.finish_and_clear();
            mmproj_dest = Some(dest);
        } else if let Some(old_mm) = &old.mmproj_path {
            // Repo dropped the projector: forget it from the row and
            // remove the dead sidecar file.
            if let Err(e) = std::fs::remove_file(old_mm) {
                if e.kind() != std::io::ErrorKind::NotFound {
                    tracing::warn!(model = %name, "could not remove stale mmproj {old_mm}: {e}");
                }
            }
        }
        build_model_row(
            name,
            target,
            info,
            selected,
            &[PathBuf::from(&old.path)],
            mmproj_dest.as_ref(),
        )
    }
}

/// Remove the replaced row's files (model shard set + sidecar),
/// skipping `keep`. Best-effort with named warnings; a shard set that
/// cannot be derived from the recorded leaf is left untouched (the
/// replacement then lands slug-disambiguated instead).
pub(crate) fn prune_replaced(name: &str, old: &ModelRow, keep: &[&Path], reason: &str) {
    let leaves = if old.shards <= 1 {
        vec![PathBuf::from(&old.path)]
    } else {
        let Some(set) = shard_set(&old.path, old.shards) else {
            tracing::warn!(
                model = %name,
                "cannot derive the {}-shard set of the previous download ({reason}); leaving files in place",
                old.shards
            );
            return;
        };
        set
    };
    let mmproj_leaf = old.mmproj_path.as_deref().map(Path::new);
    for leaf in leaves.iter().map(PathBuf::as_path).chain(mmproj_leaf) {
        if keep.contains(&leaf) {
            continue;
        }
        match std::fs::remove_file(leaf) {
            Ok(()) => {
                tracing::info!(model = %name, "pruned replaced file {} ({reason})", leaf.display());
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => tracing::warn!(
                model = %name,
                "could not prune replaced file {} ({e}) — remove it manually if disk space matters",
                leaf.display()
            ),
        }
    }
}

/// Post-download metadata + store row. Health warning: `None` when the
/// header parses; a LOUD warning otherwise (the engine will most likely
/// refuse to load the file — the qwen3.5-9b quantizer-metadata class of
/// failure). GGUF header facts win over HF-provided metadata.
#[allow(clippy::too_many_arguments)] // cohesive pull-facts tuple; splitting hides the fallback chain
fn build_model_row(
    name: &str,
    target: &PullTarget,
    info: &HfModelInfo,
    selected: &SelectedFiles,
    shard_paths: &[PathBuf],
    mmproj_dest: Option<&PathBuf>,
) -> Result<(ModelRow, Option<String>)> {
    let pull_warning = gguf_health_warning(&shard_paths[0], &target.repo, &info.siblings);
    let gguf_meta = gguf::read_metadata_file(&shard_paths[0]).ok();
    let arch = gguf_meta
        .as_ref()
        .map(|m| m.architecture.clone())
        .or_else(|| info.gguf.as_ref().and_then(|g| g.architecture.clone()));
    let ctx_train = gguf_meta
        .as_ref()
        .and_then(|m| m.context_length)
        .or_else(|| info.gguf.as_ref().and_then(|g| g.context_length));

    let bytes: u64 = selected.shards.iter().map(|s| s.bytes).sum();
    Ok((
        ModelRow {
            name: name.to_string(),
            repo: target.repo.clone(),
            quant: selected.quant.clone(),
            path: shard_paths[0].display().to_string(),
            bytes: i64::try_from(bytes).unwrap_or(i64::MAX),
            sha256: selected.shards[0].sha256.clone(),
            mmproj_path: mmproj_dest.as_ref().map(|d| d.display().to_string()),
            vae_path: None,
            llm_path: None,
            llm_vision_path: None,
            shards: i64::try_from(selected.shards.len()).unwrap_or(i64::MAX),
            arch,
            params: Some(est_params(bytes, &selected.quant)),
            ctx_train: ctx_train.and_then(|c| i64::try_from(c).ok()),
            pulled_at: i64::try_from(SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs())
                .unwrap_or(i64::MAX),
        },
        pull_warning,
    ))
}

/// Post-download GGUF health check: `None` when the header parses;
/// otherwise a warning naming the failure and the other quants of the
/// same repo (precheck for the load-refusal class of breakage).
#[must_use]
pub fn gguf_health_warning(path: &Path, repo: &str, siblings: &[HfSibling]) -> Option<String> {
    if gguf::read_metadata_file(path).is_ok() {
        return None;
    }
    let mut alts: Vec<String> = siblings
        .iter()
        .filter(|s| {
            let lower = s.rfilename.to_lowercase();
            let is_gguf = std::path::Path::new(&s.rfilename)
                .extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("gguf"));
            is_gguf && !lower.contains("-of-")
        })
        .filter_map(|s| {
            let lower = s.rfilename.to_lowercase();
            let stem = lower.strip_suffix(".gguf")?;
            stem.rsplit('-').next().map(str::to_uppercase)
        })
        .collect();
    alts.sort();
    alts.dedup();
    Some(format!(
        "GGUF metadata unreadable in {} — the engine will likely refuse to load it; `blazar rm` and try another quant of {repo}: {}",
        path.display(),
        alts.join(", ")
    ))
}

pub(crate) fn unique_dest(dir: &Path, filename: &str, repo: &str) -> PathBuf {
    let flat = Path::new(filename);
    let leaf = flat.file_name().unwrap_or_default();
    let dest = dir.join(leaf);
    if !dest.exists() {
        return dest;
    }
    // Collision: disambiguate with repo-derived path parts so two repos
    // shipping the same leaf name never clobber each other. Subdir path
    // first (matches the repo's internal layout), then the repo slug for
    // flat filenames (e.g. two Qwen3.5-9B-Q4_K_M.gguf from different
    // repos — verified live: unsloth base vs unsloth MTP collide).
    if let Some(parent) = flat.parent() {
        if !parent.as_os_str().is_empty() {
            let slug: String = parent.to_string_lossy().replace(['/', '\\'], "--");
            return dir.join(format!("{slug}--{}", leaf.to_string_lossy()));
        }
    }
    let repo_slug = repo.replace(['/', '\\'], "--");
    dir.join(format!("{repo_slug}--{}", leaf.to_string_lossy()))
}

/// A file at `candidate` that is byte-exact for `plan`: size gates a
/// full sha256 (hashing only runs on a size match, so the common miss
/// costs one stat — the rare hit is a multi-second multi-GiB read that
/// beats re-downloading the same bytes). Unknown published sha falls
/// back to size-only, matching the download lane's own verification
/// trust level.
fn reuse_byte_exact(candidate: &Path, plan: &FilePlan) -> Option<PathBuf> {
    let meta = std::fs::metadata(candidate).ok()?;
    if !meta.is_file() || meta.len() != plan.bytes || plan.bytes == 0 {
        return None;
    }
    match plan.sha256.as_deref() {
        Some(expected) if !expected.is_empty() => {
            let mut f = std::fs::File::open(candidate).ok()?;
            let mut h = Sha256::new();
            std::io::copy(&mut f, &mut h).ok()?;
            if format!("{:x}", h.finalize()).eq_ignore_ascii_case(expected) {
                Some(candidate.to_path_buf())
            } else {
                None
            }
        }
        _ => Some(candidate.to_path_buf()),
    }
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[test]
    fn unit__reuse_byte_exact__truth_table() {
        let tmp = tempfile::tempdir().unwrap();
        let body = b"component-bytes".to_vec();
        let f = tmp.path().join("te.gguf");
        std::fs::write(&f, &body).unwrap();
        let sha = {
            use sha2::Digest;
            let mut h = sha2::Sha256::new();
            h.update(&body);
            format!("{:x}", h.finalize())
        };
        let plan = |bytes: u64, sha: Option<&str>| FilePlan {
            filename: "te.gguf".into(),
            bytes,
            sha256: sha.map(str::to_string),
        };
        // Byte-exact (size + sha) reuses.
        assert_eq!(
            reuse_byte_exact(&f, &plan(body.len() as u64, Some(&sha))),
            Some(f.clone())
        );
        // Same size, different sha (hand-placed imposter) never reuses.
        let wrong = {
            use sha2::Digest;
            let mut h = sha2::Sha256::new();
            h.update(b"other");
            format!("{:x}", h.finalize())
        };
        assert_eq!(
            reuse_byte_exact(&f, &plan(body.len() as u64, Some(&wrong))),
            None
        );
        // Size mismatch short-circuits before any hashing.
        assert_eq!(reuse_byte_exact(&f, &plan(5, Some(&sha))), None);
        // Unknown published sha falls back to size-only (download-lane
        // verification trust level).
        assert_eq!(
            reuse_byte_exact(&f, &plan(body.len() as u64, None)),
            Some(f.clone())
        );
        // Missing file / zero-byte plan never reuse.
        assert_eq!(
            reuse_byte_exact(&tmp.path().join("nope"), &plan(1, Some(&sha))),
            None
        );
        assert_eq!(reuse_byte_exact(&f, &plan(0, None)), None);
    }

    #[test]
    fn unit__safetensors_byte_estimate__dtype_math_and_unknown_refusal() {
        use std::collections::BTreeMap;
        let hist = |pairs: &[(&str, u64)]| HfSafetensorsInfo {
            parameters: Some(
                pairs
                    .iter()
                    .map(|(k, v)| ((*k).to_string(), *v))
                    .collect::<BTreeMap<_, _>>(),
            ),
            total: None,
        };
        // Plain BF16 (MiniCPM5-2B shape): 2 bytes/element.
        assert_eq!(
            hist(&[("BF16", 2_516_756_480)]).byte_estimate(),
            Some(5_033_512_960)
        );
        // MLX 4-bit packing: histogram counts are wide-int CELLS (a
        // 1B-param repo live-shows {BF16: 33.8M, U32: 135M} because each
        // U32 cell packs 8 params), so on-disk bytes stay width × count:
        // 2×33.8M + 4×135M.
        assert_eq!(
            hist(&[("BF16", 33_842_688), ("U32", 135_069_696)]).byte_estimate(),
            Some(607_964_160)
        );
        // GPTQ shape: F16 embeddings kept + I32-packed weights.
        assert_eq!(
            hist(&[("F16", 803_402_992), ("I32", 497_025_024)]).byte_estimate(),
            Some(3_594_906_080)
        );
        // Unknown dtype: never guess — refuse the whole estimate.
        assert_eq!(hist(&[("BF16", 10), ("WEIRD_Q", 10)]).byte_estimate(), None);
        // Empty histogram / absent histogram: no estimate.
        assert_eq!(hist(&[]).byte_estimate(), None);
        assert_eq!(
            HfSafetensorsInfo {
                parameters: None,
                total: Some(1)
            }
            .byte_estimate(),
            None
        );
    }

    #[test]
    fn unit__search_path__format_filter_sentinels_and_browse() {
        // gguf lane keeps the historical filter.
        let gguf = search_path("minicpm", "gguf", 20);
        assert!(gguf.contains("filter=gguf"), "{gguf}");
        assert!(gguf.contains("search=minicpm"), "{gguf}");
        // any/all (case- and whitespace-tolerant) drop the filter entirely.
        for sentinel in ["any", "all", "ANY", " All "] {
            let p = search_path("q", sentinel, 5);
            assert!(!p.contains("filter="), "{sentinel}: {p}");
        }
        // Free-form tags pass through verbatim (trimmed + lowercased).
        assert!(search_path("q", " AWQ ", 5).contains("filter=awq"));
        assert!(search_path("q", "gptq", 5).contains("filter=gptq"));
        assert!(search_path("q", "mlx", 5).contains("filter=mlx"));
        // Empty query = browse mode: no search= param at all.
        assert!(!search_path("", "gguf", 5).contains("search="));
        // Every lane requests the metadata the table renders from.
        for fmt in ["gguf", "any", "awq"] {
            let p = search_path("x", fmt, 5);
            for need in [
                "expand[]=gguf",
                "expand[]=safetensors",
                "expand[]=config",
                "expand[]=siblings",
                "expand[]=tags",
            ] {
                assert!(p.contains(need), "{fmt} missing {need}: {p}");
            }
        }
    }

    #[test]
    fn unit__search_path__query_and_tag_percent_encoded() {
        // The encoder must be query-component safe: space, +, &, = all
        // percent-encoded, else a crafted query/format could splice extra
        // params into the Hub request.
        let p = search_path("mini cpm+", "a&b=c", 5);
        assert!(p.contains("search=mini%20cpm%2B"), "{p}");
        assert!(p.contains("filter=a%26b%3Dc"), "{p}");
    }

    #[test]
    fn unit__search_entry__decodes_live_hub_expand_shapes() {
        // Byte-for-byte shapes from a live `api/models?expand[]=…`
        // response (Qwen-Image-2.1 search, trimmed to the fields under
        // test): `totalFileSize` beside the parameter-count `total`, the
        // diffusers pipeline class, and repos where the Hub simply lacks
        // a field (leejet/* gguf carries no architecture).
        let decode = |body: &str| serde_json::from_str::<SearchEntry>(body).unwrap();
        let gguf_row = decode(
            r#"{"id":"Abiray/Qwen-Image-2.1-GGUF","gguf":{"total":7115124736,
                "architecture":"qwen_image","totalFileSize":3185944736}}"#,
        );
        assert_eq!(gguf_row.gguf.as_ref().unwrap().total, Some(7_115_124_736));
        assert_eq!(
            gguf_row.gguf.as_ref().unwrap().total_file_size,
            Some(3_185_944_736)
        );
        let no_arch_row = decode(
            r#"{"id":"leejet/Qwen-Image-2.1-GGUF","gguf":{"total":7115124736,
                "totalFileSize":2561716256}}"#,
        );
        assert_eq!(no_arch_row.gguf.as_ref().unwrap().architecture, None);
        let diffusers_row = decode(
            r#"{"id":"Qwen/Qwen-Image-2.1","config":{"diffusers":
                {"_class_name":"QwenImage21Pipeline"}}}"#,
        );
        assert_eq!(
            diffusers_row.config.as_ref().unwrap().arch_hint(),
            Some("qwen_image21".to_string())
        );
    }

    #[test]
    fn unit__pipeline_class_to_arch__camel_split_and_edge_cases() {
        // Pipeline classes → snake-case arch tokens; digits glue to the
        // preceding word.
        assert_eq!(
            pipeline_class_to_arch("QwenImage21Pipeline"),
            Some("qwen_image21".to_string())
        );
        assert_eq!(
            pipeline_class_to_arch("FluxPipeline"),
            Some("flux".to_string())
        );
        assert_eq!(
            pipeline_class_to_arch("StableDiffusion3Pipeline"),
            Some("stable_diffusion3".to_string())
        );
        // Degenerate inputs: nothing to show, no panic.
        assert_eq!(pipeline_class_to_arch("Pipeline"), None);
        assert_eq!(pipeline_class_to_arch(""), None);
    }

    /// Minimal `SearchEntry` — only the re-rank inputs matter.
    fn se(id: &str, downloads: u64, likes: u64) -> SearchEntry {
        SearchEntry {
            id: id.to_string(),
            downloads: Some(downloads),
            likes: Some(likes),
            siblings: Vec::new(),
            gguf: None,
            safetensors: None,
            config: None,
            tags: Vec::new(),
        }
    }

    #[test]
    fn unit__relevance_rank__bot_farmed_derivative_sinks_below_clean_mirrors() {
        // Live incident, distilled: the derivative sat at the exact same
        // 1.5M downloads as the official mirror (farmed 30-day count).
        let entries = vec![
            se("unsloth/Qwen3.5-9B-GGUF", 1_500_000, 925),
            se(
                "DavidAU/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMAX-GGUF",
                1_500_000,
                729,
            ),
            se("Jackrong/Qwen3.5-9B-DeepSeek-V4-Flash-GGUF", 411_200, 321),
        ];
        let ranked = relevance_rank(&entries, "qwen 3.5 9b gguf");
        assert_eq!(ranked[0].id, "unsloth/Qwen3.5-9B-GGUF");
        assert_eq!(ranked[1].id, "Jackrong/Qwen3.5-9B-DeepSeek-V4-Flash-GGUF");
        assert!(ranked[2].id.starts_with("DavidAU/"), "derivative last");
    }

    #[test]
    fn unit__relevance_rank__mtp_variant_outranks_junk_despite_fewer_downloads() {
        let entries = vec![
            se(
                "Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2-GGUF",
                24_900,
                395,
            ),
            se(
                "mradermacher/Qwen3.5-text-9B-NSFW-RP-RolePlay-i1-GGUF",
                29_100,
                7,
            ),
            se("unsloth/Qwen3.5-9B-MTP-GGUF", 70_300, 193),
        ];
        let ranked = relevance_rank(&entries, "qwen 3.5 9b gguf");
        assert_eq!(ranked[0].id, "unsloth/Qwen3.5-9B-MTP-GGUF");
    }

    #[test]
    fn unit__relevance_rank__exact_token_coverage_wins_over_likes() {
        // Full coverage beats partial even when the partial entry is
        // better liked: the user asked for THIS model, not a sibling.
        let entries = vec![
            se("someone/Qwen3.5-14B-GGUF", 900_000, 2_000), // misses "9b"
            se("other/Qwen3.5-9B-GGUF", 5_000, 10),         // 5/5 tokens
        ];
        let ranked = relevance_rank(&entries, "qwen 3.5 9b gguf");
        assert_eq!(ranked[0].id, "other/Qwen3.5-9B-GGUF");
    }

    #[test]
    fn unit__relevance_rank__deterministic_on_identical_keys() {
        let entries = vec![
            se("a/Qwen3.5-9B-GGUF", 1_000, 5),
            se("b/Qwen3.5-9B-GGUF", 1_000, 5),
        ];
        let ranked = relevance_rank(&entries, "qwen 3.5 9b gguf");
        assert_eq!(ranked[0].id, "a/Qwen3.5-9B-GGUF"); // input order kept
    }

    #[test]
    fn unit__relevance_rank__empty_query_is_identity() {
        // Browse mode: search() keeps Hub order, but the fn must also
        // degrade to input order if ever called without tokens.
        let entries = vec![se("z/One-GGUF", 1, 1), se("a/Two-GGUF", 2, 2)];
        let ranked = relevance_rank(&entries, "");
        assert_eq!(ranked[0].id, "z/One-GGUF"); // all-zero keys keep order
    }

    #[test]
    fn unit__fit_rows__safetensors_aggregate_when_no_gguf() {
        let sib = |name: &str, size: u64| HfSibling {
            rfilename: name.to_string(),
            size: None,
            lfs: Some(HfLfs {
                sha256: "x".to_string(),
                size: Some(size),
            }),
        };
        // Shards are parts of ONE quantization state: one aggregate row,
        // bytes summed, same KV ladder as the GGUF lane.
        let rows = fit_rows(
            &[
                sib("config.json", 512),
                sib("model-00001-of-00002.safetensors", 4_000_000_000),
                sib("model-00002-of-00002.safetensors", 2_000_000_000),
            ],
            24_000_000_000,
            32_768,
        );
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].quant, "safetensors");
        assert_eq!(rows[0].bytes, 6_000_000_000);
        assert_eq!(rows[0].file, "2 safetensors shard(s)");
        assert!(rows[0].fits_vram);
    }

    #[test]
    fn unit__fit_rows__gguf_present_beats_safetensors_lane() {
        let sib = |name: &str, size: u64| HfSibling {
            rfilename: name.to_string(),
            size: None,
            lfs: Some(HfLfs {
                sha256: "x".to_string(),
                size: Some(size),
            }),
        };
        // A repo with GGUF files is a GGUF repo: per-file rows only.
        let rows = fit_rows(
            &[
                sib("model-Q4_K_M.gguf", 1_000_000_000),
                sib("model.safetensors", 5_000_000_000),
            ],
            24_000_000_000,
            32_768,
        );
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].quant, "Q4_K_M");
    }

    #[test]
    fn unit__fit_rows__unsized_safetensors_stays_empty() {
        // No sizes from the Hub -> nothing honest to preview.
        let rows = fit_rows(
            &[HfSibling {
                rfilename: "model.safetensors".to_string(),
                size: None,
                lfs: None,
            }],
            24_000_000_000,
            32_768,
        );
        assert!(rows.is_empty());
    }

    #[test]
    fn unit__search_entry__tags_and_siblings_default_when_hub_omits() {
        let e: SearchEntry = serde_json::from_str(r#"{"id":"o/m"}"#).unwrap();
        assert_eq!(e.id, "o/m");
        assert!(e.tags.is_empty());
        assert!(e.siblings.is_empty());
        assert!(e.gguf.is_none());
    }

    #[test]
    fn unit__unique_dest__flat_collision_disambiguates_by_repo() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        std::fs::write(dir.join("m.gguf"), b"x").unwrap();
        // No collision: leaf name kept verbatim.
        assert_eq!(
            unique_dest(dir, "fresh.gguf", "unsloth/Base-GGUF"),
            dir.join("fresh.gguf")
        );
        // Flat collision: repo slug prefix — two repos shipping the same
        // leaf never clobber each other (live case: unsloth base vs MTP
        // repos both ship Qwen3.5-9B-Q4_K_M.gguf).
        assert_eq!(
            unique_dest(dir, "m.gguf", "unsloth/Qwen3.5-9B-MTP-GGUF"),
            dir.join("unsloth--Qwen3.5-9B-MTP-GGUF--m.gguf")
        );
        // Subdir filename collision: internal path slug wins (unchanged).
        assert_eq!(
            unique_dest(dir, "org/repo/m.gguf", "unsloth/Other"),
            dir.join("org--repo--m.gguf")
        );
    }

    #[test]
    fn unit__quant_tokens__real_filename_shapes() {
        // Shapes harvested live from HF nanbeige search results.
        let files = [
            "Nanbeige4.2-3B-Q4_K_M.gguf",      // dash separator
            "nanbeige4.1-3b-q8_0.gguf",        // lowercase repo style
            "nanbeige-16b-base-32k.Q2_K.gguf", // dot separator
            "Nanbeige4.2-3B-BF16.gguf",
            "Parable-Nanbeige4.2-3B-Claude-Fable-5-heretic.i1-IQ1_M.gguf",
            "model-00001-of-00002.gguf",       // shard tail: digits only
            "mmproj-model-F16.gguf",           // projector: excluded
            "README.md",                       // not gguf
            "Nanbeige4-3B-Thinking-2511.gguf", // tag tails: not quants
        ];
        assert_eq!(
            quant_tokens(files),
            ["BF16", "IQ1_M", "Q2_K", "Q4_K_M", "Q8_0"]
        );
    }

    #[test]
    fn unit__quant_tokens__version_glued_quant_after_underscore() {
        // Live shape (Abiray/AlperKTS Qwen-Image GGUFs): the version dot
        // ends the `.`/`-`-segment at `1_Q5_K_M`, so the quant only
        // parses as a `_`-suffix tail. Before this grammar extension the
        // QUANTS column showed `-` and `pull :Q5_K_M` fell back to the
        // smallest file.
        assert_eq!(
            quant_tokens([
                "qwen_image_2.1_Q3_K_M.gguf",
                "qwen_image_2.1_Q5_K_M.gguf",
                "qwen_image_2.1_Q8_0.gguf",
            ]),
            ["Q3_K_M", "Q5_K_M", "Q8_0"]
        );
        // Release-glued fp16 also parses; dates still don't.
        assert_eq!(
            quant_tokens(["model_2.1_fp16.gguf", "Nanbeige4-3B-2511.gguf"]),
            ["FP16"]
        );
    }

    #[test]
    fn unit__select_files__version_glued_quant_matches_request() {
        // Same repo shape as above, through the real pull path: the
        // requested quant must select its file, not the smallest
        // fallback.
        let sib = |name: &str| HfSibling {
            rfilename: name.to_string(),
            size: Some(1_000_000),
            lfs: None,
        };
        let siblings = [
            sib("qwen_image_2.1_Q5_K_M.gguf"),
            sib("qwen_image_2.1_Q8_0.gguf"),
            sib("README.md"),
        ];
        let picked = select_files(&siblings, "q5_k_m").unwrap();
        assert_eq!(picked.shards.len(), 1);
        assert_eq!(picked.shards[0].filename, "qwen_image_2.1_Q5_K_M.gguf");
        assert_eq!(picked.quant, "q5_k_m");
        assert!(!picked.quant_fallback);
    }

    #[test]
    fn unit__quant_tokens__dedups_case_insensitively() {
        let files = ["m.Q8_0.gguf", "m-q8_0.gguf"];
        assert_eq!(quant_tokens(files), ["Q8_0"]);
    }

    #[test]
    fn unit__quant_tokens__double_extension_and_folder_layouts() {
        // Double extension (live: ~200/4346 scanned GGUFs, e.g. Q3_K_S.GGUF.gguf
        // mirror shapes): the trailing `gguf` segment is noise, the segment
        // before it names the quant.
        assert_eq!(
            quant_tokens(["Q4_K_M.GGUF.gguf", "Q3_K_S.GGUF.gguf"]),
            ["Q3_K_S", "Q4_K_M"]
        );
        // Quant named in the parent folder, plain leaf (live: 5/4346 files).
        assert_eq!(quant_tokens(["Q4_K_M/model.gguf"]), ["Q4_K_M"]);
        assert_eq!(quant_tokens(["Q8_0/ggml-model.gguf"]), ["Q8_0"]);
        // Projector folders stay excluded exactly like mmproj leaves.
        assert_eq!(
            quant_tokens(["mmproj/model.gguf", "mmproj-f16.gguf"]),
            Vec::<String>::new()
        );
        // Non-quant folders and tag-tail leaves stay empty.
        assert_eq!(
            quant_tokens([
                "docs/qwen2.5.gguf",
                "model.gguf",
                "Nanbeige4-3B-Thinking.gguf"
            ]),
            Vec::<String>::new()
        );
        // Right-to-left order keeps the trailing segment authoritative.
        assert_eq!(quant_tokens(["model.q4.gguf"]), ["Q4"]);
    }

    #[test]
    fn unit__is_quant_token__grammar_exact_bare_tokens_only() {
        for yes in [
            "q4", "Q4_K_M", "iq4_xs", "f16", "BF16", "fp16", "q8_0_v2", "q4_0_4_4", "tq1_0",
        ] {
            assert!(is_quant_token(yes), "{yes}");
        }
        for no in [
            "qwen",
            "gguf",
            "awq",
            "gptq",
            "int4",
            "8bit",
            "q4.5",
            "qwen_image_2.1",
            "",
        ] {
            assert!(!is_quant_token(no), "{no}");
        }
    }

    #[test]
    fn unit__quant_tokens__no_gguf_files_is_empty() {
        assert!(quant_tokens(["README.md", "config.json"]).is_empty());
    }

    #[test]
    fn unit__extra_host_matches__exact_and_wildcard_semantics() {
        // Exact pattern: only itself.
        assert!(extra_host_matches(
            "registry.ollama.ai",
            "registry.ollama.ai"
        ));
        assert!(!extra_host_matches(
            "x.registry.ollama.ai",
            "registry.ollama.ai"
        ));
        // Wildcard: any subdomain, apex EXCLUDED (presigned CDN families
        // live on subdomains — the apex is a different, untrusted site).
        assert!(extra_host_matches(
            "blob-store.r2.cloudflarestorage.com",
            "*.r2.cloudflarestorage.com"
        ));
        assert!(!extra_host_matches(
            "r2.cloudflarestorage.com",
            "*.r2.cloudflarestorage.com"
        ));
        assert!(!extra_host_matches(
            "evil.r2.cloudflarestorage.com.attacker.io",
            "*.r2.cloudflarestorage.com"
        ));
        // Suffix without dot boundary must not match.
        assert!(!extra_host_matches(
            "xr2.cloudflarestorage.com",
            "*.r2.cloudflarestorage.com"
        ));
    }

    #[test]
    fn unit__gguf_health_warning__bad_header_names_alternatives() {
        let tmp = tempfile::tempdir().unwrap();
        let bad = tmp.path().join("bad.gguf");
        std::fs::write(&bad, b"not a gguf at all").unwrap();
        let sibs = vec![
            sib("m-Q4_K_M.gguf", 1, None),
            sib("m-Q8_0.gguf", 1, None),
            sib("m-00001-of-00002.gguf", 1, None), // shard: excluded
            sib("README.md", 1, None),             // not gguf: excluded
        ];
        let w = gguf_health_warning(&bad, "o/m", &sibs).expect("warning");
        assert!(w.contains("refuse to load"), "{w}");
        assert!(w.contains("o/m"), "{w}");
        assert!(w.contains("Q4_K_M") && w.contains("Q8_0"), "{w}");
        assert!(!w.contains("README"), "{w}");
    }

    #[test]
    fn unit__gguf_health_warning__valid_header_is_none() {
        // Minimal valid GGUF v3 header with one kv (architecture string).
        let tmp = tempfile::tempdir().unwrap();
        let good = tmp.path().join("good.gguf");
        let mut b: Vec<u8> = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        b.extend_from_slice(&1u64.to_le_bytes());
        let k = "general.architecture";
        b.extend_from_slice(&(k.len() as u64).to_le_bytes());
        b.extend_from_slice(k.as_bytes());
        b.extend_from_slice(&8u32.to_le_bytes());
        let v = "qwen3";
        b.extend_from_slice(&(v.len() as u64).to_le_bytes());
        b.extend_from_slice(v.as_bytes());
        std::fs::write(&good, b).unwrap();
        assert_eq!(gguf_health_warning(&good, "o/m", &[]), None);
    }

    fn sib(name: &str, size: u64, sha: Option<&str>) -> HfSibling {
        HfSibling {
            rfilename: name.to_string(),
            size: Some(size),
            lfs: sha.map(|s| HfLfs {
                sha256: s.to_string(),
                size: Some(size),
            }),
        }
    }

    fn sibling_json(name: &str, size: u64, sha: &str) -> serde_json::Value {
        serde_json::json!({
            "rfilename": name,
            "size": size,
            "lfs": {"sha256": sha, "size": size, "pointerSize": 135}
        })
    }

    #[test]
    fn unit__quant_token_of__dotted_and_dashed_segments_agree() {
        // Live regression: DevQuasar-style filenames carry the quant in
        // a '.'-separated tail; the old select_files grammar split on
        // '-' only and mis-pulled these as UNKNOWN.
        assert_eq!(
            quant_token_of("amd.instella-moe-16b-a3b-think.f16.gguf.Q2_K.gguf"),
            Some("q2_k".to_string())
        );
        assert_eq!(quant_token_of("m-q4_k_m.gguf"), Some("q4_k_m".to_string()));
        // Shard bases carry no .gguf suffix.
        assert_eq!(quant_token_of("m-q4_k_m"), Some("q4_k_m".to_string()));
        assert_eq!(
            quant_token_of("Model-IQ4_XS.gguf"),
            Some("iq4_xs".to_string())
        );
        assert_eq!(quant_token_of("m-fp16.gguf"), Some("fp16".to_string()));
        assert_eq!(quant_token_of("m-f16.gguf"), Some("f16".to_string()));
        assert_eq!(quant_token_of("m-f32.gguf"), Some("f32".to_string()));
        // Non-quants stay non-quants in both spellings.
        assert_eq!(quant_token_of("nanbeige-16b-base-32k.gguf"), None);
        assert_eq!(quant_token_of("model-heretic.gguf"), None);
        assert_eq!(quant_token_of("model-2025-11.gguf"), None);
        assert_eq!(quant_token_of("mmproj-f16.gguf"), None);
        // `q`-prefixed finetune words need the digit to qualify.
        assert_eq!(quant_token_of("model-quiet.gguf"), None);
    }

    #[test]
    fn unit__quant_token_of__strips_shard_tails() {
        assert_eq!(
            quant_token_of("m-q4_k_m-00001-of-00005.gguf"),
            Some("q4_k_m".to_string())
        );
        // Shard markers only: a real `-of-` word mid-name is not one.
        assert_eq!(strip_shard_tail("m-q4_k_m"), "m-q4_k_m");
        assert_eq!(strip_shard_tail("b-00001-of-00002"), "b");
    }

    #[test]
    fn unit__select_files__dotted_quant_tail__matches_requested_quant() {
        // The exact live shape that regressed: without '.'-splitting the
        // request fell through to the smallest-file fallback with quant
        // UNKNOWN.
        let sibs = vec![
            sib(
                "amd.instella-moe-16b-a3b-think.f16.gguf.Q2_K.gguf",
                6000,
                None,
            ),
            sib(
                "amd.instella-moe-16b-a3b-think.f16.gguf.Q4_K.gguf",
                9000,
                None,
            ),
        ];
        let sel = select_files(&sibs, "q2_k").unwrap();
        assert_eq!(
            sel.shards[0].filename,
            "amd.instella-moe-16b-a3b-think.f16.gguf.Q2_K.gguf"
        );
        assert_eq!(sel.quant, "q2_k");
        assert!(!sel.quant_fallback);
    }

    #[test]
    fn unit__registry_name__strips_gguf_and_lowercases() {
        assert_eq!(
            registry_name("Qwen/Qwen2.5-0.5B-Instruct-GGUF"),
            "qwen2.5-0.5b-instruct"
        );
        assert_eq!(registry_name("ggml-org/Qwen3-0.6B-GGUF"), "qwen3-0.6b");
        assert_eq!(registry_name("a/b"), "b");
    }

    #[test]
    fn unit__parse_target__explicit_repo_with_quant() {
        let t = parse_pull_target("Qwen/Qwen2.5-0.5B-Instruct-GGUF:q8_0").unwrap();
        assert_eq!(t.repo, "Qwen/Qwen2.5-0.5B-Instruct-GGUF");
        assert_eq!(t.quant, "Q8_0");
    }

    #[test]
    fn unit__parse_target__catalog_short_name() {
        let t = parse_pull_target("qwen3-0.6b").unwrap();
        assert_eq!(t.repo, "ggml-org/Qwen3-0.6B-GGUF");
        assert_eq!(t.quant, "Q4_K_M");
    }

    #[test]
    fn unit__parse_target__catalog_prefix_resolves() {
        let t = parse_pull_target("qwen3-coder").unwrap();
        assert!(t.repo.starts_with("unsloth/Qwen3-Coder"));
    }

    #[test]
    fn unit__select_files__quant_match_case_insensitive() {
        let sibs = vec![
            sib("model-fp16.gguf", 1000, None),
            sib("model-Q4_K_M.gguf", 400, None),
        ];
        let sel = select_files(&sibs, "q4_k_m").unwrap();
        assert_eq!(sel.shards.len(), 1);
        assert_eq!(sel.shards[0].filename, "model-Q4_K_M.gguf");
        assert!(!sel.quant_fallback);
    }

    #[test]
    fn unit__select_files__same_quant_multiple_singles__largest_wins() {
        // ggml-org layout: dflash/dspark drafters share the quant token
        // with the target. The MODEL is the largest same-quant file.
        let sibs = vec![
            sib("dflash-Qwen3-8B-Q8_0.gguf", 1120, None),
            sib("Qwen3-8B-Q8_0.gguf", 8710, None),
            sib("dspark-Qwen3-8B-Q8_0.gguf", 1200, None),
        ];
        let sel = select_files(&sibs, "Q8_0").unwrap();
        assert_eq!(sel.shards[0].filename, "Qwen3-8B-Q8_0.gguf");
        assert!(!sel.quant_fallback);
    }

    #[test]
    fn unit__select_files__exact_filename_slot__selects_drafter() {
        let sibs = vec![
            sib("dflash-Qwen3-8B-Q8_0.gguf", 1120, None),
            sib("Qwen3-8B-Q8_0.gguf", 8710, None),
        ];
        // The quant slot carrying a .gguf leaf = exact-file request; case
        // and repo-path insensitive.
        let sel = select_files(&sibs, "dflash-qwen3-8b-q8_0.gguf").unwrap();
        assert_eq!(sel.shards[0].filename, "dflash-Qwen3-8B-Q8_0.gguf");
        assert!(!sel.quant_fallback);
        // Display quant = trailing token of the filename, NOT the whole slot.
        assert_eq!(sel.quant, "Q8_0");
        // No such file: named error, no silent fallback.
        let err = select_files(&sibs, "nope.gguf").unwrap_err();
        assert!(err.to_string().contains("no file matching"), "{err}");
    }

    #[test]
    fn unit__draft_aware_name__filename_slot_names_row_by_stem() {
        assert_eq!(
            draft_aware_name("ggml-org/Qwen3-8B-GGUF", "dflash-Qwen3-8B-Q8_0.gguf"),
            "dflash-qwen3-8b-q8_0"
        );
        // Quant-style slots keep the repo-tail name.
        assert_eq!(
            draft_aware_name("ggml-org/Qwen3-0.6B-GGUF", "Q4_0"),
            "qwen3-0.6b"
        );
    }

    #[test]
    fn unit__select_files__shard_set_enumerated_in_order() {
        let sibs = vec![
            sib("big-q4_k_m-00002-of-00002.gguf", 700, None),
            sib("big-q4_k_m-00001-of-00002.gguf", 700, None),
            sib("big-q8_0.gguf", 1200, None),
        ];
        let sel = select_files(&sibs, "Q4_K_M").unwrap();
        assert_eq!(sel.shards.len(), 2, "both shards queued");
        assert_eq!(sel.shards[0].filename, "big-q4_k_m-00001-of-00002.gguf");
        assert_eq!(sel.shards[1].filename, "big-q4_k_m-00002-of-00002.gguf");
    }

    #[test]
    fn unit__select_files__mmproj_detected() {
        let sibs = vec![
            sib("model-q4_k_m.gguf", 400, None),
            sib("mmproj-model-f16.gguf", 50, None),
            sib("mmproj-model-q8_0.gguf", 30, None),
        ];
        let sel = select_files(&sibs, "Q4_K_M").unwrap();
        assert_eq!(
            sel.mmproj.as_ref().unwrap().filename,
            "mmproj-model-q8_0.gguf"
        );
    }

    #[test]
    fn unit__select_files__default_quant_fallback_smallest() {
        let sibs = vec![
            sib("model-q8_0.gguf", 900, None),
            sib("model-q3_k_m.gguf", 300, None),
        ];
        let sel = select_files(&sibs, "Q4_K_M").unwrap();
        assert!(sel.quant_fallback);
        assert_eq!(sel.quant, "Q3_K_M");
        assert_eq!(sel.shards[0].filename, "model-q3_k_m.gguf");
    }

    #[test]
    fn unit__select_files__no_gguf__named_error() {
        let sibs = vec![sib("README.md", 10, None)];
        let err = select_files(&sibs, "Q4_K_M").unwrap_err();
        assert!(err.to_string().contains("no .gguf"));
    }

    #[test]
    fn unit__est_params__sane_order_of_magnitude() {
        // 0.5B-class Q4_K_M file ~ 400 MiB
        let p = est_params(400 * 1024 * 1024, "Q4_K_M");
        assert!(p > 0.3 && p < 1.2, "got {p}B");
    }

    #[test]
    fn unit__allowlist__known_hosts_only() {
        assert!(is_allowed_download_host("huggingface.co"));
        assert!(is_allowed_download_host("cdn-lfs.huggingface.co"));
        assert!(is_allowed_download_host("cas-bridge.xethub.hf.co"));
        assert!(is_allowed_download_host("transfer.xethub.hf.co"));
        assert!(!is_allowed_download_host("evil.example.com"));
        assert!(!is_allowed_download_host("huggingface.co.evil.com"));
    }

    // ------------------------------------------------------------------
    // wiremock integration
    // ------------------------------------------------------------------

    fn payload(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    /// Minimal valid GGUF v3 file (header + one string kv) — passes
    /// `gguf::read_metadata_file`.
    fn gguf_bytes() -> Vec<u8> {
        let mut b: Vec<u8> = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes()); // tensor count
        b.extend_from_slice(&1u64.to_le_bytes()); // kv count
        let k = "general.architecture";
        b.extend_from_slice(&(k.len() as u64).to_le_bytes());
        b.extend_from_slice(k.as_bytes());
        b.extend_from_slice(&8u32.to_le_bytes()); // string type
        let v = "qwen3";
        b.extend_from_slice(&(v.len() as u64).to_le_bytes());
        b.extend_from_slice(v.as_bytes());
        b
    }

    fn seed_row(
        name: &str,
        repo: &str,
        quant: &str,
        path: &str,
        bytes: u64,
        shards: i64,
    ) -> ModelRow {
        ModelRow {
            name: name.to_string(),
            repo: repo.to_string(),
            quant: quant.to_string(),
            path: path.to_string(),
            bytes: i64::try_from(bytes).unwrap_or(i64::MAX),
            sha256: None,
            mmproj_path: None,
            vae_path: None,
            llm_path: None,
            llm_vision_path: None,
            shards,
            arch: None,
            params: None,
            ctx_train: None,
            pulled_at: 0,
        }
    }

    #[test]
    fn unit__flip_guard__format_flip_refused_until_forced() {
        let tmp = tempfile::tempdir().unwrap();
        let dir_row = seed_row("m", "r", "BF16", tmp.path().to_str().unwrap(), 1, 1);
        let gguf = tmp.path().join("m.gguf");
        std::fs::write(&gguf, b"x").unwrap();
        let gguf_row = seed_row("m", "r", "Q4_K_M", gguf.to_str().unwrap(), 1, 1);

        // GGUF pull over a safetensors dir: refused, names both
        // formats, teaches --force (the 2026-09-17 short-name incident).
        let err = flip_guard("m", Some(&dir_row), false, false).expect_err("flip refused");
        let msg = format!("{err:#}");
        assert!(msg.contains("safetensors directory"), "{msg}");
        assert!(msg.contains("--force"), "{msg}");

        // Reverse direction: safetensors pull over a GGUF row.
        let err = flip_guard("m", Some(&gguf_row), true, false).expect_err("flip refused");
        assert!(format!("{err:#}").contains("a GGUF"), "{err:#}");

        // Same format, forced flip, and a fresh name all pass.
        assert!(flip_guard("m", Some(&dir_row), true, false).is_ok());
        assert!(flip_guard("m", Some(&dir_row), false, true).is_ok());
        assert!(flip_guard("m", None, false, false).is_ok());
    }

    #[test]
    fn unit__model_file_intact__truth_table() {
        let tmp = tempfile::tempdir().unwrap();
        let good = tmp.path().join("good.gguf");
        let content = gguf_bytes();
        std::fs::write(&good, &content).unwrap();
        let row = |path: &str, bytes: u64, shards: i64| {
            seed_row("m", "o/r", "Q4_K_M", path, bytes, shards)
        };

        // Exact length + parseable header.
        assert!(model_file_intact(&row(
            good.to_str().unwrap(),
            content.len() as u64,
            1
        )));
        // Recorded length disagrees (truncation/corruption).
        assert!(!model_file_intact(&row(good.to_str().unwrap(), 5, 1)));
        // Missing file.
        assert!(!model_file_intact(&row(
            tmp.path().join("nope.gguf").to_str().unwrap(),
            0,
            1
        )));
        // Garbage header at the "right" length.
        let bad = tmp.path().join("bad.gguf");
        std::fs::write(&bad, vec![0u8; content.len()]).unwrap();
        assert!(!model_file_intact(&row(
            bad.to_str().unwrap(),
            content.len() as u64,
            1
        )));
        // Diffusion component GGUF (0 KVs, valid container): the
        // "missing general.architecture" parse still proves the
        // container walked clean — intact, or every re-pull would
        // prune + redownload a sound 4.6 GiB DiT forever.
        let mut kvless = b"GGUF".to_vec();
        kvless.extend_from_slice(&3u32.to_le_bytes());
        kvless.extend_from_slice(&297u64.to_le_bytes());
        kvless.extend_from_slice(&0u64.to_le_bytes());
        let dit = tmp.path().join("dit.gguf");
        std::fs::write(&dit, &kvless).unwrap();
        assert!(model_file_intact(&row(
            dit.to_str().unwrap(),
            kvless.len() as u64,
            1
        )));
        // Sharded: header-only check on the launch shard (bytes is a SUM
        // across files, never any single file's length).
        assert!(model_file_intact(&row(good.to_str().unwrap(), 999_999, 2)));
        assert!(!model_file_intact(&row(bad.to_str().unwrap(), 8, 2)));
    }

    #[tokio::test]
    async fn integration__repull__already_present_skips_download() {
        let tmp = tempfile::tempdir().unwrap();
        let dirs = BlazarDirs {
            config_dir: tmp.path().join("cfg"),
            data_dir: tmp.path().join("data"),
        };
        dirs.ensure().unwrap();

        // Intact on-disk state: valid GGUF at the canonical leaf + a store
        // row whose repo/quant match what select_files would choose.
        let content = gguf_bytes();
        let sha = payload(&content);
        let leaf = dirs.models_dir().join("r-q4_k_m.gguf");
        std::fs::write(&leaf, &content).unwrap();
        let row = seed_row(
            "r",
            "o/r",
            "Q4_K_M",
            &leaf.display().to_string(),
            content.len() as u64,
            1,
        );
        Store::open(&dirs).unwrap().upsert_model(&row).unwrap();
        // Stale partials from an interrupted attempt are garbage once the
        // final leaf verifies — the fast path sweeps them.
        std::fs::write(dirs.models_dir().join("r-q4_k_m.gguf.part"), b"junk").unwrap();
        std::fs::write(dirs.models_dir().join("r-q4_k_m.gguf.part.progress"), b"{}").unwrap();

        let api = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/models/o/r"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "siblings": [ sibling_json("r-q4_k_m.gguf", content.len() as u64, &sha) ]
            })))
            .mount(&api)
            .await;
        // Download server with ZERO mocks: any download attempt would 404
        // and fail the pull — the fast path must not touch it at all.
        let dl = MockServer::start().await;

        let client = HfClient::with_bases(
            &api.uri(),
            &dl.uri(),
            None,
            vec![host_of(&api.uri()), host_of(&dl.uri())],
        )
        .unwrap();
        let puller = Puller {
            dirs: dirs.clone(),
            client,
            bus: EventBus::default(),
            force: false,
        };
        let outcome = puller.pull("o/r").await.unwrap();

        assert!(outcome.already_present, "gate must short-circuit");
        assert_eq!(outcome.row.path, leaf.display().to_string());
        assert!(
            dl.received_requests().await.unwrap_or_default().is_empty(),
            "fast path must not fetch a single download byte"
        );
        assert!(!dirs.models_dir().join("r-q4_k_m.gguf.part").exists());
        assert!(!dirs
            .models_dir()
            .join("r-q4_k_m.gguf.part.progress")
            .exists());
        // Store row NOT clobbered by a re-derived slug twin.
        assert_eq!(
            Store::open(&dirs)
                .unwrap()
                .get_model("r")
                .unwrap()
                .unwrap()
                .path,
            leaf.display().to_string()
        );
    }

    #[tokio::test]
    async fn integration__repull__self_heal_replaces_corrupt_leaf() {
        let tmp = tempfile::tempdir().unwrap();
        let dirs = BlazarDirs {
            config_dir: tmp.path().join("cfg"),
            data_dir: tmp.path().join("data"),
        };
        dirs.ensure().unwrap();

        // Store row points at a truncated file: integrity fails, so the
        // pull must clear the dead leaf and land the fresh download on
        // the CANONICAL filename — not the owner--repo--leaf slug.
        let content = b"fresh-bytes-here".to_vec();
        let sha = payload(&content);
        let leaf = dirs.models_dir().join("r-q4_k_m.gguf");
        std::fs::write(&leaf, b"junk").unwrap();
        let row = seed_row(
            "r",
            "o/r",
            "Q4_K_M",
            &leaf.display().to_string(),
            content.len() as u64,
            1,
        );
        Store::open(&dirs).unwrap().upsert_model(&row).unwrap();

        let api = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/models/o/r"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "siblings": [ sibling_json("r-q4_k_m.gguf", content.len() as u64, &sha) ]
            })))
            .mount(&api)
            .await;
        let dl = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/o/r/resolve/main/r-q4_k_m.gguf"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(content.clone()))
            .mount(&dl)
            .await;

        let client = HfClient::with_bases(
            &api.uri(),
            &dl.uri(),
            None,
            vec![host_of(&api.uri()), host_of(&dl.uri())],
        )
        .unwrap();
        let puller = Puller {
            dirs: dirs.clone(),
            client,
            bus: EventBus::default(),
            force: false,
        };
        let outcome = puller.pull("o/r").await.unwrap();

        assert!(!outcome.already_present);
        assert_eq!(
            outcome.row.path,
            leaf.display().to_string(),
            "fresh download must land on the canonical leaf"
        );
        assert_eq!(std::fs::read(&leaf).unwrap(), content);
        assert!(
            !dirs.models_dir().join("o--r--r-q4_k_m.gguf").exists(),
            "no collision-slug twin"
        );
        // Row healed: byte count matches the fresh file.
        assert_eq!(
            Store::open(&dirs)
                .unwrap()
                .get_model("r")
                .unwrap()
                .unwrap()
                .bytes,
            i64::try_from(content.len()).unwrap()
        );
    }

    #[test]
    fn unit__shard_set__derives_all_leaves_and_rejects_non_shard() {
        let set = shard_set("/m/big-q4_k_m-00001-of-00003.gguf", 3).unwrap();
        assert_eq!(
            set,
            vec![
                std::path::PathBuf::from("/m/big-q4_k_m-00001-of-00003.gguf"),
                std::path::PathBuf::from("/m/big-q4_k_m-00002-of-00003.gguf"),
                std::path::PathBuf::from("/m/big-q4_k_m-00003-of-00003.gguf"),
            ]
        );
        // A single-file name does not follow the shard pattern.
        assert!(shard_set("/m/big-q4_k_m.gguf", 2).is_none());
    }

    #[tokio::test]
    async fn integration__requant__downloads_new_and_prunes_old() {
        let tmp = tempfile::tempdir().unwrap();
        let dirs = BlazarDirs {
            config_dir: tmp.path().join("cfg"),
            data_dir: tmp.path().join("data"),
        };
        dirs.ensure().unwrap();

        let q4 = b"q4-bytes".to_vec();
        let q8 = gguf_bytes();
        let q4_leaf = dirs.models_dir().join("r-q4_k_m.gguf");
        std::fs::write(&q4_leaf, &q4).unwrap();
        let row = seed_row(
            "r",
            "o/r",
            "Q4_K_M",
            &q4_leaf.display().to_string(),
            q4.len() as u64,
            1,
        );
        Store::open(&dirs).unwrap().upsert_model(&row).unwrap();

        let api = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/models/o/r"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "siblings": [
                    sibling_json("r-q4_k_m.gguf", q4.len() as u64, &payload(&q4)),
                    sibling_json("r-q8_0.gguf", q8.len() as u64, &payload(&q8)),
                ]
            })))
            .mount(&api)
            .await;
        let dl = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/o/r/resolve/main/r-q8_0.gguf"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(q8.clone()))
            .mount(&dl)
            .await;

        let client = HfClient::with_bases(
            &api.uri(),
            &dl.uri(),
            None,
            vec![host_of(&api.uri()), host_of(&dl.uri())],
        )
        .unwrap();
        let puller = Puller {
            dirs: dirs.clone(),
            client,
            bus: EventBus::default(),
            force: false,
        };
        let outcome = puller.pull("o/r:Q8_0").await.unwrap();

        assert!(!outcome.already_present);
        assert_eq!(outcome.row.quant, "Q8_0");
        assert!(dirs.models_dir().join("r-q8_0.gguf").is_file());
        assert!(
            !q4_leaf.exists(),
            "superseded quant file must be pruned, not orphaned"
        );
    }

    #[tokio::test]
    async fn integration__repull__upstream_sha_change_replaces_canonically() {
        let tmp = tempfile::tempdir().unwrap();
        let dirs = BlazarDirs {
            config_dir: tmp.path().join("cfg"),
            data_dir: tmp.path().join("data"),
        };
        dirs.ensure().unwrap();

        // Recorded sha is the OLD revision; the API now lists a different
        // sha under the same filename — the pull must replace the leaf in
        // place (canonical landing), not slug-disambiguate.
        let old_content = gguf_bytes();
        let new_content = b"new-revision-bytes".to_vec();
        let old_sha = payload(&old_content);
        let new_sha = payload(&new_content);
        let leaf = dirs.models_dir().join("r-q4_k_m.gguf");
        std::fs::write(&leaf, &old_content).unwrap();
        let row = seed_row(
            "r",
            "o/r",
            "Q4_K_M",
            &leaf.display().to_string(),
            old_content.len() as u64,
            1,
        );
        let mut row = row;
        row.sha256 = Some(old_sha);
        Store::open(&dirs).unwrap().upsert_model(&row).unwrap();

        let api = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/models/o/r"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "siblings": [ sibling_json("r-q4_k_m.gguf", new_content.len() as u64, &new_sha) ]
            })))
            .mount(&api)
            .await;
        let dl = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/o/r/resolve/main/r-q4_k_m.gguf"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(new_content.clone()))
            .mount(&dl)
            .await;

        let client = HfClient::with_bases(
            &api.uri(),
            &dl.uri(),
            None,
            vec![host_of(&api.uri()), host_of(&dl.uri())],
        )
        .unwrap();
        let puller = Puller {
            dirs: dirs.clone(),
            client,
            bus: EventBus::default(),
            force: false,
        };
        let outcome = puller.pull("o/r").await.unwrap();

        assert!(!outcome.already_present);
        assert_eq!(
            outcome.row.path,
            leaf.display().to_string(),
            "replacement must land on the canonical leaf"
        );
        assert_eq!(std::fs::read(&leaf).unwrap(), new_content);
        assert!(
            !dirs.models_dir().join("o--r--r-q4_k_m.gguf").exists(),
            "no collision-slug twin"
        );
        assert_eq!(
            Store::open(&dirs)
                .unwrap()
                .get_model("r")
                .unwrap()
                .unwrap()
                .sha256
                .as_deref(),
            Some(new_sha.as_str()),
            "row records the new revision"
        );
    }

    #[tokio::test]
    async fn integration__repull__delta_mmproj_downloads_only_sidecar() {
        let tmp = tempfile::tempdir().unwrap();
        let dirs = BlazarDirs {
            config_dir: tmp.path().join("cfg"),
            data_dir: tmp.path().join("data"),
        };
        dirs.ensure().unwrap();

        // Intact model on disk (sha matches the API); the repo now also
        // ships a projector. Only the sidecar may move.
        let model = gguf_bytes();
        let model_sha = payload(&model);
        let mm = b"mmproj-bytes".to_vec();
        let mm_sha = payload(&mm);
        let leaf = dirs.models_dir().join("r-q4_k_m.gguf");
        std::fs::write(&leaf, &model).unwrap();
        let row = seed_row(
            "r",
            "o/r",
            "Q4_K_M",
            &leaf.display().to_string(),
            model.len() as u64,
            1,
        );
        let mut row = row;
        row.sha256 = Some(model_sha.clone());
        Store::open(&dirs).unwrap().upsert_model(&row).unwrap();

        let api = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/models/o/r"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "siblings": [
                    sibling_json("r-q4_k_m.gguf", model.len() as u64, &model_sha),
                    sibling_json("mmproj-r-f16.gguf", mm.len() as u64, &mm_sha),
                ]
            })))
            .mount(&api)
            .await;
        // ONLY the sidecar endpoint is mounted: a model re-fetch would
        // 404 and fail the pull.
        let dl = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/o/r/resolve/main/mmproj-r-f16.gguf"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(mm.clone()))
            .mount(&dl)
            .await;

        let client = HfClient::with_bases(
            &api.uri(),
            &dl.uri(),
            None,
            vec![host_of(&api.uri()), host_of(&dl.uri())],
        )
        .unwrap();
        let puller = Puller {
            dirs: dirs.clone(),
            client,
            bus: EventBus::default(),
            force: false,
        };
        let outcome = puller.pull("o/r").await.unwrap();

        assert!(!outcome.already_present, "sidecar bytes did move");
        assert_eq!(
            outcome.row.mmproj_path.as_deref(),
            Some(
                dirs.models_dir()
                    .join("mmproj-r-f16.gguf")
                    .display()
                    .to_string()
                    .as_str()
            ),
            "row gains the sidecar"
        );
        assert_eq!(std::fs::read(&leaf).unwrap(), model, "model file untouched");
        assert_eq!(
            std::fs::read(dirs.models_dir().join("mmproj-r-f16.gguf")).unwrap(),
            mm
        );
        let model_hits = dl
            .received_requests()
            .await
            .unwrap_or_default()
            .iter()
            .filter(|rq| rq.url.path().ends_with("r-q4_k_m.gguf"))
            .count();
        assert_eq!(model_hits, 0, "model must not be re-downloaded");
    }

    #[tokio::test]
    async fn integration__pull_single_file__downloads_verifies_stores() {
        let tmp = tempfile::tempdir().unwrap();
        let dirs = BlazarDirs {
            config_dir: tmp.path().join("cfg"),
            data_dir: tmp.path().join("data"),
        };
        dirs.ensure().unwrap();

        let content = b"gguf-bytes-here".to_vec();
        let sha = payload(&content);
        let api = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/models/owner/m-repo"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "owner/m-repo",
                "siblings": [ sibling_json("m-repo-q4_k_m.gguf", content.len() as u64, &sha) ]
            })))
            .mount(&api)
            .await;
        let dl = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/owner/m-repo/resolve/main/m-repo-q4_k_m.gguf"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(content.clone()))
            .mount(&dl)
            .await;

        let client = HfClient::with_bases(
            &api.uri(),
            &dl.uri(),
            None,
            vec![
                api.uri().trim_start_matches("http://").to_string(),
                dl.uri().trim_start_matches("http://").to_string(),
            ],
        )
        .unwrap();
        let puller = Puller {
            dirs: dirs.clone(),
            client,
            bus: EventBus::default(),
            force: false,
        };
        let row = puller.pull("owner/m-repo:Q4_K_M").await.unwrap().row;

        assert_eq!(row.name, "m-repo");
        assert_eq!(row.quant, "Q4_K_M");
        let path = Path::new(&row.path);
        assert!(path.exists(), "file must land at stable path");
        assert_eq!(std::fs::read(path).unwrap(), content);
        assert!(!Path::new(&format!("{}.part", row.path)).exists());

        let store = Store::open(&dirs).unwrap();
        assert_eq!(store.get_model("m-repo").unwrap().unwrap().quant, "Q4_K_M");
    }

    #[tokio::test]
    async fn integration__pull_sha_mismatch__partial_deleted_and_error() {
        let tmp = tempfile::tempdir().unwrap();
        let dirs = BlazarDirs {
            config_dir: tmp.path().join("cfg"),
            data_dir: tmp.path().join("data"),
        };
        dirs.ensure().unwrap();
        let content = b"corrupt-me".to_vec();
        let wrong_sha = payload(b"something-else");
        let api = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/models/o/r"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "siblings": [ sibling_json("r-q4_k_m.gguf", content.len() as u64, &wrong_sha) ]
            })))
            .mount(&api)
            .await;
        let dl = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/o/r/resolve/main/r-q4_k_m.gguf"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(content))
            .mount(&dl)
            .await;
        let client = HfClient::with_bases(
            &api.uri(),
            &dl.uri(),
            None,
            vec![host_of(&api.uri()), host_of(&dl.uri())],
        )
        .unwrap();
        let puller = Puller {
            dirs: dirs.clone(),
            client,
            bus: EventBus::default(),
            force: false,
        };
        let err = puller.pull("o/r").await.unwrap_err();
        assert!(err.to_string().contains("sha256 mismatch"), "{err}");
        assert!(
            dirs.models_dir().read_dir().unwrap().count() == 0,
            "no partial left behind"
        );
        assert!(Store::open(&dirs)
            .unwrap()
            .list_models()
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn integration__resume_from_part_file__range_and_append() {
        let tmp = tempfile::tempdir().unwrap();
        let dirs = BlazarDirs {
            config_dir: tmp.path().join("cfg"),
            data_dir: tmp.path().join("data"),
        };
        dirs.ensure().unwrap();
        let full = b"0123456789abcdef".to_vec();
        let sha = payload(&full);
        let prefix = &full[..8];

        let api = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/models/o/r"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "siblings": [ sibling_json("r-q4_k_m.gguf", full.len() as u64, &sha) ]
            })))
            .mount(&api)
            .await;
        let dl = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/o/r/resolve/main/r-q4_k_m.gguf"))
            .and(wiremock::matchers::header("Range", "bytes=8-"))
            .respond_with(
                ResponseTemplate::new(206)
                    .insert_header("Content-Range", "bytes 8-15/16")
                    .set_body_bytes(full[8..].to_vec()),
            )
            .mount(&dl)
            .await;

        // Pre-existing partial.
        std::fs::write(dirs.models_dir().join("r-q4_k_m.gguf.part"), prefix).unwrap();

        let client = HfClient::with_bases(
            &api.uri(),
            &dl.uri(),
            None,
            vec![host_of(&api.uri()), host_of(&dl.uri())],
        )
        .unwrap();
        let puller = Puller {
            dirs,
            client,
            bus: EventBus::default(),
            force: false,
        };
        let row = puller.pull("o/r").await.unwrap().row;
        let got = std::fs::read(&row.path).unwrap();
        assert_eq!(got, full, "resumed file must equal full content");
    }

    #[tokio::test]
    async fn integration__resume_full_length_part__restarts_from_zero() {
        // A `.part` already at the expected size is a sparse parallel-lane
        // artifact (or a stale size): `bytes=have-` would 416. The classic
        // lane must restart from zero instead of Range-requesting at EOF.
        let tmp = tempfile::tempdir().unwrap();
        let dirs = BlazarDirs {
            config_dir: tmp.path().join("cfg"),
            data_dir: tmp.path().join("data"),
        };
        dirs.ensure().unwrap();
        let full = b"0123456789abcdef".to_vec();
        let sha = payload(&full);

        let dl = MockServer::start().await;
        // Any Range request answers 416 (what a real server does at EOF);
        // the plain GET serves the whole body.
        Mock::given(method("GET"))
            .and(path("/o/r/resolve/main/r.gguf"))
            .and(wiremock::matchers::header_exists("Range"))
            .respond_with(ResponseTemplate::new(416))
            .mount(&dl)
            .await;
        Mock::given(method("GET"))
            .and(path("/o/r/resolve/main/r.gguf"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(full.clone()))
            .mount(&dl)
            .await;

        let dest = dirs.models_dir().join("r.gguf");
        std::fs::write(dirs.models_dir().join("r.gguf.part"), [0xAA; 16]).unwrap();

        let client =
            HfClient::with_bases(&dl.uri(), &dl.uri(), None, vec![host_of(&dl.uri())]).unwrap();
        let url: reqwest::Url = format!("{}/o/r/resolve/main/r.gguf", dl.uri())
            .parse()
            .unwrap();
        let plan = FilePlan {
            filename: "r.gguf".into(),
            bytes: 16,
            sha256: Some(sha),
        };
        let n = client
            .download_to(url, &plan, &dest, &mut |_, _| {})
            .await
            .unwrap();
        assert_eq!(n, 16);
        assert_eq!(std::fs::read(&dest).unwrap(), full);
        assert!(
            !dirs.models_dir().join("r.gguf.part").exists(),
            "no partial left"
        );
    }

    #[tokio::test]
    async fn integration__redirect_to_non_allowlisted_host__blocked() {
        let api = MockServer::start().await;
        let dl = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/models/o/r"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "siblings": [ sibling_json("r.gguf", 4, &payload(b"abcd")) ]
            })))
            .mount(&api)
            .await;
        Mock::given(method("GET"))
            .and(path("/o/r/resolve/main/r.gguf"))
            .respond_with(
                ResponseTemplate::new(302)
                    .insert_header("Location", "http://evil.example.com/exfil"),
            )
            .mount(&dl)
            .await;

        let client = HfClient::with_bases(
            &api.uri(),
            &dl.uri(),
            None,
            vec![host_of(&api.uri()), host_of(&dl.uri())],
        )
        .unwrap();
        let plan = FilePlan {
            filename: "r.gguf".into(),
            bytes: 4,
            sha256: None,
        };
        let err = client
            .download_file("o/r", &plan, Path::new("/tmp/never-r.gguf"), |_, _| {})
            .await
            .unwrap_err();
        let chain = format!("{err:#}");
        assert!(chain.contains("evil.example.com"), "{chain}");
    }

    #[tokio::test]
    async fn integration__token_only_on_first_party_host() {
        // huggingface-mock redirects to cdn-mock; assert token arrives at
        // neither the CDN nor leaks via any second hop.
        let first = MockServer::start().await;
        let cdn = MockServer::start().await;
        let body = b"tok".to_vec();
        Mock::given(method("GET"))
            .and(path("/api/models/o/r"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "siblings": [ sibling_json("r.gguf", 3, &payload(&body)) ]
            })))
            .mount(&first)
            .await;
        // First-party download hop redirects to CDN.
        Mock::given(method("GET"))
            .and(path("/o/r/resolve/main/r.gguf"))
            .respond_with(
                ResponseTemplate::new(302)
                    .insert_header("Location", format!("{}/lfs/r.gguf", cdn.uri())),
            )
            .expect(1)
            .mount(&first)
            .await;
        Mock::given(method("GET"))
            .and(path("/lfs/r.gguf"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(body.clone()))
            .expect(1)
            .mount(&cdn)
            .await;

        let client = HfClient::with_bases(
            &first.uri(),
            &first.uri(),
            Some("SECRET-TOKEN".into()),
            vec![host_of(&first.uri()), host_of(&cdn.uri())],
        )
        .unwrap();
        let tmp = tempfile::tempdir().unwrap();
        let dest = tmp.path().join("r.gguf");
        let plan = FilePlan {
            filename: "r.gguf".into(),
            bytes: 3,
            sha256: Some(payload(&body.clone())),
        };
        client
            .download_file("o/r", &plan, &dest, |_, _| {})
            .await
            .unwrap();

        // Inspect what the CDN actually received.
        let requests = cdn.received_requests().await.unwrap();
        let cdn_req = requests
            .iter()
            .find(|r| r.url.path() == "/lfs/r.gguf")
            .expect("cdn hit");
        assert!(
            cdn_req.headers.get("authorization").is_none(),
            "token must NEVER reach a CDN host"
        );
    }

    #[tokio::test]
    async fn integration__gated_repo__clear_error() {
        let api = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/models/o/gated"))
            .respond_with(ResponseTemplate::new(401))
            .mount(&api)
            .await;
        let client =
            HfClient::with_bases(&api.uri(), &api.uri(), None, vec![host_of(&api.uri())]).unwrap();
        let err = client.model_info("o/gated").await.unwrap_err();
        assert!(
            err.to_string().contains("gated") && err.to_string().contains("HF_TOKEN"),
            "{err}"
        );
    }

    #[tokio::test]
    async fn integration__lock_contention__second_pull_errors() {
        let tmp = tempfile::tempdir().unwrap();
        let dirs = BlazarDirs {
            config_dir: tmp.path().join("cfg"),
            data_dir: tmp.path().join("data"),
        };
        dirs.ensure().unwrap();
        let held = PullLock::acquire(&dirs, "m").unwrap();
        let err = PullLock::acquire(&dirs, "m").unwrap_err();
        assert!(err.to_string().contains("already in progress"), "{err}");
        drop(held);
        assert!(
            PullLock::acquire(&dirs, "m").is_ok(),
            "lock released on drop"
        );
    }

    #[tokio::test]
    async fn integration__lock_stale__dead_owner_pid_stolen_immediately() {
        let tmp = tempfile::tempdir().unwrap();
        let dirs = BlazarDirs {
            config_dir: tmp.path().join("cfg"),
            data_dir: tmp.path().join("data"),
        };
        dirs.ensure().unwrap();
        // A real, now-exited process: its pid is genuinely dead, no race.
        let mut child =
            std::process::Command::new(std::env::var("EXE_TRUE").unwrap_or_else(|_| {
                if cfg!(windows) {
                    "cmd".into()
                } else {
                    "true".into()
                }
            }))
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .unwrap();
        let dead_pid = child.id();
        let _ = child.wait();
        let lock = dirs.run_dir().join("pull-m.lock");
        std::fs::write(&lock, format!("{dead_pid}\n")).unwrap();
        // Fresh file, well under 6h: must STILL be stolen because the
        // owner is provably dead.
        let got = PullLock::acquire(&dirs, "m");
        assert!(got.is_ok(), "dead-owner lock stolen: {}", got.unwrap_err());
    }

    #[tokio::test]
    async fn integration__lock_live__owner_pid_named_in_error() {
        let tmp = tempfile::tempdir().unwrap();
        let dirs = BlazarDirs {
            config_dir: tmp.path().join("cfg"),
            data_dir: tmp.path().join("data"),
        };
        dirs.ensure().unwrap();
        let lock = dirs.run_dir().join("pull-m.lock");
        std::fs::write(&lock, format!("{}\n", std::process::id())).unwrap();
        let err = PullLock::acquire(&dirs, "m").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("already in progress"), "{msg}");
        assert!(msg.contains(&std::process::id().to_string()), "{msg}");
        std::fs::remove_file(&lock).unwrap();
    }

    #[tokio::test]
    async fn integration__pull_shard_set__both_parts_and_count_recorded() {
        let tmp = tempfile::tempdir().unwrap();
        let dirs = BlazarDirs {
            config_dir: tmp.path().join("cfg"),
            data_dir: tmp.path().join("data"),
        };
        dirs.ensure().unwrap();
        let p1 = b"part-one--".to_vec();
        let p2 = b"part-two".to_vec();
        let api = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/models/o/big"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "siblings": [
                    sibling_json("big-q4_k_m-00001-of-00002.gguf", p1.len() as u64, &payload(&p1)),
                    sibling_json("big-q4_k_m-00002-of-00002.gguf", p2.len() as u64, &payload(&p2)),
                ]
            })))
            .mount(&api)
            .await;
        let dl = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/o/big/resolve/main/big-q4_k_m-00001-of-00002.gguf"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(p1))
            .mount(&dl)
            .await;
        Mock::given(method("GET"))
            .and(path("/o/big/resolve/main/big-q4_k_m-00002-of-00002.gguf"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(p2))
            .mount(&dl)
            .await;

        let client = HfClient::with_bases(
            &api.uri(),
            &dl.uri(),
            None,
            vec![host_of(&api.uri()), host_of(&dl.uri())],
        )
        .unwrap();
        let puller = Puller {
            dirs: dirs.clone(),
            client,
            bus: EventBus::default(),
            force: false,
        };
        let row = puller.pull("o/big:Q4_K_M").await.unwrap().row;
        assert_eq!(row.shards, 2);
        assert!(
            row.path.ends_with("-00001-of-00002.gguf"),
            "first shard is the launch path"
        );
        assert!(Path::new(&row.path).exists());
        let second = dirs.models_dir().join("big-q4_k_m-00002-of-00002.gguf");
        assert!(second.exists(), "second shard stored alongside");
    }

    #[tokio::test]
    async fn integration__pull_cancel__lock_released_partial_kept() {
        let tmp = tempfile::tempdir().unwrap();
        let dirs = BlazarDirs {
            config_dir: tmp.path().join("cfg"),
            data_dir: tmp.path().join("data"),
        };
        dirs.ensure().unwrap();
        let body = vec![7u8; 4096];
        let api = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/models/o/slow"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "siblings": [ sibling_json("slow-q4_k_m.gguf", body.len() as u64, &payload(&body)) ]
            })))
            .mount(&api)
            .await;
        let dl = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/o/slow/resolve/main/slow-q4_k_m.gguf"))
            // Slow enough for the cancel to land mid-download.
            .respond_with(
                ResponseTemplate::new(200)
                    .set_delay(std::time::Duration::from_secs(30))
                    .set_body_bytes(body.clone()),
            )
            .mount(&dl)
            .await;
        let client = HfClient::with_bases(
            &api.uri(),
            &dl.uri(),
            None,
            vec![host_of(&api.uri()), host_of(&dl.uri())],
        )
        .unwrap();
        let puller = Puller {
            dirs: dirs.clone(),
            client,
            bus: EventBus::default(),
            force: false,
        };

        // Cancellation semantics: the select! drop of the pull future is
        // exactly what an interrupt does in the CLI. Timeout fires first
        // -> future dropped mid-download.
        let fut = puller.pull("o/slow:Q4_K_M");
        let cancelled = tokio::time::timeout(std::time::Duration::from_millis(400), fut).await;
        assert!(cancelled.is_err(), "pull should still be mid-download");

        // `.part`-kept-on-cancel has no deterministic window with
        // wiremock (its delay is pre-response, before the file is even
        // created); persistence on error/cancel is pinned by the resume
        // tests. Here the contract is the LOCK: released on cancel and
        // immediately re-acquirable.
        let lock = dirs.run_dir().join("pull-slow-q4_k_m.lock");
        assert!(!lock.exists(), "lock must release on cancel, got {lock:?}");
        assert!(PullLock::acquire(&dirs, "slow-q4_k_m").is_ok());
    }

    fn sib_st(name: &str, size: u64, sha: &str) -> HfSibling {
        // Non-LFS small files carry no lfs object (config/tokenizer shape).
        let v = if sha.is_empty() || sha == "-" {
            serde_json::json!({"rfilename": name, "size": size})
        } else {
            serde_json::json!({"rfilename": name, "size": size,
                "lfs": {"sha256": sha, "size": size}})
        };
        serde_json::from_value(v).unwrap()
    }

    #[test]
    fn unit__select_safetensors_files__shards_aux_skip_nested_and_foreign() {
        let sibs = vec![
            sib_st("config.json", 10, "-"),
            sib_st("generation_config.json", 5, "-"),
            sib_st("tokenizer.json", 7, "-"),
            sib_st("model.safetensors.index.json", 9, "-"),
            sib_st("model-00002-of-00002.safetensors", 200, "bb"),
            sib_st("model-00001-of-00002.safetensors", 100, "aa"),
            sib_st("chat_template.jinja", 3, "-"),
            sib_st("original/model-00001-of-00002.safetensors", 999, "zz"), // nested
            sib_st("pytorch_model.bin", 500, "-"),                          // foreign
            sib_st("README.md", 2, "-"),                                    // doc
        ];
        let sel = select_safetensors_files(&sibs).unwrap();
        assert_eq!(sel.shard_count, 2);
        let names: Vec<&str> = sel.files.iter().map(|f| f.filename.as_str()).collect();
        // Aux sorted first, shards sorted after; nested/bin/README absent.
        assert_eq!(
            names,
            [
                "chat_template.jinja",
                "config.json",
                "generation_config.json",
                "model.safetensors.index.json",
                "tokenizer.json",
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
            ]
        );
    }

    #[test]
    fn unit__select_safetensors_files__missing_config_is_teaching_error() {
        let sibs = vec![sib_st("model.safetensors", 10, "a")];
        let err = select_safetensors_files(&sibs).unwrap_err();
        assert!(err.to_string().contains("config.json"), "{err}");
    }

    #[test]
    fn unit__select_safetensors_files__no_shards_is_error() {
        let sibs = vec![sib_st("config.json", 10, "-")];
        assert!(select_safetensors_files(&sibs).is_err());
    }

    #[test]
    fn unit__hf_quant_label__bits_dtype_default() {
        let m = |quant_bits: Option<u8>, dtype: Option<&str>| blazar_core::hfmeta::HfMeta {
            quant_bits,
            dtype: dtype.map(str::to_string),
            ..Default::default()
        };
        assert_eq!(hf_quant_label(&m(Some(4), None)), "4BIT");
        assert_eq!(hf_quant_label(&m(Some(8), Some("bfloat16"))), "8BIT");
        assert_eq!(hf_quant_label(&m(None, Some("bfloat16"))), "BF16");
        assert_eq!(hf_quant_label(&m(None, Some("float16"))), "FP16");
        assert_eq!(hf_quant_label(&m(None, Some("float32"))), "F32");
        assert_eq!(hf_quant_label(&m(None, Some("float8_e4m3fn"))), "FP8");
        assert_eq!(hf_quant_label(&m(None, Some("custom"))), "CUSTOM");
        assert_eq!(hf_quant_label(&m(None, None)), "SAFETENSORS");
    }

    #[test]
    fn unit__revision_digest__order_independent_content_sensitive() {
        let a = FilePlan {
            filename: "a".into(),
            bytes: 1,
            sha256: Some("x".into()),
        };
        let b = FilePlan {
            filename: "b".into(),
            bytes: 2,
            sha256: None,
        };
        assert_eq!(
            revision_digest(&[a.clone(), b.clone()]),
            revision_digest(&[b.clone(), a.clone()])
        );
        let c = FilePlan {
            filename: "a".into(),
            bytes: 3,
            sha256: Some("x".into()),
        };
        assert_ne!(revision_digest(&[a, b.clone()]), revision_digest(&[c, b]));
    }

    #[tokio::test]
    async fn integration__safetensors_pull__dir_row_then_idempotent() {
        let tmp = tempfile::tempdir().unwrap();
        let dirs = BlazarDirs {
            config_dir: tmp.path().join("cfg"),
            data_dir: tmp.path().join("data"),
        };
        dirs.ensure().unwrap();

        let config = br#"{"architectures":["Qwen2ForCausalLM"],"model_type":"qwen2",
            "max_position_embeddings":32768,"torch_dtype":"bfloat16","num_hidden_layers":28,
            "num_attention_heads":14,"num_key_value_heads":2,"head_dim":128}"#
            .to_vec();
        let index = br#"{"metadata":{"total_size":300},"weight_map":{
            "a":"model-00001-of-00002.safetensors","b":"model-00002-of-00002.safetensors"}}"#
            .to_vec();
        let shard1 = b"SHARD1-BYTES".to_vec();
        let shard2 = b"SHARD2-BYTES-LONGER".to_vec();
        let files: Vec<(&str, Vec<u8>, bool)> = vec![
            ("config.json", config.clone(), false),
            ("generation_config.json", b"{}".to_vec(), false),
            ("tokenizer.json", b"{}".to_vec(), false),
            ("model.safetensors.index.json", index.clone(), false),
            ("model-00001-of-00002.safetensors", shard1.clone(), true),
            ("model-00002-of-00002.safetensors", shard2.clone(), true),
        ];
        let siblings: Vec<serde_json::Value> = files
            .iter()
            .map(|(n, b, lfs)| {
                let sha = payload(b);
                sibling_json_cond(n, b.len() as u64, &sha, *lfs)
            })
            .chain([serde_json::json!({"rfilename": "original/model.bin", "size": 500})])
            .collect();

        let api = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/models/Qwen/Qwen2.5-0.5B-Instruct"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "siblings": siblings
            })))
            .mount(&api)
            .await;
        let dl = MockServer::start().await;
        for (name, body, _) in &files {
            Mock::given(method("GET"))
                .and(path(format!(
                    "/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/{name}"
                )))
                .respond_with(ResponseTemplate::new(200).set_body_bytes(body.clone()))
                .mount(&dl)
                .await;
        }

        let client = HfClient::with_bases(
            &api.uri(),
            &dl.uri(),
            None,
            vec![host_of(&api.uri()), host_of(&dl.uri())],
        )
        .unwrap();
        let puller = Puller {
            dirs: dirs.clone(),
            client,
            bus: EventBus::default(),
            force: false,
        };
        let outcome = puller.pull("Qwen/Qwen2.5-0.5B-Instruct").await.unwrap();
        let dir = dirs.models_dir().join("qwen2.5-0.5b-instruct.d");
        assert_eq!(outcome.row.path, dir.display().to_string());
        assert_eq!(outcome.row.shards, 2);
        assert_eq!(outcome.row.quant, "BF16");
        assert_eq!(outcome.row.arch.as_deref(), Some("Qwen2ForCausalLM"));
        assert_eq!(outcome.row.ctx_train, Some(32768));
        let bytes = files.iter().map(|(_, b, _)| b.len() as u64).sum::<u64>();
        assert_eq!(u64::try_from(outcome.row.bytes).unwrap(), bytes);
        for (name, body, _) in &files {
            assert_eq!(
                std::fs::read(dir.join(name)).unwrap(),
                *body,
                "{name} must be on disk"
            );
        }
        assert!(!dir.join("original").exists(), "nested files never pulled");

        // Second pull of the same revision: no-op, zero download bytes.
        let before = dl.received_requests().await.unwrap_or_default().len();
        let again = puller.pull("Qwen/Qwen2.5-0.5B-Instruct").await.unwrap();
        assert!(again.already_present, "same revision must short-circuit");
        let after = dl.received_requests().await.unwrap_or_default().len();
        assert_eq!(before, after, "no download requests on re-pull");
    }

    #[tokio::test]
    async fn integration__safetensors_pull__index_names_missing_shard_fails() {
        let tmp = tempfile::tempdir().unwrap();
        let dirs = BlazarDirs {
            config_dir: tmp.path().join("cfg"),
            data_dir: tmp.path().join("data"),
        };
        dirs.ensure().unwrap();

        let config = br#"{"architectures":["Qwen2ForCausalLM"]}"#.to_vec();
        // Index names a shard the repo listing never had: the download
        // completes, the coverage gate must refuse (H1).
        let index =
            br#"{"weight_map":{"a":"model-00001-of-00002.safetensors","b":"model-00002-of-00002.safetensors"}}"#
                .to_vec();
        let files: Vec<(&str, Vec<u8>, bool)> = vec![
            ("config.json", config, false),
            ("model.safetensors.index.json", index, false),
            ("model-00001-of-00002.safetensors", b"ONE".to_vec(), true),
        ];
        let siblings: Vec<serde_json::Value> = files
            .iter()
            .map(|(n, b, lfs)| sibling_json_cond(n, b.len() as u64, &payload(b), *lfs))
            .collect();

        let api = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/models/o/m"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "siblings": siblings
            })))
            .mount(&api)
            .await;
        let dl = MockServer::start().await;
        for (name, body, _) in &files {
            Mock::given(method("GET"))
                .and(path(format!("/o/m/resolve/main/{name}")))
                .respond_with(ResponseTemplate::new(200).set_body_bytes(body.clone()))
                .mount(&dl)
                .await;
        }
        let client = HfClient::with_bases(
            &api.uri(),
            &dl.uri(),
            None,
            vec![host_of(&api.uri()), host_of(&dl.uri())],
        )
        .unwrap();
        let puller = Puller {
            dirs,
            client,
            bus: EventBus::default(),
            force: false,
        };
        let err = puller.pull("o/m").await.unwrap_err();
        assert!(
            err.to_string().contains("missing from the download"),
            "{err}"
        );
    }

    fn sibling_json_cond(name: &str, size: u64, sha: &str, lfs: bool) -> serde_json::Value {
        serde_json::json!({
            "rfilename": name,
            "size": size,
            "lfs": lfs.then(|| serde_json::json!({"sha256": sha, "size": size}))
        })
    }

    fn host_of(uri: &str) -> String {
        uri.rsplit_once("://")
            .map_or(uri, |(_, h)| h)
            .rsplit_once(':')
            .map_or(uri, |(h, _)| h)
            .to_string()
    }
}
