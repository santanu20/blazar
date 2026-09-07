//! `pallama quantize`: orchestrate the engine's own `llama-quantize`
//! (ships in the same release tarball as llama-server) to derive a new
//! quantization of a pulled model — no separate toolchain, no uploads.
//! The output lands in the plain-GGUF store like any pulled model.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use pallama_core::{PallamaDirs, Store};

/// Locate `llama-imatrix` the same way (same release tarball family).
pub fn find_imatrix_bin(dirs: &PallamaDirs) -> Result<PathBuf> {
    let store = Store::open(dirs)?;
    let engines = store.list_engines()?;
    engines
        .iter()
        .filter(|e| e.active)
        .chain(engines.iter().filter(|e| !e.active))
        .map(|e| {
            dirs.engines_dir()
                .join(&e.tag)
                .join(format!("llama-{}", e.tag))
                .join("llama-imatrix")
        })
        .find(|p| p.exists())
        .ok_or_else(|| {
            anyhow!("no llama-imatrix found in any installed engine; run `pallama engine update`")
        })
}

/// Run llama-imatrix over a calibration file, producing `out.imatrix`
/// (upstream GGUF-embedded importance matrix). `on_line` gets progress.
pub fn imatrix<P, F>(
    bin: P,
    model: &Path,
    calibration_file: &Path,
    out: &Path,
    chunk_count: u32,
    mut on_line: F,
) -> Result<PathBuf>
where
    P: AsRef<Path>,
    F: FnMut(&str),
{
    // Verified upstream interface (tools/imatrix print_usage): -m -f -o
    // --output-format gguf. chunk_count stays advisory (0 = default).
    let _ = chunk_count;
    let result = std::process::Command::new(bin.as_ref())
        .arg("-m")
        .arg(model)
        .arg("-f")
        .arg(calibration_file)
        .arg("-o")
        .arg(out)
        .arg("--output-format")
        .arg("gguf")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output()
        .with_context(|| format!("run {}", bin.as_ref().display()))?;
    for line in String::from_utf8_lossy(&result.stdout).lines() {
        on_line(line);
    }
    for line in String::from_utf8_lossy(&result.stderr).lines() {
        on_line(line);
    }
    if !result.status.success() {
        return Err(anyhow!(
            "llama-imatrix failed ({}): {}",
            result.status,
            String::from_utf8_lossy(&result.stderr).trim()
        ));
    }
    if !out.exists() {
        return Err(anyhow!(
            "llama-imatrix reported success but wrote no matrix"
        ));
    }
    Ok(out.to_path_buf())
}

/// Locate `llama-quantize` in an installed engine directory (active
/// engine first — same discovery order as `find_bench_bin`).
pub fn find_quantize_bin(dirs: &PallamaDirs) -> Result<PathBuf> {
    let store = Store::open(dirs)?;
    let engines = store.list_engines()?;
    engines
        .iter()
        .filter(|e| e.active)
        .chain(engines.iter().filter(|e| !e.active))
        .map(|e| {
            dirs.engines_dir()
                .join(&e.tag)
                .join(format!("llama-{}", e.tag))
                .join("llama-quantize")
        })
        .find(|p| p.exists())
        .ok_or_else(|| {
            anyhow!("no llama-quantize found in any installed engine; run `pallama engine update`")
        })
}

/// Sanity-shape for a quant type (llama-quantize owns the real
/// validation — this only catches obvious shell-hostile junk early).
#[must_use]
pub fn plausible_quant_type(t: &str) -> bool {
    !t.is_empty()
        && t.len() <= 16
        && t.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        && t.chars()
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic() || c.is_ascii_digit())
}

/// Run llama-quantize, streaming its progress lines to `on_line`.
/// Returns the output path (caller registers it in the store).
pub fn quantize<P, F>(
    bin: P,
    src: &Path,
    dst: &Path,
    quant_type: &str,
    on_line: F,
) -> Result<PathBuf>
where
    P: AsRef<Path>,
    F: FnMut(&str),
{
    quantize_im(bin, src, dst, quant_type, &[], on_line)
}

/// quantize with extra leading args (e.g. `--imatrix file`).
pub fn quantize_im<P, F>(
    bin: P,
    src: &Path,
    dst: &Path,
    quant_type: &str,
    extra_args: &[String],
    mut on_line: F,
) -> Result<PathBuf>
where
    P: AsRef<Path>,
    F: FnMut(&str),
{
    let out = std::process::Command::new(bin.as_ref())
        .args(extra_args)
        .arg(src)
        .arg(dst)
        .arg(quant_type)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output()
        .with_context(|| format!("run {}", bin.as_ref().display()))?;
    // llama-quantize interleaves progress on both streams; present them
    // in order after completion (its progress ticks are not line-flushed
    // reliably enough to justify streaming complexity).
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        on_line(line);
    }
    for line in String::from_utf8_lossy(&out.stderr).lines() {
        on_line(line);
    }
    if !out.status.success() {
        return Err(anyhow!(
            "llama-quantize failed ({}): {}",
            out.status,
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    if !dst.exists() {
        return Err(anyhow!(
            "llama-quantize reported success but wrote no output"
        ));
    }
    Ok(dst.to_path_buf())
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    #[test]
    fn unit__plausible_quant_type__shape_rules() {
        assert!(plausible_quant_type("Q4_K_M"));
        assert!(plausible_quant_type("IQ4_XS"));
        assert!(plausible_quant_type("f32"));
        assert!(!plausible_quant_type(""));
        assert!(!plausible_quant_type("Q4;rm -rf"));
        assert!(!plausible_quant_type("with space"));
        assert!(!plausible_quant_type("--flag"));
    }
}
