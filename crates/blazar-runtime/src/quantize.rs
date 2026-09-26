//! `blazar quantize`: orchestrate the engine's own `llama-quantize`
//! (ships in the same release tarball as llama-server) to derive a new
//! quantization of a pulled model — no separate toolchain, no uploads.
//! The output lands in the plain-GGUF store like any pulled model.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use blazar_core::{BlazarDirs, Store};

/// Locate an engine tool binary (`llama-quantize`, `llama-imatrix`,
/// `llama-perplexity`) in an installed engine directory. Same discovery
/// contract as `find_bench_bin`: llamacpp engines only, active engine
/// first, and the engine module's symlink-safe walk (F86) finds the
/// binary under any unpack layout — release tarballs nest tools in
/// versioned dirs (`llama-b11193/`) and CUDA overlay assets in
/// vendor-suffixed dirs (`llama-b11193-cuda-bin-ubuntu-12-x64/`), so
/// deriving the inner dir from the tag misses both.
fn find_engine_tool(dirs: &BlazarDirs, tool_base: &str) -> Result<PathBuf> {
    let store = Store::open(dirs)?;
    let engines = store.list_engines()?;
    let tool_name = crate::tool_file_name(tool_base);
    let ordered: Vec<_> = engines
        .iter()
        .filter(|e| e.kind == blazar_core::engine_kind::EngineKind::LlamaCpp)
        .collect();
    let ordered: Vec<_> = ordered
        .iter()
        .filter(|e| e.active)
        .chain(ordered.iter().filter(|e| !e.active))
        .collect();
    for e in ordered {
        let engine_dir = dirs.engines_dir().join(&e.tag);
        if let Ok(tool) = crate::engine::find_engine_binary(&engine_dir, &[tool_name.as_str()]) {
            return Ok(tool);
        }
    }
    Err(anyhow!(
        "no {tool_base} found in any installed engine; run `blazar engine update`"
    ))
}

/// Locate `llama-imatrix` the same way (same release tarball family).
pub fn find_imatrix_bin(dirs: &BlazarDirs) -> Result<PathBuf> {
    find_engine_tool(dirs, "llama-imatrix")
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
    // --output-format gguf. chunk_count 0 = tool default.
    let mut cmd = std::process::Command::new(bin.as_ref());
    cmd.arg("-m")
        .arg(model)
        .arg("-f")
        .arg(calibration_file)
        .arg("-o")
        .arg(out)
        .arg("--output-format")
        .arg("gguf");
    // F99: pass the calibration chunk budget through when the caller set
    // one (0 keeps the tool's own default).
    if chunk_count > 0 {
        cmd.arg("--chunk-count").arg(chunk_count.to_string());
    }
    let result = cmd
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
pub fn find_quantize_bin(dirs: &BlazarDirs) -> Result<PathBuf> {
    find_engine_tool(dirs, "llama-quantize")
}

/// Locate `llama-perplexity` (same release tarball as llama-quantize).
pub fn find_perplexity_bin(dirs: &BlazarDirs) -> Result<PathBuf> {
    find_engine_tool(dirs, "llama-perplexity")
}

/// Fixed calibration corpus for the `--verify` gate: ~8.5KB of DIVERSE
/// one-pass natural text (no repeated sentences, no RNG) so perplexity
/// sits in a realistic range and quantization damage is actually
/// measurable — a cycled corpus drives PPL toward 1.0 and hides
/// degradation (live-proven on qwen2.5-0.5b). ≥1024 tokens satisfies
/// llama-perplexity's minimum (2× its default 512 ctx) on mainstream
/// tokenizers. Deterministic, so gate numbers reproduce run-to-run; the
/// gate measures the RELATIVE base→output delta on this same corpus.
pub const VERIFY_PROBE_CORPUS: &str = "\
The transformer architecture processes every token in a single pass. \
Quantization maps floating-point weights onto a coarse integer lattice. \
A sliding window restricts attention to the most recent positions. \
The scheduler interleaves long prefill work with latency-sensitive decode. \
Radix trees let concurrent requests share their common prefix cache. \
Water boils at one hundred degrees Celsius at standard pressure. \
The compiler emitted vector instructions for the innermost loop. \
Gradient descent follows the local slope of the loss surface. \
Basalt forms when mafic lava cools rapidly at the surface. \
The magistrate postponed the hearing until the following Tuesday. \
Kilimanjaro rises almost five thousand metres above the coastal plain. \
She folded the dough twice and let it rest for twenty minutes. \
Entropy always increases in a closed thermodynamic system. \
The train departed platform nine at a quarter past seven. \
Rust borrows are checked at compile time without runtime cost. \
Marie Curie won Nobel Prizes in two different scientific fields. \
The reef fish changed color when a predator swam overhead. \
Cache coherence protocols keep multiple cores from diverging. \
Your appointment is on the fourteenth at half past three. \
The cartographer redrew the border after the treaty was signed. \
Photosynthesis converts roughly three percent of incident sunlight. \
He asked whether the warranty covered accidental water damage. \
The symphony premiered in Vienna to a mostly empty hall. \
Memory bandwidth, not arithmetic, limits this workload. \
Aphids farm ants for honeydew in a curious reversal of roles. \
The courtroom fell silent as the witness identified the defendant. \
Fluvial deposits grade from boulders downstream into fine clay. \
Nearly forty percent of the batch failed the tensile test. \
The bakery opens at six and sells out of croissants by nine. \
Speculative decoding drafts cheap tokens and verifies them in bulk. \
The lighthouse keeper logged four ships during the night shift. \
Volcanic ash enriched the soil and the vineyards thrived. \
Please confirm the invoice total before submitting the purchase order. \
The quantum state collapses upon measurement. \
Migratory birds navigate using a combination of stars and magnetite. \
The apprentice spent months learning to tune the engine by ear. \
Diagenesis alters porosity long after sediment is buried. \
Their apartment lease expires at the end of March. \
Paged attention stores the key-value cache in fixed-size blocks. \
The detective noticed the inconsistency in the alibi immediately. \
Coral bleaching follows sustained anomalies in sea temperature. \
Grandmother threaded the needle on her third attempt. \
The kernel copies data between user and kernel address spaces. \
Whales communicate across ocean basins at infrasonic frequencies. \
The negotiation stalled over liability for prior defects. \
Erosion undercut the cliff until the farmhouse leaned seaward. \
He measured the clearance with a feeler gauge. \
Sparse attention drops token pairs with negligible similarity. \
The choir rehearsed the fugue until the entrances locked. \
Salt domes deform overlying strata into characteristic traps. \
Would you forward the minutes to everyone who abstained? \
The snapshot of the database weighed four hundred gigabytes. \
Fireflies synchronize their flashes in some Southeast Asian species. \
The tailor basted the seams before the final fitting. \
Numerical weather prediction divides the atmosphere into grid cells. \
The auditor questioned the revenue recognition schedule. \
Metamorphic foliation records the direction of ancient stress. \
The dog buried the bone beneath the hydrangea. \
Instruction-level parallelism extracts work from a single stream. \
She translated the poem twice and preferred the second version. \
Storm surges pile water against the coast ahead of landfall. \
The registrar requires transcripts from every institution attended. \
Turbidite beds thicken and coarsen upward in the core. \
The mechanic heard the bearing whine before the driver did. \
Approximate nearest-neighbor search trades recall for speed. \
The council voted seven to two against the rezoning proposal. \
Lichen grows where nothing else can cling. \
The recipe calls for a pinch of saffron and twice that of salt. \
Baroclinic instability converts horizontal to vertical shear. \
His handwriting deteriorated whenever the lecture sped up. \
Ground-penetrating radar resolves interfaces by dielectric contrast. \
The invoice references a purchase order dated last October. \
Mixture-of-experts routes each token through a sparse subset. \
The understudy performed the lead role with two days notice. \
Karst terrain drains through sinkholes rather than surface rivers. \
The jury deliberated for eleven hours over the fingerprint evidence. \
Retrieval-augmented generation cites sources it never memorized. \
The kiln reached cone ten before the glaze finally ran. \
Isotopic ratios in ice cores archive ancient atmospheres. \
Nobody expected the backup generator to fail during the test. \
The scheduler starves low-priority queues under sustained load. \
Albatrosses sleep on the wing during ocean crossings. \
The lease forbids subletting without written consent. \
Seismic reflections brighten where gas replaces brine in pores. \
She chalked the route while the others racked the gear. \
Token-level embeddings pooled over chunk spans beat naive splitting. \
The pamphlet warned of delays between the eighth and the tenth. \
Laterite forms under intense rainfall that leaches everything mobile. \
The intern archived the correspondence by decade and correspondent. \
Bandwidth costs dominate the bill at ninety-gigabyte egress. \
Moths evolved jamming to spoof bat echolocation. \
The ferry leaves twice daily except on public holidays. \
Deltaic lobes avulse when sediment raises the bed above flood stage. \
He tuned the violin by fifths and checked the harmonics. \
Prefix caching saves the recomputation of identical system prompts. \
The committee's report ran to two hundred unindexed pages. \
Peridotite from the mantle rides aboard thrust sheets. \
The barista discarded the first extraction and timed the second. \
Query planners reorder joins to minimize intermediate rows. \
The expedition cached supplies at the fifth camp. \
Evaporite sequences record the desiccation of entire basins. \
The tailor said the shoulders could not be let out further. \
Attention sinks collect disproportionate activation mass. \
The stationmaster announced a platform change at the last moment. \
Gypsum roses grow in sabkha flats where brines evaporate. \
Two of the sensors drifted and the calibration had to be redone. \
Loosely coupled services fail independently and loudly. \
The novel opens with a funeral and closes with a birth. \
Paleosols mark unconformities where exposure interrupted deposition. \
The accountant reconciled the ledger to the penny. \
She switched the experiment to a blind protocol mid-trial. \
The dam operators released pulses to mimic the spring freshet. \
Granite weathers to arenas bounded by joints. \
Our neighbor replanted the hedge with native species. \
Backpropagation chains derivatives through the computation graph. \
The ambassador declined to comment on the leaked cable. \
Turbulence mixes the surface ocean on scales of hours. \
The printer jammed on the third sheet of the appendix. \
Subduction recycles oceanic crust faster than it can age. \
The auctioneer paused when the bidding stalled at four hundred. \
Antialiasing smooths the staircase along curved edges. \
Frost heave tilts fenceposts on north-facing slopes. \
Her thesis tied reef accretion to orbit-paced sea level. \
The elevator inspection sticker expired during the renovation. \
Momentum carries the payload coasting between burns. \
Grief, the essay argues, rearranges rather than resolves. \
Cheetah populations crashed through a genetic bottleneck. \
The clerk stamped the manifest and waved the convoy through. \
Pressure solution dissolves grains at their contacts. \
He restrung the racquet and immediately won the next set. \
The archive digitized ledger books that nobody had opened since the war. \
River deltas drown when subsidence outruns sediment supply. \
A well-placed breakpoint teaches more than a page of logs. \
";

/// Extract the final perplexity from llama-perplexity output. Upstream
/// prints `Final estimate: PPL = 1.4730 +/- 0.06434` on stderr after the
/// chunk series; the LAST `PPL = <f64>` occurrence wins (progress lines
/// never carry the marker).
#[must_use]
pub fn parse_final_ppl(output: &str) -> Option<f64> {
    let idx = output.rfind("PPL = ")?;
    output[idx + "PPL = ".len()..]
        .split([',', ' ', '\n', '\r', '\t'])
        .find_map(|tok| tok.parse::<f64>().ok())
}

/// Run llama-perplexity over `model` with the fixed probe corpus,
/// returning the final PPL. `on_line` receives the tool's output lines
/// (reported after completion, matching the quantize lane's behavior).
pub fn perplexity<P, F>(bin: P, model: &Path, probe: &Path, mut on_line: F) -> Result<f64>
where
    P: AsRef<Path>,
    F: FnMut(&str),
{
    let out = std::process::Command::new(bin.as_ref())
        .arg("-m")
        .arg(model)
        .arg("-f")
        .arg(probe)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output()
        .with_context(|| format!("run {}", bin.as_ref().display()))?;
    let mut combined = String::new();
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        on_line(line);
        combined.push_str(line);
        combined.push('\n');
    }
    for line in String::from_utf8_lossy(&out.stderr).lines() {
        on_line(line);
        combined.push_str(line);
        combined.push('\n');
    }
    if !out.status.success() {
        return Err(anyhow!(
            "llama-perplexity failed ({}): {}",
            out.status,
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    parse_final_ppl(&combined)
        .ok_or_else(|| anyhow!("llama-perplexity produced no final PPL estimate"))
}

/// Pure gate: relative PPL degradation of a quantized output vs its
/// base. Returns the signed relative delta; errors (with the numbers)
/// when degradation exceeds `max_relative` or the base PPL is unusable.
pub fn verify_gate(base_ppl: f64, out_ppl: f64, max_relative: f64) -> Result<f64> {
    if !(base_ppl.is_finite() && base_ppl > 0.0) {
        return Err(anyhow!("unusable base perplexity {base_ppl}"));
    }
    let delta = (out_ppl - base_ppl) / base_ppl;
    if delta > max_relative {
        return Err(anyhow!(
            "perplexity gate FAILED: PPL {base_ppl:.4} -> {out_ppl:.4} \
             (+{:.1}% degradation, gate {:.1}%)",
            delta * 100.0,
            max_relative * 100.0
        ));
    }
    Ok(delta)
}

/// Write the fixed probe corpus and return its path. Lives under the
/// data dir so repeated runs reuse one deterministic file.
pub fn write_verify_probe(dirs: &BlazarDirs) -> Result<PathBuf> {
    std::fs::create_dir_all(&dirs.data_dir)
        .with_context(|| format!("create {}", dirs.data_dir.display()))?;
    let path = dirs.data_dir.join("blazar-verify-probe.txt");
    std::fs::write(&path, VERIFY_PROBE_CORPUS)
        .with_context(|| format!("write verify probe {}", path.display()))?;
    Ok(path)
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

    #[test]
    fn unit__parse_final_ppl__reads_last_estimate() {
        // Shape captured live from llama-perplexity b10809 (stderr).
        let stderr = "0.25.035.418 I perplexity: 24.21 seconds per pass - ETA 0.25.122.305 I Final estimate: PPL = 1.4730 +/- 0.06434";
        assert_eq!(parse_final_ppl(stderr), Some(1.4730));
        // Last occurrence wins (an interrupted earlier estimate must not).
        let two = "PPL = 9.5000 something\nFinal estimate: PPL = 2.2500 +/- 0.1";
        assert_eq!(parse_final_ppl(two), Some(2.25));
        // Chunk series alone carries no marker.
        assert_eq!(
            parse_final_ppl("[1]1.5174,[2]1.5598,[3]1.4730,\n0.30 minutes"),
            None
        );
        assert_eq!(parse_final_ppl(""), None);
    }

    #[test]
    fn unit__verify_gate__pass_improved_and_fail() {
        // 2% degradation passes the default-shaped gate.
        let d = verify_gate(10.0, 10.2, 0.10).unwrap();
        assert!((d - 0.02).abs() < 1e-12);
        // Improvement (negative delta) always passes.
        assert!(verify_gate(10.0, 9.5, 0.0).unwrap() < 0.0);
        // 15% degradation against a 10% gate fails, numbers in the error.
        let err = verify_gate(10.0, 11.5, 0.10).unwrap_err().to_string();
        assert!(
            err.contains("FAILED") && err.contains("10.0000") && err.contains("11.5000"),
            "{err}"
        );
        // Unusable base.
        assert!(verify_gate(0.0, 1.0, 0.1).is_err());
        assert!(verify_gate(f64::NAN, 1.0, 0.1).is_err());
    }

    #[test]
    fn unit__write_verify_probe__deterministic_and_large_enough() {
        let tmp = tempfile::tempdir().unwrap();
        let dirs = BlazarDirs {
            config_dir: tmp.path().join("c"),
            data_dir: tmp.path().join("d"),
        };
        let p = write_verify_probe(&dirs).unwrap();
        let a = std::fs::read_to_string(&p).unwrap();
        // ≥ 8 KB (~1024+ tokens on mainstream tokenizers; live-measured
        // 11.5 KB probe → 3 chunks on qwen2.5).
        assert!(a.len() > 8_000, "probe too small: {}", a.len());
        let b = std::fs::read_to_string(write_verify_probe(&dirs).unwrap()).unwrap();
        assert_eq!(a, b, "probe must be deterministic");
    }

    /// Regression pin (live-observed 2026-09-26): CUDA overlay tags carry
    /// a `-cuda` suffix while the tarball nests tools in a versioned
    /// `llama-bNNNN/` dir — deriving the inner dir from the tag
    /// (`engines/b11193-cuda/llama-b11193-cuda/...`) missed the binary
    /// and quantize refused with "no llama-quantize found" on a store
    /// that had it. Discovery must walk the actual layout (F86), and a
    /// whisper voice-lane row must never satisfy an llamacpp tool probe.
    #[test]
    fn unit__find_quantize_bin__nested_and_suffixed_layouts_walked() {
        use blazar_core::engine_kind::EngineKind;
        use blazar_core::store::EngineRow;

        let tmp = tempfile::tempdir().unwrap();
        let dirs = BlazarDirs {
            config_dir: tmp.path().join("c"),
            data_dir: tmp.path().join("d"),
        };
        // Whisper voice lane: first in the table, carries a same-family
        // tool name that must NOT be picked for llamacpp quantize.
        let whisper_dir = dirs
            .data_dir
            .join("engines")
            .join("b5130")
            .join("whisper-bin-ubuntu-x64");
        std::fs::create_dir_all(&whisper_dir).unwrap();
        std::fs::write(
            whisper_dir.join(crate::tool_file_name("whisper-quantize")),
            b"#!/bin/sh\n",
        )
        .unwrap();
        // CUDA overlay shape: tag suffix does not match the inner dir.
        let nested = dirs
            .data_dir
            .join("engines")
            .join("b11193-cuda")
            .join("llama-b11193");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(
            nested.join(crate::tool_file_name("llama-quantize")),
            b"#!/bin/sh\n",
        )
        .unwrap();

        let store = Store::open(&dirs).unwrap();
        store
            .upsert_engine(&EngineRow {
                tag: "b5130".into(),
                asset: String::new(),
                sha256: String::new(),
                installed_at: 30,
                active: false,
                manifest: "{}".into(),
                kind: EngineKind::Whisper,
            })
            .unwrap();
        store
            .upsert_engine(&EngineRow {
                tag: "b11193-cuda".into(),
                asset: String::new(),
                sha256: String::new(),
                installed_at: 10,
                active: true,
                manifest: "{}".into(),
                kind: EngineKind::LlamaCpp,
            })
            .unwrap();

        let bin = find_quantize_bin(&dirs).unwrap();
        assert_eq!(
            bin,
            nested.join(crate::tool_file_name("llama-quantize")),
            "walk must find the tool under the versioned inner dir"
        );
        // Same contract for the sibling tools of the tarball family.
        std::fs::write(
            nested.join(crate::tool_file_name("llama-imatrix")),
            b"#!/bin/sh\n",
        )
        .unwrap();
        assert!(find_imatrix_bin(&dirs).is_ok());
    }
}
