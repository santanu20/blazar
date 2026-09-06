//! Bench runner + measured autotune.
//!
//! `llama-bench` runs the cartesian product of comma-list args itself
//! (verified: -p/-n/-c/-t/-ctk/-ctv accept lists, `-o json` emits result
//! objects), so `tune --search` is ONE invocation. The parser is
//! header/field-name driven (json objects), order-independent.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use pallama_core::config::Config;
use pallama_core::gguf::GgufMeta;
use pallama_core::hardware::Hardware;
use pallama_core::profile::{self, Endpoint, Profile, ProfileInput, TuningOverrides};
use pallama_core::store::{ProfileRow, Store};
use pallama_core::{ModelOverride, PallamaDirs};

/// One `llama-bench` result row. Accepts both dialects:
/// - real `-o json` (verified b10816): `avg_ts`, `n_prompt`/`n_gen`
/// - stub / older: `t/s`, `test`
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct BenchRow {
    #[serde(rename = "t/s", alias = "avg_ts", default)]
    pub ts: f64,
    #[serde(default)]
    pub test: String,
    #[serde(default)]
    pub n_ctx: Option<u64>,
    #[serde(default)]
    pub n_threads: Option<u64>,
    #[serde(default, rename = "type_k")]
    pub type_k: Option<String>,
    #[serde(default, rename = "type_v")]
    pub type_v: Option<String>,
    #[serde(default)]
    pub n_prompt: Option<u64>,
    #[serde(default)]
    pub n_gen: Option<u64>,
    /// llama-bench encodes fa as -1 auto / 0 off / 1 on.
    #[serde(default)]
    pub flash_attn: Option<i64>,
    #[serde(default)]
    pub n_batch: Option<u64>,
}

impl BenchRow {
    /// Test name: explicit, or derived (`n_gen == 0` -> `pp`, else `tg`).
    #[must_use]
    pub fn test_name(&self) -> String {
        if !self.test.is_empty() {
            return self.test.clone();
        }
        match (self.n_prompt, self.n_gen) {
            (Some(p), Some(0)) => format!("pp{p}"),
            (Some(0), Some(g)) => format!("tg{g}"),
            (Some(p), Some(_)) => format!("pp{p} @ tg"),
            _ => String::new(),
        }
    }
}

/// Everything `tune` needs to compile+benchmark+persist a profile.
pub struct Tuner<'a> {
    pub dirs: &'a PallamaDirs,
    /// Path to llama-bench binary (usually next to llama-server in the
    /// engine dir).
    pub bench_bin: PathBuf,
}

/// Locate a llama-bench in an installed engine directory (upstream
/// release tarballs ship llama-bench alongside llama-server; a manually
/// registered `local` engine may not). No external fallback paths: pallama
/// is self-contained — if nothing ships a bench binary, the error says so.
pub fn find_bench_bin(dirs: &PallamaDirs) -> Result<PathBuf> {
    let store = Store::open(dirs)?;
    let engines = store.list_engines()?;
    engines
        .iter()
        .filter(|e| e.active)
        .chain(engines.iter().filter(|e| !e.active))
        .map(|e| dirs.engines_dir().join(&e.tag).join(format!("llama-{}", e.tag)).join("llama-bench"))
        .find(|p| p.exists())
        .ok_or_else(|| anyhow!("no llama-bench found in any installed engine; run `pallama engine update`"))
}

/// Default bench grid: prompt processing + generation, 2 reps.
#[must_use] 
pub fn default_bench_args() -> Vec<String> {
    vec![
        "-p".into(),
        "512,2048".into(),
        "-n".into(),
        "128".into(),
        "-r".into(),
        "2".into(),
    ]
}

impl Tuner<'_> {
    /// Run llama-bench once with the given grid args; returns parsed rows.
    pub fn run(&self, model: &Path, grid_args: &[String]) -> Result<Vec<BenchRow>> {
        let out = std::process::Command::new(&self.bench_bin)
            .arg("-o")
            .arg("json")
            .arg("-m")
            .arg(model)
            .args(grid_args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .with_context(|| format!("spawn {}", self.bench_bin.display()))?;
        if !out.status.success() {
            return Err(anyhow!(
                "llama-bench exited {}: {}",
                out.status,
                String::from_utf8_lossy(&out.stderr)
            ));
        }
        let stdout = String::from_utf8_lossy(&out.stdout);
        parse_bench_json(&stdout)
    }

    /// Plain `pallama bench <model>`: one deterministic profile, no grid.
    pub fn bench_default(&self, model: &Path) -> Result<Vec<BenchRow>> {
        self.run(model, &default_bench_args())
    }

    /// Compile + persist a profile with explicit overrides (tune --ctx).
    pub fn adopt(
        &self,
        store: &Store,
        input: &ProfileInput<'_>,
        overrides: &TuningOverrides,
    ) -> Result<pallama_core::Profile> {
        let profile = profile::compile(input, overrides)
            .map_err(|e| anyhow::anyhow!("profile: {e}"))?;
        persist_profile(store, input, overrides, &profile, &[], 0.0)?;
        Ok(profile)
    }

    /// Grid axis names (`ctx`/`nthreads`/`threads`) mirror each other on
    /// purpose: one is the loop variable, the other the winning knob.
    #[allow(clippy::similar_names)]
    /// `pallama tune <model> --search`: grid {ctx, kv-quant, threads} and
    /// adopt the argmax by mean generation (tg) throughput. Returns the
    /// winning profile plus all rows for display.
    pub fn tune_search(
        &self,
        store: &Store,
        input: &ProfileInput<'_>,
    ) -> Result<(Profile, TuningOverrides, Vec<BenchRow>)> {
        let base = profile::compile(input, &TuningOverrides::default())
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let _ = base;
        let threads = input.hardware.physical_cores.max(1);
        // Grid axes llama-bench actually supports (verified b10816):
        // threads x KV-quant x flash-attn x batch. (No -c axis: ctx is a
        // server-launch knob.)
        let mut grid = vec![
            "-t".into(),
            format!("{},{}", threads, threads.saturating_sub(2).max(1)),
            "-ctk".into(),
            "f16,q8_0".into(),
            "-ctv".into(),
            "f16,q8_0".into(),
            "-fa".into(),
            "on,off".into(),
            "-b".into(),
            "2048,1024".into(),
        ];
        grid.extend(default_bench_args());
        // Some engine/device combos reject specific configurations (e.g.
        // q8_0 KV-cache on some Vulkan drivers fails context creation and
        // llama-bench aborts the WHOLE run). Degrade the grid instead of
        // dying: drop KV-quant, then fa, then batch axes until it runs.
        let mut grid_rows = None;
        for reduction in 0..=3 {
            let mut g = grid.clone();
            match reduction {
                1 => {
                    // KV-quant axis off
                    let idx: Vec<usize> = g
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| i % 2 == 0 && g.get(*i + 1).is_some_and(|v| v == "f16,q8_0"))
                        .map(|(i, _)| i)
                        .collect();
                    for i in idx.iter().rev() {
                        g[*i + 1] = "f16".into();
                    }
                    tracing::warn!("tune: retrying grid without KV-quant axis");
                }
                2 => {
                    let pos = g.iter().position(|v| v == "on,off");
                    if let Some(p) = pos {
                        g[p] = "on".into();
                    }
                    tracing::warn!("tune: retrying grid with fa=on only");
                }
                3 => {
                    let pos = g.iter().position(|v| v == "2048,1024");
                    if let Some(p) = pos {
                        g[p] = "2048".into();
                    }
                    tracing::warn!("tune: retrying grid with default batch only");
                }
                _ => {}
            }
            match self.run(Path::new(input.model_path), &g) {
                Ok(rows) => {
                    grid_rows = Some(rows);
                    break;
                }
                Err(e) => {
                    tracing::warn!("tune: grid run failed: {e:#}");
                }
            }
        }
        let grid_rows = grid_rows.ok_or_else(|| anyhow!("llama-bench rejected every grid reduction"))?;

        // Argmax by mean tg t/s across (threads, quant, fa, batch) combos.
        let mut best: Option<(f64, Combo)> = None;
        for combo in combos(&grid_rows) {
            let score = mean_tg(&grid_rows, &combo);
            if best.as_ref().is_none_or(|(b, ..)| score > *b) {
                best = Some((score, combo));
            }
        }
        let Some((score, win)) = best else {
            return Err(anyhow!("llama-bench produced no tg rows"));
        };
        let winning = TuningOverrides {
            ctx: None,
            threads: Some(u32::try_from(win.threads).context("thread count overflow")?),
            kv_quant: Some(win.kv_q8),
            fa: Some(win.fa_on),
            batch: Some(u32::try_from(win.batch).context("batch overflow")?),
            ubatch: None,
        };
        let profile = profile::compile(input, &winning)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        persist_profile(store, input, &winning, &profile, &grid_rows, score)?;
        Ok((profile, winning, grid_rows))
    }
}

/// One measured combo: (threads, kv-q8?, fa-on?, batch).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Combo {
    pub threads: u64,
    pub kv_q8: bool,
    pub fa_on: bool,
    pub batch: u64,
}

fn combos(rows: &[BenchRow]) -> Vec<Combo> {
    let mut seen = BTreeSet::new();
    for r in rows {
        if !r.test_name().starts_with("tg") {
            continue;
        }
        seen.insert(Combo {
            threads: r.n_threads.unwrap_or(0),
            kv_q8: r.type_k.as_deref() == Some("q8_0") && r.type_v.as_deref() == Some("q8_0"),
            fa_on: r.flash_attn.unwrap_or(-1) == 1,
            batch: r.n_batch.unwrap_or(0),
        });
    }
    seen.into_iter().collect()
}

fn mean_tg(rows: &[BenchRow], want: &Combo) -> f64 {
    let hits: Vec<f64> = rows
        .iter()
        .filter(|r| {
            let cur = Combo {
                threads: r.n_threads.unwrap_or(0),
                kv_q8: r.type_k.as_deref() == Some("q8_0") && r.type_v.as_deref() == Some("q8_0"),
                fa_on: r.flash_attn.unwrap_or(-1) == 1,
                batch: r.n_batch.unwrap_or(0),
            };
            r.test_name().starts_with("tg") && &cur == want
        })
        .map(|r| r.ts)
        .collect();
    if hits.is_empty() {
        0.0
    } else {
        #[allow(clippy::cast_precision_loss)] // benchmark row counts are tiny
        return hits.iter().sum::<f64>() / hits.len() as f64;
    }
}

pub fn parse_bench_json(stdout: &str) -> Result<Vec<BenchRow>> {
    let trimmed = stdout.trim();
    if trimmed.starts_with('[') {
        return serde_json::from_str(trimmed).context("parse llama-bench json array");
    }
    // jsonl fallback
    let mut rows = Vec::new();
    for line in trimmed.lines() {
        if line.trim().starts_with('{') {
            rows.push(serde_json::from_str(line).context("parse llama-bench jsonl line")?);
        }
    }
    Ok(rows)
}

#[allow(clippy::similar_names)] // rows/bench_rows distinguished deliberately
fn persist_profile(
    store: &Store,
    input: &ProfileInput<'_>,
    tuning: &TuningOverrides,
    profile: &Profile,
    bench_rows: &[BenchRow],
    score: f64,
) -> Result<()> {
    let hash = args_hash(input, profile);
    // Tuning knobs are recorded via the winning argv (they ARE the argv);
    // `_tuning` is intentionally not serialized separately.
    let _ = tuning;
    let row = ProfileRow {
        model_name: input.model_name.to_string(),
        engine_tag: input.engine_tag.to_string(),
        args_hash: hash,
        args_json: serde_json::to_string(&profile.argv)?,
        benchmark_json: Some(serde_json::to_string(&BenchmarkPayload {
            score,
            rows: bench_rows.to_vec(),
        })?),
        updated_at: now_secs(),
    };
    store.upsert_profile(&row)?;
    Ok(())
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BenchmarkPayload {
    pub score: f64,
    pub rows: Vec<BenchRow>,
}

/// sha256 over (`engine_tag` + overlay + config-relevant fields + model
/// mtime) — recompile when any input drifts.
#[must_use] 
pub fn args_hash(input: &ProfileInput<'_>, profile: &Profile) -> String {
    let mtime = std::fs::metadata(input.model_path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_secs());
    let mut h = Sha256::new();
    h.update(input.engine_tag.as_bytes());
    h.update(format!("{:?}", input.overlay).as_bytes());
    h.update(
        format!(
            "{}|{}|{}|{}|{}|{}",
            input.config.default_ctx,
            input.config.cache_reuse,
            input.config.cache_ram_mb,
            input.config.idle_sleep_secs,
            input.config.rpc_servers,
            input.config.spec,
        )
        .as_bytes(),
    );
    h.update(mtime.to_le_bytes());
    h.update(profile.argv.join("\x1f").as_bytes());
    format!("{:x}", h.finalize())
}

fn now_secs() -> i64 {
    i64::try_from(SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs())
        .unwrap_or(i64::MAX)
}

/// Convenience: build a `ProfileInput` from stored state (used by CLI and
/// supervisor). `draft_path` resolution stays with the caller.
#[allow(clippy::too_many_arguments)]
#[must_use] 
pub fn build_input<'a>(
    model_name: &'a str,
    model_path: &'a str,
    model_bytes: u64,
    gguf: &'a GgufMeta,
    hardware: &'a Hardware,
    config: &'a Config,
    overlay: &'a ModelOverride,
    loras: &'a [(String, f64)],
    draft_path: Option<&'a str>,
    engine_tag: &'a str,
    supported_flags: &'a BTreeSet<String>,
    endpoint: Endpoint,
    data_dir: &'a str,
) -> ProfileInput<'a> {
    ProfileInput {
        model_name,
        model_path,
        model_bytes,
        gguf,
        hardware,
        config,
        overlay,
        loras,
        draft_path,
        mmproj_path: None, // vision is irrelevant to llama-bench scoring
        engine_tag,
        supported_flags,
        endpoint,
        data_dir,
    }
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    fn row(test: &str, ts: f64, ctx: u64, threads: u64, k: &str, v: &str) -> BenchRow {
        BenchRow {
            ts,
            test: test.into(),
            n_ctx: Some(ctx),
            n_threads: Some(threads),
            type_k: Some(k.into()),
            type_v: Some(v.into()),
            n_prompt: None,
            n_gen: None,
            flash_attn: None,
            n_batch: None,
        }
    }

    #[test]
    fn unit__parse_json_array() {
        let json = r#"[{"model":"m","test":"tg128","t/s":123.5,"n_ctx":8192,"n_threads":8,"type_k":"q8_0","type_v":"q8_0"}]"#;
        let rows = parse_bench_json(json).unwrap();
        assert_eq!(rows.len(), 1);
        assert!((rows[0].ts - 123.5).abs() < 1e-9);
    }

    #[test]
    fn unit__parse_jsonl_fallback() {
        let jsonl = "{\"test\":\"pp512\",\"t/s\":900}\n{\"test\":\"tg128\",\"t/s\":100}\n";
        let rows = parse_bench_json(jsonl).unwrap();
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn unit__mean_tg__filters_by_combo() {
        let rows = vec![
            row("tg128", 100.0, 8192, 8, "f16", "f16"),
            row("tg128", 140.0, 8192, 8, "q8_0", "q8_0"),
            row("tg128", 110.0, 4096, 6, "q8_0", "q8_0"),
            row("pp512", 900.0, 8192, 8, "f16", "f16"),
        ];
        let find = |t, q| {
            combos(&rows).into_iter().find(|c| c.threads == t && c.kv_q8 == q).unwrap()
        };
        assert!((mean_tg(&rows, &find(8, true)) - 140.0).abs() < 1e-9);
        assert!((mean_tg(&rows, &find(8, false)) - 100.0).abs() < 1e-9);
        assert!((mean_tg(&rows, &find(6, true)) - 110.0).abs() < 1e-9);
        assert!((mean_tg(&rows, &Combo { threads: 1, kv_q8: false, fa_on: false, batch: 0 }) - 0.0).abs() < 1e-9);
        assert_eq!(combos(&rows).len(), 3);
    }
}
