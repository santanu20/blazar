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

/// One `tune --load` A/B measurement: wall seconds (spawn→ready +
/// first-chat) for the warmup and no-warmup variants of the same argv.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoadProbe {
    pub warmup_secs: f64,
    pub no_warmup_secs: f64,
}

/// Aggregate generation tok/s with one vs two identical children
/// (`tune --replicas`). Adoption is the caller's policy (>1.3x).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReplicaSearch {
    pub r1_tps: f64,
    pub r2_tps: f64,
}

/// `--cache-reuse` grid result (`tune --cache-reuse`): measured warm
/// second-chat wall seconds per N value, plus the grid's best N.
/// Adoption is the caller's policy (best < 0.95x the current default).
#[derive(Debug, Clone, PartialEq)]
pub struct CacheReuseSearch {
    /// ((`cache_reuse` N, warm chat wall secs)) per grid point, in grid order.
    pub grid: Vec<(u32, f64)>,
    /// N with the lowest warm-chat wall time.
    pub best: u32,
}

/// Base argv for the warmup A/B: drop endpoint/session/persisted-state
/// PAIRS and any explicit warmup flags so the two variants differ ONLY
/// in the warmup axis.
#[must_use]
pub fn strip_for_load(base_argv: &[String]) -> Vec<String> {
    let mut argv: Vec<String> = Vec::new();
    let mut it = base_argv.iter().peekable();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--host" | "--port" | "--slot-save-path" | "--lookup-cache-dynamic" => {
                let _ = it.next(); // consume the value
            }
            "--no-warmup" | "--warmup" => {}
            _ => argv.push(a.clone()),
        }
    }
    argv
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
        .map(|e| {
            dirs.engines_dir()
                .join(&e.tag)
                .join(format!("llama-{}", e.tag))
                .join(crate::tool_file_name("llama-bench"))
        })
        .find(|p| p.exists())
        .ok_or_else(|| {
            anyhow!("no llama-bench found in any installed engine; run `pallama engine update`")
        })
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

    /// Live-serving slots search: for each `-np` candidate, spawn the
    /// real llama-server (argv cloned from the compiled profile, ports
    /// ephemeral), drive `clients` concurrent non-stream chats, and
    /// measure aggregate wall tokens/s. llama-bench cannot see slot
    /// concurrency — this is the only honest axis for `-np`.
    pub fn slots_search(
        &self,
        base_argv: &[String],
        clients: u32,
        candidates: &[u32],
        idle_secs: u64,
    ) -> Result<Vec<(u32, f64)>> {
        let server_bin = self
            .bench_bin
            .parent()
            .map(|p| p.join("llama-server"))
            .filter(|p| p.exists())
            .ok_or_else(|| anyhow!("llama-server not found next to llama-bench"))?;
        let mut out = Vec::new();
        for np in candidates {
            if *np > clients.max(1) {
                continue; // more slots than clients measures nothing
            }
            let port = ephemeral_port()?;
            // Strip endpoint/slot PAIRS (flag + value), then set ours.
            let mut argv: Vec<String> = Vec::new();
            let mut it = base_argv.iter().peekable();
            while let Some(a) = it.next() {
                match a.as_str() {
                    "--host" | "--port" | "-np" | "--slot-save-path" => {
                        let _ = it.next(); // consume the value
                    }
                    _ => argv.push(a.clone()),
                }
            }
            argv.extend([
                "--host".to_string(),
                "127.0.0.1".to_string(),
                "--port".to_string(),
                port.to_string(),
                "-np".to_string(),
                np.to_string(),
            ]);
            let mut child = std::process::Command::new(&server_bin)
                .args(&argv)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .with_context(|| format!("spawn {}", server_bin.display()))?;
            let ok = wait_ready(&mut child, port, 120.0);
            let tps = if ok {
                drive_concurrent(port, clients).ok()
            } else {
                None
            };
            // Single pid we spawned; never a group.
            let _ = child.kill();
            let _ = child.wait();
            match tps {
                Some(t) => out.push((*np, t)),
                None => {
                    anyhow::bail!("slots search: server at -np {np} never became ready or served")
                }
            }
            std::thread::sleep(std::time::Duration::from_secs(idle_secs.max(1)));
        }
        if out.is_empty() {
            return Err(anyhow!(
                "no -np candidates ran (clients={clients}, candidates={candidates:?})"
            ));
        }
        Ok(out)
    }

    /// Live n-gram tuning search: for each (`size_m`, `min_hits`) candidate,
    /// spawn the real llama-server with `--spec-type ngram-simple` and the
    /// candidate's typed tuning flags, drive ONE non-stream chat (spec
    /// decoding is a single-stream win; concurrency would mask it), and
    /// measure wall tokens/s. llama-bench has no spec axis — this is the
    /// only honest measurement lane for ngram knobs.
    pub fn ngram_search(
        &self,
        base_argv: &[String],
        candidates: &[(u32, u32)],
        idle_secs: u64,
    ) -> Result<Vec<((u32, u32), f64)>> {
        let server_bin = self
            .bench_bin
            .parent()
            .map(|p| p.join("llama-server"))
            .filter(|p| p.exists())
            .ok_or_else(|| anyhow!("llama-server not found next to llama-bench"))?;
        let mut out = Vec::new();
        for &(m, h) in candidates {
            let port = ephemeral_port()?;
            // Strip endpoint/session/persisted-spec PAIRS: the lookup
            // cache would leak drafts between candidates and contaminate
            // the comparison.
            let mut argv: Vec<String> = Vec::new();
            let mut it = base_argv.iter().peekable();
            while let Some(a) = it.next() {
                match a.as_str() {
                    "--host"
                    | "--port"
                    | "--slot-save-path"
                    | "--lookup-cache-dynamic"
                    | "--spec-type"
                    | "--spec-ngram-simple-size-m"
                    | "--spec-ngram-simple-size-n"
                    | "--spec-ngram-simple-min-hits"
                    | "--spec-ngram-map-k-size-m"
                    | "--spec-ngram-map-k-size-n"
                    | "--spec-ngram-map-k-min-hits"
                    | "--spec-ngram-map-k4v-size-m"
                    | "--spec-ngram-map-k4v-size-n"
                    | "--spec-ngram-map-k4v-min-hits"
                    | "--spec-ngram-mod-n-match"
                    | "--spec-ngram-mod-n-max"
                    | "--spec-ngram-mod-n-min" => {
                        let _ = it.next(); // consume the value
                    }
                    _ => argv.push(a.clone()),
                }
            }
            argv.extend([
                "--host".to_string(),
                "127.0.0.1".to_string(),
                "--port".to_string(),
                port.to_string(),
                "--spec-type".to_string(),
                "ngram-simple".to_string(),
                "--spec-ngram-simple-size-m".to_string(),
                m.to_string(),
                "--spec-ngram-simple-size-n".to_string(),
                "8".to_string(),
                "--spec-ngram-simple-min-hits".to_string(),
                h.to_string(),
            ]);
            let mut child = std::process::Command::new(&server_bin)
                .args(&argv)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .with_context(|| format!("spawn {}", server_bin.display()))?;
            let ok = wait_ready(&mut child, port, 120.0);
            let tps = if ok {
                drive_concurrent(port, 1).ok()
            } else {
                None
            };
            // Single pid we spawned; never a group.
            let _ = child.kill();
            let _ = child.wait();
            match tps {
                Some(t) => {
                    println!("  ngram size_m={m} min_hits={h}: {t:.1} tok/s");
                    out.push(((m, h), t));
                }
                None => anyhow::bail!(
                    "ngram search: server at size_m={m} min_hits={h} never became ready or served"
                ),
            }
            std::thread::sleep(std::time::Duration::from_secs(idle_secs.max(1)));
        }
        if out.is_empty() {
            anyhow::bail!("no ngram candidates ran ({candidates:?})");
        }
        Ok(out)
    }

    /// `--cache-reuse N` grid search (`tune --cache-reuse`): for each N in
    /// {0, 256, 512} spawn the real llama-server, send the SAME long prompt
    /// twice, and measure the WALL seconds of the SECOND (warm, cache-hit)
    /// chat. The prompt is built from one repeated sentence so the engine's
    /// chunked-prompt cache can reuse it. N=0 omits the flag (upstream
    /// default: cache reuse disabled); Pallama's shipped default is 256, so
    /// adoption must beat 256 by >5% — printing the whole grid keeps the
    /// verdict honest even when nothing wins.
    pub fn cache_reuse_search(
        &self,
        base_argv: &[String],
        idle_secs: u64,
    ) -> Result<CacheReuseSearch> {
        let server_bin = self
            .bench_bin
            .parent()
            .map(|p| p.join("llama-server"))
            .filter(|p| p.exists())
            .ok_or_else(|| anyhow!("llama-server not found next to llama-bench"))?;
        // ~800+ tokens of deterministic filler: one sentence repeated far
        // past the 256-token chunk watermark so partial-chunk reuse cannot
        // fake a hit.
        let long_prompt =
            "The lighthouse keeper logged the tide twice a day and mailed the ledger monthly. "
                .repeat(64);
        let mut grid = Vec::new();
        for n in [0u32, 256, 512] {
            let port = ephemeral_port()?;
            // Strip endpoint/session/persisted-cache PAIRS plus any prior
            // --cache-reuse so candidates differ ONLY in the reuse axis.
            let mut argv: Vec<String> = Vec::new();
            let mut it = base_argv.iter().peekable();
            while let Some(a) = it.next() {
                match a.as_str() {
                    "--host" | "--port" | "--slot-save-path" | "--cache-reuse" => {
                        let _ = it.next(); // consume the value
                    }
                    _ => argv.push(a.clone()),
                }
            }
            argv.extend([
                "--host".to_string(),
                "127.0.0.1".to_string(),
                "--port".to_string(),
                port.to_string(),
            ]);
            if n > 0 {
                argv.extend(["--cache-reuse".to_string(), n.to_string()]);
            }
            let mut child = std::process::Command::new(&server_bin)
                .args(&argv)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .with_context(|| format!("spawn {}", server_bin.display()))?;
            let warm_secs = if wait_ready(&mut child, port, 180.0) {
                // Prime, then measure the warm pass (the cache-hit lane).
                chat_secs(port, &long_prompt)
                    .ok()
                    .and_then(|_| chat_secs(port, &long_prompt).ok())
            } else {
                None
            };
            // Single pid we spawned; never a group.
            let _ = child.kill();
            let _ = child.wait();
            let Some(secs) = warm_secs else {
                anyhow::bail!("cache-reuse search: server at N={n} never became ready or served");
            };
            println!("  cache-reuse {n}: warm chat {secs:.2}s");
            grid.push((n, secs));
            std::thread::sleep(std::time::Duration::from_secs(idle_secs.max(1)));
        }
        let best = grid
            .iter()
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(n, _)| *n)
            .ok_or_else(|| anyhow!("cache-reuse grid ran empty"))?;
        Ok(CacheReuseSearch { grid, best })
    }

    /// Warmup-axis A/B probe (`tune --load`): spawn the real llama-server
    /// twice, differing ONLY in `--no-warmup`, and measure spawn→ready
    /// plus the first chat round-trip. Warmup trades startup seconds for
    /// first-token latency; the metric prices both into one number. The
    /// no-warmup variant runs FIRST so the second (warmup) spawn enjoys
    /// the warm page cache — biasing AGAINST adoption, which is the safe
    /// direction for a knob that disables an engine default.
    pub fn load_search(&self, base_argv: &[String], idle_secs: u64) -> Result<LoadProbe> {
        let server_bin = self
            .bench_bin
            .parent()
            .map(|p| p.join("llama-server"))
            .filter(|p| p.exists())
            .ok_or_else(|| anyhow!("llama-server not found next to llama-bench"))?;
        let base = strip_for_load(base_argv);
        let mut probe = LoadProbe {
            warmup_secs: 0.0,
            no_warmup_secs: 0.0,
        };
        for warmup in [false, true] {
            let port = ephemeral_port()?;
            let mut argv = base.clone();
            argv.extend([
                "--host".to_string(),
                "127.0.0.1".to_string(),
                "--port".to_string(),
                port.to_string(),
            ]);
            if !warmup {
                argv.push("--no-warmup".to_string());
            }
            let t0 = std::time::Instant::now();
            let mut child = std::process::Command::new(&server_bin)
                .args(&argv)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .with_context(|| format!("spawn {}", server_bin.display()))?;
            let total = if let (true, Ok(first)) =
                (wait_ready(&mut child, port, 180.0), first_chat_secs(port))
            {
                t0.elapsed().as_secs_f64() + first
            } else {
                // Single pid we spawned; never a group.
                let _ = child.kill();
                let _ = child.wait();
                anyhow::bail!("load search: server (warmup={warmup}) never became ready or served");
            };
            // Single pid we spawned; never a group.
            let _ = child.kill();
            let _ = child.wait();
            if warmup {
                probe.warmup_secs = total;
            } else {
                probe.no_warmup_secs = total;
            }
            std::thread::sleep(std::time::Duration::from_secs(idle_secs.max(1)));
        }
        Ok(probe)
    }

    /// `tune --replicas`: measure aggregate generation throughput with one
    /// vs two identical children; the caller decides adoption (>1.3x).
    pub fn replica_search(&self, base_argv: &[String], idle_secs: u64) -> Result<ReplicaSearch> {
        let server_bin = self
            .bench_bin
            .parent()
            .map(|p| p.join("llama-server"))
            .filter(|p| p.exists())
            .ok_or_else(|| anyhow!("llama-server not found next to llama-bench"))?;
        let base = strip_for_load(base_argv);
        let spawn_on = |port: u16| -> Result<std::process::Child> {
            let mut argv = base.clone();
            argv.extend([
                "--host".to_string(),
                "127.0.0.1".to_string(),
                "--port".to_string(),
                port.to_string(),
            ]);
            std::process::Command::new(&server_bin)
                .args(&argv)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .with_context(|| format!("spawn {}", server_bin.display()))
        };

        // Run A: single child.
        let p1 = ephemeral_port()?;
        let mut a = spawn_on(p1)?;
        if !wait_ready(&mut a, p1, 180.0) {
            let _ = a.kill();
            let _ = a.wait();
            anyhow::bail!("replica search: baseline child never became ready");
        }
        let r1_tps = drive_concurrent_multi(&[p1], 8, 3);
        // Single pid we spawned; never a group.
        let _ = a.kill();
        let _ = a.wait();
        let r1_tps = r1_tps?;
        std::thread::sleep(std::time::Duration::from_secs(idle_secs.max(1)));

        // Run B: two identical children (same argv, distinct ports).
        let p2 = ephemeral_port()?;
        let p3 = ephemeral_port()?;
        let mut b1 = spawn_on(p2)?;
        let mut b2 = spawn_on(p3)?;
        let ready2 = wait_ready(&mut b1, p2, 180.0);
        let ready3 = wait_ready(&mut b2, p3, 180.0);
        if !(ready2 && ready3) {
            let _ = b1.kill();
            let _ = b1.wait();
            let _ = b2.kill();
            let _ = b2.wait();
            anyhow::bail!(
                "replica search: second child never became ready — the card likely \
                 cannot hold two copies (VRAM/ RAM); keep replicas = 1"
            );
        }
        let r2_tps = drive_concurrent_multi(&[p2, p3], 8, 3);
        // Single pids we spawned; never a group.
        let _ = b1.kill();
        let _ = b1.wait();
        let _ = b2.kill();
        let _ = b2.wait();
        let r2_tps = r2_tps?;
        Ok(ReplicaSearch { r1_tps, r2_tps })
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
        let profile =
            profile::compile(input, overrides).map_err(|e| anyhow::anyhow!("profile: {e}"))?;
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
        // Validation gate: profile must compile before we spawn anything.
        profile::compile(input, &TuningOverrides::default()).map_err(|e| anyhow::anyhow!("{e}"))?;
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
                    // KV-quant axis off — value-based (F107): collapse
                    // every pair-axis value to its first leg; no
                    // positional assumption about where the axis sits.
                    for v in &mut g {
                        if v == "f16,q8_0" {
                            *v = "f16".into();
                        }
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
        let grid_rows =
            grid_rows.ok_or_else(|| anyhow!("llama-bench rejected every grid reduction"))?;

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
        let profile = profile::compile(input, &winning).map_err(|e| anyhow::anyhow!("{e}"))?;
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
    i64::try_from(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
    )
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
    spec_types: &'a [String],
    endpoint: Endpoint,
    data_dir: &'a str,
) -> ProfileInput<'a> {
    ProfileInput {
        model_name,
        instance_key: model_name,
        model_path,
        model_bytes,
        gguf,
        hardware,
        config,
        overlay,
        loras,
        draft_path,
        mmproj_path: None, // vision is irrelevant to llama-bench scoring
        mmproj_force: false,
        engine_tag,
        supported_flags,
        spec_types,
        // llama-bench scoring is llama-server-only (the mistral.rs lane
        // prints a gate skip instead of benching).
        engine_kind: pallama_core::engine_kind::EngineKind::LlamaCpp,
        sibling_devices: Vec::new(),
        auto_tensor_split: None,
        endpoint,
        data_dir,
        cache_hit_rate: None, // CLI bench: static clamp, no live hint
        device_hint: None,
        engine_census: hardware.gpus.clone(),
    }
}

/// Grab a free TCP port the OS assigns (bind to :0, read it, close).
fn ephemeral_port() -> Result<u16> {
    use std::net::TcpListener;
    let l = TcpListener::bind(("127.0.0.1", 0)).context("bind ephemeral")?;
    Ok(l.local_addr().context("local addr")?.port())
}

/// Poll GET /health until 200 or timeout.
#[allow(clippy::items_after_statements)] // io trait imports sit near their single use
fn wait_ready(child: &mut std::process::Child, port: u16, timeout_secs: f64) -> bool {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs_f64(timeout_secs);
    while std::time::Instant::now() < deadline {
        // F95: a crashed server fails fast instead of spinning the full
        // timeout against a dead port.
        if matches!(child.try_wait(), Ok(Some(_))) {
            return false;
        }
        if let Ok(mut s) = std::net::TcpStream::connect(("127.0.0.1", port)) {
            use std::io::{Read, Write};
            let _ = s.write_all(b"GET /health HTTP/1.1\r\nhost: x\r\nconnection: close\r\n\r\n");
            let mut buf = [0u8; 256];
            if let Ok(n) = s.read(&mut buf) {
                // F100: judge the STATUS LINE, not "200" anywhere (a
                // content-length or body digit would false-positive).
                let head = String::from_utf8_lossy(&buf[..n]);
                if head.starts_with("HTTP/1.1 200") || head.starts_with("HTTP/1.0 200") {
                    return true;
                }
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(250));
    }
    false
}

/// First `n` bytes of `text` cut at a char boundary — lossy-decoded socket
/// buffers can end mid-char and a raw slice would panic (F100).
fn head_bytes(text: &str, n: usize) -> &str {
    let mut cut = text.len().min(n);
    while !text.is_char_boundary(cut) {
        cut -= 1;
    }
    &text[..cut]
}

/// Wall seconds for ONE tiny non-stream chat round-trip. `tune --load`
/// adds this to spawn→ready so the no-warmup variant's first-request
/// lazy-init penalty is priced into its metric instead of silently
/// vanishing into the user's first prompt.
fn first_chat_secs(port: u16) -> Result<f64> {
    chat_secs(port, "hi")
}

/// Wall seconds for one non-stream chat with the given user content.
/// Shared by the load probe (tiny "hi") and the cache-reuse probe
/// (long repeated prefix — see `cache_reuse_search`).
#[allow(clippy::items_after_statements)] // io trait imports sit near their single use
fn chat_secs(port: u16, content: &str) -> Result<f64> {
    let t0 = std::time::Instant::now();
    let body = serde_json::json!({
        "model": "load-probe", "max_tokens": 8, "stream": false,
        "messages": [{"role": "user", "content": content}],
    });
    let payload = serde_json::to_vec(&body)?;
    let mut s = std::net::TcpStream::connect(("127.0.0.1", port))
        .with_context(|| format!("connect :{port}"))?;
    use std::io::{Read, Write};
    let req = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nhost: x\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
        payload.len()
    );
    s.write_all(req.as_bytes())?;
    s.write_all(&payload)?;
    let mut buf = Vec::new();
    s.read_to_end(&mut buf)?;
    let text = String::from_utf8_lossy(&buf);
    if !(text.starts_with("HTTP/1.1 200") || text.starts_with("HTTP/1.0 200")) {
        anyhow::bail!("load probe: first chat failed: {}", head_bytes(&text, 120));
    }
    Ok(t0.elapsed().as_secs_f64())
}

/// `clients` concurrent non-stream chats; aggregate tokens/s across all
/// walls (blocking threads — this runs in a CLI, not the runtime).
#[allow(clippy::items_after_statements)] // io trait imports sit near their single use
#[allow(clippy::cast_precision_loss)] // throughput display only
fn drive_concurrent(port: u16, clients: u32) -> Result<f64> {
    drive_concurrent_multi(&[port], clients, 1)
}

/// One blocking non-stream chat; returns completion tokens from usage.
#[allow(clippy::items_after_statements)] // io trait imports sit near their single use
fn chat_completion_tokens(port: u16, content: &str) -> Result<u64> {
    let body = serde_json::json!({
        "model": "replica-search", "max_tokens": 64, "stream": false,
        "messages": [{"role": "user", "content": content}],
    });
    let mut s = std::net::TcpStream::connect(("127.0.0.1", port))
        .with_context(|| format!("connect :{port}"))?;
    let payload = serde_json::to_vec(&body)?;
    use std::io::{Read, Write};
    let req = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nhost: x\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
        payload.len()
    );
    s.write_all(req.as_bytes())?;
    s.write_all(&payload)?;
    let mut buf = Vec::new();
    s.read_to_end(&mut buf)?;
    let text = String::from_utf8_lossy(&buf);
    if !(text.starts_with("HTTP/1.1 200") || text.starts_with("HTTP/1.0 200")) {
        anyhow::bail!(
            "replica search: chat failed on :{port}: {}",
            head_bytes(&text, 120)
        );
    }
    Ok(text
        .split("\"completion_tokens\":")
        .nth(1)
        .and_then(|rest| {
            rest.chars()
                .take_while(char::is_ascii_digit)
                .collect::<String>()
                .parse::<u64>()
                .ok()
        })
        .unwrap_or(0))
}

/// Aggregate tok/s of `clients` blocking threads round-robin across `ports`,
/// `gens` sequential requests each. The wall clock spans ALL clients, so a
/// saturated single child and a parallel pair are directly comparable.
#[allow(clippy::cast_precision_loss)] // throughput display only
fn drive_concurrent_multi(ports: &[u16], clients: u32, gens: u32) -> Result<f64> {
    if ports.is_empty() {
        anyhow::bail!("replica search: no ports to drive");
    }
    let mut handles = Vec::new();
    let start = std::time::Instant::now();
    for i in 0..clients {
        let port = ports[(i as usize) % ports.len()];
        handles.push(std::thread::spawn(move || -> Result<u64> {
            let mut total = 0u64;
            for g in 0..gens {
                total += chat_completion_tokens(
                    port,
                    &format!("count slowly: {g} one two three four five"),
                )?;
            }
            Ok(total)
        }));
    }
    let mut total = 0u64;
    for h in handles {
        total += h.join().map_err(|_| anyhow!("client thread panicked"))??;
    }
    let secs = start.elapsed().as_secs_f64();
    Ok(total as f64 / secs.max(1e-6))
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
    fn unit__strip_for_load_drops_warmup_and_state_pairs() {
        let base = [
            "--model",
            "/m.gguf",
            "--host",
            "0.0.0.0",
            "--port",
            "1",
            "--slot-save-path",
            "/s",
            "--lookup-cache-dynamic",
            "/c",
            "--no-warmup",
            "--threads",
            "4",
        ]
        .iter()
        .map(std::string::ToString::to_string)
        .collect::<Vec<_>>();
        let out = strip_for_load(&base);
        assert_eq!(out, ["--model", "/m.gguf", "--threads", "4"]);
    }

    #[test]
    fn unit__drive_concurrent_multi__rejects_empty_ports() {
        assert!(drive_concurrent_multi(&[], 4, 1).is_err());
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
            combos(&rows)
                .into_iter()
                .find(|c| c.threads == t && c.kv_q8 == q)
                .unwrap()
        };
        assert!((mean_tg(&rows, &find(8, true)) - 140.0).abs() < 1e-9);
        assert!((mean_tg(&rows, &find(8, false)) - 100.0).abs() < 1e-9);
        assert!((mean_tg(&rows, &find(6, true)) - 110.0).abs() < 1e-9);
        assert!(
            (mean_tg(
                &rows,
                &Combo {
                    threads: 1,
                    kv_q8: false,
                    fa_on: false,
                    batch: 0
                }
            ) - 0.0)
                .abs()
                < 1e-9
        );
        assert_eq!(combos(&rows).len(), 3);
    }
}
