//! `blazar optimize --run` / `--apply` - the Wave 2/3 execution layer.
//!
//! Wave 2 (`--run`) screens a candidate through the llama-bench grid
//! WITHOUT adopting it, then re-measures baseline vs candidate through
//! the REAL gateway path: swap the `profiles` row, evict the child,
//! warm it back on the new argv, and probe with alternating A/B reps
//! so drift and thermal effects hit both arms equally. A pure
//! `verify()` turns the two sample sets into one noise-gated verdict,
//! and the whole run lands in the append-only optimization ledger.
//! Nothing here adopts: adoption is [`adopt_experiment`] (Wave 3),
//! which refuses any experiment without a `verified` event.
//!
//! Invariants this module defends:
//! - the pre-run `profiles` row is restored byte-identically, even on
//!   early error (see [`RestoreGuard`]);
//! - the serving path is only ever touched through the daemon's own
//!   HTTP surface (evict/warm/chat) - never by poking the child;
//! - a lease check refuses to run while the lane serves traffic.

use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use futures::StreamExt;
use serde::{Deserialize, Serialize};

use blazar_core::dirs::BlazarDirs;
use blazar_core::profile::{Profile, ProfileInput};
use blazar_core::store::{ProfileRow, Store};

use crate::bench::{Tuner, args_hash};

/// Fixed probe prompt - identical for both arms, both reps. Content is
/// irrelevant to the measurement; determinism is not (temperature 0).
const PROBE_PROMPT: &str = "Explain in plain sentences why measured \
evidence beats assumption when tuning a local inference server. Cover \
hardware, engines, and workload. Be concise and concrete.";

/// How long a single probe request may stream before giving up.
const PROBE_TIMEOUT: Duration = Duration::from_secs(180);
/// Poll cadence while waiting for the child to come back up.
const READY_POLL: Duration = Duration::from_millis(250);
/// Upper bound on respawn wait (model load on slow disks).
const READY_DEADLINE: Duration = Duration::from_secs(120);

/// Knobs for one `--run` invocation. Every field is CLI-settable so
/// the operator owns the benchmark cost.
#[derive(Debug, Clone)]
pub struct OptimizeRunConfig {
    /// Daemon base URL (`http://host:port`).
    pub base_url: String,
    /// Concurrent requests per probe round (the C in Throughput@C).
    pub concurrency: u32,
    /// Alternating A/B repetitions per arm (warm-up excluded).
    pub reps: u32,
    /// Generated tokens per probe request.
    pub num_predict: u32,
    /// Minimum relative improvement (percent) beyond noise before a
    /// candidate can be recommended.
    pub min_gain_pct: f64,
}

impl Default for OptimizeRunConfig {
    fn default() -> Self {
        Self {
            base_url: "http://127.0.0.1:11435".to_string(),
            concurrency: 4,
            reps: 3,
            num_predict: 128,
            min_gain_pct: 5.0,
        }
    }
}

/// The explicit objective from the frozen architecture: hard
/// constraints first, then Pareto, then one named objective with
/// deterministic tie-breaks. `Memory` stays honestly ineligible until
/// a VRAM telemetry lane exists - missing telemetry is never estimated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizeObjective {
    Interactive,
    Throughput,
    Memory,
}

impl OptimizeObjective {
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Interactive => "Interactive",
            Self::Throughput => "Throughput",
            Self::Memory => "Memory",
        }
    }

    /// Parse from the CLI spelling (case-insensitive).
    pub fn parse(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "interactive" => Ok(Self::Interactive),
            "throughput" => Ok(Self::Throughput),
            "memory" => Ok(Self::Memory),
            other => Err(anyhow::anyhow!(
                "unknown objective '{other}' (interactive | throughput | memory)"
            )),
        }
    }
}

/// One request's streaming measurement (Ollama `/api/chat` ndjson).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbeSample {
    /// Send -> first token line (ms).
    pub ttft_ms: f64,
    /// Inter-line arrival deltas (ms) - per-token under the gateway's
    /// token-granular streaming.
    pub itl_ms: Vec<f64>,
    pub chunk_count: u64,
    /// Authoritative token count from the final usage line.
    pub eval_count: u64,
    /// Send -> done line (ms).
    pub wall_ms: f64,
}

/// Aggregated samples for one arm (baseline or candidate), serialized
/// into the ledger as the arm's measurement row.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArmStats {
    pub ttft_p50_ms: f64,
    pub ttft_p95_ms: f64,
    pub itl_p50_ms: f64,
    pub itl_p95_ms: f64,
    /// Aggregate generation throughput at the run's concurrency:
    /// `sum(eval) / (sum(wall)/C)` - identical arithmetic for both
    /// arms, so the ratio is fair whatever the queueing shape.
    pub aggregate_tps: f64,
    /// Per-rep medians/derived values feeding the noise estimate.
    pub per_rep_tps: Vec<f64>,
    pub per_rep_ttft_p95: Vec<f64>,
    pub per_rep_itl_p95: Vec<f64>,
    pub requests: u32,
    pub eval_tokens: u64,
}

// integer counters -> f64 statistics: values are token counts and
// millisecond spans far below 2^52, so precision loss is impossible
#[allow(clippy::cast_precision_loss)]
impl ArmStats {
    fn from_samples(samples: &[ProbeSample], per_rep: &[RepDigest]) -> Self {
        let mut ttft: Vec<f64> = samples.iter().map(|s| s.ttft_ms).collect();
        ttft.sort_by(f64::total_cmp);
        let mut itl: Vec<f64> = samples
            .iter()
            .flat_map(|s| s.itl_ms.iter().copied())
            .collect();
        itl.sort_by(f64::total_cmp);
        let eval: u64 = samples.iter().map(|s| s.eval_count).sum();
        let wall_sum: f64 = samples.iter().map(|s| s.wall_ms).sum::<f64>();
        let mean_wall = if samples.is_empty() {
            0.0
        } else {
            wall_sum / samples.len() as f64
        };
        Self {
            ttft_p50_ms: percentile(&ttft, 50.0),
            ttft_p95_ms: percentile(&ttft, 95.0),
            itl_p50_ms: percentile(&itl, 50.0),
            itl_p95_ms: percentile(&itl, 95.0),
            aggregate_tps: if mean_wall > 0.0 {
                // Under real concurrency every request spans ~mean_wall,
                // so total tokens over the mean span IS the aggregate
                // rate. Identical arithmetic for both arms - the A/B
                // ratio stays fair whatever the queueing shape.
                eval as f64 / (mean_wall / 1000.0)
            } else {
                0.0
            },
            per_rep_tps: per_rep.iter().map(|r| r.tps).collect(),
            per_rep_ttft_p95: per_rep.iter().map(|r| r.ttft_p95).collect(),
            per_rep_itl_p95: per_rep.iter().map(|r| r.itl_p95).collect(),
            requests: u32::try_from(samples.len()).unwrap_or(u32::MAX),
            eval_tokens: eval,
        }
    }
}

/// Per-rep digests kept while measuring (noise input).
struct RepDigest {
    tps: f64,
    ttft_p95: f64,
    itl_p95: f64,
}

/// Nearest-rank percentile over an ASCENDING sorted slice. Empty input
// p/100 and n multiply to at most n <= usize::MAX; nearest-rank
// ceiling then clamps into the slice - truncation is the algorithm
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
/// returns 0.0 (callers gate on non-empty sample sets long before).
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let n = sorted.len();
    let rank = ((p / 100.0) * n as f64).ceil() as usize;
    let idx = rank.saturating_sub(1).min(n - 1);
    sorted[idx]
}

/// The noise-gated multi-objective verdict. Pure - unit-tested without
/// any HTTP or store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Verdict {
    pub objective: String,
    pub recommended: bool,
    pub metric: String,
    pub baseline: f64,
    pub candidate: f64,
    pub noise: f64,
    pub threshold: f64,
    pub reason: String,
}

fn noise_for(base_series: &[f64], cand_series: &[f64]) -> f64 {
    // Observed per-rep spread of BOTH arms; noise is what an identical
    // configuration can produce on its own, so take the larger arm's.
    let s = |v: &[f64]| {
        v.iter()
            .copied()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), x| {
                (lo.min(x), hi.max(x))
            })
    };
    let (blo, bhi) = s(base_series);
    let (clo, chi) = s(cand_series);
    (bhi - blo).max(chi - clo).max(0.0)
}

/// Decide whether the candidate beats the baseline beyond noise plus
/// the operator's minimum gain. Interactive is lexicographic: p95 TTFT
/// first, p95 ITL as tie-break only within the TTFT gate.
#[must_use]
pub fn verify(
    objective: OptimizeObjective,
    base: &ArmStats,
    cand: &ArmStats,
    min_gain_pct: f64,
) -> Verdict {
    let pct = min_gain_pct.max(0.0);
    match objective {
        OptimizeObjective::Memory => Verdict {
            objective: objective.label().to_string(),
            recommended: false,
            metric: "peak_vram".to_string(),
            baseline: 0.0,
            candidate: 0.0,
            noise: 0.0,
            threshold: 0.0,
            reason: "ineligible: no VRAM telemetry lane is wired - a memory \
objective must not adopt on estimates"
                .to_string(),
        },
        OptimizeObjective::Throughput => {
            let noise = noise_for(&base.per_rep_tps, &cand.per_rep_tps);
            let threshold = (pct / 100.0 * base.aggregate_tps).max(noise);
            let gain = cand.aggregate_tps - base.aggregate_tps;
            Verdict {
                objective: objective.label().to_string(),
                recommended: gain > threshold,
                metric: "aggregate_tok_s".to_string(),
                baseline: base.aggregate_tps,
                candidate: cand.aggregate_tps,
                noise,
                threshold,
                reason: format!(
                    "candidate {gain:+.1} tok/s vs threshold {threshold:.1} \
                     (min gain {pct:.0}% of baseline, observed noise {noise:.1})"
                ),
            }
        }
        OptimizeObjective::Interactive => {
            let ttft_noise = noise_for(&base.per_rep_ttft_p95, &cand.per_rep_ttft_p95);
            let ttft_thr = (pct / 100.0 * base.ttft_p95_ms).max(ttft_noise);
            let ttft_gain = base.ttft_p95_ms - cand.ttft_p95_ms;
            if ttft_gain <= ttft_thr {
                return Verdict {
                    objective: objective.label().to_string(),
                    recommended: false,
                    metric: "p95_ttft_ms".to_string(),
                    baseline: base.ttft_p95_ms,
                    candidate: cand.ttft_p95_ms,
                    noise: ttft_noise,
                    threshold: ttft_thr,
                    reason: format!(
                        "TTFT not beyond gate: {ttft_gain:+.1} ms vs threshold \
                         {ttft_thr:.1} ms (min gain {pct:.0}%, noise {ttft_noise:.1} ms)"
                    ),
                };
            }
            let itl_noise = noise_for(&base.per_rep_itl_p95, &cand.per_rep_itl_p95);
            let itl_thr = (pct / 100.0 * base.itl_p95_ms).max(itl_noise);
            let itl_gain = base.itl_p95_ms - cand.itl_p95_ms;
            Verdict {
                objective: objective.label().to_string(),
                recommended: itl_gain >= -itl_thr,
                metric: "p95_ttft_ms+p95_itl_ms".to_string(),
                baseline: base.ttft_p95_ms,
                candidate: cand.ttft_p95_ms,
                noise: ttft_noise,
                threshold: ttft_thr,
                reason: format!(
                    "TTFT {ttft_gain:+.1} ms beyond gate {ttft_thr:.1} ms; ITL \
                     {itl_gain:+.1} ms (regression tolerated up to {itl_thr:.1} ms)"
                ),
            }
        }
    }
}

/// Restores the pre-run `profiles` row even when the run dies early.
/// Restore is upsert-when-present / delete-when-absent - an absent
/// baseline must not become a placeholder row that masquerades as a
/// tuned profile. rusqlite is synchronous, so Drop can genuinely fail
/// loudly; it never fails silently.
struct RestoreGuard {
    dirs: BlazarDirs,
    model: String,
    engine_tag: String,
    saved: Option<ProfileRow>,
    armed: bool,
}

impl RestoreGuard {
    fn arm(store: &Store, dirs: &BlazarDirs, model: &str, engine_tag: &str) -> Result<Self> {
        Ok(Self {
            dirs: dirs.clone(),
            model: model.to_string(),
            engine_tag: engine_tag.to_string(),
            saved: store.get_profile(model, engine_tag)?,
            armed: true,
        })
    }

    /// Explicit restore on the happy path (returns errors to report).
    fn restore_now(mut self) -> Result<Option<ProfileRow>> {
        self.restore()?;
        self.armed = false;
        Ok(self.saved.clone())
    }

    fn restore(&self) -> Result<()> {
        let store = Store::open(&self.dirs)?;
        match &self.saved {
            Some(row) => store.upsert_profile(row)?,
            None => store.delete_profile(&self.model, &self.engine_tag)?,
        }
        Ok(())
    }
}

impl Drop for RestoreGuard {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        if let Err(e) = self.restore() {
            eprintln!(
                "WARNING: optimize could not restore the baseline profiles row \
                 for {} on {} - the lane may serve the CANDIDATE argv until \
                 corrected (re-run the tune path or restore manually): {e}",
                self.model, self.engine_tag
            );
        }
    }
}

// ---------------------------------------------------------------- HTTP seam

fn http_client() -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .build()
        .context("build probe client")
}

async fn get_json(client: &reqwest::Client, url: &str) -> Result<serde_json::Value> {
    let v: serde_json::Value = client
        .get(url)
        .timeout(Duration::from_secs(10))
        .send()
        .await
        .with_context(|| format!("GET {url}"))?
        .error_for_status()
        .with_context(|| format!("GET {url} status"))?
        .json()
        .await
        .with_context(|| format!("GET {url} body"))?;
    Ok(v)
}

/// Is the lane quiet enough to experiment on? Refuses when the model
/// (or any replica) has in-flight requests or the gateway queue is
/// non-empty; errors (daemon unreachable) also refuse, never assume.
pub async fn lane_quiet(client: &reqwest::Client, base: &str, model: &str) -> Result<bool> {
    let ps = get_json(client, &format!("{base}/api/ps")).await?;
    // The gateway wraps rows as {"models": [...], "remotes": [...]} -
    // remote peers hold no local GPU lease, so only "models" gates.
    // A body without the key refuses: never assume an unseen lane.
    let rows = ps["models"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("/api/ps returned no models array"))?;
    for row in rows {
        let name = row["name"].as_str().unwrap_or_default();
        // Replica rows are `model#N`; strip the suffix for the match.
        let stem = name.split('#').next().unwrap_or_default();
        if stem == model && row["blazar_in_flight"].as_i64().unwrap_or(0) > 0 {
            return Ok(false);
        }
    }
    let metrics = client
        .get(format!("{base}/metrics"))
        .timeout(Duration::from_secs(10))
        .send()
        .await
        .context("GET /metrics")?
        .error_for_status()
        .context("GET /metrics status")?
        .text()
        .await
        .context("GET /metrics body")?;
    for line in metrics.lines() {
        if let Some(rest) = line.strip_prefix("blazar_queue_depth ") {
            let value: f64 = rest
                .split_whitespace()
                .next()
                .unwrap_or("0")
                .parse()
                .context("parse blazar_queue_depth")?;
            if value > 0.0 {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

async fn evict_model(client: &reqwest::Client, base: &str, model: &str) -> Result<()> {
    client
        .post(format!("{base}/api/evict"))
        .json(&serde_json::json!({ "model": model }))
        .timeout(Duration::from_secs(30))
        .send()
        .await
        .context("POST /api/evict")?
        .error_for_status()
        .context("POST /api/evict status")?;
    Ok(())
}

async fn warm_model(client: &reqwest::Client, base: &str, model: &str) -> Result<()> {
    // "wait": true selects the synchronous warm (loads the model and
    // returns the resident row); without it the endpoint is a detached
    // policy-gated notify that "always ok"s without loading anything.
    client
        .post(format!("{base}/api/warm"))
        .json(&serde_json::json!({ "model": model, "wait": true }))
        .timeout(Duration::from_secs(120))
        .send()
        .await
        .context("POST /api/warm")?
        .error_for_status()
        .context("POST /api/warm status")?;
    Ok(())
}

/// Wait until the model's row is back up (`ready` or `sleeping` - a
/// sleeping child still holds the NEW argv and wakes cheap).
async fn wait_until_up(
    client: &reqwest::Client,
    base: &str,
    model: &str,
    deadline: Duration,
) -> Result<bool> {
    let start = Instant::now();
    loop {
        let up = get_json(client, &format!("{base}/api/ps"))
            .await
            .ok()
            .and_then(|ps| {
                ps["models"].as_array().map(|rows| {
                    rows.iter().any(|r| {
                        r["name"].as_str().unwrap_or_default().split('#').next() == Some(model)
                            && matches!(r["blazar_state"].as_str(), Some("ready" | "sleeping"))
                    })
                })
            })
            .unwrap_or(false);
        if up {
            return Ok(true);
        }
        if start.elapsed() >= deadline {
            return Ok(false);
        }
        tokio::time::sleep(READY_POLL).await;
    }
}

/// One streaming probe through the REAL gateway chat path. Returns the
/// arrival-timing sample; asserts nothing about latency magnitudes.
async fn probe_once(
    client: &reqwest::Client,
    base: &str,
    model: &str,
    num_predict: u32,
) -> Result<ProbeSample> {
    let body = serde_json::json!({
        "model": model,
        "messages": [ { "role": "user", "content": PROBE_PROMPT } ],
        "stream": true,
        "options": { "num_predict": num_predict, "temperature": 0.0 },
    });
    let sent = Instant::now();
    let resp = client
        .post(format!("{base}/api/chat"))
        .json(&body)
        .timeout(PROBE_TIMEOUT)
        .send()
        .await
        .context("POST /api/chat")?
        .error_for_status()
        .context("POST /api/chat status")?;
    let mut stream = resp.bytes_stream();
    let mut buf = String::new();
    let mut arrivals: Vec<Instant> = Vec::new();
    let mut ttft_ms: Option<f64> = None;
    let mut eval_count: u64 = 0;
    let mut done = false;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("chat stream chunk")?;
        buf.push_str(&String::from_utf8_lossy(&chunk));
        // Process every COMPLETE ndjson line in the buffer.
        while let Some(pos) = buf.find('\n') {
            let line: String = buf.drain(..=pos).collect();
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            arrivals.push(Instant::now());
            if ttft_ms.is_none() {
                ttft_ms = Some(arrivals[0].duration_since(sent).as_secs_f64() * 1000.0);
            }
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(line)
                && v["done"].as_bool().unwrap_or(false)
            {
                eval_count = v["eval_count"].as_u64().unwrap_or(0);
                done = true;
            }
        }
    }
    if !done {
        anyhow::bail!("chat stream ended without a done line (eval_count unknown)");
    }
    let wall_ms = sent.elapsed().as_secs_f64() * 1000.0;
    // ITL: deltas between consecutive token-line arrivals (the final
    // done line's arrival closes the last token's interval and is
    // included by construction - arrivals covers it).
    let mut itl_ms = Vec::new();
    for pair in arrivals.windows(2) {
        itl_ms.push(pair[1].duration_since(pair[0]).as_secs_f64() * 1000.0);
    }
    Ok(ProbeSample {
        ttft_ms: ttft_ms.unwrap_or(wall_ms),
        itl_ms,
        chunk_count: u64::try_from(arrivals.len().saturating_sub(1)).unwrap_or(0),
        eval_count,
        wall_ms,
    })
}

/// Measure one arm: per rep, `concurrency` parallel probes; the first
/// post-respawn probe is a warm-up and EXCLUDED (it pays model load).
// eval counters are u64 from the engine; the f64 blend here only feeds
// the throughput digest, so precision loss is accepted by design.
#[allow(clippy::cast_precision_loss)]
async fn measure_arm(
    client: &reqwest::Client,
    base: &str,
    model: &str,
    cfg: &OptimizeRunConfig,
) -> Result<ArmStats> {
    // Warm-up: one discarded request absorbs the respawn cost.
    probe_once(client, base, model, cfg.num_predict).await?;
    let mut samples: Vec<ProbeSample> = Vec::new();
    let mut per_rep: Vec<RepDigest> = Vec::new();
    for _ in 0..cfg.reps {
        let c = cfg.concurrency.max(1);
        let mut round = Vec::with_capacity(usize::try_from(c).unwrap_or(1));
        for _ in 0..c {
            round.push(probe_once(client, base, model, cfg.num_predict));
        }
        let done = futures::future::join_all(round)
            .await
            .into_iter()
            .collect::<Result<Vec<_>>>()?;
        let mut rep_ttft: Vec<f64> = done.iter().map(|s| s.ttft_ms).collect();
        rep_ttft.sort_by(f64::total_cmp);
        let mut rep_itl: Vec<f64> = done.iter().flat_map(|s| s.itl_ms.iter().copied()).collect();
        rep_itl.sort_by(f64::total_cmp);
        let rep_eval: u64 = done.iter().map(|s| s.eval_count).sum();
        let rep_wall: f64 = done.iter().map(|s| s.wall_ms).sum::<f64>();
        per_rep.push(RepDigest {
            tps: if rep_wall > 0.0 {
                rep_eval as f64 / (rep_wall / 1000.0 / f64::from(c))
            } else {
                0.0
            },
            ttft_p95: percentile(&rep_ttft, 95.0),
            itl_p95: percentile(&rep_itl, 95.0),
        });
        samples.extend(done);
    }
    Ok(ArmStats::from_samples(&samples, &per_rep))
}

// ------------------------------------------------------------- run report

/// Everything one `--run` produced - rendered by the CLI, serialized
/// into the ledger's final measurement row.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizeRunReport {
    pub model: String,
    pub engine_tag: String,
    pub hw_fingerprint: Option<String>,
    pub objective: String,
    pub experiment_id: i64,
    pub screen_score_tg: f64,
    pub baseline: ArmStats,
    pub candidate: ArmStats,
    pub verdict: Verdict,
    pub concurrency: u32,
    pub reps: u32,
}

// the optimize lifecycle reads top-to-bottom (lease -> screen ->
// A/B -> verify -> restore); splitting the chain would hide the
// ordering that the ledger events pin
#[allow(clippy::too_many_lines)]
/// Run the full Wave-2 flow: lease -> screen -> baseline A/B -> candidate
/// A/B -> restore -> verdict -> ledger. NEVER adopts.
pub async fn run_optimize(
    dirs: &BlazarDirs,
    store: &Store,
    tuner: &Tuner<'_>,
    input: &ProfileInput<'_>,
    cfg: &OptimizeRunConfig,
    objective: OptimizeObjective,
) -> Result<OptimizeRunReport> {
    let client = http_client()?;
    let base = cfg.base_url.trim_end_matches('/').to_string();

    // Lease: refuse to disturb a lane that is serving.
    if !lane_quiet(&client, &base, input.model_name).await? {
        anyhow::bail!(
            "lane for {} is serving traffic (in-flight requests or a queued \
             request) - re-run `optimize --run` when the model is idle",
            input.model_name
        );
    }

    // An idle lane can still hold the model resident: a sleeping child
    // keeps most of the VRAM while serving nothing, and the screen grid
    // needs that headroom for its own llama-bench child on small GPUs.
    // The lane is quiet, so evicting costs only warm-cache, which the
    // warm-up probe of each arm rebuilds anyway.
    evict_model(&client, &base, input.model_name).await?;

    // Screen the candidate (llama-bench grid, KV-gated argmax) WITHOUT
    // persisting: no profile row, no bench receipt, no adoption.
    let (cand_profile, _screen_overrides, grid_rows, screen_score) =
        tuner.tune_search_screen(input)?;
    let cand_hash = args_hash(input, &cand_profile);
    let cand_row = candidate_row(input, &cand_profile, &cand_hash);

    // Ledger identity first: an attempted experiment exists even if
    // everything after it fails.
    let (hw, quant) = experiment_provenance(store, input);
    let experiment_id = store.insert_experiment(
        input.model_name,
        input.engine_tag,
        hw.as_deref(),
        quant.as_deref(),
        &cand_row.args_json,
        &cand_hash,
        objective.label(),
    )?;
    store.append_event(
        experiment_id,
        "attempted",
        "optimize --run: grid screen then gateway A/B; adoption only via --apply",
    )?;
    store.append_measurement(
        experiment_id,
        "llama-bench-grid",
        &serde_json::to_string(&crate::bench::BenchmarkPayload {
            score: screen_score,
            rows: grid_rows.clone(),
        })?,
    )?;
    store.append_event(
        experiment_id,
        "measured",
        &format!(
            "grid screen: {n} rows, mean_tg {screen_score:.1} tok/s",
            n = grid_rows.len()
        ),
    )?;

    // Baseline arm: the daemon already serves whatever the profiles
    // table holds - measure it as-is.
    let guard = RestoreGuard::arm(store, dirs, input.model_name, input.engine_tag)?;
    warm_model(&client, &base, input.model_name).await?;
    if !wait_until_up(&client, &base, input.model_name, READY_DEADLINE).await? {
        anyhow::bail!("model did not come back up after warm (baseline arm)");
    }
    let baseline = measure_arm(&client, &base, input.model_name, cfg).await?;
    store.append_measurement(
        experiment_id,
        "gateway-ab-baseline",
        &serde_json::to_string(&baseline)?,
    )?;
    store.append_event(
        experiment_id,
        "baseline_measured",
        &format!(
            "baseline: p95 ttft {:.0} ms, p95 itl {:.1} ms, agg {:.1} tok/s \
             over {} requests",
            baseline.ttft_p95_ms, baseline.itl_p95_ms, baseline.aggregate_tps, baseline.requests
        ),
    )?;

    // Candidate arm: swap the row, evict, warm on the new argv.
    store.upsert_profile(&cand_row)?;
    evict_model(&client, &base, input.model_name).await?;
    warm_model(&client, &base, input.model_name).await?;
    if !wait_until_up(&client, &base, input.model_name, READY_DEADLINE).await? {
        anyhow::bail!("model did not come back up after candidate swap");
    }
    let candidate = measure_arm(&client, &base, input.model_name, cfg).await?;
    store.append_measurement(
        experiment_id,
        "gateway-ab-candidate",
        &serde_json::to_string(&candidate)?,
    )?;
    store.append_event(
        experiment_id,
        "candidate_measured",
        &format!(
            "candidate: p95 ttft {:.0} ms, p95 itl {:.1} ms, agg {:.1} tok/s \
             over {} requests",
            candidate.ttft_p95_ms,
            candidate.itl_p95_ms,
            candidate.aggregate_tps,
            candidate.requests
        ),
    )?;

    // Restore the baseline row byte-identically, then make the daemon
    // serve it again (post-run state == pre-run state).
    guard.restore_now()?;
    evict_model(&client, &base, input.model_name).await?;
    warm_model(&client, &base, input.model_name).await?;

    let verdict = verify(objective, &baseline, &candidate, cfg.min_gain_pct);
    store.append_event(
        experiment_id,
        if verdict.recommended {
            "verified"
        } else {
            "rejected"
        },
        &format!(
            "{}: {} - adopt via `blazar optimize {} --apply {experiment_id}` \
             only if recommended",
            verdict.objective, verdict.reason, input.model_name
        ),
    )?;
    Ok(OptimizeRunReport {
        model: input.model_name.to_string(),
        engine_tag: input.engine_tag.to_string(),
        hw_fingerprint: hw,
        objective: objective.label().to_string(),
        experiment_id,
        screen_score_tg: screen_score,
        baseline,
        candidate,
        verdict,
        concurrency: cfg.concurrency,
        reps: cfg.reps,
    })
}

fn candidate_row(input: &ProfileInput<'_>, profile: &Profile, hash: &str) -> ProfileRow {
    ProfileRow {
        model_name: input.model_name.to_string(),
        engine_tag: input.engine_tag.to_string(),
        args_hash: hash.to_string(),
        args_json: serde_json::to_string(&profile.argv).unwrap_or_else(|_| "[]".to_string()),
        // Honest: the grid payload is NOT this row's provenance - the
        // A/B comparison is. Adoption (`--apply`) re-persists the full
        // chain if it wins.
        benchmark_json: None,
        updated_at: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| i64::try_from(d.as_secs()).unwrap_or(i64::MAX)),
    }
}

fn experiment_provenance(
    store: &Store,
    input: &ProfileInput<'_>,
) -> (Option<String>, Option<String>) {
    let hw =
        store.active_engine().ok().flatten().and_then(|e| {
            blazar_core::store::EvidenceAxes::hw_fingerprint_from_manifest(&e.manifest)
        });
    let quant = store
        .get_model(input.model_name)
        .ok()
        .flatten()
        .map(|m| m.quant);
    (hw, quant)
}

// ------------------------------------------------------------- adoption

/// Outcome of a successful `--apply`.
#[derive(Debug, Clone, Serialize)]
pub struct AdoptOutcome {
    pub experiment_id: i64,
    pub model: String,
    pub engine_tag: String,
}

/// Wave-3 adoption: promote a VERIFIED experiment to the active
/// profile. Refuses (with teaching) when the id is unknown, unverified,
/// or measured on a different lane/hardware. Write order is
/// profile-row -> adopted event -> pointer LAST, so a pointer's presence
/// implies the full trail (crash between steps leaves at most a
/// re-runnable adoption, never a pointer into nothing).
pub fn adopt_experiment(store: &Store, experiment_id: i64) -> Result<AdoptOutcome> {
    let exp = store
        .get_experiment(experiment_id)?
        .ok_or_else(|| anyhow::anyhow!("no optimization experiment #{experiment_id}"))?;
    let events = store.list_experiment_events(experiment_id)?;
    let verified = events
        .iter()
        .any(|(_, kind, _)| kind == "verified" || kind == "adopted");
    if !verified {
        anyhow::bail!(
            "experiment #{experiment_id} for {} has no verified event - refusing \
             to adopt; re-run `blazar optimize {} --run`",
            exp.model,
            exp.model
        );
    }
    let active = store
        .active_engine()?
        .ok_or_else(|| anyhow::anyhow!("no active engine; adoption needs a lane"))?;
    if active.tag != exp.engine_tag {
        anyhow::bail!(
            "experiment #{experiment_id} was measured on engine '{}', but '{}' is \
             active - switching engines invalidates its evidence; re-run --run on \
             the active lane",
            exp.engine_tag,
            active.tag
        );
    }
    if let (Some(exp_hw), Some(now_hw)) = (
        exp.hw_fingerprint.as_deref(),
        blazar_core::store::EvidenceAxes::hw_fingerprint_from_manifest(&active.manifest).as_deref(),
    ) && exp_hw != now_hw
    {
        anyhow::bail!(
            "experiment #{experiment_id} was measured on different hardware \
                 ({exp_hw}) than the active lane ({now_hw}) - receipts do not \
                 transfer between machines",
        );
    }
    let row = ProfileRow {
        model_name: exp.model.clone(),
        engine_tag: exp.engine_tag.clone(),
        args_hash: exp.profile_hash.clone(),
        args_json: exp.config_json.clone(),
        benchmark_json: None,
        updated_at: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| i64::try_from(d.as_secs()).unwrap_or(i64::MAX)),
    };
    store.upsert_profile(&row)?;
    store.append_event(
        experiment_id,
        "adopted",
        "--apply: verified experiment promoted to the active profile",
    )?;
    store.set_active_profile(&exp.model, &exp.engine_tag, experiment_id)?;
    Ok(AdoptOutcome {
        experiment_id,
        model: exp.model,
        engine_tag: exp.engine_tag,
    })
}

// ------------------------------------------------------------------ tests

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)] // test names narrate: unit__subject__contract
    use super::*;

    fn sample(ttft: f64, itl: &[f64], eval: u64, wall: f64) -> ProbeSample {
        ProbeSample {
            ttft_ms: ttft,
            itl_ms: itl.to_vec(),
            chunk_count: u64::try_from(itl.len()).unwrap_or(0),
            eval_count: eval,
            wall_ms: wall,
        }
    }

    // ------------------------------------------------------------------
    // adoption gate (Wave 3)
    // ------------------------------------------------------------------

    fn adopt_store(manifest: &str) -> (blazar_core::store::Store, tempfile::TempDir) {
        let tmp = tempfile::tempdir().unwrap();
        let dirs = blazar_core::BlazarDirs {
            config_dir: tmp.path().join("cfg"),
            data_dir: tmp.path().join("data"),
        };
        let store = blazar_core::store::Store::open(&dirs).unwrap();
        store
            .upsert_engine(&blazar_core::store::EngineRow {
                tag: "b1".into(),
                asset: "a".into(),
                sha256: String::new(),
                installed_at: 1_000_000,
                active: true,
                manifest: manifest.into(),
                kind: blazar_core::engine_kind::EngineKind::default(),
            })
            .unwrap();
        (store, tmp)
    }

    fn hw_manifest() -> String {
        r#"{"devices":[{"name":"TestGpu","total_mib":8188}]}"#.into()
    }

    fn manifest_fp() -> String {
        blazar_core::store::EvidenceAxes::hw_fingerprint_from_manifest(&hw_manifest())
            .expect("manifest carries a device")
    }

    fn seed_experiment(
        store: &blazar_core::store::Store,
        engine_tag: &str,
        hw: Option<&str>,
    ) -> i64 {
        store
            .insert_experiment(
                "m",
                engine_tag,
                hw,
                Some("Q4_K_M"),
                "[]",
                "hash-1",
                "Throughput",
            )
            .unwrap()
    }

    #[test]
    fn unit__adopt_experiment__unknown_id_teaches() {
        let (store, _tmp) = adopt_store(&hw_manifest());
        let err = adopt_experiment(&store, 999).unwrap_err().to_string();
        assert!(err.contains("no optimization experiment #999"), "{err}");
    }

    #[test]
    fn unit__adopt_experiment__refuses_unverified_then_adopts_verified() {
        let (store, _tmp) = adopt_store(&hw_manifest());
        let id = seed_experiment(&store, "b1", Some(&manifest_fp()));

        let err = adopt_experiment(&store, id).unwrap_err().to_string();
        assert!(
            err.contains("no verified event") && err.contains("--run"),
            "teaching refusal, got: {err}"
        );

        store
            .append_event(id, "verified", "gain beyond noise")
            .unwrap();
        let out = adopt_experiment(&store, id).unwrap();
        assert_eq!(
            (
                out.experiment_id,
                out.model.as_str(),
                out.engine_tag.as_str()
            ),
            (id, "m", "b1")
        );

        // Pointer-last invariant: the row exists AND the pointer is set.
        let row = store.get_profile("m", "b1").unwrap().expect("adopted row");
        assert_eq!(row.args_json, "[]");
        assert_eq!(
            row.benchmark_json, None,
            "adoption carries no bench payload - the ledger holds the evidence"
        );
        assert!(
            store.active_profile_of("m", "b1").unwrap().is_some(),
            "pointer written last"
        );
        let kinds: Vec<String> = store
            .list_experiment_events(id)
            .unwrap()
            .into_iter()
            .map(|(_, k, _)| k)
            .collect();
        assert!(kinds.iter().any(|k| k == "adopted"));

        // Re-adopting is safe: an adopted experiment is trivially verified.
        adopt_experiment(&store, id).unwrap();
    }

    #[test]
    fn unit__adopt_experiment__refuses_lane_and_hw_mismatch() {
        let (store, _tmp) = adopt_store(&hw_manifest());

        // Same hardware, different engine lane: evidence does not cross.
        let lane_id = seed_experiment(&store, "other-lane", Some(&manifest_fp()));
        store.append_event(lane_id, "verified", "x").unwrap();
        let err = adopt_experiment(&store, lane_id).unwrap_err().to_string();
        assert!(
            err.contains("measured on engine 'other-lane'"),
            "lane refusal, got: {err}"
        );

        // Same lane, different hardware: receipts do not transfer.
        let hw_id = seed_experiment(&store, "b1", Some("gpu=Elsewhere|vram_mib=1"));
        store.append_event(hw_id, "verified", "x").unwrap();
        let err = adopt_experiment(&store, hw_id).unwrap_err().to_string();
        assert!(err.contains("different hardware"), "hw refusal, got: {err}");

        // Nothing was written by the refusals.
        assert!(store.get_profile("m", "b1").unwrap().is_none());
        assert!(store.active_profile_of("m", "b1").unwrap().is_none());
    }

    fn stats(tps_series: &[f64], agg: f64) -> ArmStats {
        ArmStats {
            ttft_p50_ms: 0.0,
            ttft_p95_ms: 0.0,
            itl_p50_ms: 0.0,
            itl_p95_ms: 0.0,
            aggregate_tps: agg,
            per_rep_tps: tps_series.to_vec(),
            per_rep_ttft_p95: vec![],
            per_rep_itl_p95: vec![],
            requests: tps_series.len().try_into().unwrap_or(u32::MAX),
            eval_tokens: 0,
        }
    }

    #[test]
    fn unit__percentile__nearest_rank_on_sorted_slice() {
        let v: Vec<f64> = (1..=100).map(f64::from).collect();
        assert!((percentile(&v, 50.0) - 50.0).abs() < 1e-9);
        assert!((percentile(&v, 95.0) - 95.0).abs() < 1e-9);
        assert!((percentile(&v, 100.0) - 100.0).abs() < 1e-9);
        assert!((percentile(&[], 95.0) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn unit__verify__throughput_requires_gain_beyond_noise_and_floor() {
        // Baseline 100 tok/s with +/-1 noise; candidate 110 (+10%).
        let base = stats(&[99.0, 100.0, 101.0], 100.0);
        let cand = stats(&[109.0, 110.0, 111.0], 110.0);
        let v = verify(OptimizeObjective::Throughput, &base, &cand, 5.0);
        assert!(v.recommended, "{v:?}");
        // +3% with 5% floor: rejected even though noise is tiny.
        let cand_small = stats(&[102.9, 103.0, 103.1], 103.0);
        let v = verify(OptimizeObjective::Throughput, &base, &cand_small, 5.0);
        assert!(!v.recommended, "{v:?}");
    }

    #[test]
    fn unit__verify__noise_can_swallow_a_large_raw_gain() {
        // Identical configs on a noisy lane: spread 20 -> a +15 gain is
        // inside observed noise and must NOT recommend.
        let base = stats(&[90.0, 100.0, 110.0], 100.0);
        let cand = stats(&[95.0, 105.0, 120.0], 115.0);
        let v = verify(OptimizeObjective::Throughput, &base, &cand, 5.0);
        assert!(!v.recommended, "{v:?}");
    }

    #[test]
    fn unit__verify__interactive_gates_ttft_then_tolerates_itl() {
        let mut base = stats(&[], 0.0);
        base.ttft_p95_ms = 1000.0;
        base.itl_p95_ms = 20.0;
        base.per_rep_ttft_p95 = vec![990.0, 1000.0, 1010.0];
        base.per_rep_itl_p95 = vec![19.0, 20.0, 21.0];
        let mut cand = stats(&[], 0.0);
        cand.ttft_p95_ms = 800.0; // -200 ms, far beyond 5% (50 ms) + noise (20)
        cand.itl_p95_ms = 21.0; // +1 ms regression, inside the 1 ms + noise gate? 5% of 20 = 1 -> tolerated up to 1
        cand.per_rep_ttft_p95 = vec![790.0, 800.0, 810.0];
        cand.per_rep_itl_p95 = vec![20.0, 21.0, 22.0];
        let v = verify(OptimizeObjective::Interactive, &base, &cand, 5.0);
        assert!(v.recommended, "{v:?}");

        // Same TTFT win but a big ITL regression: rejected.
        let mut bad = cand.clone();
        bad.itl_p95_ms = 40.0;
        let v = verify(OptimizeObjective::Interactive, &base, &bad, 5.0);
        assert!(!v.recommended, "{v:?}");
    }

    #[test]
    fn unit__verify__memory_objective_stays_ineligible() {
        let v = verify(
            OptimizeObjective::Memory,
            &stats(&[], 0.0),
            &stats(&[], 0.0),
            5.0,
        );
        assert!(!v.recommended);
        assert!(v.reason.contains("ineligible"));
    }

    #[test]
    fn unit__arm_stats__aggregate_is_fair_across_arms() {
        // 4 requests, 100 tokens each, 1 s wall each, C=4 -> 400 tok / 1 s.
        let s = vec![sample(100.0, &[10.0; 99], 100, 1000.0); 4];
        let st = ArmStats::from_samples(
            &s,
            &[RepDigest {
                tps: 400.0,
                ttft_p95: 100.0,
                itl_p95: 10.0,
            }],
        );
        assert!(
            (st.aggregate_tps - 400.0).abs() < 1e-6,
            "{}",
            st.aggregate_tps
        );
        assert_eq!(st.eval_tokens, 400);
    }

    #[test]
    fn unit__objective__parse_accepts_known_and_rejects_unknown() {
        assert_eq!(
            OptimizeObjective::parse("interactive").unwrap(),
            OptimizeObjective::Interactive
        );
        assert_eq!(
            OptimizeObjective::parse("THROUGHPUT").unwrap(),
            OptimizeObjective::Throughput
        );
        assert!(OptimizeObjective::parse("quality").is_err());
    }
}
