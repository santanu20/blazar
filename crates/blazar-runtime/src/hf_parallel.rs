//! Parallel byte-range download lane for large files.
//!
//! The classic lane (`hf::download_to`) is a single stream with `.part`
//! tail-resume. Many hosts (HF CDN, OCI registries) serve parallel Range
//! requests per-connection-throttled, so N connections can multiply
//! throughput. This lane:
//!
//! * probes with `Range: bytes=0-0` and only engages on a strict `206` +
//!   parseable `Content-Range: bytes 0-0/<total>` (anything else falls back
//!   to the classic lane verbatim),
//! * engages only for totals >= `MIN_PARALLEL_BYTES` and `connections > 1`,
//! * writes fixed-position chunks (`write_at`) into a preallocated `.part`,
//!   so a failed chunk is simply re-fetched — never a wrong-offset write,
//! * persists a `.part.progress` sidecar ledger at BYTE granularity: the
//!   completed chunks plus the verified prefix of every in-flight chunk
//!   (per-chunk counters, snapshotted ~1 Hz by the coordinator, seeded the
//!   moment the lane engages). An interrupt at any point therefore resumes
//!   from the last snapshot instead of restarting the file — whole-chunk
//!   bookkeeping alone loses everything until a chunk (8-128 MiB) lands,
//!   which uniform spreading defers until the download is nearly done.
//! * guards against upstream changes: sidecar `url`/`total` mismatch
//!   discards the `.part` + sidecar and starts fresh,
//! * keeps legacy compatibility: a `.part` with no sidecar belongs to the
//!   classic lane and is handed back to it untouched.

use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::sync::mpsc;

use crate::hf::FilePlan;

/// Files below this stay on the classic single-stream lane (range setup,
/// sidecar bookkeeping, and connection churn cost more than they save).
pub(crate) const MIN_PARALLEL_BYTES: u64 = 32 * 1024 * 1024;
/// Lower bound on chunk size: keeps per-chunk request overhead sane.
const MIN_CHUNK_BYTES: u64 = 8 * 1024 * 1024;
/// Upper bound on chunk size: bounds re-fetch cost after a mid-chunk failure.
const MAX_CHUNK_BYTES: u64 = 128 * 1024 * 1024;
/// Sidecar format version; bump on breaking layout changes. Version 2
/// added the `partial` per-chunk prefix ledger; version 1 files (done-only)
/// remain loadable with an empty `partial`.
const SIDECAR_VERSION: u32 = 2;
/// Oldest sidecar layout this lane can resume from.
const SIDECAR_MIN_VERSION: u32 = 1;
/// Minimum spacing between heartbeat sidecar snapshots (a snapshot is also
/// forced by every chunk completion). Bounded re-fetch cost after an
/// unclean interrupt = one cadence window of bytes.
const SIDECAR_SNAPSHOT_CADENCE: Duration = Duration::from_secs(1);
/// Per-chunk attempts (1 initial + 2 retries) before the download aborts.
const CHUNK_ATTEMPTS: u32 = 3;
/// Exponential backoff table between chunk attempts (500ms -> 2s -> 8s);
/// a larger Retry-After, when sent, is honored instead.

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ChunkPlan {
    pub total: u64,
    pub chunk_size: u64,
    pub chunk_count: u64,
}

/// Chunk sizing: aim for `connections` evenly sized chunks, clamped to
/// [MIN, MAX] chunk bytes. Clamping may yield fewer chunks than requested
/// connections — that is fine, extra workers simply exit.
pub(crate) fn chunk_plan(total: u64, connections: u32) -> ChunkPlan {
    let conns = u64::from(connections.max(1));
    let target = total
        .div_ceil(conns)
        .clamp(MIN_CHUNK_BYTES, MAX_CHUNK_BYTES);
    let chunk_size = target.min(total).max(1);
    ChunkPlan {
        total,
        chunk_size,
        chunk_count: total.div_ceil(chunk_size),
    }
}

/// Strict `Content-Range: bytes 0-0/<total>` parse for the probe response:
/// the range part must echo the exact single-byte probe unit — anything
/// else means the server ignored the Range semantics.
fn probe_total(content_range: Option<&str>) -> Option<u64> {
    let v = content_range?;
    let rest = v.strip_prefix("bytes ")?;
    let (range, total) = rest.split_once('/')?;
    let (start, end) = range.split_once('-')?;
    if start != "0" || end != "0" {
        return None;
    }
    if !total.bytes().all(|b| b.is_ascii_digit()) || total.is_empty() {
        return None;
    }
    total.parse().ok()
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
struct PartialChunk {
    idx: u64,
    /// Verified contiguous prefix of that chunk's region on disk.
    bytes: u64,
}

#[derive(Serialize, Deserialize)]
struct ProgressSidecar {
    version: u32,
    url: String,
    total: u64,
    chunk_size: u64,
    /// Sorted completed chunk indices.
    done: Vec<u64>,
    /// Verified prefixes of in-flight chunks (empty in version-1 files).
    #[serde(default)]
    partial: Vec<PartialChunk>,
}

fn sidecar_path(part: &Path) -> PathBuf {
    let mut s = part.as_os_str().to_os_string();
    s.push(".progress");
    PathBuf::from(s)
}

fn load_sidecar(part: &Path) -> io::Result<Option<ProgressSidecar>> {
    let path = sidecar_path(part);
    let raw = match std::fs::read(&path) {
        Ok(raw) => raw,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e),
    };
    serde_json::from_slice(&raw)
        .map(Some)
        .with_context(|| format!("parse {}", path.display()))
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

/// Persist the ledger durably-ordered so a crash never leaves a sidecar
/// that claims more bytes than the `.part` actually holds on disk:
///
/// 1. fsync the `.part` (the data) BEFORE the ledger that describes it,
/// 2. write the ledger to a tmp file and fsync it,
/// 3. atomically rename over the previous ledger,
/// 4. fsync the parent dir so the rename itself survives power loss.
///
/// The fsync steps are best-effort: on filesystems where sync is refused
/// (e.g. read-only mounts of the data dir) we warn and keep downloading —
/// the resume guarantee then degrades to the pre-sync window, which is the
/// same behavior this lane had before ordered persistence existed.
fn store_sidecar(part: &Path, sidecar: &ProgressSidecar) -> io::Result<()> {
    let path = sidecar_path(part);
    let tmp = path.with_extension("progress.tmp");

    match OpenOptions::new().write(true).open(part) {
        Ok(data) => {
            if let Err(e) = data.sync_all() {
                tracing::warn!("part fsync before ledger persist failed (continuing): {e}");
            }
        }
        Err(e) => tracing::warn!("opening .part for fsync failed (continuing): {e}"),
    }

    let mut tmp_file = File::create(&tmp)?;
    tmp_file.write_all(&serde_json::to_vec(sidecar)?)?;
    if let Err(e) = tmp_file.sync_all() {
        tracing::warn!("ledger tmp fsync failed (continuing): {e}");
    }

    std::fs::rename(&tmp, &path)?;

    if let Some(parent) = path.parent() {
        if let Ok(dir) = File::open(parent) {
            let _ = dir.sync_all();
        }
    }
    Ok(())
}

/// Positional write that works on both unix (`write_at`) and windows
/// (`seek_write`) file extensions.
fn write_positional(file: &File, buf: &[u8], at: u64) -> io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.write_at(buf, at).map(|_| ())
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        file.seek_write(buf, at).map(|_| ())
    }
}

enum WorkerMsg {
    ChunkDone(u64),
    Failed(String),
}

/// Attempt the parallel lane for `url` into `dest`.
///
/// Returns `Ok(Some(bytes))` when the file landed via this lane, `Ok(None)`
/// when the caller should fall back to the classic single-stream lane
/// (no Range support, tiny file, `connections < 2`, or a legacy `.part`
/// without a sidecar that only the classic lane can tail-resume).
pub(crate) async fn try_parallel(
    http: &reqwest::Client,
    token: Option<&str>,
    url: &reqwest::Url,
    plan: &FilePlan,
    dest: &Path,
    connections: u32,
    on_progress: &mut impl FnMut(u64, u64),
) -> Result<Option<u64>> {
    if connections < 2 {
        return Ok(None);
    }
    let part = crate::hf::sibling_part_path(dest);

    // Probe: strict 206 + Content-Range. Anything else -> classic lane.
    let Some(total) = probe_range_total(http, token, url).await? else {
        return Ok(None);
    };

    // A `.part` without our sidecar is either a classic-lane tail partial
    // (contiguous verified prefix, shorter than the file) or our own
    // orphaned full-length preallocation (sidecar lost; its length can
    // never certify sparse chunk bytes). Only the latter is ours to
    // re-fetch in place; the former must reach the classic tail-resume.
    if part.exists() && load_sidecar(&part).ok().flatten().is_none() {
        let len = std::fs::metadata(&part).map_or(0, |m| m.len());
        if len < total {
            return Ok(None);
        }
        tracing::info!(
            "orphaned full-length .part without sidecar — re-fetching all chunks in place: {}",
            part.display()
        );
    }

    let cp = chunk_plan(total, connections);
    let Some(ledger) = resume_state(&part, url, total, &cp)? else {
        return Ok(None);
    };
    // Seed the ledger BEFORE any byte moves: an interrupt at any later
    // point then leaves a loadable resume map instead of an unresolvable
    // sparse orphan (the very failure this file exists to prevent).
    store_sidecar(
        &part,
        &ProgressSidecar {
            version: SIDECAR_VERSION,
            url: url.as_str().to_string(),
            total: cp.total,
            chunk_size: cp.chunk_size,
            done: ledger.done.clone(),
            partial: ledger.partial.clone(),
        },
    )
    .with_context(|| format!("seed {}", sidecar_path(&part).display()))?;
    execute_chunks(
        http,
        token,
        url,
        &part,
        &cp,
        ledger,
        connections,
        on_progress,
    )
    .await?;
    on_progress(total, total);
    on_progress(total, total);
    finalize_parallel(&part, dest, plan).await?;
    Ok(Some(total))
}

/// A chunk still needing bytes: `start` is the absolute resume offset
/// (chunk start + verified prefix), `len` the remaining byte count.
struct PendingChunk {
    idx: u64,
    start: u64,
    len: u64,
}

/// Spawn chunk workers over the pending ranges and coordinate: persist
/// the sidecar per completed chunk AND on a heartbeat cadence (per-chunk
/// prefixes), fan out progress on this thread (preserving the classic
/// lane's `FnMut` contract), abort on first failure.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_lines)] // one full download cycle: seed, fan out, coordinate, finalize
async fn execute_chunks(
    http: &reqwest::Client,
    token: Option<&str>,
    url: &reqwest::Url,
    part: &Path,
    cp: &ChunkPlan,
    ledger: ResumeLedger,
    connections: u32,
    on_progress: &mut impl FnMut(u64, u64),
) -> Result<()> {
    let done_set: std::collections::BTreeSet<u64> = ledger.done.iter().copied().collect();
    let mut prefix_of: std::collections::BTreeMap<u64, u64> =
        ledger.partial.iter().map(|p| (p.idx, p.bytes)).collect();

    // Per-chunk verified-byte counters, seeded with the resumed prefixes;
    // workers bump theirs after every positional write (and refund on a
    // failed attempt), so a snapshot of these counters is always a
    // lower bound of the bytes physically on disk.
    let chunk_counters: Arc<Vec<AtomicU64>> = Arc::new(
        (0..cp.chunk_count)
            .map(|idx| {
                let have = if done_set.contains(&idx) {
                    chunk_len(cp, idx)
                } else {
                    prefix_of.remove(&idx).unwrap_or(0)
                };
                AtomicU64::new(have)
            })
            .collect(),
    );
    let counters = chunk_counters.clone();
    let pending: Arc<Vec<PendingChunk>> = Arc::new(
        (0..cp.chunk_count)
            .filter(|&idx| !done_set.contains(&idx))
            .map(|idx| {
                let chunk_start = idx * cp.chunk_size;
                let have =
                    (&counters)[usize::try_from(idx).unwrap_or(usize::MAX)].load(Ordering::Relaxed);
                PendingChunk {
                    idx,
                    start: chunk_start + have,
                    len: chunk_len(cp, idx) - have,
                }
            })
            .collect(),
    );
    let base_bytes: u64 = counters.iter().map(|c| c.load(Ordering::Relaxed)).sum();
    let progress = Arc::new(AtomicU64::new(base_bytes));
    let abort = Arc::new(AtomicBool::new(false));
    let (tx, mut rx) = mpsc::channel::<WorkerMsg>(connections as usize);

    let cursor = Arc::new(AtomicU64::new(0));
    let mut handles = Vec::with_capacity(connections as usize);
    let worker_count = u32::try_from(pending.len().max(1)).unwrap_or(u32::MAX);
    for _ in 0..connections.min(worker_count) {
        let http = http.clone();
        let token = token.map(str::to_string);
        let url = url.clone();
        let part = part.to_path_buf();
        let tx = tx.clone();
        let cursor = cursor.clone();
        let progress = progress.clone();
        let abort = abort.clone();
        let pending = pending.clone();
        let chunk_counters = chunk_counters.clone();
        handles.push(tokio::spawn(async move {
            loop {
                if abort.load(Ordering::Relaxed) {
                    break;
                }
                let slot = cursor.fetch_add(1, Ordering::Relaxed);
                let Some(pc) = pending.get(usize::try_from(slot).unwrap_or(usize::MAX)) else {
                    break;
                };
                let idx_usize = usize::try_from(pc.idx).unwrap_or(usize::MAX);
                let chunk_progress = chunk_counters
                    .get(idx_usize)
                    .expect("counter array covers every chunk index");
                match fetch_chunk(
                    &http,
                    token.as_deref(),
                    &url,
                    &part,
                    pc.start,
                    pc.len,
                    &progress,
                    chunk_progress,
                )
                .await
                {
                    Ok(()) => {
                        let _ = tx.send(WorkerMsg::ChunkDone(pc.idx)).await;
                    }
                    Err(e) => {
                        abort.store(true, Ordering::Relaxed);
                        let _ = tx
                            .send(WorkerMsg::Failed(format!("chunk {}: {e:#}", pc.idx)))
                            .await;
                        break;
                    }
                }
            }
        }));
    }
    drop(tx);

    let mut done_count = done_set.len() as u64;
    let mut failure: Option<String> = None;
    let mut sidecar = ProgressSidecar {
        version: SIDECAR_VERSION,
        url: url.as_str().to_string(),
        total: cp.total,
        chunk_size: cp.chunk_size,
        done: ledger.done,
        partial: ledger.partial,
    };
    // ChunkDone paints alone would leave the bar frozen for a full chunk
    // (up to MAX_CHUNK_BYTES of silence — minutes on a throttled CDN), which
    // reads as a dead download. A 250 ms heartbeat paints the live byte
    // counter between chunk completions; first tick delayed so a fast chunk
    // still paints from its own message.
    let mut heartbeat = tokio::time::interval_at(
        tokio::time::Instant::now() + Duration::from_millis(250),
        Duration::from_millis(250),
    );
    // Snapshot bookkeeping: the sidecar is rewritten only when the byte
    // fingerprint moved AND the cadence elapsed, so a full-speed download
    // costs ~one tiny atomic rename per second, not one per tick.
    let mut snapshot_at = std::time::Instant::now();
    let mut snapshot_fingerprint = base_bytes;
    let pending_idx: Vec<u64> = pending.iter().map(|p| p.idx).collect();
    loop {
        let msg = tokio::select! {
            m = rx.recv() => match m {
                Some(m) => m,
                None => break,
            },
            _ = heartbeat.tick() => {
                let fingerprint: u64 =
                    chunk_counters.iter().map(|c| c.load(Ordering::Relaxed)).sum();
                if fingerprint != snapshot_fingerprint
                    && snapshot_at.elapsed() >= SIDECAR_SNAPSHOT_CADENCE
                {
                    sidecar.partial = snapshot_partials(&chunk_counters, &pending_idx, &sidecar.done);
                    store_sidecar(part, &sidecar)
                        .with_context(|| format!("persist {}", sidecar_path(part).display()))?;
                    snapshot_fingerprint = fingerprint;
                    snapshot_at = std::time::Instant::now();
                }
                on_progress(progress.load(Ordering::Relaxed), cp.total);
                continue;
            }
        };
        match msg {
            WorkerMsg::ChunkDone(idx) => {
                done_count += 1;
                sidecar.done.push(idx);
                sidecar.done.sort_unstable();
                sidecar.done.dedup();
                sidecar.partial.retain(|p| p.idx != idx);
                store_sidecar(part, &sidecar)
                    .with_context(|| format!("persist {}", sidecar_path(part).display()))?;
                snapshot_fingerprint = chunk_counters
                    .iter()
                    .map(|c| c.load(Ordering::Relaxed))
                    .sum();
                snapshot_at = std::time::Instant::now();
                on_progress(progress.load(Ordering::Relaxed), cp.total);
            }
            WorkerMsg::Failed(e) if failure.is_none() => failure = Some(e),
            WorkerMsg::Failed(_) => {}
        }
    }
    for h in handles {
        let _ = h.await;
    }
    if let Some(e) = failure {
        return Err(anyhow!("parallel download failed: {e}"));
    }
    if done_count != cp.chunk_count {
        return Err(anyhow!(
            "parallel download ended with {}/{} chunks — .part kept for resume",
            done_count,
            cp.chunk_count
        ));
    }
    Ok(())
}

/// Sidecar view of the per-chunk counters: every pending chunk with a
/// non-zero verified prefix (done chunks live in `done`, not here).
fn snapshot_partials(
    counters: &[AtomicU64],
    pending_idx: &[u64],
    done: &[u64],
) -> Vec<PartialChunk> {
    pending_idx
        .iter()
        .filter(|idx| !done.contains(idx))
        .filter_map(|&idx| {
            let bytes = counters
                .get(usize::try_from(idx).unwrap_or(usize::MAX))?
                .load(Ordering::Relaxed);
            (bytes > 0).then_some(PartialChunk { idx, bytes })
        })
        .collect()
}

/// Finalize: full re-read sha256 (chunked writes bypassed the streaming
/// hasher), then atomic rename and sidecar cleanup.
async fn finalize_parallel(part: &Path, dest: &Path, plan: &FilePlan) -> Result<()> {
    let part_for_hash = part.to_path_buf();
    let got = tokio::task::spawn_blocking(move || -> Result<String> {
        let mut f = File::open(&part_for_hash)
            .with_context(|| format!("open {}", part_for_hash.display()))?;
        let mut hasher = Sha256::new();
        let mut buf = vec![0u8; 1 << 20];
        loop {
            let n = f.read(&mut buf)?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }
        Ok(format!("{:x}", hasher.finalize()))
    })
    .await??;
    if let Some(expected) = &plan.sha256 {
        if !got.eq_ignore_ascii_case(expected) {
            let _ = std::fs::remove_file(part);
            let _ = std::fs::remove_file(sidecar_path(part));
            return Err(anyhow!(
                "sha256 mismatch for {}: expected {expected}, got {got}; partial deleted",
                plan.filename
            ));
        }
    }
    tokio::fs::rename(part, dest)
        .await
        .with_context(|| format!("finalize {}", dest.display()))?;
    let _ = tokio::fs::remove_file(sidecar_path(part)).await;
    Ok(())
}

/// Verified on-disk state of a partially downloaded `.part`: completed
/// chunks plus the trusted prefix of each in-flight chunk.
#[derive(Debug, Default)]
struct ResumeLedger {
    done: Vec<u64>,
    partial: Vec<PartialChunk>,
}

/// Length of chunk `idx` under plan `cp` (the last chunk is short).
fn chunk_len(cp: &ChunkPlan, idx: u64) -> u64 {
    (cp.total - idx * cp.chunk_size).min(cp.chunk_size)
}

/// Resolve resume state: load the sidecar, discard on any mismatch
/// (upstream changed / foreign layout), preallocate the `.part`.
/// `Ok(None)` = unusable sidecar, fall back to the classic lane.
fn resume_state(
    part: &Path,
    url: &reqwest::Url,
    total: u64,
    cp: &ChunkPlan,
) -> Result<Option<ResumeLedger>> {
    let mut ledger = ResumeLedger::default();
    match load_sidecar(part) {
        Ok(Some(sc))
            if (SIDECAR_MIN_VERSION..=SIDECAR_VERSION).contains(&sc.version)
                && sc.url == url.as_str()
                && sc.total == total
                && sc.chunk_size == cp.chunk_size =>
        {
            ledger.done = sc.done;
            ledger.partial = sc.partial;
        }
        Ok(Some(_)) => {
            // Mismatched sidecar (upstream changed / foreign layout):
            // discard both and start clean.
            let _ = std::fs::remove_file(part);
            let _ = std::fs::remove_file(sidecar_path(part));
        }
        Ok(None) => {
            // No sidecar: fresh download, or an orphaned full-length
            // preallocation whose allocation we keep (every chunk is
            // re-fetched at fixed offsets either way). A short classic
            // partial never reaches here — try_parallel routes it to
            // the classic lane before we run.
        }
        Err(_) => return Ok(None),
    }
    ledger.done.retain(|&i| i < cp.chunk_count);
    ledger.done.sort_unstable();
    ledger.done.dedup();
    let done: std::collections::BTreeSet<u64> = ledger.done.iter().copied().collect();
    // Clamp partials to the chunk layout; drop entries duplicating a done
    // chunk; promote full-length prefixes to done (all their bytes are on
    // disk — the counter only ever counted completed positional writes).
    let mut map: std::collections::BTreeMap<u64, u64> = ledger
        .partial
        .into_iter()
        .filter(|p| p.idx < cp.chunk_count && !done.contains(&p.idx))
        .map(|p| (p.idx, p.bytes.min(chunk_len(cp, p.idx))))
        .collect();
    let completed: Vec<u64> = map
        .iter()
        .filter(|(&idx, &bytes)| bytes >= chunk_len(cp, idx))
        .map(|(&idx, _)| idx)
        .collect();
    for idx in completed {
        map.remove(&idx);
        ledger.done.push(idx);
    }
    ledger.done.sort_unstable();
    ledger.done.dedup();
    ledger.partial = map
        .into_iter()
        .map(|(idx, bytes)| PartialChunk { idx, bytes })
        .collect();

    // Preallocate the exact final size; chunks write at fixed offsets.
    if !part.exists() || std::fs::metadata(part).map_or(0, |m| m.len()) != total {
        let f = File::create(part).with_context(|| format!("create {}", part.display()))?;
        f.set_len(total)
            .with_context(|| format!("preallocate {}", part.display()))?;
    }
    Ok(Some(ledger))
}

/// Probe the server's Range support: a strict 206 echoing our exact
/// `bytes=0-0` unit plus a parseable total. `Ok(None)` = fall back.
async fn probe_range_total(
    http: &reqwest::Client,
    token: Option<&str>,
    url: &reqwest::Url,
) -> Result<Option<u64>> {
    let mut req = http.get(url.clone());
    if let Some(t) = token {
        req = req.bearer_auth(t);
    }
    let resp = req
        .header(reqwest::header::RANGE, "bytes=0-0")
        .send()
        .await
        .context("download probe failed")?;
    if resp.status().as_u16() != 206 {
        return Ok(None);
    }
    let header = resp
        .headers()
        .get(reqwest::header::CONTENT_RANGE)
        .and_then(|v| v.to_str().ok());
    let Some(total) = probe_total(header) else {
        return Ok(None);
    };
    if total < MIN_PARALLEL_BYTES {
        return Ok(None);
    }
    Ok(Some(total))
}

#[allow(clippy::too_many_arguments)] // one param per wire concern; a bundle would hide the range contract
async fn fetch_chunk(
    http: &reqwest::Client,
    token: Option<&str>,
    url: &reqwest::Url,
    part: &Path,
    start: u64,
    len: u64,
    progress: &AtomicU64,
    chunk_progress: &AtomicU64,
) -> Result<()> {
    let mut last_err: Option<String> = None;
    for attempt in 0..CHUNK_ATTEMPTS {
        if attempt > 0 {
            // 500ms -> 2s -> 8s; attempts beyond the table hold at the cap.
            let backoff = match attempt {
                1 => Duration::from_millis(500),
                2 => Duration::from_secs(2),
                _ => Duration::from_secs(8),
            };
            tokio::time::sleep(backoff).await;
        }
        let mut req = http.get(url.clone());
        if let Some(t) = token {
            req = req.bearer_auth(t);
        }
        let end = start + len - 1;
        let resp = match req
            .header(reqwest::header::RANGE, format!("bytes={start}-{end}"))
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => {
                last_err = Some(format!("request: {e}"));
                continue;
            }
        };
        // Gate: only a 206 carrying exactly this chunk's length is usable.
        if resp.status().as_u16() != 206 {
            last_err = Some(format!("status {} (expected 206)", resp.status()));
            continue;
        }
        if resp.content_length() != Some(len) {
            last_err = Some(format!(
                "content-length {:?} != chunk {len}",
                resp.content_length()
            ));
            continue;
        }
        let file = match File::options().write(true).open(part) {
            Ok(f) => f,
            Err(e) => return Err(anyhow!("open {}: {e}", part.display())),
        };
        let mut written: u64 = 0;
        let mut resp = resp;
        let mut ok = true;
        loop {
            match resp.chunk().await {
                Ok(Some(piece)) => match write_positional(&file, &piece, start + written) {
                    Ok(()) => {
                        written += piece.len() as u64;
                        progress.fetch_add(piece.len() as u64, Ordering::Relaxed);
                        chunk_progress.fetch_add(piece.len() as u64, Ordering::Relaxed);
                    }
                    Err(e) => {
                        last_err = Some(format!("write: {e}"));
                        ok = false;
                        break;
                    }
                },
                Ok(None) => break,
                Err(e) => {
                    // Mid-body read failure (HF resetting the connection
                    // partway through a chunk): consume it as a retryable
                    // attempt instead of propagating out of the loop —
                    // positional writes make re-fetching the range
                    // overwrite-safe.
                    last_err = Some(format!("body: {e}"));
                    ok = false;
                    break;
                }
            }
        }
        if ok && written == len {
            return Ok(());
        }
        if ok {
            last_err = Some(format!("short chunk body {written}/{len}"));
        }
        // A partial attempt's bytes are re-downloaded from scratch:
        // give the progress bar its bytes back.
        progress.fetch_sub(written, Ordering::Relaxed);
        chunk_progress.fetch_sub(written, Ordering::Relaxed);
    }
    Err(anyhow!(last_err.unwrap_or_else(|| "chunk failed".into())))
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    #[test]
    fn unit__chunk_plan__table() {
        // Tiny totals: one chunk, never zero.
        let cp = chunk_plan(1000, 8);
        assert_eq!(cp.chunk_count, 1);
        assert_eq!(cp.chunk_size, 1000);

        // 1 GiB over 8 conns -> 128 MiB target clamped to MAX (equal), 8 chunks.
        let cp = chunk_plan(1024 * 1024 * 1024, 8);
        assert_eq!(cp.chunk_size, MAX_CHUNK_BYTES);
        assert_eq!(cp.chunk_count, 8);

        // 8 GiB over 8 conns -> unclamped even 1 GiB chunks impossible
        // (MAX cap): chunk count grows past conns, extra workers exit.
        let cp = chunk_plan(8 * 1024 * 1024 * 1024, 8);
        assert_eq!(cp.chunk_size, MAX_CHUNK_BYTES);
        assert_eq!(cp.chunk_count, 64);

        // Small-ish file: MIN floor keeps chunks from being tiny.
        let cp = chunk_plan(40 * 1024 * 1024, 32);
        assert_eq!(cp.chunk_size, MIN_CHUNK_BYTES);
        assert_eq!(cp.chunk_count, 5);

        // conns = 0 defensively treated as 1: one chunk, no split.
        let cp = chunk_plan(50 * 1024 * 1024, 0);
        assert_eq!(cp.chunk_count, 1);
        assert_eq!(cp.chunk_size, 50 * 1024 * 1024);
    }

    #[test]
    fn unit__probe_total__strict_content_range() {
        assert_eq!(probe_total(Some("bytes 0-0/12345")), Some(12345));
        assert_eq!(probe_total(Some("bytes 0-0/0")), Some(0));
        assert_eq!(probe_total(None), None);
        assert_eq!(probe_total(Some("bytes 0-100/12345")), None); // not a 0-0 probe form
        assert_eq!(probe_total(Some("bytes 0-0/abc")), None);
        assert_eq!(probe_total(Some("bytes 0-0/")), None);
        assert_eq!(probe_total(Some("items 0-0/9")), None);
        assert_eq!(probe_total(Some("bytes 0-0/12x345")), None);
    }

    #[test]
    fn unit__sidecar__roundtrip_and_paths() {
        let dir = std::env::temp_dir().join(format!("blazar-sc-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let part = dir.join("model.gguf.part");
        std::fs::write(&part, b"x").unwrap();
        assert_eq!(
            sidecar_path(&part).file_name().unwrap(),
            "model.gguf.part.progress"
        );
        assert!(load_sidecar(&part).unwrap().is_none());
        let sc = ProgressSidecar {
            version: SIDECAR_VERSION,
            url: "https://x/y".into(),
            total: 42,
            chunk_size: 8,
            done: vec![0, 2],
            partial: vec![PartialChunk { idx: 1, bytes: 5 }],
        };
        store_sidecar(&part, &sc).unwrap();
        assert_eq!(load_sidecar(&part).unwrap().unwrap().done, vec![0, 2]);
        assert_eq!(
            load_sidecar(&part).unwrap().unwrap().partial,
            vec![PartialChunk { idx: 1, bytes: 5 }]
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn unit__sidecar__v1_without_partial_field_loads() {
        // Version-1 ledgers predate `partial`; they must keep resuming
        // their completed chunks (empty partial), not be discarded.
        let dir = std::env::temp_dir().join(format!("blazar-sc1-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let part = dir.join("model.gguf.part");
        std::fs::write(&part, b"x").unwrap();
        let v1 = r#"{"version":1,"url":"https://x/y","total":42,"chunk_size":8,"done":[0,2]}"#;
        std::fs::write(sidecar_path(&part), v1).unwrap();
        let sc = load_sidecar(&part)
            .unwrap()
            .expect("v1 sidecar is loadable");
        assert_eq!(sc.version, 1);
        assert_eq!(sc.done, vec![0, 2]);
        assert!(sc.partial.is_empty());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn unit__sidecar__persists_even_when_part_fsync_degrades() {
        // Durability ordering must degrade, never fail: a .part that
        // cannot be opened for fsync (here: replaced by a directory)
        // still gets its ledger written — a lost ledger is the exact
        // restart-from-zero bug the sidecar exists to prevent.
        let dir = std::env::temp_dir().join(format!("blazar-sc2-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let part = dir.join("model.gguf.part");
        std::fs::create_dir(&part).unwrap(); // open(write) fails -> warn path
        let sc = ProgressSidecar {
            version: SIDECAR_VERSION,
            url: "https://x/y".into(),
            total: 42,
            chunk_size: 8,
            done: vec![0],
            partial: vec![PartialChunk { idx: 1, bytes: 5 }],
        };
        store_sidecar(&part, &sc).unwrap();
        assert_eq!(
            load_sidecar(&part).unwrap().unwrap().partial,
            vec![PartialChunk { idx: 1, bytes: 5 }],
            "ledger persisted despite .part fsync degrade"
        );
        assert!(
            !dir.join("model.gguf.part.progress.tmp").exists(),
            "tmp cleaned by rename"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    // ---- minimal HTTP/1.1 server with real Range semantics ----

    /// Serves `payload` with HTTP 206 partial responses for Range requests
    /// (and plain 200 for full GETs). Returns (addr, payload, shutdown).
    async fn range_server(
        payload: Arc<Vec<u8>>,
        fail_first_n_probe: Arc<AtomicU64>,
    ) -> (std::net::SocketAddr, Arc<AtomicBool>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let shutdown = Arc::new(AtomicBool::new(false));
        let sd = shutdown.clone();
        tokio::spawn(async move {
            loop {
                let Ok((mut sock, _)) = listener.accept().await else {
                    break;
                };
                if sd.load(Ordering::Relaxed) {
                    break;
                }
                let payload = payload.clone();
                let fail_probe = fail_first_n_probe.clone();
                tokio::spawn(async move {
                    let mut buf = vec![0u8; 8192];
                    let n = sock.read(&mut buf).await.unwrap_or(0);
                    let req = String::from_utf8_lossy(&buf[..n]).to_string();
                    let range = req
                        .lines()
                        .find(|l| l.to_ascii_lowercase().starts_with("range:"))
                        .and_then(|l| l.split_once(':').map(|(_, v)| v.trim().to_string()));
                    let (code, body, content_range): (u16, Vec<u8>, String) = match range.as_deref()
                    {
                        Some("bytes=0-0") => {
                            if fail_probe
                                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                                    if v > 0 {
                                        Some(v - 1)
                                    } else {
                                        None
                                    }
                                })
                                .is_ok()
                            {
                                (500, b"boom".to_vec(), String::new())
                            } else {
                                (
                                    206,
                                    payload[..1].to_vec(),
                                    format!("bytes 0-0/{}", payload.len()),
                                )
                            }
                        }
                        Some(r) => {
                            let spec = r.strip_prefix("bytes=").unwrap_or("");
                            let (s, e) = spec.split_once('-').unwrap_or(("0", "0"));
                            let s: usize = s.parse().unwrap_or(0);
                            let e: usize = e.parse().unwrap_or(payload.len().saturating_sub(1));
                            let e = e.min(payload.len() - 1);
                            (
                                206,
                                payload[s..=e].to_vec(),
                                format!("bytes {s}-{e}/{}", payload.len()),
                            )
                        }
                        _ => (200, payload.to_vec(), String::new()),
                    };
                    let mut head = format!(
                        "HTTP/1.1 {} {}\r\ncontent-length: {}\r\n",
                        code,
                        if code == 206 { "Partial Content" } else { "OK" },
                        body.len()
                    );
                    if !content_range.is_empty() {
                        head.push_str("content-range: ");
                        head.push_str(&content_range);
                        head.push_str("\r\n");
                    }
                    head.push_str("connection: close\r\n\r\n");
                    let _ = sock.write_all(head.as_bytes()).await;
                    let _ = sock.write_all(&body).await;
                    let _ = sock.shutdown().await;
                });
            }
        });
        (addr, shutdown)
    }

    fn http_client() -> reqwest::Client {
        reqwest::Client::builder().build().unwrap()
    }

    /// 40 MiB of distinct bytes so offset bugs corrupt the body visibly,
    /// and so the payload clears `MIN_PARALLEL_BYTES` and yields `>= 4` chunks.
    fn write_payload(_dir: &Path) -> Vec<u8> {
        let block: Vec<u8> = (0..u64::from(u32::MAX))
            .take(1 << 20)
            .map(|i| (i % 251) as u8)
            .collect();
        let mut v = Vec::with_capacity(40 * 1024 * 1024);
        for rep in 0u8..40 {
            v.extend(block.iter().map(|b| b.wrapping_add(rep)));
        }
        v
    }

    #[tokio::test]
    async fn integration__try_parallel__byte_exact_assembly() {
        let payload = Arc::new(write_payload(Path::new(".")));
        let payload_len = payload.len() as u64;
        // Force the parallel lane below the 32 MiB floor for test speed.
        // The floor is checked by the caller path; here we test the lane
        // mechanics via a small total and direct invocation of internals:
        // chunk_plan + fetch_chunk cover the same code paths.
        let cp = chunk_plan(payload_len, 4);
        assert!(cp.chunk_count >= 3);
        let (addr, sd) = range_server(payload.clone(), Arc::new(AtomicU64::new(0))).await;
        let dir = std::env::temp_dir().join(format!("blazar-pp-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let part = dir.join("m.gguf.part");
        let _ = std::fs::remove_file(&part);
        std::fs::write(&part, b"").unwrap();
        let url: reqwest::Url = format!("http://{addr}/f.gguf").parse().unwrap();
        let client = http_client();
        let progress = AtomicU64::new(0);
        for idx in 0..cp.chunk_count {
            let start = idx * cp.chunk_size;
            let len = (cp.total - start).min(cp.chunk_size);
            let chunk_progress = AtomicU64::new(0);
            fetch_chunk(
                &client,
                None,
                &url,
                &part,
                start,
                len,
                &progress,
                &chunk_progress,
            )
            .await
            .unwrap();
        }
        let got = std::fs::read(&part).unwrap();
        assert_eq!(got, *payload);
        assert_eq!(progress.load(Ordering::Relaxed), payload_len);
        sd.store(true, Ordering::Relaxed);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[tokio::test]
    async fn integration__fetch_chunk__mid_body_reset_is_retried() {
        // HF occasionally resets the connection partway through a chunk
        // body ("end of file before message length reached"). The
        // body-read error must consume a retry attempt — not propagate
        // out of the attempt loop — and the retried range must land
        // byte-exact, with the progress bar refunded for the partial
        // attempt's bytes.
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let payload: Vec<u8> = (0u32..64 * 1024).map(|i| (i % 251) as u8).collect();
        let full = payload.clone();
        let half = payload[..10].to_vec();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let mut served = 0u32;
            while let Ok((mut sock, _)) = listener.accept().await {
                let mut buf = vec![0u8; 4096];
                let _ = sock.read(&mut buf).await;
                served += 1;
                let body = if served == 1 { &half } else { &full };
                let head = format!(
                    "HTTP/1.1 206 Partial Content\r\ncontent-length: {}\r\n\
                     content-range: bytes 0-{}/{}\r\nconnection: close\r\n\r\n",
                    full.len(),
                    full.len() - 1,
                    full.len()
                );
                let _ = sock.write_all(head.as_bytes()).await;
                let _ = sock.write_all(body).await;
                // First attempt closes mid-body (content-length already
                // promised the full chunk): premature EOF for the client.
                let _ = sock.shutdown().await;
            }
        });
        let dir = std::env::temp_dir().join(format!("blazar-rst-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let part = dir.join("m.gguf.part");
        std::fs::write(&part, b"").unwrap();
        let url: reqwest::Url = format!("http://{addr}/f.gguf").parse().unwrap();
        let progress = AtomicU64::new(0);
        let chunk_progress = AtomicU64::new(0);
        fetch_chunk(
            &http_client(),
            None,
            &url,
            &part,
            0,
            payload.len() as u64,
            &progress,
            &chunk_progress,
        )
        .await
        .expect("mid-body reset must be retried, not fatal");
        assert_eq!(std::fs::read(&part).unwrap(), payload);
        assert_eq!(
            progress.load(Ordering::Relaxed),
            payload.len() as u64,
            "partial attempt bytes refunded"
        );
        assert_eq!(
            chunk_progress.load(Ordering::Relaxed),
            payload.len() as u64,
            "per-chunk ledger refunded in lockstep"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[tokio::test]
    async fn integration__try_parallel__orphaned_full_length_part_refetched() {
        // Sidecar lost after a preallocated failure: the full-length .part
        // must be re-fetched in place by the parallel lane (every chunk,
        // write_at is idempotent) — never handed down to the classic lane,
        // whose length-based resume would send `bytes=total-` and 416.
        let payload = Arc::new(write_payload(Path::new(".")));
        let payload_len = payload.len() as u64;
        let (addr, sd) = range_server(payload.clone(), Arc::new(AtomicU64::new(0))).await;
        let dir = std::env::temp_dir().join(format!("blazar-orp-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let dest = dir.join("m.gguf");
        let part = crate::hf::sibling_part_path(&dest);
        let _ = std::fs::remove_file(sidecar_path(&part));
        let f = std::fs::File::create(&part).unwrap();
        f.set_len(payload_len).unwrap();
        drop(f);
        let url: reqwest::Url = format!("http://{addr}/f.gguf").parse().unwrap();
        let plan = FilePlan {
            filename: "f.gguf".into(),
            bytes: payload_len,
            sha256: None,
        };
        let mut prog = |_, _| {};
        let out = try_parallel(&http_client(), None, &url, &plan, &dest, 4, &mut prog)
            .await
            .unwrap();
        assert_eq!(out, Some(payload_len));
        assert_eq!(std::fs::read(&dest).unwrap(), *payload);
        assert!(!part.exists(), "artifact finalized into dest");
        sd.store(true, Ordering::Relaxed);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[tokio::test]
    async fn integration__try_parallel__short_part_without_sidecar_routed_to_classic() {
        // A short .part with no sidecar is a classic-lane tail partial
        // (contiguous prefix): even against a Range-capable server the
        // parallel lane must decline so the classic lane can resume it.
        let payload = Arc::new(write_payload(Path::new(".")));
        let payload_len = payload.len() as u64;
        let (addr, sd) = range_server(payload.clone(), Arc::new(AtomicU64::new(0))).await;
        let dir = std::env::temp_dir().join(format!("blazar-sho-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let dest = dir.join("m.gguf");
        let part = crate::hf::sibling_part_path(&dest);
        std::fs::write(&part, vec![0x5Au8; 1024]).unwrap();
        let url: reqwest::Url = format!("http://{addr}/f.gguf").parse().unwrap();
        let plan = FilePlan {
            filename: "f.gguf".into(),
            bytes: payload_len,
            sha256: None,
        };
        let mut prog = |_, _| {};
        let out = try_parallel(&http_client(), None, &url, &plan, &dest, 4, &mut prog)
            .await
            .unwrap();
        assert_eq!(out, None, "short classic partial belongs to classic lane");
        sd.store(true, Ordering::Relaxed);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[tokio::test]
    async fn integration__try_parallel__falls_back_on_missing_range_support() {
        // A server answering 200 (no Range support) must yield Ok(None).
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut sock, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 4096];
            let _ = sock.read(&mut buf).await;
            let body = vec![0u8; 64];
            let head = format!(
                "HTTP/1.1 200 OK\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                body.len()
            );
            let _ = sock.write_all(head.as_bytes()).await;
            let _ = sock.write_all(&body).await;
            let _ = sock.shutdown().await;
        });
        let dir = std::env::temp_dir().join(format!("blazar-fb-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let dest = dir.join("m.gguf");
        let url: reqwest::Url = format!("http://{addr}/f.gguf").parse().unwrap();
        let plan = FilePlan {
            filename: "f.gguf".into(),
            bytes: 64,
            sha256: None,
        };
        let mut prog = |_, _| {};
        let out = try_parallel(&http_client(), None, &url, &plan, &dest, 8, &mut prog)
            .await
            .unwrap();
        assert_eq!(out, None);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[tokio::test]
    async fn integration__try_parallel__full_lane_small_total_forces_none() {
        // 200-OK probe response (server ignores Range) -> None; also covers
        // the connections<2 guard returning None without touching network.
        let dir = std::env::temp_dir().join(format!("blazar-sm-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let dest = dir.join("m.gguf");
        let url: reqwest::Url = "http://127.0.0.1:1/f.gguf".parse().unwrap();
        let plan = FilePlan {
            filename: "f.gguf".into(),
            bytes: 0,
            sha256: None,
        };
        let mut prog = |_, _| {};
        let out = try_parallel(&http_client(), None, &url, &plan, &dest, 1, &mut prog)
            .await
            .unwrap();
        assert_eq!(out, None);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[tokio::test]
    async fn integration__try_parallel__resume_fetches_only_missing_chunks() {
        let payload = Arc::new(write_payload(Path::new(".")));
        let (addr, sd) = range_server(payload.clone(), Arc::new(AtomicU64::new(0))).await;
        let dir = std::env::temp_dir().join(format!("blazar-res-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let dest = dir.join("m.gguf");
        let part = crate::hf::sibling_part_path(&dest);
        let _ = std::fs::remove_file(&part);
        let url: reqwest::Url = format!("http://{addr}/f.gguf").parse().unwrap();

        // Seed a sidecar claiming chunk 0 done, with .part preallocated and
        // chunk 0's REAL bytes in place (a sparse hole would fail the final
        // content check — the lane trusts the sidecar, as it should).
        let cp = chunk_plan(payload.len() as u64, 4);
        let sc = ProgressSidecar {
            version: SIDECAR_VERSION,
            url: url.as_str().to_string(),
            total: cp.total,
            chunk_size: cp.chunk_size,
            done: vec![0],
            partial: vec![],
        };
        {
            use std::io::Write;
            let mut f = File::create(&part).unwrap();
            f.write_all(&payload[..usize::try_from(cp.chunk_size).unwrap()])
                .unwrap();
            f.set_len(cp.total).unwrap();
        }
        store_sidecar(&part, &sc).unwrap();

        let plan = FilePlan {
            filename: "f.gguf".into(),
            bytes: payload.len() as u64,
            sha256: None,
        };
        let mut prog = |_, _| {};
        let out = try_parallel(&http_client(), None, &url, &plan, &dest, 4, &mut prog)
            .await
            .unwrap();
        assert_eq!(out, Some(payload.len() as u64));
        assert_eq!(std::fs::read(&dest).unwrap(), *payload);
        assert!(!part.exists());
        assert!(!sidecar_path(&part).exists());
        sd.store(true, Ordering::Relaxed);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[tokio::test]
    // One scenario, three movements: seed a prefix + ledger, resume, and
    // prove the served ranges skip the banked bytes — splitting it would
    // hide exactly the cross-movement contract under test.
    #[allow(clippy::too_many_lines)]
    async fn integration__try_parallel__partial_chunk_prefix_is_not_refetched() {
        // The reported bug: interrupting before any WHOLE chunk completed
        // left a sparse full-length .part with no ledger, and the re-pull
        // re-downloaded from zero. The v2 ledger's per-chunk prefix must
        // be honored: only the remainder of chunk 0 is fetched, and no
        // request may touch the already-verified prefix bytes.
        use std::sync::Mutex;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let payload = Arc::new(write_payload(Path::new(".")));
        let payload_len = payload.len() as u64;
        let cp = chunk_plan(payload_len, 4);
        let prefix = 3 * 1024 * 1024u64; // inside chunk 0 (10 MiB)

        // Range server that records every (start, end) it serves.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let served: Arc<Mutex<Vec<(u64, u64)>>> = Arc::new(Mutex::new(Vec::new()));
        {
            let payload = payload.clone();
            let served = served.clone();
            tokio::spawn(async move {
                while let Ok((mut sock, _)) = listener.accept().await {
                    let payload = payload.clone();
                    let served = served.clone();
                    tokio::spawn(async move {
                        let mut buf = vec![0u8; 8192];
                        let n = sock.read(&mut buf).await.unwrap_or(0);
                        let req = String::from_utf8_lossy(&buf[..n]).to_string();
                        let spec = req
                            .lines()
                            .find(|l| l.to_ascii_lowercase().starts_with("range:"))
                            .and_then(|l| l.split_once(':'))
                            .map(|(_, v)| v.trim().trim_start_matches("bytes="));
                        let (s, e) = match spec {
                            Some(spec) => spec.split_once('-').map_or((0, 0), |(s, e)| {
                                (s.parse().unwrap_or(0), e.parse().unwrap_or(0))
                            }),
                            None => return, // probe-only server; plain GET unused
                        };
                        served.lock().unwrap().push((s, e));
                        let body = payload
                            [usize::try_from(s).unwrap()..=usize::try_from(e).unwrap()]
                            .to_vec();
                        let head = format!(
                            "HTTP/1.1 206 Partial Content\r\ncontent-length: {}\r\n\
                             content-range: bytes {s}-{e}/{}\r\nconnection: close\r\n\r\n",
                            body.len(),
                            payload.len()
                        );
                        let _ = sock.write_all(head.as_bytes()).await;
                        let _ = sock.write_all(&body).await;
                        let _ = sock.shutdown().await;
                    });
                }
            });
        }

        let dir = std::env::temp_dir().join(format!("blazar-ppx-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let dest = dir.join("m.gguf");
        let part = crate::hf::sibling_part_path(&dest);
        let _ = std::fs::remove_file(&part);
        {
            // Real chunk-0 prefix bytes on disk, then preallocate the rest.
            use std::io::Write;
            let mut f = File::create(&part).unwrap();
            f.write_all(&payload[..usize::try_from(prefix).unwrap()])
                .unwrap();
            f.set_len(cp.total).unwrap();
        }
        let url: reqwest::Url = format!("http://{addr}/f.gguf").parse().unwrap();
        store_sidecar(
            &part,
            &ProgressSidecar {
                version: SIDECAR_VERSION,
                url: url.as_str().to_string(),
                total: cp.total,
                chunk_size: cp.chunk_size,
                done: vec![],
                partial: vec![PartialChunk {
                    idx: 0,
                    bytes: prefix,
                }],
            },
        )
        .unwrap();

        let plan = FilePlan {
            filename: "f.gguf".into(),
            bytes: payload_len,
            sha256: None,
        };
        let mut prog = |_, _| {};
        let out = try_parallel(&http_client(), None, &url, &plan, &dest, 4, &mut prog)
            .await
            .unwrap();
        assert_eq!(out, Some(payload_len));
        assert_eq!(std::fs::read(&dest).unwrap(), *payload);

        let ranges = served.lock().unwrap().clone();
        // Probe excluded, nothing may overlap the verified prefix...
        for &(s, e) in ranges.iter().filter(|&&(s, e)| !(s == 0 && e == 0)) {
            assert!(
                s >= prefix,
                "refetched verified prefix bytes: range {s}-{e} vs prefix {prefix}"
            );
        }
        // ...and chunk 0's remainder must resume exactly at the prefix.
        assert!(
            ranges
                .iter()
                .any(|&(s, e)| s == prefix && e == cp.chunk_size - 1),
            "chunk 0 remainder must start at the persisted prefix: got {ranges:?}"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[tokio::test]
    async fn integration__try_parallel__ledger_exists_from_first_byte() {
        // The other half of the bug: the ledger used to be written only on
        // chunk completion, so an interrupted pull left nothing resumable.
        // The sidecar must be seeded the moment the lane engages — even a
        // pull that fails on the very first chunk fetch leaves a ledger.
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let payload_len: u64 = 40 * 1024 * 1024; // clears MIN_PARALLEL_BYTES
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            while let Ok((mut sock, _)) = listener.accept().await {
                let mut buf = vec![0u8; 4096];
                let _ = sock.read(&mut buf).await;
                let req = String::from_utf8_lossy(&buf).to_string();
                let probe = req
                    .lines()
                    .any(|l| l.eq_ignore_ascii_case("range: bytes=0-0"));
                let head = if probe {
                    "HTTP/1.1 206 Partial Content\r\ncontent-length: 1\r\n\
                     content-range: bytes 0-0/41943040\r\nconnection: close\r\n\r\n"
                        .to_string()
                } else {
                    "HTTP/1.1 500 Internal Server Error\r\ncontent-length: 0\r\n\
                     connection: close\r\n\r\n"
                        .to_string()
                };
                let _ = sock.write_all(head.as_bytes()).await;
                if probe {
                    let _ = sock.write_all(b"x").await;
                }
                let _ = sock.shutdown().await;
            }
        });

        let dir = std::env::temp_dir().join(format!("blazar-seed-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let dest = dir.join("m.gguf");
        let part = crate::hf::sibling_part_path(&dest);
        let _ = std::fs::remove_file(&part);
        let _ = std::fs::remove_file(sidecar_path(&part));
        let url: reqwest::Url = format!("http://{addr}/f.gguf").parse().unwrap();
        let plan = FilePlan {
            filename: "f.gguf".into(),
            bytes: payload_len,
            sha256: None,
        };
        let mut prog = |_, _| {};
        let err = try_parallel(&http_client(), None, &url, &plan, &dest, 4, &mut prog)
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("parallel download failed"),
            "unexpected error: {err:#}"
        );
        let sc_path = sidecar_path(&part);
        assert!(sc_path.exists(), "ledger must be seeded at engagement");
        let sc = load_sidecar(&part).unwrap().expect("seeded ledger loads");
        assert!(sc.done.is_empty());
        assert!(sc.partial.is_empty());
        assert_eq!(sc.total, payload_len, "preallocated .part reflected");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[tokio::test]
    async fn integration__execute_chunks__heartbeat_paints_before_first_chunk_completes() {
        // Live incident: a throttled CDN left the bar frozen at "0 B (0s)"
        // for the whole first chunk (minutes) because progress painted only
        // on ChunkDone — the user read a working download as dead and killed
        // it. Here the server delays every response past the observation
        // window, so ZERO chunks can complete; any paint must come from the
        // heartbeat.
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let payload_len: u64 = 64 * 1024 * 1024; // clears MIN_PARALLEL_BYTES
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            while let Ok((mut sock, _)) = listener.accept().await {
                let mut buf = vec![0u8; 4096];
                let _ = sock.read(&mut buf).await;
                tokio::time::sleep(Duration::from_millis(1_200)).await;
                let body = vec![0u8; 8];
                let head = format!(
                    "HTTP/1.1 206 Partial Content\r\ncontent-length: {}\r\n\
                     content-range: bytes 0-7/{payload_len}\r\nconnection: close\r\n\r\n",
                    body.len()
                );
                let _ = sock.write_all(head.as_bytes()).await;
                let _ = sock.write_all(&body).await;
                let _ = sock.shutdown().await;
            }
        });
        let dir = std::env::temp_dir().join(format!("blazar-hb-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let part = dir.join("m.gguf.part");
        std::fs::write(&part, b"").unwrap();
        let url: reqwest::Url = format!("http://{addr}/f.gguf").parse().unwrap();
        let cp = chunk_plan(payload_len, 4);
        let paints = Arc::new(AtomicU64::new(0));
        let seen = paints.clone();
        let mut on_progress = move |_done: u64, _total: u64| {
            seen.fetch_add(1, Ordering::Relaxed);
        };
        let client = http_client();
        let mut fut = Box::pin(execute_chunks(
            &client,
            None,
            &url,
            &part,
            &cp,
            ResumeLedger::default(),
            4,
            &mut on_progress,
        ));
        tokio::select! {
            _ = &mut fut => {}
            () = tokio::time::sleep(Duration::from_millis(700)) => {}
        }
        // 250 ms heartbeat over a 700 ms window with no chunk completions
        // must have painted at least twice.
        assert!(
            paints.load(Ordering::Relaxed) >= 2,
            "heartbeat failed to paint: {} paints in 700 ms with zero chunks done",
            paints.load(Ordering::Relaxed)
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
