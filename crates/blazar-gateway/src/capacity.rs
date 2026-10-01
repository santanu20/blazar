//! Per-GPU capacity observability (`GET /api/capacity`). v0.15.
//!
//! The planner already knows placement decisions; llama.cpp#26129-class
//! tooling wants the LIVE numbers as an API, not logs. This endpoint
//! joins three sources, each honest about its own limits:
//!
//! - `nvidia-smi` device census (index, name, total/free VRAM,
//!   utilization) — absent on non-NVIDIA boxes, reported as an empty
//!   device list with a note, never guessed.
//! - `nvidia-smi --query-compute-apps` (pid + used VRAM) — the ONLY
//!   per-process memory authority on the box.
//! - the supervisor's live instance rows (`Supervisor::ps`) — model,
//!   engine, slots, state, placement. Joining by pid attributes VRAM to
//!   Blazar residents; unmatched compute apps surface as `external`
//!   (another process eating the GPU — ollama users' "where did my VRAM
//!   go" question, answered with data).
//!
//! Weights/KV/compute SPLIT per resident is deliberately not claimed:
//! that split lives inside each engine child and would be an estimate
//! here. Per-process totals are measured; the split lands when engines
//! expose it natively.

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

use crate::state::AppState;

/// Bounded external-process run: kill past the deadline, drain both pipes
/// through threads (mirrors the runtime probe helper's fork-pressure
/// discipline — a wedged probe must never wedge the endpoint).
fn bounded_output(cmd: &mut std::process::Command, secs: u64) -> Option<std::process::Output> {
    use std::io::Read;
    use std::process::Stdio;
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = cmd.spawn().ok()?;
    let mut stdout = child.stdout.take().map(|mut p| {
        std::thread::spawn(move || {
            let mut b = Vec::new();
            p.read_to_end(&mut b).ok();
            b
        })
    });
    let mut stderr = child.stderr.take().map(|mut p| {
        std::thread::spawn(move || {
            let mut b = Vec::new();
            p.read_to_end(&mut b).ok();
            b
        })
    });
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(secs);
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {
                if std::time::Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    return None;
                }
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
            Err(_) => return None,
        }
    };
    let stdout = stdout.take().map(|h| h.join().unwrap_or_default());
    let stderr = stderr.take().map(|h| h.join().unwrap_or_default());
    Some(std::process::Output {
        status,
        stdout: stdout.unwrap_or_default(),
        stderr: stderr.unwrap_or_default(),
    })
}

struct DeviceRow {
    index: u32,
    name: String,
    total_mib: u64,
    free_mib: u64,
    util_pct: u64,
}

/// `index, name, total MiB, free MiB, util %` per line
/// (`--format=csv,noheader,nounits`). Rows with unparseable fields drop
/// silently-unknown rather than half-guessed.
fn parse_devices(text: &str) -> Vec<DeviceRow> {
    text.lines()
        .filter_map(|ln| {
            let mut parts = ln.split(',').map(str::trim);
            let index: u32 = parts.next()?.parse().ok()?;
            let name = parts.next()?.to_string();
            if name.is_empty() {
                return None;
            }
            Some(DeviceRow {
                index,
                name,
                total_mib: parts.next()?.parse().ok()?,
                free_mib: parts.next()?.parse().ok()?,
                util_pct: parts.next().unwrap_or("0").parse().unwrap_or(0),
            })
        })
        .collect()
}

/// `pid, used MiB` per line. Drivers without the number print `[N/A]` —
/// those rows report the pid with an unknown footprint (0), never a guess.
fn parse_compute_apps(text: &str) -> Vec<(u32, u64)> {
    text.lines()
        .filter_map(|ln| {
            let mut parts = ln.split(',').map(str::trim);
            let pid: u32 = parts.next()?.parse().ok()?;
            let mib: u64 = parts.next().and_then(|v| v.parse().ok()).unwrap_or(0);
            Some((pid, mib))
        })
        .collect()
}

/// GET /api/capacity — device census + per-resident VRAM attribution.
pub async fn capacity(State(state): State<Arc<AppState>>) -> Response {
    // Two nvidia-smi spawns (~30-60 ms each) + pipe polling: blocking
    // work, off the async runtime threads.
    let (devices_raw, apps_raw) = tokio::task::spawn_blocking(|| {
        let devices = bounded_output(
            std::process::Command::new("nvidia-smi").args([
                "--query-gpu=index,name,memory.total,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ]),
            5,
        )
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).into_owned());
        let apps = bounded_output(
            std::process::Command::new("nvidia-smi").args([
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ]),
            5,
        )
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).into_owned());
        (devices, apps)
    })
    .await
    .unwrap_or((None, None));

    let mut notes = Vec::new();
    let devices: Vec<serde_json::Value> = match devices_raw.as_deref() {
        Some(text) => parse_devices(text)
            .into_iter()
            .map(|d| {
                serde_json::json!({
                    "id": format!("cuda:{}", d.index),
                    "index": d.index,
                    "name": d.name,
                    "total_vram_bytes": d.total_mib * 1024 * 1024,
                    "free_vram_bytes": d.free_mib * 1024 * 1024,
                    "utilization_percent": d.util_pct,
                })
            })
            .collect(),
        None => {
            notes.push(
                "no nvidia-smi census on this box — non-NVIDIA GPU boxes report an empty \
                 device list (residents below still come from the supervisor)",
            );
            Vec::new()
        }
    };

    // Supervisor rows are the resident truth; compute-apps attributes VRAM.
    let ps_rows = state.sup.ps();
    let apps = apps_raw.as_deref().map(|t| parse_compute_apps(t));
    let mut external: Vec<serde_json::Value> = Vec::new();
    let residents: Vec<serde_json::Value> = ps_rows
        .iter()
        .map(|p| {
            let mut v = serde_json::json!({
                "model": p.name,
                "engine": p.engine,
                "pid": p.pid,
                "state": p.state,
                "slots": p.slots,
                "slots_configured": p.slots_configured,
                "in_flight": p.in_flight,
                "device": p.device,
                "device_id": p.device_id,
                "model_bytes": p.bytes,
            });
            if let Some(apps) = &apps {
                match apps.iter().find(|(pid, _)| *pid == p.pid) {
                    Some((_, mib)) => {
                        v["vram_bytes"] = serde_json::json!(mib * 1024 * 1024);
                    }
                    // Resident but not (yet) holding GPU memory — CPU
                    // instances and engine kinds without compute apps.
                    None => {
                        v["vram_bytes"] = serde_json::Value::Null;
                    }
                }
            }
            v
        })
        .collect();

    if let Some(apps) = &apps {
        let known: std::collections::HashSet<u32> = ps_rows.iter().map(|p| p.pid).collect();
        for (pid, mib) in apps {
            if !known.contains(pid) {
                external.push(serde_json::json!({
                    "pid": pid,
                    "vram_bytes": mib * 1024 * 1024,
                    "note": "external process (not a Blazar child)",
                }));
            }
        }
    }
    notes.push(
        "per-resident vram is the process total from nvidia-smi compute-apps; \
         the weights/KV/compute split lives inside each engine child and is \
         not estimated here",
    );

    (
        StatusCode::OK,
        axum::Json(serde_json::json!({
            "object": "blazar.capacity",
            "devices": devices,
            "residents": residents,
            "external": external,
            "notes": notes,
        })),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit__parse_devices__census_rows_and_drops_malformed() {
        let rows = parse_devices(
            "0, NVIDIA GeForce RTX 4070, 8192, 4096, 37\n\
             1, NVIDIA GeForce RTX 3090, 24576, 24575, 0\n\
             garbage line\n",
        );
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].name, "NVIDIA GeForce RTX 4070");
        assert_eq!(rows[0].total_mib, 8192);
        assert_eq!(rows[0].free_mib, 4096);
        assert_eq!(rows[0].util_pct, 37);
        assert_eq!(rows[1].index, 1);
    }

    #[test]
    fn unit__parse_compute_apps__na_footprint_is_zero_not_guessed() {
        let apps = parse_compute_apps("12345, 2048\n67890, [N/A]\nnope\n");
        assert_eq!(apps.len(), 2);
        assert_eq!(apps[0], (12345, 2048));
        assert_eq!(apps[1], (67890, 0), "[N/A] footprint stays unknown(0)");
    }
}
