//! `ComputeFabric` read-model (`GET /api/fabric`). v0.21, Wave 3.
//!
//! One consolidated inventory of everything compute-ish this daemon
//! can see, assembled read-only from sources that each stay honest
//! about their own limits:
//!
//! - CPU: sysinfo physical cores + total RAM (`probe_hardware`).
//! - GPUs: live `nvidia-smi` census when present (the same bounded
//!   spawn `capacity.rs` uses), falling back to the engine manifest's
//!   device list, falling back to an empty list plus a note. Never a
//!   guess, never an error.
//! - Residents: supervisor live rows joined onto their device by the
//!   census key (`CUDA0` and `cuda:0` are the same card to the join),
//!   with per-process VRAM from compute-apps where the driver reports
//!   it. External (non-Blazar) GPU processes surface per device.
//! - Engines: the store's installed rows with the active one flagged.
//! - Peers: cached `PeerCapacity` from configured remotes, named by
//!   the remote's routing prefix; an unprobed peer is simply absent.
//!
//! This is a READ MODEL, not a scheduler: placement stays where it is
//! (spawn planner + one-active-engine-per-daemon). The fabric answers
//! "what is there and what is it doing", it never moves work.

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use blazar_core::Hardware;
use blazar_core::store::EngineRow;
use blazar_runtime::supervisor::PsRow;

use crate::capacity::{DeviceRow, bounded_output, parse_compute_apps, parse_devices};
use crate::remotes::PeerCapacity;
use crate::state::AppState;

/// Normalize a census device id for joins: the supervisor writes
/// backend-namespace keys (`CUDA0`), the capacity census writes
/// lowercase path ids (`cuda:0`) — to this fn they are the same card.
fn join_key(id: &str) -> String {
    id.chars()
        .filter(char::is_ascii_alphanumeric)
        .collect::<String>()
        .to_ascii_lowercase()
}

/// Census pair handed to [`fabric_snapshot`]: device rows from
/// `nvidia-smi --query-gpu` plus per-pid VRAM from compute-apps.
pub(crate) type Census<'a> = Option<(&'a [DeviceRow], &'a [(u32, u64)])>;

/// Pure assembly of the fabric snapshot — every input injected, so the
/// whole read-model is unit-pinnable without a live daemon.
#[allow(clippy::too_many_lines)] // one JSON block per source, cohesive
#[must_use]
pub(crate) fn fabric_snapshot(
    hardware: &Hardware,
    census: Census<'_>,
    ps_rows: &[PsRow],
    engines: &[EngineRow],
    peers: &[(String, String, PeerCapacity)],
) -> serde_json::Value {
    let mut notes = Vec::new();

    // CPU block: always measured, never absent.
    let cpu = serde_json::json!({
        "physical_cores": hardware.physical_cores,
        "total_ram_bytes": hardware.total_ram_mib * 1024 * 1024,
    });

    // GPU block: live census > manifest devices > honest absence.
    let apps = census.map(|(_, apps)| apps);
    let (devices, backend) = match census {
        Some((rows, _)) if !rows.is_empty() => {
            let n = rows.len();
            (rows.to_vec(), format!("nvidia-smi census ({n} device(s))"))
        }
        _ => {
            let gpus: Vec<DeviceRow> = hardware
                .gpus
                .iter()
                .enumerate()
                .map(|(i, g)| DeviceRow {
                    index: u32::try_from(i).expect("device index fits u32"),
                    name: g.name.clone(),
                    total_mib: g.total_mib,
                    free_mib: g.free_mib,
                    // Manifest censuses carry no utilization signal.
                    util_pct: 0,
                })
                .collect();
            if gpus.is_empty() {
                notes.push(
                    "no GPU census available — non-NVIDIA boxes without an engine manifest \
                     census report an empty device list; residents below still come from \
                     the supervisor",
                );
                (Vec::new(), "none".to_string())
            } else {
                notes.push(
                    "nvidia-smi unavailable — device list comes from the active engine \
                     manifest census (no utilization, no per-process VRAM)",
                );
                (gpus, "engine manifest census".to_string())
            }
        }
    };

    let devices: Vec<serde_json::Value> = devices
        .iter()
        .map(|d| {
            let id = format!("cuda:{}", d.index);
            let key = join_key(&id);
            let mut residents = Vec::new();
            let mut external = Vec::new();
            for p in ps_rows {
                if p.device_id
                    .as_deref()
                    .is_some_and(|pid| join_key(pid) == key)
                {
                    let vram = apps.and_then(|a| {
                        a.iter()
                            .find(|(pid, _)| *pid == p.pid)
                            .map(|&(_, mib)| mib * 1024 * 1024)
                    });
                    residents.push(serde_json::json!({
                        "model": p.name,
                        "engine": p.engine,
                        "pid": p.pid,
                        "state": p.state,
                        "slots": p.slots,
                        "in_flight": p.in_flight,
                        "vram_bytes": vram,
                    }));
                }
            }
            if let Some(a) = apps {
                let known: std::collections::HashSet<u32> = ps_rows.iter().map(|p| p.pid).collect();
                for (pid, mib) in a {
                    if !known.contains(pid) {
                        external.push(serde_json::json!({
                            "pid": pid,
                            "vram_bytes": mib * 1024 * 1024,
                            "note": "external process (not a Blazar child)",
                        }));
                    }
                }
            }
            serde_json::json!({
                "id": id,
                "name": d.name,
                "backend": "cuda",
                "total_vram_bytes": d.total_mib * 1024 * 1024,
                "free_vram_bytes": d.free_mib * 1024 * 1024,
                "utilization_percent": d.util_pct,
                "residents": residents,
                "external": external,
            })
        })
        .collect();

    // Assignments: every supervisor row, wherever it sits (CPU rows
    // and card-spanning tensor-split rows carry no device_id).
    let assignments: Vec<serde_json::Value> = ps_rows
        .iter()
        .map(|p| {
            serde_json::json!({
                "model": p.name,
                "engine": p.engine,
                "replica": p.replica,
                "state": p.state,
                "gpu": p.gpu,
                "device": p.device,
                "device_id": p.device_id,
                "ctx": p.ctx,
                "slots": p.slots,
                "slots_configured": p.slots_configured,
                "in_flight": p.in_flight,
                "idle_secs": p.idle_secs,
            })
        })
        .collect();

    let engines: Vec<serde_json::Value> = engines
        .iter()
        .map(|e| {
            serde_json::json!({
                "tag": e.tag,
                "kind": e.kind.as_str(),
                "active": e.active,
                "installed_at": e.installed_at,
            })
        })
        .collect();

    let peers: Vec<serde_json::Value> = peers
        .iter()
        .map(|(name, url, cap)| {
            serde_json::json!({
                "name": name,
                "url": url,
                "devices": cap.devices.iter().map(|d| serde_json::json!({
                    "name": d.name,
                    "total_vram_bytes": d.total_vram_bytes,
                    "free_vram_bytes": d.free_vram_bytes,
                })).collect::<Vec<_>>(),
                "residents": cap.residents.iter().map(|r| serde_json::json!({
                    "model": r.model,
                    "engine": r.engine,
                    "state": r.state,
                    "in_flight": r.in_flight,
                })).collect::<Vec<_>>(),
            })
        })
        .collect();

    notes.push("read-model inventory — placement decisions stay with the spawn planner");
    serde_json::json!({
        "object": "blazar.fabric",
        "backend": backend,
        "cpu": cpu,
        "devices": devices,
        "assignments": assignments,
        "engines": engines,
        "peers": peers,
        "notes": notes,
    })
}

/// `GET /api/fabric` — the `ComputeFabric` read-model.
pub async fn fabric(State(state): State<Arc<AppState>>) -> Response {
    // Same two bounded nvidia-smi spawns as /api/capacity (~30-60 ms):
    // blocking work, off the async runtime threads.
    let census = tokio::task::spawn_blocking(|| {
        let devices = bounded_output(
            std::process::Command::new("nvidia-smi").args([
                "--query-gpu=index,name,memory.total,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ]),
            5,
        )
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
        .map(|t| parse_devices(&t));
        let apps = bounded_output(
            std::process::Command::new("nvidia-smi").args([
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ]),
            5,
        )
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
        .map(|t| parse_compute_apps(&t));
        devices.map(|d| (d, apps.unwrap_or_default()))
    })
    .await
    .ok()
    .flatten();

    let hardware = blazar_runtime::probe_hardware(None);
    let ps_rows = state.sup.ps();
    // Flat, short store borrow (never nested inside another with_store).
    let engines = state
        .with_store(blazar_core::Store::list_engines)
        .and_then(std::result::Result::ok)
        .unwrap_or_default();
    // Peers: snapshot the capacity cache once, name entries by their
    // configured remote (an unprobed remote is simply absent).
    let caps = state
        .remote_capacity
        .lock()
        .expect("remote_capacity lock poisoned")
        .clone();
    let peers: Vec<(String, String, PeerCapacity)> = state
        .config
        .remotes
        .iter()
        .filter_map(|r| {
            let cap = caps.get(&crate::remotes::health_key(r))?.clone();
            Some((r.name.clone(), r.url.clone(), cap))
        })
        .collect();

    (
        StatusCode::OK,
        axum::Json(fabric_snapshot(
            &hardware,
            census.as_ref().map(|(d, a)| (d.as_slice(), a.as_slice())),
            &ps_rows,
            &engines,
            &peers,
        )),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]
    use super::*;

    fn ps_row(name: &str, device_id: Option<&str>, pid: u32) -> PsRow {
        PsRow {
            name: name.to_string(),
            engine: "llamacpp-b1".to_string(),
            replica: None,
            state: "running",
            endpoint: "http://127.0.0.1:1".to_string(),
            idle_secs: 4,
            in_flight: 1,
            ctx: 16384,
            slots: Some(2),
            slots_configured: Some(2),
            gpu: "full".to_string(),
            device: Some("RTX 4070".to_string()),
            device_id: device_id.map(str::to_string),
            warnings: Vec::new(),
            spec_mode: "off".to_string(),
            draft: None,
            heat: 0,
            keep_alive_secs: None,
            pid,
            bytes: 1024,
        }
    }

    fn hardware(gpus: Vec<blazar_core::GpuInfo>) -> Hardware {
        Hardware {
            physical_cores: 8,
            total_ram_mib: 32768,
            gpus,
        }
    }

    fn census() -> (Vec<DeviceRow>, Vec<(u32, u64)>) {
        (
            vec![DeviceRow {
                index: 0,
                name: "NVIDIA GeForce RTX 4070".to_string(),
                total_mib: 8192,
                free_mib: 4096,
                util_pct: 37,
            }],
            vec![(4242, 2048), (9999, 512)],
        )
    }

    #[test]
    #[allow(non_snake_case)] // pin names read as scenario sentences (house pattern)
    fn unit__fabric_snapshot__census_devices_with_residents_joined() {
        let rows = [
            ps_row("qwen3", Some("CUDA0"), 4242),
            ps_row("cpu-model", None, 5555),
        ];
        let (dev, apps) = census();
        let v = fabric_snapshot(&hardware(Vec::new()), Some((&dev, &apps)), &rows, &[], &[]);
        assert_eq!(v["object"], "blazar.fabric");
        assert_eq!(v["backend"], "nvidia-smi census (1 device(s))");
        let d = &v["devices"][0];
        assert_eq!(d["id"], "cuda:0");
        assert_eq!(d["total_vram_bytes"], 8192u64 * 1024 * 1024);
        // CUDA0 == cuda:0 to the join, and the resident's VRAM comes
        // from the compute-apps row for its pid.
        assert_eq!(d["residents"][0]["model"], "qwen3");
        assert_eq!(d["residents"][0]["vram_bytes"], 2048u64 * 1024 * 1024);
        // Unmatched compute app = external, attributed to the device.
        assert_eq!(d["external"][0]["pid"], 9999);
        // The CPU-placed row stays in assignments with a null device_id.
        assert_eq!(v["assignments"].as_array().map(Vec::len), Some(2));
        assert_eq!(v["assignments"][1]["device_id"], serde_json::Value::Null);
    }

    #[test]
    #[allow(non_snake_case)] // pin names read as scenario sentences (house pattern)
    fn unit__fabric_snapshot__manifest_fallback_and_honest_absence() {
        let gpus = vec![blazar_core::GpuInfo {
            name: "RTX 4070".to_string(),
            description: "vulkan".to_string(),
            total_mib: 8192,
            free_mib: 8192,
        }];
        // Census absent, manifest present: devices render from the
        // manifest, the note names the degraded source.
        let v = fabric_snapshot(&hardware(gpus), None, &[], &[], &[]);
        assert_eq!(v["backend"], "engine manifest census");
        assert_eq!(v["devices"][0]["name"], "RTX 4070");
        assert_eq!(
            v["devices"][0]["residents"].as_array().map(Vec::len),
            Some(0)
        );
        assert!(
            v["notes"].to_string().contains("manifest"),
            "degraded source must be named: {}",
            v["notes"]
        );

        // Neither source: empty list plus an honest note, never a guess.
        let empty = fabric_snapshot(&hardware(Vec::new()), None, &[], &[], &[]);
        assert_eq!(empty["devices"].as_array().map(Vec::len), Some(0));
        assert!(empty["notes"].to_string().contains("no GPU census"));
        assert_eq!(empty["cpu"]["physical_cores"], 8);
        assert_eq!(empty["cpu"]["total_ram_bytes"], 32768u64 * 1024 * 1024);
    }

    #[test]
    #[allow(non_snake_case)] // pin names read as scenario sentences (house pattern)
    fn unit__fabric_snapshot__engines_and_peers_shapes() {
        let engines = [EngineRow {
            tag: "b11370-cuda".to_string(),
            asset: "llamacpp.zip".to_string(),
            sha256: "abc".to_string(),
            installed_at: 1_790_000_000,
            active: true,
            manifest: String::new(),
            kind: blazar_core::engine_kind::EngineKind::LlamaCpp,
        }];
        let peer = PeerCapacity {
            fetched: std::time::Instant::now(),
            devices: vec![crate::remotes::PeerDevice {
                name: "RTX 3090".to_string(),
                total_vram_bytes: 24,
                free_vram_bytes: 12,
            }],
            residents: vec![crate::remotes::PeerResident {
                model: "m2".to_string(),
                engine: "e".to_string(),
                state: "running".to_string(),
                slots: None,
                slots_configured: None,
                in_flight: 3,
            }],
        };
        let v = fabric_snapshot(
            &hardware(Vec::new()),
            None,
            &[],
            &engines,
            &[(
                "workstation".to_string(),
                "http://10.0.0.4:11435".to_string(),
                peer,
            )],
        );
        assert_eq!(v["engines"][0]["tag"], "b11370-cuda");
        assert_eq!(v["engines"][0]["kind"], "llamacpp");
        assert_eq!(v["engines"][0]["active"], true);
        assert_eq!(v["peers"][0]["name"], "workstation");
        assert_eq!(v["peers"][0]["devices"][0]["total_vram_bytes"], 24);
        assert_eq!(v["peers"][0]["residents"][0]["in_flight"], 3);
        // The read-model disclaimer rides every snapshot.
        assert!(v["notes"].to_string().contains("read-model"));
    }
}
