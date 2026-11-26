//! System GPU census through the public probe: Blazar must see the GPUs
//! `nvidia-smi` sees even when that tool is not installed.

use std::io;
use std::process::Command;
use std::sync::{Arc, Mutex};

use tracing_subscriber::fmt::MakeWriter;

const CHILD_MARKER: &str = "BLAZAR_TEST_GPU_CENSUS_CHILD";
const TEST_NAME: &str = "integration__probe_hardware__sees_nvidia_gpus_without_nvidia_smi";
const QUIET_CHILD_MARKER: &str = "BLAZAR_TEST_QUIET_PROBE_CHILD";
const QUIET_TEST_NAME: &str = "integration__probe_helpers__missing_nvidia_smi_is_not_a_warning";

/// Log sink that shares its buffer with the test: a clone writes through
/// to the same allocation, so the subscriber's per-event writer clone
/// still lands where the test reads.
#[derive(Clone)]
struct SharedLogBuffer(Arc<Mutex<Vec<u8>>>);

impl io::Write for SharedLogBuffer {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.lock().expect("log buffer lock").write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.0.lock().expect("log buffer lock").flush()
    }
}

impl<'a> MakeWriter<'a> for SharedLogBuffer {
    type Writer = Self;

    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

/// `nvidia-smi` memory totals in MiB; empty when the tool is not installed.
fn nvidia_smi_totals() -> Vec<u64> {
    let Ok(out) = Command::new("nvidia-smi")
        .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
        .output()
    else {
        return Vec::new();
    };
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(|line| line.trim().parse::<u64>())
        .collect::<Result<Vec<_>, _>>()
        .unwrap_or_default()
}

/// Run one test of this binary again in a second process whose PATH
/// holds no `nvidia-smi`; the child reports through a printed line.
fn rerun_without_nvidia_smi(test_name: &str, marker: &str) -> String {
    let empty_path = tempfile::tempdir().expect("temp dir");
    let child = Command::new(std::env::current_exe().expect("test binary path"))
        .args(["--exact", test_name, "--nocapture"])
        .env("PATH", empty_path.path())
        .env(marker, "1")
        .output()
        .expect("spawn the test binary again");
    String::from_utf8_lossy(&child.stdout).into_owned()
}

/// The GPU rows a process on a PATH without `nvidia-smi` reported.
fn census_without_nvidia_smi() -> Vec<(String, u64, u64)> {
    let stdout = rerun_without_nvidia_smi(TEST_NAME, CHILD_MARKER);
    let Some(json) = stdout
        .lines()
        .find_map(|line| line.strip_prefix("GPU_CENSUS "))
    else {
        panic!("child process printed no census report\n{stdout}");
    };
    serde_json::from_str(json).expect("census report json")
}

#[test]
#[allow(non_snake_case)]
fn integration__probe_hardware__sees_nvidia_gpus_without_nvidia_smi() {
    if std::env::var_os(CHILD_MARKER).is_some() {
        let rows: Vec<(String, u64, u64)> = blazar_runtime::probe_hardware(None)
            .gpus
            .into_iter()
            .map(|g| (g.name, g.total_mib, g.free_mib))
            .collect();
        println!(
            "GPU_CENSUS {}",
            serde_json::to_string(&rows).expect("serialize census")
        );
        return;
    }

    let rows = census_without_nvidia_smi();
    for (name, total_mib, free_mib) in &rows {
        assert!(
            *total_mib > 0,
            "{name}: a row without a memory total is not a capacity row"
        );
        assert!(
            *free_mib <= *total_mib,
            "{name}: free {free_mib} MiB exceeds total {total_mib} MiB"
        );
    }
    let totals = rows.iter().map(|row| row.1).collect::<Vec<_>>();
    for total_mib in nvidia_smi_totals() {
        assert!(
            totals.contains(&total_mib),
            "GPU of {total_mib} MiB listed by nvidia-smi is invisible without the tool (census: {rows:?})"
        );
    }
}

/// The census module on its own, on whatever hardware the host has: the
/// merge policy promises every row is a capacity row, an integrated row
/// stands in only when no discrete card was found, and merging never
/// invents rows.
#[test]
#[allow(non_snake_case)]
fn integration__system_gpus__rows_are_capacity_rows() {
    let census = blazar_runtime::gpu_census::system_gpus();
    for gpu in &census.gpus {
        assert!(
            gpu.total_mib > 0,
            "{}: a row without a memory total is not a capacity row",
            gpu.name
        );
        assert!(
            gpu.free_mib <= gpu.total_mib,
            "{}: free {} MiB exceeds total {} MiB",
            gpu.name,
            gpu.free_mib,
            gpu.total_mib
        );
    }
    if census.gpus.iter().any(blazar_core::GpuInfo::is_integrated) {
        assert!(
            census.gpus.iter().all(blazar_core::GpuInfo::is_integrated),
            "an integrated row stands in only when the box has no discrete card: {:?}",
            census.gpus
        );
    }
    assert!(
        census.gpus.len() <= census.nvml + census.sysfs + census.vulkan + census.displays,
        "merged rows never exceed the rows the sources reported: {census:?}"
    );
}

/// Where `nvidia-smi` exists it is the oracle: the vendor-neutral sources
/// must see the same cards. Boxes without the tool have nothing to check.
#[test]
#[allow(non_snake_case)]
fn integration__system_gpus__nvidia_rows_match_nvidia_smi_totals() {
    let totals = nvidia_smi_totals();
    if totals.is_empty() {
        return;
    }
    let census = blazar_runtime::gpu_census::system_gpus();
    let listed: Vec<u64> = census.gpus.iter().map(|g| g.total_mib).collect();
    for total_mib in totals {
        assert!(
            listed.contains(&total_mib),
            "GPU of {total_mib} MiB listed by nvidia-smi is missing from the system census {census:?}"
        );
    }
}

/// The driver library is an installer's oracle for "an NVIDIA card is
/// present". Where `nvidia-smi` exists, the largest card it lists is the
/// card NVML reports.
#[test]
#[allow(non_snake_case)]
fn integration__nvml_total_mib__matches_the_largest_card_nvidia_smi_lists() {
    let Some(largest) = nvidia_smi_totals().into_iter().max() else {
        return;
    };
    assert_eq!(blazar_runtime::gpu_census::nvml_total_mib(), Some(largest));
}

/// A box without `nvidia-smi` is a normal box (AMD, Intel, a Mac, a
/// distribution that ships the driver without the CLI). The probe helpers
/// that read the tool must say so at debug, not warn on every call: the
/// serve banner and each `/api/capacity` call ask, and a warn per ask
/// buries real failures.
#[test]
#[allow(non_snake_case)]
fn integration__probe_helpers__missing_nvidia_smi_is_not_a_warning() {
    if std::env::var_os(QUIET_CHILD_MARKER).is_some() {
        let buffer = Arc::new(Mutex::new(Vec::new()));
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::WARN)
            .with_ansi(false)
            .with_writer(SharedLogBuffer(buffer.clone()))
            .try_init()
            .expect("child process installs its own log sink");
        let _ = blazar_runtime::probe::gpu_compute_tenants();
        let _ = blazar_runtime::probe::nvidia_free_vram_mib();
        let _ = blazar_runtime::probe::nvidia_used_vram_mib();
        let lines = buffer
            .lock()
            .expect("log buffer lock")
            .split(|byte| *byte == b'\n')
            .filter(|line| !line.is_empty())
            .count();
        println!("PROBE_HELPER_WARNS {lines}");
        return;
    }

    let stdout = rerun_without_nvidia_smi(QUIET_TEST_NAME, QUIET_CHILD_MARKER);
    let Some(report) = stdout
        .lines()
        .find_map(|line| line.strip_prefix("PROBE_HELPER_WARNS "))
    else {
        panic!("child process printed no warning report\n{stdout}");
    };
    assert_eq!(
        report.trim(),
        "0",
        "a missing nvidia-smi is a box property, not a failure:\n{stdout}"
    );
}
