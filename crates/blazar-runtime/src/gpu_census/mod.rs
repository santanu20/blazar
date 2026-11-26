//! Vendor-neutral GPU census.
//!
//! [`crate::probe_hardware`] knows a machine's GPUs through the active
//! engine's device list and through `nvidia-smi`. Both can be missing at
//! once: an engine without a device listing (sglang, vllm, MLX) as the
//! active lane, on a distribution that does not ship `nvidia-smi`. The
//! hardware then read "0 GPUs", every GGUF model started CPU-only
//! (measured 47 tok/s against 340 on the same card) and nothing said why.
//!
//! This module asks the system itself, in the order that costs least and
//! trusts a vendor's own numbers first:
//!
//! 1. NVML, the driver library behind `nvidia-smi`. Its rows are identical
//!    to the `nvidia-smi` rows.
//! 2. Linux sysfs: amdgpu is the one kernel driver that publishes VRAM
//!    counters.
//! 3. The Vulkan loader, the one vendor-neutral inventory every desktop
//!    distribution ships. Skipped once an earlier source saw a discrete
//!    card.
//!
//! On macOS only `system_profiler` reports GPUs. On Windows NVML and the
//! Vulkan loader serve.
//!
//! Rows of this census describe the machine. They are never handed to an
//! engine as a device name: what an engine accepts is only what its own
//! `--list-devices` prints.

// NVML and the Vulkan loader are C libraries without a safe wrapper in
// this workspace. The unsafe surface is the symbol lookup and the calls
// behind it; each has its own SAFETY note and a layout pin in its tests.
#![allow(unsafe_code)]

#[cfg(not(target_os = "macos"))]
mod dynlib;
#[cfg(target_os = "linux")]
mod linux_drm;
#[cfg(any(target_os = "macos", test))]
mod mac_displays;
#[cfg(not(target_os = "macos"))]
mod nvml;
#[cfg(not(target_os = "macos"))]
mod vulkan;

use blazar_core::GpuInfo;

/// A GPU as one source of the census reports it.
struct CensusGpu {
    name: String,
    description: String,
    total_mib: u64,
    free_mib: u64,
    /// PCI vendor id, `0` when the source does not know it.
    vendor_id: u32,
    /// Shares system memory instead of owning VRAM.
    integrated: bool,
}

impl From<CensusGpu> for GpuInfo {
    fn from(gpu: CensusGpu) -> Self {
        let CensusGpu {
            name,
            description,
            total_mib,
            free_mib,
            ..
        } = gpu;
        Self {
            name,
            description,
            total_mib,
            free_mib,
        }
    }
}

/// The system's GPUs and where each list came from.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SystemGpuCensus {
    /// One row per GPU, in source precedence order.
    pub gpus: Vec<GpuInfo>,
    /// Rows NVML returned.
    pub nvml: usize,
    /// Rows the Linux sysfs source returned.
    pub sysfs: usize,
    /// Rows the Vulkan loader returned.
    pub vulkan: usize,
    /// Rows `system_profiler` returned.
    pub displays: usize,
    /// PCI vendor id of each row, each vendor once, in row order.
    pub vendors: Vec<u32>,
}

/// Asks the system for its GPUs. A row never lacks a memory size: a GPU
/// without one is not a capacity signal.
#[must_use]
pub fn system_gpus() -> SystemGpuCensus {
    let (sources, counts) = platform_sources();
    let SourceCounts {
        nvml,
        sysfs,
        vulkan,
        displays,
    } = counts;
    let merged = merge_policy(sources);
    let vendors = vendors_of(&merged);
    SystemGpuCensus {
        gpus: merged.into_iter().map(GpuInfo::from).collect(),
        nvml,
        sysfs,
        vulkan,
        displays,
        vendors,
    }
}

/// The PCI vendor of each row, each vendor once, in row order.
fn vendors_of(gpus: &[CensusGpu]) -> Vec<u32> {
    let mut vendors: Vec<u32> = Vec::new();
    for gpu in gpus {
        if !vendors.contains(&gpu.vendor_id) {
            vendors.push(gpu.vendor_id);
        }
    }
    vendors
}

/// Whether the Vulkan loader enumerates a discrete GPU on this host. An
/// installer that has no engine to probe yet asks here before it decides
/// whether a card without its vendor runtime can be served.
///
/// An integrated GPU does not count. On a small iGPU the Vulkan build's GPU
/// path decoded several times slower than the CPU backend of the same
/// build (live run recorded under
/// `bench-artifacts/2026-11-23-sota-roadmap/`), so an integrated-only host
/// keeps the CPU build it always got.
#[cfg(not(target_os = "macos"))]
#[must_use]
pub fn vulkan_discrete_gpu_present() -> bool {
    has_discrete_gpu(&vulkan::gpus())
}

/// No Blazar engine build runs on Vulkan on macOS.
#[cfg(target_os = "macos")]
#[must_use]
pub fn vulkan_discrete_gpu_present() -> bool {
    false
}

/// Whether any row is a discrete card. Built wherever its caller is, or
/// under test: macOS has no caller and must not carry dead code.
#[cfg(any(not(target_os = "macos"), test))]
fn has_discrete_gpu(gpus: &[CensusGpu]) -> bool {
    gpus.iter().any(|gpu| !gpu.integrated)
}

/// The memory of the largest NVIDIA card the driver library reports, in
/// MiB. The driver answers on hosts that lack the `nvidia-smi` tool, so an
/// installer that needs an NVIDIA card asks here and not for the tool.
#[cfg(not(target_os = "macos"))]
#[must_use]
pub fn nvml_total_mib() -> Option<u64> {
    nvml::gpus()
        .iter()
        .map(|gpu| gpu.total_mib)
        .max()
        .filter(|total_mib| *total_mib > 0)
}

/// macOS ships no NVML.
#[cfg(target_os = "macos")]
#[must_use]
pub fn nvml_total_mib() -> Option<u64> {
    None
}

/// How many rows each source returned, before de-duplication.
struct SourceCounts {
    nvml: usize,
    sysfs: usize,
    vulkan: usize,
    displays: usize,
}

#[cfg(target_os = "linux")]
fn platform_sources() -> (Vec<Vec<CensusGpu>>, SourceCounts) {
    let nvml = nvml::gpus();
    let sysfs = linux_drm::gpus();
    let vulkan = vulkan::gpus_if_needed(&[&nvml, &sysfs]);
    let counts = SourceCounts {
        nvml: nvml.len(),
        sysfs: sysfs.len(),
        vulkan: vulkan.len(),
        displays: 0,
    };
    (vec![nvml, sysfs, vulkan], counts)
}

#[cfg(target_os = "windows")]
fn platform_sources() -> (Vec<Vec<CensusGpu>>, SourceCounts) {
    let nvml = nvml::gpus();
    let vulkan = vulkan::gpus_if_needed(&[&nvml]);
    let counts = SourceCounts {
        nvml: nvml.len(),
        sysfs: 0,
        vulkan: vulkan.len(),
        displays: 0,
    };
    (vec![nvml, vulkan], counts)
}

#[cfg(target_os = "macos")]
fn platform_sources() -> (Vec<Vec<CensusGpu>>, SourceCounts) {
    let displays = mac_displays::gpus();
    let counts = SourceCounts {
        nvml: 0,
        sysfs: 0,
        vulkan: 0,
        displays: displays.len(),
    };
    (vec![displays], counts)
}

#[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
fn platform_sources() -> (Vec<Vec<CensusGpu>>, SourceCounts) {
    let counts = SourceCounts {
        nvml: 0,
        sysfs: 0,
        vulkan: 0,
        displays: 0,
    };
    (Vec::new(), counts)
}

/// Sources arrive in precedence order. A source may not repeat a vendor an
/// earlier source already reported: NVML and Vulkan both list the same
/// NVIDIA card under different names, and two rows for one card would
/// double every VRAM total. Cards of one vendor listed by one source (two
/// AMD cards through sysfs) all stay.
///
/// A card that shares system RAM counts only on a machine without a
/// discrete card. With a discrete card present it must not add to any
/// capacity sum, the rule `GpuInfo::is_integrated` applies downstream.
fn merge_policy(sources: Vec<Vec<CensusGpu>>) -> Vec<CensusGpu> {
    let mut reported: Vec<u32> = Vec::new();
    let mut merged: Vec<CensusGpu> = Vec::new();
    for source in sources {
        let mut kept_vendors: Vec<u32> = Vec::new();
        for mut gpu in source {
            if gpu.total_mib == 0 || reported.contains(&gpu.vendor_id) {
                continue;
            }
            gpu.free_mib = gpu.free_mib.min(gpu.total_mib);
            if !kept_vendors.contains(&gpu.vendor_id) {
                kept_vendors.push(gpu.vendor_id);
            }
            merged.push(gpu);
        }
        reported.extend(kept_vendors);
    }
    if merged.iter().any(|gpu| !gpu.integrated) {
        merged.retain(|gpu| !gpu.integrated);
    }
    merged
}

/// A fixed-size C string buffer: up to the first NUL, without padding.
#[cfg(not(target_os = "macos"))]
fn c_string(bytes: &[u8]) -> String {
    let end = bytes
        .iter()
        .position(|&byte| byte == 0)
        .unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).trim().to_string()
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use blazar_core::GpuInfo;

    use super::{CensusGpu, merge_policy};

    const NVIDIA: u32 = 0x10de;
    const AMD: u32 = 0x1002;
    const INTEL: u32 = 0x8086;

    fn gpu(
        name: &str,
        vendor_id: u32,
        integrated: bool,
        total_mib: u64,
        free_mib: u64,
    ) -> CensusGpu {
        CensusGpu {
            name: name.into(),
            description: name.into(),
            total_mib,
            free_mib,
            vendor_id,
            integrated,
        }
    }

    fn names(rows: &[CensusGpu]) -> Vec<&str> {
        rows.iter().map(|row| row.name.as_str()).collect()
    }

    #[test]
    fn unit__merge_policy__later_source_does_not_repeat_a_vendor() {
        let rows = merge_policy(vec![
            vec![gpu(
                "NVIDIA GeForce RTX 4070 Laptop GPU",
                NVIDIA,
                false,
                8_188,
                6_842,
            )],
            Vec::new(),
            vec![
                gpu(
                    "NVIDIA GeForce RTX 4070 Laptop GPU",
                    NVIDIA,
                    false,
                    8_188,
                    6_842,
                ),
                gpu("Intel(R) Graphics (RPL-S)", INTEL, true, 4_236, 4_236),
            ],
        ]);
        assert_eq!(names(&rows), ["NVIDIA GeForce RTX 4070 Laptop GPU"]);
    }

    #[test]
    fn unit__merge_policy__same_vendor_cards_from_one_source_all_stay() {
        let rows = merge_policy(vec![vec![
            gpu("CUDA0", NVIDIA, false, 8_188, 8_188),
            gpu("CUDA1", NVIDIA, false, 8_188, 8_188),
        ]]);
        assert_eq!(names(&rows), ["CUDA0", "CUDA1"]);
    }

    #[test]
    fn unit__merge_policy__integrated_cards_drop_when_a_discrete_card_exists() {
        let rows = merge_policy(vec![
            vec![gpu("sys:pci:0000:0a:00.0", AMD, false, 16_384, 12_000)],
            vec![gpu("Intel(R) Graphics (RPL-S)", INTEL, true, 4_236, 4_236)],
        ]);
        assert_eq!(names(&rows), ["sys:pci:0000:0a:00.0"]);
    }

    #[test]
    fn unit__merge_policy__integrated_only_machine_keeps_its_gpu() {
        let rows = merge_policy(vec![vec![gpu("Apple M2 Pro", 0x106b, true, 16_384, 9_000)]]);
        assert_eq!(names(&rows), ["Apple M2 Pro"]);
    }

    #[test]
    fn unit__merge_policy__row_without_memory_is_dropped_and_does_not_hide_its_vendor() {
        let rows = merge_policy(vec![
            vec![gpu("NVIDIA broken", NVIDIA, false, 0, 0)],
            vec![gpu(
                "NVIDIA GeForce RTX 4070 Laptop GPU",
                NVIDIA,
                false,
                8_188,
                6_842,
            )],
        ]);
        assert_eq!(names(&rows), ["NVIDIA GeForce RTX 4070 Laptop GPU"]);
    }

    #[test]
    fn unit__merge_policy__free_memory_never_exceeds_total() {
        let rows = merge_policy(vec![vec![gpu("oversold", AMD, false, 8_188, 9_000)]]);
        assert_eq!(rows.len(), 1);
        assert_eq!((rows[0].total_mib, rows[0].free_mib), (8_188, 8_188));
    }

    #[test]
    fn unit__census_gpu__maps_to_the_hardware_row_unchanged() {
        let row: GpuInfo = gpu("Card", NVIDIA, false, 8_188, 6_842).into();
        assert_eq!(
            row,
            GpuInfo {
                name: "Card".into(),
                description: "Card".into(),
                total_mib: 8_188,
                free_mib: 6_842,
            }
        );
    }

    #[test]
    fn unit__vendors_of__each_vendor_once_in_row_order() {
        let rows = [
            gpu("CUDA0", NVIDIA, false, 8_188, 8_188),
            gpu("sys:pci:0000:0a:00.0", AMD, false, 16_384, 12_000),
            gpu("CUDA1", NVIDIA, false, 8_188, 8_188),
        ];
        assert_eq!(super::vendors_of(&rows), [NVIDIA, AMD]);
        assert!(super::vendors_of(&[]).is_empty());
    }

    #[test]
    fn unit__has_discrete_gpu__integrated_only_host_is_not_a_discrete_gpu_host() {
        let integrated_only = [gpu("Vulkan0", INTEL, true, 10_256, 4_236)];
        assert!(!super::has_discrete_gpu(&integrated_only));
        assert!(!super::has_discrete_gpu(&[]));

        let with_a_card = [
            gpu("Vulkan0", INTEL, true, 10_256, 4_236),
            gpu("Vulkan1", NVIDIA, false, 8_188, 6_732),
        ];
        assert!(super::has_discrete_gpu(&with_a_card));
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn unit__c_string__stops_at_the_first_nul_and_drops_padding() {
        let mut name = [0u8; 16];
        name[..8].copy_from_slice(b"RTX 4070");
        name[12..].copy_from_slice(b"junk");
        assert_eq!(super::c_string(&name), "RTX 4070");
        assert_eq!(super::c_string(b"  padded  "), "padded");
        assert_eq!(super::c_string(b"no terminator"), "no terminator");
        assert_eq!(super::c_string(&[]), "");
    }
}
