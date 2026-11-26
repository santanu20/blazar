//! Dedicated GPUs on Linux through sysfs.
//!
//! Only amdgpu publishes the memory numbers a capacity decision needs
//! (`mem_info_vram_total`, `mem_info_vram_used`). The xe, i915, nouveau and
//! NVIDIA kernel drivers expose no VRAM counters, so Intel GPUs on Linux
//! are found by the Vulkan source and NVIDIA cards by NVML.
//!
//! Rows are read from render nodes (`renderD<N>`): the primary `card<N>`
//! node and every connector directory of the same device are other names
//! for it and would be counted again.

use std::fs;
use std::path::{Path, PathBuf};

use blazar_core::Hardware;

use super::CensusGpu;

const SYSFS_DRM: &str = "/sys/class/drm";
const RENDER_NODE_PREFIX: &str = "renderD";
const VENDOR_AMD: u32 = 0x1002;
const VENDOR_NVIDIA: u32 = 0x10de;
const VENDOR_INTEL: u32 = 0x8086;
/// On-package memory below this size is an APU carve-out out of system
/// RAM, not dedicated VRAM. Sizing offload and slots against it would
/// budget a slice of the RAM every other process shares. It is the line
/// ollama draws as well: 4 GiB of VRAM or less reads as integrated.
const MIN_DEDICATED_VRAM_BYTES: u64 = 4 * 1_024 * 1_024 * 1_024;

pub(super) fn gpus() -> Vec<CensusGpu> {
    gpus_in(Path::new(SYSFS_DRM))
}

fn gpus_in(drm: &Path) -> Vec<CensusGpu> {
    render_nodes(drm)
        .iter()
        .map(std::path::PathBuf::as_path)
        .filter_map(read_node)
        .collect()
}

fn render_nodes(drm: &Path) -> Vec<PathBuf> {
    let Ok(entries) = fs::read_dir(drm) else {
        return Vec::new();
    };
    let mut nodes: Vec<PathBuf> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(is_render_node)
        })
        .collect();
    nodes.sort();
    nodes
}

/// Render nodes only: `card1` and every `card1-<connector>` directory are
/// the same devices under other names.
fn is_render_node(name: &str) -> bool {
    name.strip_prefix(RENDER_NODE_PREFIX)
        .is_some_and(|minor| !minor.is_empty() && minor.bytes().all(|b| b.is_ascii_digit()))
}

fn read_node(node: &Path) -> Option<CensusGpu> {
    let device = node.join("device");
    let total_mib = Hardware::mib(read_number(&device.join("mem_info_vram_total"))?);
    if total_mib < Hardware::mib(MIN_DEDICATED_VRAM_BYTES) {
        return None;
    }
    let used_mib = read_number(&device.join("mem_info_vram_used"))
        .map_or(0, |used| Hardware::mib(used).min(total_mib));
    let vendor = u32::try_from(read_number(&device.join("vendor"))?).ok()?;
    let bdf = device
        .canonicalize()
        .ok()?
        .file_name()?
        .to_str()?
        .to_string();
    let driver = fs::read_link(device.join("driver"))
        .ok()?
        .file_name()?
        .to_str()?
        .to_string();
    Some(CensusGpu {
        name: format!("sys:pci:{bdf}"),
        description: format!(
            "{} graphics adapter ({driver}, pci {bdf})",
            vendor_name(vendor)
        ),
        total_mib,
        free_mib: total_mib - used_mib,
        vendor_id: vendor,
        integrated: false,
    })
}

/// A sysfs attribute is one number: `0x1002` for PCI ids, plain decimal
/// byte counts for the memory files.
fn read_number(path: &Path) -> Option<u64> {
    parse_sysfs_number(&fs::read_to_string(path).ok()?)
}

fn parse_sysfs_number(text: &str) -> Option<u64> {
    let text = text.trim();
    match text.strip_prefix("0x") {
        Some(hex) => u64::from_str_radix(hex, 16).ok(),
        None => text.parse().ok(),
    }
}

fn vendor_name(vendor: u32) -> &'static str {
    match vendor {
        VENDOR_AMD => "AMD",
        VENDOR_NVIDIA => "NVIDIA",
        VENDOR_INTEL => "Intel",
        _ => "Unknown",
    }
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use std::fs;
    use std::os::unix::fs::symlink;
    use std::path::Path;

    use super::{gpus_in, is_render_node, parse_sysfs_number};

    const GIB: u64 = 1_024 * 1_024 * 1_024;

    /// A DRM class directory holding one render node whose `device` is a
    /// PCI device with amdgpu's memory counters. Returns the directory
    /// holding the PCI device so a test can add or remove its files.
    fn drm_tree(root: &Path, vram_total: Option<&str>) -> (std::path::PathBuf, std::path::PathBuf) {
        let pci_device = root.join("pci/0000:0a:00.0");
        fs::create_dir_all(&pci_device).unwrap();
        fs::create_dir_all(root.join("drivers/amdgpu")).unwrap();
        let node = root.join("drm/renderD128");
        fs::create_dir_all(&node).unwrap();
        symlink(&pci_device, node.join("device")).unwrap();
        symlink(root.join("drivers/amdgpu"), pci_device.join("driver")).unwrap();
        fs::write(pci_device.join("vendor"), "0x1002\n").unwrap();
        if let Some(total) = vram_total {
            fs::write(pci_device.join("mem_info_vram_total"), total).unwrap();
        }
        (root.join("drm"), pci_device)
    }

    #[test]
    fn unit__linux_drm_gpus_in__amd_card_with_memory_counters_is_listed() {
        let tmp = tempfile::tempdir().unwrap();
        let (drm, pci_device) = drm_tree(tmp.path(), Some(&(16 * GIB).to_string()));
        fs::write(pci_device.join("mem_info_vram_used"), (GIB).to_string()).unwrap();
        let rows = gpus_in(&drm);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].name, "sys:pci:0000:0a:00.0");
        assert_eq!(
            rows[0].description,
            "AMD graphics adapter (amdgpu, pci 0000:0a:00.0)"
        );
        assert_eq!((rows[0].total_mib, rows[0].free_mib), (16_384, 15_360));
        assert_eq!(rows[0].vendor_id, 0x1002);
        assert!(!rows[0].integrated);
    }

    #[test]
    fn unit__linux_drm_gpus_in__small_carveout_is_not_dedicated_memory() {
        let tmp = tempfile::tempdir().unwrap();
        let (drm, _) = drm_tree(tmp.path(), Some(&(2 * GIB).to_string()));
        assert!(gpus_in(&drm).is_empty());
    }

    #[test]
    fn unit__linux_drm_gpus_in__missing_total_counter_is_dropped() {
        let tmp = tempfile::tempdir().unwrap();
        let (drm, _) = drm_tree(tmp.path(), None);
        assert!(gpus_in(&drm).is_empty());
    }

    #[test]
    fn unit__linux_drm_gpus_in__missing_used_counter_reports_all_memory_free() {
        let tmp = tempfile::tempdir().unwrap();
        let (drm, _) = drm_tree(tmp.path(), Some(&(8 * GIB).to_string()));
        let rows = gpus_in(&drm);
        assert_eq!(rows.len(), 1);
        assert_eq!((rows[0].total_mib, rows[0].free_mib), (8_192, 8_192));
    }

    #[test]
    fn unit__linux_drm_gpus_in__card_and_connector_nodes_are_not_counted() {
        let tmp = tempfile::tempdir().unwrap();
        let (drm, pci_device) = drm_tree(tmp.path(), Some(&(16 * GIB).to_string()));
        for other_name in ["card1", "card1-eDP-1"] {
            fs::create_dir_all(drm.join(other_name)).unwrap();
            symlink(&pci_device, drm.join(other_name).join("device")).unwrap();
        }
        let rows = gpus_in(&drm);
        assert_eq!(
            rows.len(),
            1,
            "the render node is the only counted name of the device"
        );
    }

    #[test]
    fn unit__linux_drm_gpus_in__missing_drm_class_directory_is_empty() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(gpus_in(&tmp.path().join("drm")).is_empty());
    }

    #[test]
    fn unit__is_render_node__accepts_render_nodes_with_numeric_minors_only() {
        for name in ["renderD128", "renderD0"] {
            assert!(is_render_node(name), "{name} is a render node");
        }
        for name in ["card1", "renderD", "renderD12a", "renderD1-eDP-1", "render"] {
            assert!(!is_render_node(name), "{name} is not a render node");
        }
    }

    #[test]
    fn unit__parse_sysfs_number__reads_hex_ids_and_decimal_byte_counts() {
        assert_eq!(parse_sysfs_number("0x1002\n"), Some(0x1002));
        assert_eq!(parse_sysfs_number(" 17179869184\n"), Some(17_179_869_184));
        assert_eq!(parse_sysfs_number("abc"), None);
        assert_eq!(parse_sysfs_number(""), None);
    }
}
