//! Hardware assembly: sysinfo (CPU/RAM) + engine manifest devices.

/// Live `MemAvailable` (MiB). The spawn-time memory guard: below a hard
/// floor the mmap streaming engine will thrash swap for minutes — fail
/// the load with a named error instead (the validate.py heuristic,
/// tightened to a zero-false-positive floor).
#[must_use]
pub fn mem_available_mib() -> u64 {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    sys.available_memory() / (1024 * 1024)
}

use pallama_core::{GpuInfo, Hardware};

use crate::engine::manifest::Manifest;

#[must_use]
pub fn probe_hardware(manifest: Option<&Manifest>) -> Hardware {
    let mut sys = sysinfo::System::new();
    sys.refresh_cpu_usage();
    sys.refresh_memory();
    let physical_cores = sys
        .physical_core_count()
        .map_or(1, |c| u32::try_from(c).unwrap_or(1))
        .max(1);
    let total_ram_mib = sys.total_memory() / (1024 * 1024); // sysinfo returns bytes
    let gpus = manifest
        .map(|m| {
            m.devices
                .iter()
                .map(|d| GpuInfo {
                    name: d.name.clone(),
                    description: d.description.clone(),
                    total_mib: d.total_mib,
                    free_mib: d.free_mib,
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    Hardware {
        physical_cores,
        total_ram_mib,
        gpus,
    }
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    #[test]
    fn unit__probe_hardware__merges_sysinfo_and_devices() {
        let m = Manifest {
            tag: "t".into(),
            build_number: 1,
            version_raw: "version: 1".into(),
            devices: vec![crate::engine::manifest::DeviceDesc {
                name: "RTX".into(),
                description: "NVIDIA CUDA".into(),
                total_mib: 8_188,
                free_mib: 7_000,
            }],
            flags: std::collections::BTreeSet::default(),
            spec_types: vec![],
            server_path: "/x".into(),
        };
        let hw = probe_hardware(Some(&m));
        assert!(hw.physical_cores >= 1);
        assert!(hw.total_ram_mib > 0);
        assert_eq!(hw.gpus.len(), 1);
        assert_eq!(hw.total_vram_mib(), 8_188);
        let cpu_only = probe_hardware(None);
        assert!(cpu_only.gpus.is_empty());
    }
}
