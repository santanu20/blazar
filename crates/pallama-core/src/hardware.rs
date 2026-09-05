//! Hardware snapshot consumed by the profile compiler. Pure data; the
//! runtime assembles it from sysinfo (CPU/RAM) + engine manifest devices.

#[derive(Debug, Clone, PartialEq, Default)]
pub struct GpuInfo {
    pub name: String,
    pub description: String,
    pub total_mib: u64,
    pub free_mib: u64,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Hardware {
    pub physical_cores: u32,
    pub total_ram_mib: u64,
    pub gpus: Vec<GpuInfo>,
}

impl Hardware {
    #[must_use] 
    pub fn total_vram_mib(&self) -> u64 {
        self.gpus.iter().map(|g| g.total_mib).sum()
    }

    #[must_use] 
    pub fn has_gpu(&self) -> bool {
        !self.gpus.is_empty()
    }

    #[must_use] 
    pub fn mib(bytes: u64) -> u64 {
        bytes / (1024 * 1024)
    }

    #[must_use] 
    pub fn bytes(mib: u64) -> u64 {
        mib.saturating_mul(1024 * 1024)
    }
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    #[test]
    fn unit__hardware__vram_sums_across_gpus() {
        let hw = Hardware {
            physical_cores: 8,
            total_ram_mib: 32_000,
            gpus: vec![
                GpuInfo { name: "a".into(), description: "NVIDIA CUDA".into(), total_mib: 8_188, free_mib: 7_000 },
                GpuInfo { name: "b".into(), description: "Vulkan".into(), total_mib: 4_000, free_mib: 4_000 },
            ],
        };
        assert_eq!(hw.total_vram_mib(), 12_188);
        assert!(hw.has_gpu());
        assert_eq!(Hardware::bytes(2) / (1024 * 1024), 2);
        assert_eq!(Hardware::mib(Hardware::bytes(5)), 5);
    }
}
