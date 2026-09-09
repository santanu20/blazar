//! Hardware snapshot consumed by the profile compiler. Pure data; the
//! runtime assembles it from sysinfo (CPU/RAM) + engine manifest devices.

#[derive(Debug, Clone, PartialEq, Default)]
pub struct GpuInfo {
    pub name: String,
    pub description: String,
    pub total_mib: u64,
    pub free_mib: u64,
}

/// Vulkan device-description substrings that identify INTEGRATED GPUs
/// (silicon shares system RAM: huge "free" numbers are a fiction and the
/// effective bandwidth is a fraction of a discrete card). Matched
/// case-insensitively against `--list-devices` descriptions. Discrete
/// parts from the same vendors (Intel Arc, Radeon RX) deliberately do
/// NOT match.
const INTEGRATED_PATTERNS: &[&str] = &[
    // Intel iGPU as llama.cpp reports it (verified live: "Intel(R)
    // Graphics (RPL-S)"), plus the Iris Xemarketing spelling.
    "intel(r) graphics",
    "intel(r) iris",
    // AMD mobile APU graphics (Ryzen iGPU); "Radeon RX"/"Radeon PRO"
    // discrete cards stay unmatched.
    "radeon(tm) graphics",
    "amd radeon(tm) graphics",
];

impl GpuInfo {
    /// Integrated-GPU heuristic from the device description. Used by the
    /// auto-pick to prefer discrete cards (bandwidth-bound serving) and
    /// never as a hard exclusion — an integrated-only box still serves.
    #[must_use]
    pub fn is_integrated(&self) -> bool {
        let d = self.description.to_lowercase();
        INTEGRATED_PATTERNS.iter().any(|p| d.contains(p))
    }
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
                GpuInfo {
                    name: "a".into(),
                    description: "NVIDIA CUDA".into(),
                    total_mib: 8_188,
                    free_mib: 7_000,
                },
                GpuInfo {
                    name: "b".into(),
                    description: "Vulkan".into(),
                    total_mib: 4_000,
                    free_mib: 4_000,
                },
            ],
        };
        assert_eq!(hw.total_vram_mib(), 12_188);
        assert!(hw.has_gpu());
        assert_eq!(Hardware::bytes(2) / (1024 * 1024), 2);
        assert_eq!(Hardware::mib(Hardware::bytes(5)), 5);
    }

    #[test]
    fn unit__gpuinfo__integrated_heuristic() {
        let gpu = |desc: &str| GpuInfo {
            name: "Vulkan0".into(),
            description: desc.into(),
            total_mib: 10256,
            free_mib: 10256,
        };
        // Integrated (live-verified Vulkan descriptions).
        for d in [
            "Intel(R) Graphics (RPL-S)",
            "Intel(R) Iris(R) Xe Graphics",
            "AMD Radeon(TM) Graphics",
        ] {
            assert!(gpu(d).is_integrated(), "{d} should read integrated");
        }
        // Discrete (must never match).
        for d in [
            "NVIDIA GeForce RTX 4070 Laptop GPU",
            "Intel(R) Arc(TM) A770 Graphics",
            "AMD Radeon RX 7900 XTX",
            "AMD Radeon PRO W7900",
        ] {
            assert!(!gpu(d).is_integrated(), "{d} should read discrete");
        }
    }
}
