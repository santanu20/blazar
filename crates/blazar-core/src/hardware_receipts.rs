//! Which (engine lane, GPU backend, GPU vendor) combinations have run
//! live, and where the artifact of that run lies.
//!
//! Blazar installs vendor-specific upstream builds (CUDA, `ROCm`, SYCL,
//! `OpenVINO`, Vulkan, Metal), but a build that installs is not a build that
//! was measured. A lane without a receipt on a hardware class is experimental
//! there. `blazar doctor` says so for the lanes installed on the host and
//! `docs/engines.md` prints this table. The capability matrix records what
//! a lane can do; this table records where it was seen doing it.
//!
//! A receipt is cited only for an artifact that names the GPU and the
//! backend the run exercised. A row without a receipt is an honest gap,
//! never filled by inference.

/// The accelerator backend an engine build was made for. `OpenVINO` is the
/// odd one: one build drives the CPU, the GPU or the NPU of an Intel host.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum GpuBackend {
    Cuda,
    Rocm,
    Sycl,
    Vulkan,
    Metal,
    OpenVino,
}

impl GpuBackend {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cuda => "cuda",
            Self::Rocm => "rocm",
            Self::Sycl => "sycl",
            Self::Vulkan => "vulkan",
            Self::Metal => "metal",
            Self::OpenVino => "openvino",
        }
    }

    /// The one GPU vendor a backend is tied to. Vulkan is vendor-neutral:
    /// its loader drives every driver installed on the host.
    #[must_use]
    pub const fn implied_vendor(self) -> Option<&'static str> {
        match self {
            Self::Cuda => Some("nvidia"),
            Self::Rocm => Some("amd"),
            Self::Sycl | Self::OpenVino => Some("intel"),
            Self::Metal => Some("apple"),
            Self::Vulkan => None,
        }
    }
}

/// One hardware class of one lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct HardwareReceipt {
    /// Engine lane as the store spells the kind: `llamacpp`, `mlx`, ...
    pub lane: &'static str,
    pub backend: GpuBackend,
    /// GPU vendor name: `nvidia`, `amd`, `intel`, `apple`.
    pub vendor: &'static str,
    /// Repository-relative artifact of a live run on this hardware class.
    pub receipt: Option<&'static str>,
    /// Why a gap stands, when there is something to say.
    pub note: Option<&'static str>,
}

/// Every hardware class this table tracks: the classes with a live run
/// and the vendor classes the roadmap tracks (AMD `ROCm`, Intel SYCL, Intel
/// `OpenVINO`, Apple Metal, Vulkan). A class missing here has no receipt
/// either.
pub const RECEIPTS: &[HardwareReceipt] = &[
    HardwareReceipt {
        lane: "llamacpp",
        backend: GpuBackend::Cuda,
        vendor: "nvidia",
        receipt: Some("bench-artifacts/20261005-kv-quality/benchmark.md"),
        note: None,
    },
    HardwareReceipt {
        lane: "llamacpp",
        backend: GpuBackend::Rocm,
        vendor: "amd",
        receipt: None,
        note: Some(
            "no host with an AMD GPU was at hand; the current release ships ubuntu-rocm-10.0-x64, which carries its runtime, installs on any host and lists no device without an AMD GPU",
        ),
    },
    HardwareReceipt {
        lane: "llamacpp",
        backend: GpuBackend::Sycl,
        vendor: "intel",
        receipt: None,
        note: Some(
            "no host with the Intel oneAPI runtime was at hand; the SYCL prebuilt needs twelve oneAPI libraries it does not ship and does not start without them",
        ),
    },
    HardwareReceipt {
        lane: "llamacpp",
        backend: GpuBackend::Vulkan,
        vendor: "amd",
        receipt: None,
        note: None,
    },
    HardwareReceipt {
        lane: "llamacpp",
        backend: GpuBackend::Vulkan,
        vendor: "intel",
        receipt: None,
        note: Some(
            "runs on an Intel RPL-S iGPU with correct output at 9 to 14 tok/s, against 25 to 44 tok/s on the CPU backend of the same build (bench-artifacts/2026-11-23-sota-roadmap/llamacpp-vulkan-receipt.md); not a performance receipt",
        ),
    },
    HardwareReceipt {
        lane: "llamacpp",
        backend: GpuBackend::Vulkan,
        vendor: "nvidia",
        receipt: Some("bench-artifacts/2026-11-23-sota-roadmap/llamacpp-vulkan-receipt.md"),
        note: None,
    },
    HardwareReceipt {
        lane: "llamacpp",
        backend: GpuBackend::Metal,
        vendor: "apple",
        receipt: None,
        note: None,
    },
    HardwareReceipt {
        lane: "llamacpp",
        backend: GpuBackend::OpenVino,
        vendor: "intel",
        receipt: None,
        note: Some(
            "the build served through its CPU device on a host without an NPU or an Intel GPU at 41.0 tok/s median, against 52.2 tok/s on the CPU backend of the CUDA build; the GPU and NPU devices never ran (bench-artifacts/2026-11-23-sota-roadmap/llamacpp-openvino-receipt.md); not an accelerator receipt",
        ),
    },
    HardwareReceipt {
        lane: "mistralrs",
        backend: GpuBackend::Cuda,
        vendor: "nvidia",
        receipt: None,
        note: Some(
            "a live spawn proof exists (bench-artifacts/20261010-engine-utilization/live_spawn_proofs.md) but names no backend, so it is not counted",
        ),
    },
    HardwareReceipt {
        lane: "mistralrs",
        backend: GpuBackend::Metal,
        vendor: "apple",
        receipt: None,
        note: None,
    },
    HardwareReceipt {
        lane: "mlx",
        backend: GpuBackend::Cuda,
        vendor: "nvidia",
        receipt: Some("bench-artifacts/20261010-engine-utilization/mlx_kv_bits_ab.md"),
        note: None,
    },
    HardwareReceipt {
        lane: "mlx",
        backend: GpuBackend::Metal,
        vendor: "apple",
        receipt: None,
        note: None,
    },
    HardwareReceipt {
        lane: "sdcpp",
        backend: GpuBackend::Vulkan,
        vendor: "nvidia",
        receipt: Some("bench-artifacts/20261103-media-lanes/REPORT.md"),
        note: None,
    },
];

/// The GPU backend an installed engine build was made for, read from its
/// asset label. `None` when the label states none: CPU builds, fork builds
/// (their own capability lanes, a mainstream receipt does not cover them)
/// and pip lanes other than mlx.
#[must_use]
pub fn backend_of(kind: &str, asset: &str, os: &str) -> Option<GpuBackend> {
    if let Some(pin) = asset.strip_prefix("pip:") {
        return match kind {
            "mlx" if pin.contains("mlx[cuda") => Some(GpuBackend::Cuda),
            "mlx" if os == "macos" => Some(GpuBackend::Metal),
            _ => None,
        };
    }
    if asset == "built-fork" {
        return None;
    }
    let asset = asset.to_ascii_lowercase();
    let backend = if asset.contains("cuda") || asset.contains("cublas") {
        GpuBackend::Cuda
    } else if asset.contains("rocm") {
        GpuBackend::Rocm
    } else if asset.contains("sycl") {
        GpuBackend::Sycl
    } else if asset.contains("vulkan") {
        GpuBackend::Vulkan
    } else if asset.contains("metal") || asset.starts_with("macos-") {
        GpuBackend::Metal
    } else if asset.contains("openvino") {
        GpuBackend::OpenVino
    } else {
        return None;
    };
    Some(backend)
}

/// The tracked hardware class of one installed build. `None` when the
/// table has no row for the class, which is an unreceipted class too.
#[must_use]
pub fn lookup(lane: &str, backend: GpuBackend, vendor: &str) -> Option<&'static HardwareReceipt> {
    RECEIPTS
        .iter()
        .find(|row| row.lane == lane && row.backend == backend && row.vendor == vendor)
}

/// The table `docs/engines.md` prints, followed by a line for every row
/// that carries a note. A test keeps the two equal.
#[must_use]
pub fn doc_table() -> String {
    use std::fmt::Write;

    let mut table =
        String::from("| Lane | Backend | GPU vendor | Live receipt |\n|---|---|---|---|\n");
    for row in RECEIPTS {
        let receipt = match row.receipt {
            Some(path) => format!("`{path}`"),
            None => "none (experimental)".to_string(),
        };
        let _ = writeln!(
            table,
            "| {} | {} | {} | {receipt} |",
            row.lane,
            row.backend.as_str(),
            row.vendor
        );
    }
    for row in RECEIPTS.iter().filter(|row| row.note.is_some()) {
        let _ = writeln!(
            table,
            "\n- {} ({}, {}): {}",
            row.lane,
            row.backend.as_str(),
            row.vendor,
            row.note.unwrap_or_default()
        );
    }
    table
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::{GpuBackend, RECEIPTS, backend_of, doc_table, lookup};

    #[test]
    fn unit__backend_of__reads_the_label_of_every_install_shape() {
        let cases = [
            // llama.cpp prebuilts, source builds, macOS builds.
            (
                "llamacpp",
                "ubuntu-cuda-12.8-x64",
                "linux",
                GpuBackend::Cuda,
            ),
            (
                "llamacpp",
                "ubuntu-rocm-6.3.4-x64",
                "linux",
                GpuBackend::Rocm,
            ),
            (
                "llamacpp",
                "ubuntu-sycl-fp16-x64",
                "linux",
                GpuBackend::Sycl,
            ),
            ("llamacpp", "ubuntu-vulkan-x64", "linux", GpuBackend::Vulkan),
            ("llamacpp", "win-cuda-12.4-x64", "windows", GpuBackend::Cuda),
            ("llamacpp", "macos-arm64", "macos", GpuBackend::Metal),
            ("llamacpp", "built-cuda", "linux", GpuBackend::Cuda),
            // bare labels of the other prebuilt lanes
            ("sdcpp", "vulkan", "linux", GpuBackend::Vulkan),
            ("sdcpp", "metal", "macos", GpuBackend::Metal),
            ("whisper", "cublas", "windows", GpuBackend::Cuda),
            ("mistralrs", "cuda132-sm89", "linux", GpuBackend::Cuda),
            ("mistralrs", "metal", "macos", GpuBackend::Metal),
            // the mlx venv names its backend extra in the label
            (
                "mlx",
                "pip:mlx-lm==0.32.2+mlx[cuda12]==0.32.2",
                "linux",
                GpuBackend::Cuda,
            ),
            ("mlx", "pip:mlx-lm==0.32.2", "macos", GpuBackend::Metal),
        ];
        for (kind, asset, os, expected) in cases {
            assert_eq!(
                backend_of(kind, asset, os),
                Some(expected),
                "{kind} {asset}"
            );
        }
    }

    #[test]
    fn unit__backend_of__labels_that_state_no_backend_read_as_none() {
        for (kind, asset, os) in [
            ("llamacpp", "ubuntu-x64", "linux"),
            ("llamacpp", "cpu", "linux"),
            ("llamacpp", "built-cpu", "linux"),
            ("llamacpp", "built-fork", "linux"),
            ("mlx", "pip:mlx-lm==0.32.2", "linux"),
            ("sglang", "pip:sglang==0.5.22", "linux"),
            ("vllm", "pip:vllm==0.34.0", "linux"),
            ("openaihost", "pip:litert-lm==0.19.2", "linux"),
        ] {
            assert_eq!(backend_of(kind, asset, os), None, "{kind} {asset}");
        }
    }

    /// The llama.cpp `OpenVINO` prebuilts, named after the `OpenVINO` release
    /// they bundle. Skipped today, they would install without a word about
    /// being the most experimental accelerator build Blazar can register.
    #[test]
    fn unit__backend_of__reads_the_openvino_build_labels() {
        for (asset, os) in [
            ("ubuntu-openvino-2026.4.1-x64", "linux"),
            ("win-openvino-2026.4.1-x64", "windows"),
        ] {
            assert_eq!(
                backend_of("llamacpp", asset, os),
                Some(GpuBackend::OpenVino),
                "{asset}"
            );
        }
    }

    /// No Intel GPU or NPU has run this build under Blazar: the class is
    /// tracked so doctor can flag it, and it stays a gap.
    #[test]
    fn unit__receipts__openvino_class_is_tracked_and_stays_unreceipted() {
        let openvino = lookup("llamacpp", GpuBackend::OpenVino, "intel")
            .expect("the OpenVINO build is a tracked hardware class");
        assert_eq!(openvino.receipt, None);
    }

    #[test]
    fn unit__lookup__matches_the_exact_class_and_keeps_a_gap_a_gap() {
        let cuda = lookup("llamacpp", GpuBackend::Cuda, "nvidia").expect("a receipted class");
        assert_eq!(
            cuda.receipt,
            Some("bench-artifacts/20261005-kv-quality/benchmark.md")
        );
        let metal = lookup("mlx", GpuBackend::Metal, "apple").expect("a tracked class");
        assert_eq!(metal.receipt, None);
        assert!(lookup("llamacpp", GpuBackend::Cuda, "amd").is_none());
    }

    #[test]
    fn unit__receipts__sorted_and_no_class_twice() {
        let mut sorted = RECEIPTS.to_vec();
        sorted.sort();
        assert_eq!(RECEIPTS, sorted.as_slice());

        let mut classes: Vec<(&str, GpuBackend, &str)> = RECEIPTS
            .iter()
            .map(|row| (row.lane, row.backend, row.vendor))
            .collect();
        let total = classes.len();
        classes.sort();
        classes.dedup();
        assert_eq!(classes.len(), total);
    }

    #[test]
    fn unit__receipts__every_cited_artifact_exists() {
        // The workspace root, two levels above this crate.
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        for row in RECEIPTS {
            if let Some(receipt) = row.receipt {
                let path = root.join(receipt);
                assert!(
                    path.is_file(),
                    "{} is cited as a receipt but does not exist",
                    path.display()
                );
            }
        }
    }

    #[test]
    fn unit__doc_table__engines_doc_prints_the_current_table() {
        let docs = include_str!("../../../docs/engines.md");
        let table = doc_table();
        assert!(
            docs.contains(&table),
            "docs/engines.md does not print the current hardware receipt table:\n{table}"
        );
    }
}
