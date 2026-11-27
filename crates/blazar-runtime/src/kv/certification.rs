//! Pair-wise cross-engine KV interoperability certification.
//!
//! The interchange substrate ships codecs ([`super::interchange::CodecLane`])
//! and a strict compatibility policy ([`super::compatibility`]); this module
//! is the evidence layer on top: one explicit verdict per engine pair, so a
//! reader never has to presume what a conversion implies. Verdicts follow the
//! same discipline as the engine capability matrix: `live_certified` requires
//! a receipt from real engines proving token reuse (a successful file
//! conversion alone proves nothing), `codec_shipped` means the codec landed
//! with byte-identity unit proofs but the live pair certificate is still
//! pending, and `unavailable` pairs name the missing lane.
//!
//! Cross-engine migration stays opt-in behind `[kv.translation] enabled`
//! (default off) until a pair earns `live_certified` with measured prefill
//! savings - the matrix is the tracking surface for that promotion.

use std::fmt::Write as _;

use blazar_core::config::TranslationSection;

use super::interchange::admitted_pairs;

/// Verdict for one source -> target engine pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvCertStatus {
    /// Same-engine round trip through the engine's own restore path
    /// (slot-save bank, fs tier, hicache, prompt-cache file).
    Exact,
    /// Live engines on both sides, token reuse proven, receipt recorded.
    LiveCertified,
    /// Codec landed with byte-identity unit proofs; the live pair
    /// certificate (real engines, proven reuse) is still pending.
    CodecShipped,
    /// No codec lane exists for this pair today.
    Unavailable,
}

impl KvCertStatus {
    /// Stable kebab label used by the committed JSON and doc renders.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::LiveCertified => "live_certified",
            Self::CodecShipped => "codec_shipped",
            Self::Unavailable => "unavailable",
        }
    }
}

/// One row of the certification matrix.
#[derive(Debug, Clone)]
pub struct KvPairCert {
    /// Pair name, `source -> target` (matches [`CodecLane::pair_name`]
    /// for off-diagonal codec pairs).
    pub pair: &'static str,
    pub status: KvCertStatus,
    /// Path to the live-evidence receipt (`live_certified` only).
    pub receipt: Option<&'static str>,
    pub note: &'static str,
}

/// The canonical cross-engine KV certification matrix: all four
/// KV-carrying engines in both directions. Verdicts cite receipts where
/// live evidence exists and name the gap where it does not.
/// A literal 4x4 table; the tests pin it to the codec lanes and
/// the committed JSON.
#[allow(clippy::too_many_lines)] // a data table, not logic
#[must_use]
pub fn kv_certification_matrix() -> &'static [KvPairCert] {
    &[
        // Diagonals: the engine's own save/restore path.
        KvPairCert {
            pair: "llama.cpp -> llama.cpp",
            status: KvCertStatus::Exact,
            receipt: None,
            note: "slot-save bank round trip - the native restore path",
        },
        KvPairCert {
            pair: "vllm -> vllm",
            status: KvCertStatus::Exact,
            receipt: None,
            note: "fs offload tier round trip - content-addressed reload",
        },
        KvPairCert {
            pair: "sglang -> sglang",
            status: KvCertStatus::Exact,
            receipt: None,
            note: "hicache L3 file tier round trip",
        },
        KvPairCert {
            pair: "mlx -> mlx",
            status: KvCertStatus::Exact,
            receipt: None,
            note: "mlx-lm prompt-cache file round trip",
        },
        // Off-diagonal codec pairs.
        KvPairCert {
            pair: "llama.cpp -> vllm",
            status: KvCertStatus::LiveCertified,
            receipt: Some("docs/research/2026-11-19-llamacpp-vllm-kv-import-data-plane.md"),
            note: "gateway chat through the imported prefix reported \
                   cached_tokens > 0 on live engines",
        },
        KvPairCert {
            pair: "vllm -> llama.cpp",
            status: KvCertStatus::LiveCertified,
            receipt: Some("bench-artifacts/2026-11-21-kv-codec-live-certs/README.md"),
            note: "tier -> bank -> native slot restore proven live \
                   (qwen2.5-coder-0.5b pair: 24 blocks x 64 tokens imported; \
                   slot restore of 1536 tokens logged before first use)",
        },
        KvPairCert {
            pair: "mlx -> vllm",
            status: KvCertStatus::CodecShipped,
            receipt: None,
            note: "mlx-lm export codec landed; live pair certificate \
                   needs a Metal host for the mlx lane",
        },
        KvPairCert {
            pair: "vllm -> mlx",
            status: KvCertStatus::CodecShipped,
            receipt: None,
            note: "staged preload landed; live pair certificate needs \
                   a Metal host for the mlx lane",
        },
        KvPairCert {
            pair: "sglang -> vllm",
            status: KvCertStatus::CodecShipped,
            receipt: None,
            note: "hicache L3 page codec landed; live pair certificate \
                   (both engines, proven reuse) pending \
                   - the hicache host-tier guard needs a RAM class with ~15 GiB \
                   headroom (guard refusal verified live on the 13.7 GiB cert box)",
        },
        KvPairCert {
            pair: "vllm -> sglang",
            status: KvCertStatus::CodecShipped,
            receipt: None,
            note: "hicache L3 page codec landed; live pair certificate \
                   (both engines, proven reuse) pending \
                   - the hicache host-tier guard needs a RAM class with ~15 GiB \
                   headroom (guard refusal verified live on the 13.7 GiB cert box)",
        },
        // Pairs with no codec lane at all.
        KvPairCert {
            pair: "llama.cpp -> mlx",
            status: KvCertStatus::Unavailable,
            receipt: None,
            note: "no codec lane exists for this pair today",
        },
        KvPairCert {
            pair: "mlx -> llama.cpp",
            status: KvCertStatus::Unavailable,
            receipt: None,
            note: "no codec lane exists for this pair today",
        },
        KvPairCert {
            pair: "llama.cpp -> sglang",
            status: KvCertStatus::Unavailable,
            receipt: None,
            note: "no codec lane exists for this pair today",
        },
        KvPairCert {
            pair: "sglang -> llama.cpp",
            status: KvCertStatus::Unavailable,
            receipt: None,
            note: "no codec lane exists for this pair today",
        },
        KvPairCert {
            pair: "sglang -> mlx",
            status: KvCertStatus::Unavailable,
            receipt: None,
            note: "no codec lane exists for this pair today",
        },
        KvPairCert {
            pair: "mlx -> sglang",
            status: KvCertStatus::Unavailable,
            receipt: None,
            note: "no codec lane exists for this pair today",
        },
    ]
}

/// Markdown render of the matrix for docs and CLI reports.
#[must_use]
pub fn kv_certification_markdown() -> String {
    let mut out = String::new();
    writeln!(out, "| Pair | Verdict | Notes |").unwrap();
    writeln!(out, "|---|---|---|").unwrap();
    for row in kv_certification_matrix() {
        match row.receipt {
            Some(path) => writeln!(
                out,
                "| {} | [{}]({}) | {} |",
                row.pair,
                row.status.label(),
                path,
                row.note
            )
            .unwrap(),
            None => writeln!(
                out,
                "| {} | {} | {} |",
                row.pair,
                row.status.label(),
                row.note
            )
            .unwrap(),
        }
    }
    out
}

/// The automatic-migration gate: cross-engine admission stays empty under
/// the default config, so no pair fires without the explicit opt-in knob.
#[must_use]
pub fn migration_off_by_default() -> bool {
    let default = TranslationSection::default();
    !default.enabled && admitted_pairs(&default).is_empty()
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]

    use super::super::interchange::CodecLane;
    use super::*;

    const ENGINES: [&str; 4] = ["llama.cpp", "vllm", "sglang", "mlx"];

    #[test]
    fn unit__kv_certification__matrix_covers_every_pair_exactly_once() {
        let rows = kv_certification_matrix();
        assert_eq!(rows.len(), 16, "4x4 engine pairs");
        for src in ENGINES {
            for dst in ENGINES {
                let pair = format!("{src} -> {dst}");
                let hits = rows.iter().filter(|r| r.pair == pair).count();
                assert_eq!(hits, 1, "pair {pair} must appear exactly once");
            }
        }
    }

    #[test]
    fn unit__kv_certification__codec_lanes_carry_non_exact_verdicts() {
        let rows = kv_certification_matrix();
        for lane in [
            CodecLane::LlamaCppToVllm,
            CodecLane::VllmToLlamaCpp,
            CodecLane::MlxToVllm,
            CodecLane::VllmToMlx,
            CodecLane::SglangToVllm,
            CodecLane::VllmToSglang,
        ] {
            let row = rows
                .iter()
                .find(|r| r.pair == lane.pair_name())
                .unwrap_or_else(|| panic!("codec lane {} missing from matrix", lane.pair_name()));
            assert_ne!(row.status, KvCertStatus::Exact);
            assert_ne!(row.status, KvCertStatus::Unavailable);
        }
    }

    #[test]
    fn unit__kv_certification__live_certified_requires_a_receipt() {
        for row in kv_certification_matrix() {
            if row.status == KvCertStatus::LiveCertified {
                assert!(
                    row.receipt.is_some(),
                    "pair {} claims live certification without a receipt",
                    row.pair
                );
            }
        }
    }

    #[test]
    fn unit__kv_certification__migration_stays_off_by_default() {
        // The knob-off posture: the default registry admits no pair, and
        // the compatibility policy reports the gap with the opt-in
        // teaching instead of a silent maybe.
        assert!(!TranslationSection::default().enabled);
        assert_eq!(
            admitted_pairs(&TranslationSection::default()),
            [] as [crate::kv::interchange::CodecLane; 0]
        );
        assert!(migration_off_by_default());
    }

    #[test]
    fn unit__kv_certification__markdown_renders_every_row() {
        let md = kv_certification_markdown();
        assert_eq!(md.lines().count(), 16 + 2, "header + 16 rows");
        assert!(md.contains("[live_certified](docs/research/"));
        assert!(md.contains("| llama.cpp -> llama.cpp | exact |"));
    }
    /// Crash-consistency: the sglang page writer publishes objects
    /// write-aside + rename, and the reader walks the digest chain -
    /// so an interrupted publish can only ever leave the object
    /// ABSENT (a cold page), never partial bytes under a live name.
    #[test]
    fn unit__kv_certification__interrupted_page_publish_never_exposes_partial_cache() {
        use super::super::interchange::{
            SglangPageSpec, carrier_to_sglang_pages, sglang_pages_to_carrier,
        };
        use super::super::repack::RepackGeometry;
        use super::super::translation::{KV_DTYPE_F16, LLAMACPP_LAYOUT};

        let geo = RepackGeometry {
            layers: 2,
            kv_heads: 2,
            head_dim: 4,
        };
        // Two full 64-token pages: 128 cells, 16-byte rows
        // (2 heads x 4 dim x F16).
        let row = (geo.kv_heads * geo.head_dim) as usize * 2;
        let cells = 128u32;
        let layer_bytes = cells as usize * row;
        let carrier = super::super::interchange::KvCarrier {
            tokens: (0..cells).collect(),
            cells,
            geometry: geo,
            dtype: KV_DTYPE_F16,
            block_tokens: 64,
            attention_layout: LLAMACPP_LAYOUT,
            k_rows: vec![0xA5u8; layer_bytes * geo.layers as usize],
            v_rows: vec![0x5Au8; layer_bytes * geo.layers as usize],
        };
        let spec = SglangPageSpec {
            model_name: SglangPageSpec::model_name_of("/models/tiny.d"),
        };
        let dir = tempfile::tempdir().unwrap();

        // 1. Successful publish: digest-named objects only, no .part.
        let report = carrier_to_sglang_pages(&carrier, &spec, dir.path(), 64).unwrap();
        assert_eq!(report.blocks_written, 2);
        let parts: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter(|e| {
                e.as_ref()
                    .unwrap()
                    .path()
                    .extension()
                    .is_some_and(|x| x == "part")
            })
            .collect();
        assert!(
            parts.is_empty(),
            "successful publish must not leave .part files"
        );

        // 2. Round trip: the chain reads back both pages.
        let (back, _) =
            sglang_pages_to_carrier(dir.path(), &spec, &carrier.tokens, &geo, 64).unwrap();
        assert_eq!(back.cells, cells);

        // 3. Interrupted publish of a would-be third page: garbage
        // under a .part name + no digest object. The reader walk must
        // stop at the missing object and never surface the bytes.
        let garbage_path = dir
            .path()
            .join(format!("deadbeef{}.part", spec.config_suffix()));
        std::fs::write(&garbage_path, [0xFFu8; 1024]).unwrap();
        let more_tokens: Vec<u32> = (0..192).collect();
        let (again, _) =
            sglang_pages_to_carrier(dir.path(), &spec, &more_tokens, &geo, 64).unwrap();
        assert_eq!(
            again.cells, cells,
            "the chain must stop at the missing object - partial pages are cold, never read"
        );
    }
}
