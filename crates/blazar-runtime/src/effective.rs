//! Post-spawn effective-configuration verification.
//!
//! A requested configuration is not proof that an engine accepted or
//! applied it: argv is what Blazar asked for, the child's own runtime
//! surface is what it actually booted with. This module closes that gap
//! for the llama.cpp lane, whose `GET /props` reports the live serving
//! shape (`total_slots`, per-slot `n_ctx` under
//! `default_generation_settings`).
//!
//! Verification policy:
//! - Only argv-pinned values are verified. When the argv carries no
//!   `-np`/`--ctx-size`, the child runs engine-declared defaults (this
//!   build: 4 slots) - that is engine business, not drift, and is never
//!   reported as a mismatch.
//! - `--kv-unified` changes the READING, not the capacity: the shared
//!   pool reports its own size as every slot's `n_ctx` (live-measured
//!   b12578, `-c 32768 -np 2`: child VRAM 956 MiB classic vs 972 MiB
//!   unified - the pool is flat in `-np`), so the effective total is
//!   the per-slot reading itself, never `n_ctx * total_slots`.
//! - `n_ctx` padding is tolerated: llama.cpp rounds the per-slot context
//!   up to KV-cache alignment, so an observed total up to 256 tokens per
//!   slot ABOVE the planned total is rounding, not loss.
//! - Cache-type keys (`-ctk`/`-ctv`) are parsed from the argv but this
//!   build's `/props` exposes no cache-type field, so they are currently
//!   unobservable and never compared (documented, not guessed).
//! - Observability only: mismatches are reported (instance warnings +
//!   tracing), never fatal - a healthy child serving a slightly different
//!   shape beats a rejected spawn.
//!
//! Other lanes (mistral.rs, sglang, vllm, mlx, ...) have no equivalent
//! queryable effective-config surface wired here yet; they skip the probe
//! silently and gain a lane when a surface is certified live.

use blazar_core::Endpoint;

/// How the child lays out its KV cache, read off the spawn argv. The
/// geometry decides how `/props` numbers must be interpreted - the wrong
/// reading turns a correct spawn into a false mismatch (or hides a real
/// one).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum KvGeometry {
    /// Isolated per-slot buffers (upstream default, `--no-kv-unified`):
    /// observed total = `n_ctx * total_slots`.
    #[default]
    Classic,
    /// One shared buffer of `--ctx-size` tokens: every slot's reported
    /// `n_ctx` IS the pool size, so the effective total is the per-slot
    /// reading itself (live-verified b12578 - pool VRAM is flat in
    /// `-np`, it never multiplies).
    Unified,
    /// `--kv-unified-per-slot N` fork geometry: not live-certified, so
    /// the ctx comparison is skipped rather than guessed.
    UnifiedPerSlot,
}

/// What the spawn argv explicitly pinned for the child.
#[derive(Debug, Default, PartialEq)]
pub struct PlannedFromArgv {
    /// `-np` / `--parallel` parallel slot count, when pinned.
    pub slots: Option<u32>,
    /// `--ctx-size` / `-c` TOTAL context tokens across slots, when pinned.
    pub ctx_total: Option<u32>,
    /// `--cache-type-k` / `-ctk`, when pinned (currently unobservable).
    pub cache_k: Option<String>,
    /// `--cache-type-v` / `-ctv`, when pinned (currently unobservable).
    pub cache_v: Option<String>,
    /// KV buffer geometry implied by the argv flags.
    pub kv_geometry: KvGeometry,
}

/// Effective values parsed from the child's `GET /props`.
#[derive(Debug, Default, PartialEq)]
pub struct EffectiveProps {
    /// `total_slots`: concurrent request slots the child actually runs.
    pub total_slots: Option<u32>,
    /// `default_generation_settings.n_ctx`: the PER-SLOT context window.
    /// Under `--kv-unified` this is the shared pool size - every slot
    /// addresses the whole pool, so the value doubles as the effective
    /// total (see [`KvGeometry::Unified`]).
    pub per_slot_ctx: Option<u32>,
}

/// Parse the spawn argv for pinned capacity values. Short and long flag
/// spellings are both honored (`-np`/`--parallel`, `-c`/`--ctx-size`,
/// `-ctk`/`--cache-type-k`, `-ctv`/`--cache-type-v`); `--flag=value`
/// forms are not - the profile compiler always emits space-separated
/// pairs.
#[must_use]
pub fn planned_from_argv(argv: &[String]) -> PlannedFromArgv {
    let flag_value = |long: &str, short: &str| -> Option<String> {
        argv.windows(2)
            .find(|w| w[0] == long || w[0] == short)
            .map(|w| w[1].clone())
    };
    // Exact-token matching keeps `--kv-unified` from colliding with
    // `--kv-unified-per-slot`; the per-slot budget wins the verdict
    // because its pool sizing is the stricter, unverified shape.
    let kv_geometry = if argv.iter().any(|t| t == "--kv-unified-per-slot") {
        KvGeometry::UnifiedPerSlot
    } else if argv.iter().any(|t| t == "--kv-unified") {
        KvGeometry::Unified
    } else {
        KvGeometry::Classic
    };
    PlannedFromArgv {
        slots: flag_value("--parallel", "-np").and_then(|v| v.parse().ok()),
        ctx_total: flag_value("--ctx-size", "-c").and_then(|v| v.parse().ok()),
        cache_k: flag_value("--cache-type-k", "-ctk"),
        cache_v: flag_value("--cache-type-v", "-ctv"),
        kv_geometry,
    }
}

/// Parse the child's `/props` body into the comparable shape. Absent or
/// non-numeric fields stay `None` - a surface that does not answer a
/// field is never treated as a zero or a mismatch.
#[must_use]
pub fn parse_props(body: &serde_json::Value) -> EffectiveProps {
    EffectiveProps {
        total_slots: body
            .get("total_slots")
            .and_then(serde_json::Value::as_u64)
            .and_then(|v| u32::try_from(v).ok()),
        per_slot_ctx: body
            .get("default_generation_settings")
            .and_then(|s| s.get("n_ctx"))
            .and_then(serde_json::Value::as_u64)
            .and_then(|v| u32::try_from(v).ok()),
    }
}

/// Compare planned argv pins against the child's effective props;
/// returns one human-readable line per finding (empty = verified clean).
#[must_use]
pub fn compare(planned: &PlannedFromArgv, observed: &EffectiveProps) -> Vec<String> {
    let mut findings = Vec::new();
    if let (Some(want), Some(got)) = (planned.slots, observed.total_slots)
        && want != got
    {
        findings.push(format!(
            "effective config mismatch: argv pinned {want} slot(s) (-np) but the child reports total_slots {got}"
        ));
    }
    if let (Some(want_total), Some(per_slot), Some(slots)) = (
        planned.ctx_total,
        observed.per_slot_ctx,
        observed.total_slots,
    ) {
        match planned.kv_geometry {
            KvGeometry::UnifiedPerSlot => {
                // Fork geometry without a live certification: issuing a
                // verdict would be a guess, so none is issued (the slot
                // count check above still applies).
            }
            KvGeometry::Classic => {
                // Saturating on purpose: a hostile/huge observed product
                // must not panic the verification path.
                let observed_total = per_slot.saturating_mul(slots);
                let padding_allowance = 256u32.saturating_mul(slots);
                if observed_total < want_total {
                    findings.push(format!(
                        "effective config mismatch: argv pinned --ctx-size {want_total} total but the child serves {observed_total} ({per_slot} x {slots} slots)"
                    ));
                } else if observed_total - want_total > padding_allowance {
                    findings.push(format!(
                        "effective config mismatch: argv pinned --ctx-size {want_total} total but the child serves {observed_total} ({per_slot} x {slots} slots) - far above the request"
                    ));
                }
            }
            KvGeometry::Unified => {
                // One shared pool: every slot reports the pool size, so
                // the per-slot reading IS the effective total. A single
                // pool gets a single alignment window, not one per slot.
                let observed_total = per_slot;
                if observed_total < want_total {
                    findings.push(format!(
                        "effective config mismatch: argv pinned --ctx-size {want_total} total but the shared KV pool serves {observed_total} tokens"
                    ));
                } else if observed_total - want_total > 256 {
                    findings.push(format!(
                        "effective config mismatch: argv pinned --ctx-size {want_total} total but the shared KV pool serves {observed_total} tokens - far above the request"
                    ));
                }
            }
        }
    }
    findings
}

/// Probe a fresh llama.cpp child's `/props` and compare against its argv.
/// Returns the finding lines (empty = verified clean); transport or parse
/// failures return a single `unverified` line so the gap is visible, not
/// silent. Bounded by a short timeout: this runs inline on the spawn path
/// (loopback, typically single-digit milliseconds).
pub async fn probe_llamacpp(
    endpoint: &Endpoint,
    auth: Option<&str>,
    argv: &[String],
) -> Vec<String> {
    let (client, base) = crate::supervisor::child_dial(endpoint);
    let mut get = client
        .get(format!("{base}/props"))
        .timeout(std::time::Duration::from_secs(5));
    if let Some(secret) = auth {
        get = get.bearer_auth(secret);
    }
    let resp = match get.send().await {
        Ok(r) => r,
        Err(e) => {
            return vec![format!(
                "effective config unverified: GET /props failed on the fresh child: {e}"
            )];
        }
    };
    if !resp.status().is_success() {
        return vec![format!(
            "effective config unverified: GET /props answered {} on the fresh child",
            resp.status()
        )];
    }
    let body: serde_json::Value = match resp.json().await {
        Ok(v) => v,
        Err(e) => {
            return vec![format!(
                "effective config unverified: /props body unreadable: {e}"
            )];
        }
    };
    let observed = parse_props(&body);
    let planned = planned_from_argv(argv);
    compare(&planned, &observed)
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    fn argv(pairs: &[&str]) -> Vec<String> {
        pairs.iter().map(|s| (*s).to_string()).collect()
    }

    /// Live-captured shape (llama-server b12578, `-c 4096 -np 2`):
    /// per-slot `n_ctx` 2048, `total_slots` 2. Only the comparable keys are
    /// embedded; the full surface is far larger and irrelevant here.
    fn props_np2() -> serde_json::Value {
        serde_json::json!({
            "total_slots": 2,
            "default_generation_settings": {"n_ctx": 2048},
        })
    }

    #[test]
    fn unit__planned_from_argv__parses_long_and_short_pins() {
        let planned = planned_from_argv(&argv(&[
            "-m", "x.gguf", "-np", "4", "-c", "8192", "-ctk", "q8_0", "-ctv", "q4_0",
        ]));
        assert_eq!(planned.slots, Some(4));
        assert_eq!(planned.ctx_total, Some(8192));
        assert_eq!(planned.cache_k.as_deref(), Some("q8_0"));
        assert_eq!(planned.cache_v.as_deref(), Some("q4_0"));

        let planned = planned_from_argv(&argv(&["--parallel", "2", "--ctx-size", "4096"]));
        assert_eq!(planned.slots, Some(2));
        assert_eq!(planned.ctx_total, Some(4096));
    }

    #[test]
    fn unit__planned_from_argv__absent_pins_stay_none() {
        let planned = planned_from_argv(&argv(&["-m", "x.gguf", "--host", "127.0.0.1"]));
        assert_eq!(planned, PlannedFromArgv::default());
    }

    #[test]
    fn unit__parse_props__reads_live_shape() {
        let observed = parse_props(&props_np2());
        assert_eq!(observed.total_slots, Some(2));
        assert_eq!(observed.per_slot_ctx, Some(2048));
    }

    #[test]
    fn unit__parse_props__absent_fields_stay_none() {
        let observed = parse_props(&serde_json::json!({"model_path": "/x"}));
        assert_eq!(observed, EffectiveProps::default());
    }

    #[test]
    fn unit__compare__clean_match_is_empty() {
        let planned = planned_from_argv(&argv(&["-np", "2", "-c", "4096"]));
        assert!(compare(&planned, &parse_props(&props_np2())).is_empty());
    }

    #[test]
    fn unit__compare__slot_shortfall_reported() {
        // argv asked 4 slots; child runs the engine default 2.
        let planned = planned_from_argv(&argv(&["-np", "4"]));
        let findings = compare(&planned, &parse_props(&props_np2()));
        assert_eq!(findings.len(), 1);
        assert!(findings[0].contains("total_slots 2"));
    }

    #[test]
    fn unit__compare__ctx_loss_reported() {
        // argv asked 8192 total; child serves 4096.
        let planned = planned_from_argv(&argv(&["-c", "8192"]));
        let findings = compare(&planned, &parse_props(&props_np2()));
        assert_eq!(findings.len(), 1);
        assert!(findings[0].contains("2048 x 2 slots"));
    }

    #[test]
    fn unit__compare__kv_padding_tolerated() {
        // Asked 3904 total; child padded per-slot up to 2048 x 2 = 4096
        // (delta 192 <= 256/slot allowance) - rounding, not loss.
        let planned = planned_from_argv(&argv(&["-np", "2", "-c", "3904"]));
        assert!(compare(&planned, &parse_props(&props_np2())).is_empty());
    }

    #[test]
    fn unit__compare__gross_overshoot_reported() {
        let planned = planned_from_argv(&argv(&["-np", "2", "-c", "1024"]));
        let findings = compare(&planned, &parse_props(&props_np2()));
        assert_eq!(findings.len(), 1);
        assert!(findings[0].contains("far above the request"));
    }

    #[test]
    fn unit__compare__engine_default_slots_never_a_mismatch() {
        // No -np in argv, child runs its declared default (live-observed
        // 4 slots on b12578) - nothing was pinned, nothing can drift.
        let planned = planned_from_argv(&argv(&["-m", "x.gguf"]));
        let observed = EffectiveProps {
            total_slots: Some(4),
            per_slot_ctx: Some(4096),
        };
        assert!(compare(&planned, &observed).is_empty());
    }

    #[test]
    fn unit__compare__unified_pool_is_not_slot_multiplied() {
        // The live shape that motivated the geometry split: a daemon
        // child pinned `--ctx-size 32768 -np 2 --kv-unified` and /props
        // reports n_ctx 32768 with total_slots 2. Multiplying would
        // double-count the one shared pool - VRAM measured flat in -np
        // (956 MiB classic vs 972 MiB unified, same argv otherwise).
        let planned =
            planned_from_argv(&argv(&["--ctx-size", "32768", "-np", "2", "--kv-unified"]));
        let observed = EffectiveProps {
            total_slots: Some(2),
            per_slot_ctx: Some(32768),
        };
        assert!(compare(&planned, &observed).is_empty());
    }

    #[test]
    fn unit__compare__unified_pool_undershoot_reported() {
        let planned = planned_from_argv(&argv(&["-c", "65536", "--kv-unified"]));
        let observed = EffectiveProps {
            total_slots: Some(2),
            per_slot_ctx: Some(32768),
        };
        let findings = compare(&planned, &observed);
        assert_eq!(findings.len(), 1);
        assert!(findings[0].contains("shared KV pool serves 32768"));
    }

    #[test]
    fn unit__compare__unified_pool_padding_tolerated() {
        // One pool, one alignment window: 32512 planned vs 32768 pooled
        // (delta 256, not 256 x slots) is rounding, not loss.
        let planned = planned_from_argv(&argv(&["-c", "32512", "--kv-unified"]));
        let observed = EffectiveProps {
            total_slots: Some(2),
            per_slot_ctx: Some(32768),
        };
        assert!(compare(&planned, &observed).is_empty());
    }

    #[test]
    fn unit__compare__unified_per_slot_geometry_skips_ctx_verdict() {
        // Fork geometry without a live certification: no ctx verdict is
        // issued (the slot pin still verifies).
        let planned = planned_from_argv(&argv(&[
            "-c",
            "32768",
            "-np",
            "2",
            "--kv-unified-per-slot",
            "4096",
        ]));
        let observed = EffectiveProps {
            total_slots: Some(2),
            per_slot_ctx: Some(32768),
        };
        assert!(compare(&planned, &observed).is_empty());
    }

    #[test]
    fn unit__planned_from_argv__detects_geometry_from_flags() {
        assert_eq!(
            planned_from_argv(&argv(&["--ctx-size", "8", "--kv-unified"])).kv_geometry,
            KvGeometry::Unified
        );
        assert_eq!(
            planned_from_argv(&argv(&["--ctx-size", "8", "--no-kv-unified"])).kv_geometry,
            KvGeometry::Classic
        );
        assert_eq!(
            planned_from_argv(&argv(&["--ctx-size", "8"])).kv_geometry,
            KvGeometry::Classic
        );
        // Exact-token matching: the per-slot budget wins over plain
        // unified whenever both shapes are even possible.
        assert_eq!(
            planned_from_argv(&argv(&["--kv-unified-per-slot", "4096"])).kv_geometry,
            KvGeometry::UnifiedPerSlot
        );
    }
}
