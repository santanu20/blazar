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
//! Other lanes:
//! - sglang exposes its resolved startup configuration through
//!   `GET /server_info` (0.5.22+; the legacy alias `/get_server_info`
//!   still answers with a deprecation warning). The body carries the
//!   RESOLVED server args - capacity pins (`max_running_requests`,
//!   `context_length`, `mem_fraction_static`) land top-level or inside
//!   a nested arg record depending on build; both shapes are searched,
//!   never deeper (per-DP `internal_states` is runtime state, not
//!   config). One derivation is known-legitimate: some attention
//!   configs scale `mem_fraction_static` by 0.85 at startup, so a
//!   differing fraction is reported with that possibility named rather
//!   than as silent drift.
//! - The vllm lane exposes no certified capacity introspection endpoint
//!   endpoint; the lane probes `/v1/models` for liveness and reports an
//!   honest `unverified` line instead of guessing capacity.
//! - Remaining lanes (mistral.rs, mlx, ...) have no equivalent queryable
//!   effective-config surface wired here yet; they skip the probe
//!   silently and gain a lane when a surface is certified live.

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

/// What the sglang spawn argv explicitly pinned for the child.
#[derive(Debug, Default, PartialEq)]
pub struct PlannedSglang {
    /// `--max-running-requests`: concurrent request capacity, when pinned.
    pub max_running_requests: Option<u32>,
    /// `--context-length`: model context window override, when pinned.
    pub context_length: Option<u32>,
    /// `--mem-fraction-static`: static memory fraction, when pinned.
    pub mem_fraction_static: Option<f64>,
    /// `--enable-paged-experts`: routed experts paged from pinned host
    /// memory, when the flag is present.
    pub paged_experts: bool,
    /// `--paged-experts-num-resident`: experts kept resident on the GPU,
    /// when paged.
    pub paged_num_resident: Option<u64>,
}

/// Effective values parsed from sglang's `/server_info` body.
#[derive(Debug, Default, PartialEq)]
pub struct EffectiveSglang {
    pub max_running_requests: Option<u32>,
    pub context_length: Option<u32>,
    pub mem_fraction_static: Option<f64>,
}

/// Parse the sglang spawn argv for pinned capacity values (long flags,
/// space-separated pairs - the profile compiler's emission form).
#[must_use]
pub fn planned_from_argv_sglang(argv: &[String]) -> PlannedSglang {
    let flag_value = |long: &str| -> Option<String> {
        argv.windows(2).find(|w| w[0] == long).map(|w| w[1].clone())
    };
    PlannedSglang {
        max_running_requests: flag_value("--max-running-requests").and_then(|v| v.parse().ok()),
        context_length: flag_value("--context-length").and_then(|v| v.parse().ok()),
        mem_fraction_static: flag_value("--mem-fraction-static").and_then(|v| v.parse().ok()),
        paged_experts: argv.iter().any(|a| a == "--enable-paged-experts"),
        paged_num_resident: flag_value("--paged-experts-num-resident").and_then(|v| v.parse().ok()),
    }
}

/// Look a numeric key up in a resolved-server-args body: top level
/// first, then inside object-valued children one nesting level deep
/// (newer builds group args into nested records). Bounded to depth one
/// on purpose - `internal_states` entries are per-DP runtime state,
/// not configuration, and must never answer a config lookup.
fn resolved_u64(body: &serde_json::Value, key: &str) -> Option<u64> {
    let at = |v: &serde_json::Value| v.get(key).and_then(serde_json::Value::as_u64);
    at(body).or_else(|| {
        body.as_object()?
            .values()
            .filter(|v| v.is_object())
            .find_map(at)
    })
}

/// Float sibling of [`resolved_u64`] (same depth bound).
fn resolved_f64(body: &serde_json::Value, key: &str) -> Option<f64> {
    let at = |v: &serde_json::Value| v.get(key).and_then(serde_json::Value::as_f64);
    at(body).or_else(|| {
        body.as_object()?
            .values()
            .filter(|v| v.is_object())
            .find_map(at)
    })
}

/// Parse the `/server_info` body into the comparable shape. Absent or
/// non-numeric fields stay `None` - never treated as zero or mismatch.
#[must_use]
pub fn parse_server_info(body: &serde_json::Value) -> EffectiveSglang {
    EffectiveSglang {
        max_running_requests: resolved_u64(body, "max_running_requests")
            .and_then(|v| u32::try_from(v).ok()),
        context_length: resolved_u64(body, "context_length").and_then(|v| u32::try_from(v).ok()),
        mem_fraction_static: resolved_f64(body, "mem_fraction_static"),
    }
}

/// Compare planned sglang argv pins against the resolved server args;
/// one human-readable line per finding (empty = verified clean).
#[must_use]
pub fn compare_sglang(planned: &PlannedSglang, observed: &EffectiveSglang) -> Vec<String> {
    let mut findings = Vec::new();
    if let (Some(want), Some(got)) = (planned.max_running_requests, observed.max_running_requests)
        && want != got
    {
        findings.push(format!(
            "effective config mismatch: argv pinned --max-running-requests {want} but the server resolves {got}"
        ));
    }
    if let (Some(want), Some(got)) = (planned.context_length, observed.context_length)
        && want != got
    {
        findings.push(format!(
            "effective config mismatch: argv pinned --context-length {want} but the server resolves {got}"
        ));
    }
    if let (Some(want), Some(got)) = (planned.mem_fraction_static, observed.mem_fraction_static)
        && (want - got).abs() > 1e-6
    {
        // Known-legitimate derivation: some attention configs scale the
        // fraction by 0.85 at startup - the line names it instead of
        // crying drift, and stays warn-only either way.
        findings.push(format!(
            "effective config mismatch: argv pinned --mem-fraction-static {want} but the server resolves {got:.4} (sglang may derive a scaled fraction in some attention configs - see its startup logs)"
        ));
    }
    findings
}

/// Probe a fresh sglang child's resolved startup configuration and
/// compare against its argv. `GET /server_info` is the canonical route
/// (0.5.22+); the legacy `/get_server_info` alias is the fallback for
/// older installs. Bounded by a short timeout: this runs inline on the
/// spawn path (loopback). Transport or parse failures return a single
/// `unverified` line so the gap is visible, never silent.
pub async fn probe_sglang(endpoint: &Endpoint, auth: Option<&str>, argv: &[String]) -> Vec<String> {
    let (client, base) = crate::supervisor::child_dial(endpoint);
    let mut body: Option<serde_json::Value> = None;
    for path in ["/server_info", "/get_server_info"] {
        let mut get = client
            .get(format!("{base}{path}"))
            .timeout(std::time::Duration::from_secs(5));
        if let Some(secret) = auth {
            get = get.bearer_auth(secret);
        }
        match get.send().await {
            Ok(r) if r.status().is_success() => match r.json().await {
                Ok(v) => {
                    body = Some(v);
                    break;
                }
                Err(e) => {
                    return vec![format!(
                        "effective config unverified: {path} body unreadable: {e}"
                    )];
                }
            },
            // Canonical route absent on an older build: fall through to
            // the legacy alias on the next loop iteration.
            Ok(_) if path == "/server_info" => {}
            Ok(r) => {
                return vec![format!(
                    "effective config unverified: GET /get_server_info answered {} on the fresh child",
                    r.status()
                )];
            }
            Err(e) => {
                return vec![format!(
                    "effective config unverified: server-info probe failed on the fresh child: {e}"
                )];
            }
        }
    }
    let Some(body) = body else {
        return vec![
            "effective config unverified: no server-info route answered on the fresh child"
                .to_string(),
        ];
    };
    compare_sglang(&planned_from_argv_sglang(argv), &parse_server_info(&body))
}

/// Planned speculative-decoding facts pinned in a fresh vLLM child's
/// argv. vLLM exposes no certified introspection surface to verify
/// them against, so these stay planned-only facts - the liveness
/// probe reports them beside its unverified line, never as a
/// verified effective config.
pub struct PlannedVllm {
    /// `--spec-method`: speculative family, when pinned.
    pub spec_method: Option<String>,
    /// `--spec-tokens`: proposals per step, when pinned.
    pub spec_num_tokens: Option<u64>,
}

/// Parse the spec-decode pins out of a vLLM child's argv (planned
/// facts only - see [`PlannedVllm`]).
#[must_use]
pub fn planned_from_argv_vllm(argv: &[String]) -> PlannedVllm {
    let flag_value = |long: &str| -> Option<String> {
        argv.windows(2).find(|w| w[0] == long).map(|w| w[1].clone())
    };
    PlannedVllm {
        spec_method: flag_value("--spec-method"),
        spec_num_tokens: flag_value("--spec-tokens").and_then(|v| v.parse().ok()),
    }
}

/// Probe a fresh vLLM child. vLLM's OpenAI-compatible surface exposes
/// no certified capacity introspection endpoint, so no argv pin can be
/// verified; the liveness probe distinguishes "surface absent" (line
/// below) from "probe never ran", and capacity stays honestly
/// unverified rather than guessed from argv.
pub async fn probe_vllm(endpoint: &Endpoint, auth: Option<&str>, argv: &[String]) -> Vec<String> {
    let (client, base) = crate::supervisor::child_dial(endpoint);
    let mut get = client
        .get(format!("{base}/v1/models"))
        .timeout(std::time::Duration::from_secs(5));
    if let Some(secret) = auth {
        get = get.bearer_auth(secret);
    }
    match get.send().await {
        Ok(r) if r.status().is_success() => {
            let mut lines = vec![
                "effective config unverified: vllm exposes no capacity introspection surface (liveness confirmed via /v1/models)".to_string(),
            ];
            // Planned spec-decode pins ride the same honest line:
            // reported, never dressed up as a verified config.
            let planned = planned_from_argv_vllm(argv);
            if let Some(method) = &planned.spec_method {
                let window = planned
                    .spec_num_tokens
                    .map_or_else(|| "the engine default".to_string(), |k| k.to_string());
                lines.push(format!(
                    "planned spec decode: {method} with {window} speculative tokens per step (unverified)"
                ));
            }
            lines
        }
        Ok(r) => vec![format!(
            "effective config unverified: GET /v1/models answered {} on the fresh child - capacity and liveness both unconfirmed",
            r.status()
        )],
        Err(e) => vec![format!(
            "effective config unverified: /v1/models probe failed on the fresh child: {e}"
        )],
    }
}

/// Lane dispatch for post-spawn effective-config verification. Lanes
/// without a certified introspection surface return no lines today;
/// each gained surface wires in here.
pub async fn probe(
    kind: blazar_core::engine_kind::EngineKind,
    endpoint: &Endpoint,
    auth: Option<&str>,
    argv: &[String],
) -> Vec<String> {
    use blazar_core::engine_kind::EngineKind;
    match kind {
        EngineKind::LlamaCpp => probe_llamacpp(endpoint, auth, argv).await,
        EngineKind::Sglang => probe_sglang(endpoint, auth, argv).await,
        EngineKind::Vllm => probe_vllm(endpoint, auth, argv).await,
        _ => Vec::new(),
    }
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

    /// sglang `/server_info` shape: resolved server args land top-level
    /// (the `result.update(scheduler_info)` merge) or inside a nested
    /// arg record depending on build - field names verified against the
    /// installed 0.5.22 package source AND against a live capture
    /// (2026-11-22, sglang 0.5.22 child serving qwen3-1.7b: top-level
    /// `context_length` 16384 + `mem_fraction_static` 0.755 matching the
    /// pinned argv, `max_running_requests` null when unpinned; receipt
    /// bench-artifacts/20261122-w7-live/).
    fn server_info_shape() -> serde_json::Value {
        serde_json::json!({
            "max_running_requests": 4,
            "context_length": 8192,
            "mem_fraction_static": 0.82,
            "version": "0.5.22",
            "launch_command": "python -m sglang.launch_server ...",
            "max_total_num_tokens": 22026,
            "internal_states": [{"dp_rank": 0, "load_generation": 1}],
        })
    }

    /// Live-captured unpinned shape (sglang 0.5.22, 2026-11-22): absent
    /// pins are null at top level, never zero - the parser must read
    /// them as "engine default" and never mismatch.
    #[test]
    fn unit__parse_server_info__live_unpinned_shape_all_nulls() {
        let observed = parse_server_info(&serde_json::json!({
            "context_length": 16384,
            "mem_fraction_static": 0.755,
            "max_running_requests": serde_json::Value::Null,
            "version": "0.5.22",
            "max_total_num_tokens": 22026,
            "launch_command": "--model-path ... --context-length 16384 --mem-fraction-static 0.755",
        }));
        assert_eq!(observed.max_running_requests, None);
        assert_eq!(observed.context_length, Some(16384));
        assert_eq!(observed.mem_fraction_static, Some(0.755));
        // The live argv pinned context-length + mem-fraction only.
        let planned = planned_from_argv_sglang(&argv(&[
            "--model-path",
            "/x",
            "--context-length",
            "16384",
            "--mem-fraction-static",
            "0.755",
        ]));
        assert!(
            compare_sglang(&planned, &observed).is_empty(),
            "live shape must verify clean"
        );
    }

    #[test]
    fn unit__planned_from_argv_sglang__parses_capacity_pins() {
        let planned = planned_from_argv_sglang(&argv(&[
            "--model-path",
            "/x",
            "--max-running-requests",
            "8",
            "--context-length",
            "16384",
            "--mem-fraction-static",
            "0.87",
        ]));
        assert_eq!(planned.max_running_requests, Some(8));
        assert_eq!(planned.context_length, Some(16384));
        assert_eq!(planned.mem_fraction_static, Some(0.87));

        let planned = planned_from_argv_sglang(&argv(&["--model-path", "/x"]));
        assert_eq!(planned, PlannedSglang::default());
    }

    #[test]
    fn unit__planned_from_argv_vllm__parses_spec_pins() {
        let planned =
            planned_from_argv_vllm(&argv(&["--spec-method", "suffix", "--spec-tokens", "4"]));
        assert_eq!(planned.spec_method.as_deref(), Some("suffix"));
        assert_eq!(planned.spec_num_tokens, Some(4));
        // Bare method: the window stays unset, never guessed.
        let planned = planned_from_argv_vllm(&argv(&["--spec-method", "ngram"]));
        assert_eq!(planned.spec_method.as_deref(), Some("ngram"));
        assert_eq!(planned.spec_num_tokens, None);
        // No spec flags: nothing parsed.
        let planned = planned_from_argv_vllm(&argv(&["--max-model-len", "4096"]));
        assert!(planned.spec_method.is_none() && planned.spec_num_tokens.is_none());
    }

    #[test]
    fn unit__planned_from_argv_sglang__parses_paged_expert_pins() {
        let planned = planned_from_argv_sglang(&argv(&[
            "--model-path",
            "/x",
            "--enable-paged-experts",
            "--paged-experts-num-resident",
            "12",
        ]));
        assert!(planned.paged_experts, "the bare flag marks paging planned");
        assert_eq!(planned.paged_num_resident, Some(12));

        // A bare flag with no K must not invent a resident count, and a
        // plan without the flag reports paging off with no K.
        let planned = planned_from_argv_sglang(&argv(&["--enable-paged-experts"]));
        assert!(planned.paged_experts);
        assert_eq!(planned.paged_num_resident, None);
        let planned = planned_from_argv_sglang(&argv(&["--mem-fraction-static", "0.87"]));
        assert!(!planned.paged_experts);
        assert_eq!(planned.paged_num_resident, None);
    }

    #[test]
    fn unit__parse_server_info__reads_top_level_shape() {
        let observed = parse_server_info(&server_info_shape());
        assert_eq!(observed.max_running_requests, Some(4));
        assert_eq!(observed.context_length, Some(8192));
        assert!((observed.mem_fraction_static.unwrap() - 0.82).abs() < 1e-9);
    }

    #[test]
    fn unit__parse_server_info__nested_arg_record_searched_one_level() {
        // Newer builds group resolved args into nested records; the
        // lookup must find them there without diving into
        // internal_states (per-DP runtime state, not config).
        let body = serde_json::json!({
            "schedule": {"max_running_requests": 2, "context_length": 4096},
            "memory": {"mem_fraction_static": 0.9},
            "internal_states": [{"max_running_requests": 999}],
        });
        let observed = parse_server_info(&body);
        assert_eq!(observed.max_running_requests, Some(2));
        assert_eq!(observed.context_length, Some(4096));
        assert!((observed.mem_fraction_static.unwrap() - 0.9).abs() < 1e-9);
    }

    #[test]
    fn unit__parse_server_info__absent_fields_stay_none() {
        let observed = parse_server_info(&serde_json::json!({"version": "0.4.1"}));
        assert_eq!(observed, EffectiveSglang::default());
    }

    #[test]
    fn unit__compare_sglang__clean_match_is_empty() {
        let planned = planned_from_argv_sglang(&argv(&[
            "--max-running-requests",
            "4",
            "--context-length",
            "8192",
            "--mem-fraction-static",
            "0.82",
        ]));
        assert!(compare_sglang(&planned, &parse_server_info(&server_info_shape())).is_empty());
    }

    #[test]
    fn unit__compare_sglang__unpinned_capacity_never_a_mismatch() {
        // Nothing pinned, engine derived its own capacity: engine
        // business, not drift (same policy as the llamacpp defaults).
        let planned = planned_from_argv_sglang(&argv(&["--model-path", "/x"]));
        assert!(compare_sglang(&planned, &parse_server_info(&server_info_shape())).is_empty());
    }

    #[test]
    fn unit__compare_sglang__slot_and_ctx_drift_reported() {
        let planned = planned_from_argv_sglang(&argv(&[
            "--max-running-requests",
            "8",
            "--context-length",
            "16384",
        ]));
        let findings = compare_sglang(&planned, &parse_server_info(&server_info_shape()));
        assert_eq!(findings.len(), 2);
        assert!(
            findings
                .iter()
                .any(|f| f.contains("max-running-requests 8"))
        );
        assert!(findings.iter().any(|f| f.contains("context-length 16384")));
    }

    #[test]
    fn unit__compare_sglang__derived_mem_fraction_names_the_derivation() {
        // Attention configs may scale the pinned fraction by 0.85 - the
        // line must name that possibility, not report bare drift.
        let planned = planned_from_argv_sglang(&argv(&["--mem-fraction-static", "0.87"]));
        let observed = EffectiveSglang {
            mem_fraction_static: Some(0.87 * 0.85),
            ..EffectiveSglang::default()
        };
        let findings = compare_sglang(&planned, &observed);
        assert_eq!(findings.len(), 1);
        assert!(findings[0].contains("may derive a scaled fraction"));
    }
}
