//! Co-residency planner: which local models can stay loaded together in
//! a VRAM budget (weights + f16 KV at each model's effective ctx).
//! Greedy: hot models first (they survive capacity cycles anyway), then
//! smallest — a plan, not a scheduler; the supervisor still owns eviction.

/// f16 KV (MiB) for a GGUF at a ctx — shared by the planner and the
/// gateway's `num_ctx` preflight. Geometry-strict when the GGUF carries
/// attention fields (`GgufMeta::kv_f16_bytes`, MLA/SWA-aware); lossy
/// defaults (1 layer/head, 128 dim) only for metadata-poor files, keeping
/// this a coarse co-residency heuristic rather than a hard gate.
#[must_use]
pub fn kv_f16_mib(meta: &crate::GgufMeta, ctx: u64) -> u64 {
    if let Some(bytes) = meta.kv_f16_bytes(ctx) {
        return bytes / (1024 * 1024);
    }
    let layers = meta.block_count.unwrap_or(0).max(1);
    let kv_heads = meta
        .head_count_kv
        .unwrap_or(meta.head_count.unwrap_or(1))
        .max(1);
    let head_dim = meta.derived_head_dim().unwrap_or(128);
    2u64.saturating_mul(layers)
        .saturating_mul(kv_heads)
        .saturating_mul(head_dim)
        .saturating_mul(ctx)
        .saturating_mul(2)
        / (1024 * 1024)
}

/// One model's footprint for planning.
#[derive(Debug, Clone, PartialEq)]
pub struct Footprint {
    pub name: String,
    /// Model weights (MiB).
    pub weights_mib: u64,
    /// f16 KV at `ctx` (MiB).
    pub kv_mib: u64,
    /// The ctx the KV figure assumes.
    pub ctx: u32,
    /// Recency heat (0 = cold); higher loads first.
    pub heat: u64,
}

/// Greedy plan: hot-first, then smallest-first, accumulating while the
/// budget holds. Returns (resident, deferred) in load-priority order.
#[must_use]
pub fn plan(footprints: &[Footprint], vram_mib: u64) -> (Vec<&Footprint>, Vec<&Footprint>) {
    let mut order: Vec<&Footprint> = footprints.iter().collect();
    order.sort_by(|a, b| {
        b.heat
            .cmp(&a.heat)
            .then_with(|| a.weights_mib.cmp(&b.weights_mib))
            .then_with(|| a.name.cmp(&b.name))
    });
    let mut used = 0u64;
    let mut resident = Vec::new();
    let mut deferred = Vec::new();
    for f in order {
        let need = f.weights_mib.saturating_add(f.kv_mib);
        // 512 MiB headroom for activations/context scratch.
        if used.saturating_add(need).saturating_add(512) <= vram_mib {
            used = used.saturating_add(need);
            resident.push(f);
        } else {
            deferred.push(f);
        }
    }
    (resident, deferred)
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    fn fp(name: &str, w: u64, kv: u64, heat: u64) -> Footprint {
        Footprint {
            name: name.into(),
            weights_mib: w,
            kv_mib: kv,
            ctx: 16_384,
            heat,
        }
    }

    #[test]
    fn unit__plan__hot_first_then_smallest() {
        let fps = vec![
            fp("big", 8000, 512, 0),
            fp("hot-small", 1000, 128, 5),
            fp("cold-small", 900, 128, 0),
        ];
        let (resident, deferred) = plan(&fps, 4000);
        let names: Vec<&str> = resident.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(
            names,
            vec!["hot-small", "cold-small"],
            "heat first, then size"
        );
        assert_eq!(deferred.len(), 1);
        assert_eq!(deferred[0].name, "big");
    }

    #[test]
    fn unit__plan__headroom_guard() {
        // weights+kv+512 must fit: 4039 refuses, 4040 fits exactly.
        let fps = vec![fp("m", 3400, 128, 0)];
        let (resident, _) = plan(&fps, 4039);
        assert!(resident.is_empty(), "needs weights+kv+512");
        let (resident, _) = plan(&fps, 4040);
        assert_eq!(resident.len(), 1);
    }
}
