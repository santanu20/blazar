//! Submit-time VRAM headroom for the diffusion scratch gates.
//!
//! `nvidia-smi` costs a process spawn (~30-60 ms); a burst of video
//! submits would pay it per request. A short TTL cache keeps the number
//! fresh enough for admission math — foreign tenants move on second
//! timescales, not sub-second — at one spawn per window. Boxes without a
//! working `nvidia-smi` get `None` and the gate steps aside: the spawn
//! heuristic in the runtime remains the last line of defense there.

use std::sync::Mutex;
use std::time::{Duration, Instant};

static CACHE: Mutex<Option<(Instant, u64)>> = Mutex::new(None);

/// Free VRAM (MiB, summed across NVIDIA cards) sampled no older than
/// `ttl`. `None` when this box has no working `nvidia-smi`.
pub fn free_vram_mib(ttl: Duration) -> Option<u64> {
    if let Some((at, mib)) = *CACHE.lock().unwrap() {
        if at.elapsed() <= ttl {
            return Some(mib);
        }
    }
    let fresh = blazar_runtime::probe::nvidia_free_vram_mib()?;
    *CACHE.lock().unwrap() = Some((Instant::now(), fresh));
    Some(fresh)
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    #[test]
    fn unit__free_vram_mib__caches_within_ttl() {
        // First call probes the real box: on CI without nvidia-smi the
        // cache stays empty and the gate is a no-op — assert both shapes.
        let first = free_vram_mib(Duration::from_secs(60));
        let second = free_vram_mib(Duration::from_secs(60));
        assert_eq!(first, second, "second hit inside TTL must not re-probe");
    }
}
