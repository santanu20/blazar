//! R4: opt-in semantic cache for non-stream chat responses.
//!
//! Bifrost-shaped (per-request headers + `cache_debug` in responses) but
//! stricter: entries are exact-filtered by lane + chat model + API key,
//! so a hit never crosses tenants; similarity only decides whether a
//! stored prompt is "the same question". Correctness-sensitive lanes
//! stay off (config default) or opt out per request
//! (`x-blazar-cache: off`).

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde_json::Value;

/// Ollama lane (`/api/chat`). The `OpenAI` byte-proxy lane does not
/// participate yet; the tag keeps a future lane from serving another
/// lane's response shape.
pub const LANE_OLLAMA: u8 = 1;

/// Counters surfaced in `/metrics` (`cache_metrics`).
#[derive(Default)]
pub struct SemMetrics {
    pub hits: AtomicU64,
    pub misses: AtomicU64,
    pub stores: AtomicU64,
    pub embed_failures: AtomicU64,
}

struct Entry {
    lane: u8,
    model: String,
    /// API key name that produced the entry (None = authless loopback).
    key: Option<String>,
    emb: Vec<f32>,
    /// Stored response, always in `OpenAI` shape; the ollama lane
    /// re-translates on hit (same code path as a live response).
    response: Value,
    expires: Instant,
    last_hit: Instant,
}

/// In-memory semantic cache. Lazy TTL sweep on access; LRU eviction
/// (min `last_hit`) when `max_entries` is exceeded.
#[derive(Default)]
pub struct SemanticCache {
    inner: Mutex<HashMap<u64, Entry>>,
    next_id: AtomicU64,
}

impl SemanticCache {
    /// Empty cache; `AppState::new` wires the shared instance.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn sweep_locked(map: &mut HashMap<u64, Entry>) {
        let now = Instant::now();
        map.retain(|_, e| e.expires > now);
    }

    /// Best-match lookup: exact (lane, model, key) filter, then cosine
    /// similarity ≥ threshold. Returns `(cache_id, similarity, response)`
    /// of the best entry.
    pub fn lookup(
        &self,
        lane: u8,
        model: &str,
        key: Option<&str>,
        emb: &[f32],
        threshold: f32,
    ) -> Option<(u64, f32, Value)> {
        let mut map = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        Self::sweep_locked(&mut map);
        let mut best: Option<(u64, f32, f32)> = None; // (id, sim, -unused rank)
        for (id, e) in map.iter() {
            if e.lane != lane || e.model != model || e.key.as_deref() != key {
                continue;
            }
            let sim = cosine(emb, &e.emb);
            if sim >= threshold && best.is_none_or(|(_, b, _)| sim > b) {
                best = Some((*id, sim, 0.0));
            }
        }
        let (id, sim) = best.map(|(i, s, _)| (i, s))?;
        let e = map.get_mut(&id)?;
        e.last_hit = Instant::now();
        Some((id, sim, e.response.clone()))
    }

    /// Store a response. Returns the assigned `cache_id`. Evicts the
    /// least-recently-hit entry when over capacity.
    // One cohesive cache-put: identity filter triplet + payload + TTL + cap.
    #[allow(clippy::too_many_arguments)]
    pub fn store(
        &self,
        lane: u8,
        model: &str,
        key: Option<&str>,
        emb: Vec<f32>,
        response: Value,
        ttl: Duration,
        max_entries: usize,
    ) -> u64 {
        let mut map = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        Self::sweep_locked(&mut map);
        while map.len() >= max_entries.max(1) {
            if let Some((&evict, _)) = map.iter().min_by_key(|(_, e)| e.last_hit) {
                map.remove(&evict);
            } else {
                break;
            }
        }
        let id = self.next_id.fetch_add(1, Ordering::Relaxed) + 1;
        map.insert(
            id,
            Entry {
                lane,
                model: model.to_string(),
                key: key.map(str::to_string),
                emb,
                response,
                expires: Instant::now() + ttl,
                last_hit: Instant::now(),
            },
        );
        id
    }

    /// Live (non-expired) entry count, for the metrics gauge.
    pub fn live_len(&self) -> usize {
        let mut map = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        Self::sweep_locked(&mut map);
        map.len()
    }
}

/// Cosine similarity over L2-normalized vectors (still guards zero
/// norms). Inputs need not be normalized.
#[must_use]
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for i in 0..n {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// L2-normalize in place (zero vector stays zero).
pub fn l2_normalize(v: &mut [f32]) {
    let mut norm = 0.0f32;
    for x in &*v {
        norm += x * x;
    }
    norm = norm.sqrt();
    if norm > 0.0 {
        for x in v {
            *x /= norm;
        }
    }
}

/// Per-request cache directive parsed from headers + config.
#[derive(Debug, Clone, PartialEq)]
pub struct Directive {
    pub ttl: Duration,
    pub threshold: f32,
}

/// Carried from `chat()` (miss path) into `proxy_core_chat`'s non-stream
/// store step: everything needed to file the response after it completes.
pub struct SemCtx {
    pub emb: Vec<f32>,
    pub directive: Directive,
    /// API key name that produced the response (None = authless loopback).
    pub key: Option<String>,
}

/// Header names (also advertised in `/.well-known/blazar`).
pub const HDR_CACHE: &str = "x-blazar-cache";
pub const HDR_CACHE_TTL: &str = "x-blazar-cache-ttl";
pub const HDR_CACHE_THRESHOLD: &str = "x-blazar-cache-threshold";

/// Resolve whether the cache participates for this request and with
/// what parameters. `None` = bypass (no headers touched beyond the
/// bypass marker set by the caller). Malformed override values fail
/// fast with `Err(msg)` (surfaced as 400 — never silently ignored).
#[allow(clippy::cast_possible_truncation)] // config threshold is f64; comparisons run in f32
pub fn directive(
    enabled: bool,
    model_configured: bool,
    cfg_ttl: u64,
    cfg_threshold: f64,
    get_header: impl Fn(&str) -> Option<String>,
) -> Result<Option<Directive>, String> {
    if !model_configured {
        return Ok(None);
    }
    let on = match get_header(HDR_CACHE).as_deref() {
        Some(v) if v.eq_ignore_ascii_case("on") => true,
        Some(v) if v.eq_ignore_ascii_case("off") => false,
        Some(_) => return Err(format!("invalid {HDR_CACHE} value (expected on|off)")),
        None => enabled,
    };
    if !on {
        return Ok(None);
    }
    let ttl = match get_header(HDR_CACHE_TTL) {
        None => cfg_ttl,
        Some(v) => match v.parse::<u64>() {
            Ok(t) if (1..=86_400).contains(&t) => t,
            _ => {
                return Err(format!(
                    "invalid {HDR_CACHE_TTL} value (expected 1..=86400 seconds)"
                ))
            }
        },
    };
    let threshold = match get_header(HDR_CACHE_THRESHOLD) {
        None => cfg_threshold,
        Some(v) => match v.parse::<f64>() {
            Ok(t) if (0.01..=1.0).contains(&t) => t,
            _ => {
                return Err(format!(
                    "invalid {HDR_CACHE_THRESHOLD} value (expected 0.01..=1.0)"
                ))
            }
        },
    };
    Ok(Some(Directive {
        ttl: Duration::from_secs(ttl),
        threshold: threshold as f32,
    }))
}

/// Mean-pool a per-token embedding matrix into one L2-normalized vector.
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)] // f64 child vectors -> f32 cache vectors; row count -> f64
pub fn mean_pool(matrix: &[Vec<f64>]) -> Vec<f32> {
    if matrix.is_empty() {
        return Vec::new();
    }
    let dim = matrix[0].len();
    let mut acc = vec![0.0f64; dim];
    for row in matrix {
        for (a, x) in acc.iter_mut().zip(row.iter()) {
            *a += x;
        }
    }
    let n = matrix.len() as f64;
    let mut out: Vec<f32> = acc.into_iter().map(|x| (x / n) as f32).collect();
    l2_normalize(&mut out);
    out
}

/// Interpret a child `/embedding` response's `embedding` field: a
/// matrix (pooling=none) is mean-pooled; a flat vector (pooling=mean)
/// is L2-normalized as-is. Returns None on unexpected shapes.
#[must_use]
#[allow(clippy::cast_possible_truncation)] // child embeddings are f64; cache vectors are f32
pub fn embed_from_value(embedding: &Value) -> Option<Vec<f32>> {
    let arr = embedding.as_array()?;
    if arr.is_empty() {
        return None;
    }
    if arr[0].is_array() {
        // Per-token matrix.
        let matrix: Vec<Vec<f64>> = arr
            .iter()
            .map(|row| {
                row.as_array()?
                    .iter()
                    .map(Value::as_f64)
                    .collect::<Option<Vec<f64>>>()
            })
            .collect::<Option<Vec<Vec<f64>>>>()?;
        let pooled = mean_pool(&matrix);
        (!pooled.is_empty()).then_some(pooled)
    } else {
        let mut out: Vec<f32> = arr
            .iter()
            .map(Value::as_f64)
            .collect::<Option<Vec<f64>>>()?
            .into_iter()
            .map(|v| v as f32)
            .collect();
        l2_normalize(&mut out);
        (!out.is_empty()).then_some(out)
    }
}

/// Embed `text` with the configured embed model via the engine child
/// (tokenize → legacy `/embedding` with raw ids → mean-pool/L2). Works
/// on any `--embeddings` child regardless of pooling mode.
pub async fn embed_prompt(
    state: &std::sync::Arc<crate::state::AppState>,
    embed_model: &str,
    text: &str,
) -> Result<Vec<f32>, String> {
    let (engine, _load_ms) = crate::proxy::ensure_with_admission(
        state,
        embed_model,
        crate::queue::Priority::Normal,
        // Short, tool-free, never raw — interactive-class admission.
        crate::queue::WorkClass::Interactive,
        None,
        false,
        false, // text surface: diffusion rows teach the images lane
    )
    .await
    .map_err(|e| format!("embed model admission failed: {e:?}"))?;
    let base = crate::proxy::child_base(&engine.endpoint);
    // Tokenize exactly (no BOS so ids map 1:1 to child tokens).
    let ids: Vec<u64> = crate::proxy::child_auth(
        state
            .http
            .post(format!("{base}/tokenize"))
            .json(&serde_json::json!({"content": text, "add_special": false})),
        &engine,
    )
    .send()
    .await
    .map_err(|e| format!("tokenize call failed: {e}"))?
    .json::<Value>()
    .await
    .map_err(|e| format!("tokenize parse failed: {e}"))?["tokens"]
        .as_array()
        .ok_or_else(|| "tokenize response missing tokens".to_string())?
        .iter()
        .filter_map(Value::as_u64)
        .collect();
    if ids.is_empty() {
        return Err("empty prompt embedding".into());
    }
    let resp = crate::proxy::child_auth(
        state
            .http
            .post(format!("{base}/embedding"))
            .json(&serde_json::json!({"content": ids})),
        &engine,
    )
    .send()
    .await
    .map_err(|e| format!("embedding call failed: {e}"))?
    .json::<Value>()
    .await
    .map_err(|e| format!("embedding parse failed: {e}"))?;
    let embedding = resp
        .as_array()
        .and_then(|a| a.first())
        .map(|e| &e["embedding"])
        .ok_or_else(|| "embedding response missing data".to_string())?;
    embed_from_value(embedding).ok_or_else(|| "unexpected embedding shape".to_string())
}

#[cfg(test)]
#[allow(non_snake_case)]
#[allow(clippy::duration_suboptimal_units)]
mod tests {
    use super::*;

    #[test]
    fn unit__cosine__identical_orthogonal_unnormalized() {
        let a = [1.0f32, 0.0];
        let b = [2.0f32, 0.0];
        let c = [0.0f32, 3.0];
        assert!((cosine(&a, &b) - 1.0).abs() < 1e-6);
        assert!(cosine(&a, &c).abs() < 1e-6);
        assert!(cosine(&a, &[0.0, 0.0]).abs() < 1e-6);
    }

    #[test]
    fn unit__directive__config_header_matrix() {
        let none = |_: &str| None::<String>;
        // config off, no header -> bypass
        assert_eq!(directive(false, true, 600, 0.9, none).unwrap(), None);
        // config on -> active with config values
        assert_eq!(
            directive(true, true, 600, 0.9, none).unwrap(),
            Some(Directive {
                ttl: Duration::from_secs(600),
                threshold: 0.9
            })
        );
        // header on beats config off
        let on = |h: &str| (h == HDR_CACHE).then(|| "on".to_string());
        assert!(directive(false, true, 600, 0.9, on).unwrap().is_some());
        // header off beats config on
        let off = |h: &str| (h == HDR_CACHE).then(|| "off".to_string());
        assert_eq!(directive(true, true, 600, 0.9, off).unwrap(), None);
        // model not configured -> always bypass
        assert_eq!(directive(true, false, 600, 0.9, on).unwrap(), None);
        // overrides
        let both = |h: &str| match h {
            HDR_CACHE => Some("on".to_string()),
            HDR_CACHE_TTL => Some("60".to_string()),
            HDR_CACHE_THRESHOLD => Some("0.75".to_string()),
            _ => None,
        };
        assert_eq!(
            directive(false, true, 600, 0.9, both).unwrap(),
            Some(Directive {
                ttl: Duration::from_secs(60),
                threshold: 0.75
            })
        );
    }

    #[test]
    fn unit__directive__malformed_headers_err() {
        let bad_cache = |h: &str| (h == HDR_CACHE).then(|| "maybe".to_string());
        assert!(directive(true, true, 600, 0.9, bad_cache).is_err());
        let bad_ttl = |h: &str| (h == HDR_CACHE_TTL).then(|| "0".to_string());
        assert!(directive(true, true, 600, 0.9, bad_ttl).is_err());
        let bad_thr = |h: &str| (h == HDR_CACHE_THRESHOLD).then(|| "1.5".to_string());
        assert!(directive(true, true, 600, 0.9, bad_thr).is_err());
    }

    #[test]
    fn unit__store_lookup__exact_filters_and_threshold() {
        let sc = SemanticCache::new();
        let emb = vec![1.0f32, 0.0];
        let id = sc.store(
            LANE_OLLAMA,
            "m1",
            Some("k1"),
            emb.clone(),
            serde_json::json!({"a": 1}),
            Duration::from_secs(60),
            16,
        );
        // exact lane+model+key, sim 1.0
        let hit = sc
            .lookup(LANE_OLLAMA, "m1", Some("k1"), &emb, 0.99)
            .unwrap();
        assert_eq!(hit.0, id);
        assert!((hit.1 - 1.0).abs() < 1e-6);
        assert_eq!(hit.2, serde_json::json!({"a": 1}));
        // different model / key / lane -> miss
        assert!(sc
            .lookup(LANE_OLLAMA, "m2", Some("k1"), &emb, 0.5)
            .is_none());
        assert!(sc
            .lookup(LANE_OLLAMA, "m1", Some("k2"), &emb, 0.5)
            .is_none());
        assert!(sc.lookup(LANE_OLLAMA, "m1", None, &emb, 0.5).is_none());
        // below threshold -> miss
        let orth = [0.0f32, 1.0];
        assert!(sc
            .lookup(LANE_OLLAMA, "m1", Some("k1"), &orth, 0.5)
            .is_none());
    }

    #[test]
    fn unit__store__ttl_expiry_and_lru_cap() {
        let sc = SemanticCache::new();
        let emb = vec![1.0f32, 0.0];
        sc.store(
            LANE_OLLAMA,
            "m",
            None,
            emb.clone(),
            serde_json::json!({}),
            Duration::from_millis(0),
            16,
        );
        std::thread::sleep(Duration::from_millis(2));
        assert!(sc.lookup(LANE_OLLAMA, "m", None, &emb, 0.1).is_none());
        assert_eq!(sc.live_len(), 0);
        // LRU: cap 2, third store evicts the least-recently-hit. Each
        // entry gets a distinct orthogonal embedding so lookups touch a
        // deterministic entry regardless of HashMap iteration order.
        let sc2 = SemanticCache::new();
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        let e1 = sc2.store(
            1,
            "m",
            None,
            a.clone(),
            json_v(1),
            Duration::from_secs(60),
            2,
        );
        let _e2 = sc2.store(
            1,
            "m",
            None,
            b.clone(),
            json_v(2),
            Duration::from_secs(60),
            2,
        );
        std::thread::sleep(Duration::from_micros(200));
        let _ = sc2.lookup(1, "m", None, &a, 0.99); // touches e1 only (cos(a,b)=0)
        std::thread::sleep(Duration::from_micros(200));
        let _e3 = sc2.store(
            1,
            "m",
            None,
            b.clone(),
            json_v(3),
            Duration::from_secs(60),
            2,
        );
        let hit = sc2.lookup(1, "m", None, &a, 0.99).unwrap();
        assert_eq!(hit.2, json_v(1)); // e2 evicted (least-recently-hit)
        assert_eq!(hit.0, e1);
        // And the newest (e3, embedding b) is retrievable too.
        let hit3 = sc2.lookup(1, "m", None, &b, 0.99).unwrap();
        assert_eq!(hit3.2, json_v(3));
    }

    fn json_v(n: u64) -> Value {
        serde_json::json!({"n": n})
    }

    #[test]
    fn unit__embed_from_value__matrix_and_flat_shapes() {
        let matrix = serde_json::json!([[1.0, 0.0], [0.0, 1.0]]);
        let v = embed_from_value(&matrix).unwrap();
        assert_eq!(v.len(), 2);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
        let flat = serde_json::json!([3.0, 4.0]);
        let v2 = embed_from_value(&flat).unwrap();
        assert!((v2[0] - 0.6).abs() < 1e-6);
        assert!(embed_from_value(&serde_json::json!([])).is_none());
        assert!(embed_from_value(&serde_json::json!("junk")).is_none());
    }

    #[test]
    fn unit__mean_pool__uniform_rows_average() {
        let m = vec![vec![2.0f64, 0.0], vec![0.0, 2.0]];
        let v = mean_pool(&m);
        assert!((v[0] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);
        assert!(mean_pool(&[]).is_empty());
    }
}
