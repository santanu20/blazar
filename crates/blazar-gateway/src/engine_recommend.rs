//! `/api/engine/recommend` - the ranked engine-build view, read-only.
//!
//! Same pure ranking the CLI prints (`blazar engine recommend`):
//! hardware match and stored bench receipts decide, recency and active
//! status break ties, and every axis either shows its datum or states
//! its absence. Nothing here re-runs a benchmark or touches an engine.

use std::sync::Arc;

use axum::Json;
use axum::extract::{Query, State};
use serde_json::{Value, json};

use crate::state::AppState;

#[derive(serde::Deserialize, Default)]
pub struct RecommendQuery {
    /// Scope the bench axis to one model's receipts (optional).
    pub model: Option<String>,
}

/// `GET /api/engine/recommend?model=<name>`
pub async fn engine_recommend(
    State(state): State<Arc<AppState>>,
    Query(q): Query<RecommendQuery>,
) -> Json<Value> {
    // The hardware probe spawns OS census subprocesses (nvidia-smi /
    // lspci) - keep them off the async workers, like capacity does.
    let state_for_task = Arc::clone(&state);
    let model = q.model.clone();
    let ranked = tokio::task::spawn_blocking(move || {
        let hw = blazar_runtime::probe_hardware(None);
        let live_gpu = hw.gpus.first().map(|g| g.name.clone());
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or_default();
        #[allow(clippy::cast_possible_wrap)] // epoch seconds fit i64 by construction
        let now = secs as i64;
        let rows = state_for_task.with_store(|store| {
            blazar_core::optimize::engine_recommend(
                store,
                model.as_deref(),
                &blazar_core::optimize::EngineRecommendInputs {
                    hw: live_gpu.as_deref(),
                    now,
                },
            )
        });
        (live_gpu, rows)
    })
    .await
    .unwrap_or_else(|join_err| {
        // A JoinError here means the store probe task panicked or was
        // cancelled - without this line the handler would quietly
        // present "store unavailable" + "no GPU" notes as if they were
        // measured facts.
        tracing::warn!("engine recommend: store probe task failed ({join_err})");
        (None, None)
    });

    let mut notes = vec![
        "ranking axes: hardware match and bench receipts decide; recency and active status break ties".to_string(),
        "architecture coverage is not ranked - engine manifests carry no architecture lists".to_string(),
    ];
    let rows: Vec<Value> = match ranked.1 {
        Some(Ok(recs)) => recs
            .iter()
            .map(|r| serde_json::to_value(r).unwrap_or(Value::Null))
            .collect(),
        Some(Err(e)) => {
            notes.push(format!("ranking failed: {e}"));
            Vec::new()
        }
        None => {
            notes.push("store unavailable - no engines ranked".to_string());
            Vec::new()
        }
    };
    if ranked.0.is_none() {
        notes.push(
            "no GPU detected by the hardware probe - hardware axis skipped for every build"
                .to_string(),
        );
    }
    Json(json!({
        "object": "blazar.engine_recommend",
        "model": q.model,
        "hardware": ranked.0,
        "rows": rows,
        "notes": notes,
    }))
}
