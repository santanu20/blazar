//! `/api/engines` - installed engine rows joined with their derived
//! capability cards: one object per row carrying the store columns
//! (`tag`/`kind`/`active`/...) and the same card `blazar engine card`
//! renders (lane contract + probed build facts + honest KV caps under
//! the live `[kv.lmcache]` truth). Read-only; no spawn, no probe -
//! everything shown was probed at install time.

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use serde_json::{Value, json};

use crate::state::AppState;

pub async fn engines(State(state): State<Arc<AppState>>) -> Json<Value> {
    let cfg = state.sup.cfg();
    let sglang_lmcache = cfg.kv.lmcache.enabled && cfg.sglang.lmcache.enabled;
    let rows = state
        .with_store(|s| s.list_engines().unwrap_or_default())
        .unwrap_or_default();
    let cards: Vec<Value> = rows
        .iter()
        .map(|r| {
            // Same join the CLI renders; an unreadable manifest keeps
            // the row listed with the honest lane-only card (engine
            // reprobe heals the row) instead of dropping it silently.
            let card = serde_json::from_str::<blazar_runtime::Manifest>(&r.manifest).map_or_else(
                |_| blazar_runtime::lane_only_capability_card(r.kind, sglang_lmcache),
                |m| m.capability_card(r.kind, sglang_lmcache),
            );
            json!({
                "tag": r.tag,
                "kind": r.kind.as_str(),
                "active": r.active,
                "capabilities": card,
            })
        })
        .collect();
    Json(json!({
        "object": "blazar.engines",
        "engines": cards,
    }))
}
