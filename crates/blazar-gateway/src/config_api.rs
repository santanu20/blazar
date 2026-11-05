//! `/api/config/reload` - one-shot hot-reload surface shared by the CLI
//! (`blazar config reload`) and any orchestration that edits the config
//! file in place. The watcher task drives the same supervisor method on
//! file-change; both paths share the (mtime, size) fingerprint debounce
//! so a manual reload racing the watcher applies exactly once.

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use std::sync::Arc;

use crate::state::AppState;

/// POST /api/config/reload - validate and swap the running config in
/// place. Bind-time knobs (host, port, TLS, ...) are detected and named
/// as `cold_only`: they take effect at the next daemon start, never
/// mid-flight. A parse or validation failure keeps the old config
/// running and answers 422 with the reason.
pub async fn reload(State(state): State<Arc<AppState>>) -> Response {
    match state.sup.reload_config_now() {
        Ok(outcome) => {
            if outcome.preload_changed {
                // Mirror the boot-time behavior: a changed preload list
                // engages now, not at next start.
                let sup = Arc::clone(&state.sup);
                tokio::spawn(async move { sup.preload_listed().await });
            }
            let note = if outcome.cold_only.is_empty() {
                "all changed knobs applied live".to_string()
            } else {
                format!(
                    "applied live; these bind-time knobs need a daemon restart: {}",
                    outcome.cold_only.join(", ")
                )
            };
            (
                StatusCode::OK,
                Json(serde_json::json!({
                    "applied": outcome.applied,
                    "cold_only": outcome.cold_only,
                    "preload_changed": outcome.preload_changed,
                    "note": note,
                })),
            )
                .into_response()
        }
        Err(reason) => (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(serde_json::json!({
                "error": {
                    "message": reason,
                    "type": "config_error",
                }
            })),
        )
            .into_response(),
    }
}
