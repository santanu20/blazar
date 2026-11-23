//! `/api/config` family - the settings surface behind the web console
//! and any orchestration tooling. `reload` is the one-shot hot-reload
//! shared with the CLI (`blazar config reload`); `GET /api/config`
//! serves the RUNNING config (secrets redacted) plus the settable-key
//! schema; `set`/`unset` route through blazar-core's dotted-key editor
//! - the exact same grammar, oracle validation, and atomic persist the
//! CLI uses, then hot-reload so the change engages immediately (the
//! watcher task and these paths share the (mtime, size) fingerprint
//! debounce, so a manual reload racing the watcher applies exactly
//! once).

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use std::sync::Arc;

use crate::state::AppState;

/// POST /api/config/reload - validate and swap the running config in
/// place. Bind-time knobs (host, port, TLS, ...) are detected and named
/// as `cold_only`: they take effect at the next daemon start, never
/// mid-flight. A parse or validation failure keeps the old config
/// running and answers 422 with the reason.
///
/// Management-class surface: rewriting the RUNNING daemon's config is
/// an operator action, so when keys exist only an unscoped (admin) key
/// may trigger it (authless daemons stay open - the local CLI
/// contract; the CLI attaches an admin bearer when keys exist).
pub async fn reload(
    State(state): State<Arc<AppState>>,
    key: Option<axum::Extension<crate::keys::KeyCtx>>,
) -> Response {
    if !state.keys.is_empty()
        && let Err(denied) = crate::management_authorized(&state, key.as_ref().map(|e| &e.0))
    {
        return denied;
    }
    match state.sup.reload_config_now() {
        Ok(outcome) => {
            let receipt = reload_receipt(&state, &outcome);
            (StatusCode::OK, Json(receipt)).into_response()
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

/// Post-reload actions + the JSON receipt shared by `reload`, `set` and
/// `unset`: a changed preload list engages now (mirroring boot), and a
/// changed `[[keys]]` set is hot-swapped auth state, not a bind-time
/// knob. Bind-time knobs are named as `cold_only` for the next start.
fn reload_receipt(
    state: &AppState,
    outcome: &blazar_runtime::supervisor::ConfigReloadOutcome,
) -> serde_json::Value {
    if outcome.preload_changed {
        // Mirror the boot-time behavior: a changed preload list
        // engages now, not at next start.
        let sup = Arc::clone(&state.sup);
        tokio::spawn(async move { sup.preload_listed().await });
    }
    if outcome.keys_changed {
        // The `[[keys]]` set is hot-swappable auth state, not a
        // bind-time knob: push the new generation into the live
        // limiter so enforcement reflects the file immediately.
        state.keys.replace_all(state.cfg().keys.clone());
        tracing::info!(
            "api keys hot-reloaded - {} live key(s)",
            state.keys.entries().len()
        );
    }
    let mut note = if outcome.cold_only.is_empty() {
        "all changed knobs applied live".to_string()
    } else {
        format!(
            "applied live; these bind-time knobs need a daemon restart: {}",
            outcome.cold_only.join(", ")
        )
    };
    if outcome.keys_changed {
        note.push_str("; api keys applied live");
    }
    serde_json::json!({
        "applied": outcome.applied,
        "cold_only": outcome.cold_only,
        "preload_changed": outcome.preload_changed,
        "note": note,
    })
}

/// GET /api/config - the RUNNING config (secrets redacted) plus the
/// settable-key schema for editors. Read-plane disclosure of topology
/// (remotes, fleet wiring), so it rides the same management gate as
/// reload: keys-mode daemons need an unscoped admin bearer; authless
/// daemons stay open (the local CLI contract).
pub async fn get(
    State(state): State<Arc<AppState>>,
    key: Option<axum::Extension<crate::keys::KeyCtx>>,
) -> Response {
    if !state.keys.is_empty()
        && let Err(denied) = crate::management_authorized(&state, key.as_ref().map(|e| &e.0))
    {
        return denied;
    }
    let mut cfg = serde_json::to_value(state.cfg().as_ref())
        .unwrap_or_else(|e| serde_json::json!({"error": format!("serialize running config: {e}")}));
    redact_secrets(&mut cfg);
    let raw_file = std::fs::read_to_string(state.dirs.config_file()).unwrap_or_default();
    let schema: Vec<serde_json::Value> = blazar_core::config_edit::schema_sections(&raw_file)
        .into_iter()
        .map(|s| {
            serde_json::json!({
                "name": s.name,
                "leaves": s.leaves,
                "is_map": s.is_map,
            })
        })
        .collect();
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "config": cfg,
            "schema": schema,
            "file": {"present": state.dirs.config_file().is_file()},
        })),
    )
        .into_response()
}

/// Secret-bearing field names redacted in every HTTP read plane: key
/// material (`[[keys]].key`), MCP child environment values (provider
/// API tokens ride there), and anything named like a credential. The
/// redaction keeps an 8-char prefix so an operator can still tell
/// entries apart without learning the secret.
fn redact_secrets(v: &mut serde_json::Value) {
    match v {
        serde_json::Value::Object(map) => {
            for (k, val) in map.iter_mut() {
                let secret_name = matches!(
                    k.as_str(),
                    "key" | "token" | "secret" | "password" | "api_key" | "bearer"
                );
                if secret_name
                    && let serde_json::Value::String(s) = val
                    && !s.is_empty()
                {
                    *val = serde_json::Value::String(crate::redact(s));
                } else if k == "env"
                    && let serde_json::Value::Object(env) = val
                {
                    for (_name, ev) in env.iter_mut() {
                        if let serde_json::Value::String(s) = ev {
                            *ev = serde_json::Value::String(crate::redact(s));
                        }
                    }
                } else {
                    redact_secrets(val);
                }
            }
        }
        serde_json::Value::Array(items) => {
            for item in items.iter_mut() {
                redact_secrets(item);
            }
        }
        _ => {}
    }
}

/// The management gate `set`/`unset`/`reload` share: keys-mode daemons
/// require an unscoped admin bearer; authless daemons stay open (an
/// operator on the box can already edit config.toml directly - the CLI
/// is that operator's hands, and this endpoint is the console's).
fn operator_authorized(state: &AppState, key: Option<&crate::keys::KeyCtx>) -> Option<Response> {
    if state.keys.is_empty() {
        return None;
    }
    crate::management_authorized(state, key).err()
}

/// POST /api/config/set - body `{"key": "...", "value": "..."}`.
/// Routes through [`blazar_core::config_edit::set_candidate`] (the CLI's
/// exact grammar: dotted table-leaf surgery, bare-key routing, merged
/// model-name retries, quoted-string twin hint), persists atomically,
/// then hot-reloads so the change engages now. The schema oracle gates
/// everything - a rejected candidate never touches the file.
pub async fn set(
    State(state): State<Arc<AppState>>,
    key: Option<axum::Extension<crate::keys::KeyCtx>>,
    Json(body): Json<serde_json::Value>,
) -> Response {
    if let Some(denied) = operator_authorized(&state, key.as_ref().map(|e| &e.0)) {
        return denied;
    }
    let (Some(key), Some(value)) = (
        body.get("key").and_then(serde_json::Value::as_str),
        body.get("value").and_then(serde_json::Value::as_str),
    ) else {
        return crate::error_response(400, "body needs string `key` and string `value`");
    };
    let _guard = config_write_lock();
    let raw = match state.config_raw() {
        Ok(raw) => raw,
        Err(resp) => return resp,
    };
    let outcome = blazar_core::config_edit::set_candidate(&raw, key, value);
    match outcome {
        blazar_core::config_edit::SetOutcome::Applied { body, stored } => {
            if let Err(resp) = state.persist_config_raw(&body) {
                return resp;
            }
            drop(_guard);
            let reload = state.sup.reload_config_now().ok();
            let mut receipt = serde_json::json!({
                "applied": true,
                "key": key,
                "stored": stored,
            });
            if let Some(out) = reload.as_ref() {
                receipt["reload"] = reload_receipt(&state, out);
            } else {
                receipt["reload"] = serde_json::json!({
                    "applied": false,
                    "note": "file persisted; live reload failed - the 2s watcher retries, or POST /api/config/reload",
                });
            }
            (StatusCode::OK, Json(receipt)).into_response()
        }
        blazar_core::config_edit::SetOutcome::Rejected {
            reason,
            string_hint,
        } => {
            let mut err = serde_json::json!({
                "message": format!("rejected, file unchanged: {reason}"),
                "type": "config_error",
            });
            if let Some(quoted) = string_hint {
                err["hint"] = serde_json::Value::String(format!(
                    "{key} takes a string - quote the value: {quoted}"
                ));
            }
            (StatusCode::UNPROCESSABLE_ENTITY, Json(serde_json::json!({"error": err}))).into_response()
        }
        blazar_core::config_edit::SetOutcome::TableNotValue { leaves } => (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(serde_json::json!({"error": {
                "message": format!("{key} is a table, not a value - set individual leaves: {key}.<leaf> <value>"),
                "type": "config_error",
                "leaves": leaves,
            }})),
        )
            .into_response(),
        blazar_core::config_edit::SetOutcome::Ambiguous { sections } => (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(serde_json::json!({"error": {
                "message": format!(
                    "ambiguous key: {key} is a leaf of several sections ({}) - use the dotted form",
                    sections.join(", ")
                ),
                "type": "config_error",
                "sections": sections,
            }})),
        )
            .into_response(),
    }
}

/// POST /api/config/unset - body `{"key": "..."}`. Same routing grammar
/// as `set`, mirrored for removal: the pin leaves the file, validation
/// gates the candidate, then a hot-reload restores the built-in default
/// (or documents the bind-time knob for the next start).
pub async fn unset(
    State(state): State<Arc<AppState>>,
    key: Option<axum::Extension<crate::keys::KeyCtx>>,
    Json(body): Json<serde_json::Value>,
) -> Response {
    if let Some(denied) = operator_authorized(&state, key.as_ref().map(|e| &e.0)) {
        return denied;
    }
    let Some(key) = body.get("key").and_then(serde_json::Value::as_str) else {
        return crate::error_response(400, "body needs a string `key`");
    };
    let _guard = config_write_lock();
    let raw = match state.config_raw() {
        Ok(raw) => raw,
        Err(resp) => return resp,
    };
    let outcome = blazar_core::config_edit::unset_candidate(&raw, key);
    match outcome {
        blazar_core::config_edit::UnsetOutcome::Removed { body, removed_line } => {
            if let Err(resp) = state.persist_config_raw(&body) {
                return resp;
            }
            drop(_guard);
            let reload = state.sup.reload_config_now().ok();
            let mut receipt = serde_json::json!({
                "applied": true,
                "key": key,
                "removed": removed_line,
            });
            if let Some(out) = reload.as_ref() {
                receipt["reload"] = reload_receipt(&state, out);
            } else {
                receipt["reload"] = serde_json::json!({
                    "applied": false,
                    "note": "file persisted; live reload failed - the 2s watcher retries, or POST /api/config/reload",
                });
            }
            (StatusCode::OK, Json(receipt)).into_response()
        }
        blazar_core::config_edit::UnsetOutcome::NotPinned => (
            StatusCode::OK,
            Json(serde_json::json!({
                "applied": false,
                "key": key,
                "note": "not pinned - already at the built-in default",
            })),
        )
            .into_response(),
        blazar_core::config_edit::UnsetOutcome::UnknownKey => crate::error_response(
            404,
            &format!(
                "unknown config key: {key} (table knobs use dotted keys, e.g. sglang.stream_interval)"
            ),
        ),
        blazar_core::config_edit::UnsetOutcome::TableNotValue => (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(serde_json::json!({"error": {
                "message": format!("{key} is a table - unset individual leaves: {key}.<leaf>"),
                "type": "config_error",
            }})),
        )
            .into_response(),
        blazar_core::config_edit::UnsetOutcome::Ambiguous { sections } => (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(serde_json::json!({"error": {
                "message": format!(
                    "ambiguous key: {key} is a leaf of several sections ({}) - use the dotted form",
                    sections.join(", ")
                ),
                "type": "config_error",
                "sections": sections,
            }})),
        )
            .into_response(),
        blazar_core::config_edit::UnsetOutcome::Rejected { reason } => (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(serde_json::json!({"error": {
                "message": format!("rejected, file unchanged: {reason}"),
                "type": "config_error",
            }})),
        )
            .into_response(),
    }
}

/// Read-modify-write scope for the config file: holds the process-wide
/// write lock (the same exclusion `persist_keys` uses - hoisted so key
/// rotations and config edits serialize against each other too). The
/// lock spans read -> edit -> persist, so a racing writer can never
/// interleave read/write. The guard's existence IS the exclusion.
/// The guard's existence IS the exclusion - the MutexGuard inside is
/// the point, never read.
#[allow(dead_code)]
struct ConfigWriteGuard(std::sync::MutexGuard<'static, ()>);

/// Take the config write lock; the guard's mere existence is the
/// exclusion (drop ends it). `config_raw`/`persist_config_raw` below do
/// the IO while it is held.
fn config_write_lock() -> ConfigWriteGuard {
    ConfigWriteGuard(
        crate::CONFIG_WRITE_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
    )
}

trait ConfigFileIo {
    fn config_raw(&self) -> Result<String, Response>;
    fn persist_config_raw(&self, body: &str) -> Result<(), Response>;
}

impl ConfigFileIo for AppState {
    fn config_raw(&self) -> Result<String, Response> {
        let path = self.dirs.config_file();
        if !path.exists() {
            // Fresh-box bootstrap, same as the CLI: `set` may be the
            // first write ever - Config::load materializes the default.
            if let Err(e) = blazar_core::Config::load(&self.dirs) {
                return Err(crate::error_response(
                    500,
                    &format!("bootstrap config.toml: {e}"),
                ));
            }
        }
        std::fs::read_to_string(&path)
            .map_err(|e| crate::error_response(500, &format!("read {}: {e}", path.display())))
    }

    fn persist_config_raw(&self, body: &str) -> Result<(), Response> {
        let path = self.dirs.config_file();
        blazar_core::persist_config(&path, body)
            .map_err(|e| crate::error_response(500, &format!("write {}: {e}", path.display())))
    }
}
