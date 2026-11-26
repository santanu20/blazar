//! E4 - Gemini `generateContent` translate lane.
//!
//! Gemini-dialect clients (google-genai SDKs, Gemini CLI, the large
//! body of Gemini-SDK code) speak `POST /v1beta/models/{model}:
//! {generateContent|streamGenerateContent}`: `contents` turn arrays
//! with `parts`, `systemInstruction`, `functionDeclarations` /
//! `functionCall` / `functionResponse` tool blocks, camelCase
//! `generationConfig`, and `usageMetadata` accounting. The engine
//! speaks `OpenAI` chat. This module translates BOTH directions
//! locally: request -> `OpenAI` chat body, child response -> Gemini
//! shape (`JSON` + `SSE`). Local engine only - a remote-prefixed
//! model gets a teaching error (remote fleet speaks its own
//! dialect), same posture as the Anthropic lane.
//!
//! Gemini `functionCall`/`functionResponse` parts carry NO call ids,
//! while `OpenAI` tool messages must reference the assistant's
//! `tool_call_id`. The translator therefore mints deterministic ids
//! from the function NAME on both sides of the round trip, so a
//! Gemini client that echoes a `functionResponse` for a name it
//! received in a `functionCall` matches the id the engine sees.
//!
//! `candidateCount > 1` is refused with teaching: the translate lane
//! must not silently drop candidates (engines differ on `n`).
//! `generationConfig.topK` has no `OpenAI` chat equivalent - it is
//! warn-logged and dropped, never silently.
//!
//! Mount is flag-gated (`gemini_translate`, default on - the kill
//! switch). Route shape: axum/matchit refuses a parameter plus a
//! literal suffix in one path segment (`{model}:generateContent`
//! panics 'Only one parameter is allowed per path segment'), so the
//! route is a single-segment catch-all `{model_verb}` and the
//! handler splits the LAST `:` - model names containing `:` (Ollama
//! tag style) still parse because the verb is the final segment.

use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use futures::StreamExt;
use serde_json::{Value, json};
use std::sync::Arc;

use crate::proxy::{admission_gate_slo, child_base, ensure_with_admission};
use crate::queue::Priority;
use crate::translate::image_data_url;

/// Split a `{model_verb}` path segment into `(model, verb)`. The verb
/// is the suffix after the LAST `:` so model names that themselves
/// contain `:` (tag style) keep working; the verb must be one of the
/// two generateContent verbs, else the caller answers a Gemini-shaped
/// 404.
pub fn parse_model_verb(seg: &str) -> Result<(String, String), String> {
    let Some(colon) = seg.rfind(':') else {
        return Err(format!(
            "path segment {seg:?} has no :{{verb}} - Gemini calls are \
             /v1beta/models/{{model}}:generateContent or :streamGenerateContent"
        ));
    };
    let model = seg[..colon].to_string();
    let verb = seg[colon + 1..].to_string();
    if model.is_empty() {
        return Err("empty model name before the :{verb} suffix".to_string());
    }
    match verb.as_str() {
        "generateContent" | "streamGenerateContent" => Ok((model, verb)),
        _ => Err(format!(
            "unknown verb {verb:?} - this lane serves :generateContent and \
             :streamGenerateContent"
        )),
    }
}

/// POST `/v1beta/models/{model}:{verb}` - Gemini generateContent
/// translate (the catch-all segment carries both).
#[allow(clippy::too_many_lines)]
pub async fn generate_content(
    State(state): State<Arc<crate::state::AppState>>,
    axum::extract::Path(model_verb): axum::extract::Path<String>,
    key_ext: Option<axum::extract::Extension<crate::keys::KeyCtx>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let (model, verb) = match parse_model_verb(&model_verb) {
        Ok(pair) => pair,
        Err(msg) => return gemini_error(404, "NOT_FOUND", &msg),
    };
    let mut parsed: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => {
            return gemini_error(400, "INVALID_ARGUMENT", &format!("invalid JSON: {e}"));
        }
    };
    let mut body = body;
    // Model pools before the translate-lane gates: the member string
    // is what scope/admission and the local translator key off. This
    // lane translates local engines only - a pool pick that lands on a
    // remote member refuses with the pool named (same posture as the
    // Anthropic lane, one lane, one contract). Gemini bodies carry no
    // model field (the model rides the URL), so the translated OpenAI
    // body below stamps the effective member itself.
    let requested_model = model.clone();
    let wf = crate::pools::workflow_hint(&headers).map(|h| h.id);
    let mut model = model;
    if let Some((effective, rewritten)) =
        crate::pools::expand_request(&state, &model, &body, wf.as_deref())
    {
        if crate::remotes::split_remote(&effective, &state.cfg()).is_some() {
            return gemini_error(
                400,
                "INVALID_ARGUMENT",
                &format!(
                    "model pool {model:?} picked remote member {effective:?} - the Gemini \
                     generateContent translate lane serves the local engine only; call the \
                     member directly or pin a local member"
                ),
            );
        }
        if effective != model {
            tracing::info!(target: "blazar::pools", member = %effective, "gemini generateContent: pool routed to member");
            model = effective;
            // The pool rewrite stamps the member into the body
            // (anthropic-style); the gemini schema has no model field
            // but tolerates the extra key - re-parse defensively.
            body = rewritten;
            if let Ok(v) = serde_json::from_slice(&body) {
                parsed = v;
            }
        }
    }
    if crate::remotes::split_remote(&model, &state.cfg()).is_some() {
        return gemini_error(
            400,
            "INVALID_ARGUMENT",
            "the Gemini generateContent translate lane serves the local engine only; \
             remote-prefixed models are not translated (their dialect is the remote's own)",
        );
    }
    // Pool attribution (see the OpenAI/ollama/anthropic lanes): stamp
    // the member that actually served so the usage meter double-rows
    // the request.
    let pool_member = (model != requested_model).then(|| model.clone());
    // Failover chains (Gemini dialect): the translate lane serves
    // LOCAL targets only - a remote hop would need dialect
    // re-translation both ways, which this lane refuses by design.
    // The re-entry rewrites the model in the PATH (Gemini bodies do
    // not carry the model).
    if !state.failover.is_empty()
        && !headers.contains_key(crate::failover::ATTEMPT_HEADER)
        && let Some(mut attempt) = state.failover.plan(&model)
    {
        loop {
            if attempt.label.starts_with("remote:") {
                return gemini_error(
                    400,
                    "INVALID_ARGUMENT",
                    &format!(
                        "failover chain {:?} includes remote targets; the Gemini \
                         generateContent translate lane serves local targets only - use the \
                         chain on the OpenAI or ollama lane, or keep its targets local",
                        attempt.chain
                    ),
                );
            }
            let mut chain_headers = headers.clone();
            if let Ok(v) =
                axum::http::HeaderValue::from_str(&format!("{}:{}", attempt.chain, attempt.index))
            {
                chain_headers.insert(crate::failover::ATTEMPT_HEADER, v);
            }
            let chain_path = format!("{}:{verb}", attempt.serve_model);
            let mut resp = Box::pin(generate_content(
                axum::extract::State(state.clone()),
                axum::extract::Path(chain_path),
                key_ext.clone(),
                chain_headers,
                body.clone(),
            ))
            .await;
            if crate::failover::escalate_response(&mut resp).await {
                if let Some(sw) = state.failover.record_failure(&model, attempt.index) {
                    state
                        .bus
                        .publish(blazar_runtime::BlazarEvent::FailoverSwitched {
                            chain: sw.attempt.chain.clone(),
                            from: sw.benched_index,
                            to: sw.attempt.index,
                            reason: format!(
                                "target {} failed with HTTP {} - benched, escalating",
                                attempt.label,
                                resp.status().as_u16()
                            ),
                        });
                    attempt = sw.attempt;
                    continue;
                }
            } else if let Some((from, to)) = state.failover.record_success(&model, attempt.index) {
                state
                    .bus
                    .publish(blazar_runtime::BlazarEvent::FailoverSwitched {
                        chain: attempt.chain.clone(),
                        from,
                        to,
                        reason: format!("served by target {} ({})", attempt.index, attempt.label),
                    });
            }
            crate::failover::stamp(&mut resp, &attempt);
            return resp;
        }
    }
    // The translate lane serves ONE candidate: engines differ on `n`
    // and the OpenAI child path would either drop the extra candidates
    // or change sampling - both silently. Teach instead.
    if parsed
        .pointer("/generationConfig/candidateCount")
        .and_then(Value::as_u64)
        .is_some_and(|n| n > 1)
    {
        return gemini_error(
            400,
            "INVALID_ARGUMENT",
            "candidateCount > 1 is not served on the Gemini translate lane - it would \
             silently drop candidates; call n times, or use the OpenAI surface with n",
        );
    }
    // topK has no OpenAI chat equivalent: warn-and-drop, never silent.
    if let Some(k) = parsed
        .pointer("/generationConfig/topK")
        .and_then(Value::as_u64)
    {
        tracing::warn!(target: "blazar::gemini", top_k = k, "generationConfig.topK has no OpenAI chat equivalent - dropped");
    }
    // The stream flag comes from the URL VERB (that is the Gemini
    // wire), not the body.
    let stream = verb == "streamGenerateContent";
    let mut openai_body = match translate_request_gemini(&parsed, &model, stream) {
        Ok(b) => b,
        Err(msg) => return gemini_error(400, "INVALID_ARGUMENT", &msg),
    };
    // Best-of-N fan-out knob (blazar extension): header is the
    // dialect-universal spelling. The Gemini body has no best_of
    // field, so the header is the only entry point here.
    // Non-streaming only: a winner cannot be judged before completion.
    let best_of = match crate::bestof::resolve_best_of(&headers, None) {
        Ok(n) => n,
        Err(msg) => return gemini_error(400, "INVALID_ARGUMENT", &msg),
    };
    if best_of.is_some() && stream {
        return gemini_error(
            400,
            "INVALID_ARGUMENT",
            "best_of requires the non-streaming generateContent verb - a winner cannot be \
             judged before completion",
        );
    }
    // F53 + BLZ-3: atomic admission AFTER validation so 400s consume
    // no budget, with the gates and the charge in one lock step.
    if let Some(key) = key_ext.as_ref().map(|axum::extract::Extension(k)| k)
        && let Some(entry) = state.keys.entry(&key.name)
        && let Err(rej) = state.keys.admit(&entry, &model)
    {
        return rej.to_response();
    }
    // Strict token-budget lane (`reserve_budgets`): hold the completion
    // ceiling (`maxOutputTokens`, else the configured default; a
    // best-of-N fan scales it by the candidate count) before
    // generation. Settles from the child's raw usage below - both the
    // streaming tap and the buffered parse.
    let mut gemini_lease = None;
    if state.cfg().reserve_budgets
        && let Some(key) = key_ext.as_ref().map(|axum::extract::Extension(k)| k)
        && let Some(entry) = state.keys.entry(&key.name)
    {
        let base =
            crate::keys::completion_ceiling(&openai_body, state.cfg().reserve_default_tokens);
        let estimate = base.saturating_mul(best_of.filter(|n| *n >= 2).unwrap_or(1));
        match state.keys.reserve(&entry, estimate) {
            Ok(l) => gemini_lease = l,
            Err(rej) => return rej.to_response(),
        }
    }
    if let Some(err) = state.sentinel.strict_tool_def_error_cached(&openai_body) {
        return gemini_error(400, "INVALID_ARGUMENT", &format!("invalid tools: {err}"));
    }
    let eff = crate::preflight::admission_ctx(&state, &model);
    // A cold model with a straddling estimate is counted exactly once
    // loaded (right after the admission below).
    let recheck_prompt =
        match crate::preflight::enforce_prompt_fits(&state, &model, &openai_body, eff).await {
            Ok(needed) => needed,
            Err(resp) => return *resp,
        };
    let priority = Priority::from_header(
        headers
            .get("x-blazar-priority")
            .and_then(|v| v.to_str().ok()),
    );
    let prefix = crate::proxy::affinity_hash_bytes(
        &serde_json::to_vec(&openai_body).unwrap_or_else(|_| body.to_vec()),
    );
    let class = crate::workload::classify_for(
        &state,
        &model,
        crate::workload::WorkloadSurface::Chat,
        openai_body
            .get("tools")
            .is_some_and(serde_json::Value::is_array),
        false,
    );
    // Request-scope clock for the local service EWMA (both branches).
    let served_t0 = std::time::Instant::now();
    let (engine, _load_ms) = match ensure_with_admission(
        &state,
        &model,
        priority,
        class,
        prefix,
        crate::proxy::body_needs_vision(&openai_body, false),
        false, // text surface: diffusion rows teach the images lane
        false,
        true, // generateContent: generation surface (llama.cpp variant ride)
    )
    .await
    {
        Ok(ok) => ok,
        Err(resp) => return *resp,
    };
    if recheck_prompt
        && let Err(resp) =
            crate::preflight::recheck_prompt_fits(&state, &model, &openai_body, &engine).await
    {
        return *resp;
    }
    // llama.cpp LoRA-variant ride (see proxy.rs::llamacpp_variant_lora):
    // `base+adapter` on the resident base child carries the per-request
    // `lora` field and the engine's own model name in the translated
    // OpenAI body. Teaching surfaces as 502 upstream - never a silent
    // fall back to base weights.
    match crate::proxy::llamacpp_variant_lora(&state, &engine, Some(&model)).await {
        Some(Ok(pairs)) => {
            crate::proxy::set_child_lora(&mut openai_body, &pairs);
            crate::proxy::set_child_model(&mut openai_body, &engine.name);
        }
        Some(Err(teach)) => return gemini_error(502, "UNAVAILABLE", &teach),
        None => {}
    }
    let key_name = engine.name.clone();
    state.sup.note_prefix_hit(&key_name);
    // Think-dialect bridge for mistral.rs children (see proxy.rs): the
    // translated body speaks the kwargs dialect, which mistral.rs ignores
    // - mirror the think state into the top-level fields it reads.
    crate::proxy::normalize_think_in_value(engine.kind, &mut openai_body);
    let deadline_ms = headers
        .get("x-blazar-deadline-ms")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok());
    let guard = match admission_gate_slo(
        &state,
        &key_name,
        priority,
        class,
        deadline_ms,
        body.len(),
        key_ext
            .as_ref()
            .map(|axum::extract::Extension(k)| (k.name.as_str(), k.weight)),
        // Gemini generationConfig has no seed in common use; when one
        // rides through, the translated body carries it and the
        // reproduce-exclusivity check applies.
        openai_body.get("seed").is_some(),
    )
    .await
    {
        Ok(g) => g,
        Err(resp) => return *resp,
    };
    let url = format!("{}/v1/chat/completions", child_base(&engine.endpoint));
    // Gateway-mediated MCP tools (opt-in via `x-blazar-mcp` header -
    // the Gemini body has no selector field). The loop is buffered
    // this release, so a streaming request that opts in is taught the
    // constraint instead of silently degrading.
    let mcp_sel = match state
        .mcp
        .resolve(crate::mcp::selector_read(&headers, &Value::Null))
    {
        Err(msg) => return gemini_error(400, "INVALID_ARGUMENT", &msg),
        Ok(sel) => sel,
    };
    if stream && mcp_sel.is_some() {
        return gemini_error(
            400,
            "INVALID_ARGUMENT",
            "mcp tool loop requires the non-streaming generateContent verb in this release - \
             the gateway mediates buffered tool rounds",
        );
    }
    // F44: pooled client (10-min total timeout) instead of a per-request
    // build - same transport every other child lane uses (socket-pinned
    // for unix children).
    let client = crate::state::child_client(&state, &engine.endpoint);
    let send = crate::proxy::child_auth(
        client
            .post(&url)
            .header("content-type", "application/json")
            .body(serde_json::to_vec(&openai_body).unwrap_or_default()),
        &engine,
    );
    if stream {
        let resp = match send.send().await {
            Ok(r) => r,
            Err(e) => return gemini_error(502, "UNAVAILABLE", &format!("engine: {e}")),
        };
        // F45: a non-2xx child body is a JSON error, not an SSE stream -
        // surfacing it as 200+empty-events hangs clients. Translate the
        // status instead of framing the error bytes as events.
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            let detail = serde_json::from_str::<serde_json::Value>(&body)
                .ok()
                .and_then(|v| v["error"]["message"].as_str().map(str::to_string))
                .unwrap_or_else(|| body.chars().take(200).collect());
            return gemini_error(502, "UNAVAILABLE", &format!("engine {status}: {detail}"));
        }
        let model_label = model.clone();
        // Token accounting rides the RAW child chunks (openai dialect
        // in the SSE frames - the translated Gemini chunks carry a
        // different shape the sniffer does not parse): constructed only
        // when the key carries budgets, exactly like the proxy lanes.
        // Frame counting is on: this leg always streams SSE, and a
        // child that never emits a usage chunk still settles budgets
        // from the observed token frames.
        let budget_sniffer = key_ext
            .as_ref()
            .map(|axum::extract::Extension(k)| k.name.clone())
            .and_then(|name| state.keys.entry(&name))
            .filter(|e| e.tpm > 0 || e.daily_tokens > 0)
            .map(|e| crate::keys::UsageSniffer::new(&e.name).with_frame_counting(true));
        let child_stream = if budget_sniffer.is_some() || gemini_lease.is_some() {
            futures::StreamExt::boxed(crate::keys::tap_usage(
                resp.bytes_stream(),
                budget_sniffer,
                std::sync::Arc::clone(&state.keys),
                gemini_lease,
                Some(crate::keys::LocalServiceStamp::from_started(
                    std::sync::Arc::clone(&state.service_ewma_local),
                    &engine.name,
                    served_t0,
                )),
            ))
        } else {
            // Keyless traffic still feeds the local service EWMA
            // (stamp-only tap).
            futures::StreamExt::boxed(crate::keys::tap_usage(
                resp.bytes_stream(),
                None,
                std::sync::Arc::clone(&state.keys),
                None,
                Some(crate::keys::LocalServiceStamp::from_started(
                    std::sync::Arc::clone(&state.service_ewma_local),
                    &engine.name,
                    served_t0,
                )),
            ))
        };
        let events = gemini_sse_stream(child_stream, GeminiStreamState::new(&model_label), guard);
        let mut resp = Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "text/event-stream")
            .body(Body::from_stream(events))
            .unwrap_or_else(|e| gemini_error(500, "INTERNAL", &format!("stream: {e}")));
        if let Some(member) = &pool_member
            && let Ok(v) = axum::http::HeaderValue::from_str(member)
        {
            resp.headers_mut()
                .insert(crate::pools::EFFECTIVE_MODEL_HEADER, v);
        }
        return resp;
    }
    // Best-of-N: judge N candidates on the translated child body, return
    // the winner as a normal child response (usage summed across
    // candidates). Skip/Degraded (knob off, guard collapse, first-copy
    // transport failure) falls through to the single-send path below; a
    // guard degrade still stamps the reason header.
    let mut bestof_hdr: Option<String> = None;
    // MCP receipt survives the dialect rebuild via this capture (see
    // `mcp::stamp`).
    let mut mcp_hdr: Option<String> = None;
    // A2: wall anchor for the decode-throughput observation below -
    // predates both the fan and the single send. Also the D5
    // class-routing anchor: the guard is held from here through the
    // note, so both live counts include this request (`others_peak`).
    let t0 = std::time::Instant::now();
    let entry_live = state.sup.model_live_streams(&model);
    let fan = if let Some(want) = best_of.filter(|n| *n >= 2) {
        crate::bestof::fan_out(
            &state,
            &engine,
            &url,
            &serde_json::to_vec(&openai_body).unwrap_or_default(),
            want,
        )
        .await
    } else {
        crate::bestof::FanOut::Skip
    };
    if let crate::bestof::FanOut::Degraded(h) = &fan {
        bestof_hdr = Some(h.clone());
    }
    // MCP mediation takes precedence over the fan on the requests that
    // opt in (the loop owns every child round; judging candidates on
    // top would double-call the engine).
    let mcp_child = if let Some(sel) = &mcp_sel {
        match crate::mcp::chat_via_mcp(&state, &engine, &mut openai_body, sel, key_ext.as_deref())
            .await
        {
            Ok(child) => Some(child),
            Err(resp) => return resp,
        }
    } else {
        None
    };
    let resp = match mcp_child {
        Some(child) => {
            mcp_hdr = child
                .headers()
                .get("x-blazar-mcp")
                .and_then(|v| v.to_str().ok())
                .map(str::to_string);
            child
        }
        None => match fan {
            crate::bestof::FanOut::Ran(outcome) => {
                bestof_hdr = Some(outcome.hdr);
                outcome.resp
            }
            _ => match send.send().await {
                Ok(r) => r,
                Err(e) => return gemini_error(502, "UNAVAILABLE", &format!("engine: {e}")),
            },
        },
    };
    let status = resp.status();
    let bytes = match resp.bytes().await {
        Ok(b) => b,
        Err(e) => return gemini_error(502, "UNAVAILABLE", &format!("engine: {e}")),
    };
    let openai: Value = match serde_json::from_slice(&bytes) {
        Ok(v) => v,
        Err(_) => {
            // Unparseable child body: the strict-lane lease drops here
            // -> full release (nothing sniffable was generated).
            return gemini_error(
                status.as_u16(),
                "UNAVAILABLE",
                &String::from_utf8_lossy(&bytes[..bytes.len().min(400)]),
            );
        }
    };
    // Token accounting for the buffered lane: charge the child's
    // reported usage - error bodies carry none and settle to zero. The
    // strict-lane reservation settles through the same numbers;
    // best-of fans and MCP loops arrive here with usage already
    // summed by their builders.
    if let Some(key) = key_ext.as_ref().map(|axum::extract::Extension(k)| k) {
        let total = openai
            .pointer("/usage/prompt_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0)
            .saturating_add(
                openai
                    .pointer("/usage/completion_tokens")
                    .and_then(Value::as_u64)
                    .unwrap_or(0),
            );
        match gemini_lease.take() {
            Some(l) => l.settle(total),
            None => state.keys.charge_tokens(&key.name, total),
        }
    }
    if !status.is_success() {
        let mut msg = openai
            .pointer("/error/message")
            .and_then(Value::as_str)
            .unwrap_or("engine error")
            .to_string();
        // Constraint-pinned chat failure: append the pin + workaround
        // teaching (signature-gated; see proxy::constraint_chat_failure_text).
        if let Some(teach) = crate::proxy::constraint_chat_failure_teaching(&state, &model, &msg) {
            msg = format!("{msg} - {teach}");
        }
        let g_status = match status.as_u16() {
            429 => "RESOURCE_EXHAUSTED",
            500..=599 => "UNAVAILABLE",
            _ => "FAILED_PRECONDITION",
        };
        let mut r = gemini_error(status.as_u16(), g_status, &msg);
        crate::bestof::stamp(&mut r, bestof_hdr.as_deref());
        crate::mcp::stamp(&mut r, mcp_hdr.as_deref());
        return r;
    }
    let translated = translate_response_gemini(&openai, &model);
    // Local service-time probe (buffered lane): measures the full
    // round trip from t0, stamps on scope drop when the response
    // hands off complete.
    let _service = crate::keys::LocalServiceStamp::from_started(
        std::sync::Arc::clone(&state.service_ewma_local),
        &engine.name,
        served_t0,
    );
    // A2: blend the completion's decode rate into the model's EWMA.
    if let Some(n) = openai
        .pointer("/usage/completion_tokens")
        .and_then(Value::as_u64)
    {
        state.sup.note_model_throughput(
            &model,
            n,
            t0.elapsed().as_secs_f64(),
            crate::proxy::others_peak(entry_live, state.sup.model_live_streams(&model)),
        );
    }
    let mut resp = (StatusCode::OK, axum::Json(translated)).into_response();
    crate::bestof::stamp(&mut resp, bestof_hdr.as_deref());
    crate::mcp::stamp(&mut resp, mcp_hdr.as_deref());
    if let Some(member) = &pool_member
        && let Ok(v) = axum::http::HeaderValue::from_str(member)
    {
        resp.headers_mut()
            .insert(crate::pools::EFFECTIVE_MODEL_HEADER, v);
    }
    resp
}

/// Gemini `generateContent` request -> `OpenAI` chat body.
///
/// `contents` turns map to `messages` (`role: "model"` -> assistant),
/// `systemInstruction` to `system`, `functionDeclarations` to `tools`,
/// `functionCallingConfig` to `tool_choice`, camelCase
/// `generationConfig` keys to their `OpenAI` chat spellings.
/// `functionCall` parts (model turns) become assistant `tool_calls`
/// with the NAME as the deterministic id; `functionResponse` parts
/// (user turns) become `tool` messages referencing that name (see the
/// module doc - Gemini parts carry no call ids).
#[allow(clippy::too_many_lines)]
pub fn translate_request_gemini(v: &Value, model: &str, stream: bool) -> Result<Value, String> {
    let contents = v
        .get("contents")
        .and_then(Value::as_array)
        .ok_or_else(|| "missing `contents` array".to_string())?;
    if contents.is_empty() {
        return Err("`contents` must hold at least one turn".to_string());
    }
    let mut messages: Vec<Value> = Vec::new();
    if let Some(sys) = system_instruction_text(v) {
        messages.push(json!({"role": "system", "content": sys}));
    }
    for (i, turn) in contents.iter().enumerate() {
        let role = turn.get("role").and_then(Value::as_str).unwrap_or("user");
        let oai_role = match role {
            "model" => "assistant",
            "user" => "user",
            other => {
                return Err(format!(
                    "contents[{i}].role {other:?} is not \"user\" or \"model\""
                ));
            }
        };
        let parts = turn
            .get("parts")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("contents[{i}] has no `parts` array"))?;
        let mut text = String::new();
        let mut content_parts: Vec<Value> = Vec::new();
        let mut tool_calls: Vec<Value> = Vec::new();
        let mut tool_results: Vec<Value> = Vec::new();
        for part in parts {
            if let Some(t) = part.get("text").and_then(Value::as_str) {
                if oai_role == "user" || tool_calls.is_empty() {
                    // A user turn (or a pre-tool-call assistant prefix)
                    // accumulates text; text after tool calls in one
                    // assistant turn is not representable - the engine
                    // dialect emits tool calls last.
                    if content_parts.is_empty() && tool_results.is_empty() {
                        text.push_str(t);
                    } else {
                        content_parts.push(json!({"type": "text", "text": t}));
                    }
                } else {
                    content_parts.push(json!({"type": "text", "text": t}));
                }
            } else if let Some(fc) = part
                .get("functionCall")
                .or_else(|| part.get("function_call"))
            {
                let name = fc
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let args = fc.get("args").or_else(|| fc.get("arguments")).cloned();
                tool_calls.push(json!({
                    "type": "function",
                    "id": name,
                    "function": {"name": name, "arguments": serde_json::to_string(&args.unwrap_or(json!({}))).unwrap_or_else(|_| "{}".into())}
                }));
            } else if let Some(fr) = part
                .get("functionResponse")
                .or_else(|| part.get("function_response"))
            {
                let name = fr
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let payload = fr.get("response").cloned().unwrap_or(json!({}));
                tool_results.push(json!({
                    "role": "tool",
                    "tool_call_id": name,
                    "content": serde_json::to_string(&payload).unwrap_or_else(|_| "{}".into())
                }));
            } else if let Some(data) = inline_image_b64(part) {
                match image_data_url(&data) {
                    Ok(data_url) => {
                        content_parts
                            .push(json!({"type": "image_url", "image_url": {"url": data_url}}));
                    }
                    Err(msg) => {
                        return Err(format!("contents[{i}] inline data: {msg}"));
                    }
                }
            } else {
                return Err(format!(
                    "contents[{i}] carries a part this lane does not translate \
                     (supported: text, functionCall, functionResponse, inlineData images)"
                ));
            }
        }
        if oai_role == "user" {
            // A turn that is ONLY functionResponses must emit just the
            // tool messages - a padding user message between the
            // assistant's tool_calls and the tool results breaks the
            // OpenAI turn contract engines enforce.
            if !text.is_empty() || !content_parts.is_empty() {
                let content = if content_parts.is_empty() {
                    Value::String(text)
                } else if !text.is_empty() {
                    let mut all = vec![json!({"type": "text", "text": text})];
                    all.extend(content_parts);
                    Value::Array(all)
                } else {
                    Value::Array(content_parts)
                };
                messages.push(json!({"role": "user", "content": content}));
            }
            messages.extend(tool_results);
        } else {
            let mut assistant = json!({});
            if !text.is_empty() || !content_parts.is_empty() {
                let content = if content_parts.is_empty() {
                    Value::String(text)
                } else if !text.is_empty() {
                    let mut all = vec![json!({"type": "text", "text": text})];
                    all.extend(content_parts);
                    Value::Array(all)
                } else {
                    Value::Array(content_parts)
                };
                assistant["content"] = content;
            }
            if !tool_calls.is_empty() {
                assistant["tool_calls"] = Value::Array(tool_calls);
            }
            if assistant.as_object().is_none_or(serde_json::Map::is_empty) {
                return Err(format!(
                    "contents[{i}] (model turn) carries no translatable content"
                ));
            }
            assistant["role"] = Value::String("assistant".into());
            messages.push(assistant);
        }
    }
    let mut body = json!({"model": model, "messages": messages, "stream": stream});
    if let Some(cfg) = v.get("generationConfig") {
        for (gk, ok) in [
            ("temperature", "temperature"),
            ("topP", "top_p"),
            ("maxOutputTokens", "max_tokens"),
            ("seed", "seed"),
        ] {
            if let Some(val) = cfg.get(gk)
                && !val.is_null()
            {
                body[ok] = val.clone();
            }
        }
        if let Some(stops) = cfg.get("stopSequences").and_then(Value::as_array) {
            let stops: Vec<Value> = stops
                .iter()
                .filter_map(|s| s.as_str().map(|v| Value::String(v.to_string())))
                .collect();
            if !stops.is_empty() {
                body["stop"] = Value::Array(stops);
            }
        }
        if cfg.get("responseMimeType").and_then(Value::as_str) == Some("application/json") {
            body["response_format"] = json!({"type": "json_object"});
        }
    }
    let declarations: Vec<Value> = v
        .get("tools")
        .and_then(Value::as_array)
        .map(|tools| {
            tools
                .iter()
                .filter_map(|t| {
                    t.get("functionDeclarations")
                        .or_else(|| t.get("function_declarations"))
                        .and_then(Value::as_array)
                })
                .flatten()
                .cloned()
                .collect()
        })
        .unwrap_or_default();
    if !declarations.is_empty() {
        let tools: Vec<Value> = declarations
            .iter()
            .map(|d| {
                json!({"type": "function", "function": {
                    "name": d.get("name").and_then(Value::as_str).unwrap_or_default(),
                    "description": d.get("description").and_then(Value::as_str).unwrap_or_default(),
                    "parameters": d.get("parameters").cloned().unwrap_or(json!({"type": "object"}))
                }})
            })
            .collect();
        body["tools"] = Value::Array(tools);
        // toolConfig.functionCallingConfig -> tool_choice (only with
        // tools declared; a tool_choice without tools is an error in
        // the child).
        if let Some(mode) = v
            .pointer("/toolConfig/functionCallingConfig/mode")
            .or_else(|| v.pointer("/tool_config/function_calling_config/mode"))
            .and_then(Value::as_str)
        {
            let allowed = v
                .pointer("/toolConfig/functionCallingConfig/allowedFunctionNames")
                .or_else(|| {
                    v.pointer("/tool_config/function_calling_config/allowed_function_names")
                })
                .and_then(Value::as_array)
                .map(|a| {
                    a.iter()
                        .filter_map(|n| n.as_str().map(str::to_string))
                        .collect::<Vec<String>>()
                })
                .unwrap_or_default();
            match mode {
                "NONE" => {
                    body.as_object_mut().map(|m| m.remove("tools"));
                }
                "ANY" => {
                    if allowed.len() == 1 {
                        body["tool_choice"] = json!({
                            "type": "function",
                            "function": {"name": allowed[0]}
                        });
                    } else {
                        // OpenAI has no any-of-N spelling; required with
                        // the full tool set is the closest contract.
                        body["tool_choice"] = json!({"type": "required"});
                    }
                }
                _ => {} // AUTO (default): omit tool_choice.
            }
        }
    }
    Ok(body)
}

/// `systemInstruction` -> plain system text (string or parts form).
fn system_instruction_text(v: &Value) -> Option<String> {
    let si = v
        .get("systemInstruction")
        .or_else(|| v.get("system_instruction"))?;
    if let Some(s) = si.as_str() {
        return (!s.is_empty()).then(|| s.to_string());
    }
    let parts = si.get("parts").and_then(Value::as_array)?;
    let text: String = parts
        .iter()
        .filter_map(|p| p.get("text").and_then(Value::as_str))
        .collect::<Vec<&str>>()
        .join("\n");
    (!text.is_empty()).then_some(text)
}

/// `inlineData` part -> base64 payload when it is an image (the
/// data-url builder carries the media prefix); other media kinds
/// return `None` and the caller's unknown branch teaches.
fn inline_image_b64(part: &Value) -> Option<String> {
    let d = part.get("inlineData").or_else(|| part.get("inline_data"))?;
    let mime = d.get("mimeType").or_else(|| d.get("mime_type"))?.as_str()?;
    if !mime.starts_with("image/") {
        return None;
    }
    d.get("data").and_then(Value::as_str).map(str::to_string)
}

/// `OpenAI` chat response -> Gemini `generateContent` response.
#[must_use]
pub fn translate_response_gemini(openai: &Value, model: &str) -> Value {
    let choice = openai.pointer("/choices/0").cloned().unwrap_or(Value::Null);
    let msg = choice.get("message").cloned().unwrap_or(Value::Null);
    let mut parts: Vec<Value> = Vec::new();
    // F47-style bridge: engines exposing reasoning surface it as a
    // Gemini thought part ahead of the text.
    if let Some(rc) = msg.get("reasoning_content").and_then(Value::as_str)
        && !rc.is_empty()
    {
        parts.push(json!({"text": rc, "thought": true}));
    }
    if let Some(text) = msg.get("content").and_then(Value::as_str)
        && !text.is_empty()
    {
        parts.push(json!({"text": text}));
    }
    if let Some(calls) = msg.get("tool_calls").and_then(Value::as_array) {
        for c in calls {
            let name = c
                .pointer("/function/name")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let args = c
                .pointer("/function/arguments")
                .and_then(Value::as_str)
                .and_then(|s| serde_json::from_str::<Value>(s).ok())
                .unwrap_or_else(|| json!({}));
            parts.push(json!({"functionCall": {"name": name, "args": args}}));
        }
    }
    if parts.is_empty() {
        parts.push(json!({"text": ""}));
    }
    let finish = choice
        .get("finish_reason")
        .and_then(Value::as_str)
        .unwrap_or("stop");
    let finish_reason = match finish {
        "length" => "MAX_TOKENS",
        "content_filter" => "SAFETY",
        // Gemini signals tool calls with STOP; the client keys on the
        // functionCall part, not the reason.
        _ => "STOP",
    };
    let mut resp = json!({
        "candidates": [{
            "content": {"role": "model", "parts": parts},
            "finishReason": finish_reason,
            "index": 0
        }],
        "usageMetadata": gemini_usage(openai.pointer("/usage")),
        "modelVersion": model
    });
    if let Some(id) = openai.get("id").and_then(Value::as_str)
        && !id.is_empty()
    {
        resp["responseId"] = Value::String(id.to_string());
    }
    resp
}

/// `OpenAI` usage object -> Gemini `usageMetadata`. `cached prompt`
/// reads map to `cachedContentTokenCount` (llama-server reports cache
/// reads only, so the tier is read-only here too).
#[must_use]
pub fn gemini_usage(openai_usage: Option<&Value>) -> Value {
    let u = openai_usage.unwrap_or(&Value::Null);
    let prompt = u.get("prompt_tokens").and_then(Value::as_u64).unwrap_or(0);
    let completion = u
        .get("completion_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let cached = crate::translate::cached_prompt_tokens(Some(u));
    json!({
        "promptTokenCount": prompt,
        "candidatesTokenCount": completion,
        "totalTokenCount": prompt.saturating_add(completion),
        "cachedContentTokenCount": cached
    })
}

/// Streaming state for the Gemini `SSE` translation: the model label,
/// captured prompt tokens, open tool-call accumulators (`OpenAI`
/// fragments arguments across deltas; Gemini ships each
/// `functionCall` part COMPLETE, so fragments accumulate here and
/// flush in the final chunk), and the done marker for the truncation
/// guard.
#[derive(Debug)]
pub struct GeminiStreamState {
    model: String,
    prompt_tokens: u64,
    tools: std::collections::BTreeMap<u64, (String, String)>,
    done: bool,
}

impl GeminiStreamState {
    fn new(model: &str) -> Self {
        Self {
            model: model.to_string(),
            prompt_tokens: 0,
            tools: std::collections::BTreeMap::new(),
            done: false,
        }
    }
}

/// One `OpenAI` `SSE` chunk -> zero or more Gemini `SSE` data frames
/// (each returned `Value` is a complete `GenerateContentResponse`).
#[allow(clippy::too_many_lines)]
pub fn chunk_events_gemini(chunk: &Value, state: &mut GeminiStreamState) -> Vec<Value> {
    let mut frames: Vec<Value> = Vec::new();
    if let Some(p) = chunk
        .pointer("/usage/prompt_tokens")
        .and_then(Value::as_u64)
    {
        state.prompt_tokens = state.prompt_tokens.max(p);
    }
    let choice = chunk.pointer("/choices/0").cloned().unwrap_or(Value::Null);
    let mut parts: Vec<Value> = Vec::new();
    if let Some(rc) = choice
        .pointer("/delta/reasoning_content")
        .and_then(Value::as_str)
        && !rc.is_empty()
    {
        parts.push(json!({"text": rc, "thought": true}));
    }
    if let Some(text) = choice.pointer("/delta/content").and_then(Value::as_str)
        && !text.is_empty()
    {
        parts.push(json!({"text": text}));
    }
    if let Some(calls) = choice
        .pointer("/delta/tool_calls")
        .and_then(Value::as_array)
    {
        for c in calls {
            let idx = c.get("index").and_then(Value::as_u64).unwrap_or(0);
            let entry = state
                .tools
                .entry(idx)
                .or_insert_with(|| (String::new(), String::new()));
            if let Some(name) = c.pointer("/function/name").and_then(Value::as_str)
                && !name.is_empty()
            {
                entry.0 = name.to_string();
            }
            if let Some(frag) = c.pointer("/function/arguments").and_then(Value::as_str)
                && !frag.is_empty()
            {
                entry.1.push_str(frag);
            }
        }
    }
    let finish = choice.get("finish_reason").and_then(Value::as_str);
    let usage = chunk.get("usage").filter(|u| !u.is_null());
    if !parts.is_empty() {
        let mut frame = json!({
            "candidates": [{"content": {"role": "model", "parts": parts}, "index": 0}],
            "modelVersion": state.model
        });
        if let Some(u) = usage {
            frame["usageMetadata"] = gemini_usage(Some(u));
        }
        frames.push(frame);
    }
    if let Some(reason) = finish
        && !state.done
    {
        state.done = true;
        let finish_reason = match reason {
            "length" => "MAX_TOKENS",
            "content_filter" => "SAFETY",
            _ => "STOP",
        };
        // Flush accumulated tool calls as COMPLETE functionCall parts
        // in the terminal chunk (the Gemini wire has no partial-args
        // protocol).
        let mut final_parts: Vec<Value> = Vec::new();
        for (name, args) in state.tools.values() {
            let parsed_args = serde_json::from_str::<Value>(args).unwrap_or_else(|_| json!({}));
            final_parts.push(json!({"functionCall": {"name": name, "args": parsed_args}}));
        }
        // Terminal frame always carries usageMetadata: the child's own
        // usage when it sent one, else the observed prompt count with
        // an honest zero for the unmeasured completion side.
        let usage_metadata = match usage {
            Some(u) => gemini_usage(Some(u)),
            None => json!({
                "promptTokenCount": state.prompt_tokens,
                "candidatesTokenCount": 0,
                "totalTokenCount": state.prompt_tokens,
                "cachedContentTokenCount": 0
            }),
        };
        frames.push(json!({
            "candidates": [{
                "content": {"role": "model", "parts": final_parts},
                "finishReason": finish_reason,
                "index": 0
            }],
            "modelVersion": state.model,
            "usageMetadata": usage_metadata
        }));
    }
    frames
}

/// Child `OpenAI` `SSE` -> Gemini `SSE` data frames. The in-flight
/// guard rides in the stream state so accounting holds until the last
/// frame ships. Gemini `SSE` frames are plain `data: {json}` lines -
/// no `event:` prefix.
fn gemini_sse_stream<S>(
    child: S,
    st: GeminiStreamState,
    guard: crate::proxy::InFlightGuard,
) -> impl futures::Stream<Item = Result<Bytes, std::io::Error>>
where
    S: futures::Stream<Item = Result<Bytes, reqwest::Error>> + Unpin,
{
    use std::collections::VecDeque;
    futures::stream::unfold(
        (
            child,
            st,
            String::new(),
            crate::translate::LineBuffer::new(),
            VecDeque::new(),
            Some(guard),
        ),
        |(mut child, mut st, mut buf, mut lines, mut queue, guard)| async move {
            loop {
                if let Some(frame) = queue.pop_front() {
                    return Some((Ok(frame), (child, st, buf, lines, queue, guard)));
                }
                match child.next().await {
                    Some(Ok(chunk)) => {
                        buf.push_str(&lines.feed(&chunk));
                        while let Some(pos) = buf.find('\n') {
                            let line: String = buf.drain(..=pos).collect();
                            let line = line.trim();
                            let Some(payload) = line.strip_prefix("data: ") else {
                                continue;
                            };
                            if payload == "[DONE]" {
                                continue;
                            }
                            let Ok(v) = serde_json::from_str::<Value>(payload) else {
                                continue;
                            };
                            for frame in chunk_events_gemini(&v, &mut st) {
                                queue.push_back(Bytes::from(format!("data: {frame}\n\n")));
                            }
                        }
                    }
                    Some(Err(e)) => {
                        let frame = Bytes::from(format!(
                            "data: {}\n\n",
                            json!({"error": {"code": 502, "message": e.to_string(),
                                   "status": "UNAVAILABLE"}})
                        ));
                        queue.push_back(frame);
                    }
                    None => {
                        // F51-style guard: child ended without a finish
                        // chunk - emit a terminal error frame so clients
                        // see the truncation instead of a silent bare
                        // EOF.
                        if !st.done {
                            st.done = true;
                            queue.push_back(Bytes::from(format!(
                                "data: {}\n\n",
                                json!({"error": {"code": 502, "message":
                                       "upstream stream ended before completion",
                                       "status": "UNAVAILABLE"}})
                            )));
                            continue;
                        }
                        return None;
                    }
                }
            }
        },
    )
}

/// Gemini error envelope: `{"error": {code, message, status}}` where
/// `status` is the googleapi canonical code string.
fn gemini_error(status: u16, g_status: &str, message: &str) -> Response {
    let body = json!({"error": {"code": status, "message": message, "status": g_status}});
    Response::builder()
        .status(StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR))
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap_or_else(|_| Response::new(Body::from("{}")))
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    #[test]
    fn unit__parse_model_verb__splits_model_and_verb() {
        let (m, v) = parse_model_verb("qwen3:generateContent").expect("plain split");
        assert_eq!((m.as_str(), v.as_str()), ("qwen3", "generateContent"));
        // Tag-style model names: the verb is the LAST segment.
        let (m, v) = parse_model_verb("llama3:8b:streamGenerateContent").expect("tag split");
        assert_eq!(
            (m.as_str(), v.as_str()),
            ("llama3:8b", "streamGenerateContent")
        );
        assert!(parse_model_verb("qwen3").is_err(), "no verb");
        assert!(
            parse_model_verb("qwen3:countTokens").is_err(),
            "unknown verb"
        );
        assert!(parse_model_verb(":generateContent").is_err(), "empty model");
    }

    #[test]
    fn unit__translate_request_gemini__contents_system_and_config() {
        let g = json!({
            "systemInstruction": {"parts": [{"text": "be terse"}]},
            "contents": [
                {"role": "user", "parts": [{"text": "hello "}, {"text": "world"}]},
                {"role": "model", "parts": [{"text": "hi"}]},
                {"role": "user", "parts": [{"text": "go"}]}
            ],
            "generationConfig": {
                "temperature": 0.2, "topP": 0.9, "maxOutputTokens": 128,
                "stopSequences": ["END"], "candidateCount": 1
            }
        });
        let out = translate_request_gemini(&g, "qwen3", false).expect("translate");
        assert_eq!(
            out.pointer("/messages/0"),
            Some(&json!({"role": "system", "content": "be terse"}))
        );
        // Two text parts in one user turn concatenate (single string
        // content keeps the engine dialect canonical).
        assert_eq!(
            out.pointer("/messages/1/content"),
            Some(&json!("hello world"))
        );
        assert_eq!(out.pointer("/messages/2/content"), Some(&json!("hi")));
        assert_eq!(out.pointer("/messages/3/content"), Some(&json!("go")));
        assert_eq!(out.pointer("/model"), Some(&json!("qwen3")));
        assert_eq!(out.pointer("/stream"), Some(&json!(false)));
        assert_eq!(out.pointer("/temperature"), Some(&json!(0.2)));
        assert_eq!(out.pointer("/top_p"), Some(&json!(0.9)));
        assert_eq!(out.pointer("/max_tokens"), Some(&json!(128)));
        assert_eq!(out.pointer("/stop"), Some(&json!(["END"])));
    }

    #[test]
    fn unit__translate_request_gemini__function_round_trip_and_tool_config() {
        let g = json!({
            "contents": [
                {"role": "user", "parts": [{"text": "weather?"}]},
                {"role": "model", "parts": [{"functionCall":
                    {"name": "get_weather", "args": {"city": "sf"}}}]},
                {"role": "user", "parts": [{"functionResponse":
                    {"name": "get_weather", "response": {"temp": 60}}}]}
            ],
            "tools": [{"functionDeclarations": [{"name": "get_weather",
                "description": "weather", "parameters": {"type": "object"}}]}],
            "toolConfig": {"functionCallingConfig": {"mode": "ANY",
                "allowedFunctionNames": ["get_weather"]}}
        });
        let out = translate_request_gemini(&g, "m1", false).expect("translate");
        assert_eq!(
            out.pointer("/tools/0/function/name"),
            Some(&json!("get_weather"))
        );
        assert_eq!(
            out.pointer("/tool_choice/function/name"),
            Some(&json!("get_weather"))
        );
        // functionCall -> assistant tool_call with the NAME as the id
        // (messages: [1] = the model turn carrying tool_calls).
        assert_eq!(
            out.pointer("/messages/1/tool_calls/0/id"),
            Some(&json!("get_weather"))
        );
        assert_eq!(
            out.pointer("/messages/1/tool_calls/0/function/arguments"),
            Some(&json!("{\"city\":\"sf\"}"))
        );
        // functionResponse -> tool message keyed by the same name
        // ([2] = the tool result turn).
        assert_eq!(out.pointer("/messages/2/role"), Some(&json!("tool")));
        assert_eq!(
            out.pointer("/messages/2/tool_call_id"),
            Some(&json!("get_weather"))
        );
        assert_eq!(
            out.pointer("/messages/2/content"),
            Some(&json!("{\"temp\":60}"))
        );
    }

    #[test]
    fn unit__translate_request_gemini__refuses_empty_and_unknown_shapes() {
        assert!(
            translate_request_gemini(&json!({}), "m", false).is_err(),
            "no contents"
        );
        let bad_role = json!({"contents": [{"role": "system", "parts": [{"text": "x"}]}]});
        assert!(
            translate_request_gemini(&bad_role, "m", false).is_err(),
            "bad role"
        );
        let no_parts = json!({"contents": [{"role": "user"}]});
        assert!(
            translate_request_gemini(&no_parts, "m", false).is_err(),
            "no parts"
        );
    }

    #[test]
    fn unit__translate_response_gemini__text_tools_and_finish_reasons() {
        let openai = json!({
            "id": "chatcmpl-42",
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "content": "calling",
                "tool_calls": [{"function": {"name": "get_weather",
                    "arguments": "{\"city\":\"sf\"}"}}]
            }}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 7}
        });
        let out = translate_response_gemini(&openai, "m1");
        assert_eq!(
            out.pointer("/candidates/0/finishReason"),
            Some(&json!("STOP"))
        );
        assert_eq!(
            out.pointer("/candidates/0/content/parts/0/text"),
            Some(&json!("calling"))
        );
        assert_eq!(
            out.pointer("/candidates/0/content/parts/1/functionCall/name"),
            Some(&json!("get_weather"))
        );
        assert_eq!(
            out.pointer("/candidates/0/content/parts/1/functionCall/args/city"),
            Some(&json!("sf"))
        );
        assert_eq!(
            out.pointer("/usageMetadata/promptTokenCount"),
            Some(&json!(12))
        );
        assert_eq!(
            out.pointer("/usageMetadata/candidatesTokenCount"),
            Some(&json!(7))
        );
        assert_eq!(out.pointer("/responseId"), Some(&json!("chatcmpl-42")));
        // length -> MAX_TOKENS
        let long = json!({"choices": [{"finish_reason": "length", "message": {"content": "x"}}]});
        assert_eq!(
            translate_response_gemini(&long, "m").pointer("/candidates/0/finishReason"),
            Some(&json!("MAX_TOKENS"))
        );
    }

    #[test]
    fn unit__chunk_events_gemini__text_deltas_then_finish() {
        let mut st = GeminiStreamState::new("m1");
        let d1 = json!({"choices": [{"delta": {"content": "one "}}]});
        let d2 = json!({"choices": [{"delta": {"content": "two"}}],
                        "usage": {"prompt_tokens": 5}});
        let fin = json!({"choices": [{"finish_reason": "stop", "delta": {}}],
                         "usage": {"prompt_tokens": 5, "completion_tokens": 2}});
        let f1 = chunk_events_gemini(&d1, &mut st);
        assert_eq!(f1.len(), 1);
        assert_eq!(
            f1[0].pointer("/candidates/0/content/parts/0/text"),
            Some(&json!("one "))
        );
        assert!(f1[0].pointer("/candidates/0/finishReason").is_none());
        let _ = chunk_events_gemini(&d2, &mut st);
        let ff = chunk_events_gemini(&fin, &mut st);
        assert_eq!(ff.len(), 1);
        assert_eq!(
            ff[0].pointer("/candidates/0/finishReason"),
            Some(&json!("STOP"))
        );
        assert_eq!(
            ff[0].pointer("/usageMetadata/totalTokenCount"),
            Some(&json!(7))
        );
        // A second finish chunk emits nothing (done marker).
        assert!(chunk_events_gemini(&fin, &mut st).is_empty());
    }

    #[test]
    fn unit__chunk_events_gemini__accumulates_fragmented_tool_arguments() {
        let mut st = GeminiStreamState::new("m1");
        let head = json!({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"name": "get_weather", "arguments": "{\"ci"}}]}}]});
        let tail = json!({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": "ty\":\"sf\"}"}}]}}]});
        let fin = json!({"choices": [{"finish_reason": "tool_calls", "delta": {}}]});
        // The finish chunk in the loop emits the flush frame; capture
        // it there (a second finish call would hit the done marker).
        let mut fin_frame: Vec<Value> = Vec::new();
        for (i, c) in [&head, &tail, &fin].iter().enumerate() {
            let frames = chunk_events_gemini(c, &mut st);
            if i < 2 {
                assert!(frames.is_empty(), "no data frames while args accumulate");
            } else {
                fin_frame = frames;
            }
        }
        // The finish flush carries ONE complete functionCall part.
        assert_eq!(fin_frame.len(), 1);
        assert_eq!(
            fin_frame[0].pointer("/candidates/0/content/parts/0/functionCall/args/city"),
            Some(&json!("sf"))
        );
        assert_eq!(
            fin_frame[0].pointer("/candidates/0/finishReason"),
            Some(&json!("STOP"))
        );
    }

    #[test]
    fn unit__gemini_usage__maps_openai_counts_and_cache() {
        let u = gemini_usage(Some(&json!({
            "prompt_tokens": 10, "completion_tokens": 4,
            "prompt_tokens_details": {"cached_tokens": 6}
        })));
        assert_eq!(u.pointer("/promptTokenCount"), Some(&json!(10)));
        assert_eq!(u.pointer("/candidatesTokenCount"), Some(&json!(4)));
        assert_eq!(u.pointer("/totalTokenCount"), Some(&json!(14)));
        assert_eq!(u.pointer("/cachedContentTokenCount"), Some(&json!(6)));
        let empty = gemini_usage(None);
        assert_eq!(empty.pointer("/promptTokenCount"), Some(&json!(0)));
    }
}
