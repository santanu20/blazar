//! Stable, machine-readable error codes for the gateway's OpenAI-shaped
//! error envelope. Every error rendered through `proxy::openai_error`
//! (and the auth/key middleware refusals) carries a `blazar_code`
//! string that is stable across releases, unlike free-form messages
//! and HTTP statuses which can shift with routing internals.
//!
//! Layering:
//! - Typed choke points (`supervision_error`, preflight refusals,
//!   auth/key middleware) emit PRECISE codes (`MODEL_NOT_FOUND`,
//!   `ENGINE_CIRCUIT_OPEN`, ...).
//! - Every other `openai_error(status, msg)` site derives a GENERIC
//!   but still stable code from the HTTP status (`BAD_REQUEST`,
//!   `INTERNAL`, ...), so the field is present on every path.
//!
//! The catalog is self-describing over HTTP at `GET /api/errors`.
//! Codes are ADDITIVE ONLY: a code, once shipped, keeps its string,
//! its HTTP status, and its meaning. `docs/error-codes.md` mirrors
//! this table; a unit test pins the two together.

use axum::response::IntoResponse;
use serde_json::json;

/// Bump when the catalog shape (not individual code additions) changes.
pub const CATALOG_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlazarCode {
    // Precise: supervision lifecycle (proxy::supervision_error).
    ModelNotFound,
    ModelUnsupportedEngine,
    ModelLoadTimeout,
    EngineCircuitOpen,
    AllSlotsBusy,
    ModelTooLarge,
    InsufficientMemory,
    EngineCrashed,
    RpcEndpointUnreachable,
    Internal,
    // Precise: admission / preflight teaching refusals.
    PromptTooLong,
    VisionProjectorMissing,
    CapabilityVerifiedFailed,
    DiffusionTextRefusal,
    // Precise: gateway middleware (auth, key budgets, host guard).
    Unauthorized,
    KeyScopeForbidden,
    HostForbidden,
    RateLimited,
    FleetTokenRequired,
    // Generic status-derived fallbacks (openai_error without a typed map).
    BadRequest,
    NotFound,
    Forbidden,
    PayloadTooLarge,
    Unprocessable,
    BadGateway,
    Unavailable,
    GatewayTimeout,
    InsufficientStorage,
    // Method/content-type mismatches axum itself rejects before any handler runs.
    MethodNotAllowed,
    UnsupportedMediaType,
    Conflict,
}

impl BlazarCode {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ModelNotFound => "MODEL_NOT_FOUND",
            Self::ModelUnsupportedEngine => "MODEL_UNSUPPORTED_ENGINE",
            Self::ModelLoadTimeout => "MODEL_LOAD_TIMEOUT",
            Self::EngineCircuitOpen => "ENGINE_CIRCUIT_OPEN",
            Self::AllSlotsBusy => "ALL_SLOTS_BUSY",
            Self::ModelTooLarge => "MODEL_TOO_LARGE",
            Self::InsufficientMemory => "INSUFFICIENT_MEMORY",
            Self::EngineCrashed => "ENGINE_CRASHED",
            Self::RpcEndpointUnreachable => "RPC_ENDPOINT_UNREACHABLE",
            Self::Internal => "INTERNAL",
            Self::PromptTooLong => "PROMPT_TOO_LONG",
            Self::VisionProjectorMissing => "VISION_PROJECTOR_MISSING",
            Self::CapabilityVerifiedFailed => "CAPABILITY_VERIFIED_FAILED",
            Self::DiffusionTextRefusal => "DIFFUSION_TEXT_REFUSAL",
            Self::Unauthorized => "UNAUTHORIZED",
            Self::FleetTokenRequired => "FLEET_TOKEN_REQUIRED",
            Self::KeyScopeForbidden => "KEY_SCOPE_FORBIDDEN",
            Self::HostForbidden => "HOST_FORBIDDEN",
            Self::RateLimited => "RATE_LIMITED",
            Self::Forbidden => "FORBIDDEN",
            Self::BadRequest => "BAD_REQUEST",
            Self::NotFound => "NOT_FOUND",
            Self::MethodNotAllowed => "METHOD_NOT_ALLOWED",
            Self::UnsupportedMediaType => "UNSUPPORTED_MEDIA_TYPE",
            Self::Conflict => "CONFLICT",
            Self::PayloadTooLarge => "PAYLOAD_TOO_LARGE",
            Self::Unprocessable => "UNPROCESSABLE",
            Self::BadGateway => "BAD_GATEWAY",
            Self::Unavailable => "UNAVAILABLE",
            Self::GatewayTimeout => "GATEWAY_TIMEOUT",
            Self::InsufficientStorage => "INSUFFICIENT_STORAGE",
        }
    }

    /// HTTP status this code renders with. Pairings are part of the
    /// contract: a code never changes its status family.
    #[must_use]
    pub fn http(self) -> u16 {
        match self {
            Self::ModelNotFound | Self::NotFound => 404,
            Self::MethodNotAllowed => 405,
            Self::UnsupportedMediaType => 415,
            Self::Conflict => 409,
            Self::ModelUnsupportedEngine
            | Self::PromptTooLong
            | Self::VisionProjectorMissing
            | Self::CapabilityVerifiedFailed
            | Self::DiffusionTextRefusal
            | Self::BadRequest => 400,
            Self::Unauthorized | Self::FleetTokenRequired => 401,
            Self::KeyScopeForbidden | Self::HostForbidden | Self::Forbidden => 403,
            Self::RateLimited => 429,
            Self::PayloadTooLarge => 413,
            Self::Unprocessable => 422,
            Self::ModelLoadTimeout
            | Self::EngineCircuitOpen
            | Self::AllSlotsBusy
            | Self::ModelTooLarge
            | Self::RpcEndpointUnreachable
            | Self::Unavailable => 503,
            Self::EngineCrashed | Self::BadGateway => 502,
            Self::GatewayTimeout => 504,
            Self::InsufficientMemory | Self::InsufficientStorage => 507,
            Self::Internal => 500,
        }
    }

    /// One-line plain statement of what happened (no model-specific
    /// detail - that lives in `message`).
    #[must_use]
    pub fn description(self) -> &'static str {
        match self {
            Self::ModelNotFound => "no model with this name is installed",
            Self::ModelUnsupportedEngine => {
                "the model's format or quantization is not supported by the engine lane it reached"
            }
            Self::ModelLoadTimeout => "the model did not become healthy within model_load_timeout",
            Self::EngineCircuitOpen => {
                "the engine process keeps crashing and its circuit breaker is open"
            }
            Self::AllSlotsBusy => "all serving slots are busy; the bounded admission wait expired",
            Self::ModelTooLarge => {
                "the model cannot fit this machine even on an empty box (admission floor)"
            }
            Self::InsufficientMemory => {
                "the box lacks the memory right now; freeing co-resident engines would change the verdict"
            }
            Self::EngineCrashed => "the engine child process crashed while loading or serving",
            Self::RpcEndpointUnreachable => {
                "an rpc fleet worker endpoint named in rpc_servers is unreachable"
            }
            Self::Internal => "unexpected internal error",
            Self::PromptTooLong => {
                "the prompt exceeds the effective context window and would be silently truncated"
            }
            Self::VisionProjectorMissing => {
                "the request needs vision but the model has no projector sidecar attached"
            }
            Self::CapabilityVerifiedFailed => {
                "the model's verified capability certificate shows FAIL for a capability this request needs"
            }
            Self::DiffusionTextRefusal => {
                "a diffusion component-set model was addressed on a text/embedding surface"
            }
            Self::Unauthorized => "missing or invalid API key",
            Self::FleetTokenRequired => {
                "a fleet worker tried to join without the shared secret this gateway requires"
            }
            Self::KeyScopeForbidden => "the API key is not scoped for this model",
            Self::HostForbidden => {
                "unrecognized Host header on a loopback bind (possible DNS rebinding)"
            }
            Self::RateLimited => "the API key exceeded one of its budgets",
            Self::BadRequest => "malformed or invalid request",
            Self::NotFound => "resource not found",
            Self::Forbidden => "request refused by an authorization gate",
            Self::MethodNotAllowed => "the route exists but not for this HTTP method",
            Self::UnsupportedMediaType => "the endpoint requires an application/json body",
            Self::Conflict => "the resource already exists in the requested state",
            Self::PayloadTooLarge => "request body too large",
            Self::Unprocessable => "request semantically invalid for this surface",
            Self::BadGateway => "invalid response from the engine child",
            Self::Unavailable => "service temporarily unavailable",
            Self::GatewayTimeout => "gateway timeout",
            Self::InsufficientStorage => "insufficient storage or memory",
        }
    }

    /// Actionable next step, phrased against shipped commands only.
    #[must_use]
    pub fn remediation(self) -> &'static str {
        match self {
            Self::ModelNotFound => "pull the model first: `blazar pull <model>`",
            Self::ModelUnsupportedEngine => {
                "pick a sibling quant/format for this engine, or install/select another engine (see `blazar engine --help`)"
            }
            Self::ModelLoadTimeout => {
                "check `blazar ps` for the loading state; raise model_load_timeout if the box is slow"
            }
            Self::EngineCircuitOpen => {
                "inspect crash causes, then run `blazar ps --reset` to clear the breaker"
            }
            Self::AllSlotsBusy => "retry when capacity frees, or lower concurrent load",
            Self::ModelTooLarge => "use a smaller quantization or a machine with more VRAM/RAM",
            Self::InsufficientMemory => {
                "evict co-resident models (`blazar ps` shows them) or use a smaller quantization"
            }
            Self::EngineCrashed => {
                "check daemon logs for the child's stderr; `blazar ps` shows restart counts"
            }
            Self::RpcEndpointUnreachable => {
                "start the worker on that host (`blazar rpc worker`) or fix rpc_servers in the config"
            }
            Self::Internal => {
                "check daemon logs; report with the x-blazar-trace-id of the failed request"
            }
            Self::PromptTooLong => {
                "shorten the prompt, raise ctx ([model_overrides] or X-Blazar-Num-Ctx), or set prompt_preflight = false"
            }
            Self::VisionProjectorMissing => {
                "attach one while the model is stopped: `blazar mmproj <model> <mmproj.gguf path>`"
            }
            Self::CapabilityVerifiedFailed => {
                "pick a sibling whose certificate passes (`blazar scorecard <model>`), or re-verify: `blazar model-doctor <model>`"
            }
            Self::DiffusionTextRefusal => {
                "address the images/videos surface for this model, or use a text model"
            }
            Self::Unauthorized => "send Authorization: Bearer <key> or x-api-key",
            Self::FleetTokenRequired => {
                "start the worker with the gateway's fleet_token (blazar rpc worker --fleet-token <token>) or set fleet_open = true on a trusted network"
            }
            Self::KeyScopeForbidden => "use a key whose scopes cover this model",
            Self::HostForbidden => {
                "bind blazar to your LAN address, or set host to match the Host header you send"
            }
            Self::RateLimited => "retry after retry_after_secs, or raise the key's budget",
            Self::BadRequest => "fix the request body per the API surface docs",
            Self::NotFound => "check the resource id and route",
            Self::Forbidden => {
                "check the key scopes / fleet token the gate names, or use a management key"
            }
            Self::MethodNotAllowed => "check the Allow header this response carries",
            Self::UnsupportedMediaType => "send Content-Type: application/json",
            Self::Conflict => "delete or rename the existing resource first",
            Self::PayloadTooLarge => "send a smaller body",
            Self::Unprocessable => "fix the request semantics for this surface",
            Self::BadGateway => "retry; if it persists, check daemon logs for the engine child",
            Self::Unavailable => "retry after capacity frees; `blazar ps` shows resident state",
            Self::GatewayTimeout => "retry, or raise the client timeout for long generations",
            Self::InsufficientStorage => "free memory or storage, or use a smaller footprint",
        }
    }
}

/// Every shipped code, in catalog order. `GET /api/errors` renders this.
pub const ALL: &[BlazarCode] = &[
    BlazarCode::ModelNotFound,
    BlazarCode::ModelUnsupportedEngine,
    BlazarCode::ModelLoadTimeout,
    BlazarCode::EngineCircuitOpen,
    BlazarCode::AllSlotsBusy,
    BlazarCode::ModelTooLarge,
    BlazarCode::InsufficientMemory,
    BlazarCode::EngineCrashed,
    BlazarCode::RpcEndpointUnreachable,
    BlazarCode::Internal,
    BlazarCode::PromptTooLong,
    BlazarCode::VisionProjectorMissing,
    BlazarCode::CapabilityVerifiedFailed,
    BlazarCode::DiffusionTextRefusal,
    BlazarCode::Unauthorized,
    BlazarCode::KeyScopeForbidden,
    BlazarCode::HostForbidden,
    BlazarCode::Forbidden,
    BlazarCode::RateLimited,
    BlazarCode::FleetTokenRequired,
    BlazarCode::BadRequest,
    BlazarCode::NotFound,
    BlazarCode::PayloadTooLarge,
    BlazarCode::Unprocessable,
    BlazarCode::BadGateway,
    BlazarCode::Unavailable,
    BlazarCode::GatewayTimeout,
    BlazarCode::InsufficientStorage,
    BlazarCode::MethodNotAllowed,
    BlazarCode::UnsupportedMediaType,
    BlazarCode::Conflict,
];

/// Total fallback for `openai_error(status, ...)` sites that have no
/// typed mapping: a generic-but-stable code per status. Unknown or
/// invalid statuses collapse to `INTERNAL`, matching the
/// `unwrap_or(INTERNAL_SERVER_ERROR)` the envelope already applies.
#[must_use]
pub fn generic_for_status(status: u16) -> BlazarCode {
    match status {
        400 => BlazarCode::BadRequest,
        403 => BlazarCode::Forbidden,
        404 => BlazarCode::NotFound,
        409 => BlazarCode::Conflict,
        413 => BlazarCode::PayloadTooLarge,
        422 => BlazarCode::Unprocessable,
        429 => BlazarCode::RateLimited,
        502 => BlazarCode::BadGateway,
        503 => BlazarCode::Unavailable,
        504 => BlazarCode::GatewayTimeout,
        507 => BlazarCode::InsufficientStorage,
        _ => BlazarCode::Internal,
    }
}

/// OpenAI-shaped error body carrying a PRECISE stable code. Typed
/// choke points render through this; everything else goes through
/// `proxy::openai_error` (generic status-derived code).
#[must_use]
pub fn error_response(code: BlazarCode, message: &str) -> axum::response::Response {
    let body = json!({
        "error": {
            "message": message,
            "type": "blazar_error",
            "code": code.http(),
            "blazar_code": code.as_str(),
        }
    });
    (
        axum::http::StatusCode::from_u16(code.http())
            .unwrap_or(axum::http::StatusCode::INTERNAL_SERVER_ERROR),
        axum::Json(body),
    )
        .into_response()
}

/// Exact-status variant for raw `u16` call sites (lib.rs router paths):
/// the body carries the generic typed code for the status, but the HTTP
/// status is EXACTLY the one the caller asked for - unmapped statuses
/// keep their status instead of collapsing to INTERNAL.
#[must_use]
pub fn generic_error_response(status: u16, message: &str) -> axum::response::Response {
    let code = generic_for_status(status);
    let body = json!({
        "error": {
            "message": message,
            "type": "blazar_error",
            "code": status,
            "blazar_code": code.as_str(),
        }
    });
    (
        axum::http::StatusCode::from_u16(status)
            .unwrap_or(axum::http::StatusCode::INTERNAL_SERVER_ERROR),
        axum::Json(body),
    )
        .into_response()
}

/// `GET /api/errors` - the machine-readable catalog. Static by
/// construction (no state), same auth posture as every other /api route.
pub async fn errors_catalog() -> axum::response::Response {
    let codes: Vec<serde_json::Value> = ALL
        .iter()
        .map(|c| {
            json!({
                "code": c.as_str(),
                "http": c.http(),
                "description": c.description(),
                "remediation": c.remediation(),
            })
        })
        .collect();
    axum::Json(json!({
        "catalog_version": CATALOG_VERSION,
        "codes": codes,
    }))
    .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn body_json(resp: axum::response::Response) -> serde_json::Value {
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("body bytes");
        serde_json::from_slice(&bytes).expect("valid JSON body")
    }

    #[test]
    #[allow(non_snake_case)]
    fn unit__catalog__every_code_unique_with_metadata() {
        let mut seen = std::collections::HashSet::new();
        for code in ALL {
            assert!(
                seen.insert(code.as_str()),
                "duplicate code string {}",
                code.as_str()
            );
            assert_eq!(
                code.as_str(),
                code.as_str().to_uppercase(),
                "SCREAMING_SNAKE"
            );
            assert!(code.description().len() > 10, "description present");
            assert!(code.remediation().len() > 5, "remediation present");
            assert!(
                (400..=599).contains(&code.http()),
                "valid HTTP status for {}",
                code.as_str()
            );
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn unit__generic_for_status__total_with_internal_fallback() {
        assert_eq!(generic_for_status(400), BlazarCode::BadRequest);
        assert_eq!(generic_for_status(507), BlazarCode::InsufficientStorage);
        // Off-map statuses (418, 599, 302...) collapse, never panic.
        for odd in [0, 302, 418, 499, 598, 999] {
            assert_eq!(
                generic_for_status(odd),
                BlazarCode::Internal,
                "status {odd}"
            );
        }
        // Every fallback's own status re-derives to itself (pairing coherence).
        for code in [
            BlazarCode::BadRequest,
            BlazarCode::NotFound,
            BlazarCode::PayloadTooLarge,
            BlazarCode::Unprocessable,
            BlazarCode::RateLimited,
            BlazarCode::Internal,
            BlazarCode::BadGateway,
            BlazarCode::Unavailable,
            BlazarCode::GatewayTimeout,
            BlazarCode::InsufficientStorage,
        ] {
            assert_eq!(
                generic_for_status(code.http()),
                code,
                "fallback for {} must be idempotent",
                code.http()
            );
        }
    }

    #[tokio::test]
    #[allow(non_snake_case)]
    async fn unit__error_response__envelope_carries_stable_code() {
        let resp = error_response(BlazarCode::ModelNotFound, "no such model: foo");
        assert_eq!(resp.status(), axum::http::StatusCode::NOT_FOUND);
        let v = body_json(resp).await;
        assert_eq!(
            v.pointer("/error/type"),
            Some(&serde_json::json!("blazar_error"))
        );
        assert_eq!(v.pointer("/error/code"), Some(&serde_json::json!(404)));
        assert_eq!(
            v.pointer("/error/blazar_code"),
            Some(&serde_json::json!("MODEL_NOT_FOUND"))
        );
        assert_eq!(
            v.pointer("/error/message"),
            Some(&serde_json::json!("no such model: foo"))
        );
    }

    #[tokio::test]
    #[allow(non_snake_case)]
    async fn unit__errors_catalog__versioned_and_complete() {
        let v = body_json(errors_catalog().await).await;
        assert_eq!(v["catalog_version"], serde_json::json!(CATALOG_VERSION));
        let codes = v["codes"].as_array().expect("codes array");
        assert_eq!(
            codes.len(),
            ALL.len(),
            "catalog lists every code exactly once"
        );
        for entry in codes {
            assert!(entry["code"].is_string());
            assert!(entry["http"].is_u64());
            assert!(entry["description"].is_string());
            assert!(entry["remediation"].is_string());
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn unit__generic_for_status__conflict_maps_and_status_is_preserved() {
        // Raw-u16 router paths (lib.rs) must keep their exact HTTP status:
        // 409 stays 409 with the typed CONFLICT code, unmapped statuses
        // keep their status instead of collapsing to INTERNAL.
        assert_eq!(generic_for_status(409), BlazarCode::Conflict);
        let resp = generic_error_response(409, "key \"x\" already exists");
        assert_eq!(resp.status(), axum::http::StatusCode::CONFLICT);
        let odd = generic_error_response(418, "teapot");
        assert_eq!(odd.status(), axum::http::StatusCode::from_u16(418).unwrap());
        assert_eq!(generic_for_status(418), BlazarCode::Internal);
    }

    #[test]
    #[allow(non_snake_case)]
    fn unit__docs_error_codes__table_matches_enum() {
        // Drift pin: every shipped code has a docs row and vice versa.
        let doc = include_str!("../../../docs/error-codes.md");
        let rows: Vec<&str> = doc.lines().filter(|l| l.starts_with("| `")).collect();
        assert_eq!(
            rows.len(),
            ALL.len(),
            "docs/error-codes.md row count {} != {} enum variants - regenerate the table",
            rows.len(),
            ALL.len()
        );
        for code in ALL {
            assert!(
                doc.contains(&format!("| `{}` |", code.as_str())),
                "docs/error-codes.md missing row for {}",
                code.as_str()
            );
        }
    }
    mod status_enrich_tests {
        use super::*;
        use axum::body::Body;
        use axum::http::{Response, StatusCode, header};

        #[test]
        #[allow(non_snake_case)]
        fn unit__generic_for_status__forbidden_is_not_internal() {
            // A bare 403 used to collapse to INTERNAL_ERROR; it now keeps its own code.
            assert_eq!(generic_for_status(403), BlazarCode::Forbidden);
            assert_eq!(BlazarCode::Forbidden.http(), 403);
            assert_eq!(BlazarCode::Forbidden.as_str(), "FORBIDDEN");
        }

        async fn enrich(status: StatusCode, allow: Option<&str>) -> axum::response::Response {
            let mut builder = Response::builder().status(status);
            if let Some(a) = allow {
                builder = builder.header(header::ALLOW, a);
            }
            let resp = builder.body(Body::empty()).unwrap();
            crate::status_enrich_mw(resp).await
        }

        async fn body_json(resp: axum::response::Response) -> serde_json::Value {
            let bytes = axum::body::to_bytes(resp.into_body(), 64 * 1024)
                .await
                .unwrap();
            serde_json::from_slice(&bytes).unwrap()
        }

        #[tokio::test]
        #[allow(non_snake_case)]
        async fn unit__status_enrich__405_gets_envelope_and_allow_header() {
            // Axum's default 405 carries an empty body (no blazar envelope) and an
            // auto-generated Allow header; enrichment must wrap the body AND keep Allow.
            let resp = enrich(StatusCode::METHOD_NOT_ALLOWED, Some("POST")).await;
            assert_eq!(resp.status(), 405);
            let allow = resp
                .headers()
                .get(header::ALLOW)
                .expect("Allow header survives enrichment")
                .to_str()
                .unwrap()
                .to_ascii_uppercase();
            assert!(
                allow.contains("POST"),
                "Allow lists the accepted verb: {allow}"
            );
            let v = body_json(resp).await;
            assert_eq!(
                v["error"]["blazar_code"], "METHOD_NOT_ALLOWED",
                "405 body is the blazar envelope: {v}"
            );
            assert!(v["error"]["message"].as_str().unwrap().contains("Allow"));
        }

        #[tokio::test]
        #[allow(non_snake_case)]
        async fn unit__status_enrich__415_gets_envelope() {
            // Wrong content-type: axum rejects before any handler runs - with an
            // empty body or a plain-text line like "Expected request with
            // `Content-Type: application/json`"; both get the blazar envelope.
            let resp = enrich(StatusCode::UNSUPPORTED_MEDIA_TYPE, None).await;
            assert_eq!(resp.status(), 415);
            let v = body_json(resp).await;
            assert_eq!(
                v["error"]["blazar_code"], "UNSUPPORTED_MEDIA_TYPE",
                "415 body is the blazar envelope: {v}"
            );
            assert!(v["error"]["message"].as_str().unwrap().contains("json"));

            // The extractor's real-world shape: text/plain content-type + body.
            let resp = Response::builder()
                .status(StatusCode::UNSUPPORTED_MEDIA_TYPE)
                .header("content-type", "text/plain")
                .body(Body::from(
                    "Expected request with `Content-Type: application/json`\n",
                ))
                .unwrap();
            let out = crate::status_enrich_mw(resp).await;
            let v = body_json(out).await;
            assert_eq!(
                v["error"]["blazar_code"], "UNSUPPORTED_MEDIA_TYPE",
                "axum's plain-text 415 is enveloped: {v}"
            );

            // A handler that already answered JSON keeps its payload.
            let resp = Response::builder()
                .status(StatusCode::UNSUPPORTED_MEDIA_TYPE)
                .header("content-type", "application/json")
                .body(Body::from(r#"{"already":"json"}"#))
                .unwrap();
            let out = crate::status_enrich_mw(resp).await;
            let v = body_json(out).await;
            assert_eq!(v["already"], "json", "JSON 415 body passes through");
        }

        #[tokio::test]
        #[allow(non_snake_case)]
        async fn unit__status_enrich__bodies_are_left_alone() {
            // Handlers that already produced a body (our own envelopes) must not be
            // double-wrapped; only axum's empty default rejections are enriched.
            let resp = Response::builder()
                .status(StatusCode::METHOD_NOT_ALLOWED)
                .body(Body::from(r#"{"already":"enveloped"}"#))
                .unwrap();
            let out = crate::status_enrich_mw(resp).await;
            let v = body_json(out).await;
            assert_eq!(v["already"], "enveloped", "non-empty body passes through");
        }
    }
}
