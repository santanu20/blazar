//! `/ui` - the embedded web console, served straight from the binary.
//!
//! The page is `console.html`, compiled in with `include_str!`: no
//! external assets, no CDN, no build pipeline - the console works
//! offline exactly where the daemon works. Observability composes the
//! existing JSON endpoints (`/api/version`, `/api/tags`, `/api/ps`,
//! `/api/fabric`, `/api/quantiles`, `/api/benchmarks`, `/api/usage`,
//! `/api/capacity`, `/v1/jobs`, `/api/sessions`, `/api/pools`,
//! `/api/failover`, `/api/why`, plus the `/api/events` SSE stream);
//! mutations go through the validated write planes (`/api/config/set`,
//! `/api/config/unset`, `/api/config/reload`, `/api/warm`, `/api/evict`,
//! job cancel, key add/remove/rotate, failover and pool member ops) -
//! the same contract the CLI uses, so the GUI can never drift from it.

use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};

/// `GET /ui` - the console document, `text/html` + no-cache (live
/// daemon state should never be served stale from a browser cache).
pub async fn ui() -> Response {
    let mut res = axum::response::Html(include_str!("console.html")).into_response();
    let headers = res.headers_mut();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/html; charset=utf-8"),
    );
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    *res.status_mut() = StatusCode::OK;
    res
}

#[cfg(test)]
mod tests {
    /// The console must be self-contained: any external URL would break
    /// the offline guarantee (and leak browsing off-machine).
    #[test]
    #[allow(non_snake_case)] // pin names read as scenario sentences (house pattern)
    fn unit__console_html__self_contained_offline_document() {
        let html = include_str!("console.html");
        assert!(!html.contains("http://"), "no plain-external refs");
        assert!(!html.contains("https://"), "no CDN refs");
        assert!(!html.contains("src=\""), "no script/img/network sources");
        for tab in [
            "dashboard",
            "models",
            "fabric",
            "routing",
            "config",
            "benchmarks",
            "jobs",
            "sessions",
            "keys",
        ] {
            assert!(
                html.contains(&format!("data-tab=\"{tab}\"")),
                "tab {tab} present"
            );
            assert!(
                html.contains(&format!("id=\"{tab}\"")),
                "section {tab} present"
            );
        }
        // Read planes the page composes - if one moves, this pin flags it.
        for ep in [
            "/api/version",
            "/api/tags",
            "/api/ps",
            "/api/fabric",
            "/api/quantiles",
            "/api/usage?days=7",
            "/api/capacity",
            "/api/admission-accuracy",
            "/api/why?limit=10",
            "/api/mcp",
            "/api/pools",
            "/api/failover",
            "/api/fleet/workers",
            "/api/config",
            "/api/benchmarks",
            "/v1/jobs",
            "/api/sessions",
            "/api/keys",
            // live strip
            "/api/events",
        ] {
            assert!(html.contains(ep), "fetches {ep}");
        }
        // Write planes: the console's mutation surface. One pin each so
        // a renamed route fails here, not silently in the browser.
        for ep in [
            "/api/config/set",
            "/api/config/unset",
            "/api/config/reload",
            "/api/warm",
            "/api/evict",
            "/api/explain/",
            "/api/diagnose/",
            "/api/route/",
            "/cancel",
            "/api/keys/rotate",
            "/api/failover/",
            "/api/pools/",
        ] {
            assert!(html.contains(ep), "mutates through {ep}");
        }
        // Every mutation and every dynamic string must route through the
        // escaping dispatcher - no inline handler strings anywhere.
        assert!(!html.contains("onclick="), "no inline handlers");
        assert!(html.contains("data-action="), "action delegation present");
        assert!(html.contains("encodeURIComponent"), "path args encoded");
    }
}
