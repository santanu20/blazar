//! `/ui` — the embedded web console, served straight from the binary.
//!
//! The page is `console.html`, compiled in with `include_str!`: no
//! external assets, no CDN, no build pipeline — the console works
//! offline exactly where the daemon works. It is read-only and
//! composes the existing JSON endpoints (`/api/version`, `/api/tags`,
//! `/api/ps`, `/api/fabric`, `/api/quantiles`, `/api/benchmarks`,
//! `/v1/jobs`, `/api/sessions`); it holds no state of its own.

use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};

/// `GET /ui` — the console document, `text/html` + no-cache (live
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
            "engines",
            "compute",
            "benchmarks",
            "jobs",
            "sessions",
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
        // Endpoints the page composes — if one moves, this pin flags it.
        for ep in [
            "/api/version",
            "/api/tags",
            "/api/ps",
            "/api/fabric",
            "/api/quantiles",
            "/api/benchmarks",
            "/v1/jobs",
            "/api/sessions",
        ] {
            assert!(html.contains(ep), "fetches {ep}");
        }
    }
}
