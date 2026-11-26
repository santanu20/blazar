//! W3C Trace Context intake (<https://www.w3.org/TR/trace-context/>).
//!
//! With an exporter configured the gateway takes part in the caller's
//! trace: it continues a valid context (or restarts an invalid one),
//! becomes the parent of whatever the engine child records, and reports
//! its own span back to the caller. Without an exporter it is a
//! pass-through proxy: `traceparent` and `tracestate` reach the child as
//! sent.

use axum::http::{HeaderMap, HeaderValue};

/// `version "-" trace-id "-" parent-id "-" trace-flags` at version 00.
const TRACEPARENT_LEN: usize = 55;

/// A `tracestate` list holds at most this many members.
const TRACESTATE_MAX_MEMBERS: usize = 32;

/// The caller's operation, from a valid `traceparent` header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TraceParent {
    trace_id: String,
    span_id: String,
    /// Version 00 carries one opaque flag byte; for later versions only
    /// the sampled bit is understood and the rest leaves as zero.
    trace_flags: u8,
}

/// The trace a request is served in, as the gateway decided it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Propagation {
    pub trace_id: String,
    /// The gateway's own operation - the parent of every child span.
    pub span_id: String,
    /// The caller's operation while the trace continues; `None` after a
    /// restart.
    pub parent_span_id: Option<String>,
    /// Vendor state that travels on; `None` when none arrived or none
    /// of it parsed.
    pub trace_state: Option<String>,
    pub trace_flags: u8,
}

/// Lowercase hex only: uppercase digits make the header invalid.
fn lower_hex(digits: &[u8]) -> Option<String> {
    digits
        .iter()
        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(b))
        .then(|| String::from_utf8_lossy(digits).into_owned())
}

fn hex_pair(digits: &[u8]) -> Option<u8> {
    u8::from_str_radix(&lower_hex(digits)?, 16).ok()
}

fn all_zero(digits: &str) -> bool {
    digits.bytes().all(|b| b == b'0')
}

/// Parse a `traceparent` header value. Anything that does not match the
/// version 00 layout (or a longer future version of it) is `None`: the
/// caller's context cannot be continued.
pub(crate) fn parse_traceparent(value: &str) -> Option<TraceParent> {
    // Padding (space or tab) around the field value is not part of it.
    let value = value.trim_matches([' ', '\t']);
    let bytes = value.as_bytes();
    // Too short for version, both ids and flags.
    if bytes.len() < TRACEPARENT_LEN {
        return None;
    }
    let version = hex_pair(&bytes[0..2])?;
    if version == 0xff || bytes[2] != b'-' {
        return None;
    }
    let trace_id = lower_hex(&bytes[3..35])?;
    if bytes[35] != b'-' {
        return None;
    }
    let span_id = lower_hex(&bytes[36..52])?;
    if bytes[52] != b'-' {
        return None;
    }
    let flags = hex_pair(&bytes[53..55])?;
    // A later format may append a tail, but only behind a dash.
    if bytes.get(55).is_some_and(|&b| b != b'-') {
        return None;
    }
    let trace_flags = if version == 0 {
        // Version 00 defines nothing after the flags.
        if bytes.len() != TRACEPARENT_LEN {
            return None;
        }
        flags
    } else {
        // Flags this level cannot read are dropped on the way out.
        flags & 0x01
    };
    if all_zero(&trace_id) || all_zero(&span_id) {
        return None;
    }
    Some(TraceParent {
        trace_id,
        span_id,
        trace_flags,
    })
}

fn is_ows(b: u8) -> bool {
    b == b' ' || b == b'\t'
}

fn key_char(b: u8) -> bool {
    b.is_ascii_lowercase() || b.is_ascii_digit() || matches!(b, b'_' | b'-' | b'*' | b'/')
}

/// `simple-key` or `tenant-id "@" system-id`, each part within its own
/// length limit.
fn valid_key(key: &str) -> bool {
    let part = |bytes: &[u8], max: usize, digit_first: bool| {
        (1..=max).contains(&bytes.len())
            && (bytes[0].is_ascii_lowercase() || (digit_first && bytes[0].is_ascii_digit()))
            && bytes.iter().all(|&b| key_char(b))
    };
    match key.split_once('@') {
        None => part(key.as_bytes(), 256, false),
        Some((tenant, system)) if !system.contains('@') => {
            part(tenant.as_bytes(), 241, true) && part(system.as_bytes(), 14, false)
        }
        Some(_) => false,
    }
}

fn nblk(b: u8) -> bool {
    matches!(b, 0x21..=0x2B | 0x2D..=0x3C | 0x3E..=0x7E)
}

/// `0*255(chr) nblk-chr`: up to 256 bytes, plain ASCII, not ending blank.
fn valid_value(value: &str) -> bool {
    let bytes = value.as_bytes();
    (1..=256).contains(&bytes.len())
        && bytes[..bytes.len() - 1]
            .iter()
            .all(|&b| b == b' ' || nblk(b))
        && nblk(bytes[bytes.len() - 1])
}

fn valid_member(member: &str) -> bool {
    let member = member.trim_matches([' ', '\t']);
    // Empty and blank members are legal.
    if member.is_empty() {
        return true;
    }
    member
        .split_once('=')
        .is_some_and(|(key, value)| valid_key(key) && valid_value(value))
}

/// Validate the `tracestate` header lines of one request and return them
/// combined as they travel on; `None` when nothing arrived or any member
/// is malformed (the whole header is then not propagated).
pub(crate) fn parse_tracestate(values: &[&str]) -> Option<String> {
    // Several header lines combine into one list (RFC 7230).
    let combined = values.join(", ");
    if combined.bytes().all(is_ows) {
        return None;
    }
    let members: Vec<&str> = combined.split(',').collect();
    if members.len() > TRACESTATE_MAX_MEMBERS || !members.iter().all(|m| valid_member(m)) {
        return None;
    }
    // A repeated key is left as sent: the list is passed on, not merged.
    // A list made only of empty members (`,` or ` , `) carries no vendor
    // state at all: nothing to propagate, so it reads as absent.
    if members
        .iter()
        .all(|member| member.trim_matches([' ', '\t']).is_empty())
    {
        return None;
    }
    Some(combined.trim_matches([' ', '\t']).to_string())
}

impl Propagation {
    /// Decide the trace for a request: continue the caller's context
    /// when it parses, otherwise start a new trace under a fresh span.
    pub(crate) fn for_request(headers: &HeaderMap) -> Self {
        // Two traceparent headers name two parents: no single context.
        let mut parents = headers.get_all("traceparent").iter();
        let parent = parents
            .next()
            .filter(|_| parents.next().is_none())
            .and_then(|value| value.to_str().ok())
            .and_then(parse_traceparent);
        let states: Vec<&str> = headers
            .get_all("tracestate")
            .iter()
            .filter_map(|value| value.to_str().ok())
            .collect();
        let (fresh_trace_id, span_id) = crate::otlp::new_ids();
        match parent {
            Some(parent) => Self {
                trace_id: parent.trace_id,
                span_id,
                parent_span_id: Some(parent.span_id),
                trace_state: parse_tracestate(&states),
                trace_flags: parent.trace_flags,
            },
            None => Self {
                trace_id: fresh_trace_id,
                span_id,
                parent_span_id: None,
                trace_state: None,
                trace_flags: 0,
            },
        }
    }

    /// `traceparent` naming the gateway's operation.
    pub(crate) fn header(&self) -> String {
        format!(
            "00-{}-{}-{:02x}",
            self.trace_id, self.span_id, self.trace_flags
        )
    }

    /// Whether the caller's context continues through this request.
    pub(crate) fn inherited(&self) -> bool {
        self.parent_span_id.is_some()
    }

    /// Make `headers` carry the gateway's operation toward the child.
    pub(crate) fn rewrite(&self, headers: &mut HeaderMap) {
        match HeaderValue::from_str(&self.header()) {
            Ok(value) => {
                headers.insert("traceparent", value);
            }
            // Unreachable for hex ids; a wrong parent is worse than none.
            Err(_) => {
                headers.remove("traceparent");
            }
        }
        if self.trace_state.is_none() {
            // A restarted trace, or state that did not parse: the
            // exported span carries none, so the child gets none either.
            headers.remove("tracestate");
        }
    }
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    const TRACE: &str = "4bf92f3577b34da6a3ce929d0e0e4736";
    const SPAN: &str = "00f067aa0ba902b7";

    fn parent(value: &str) -> Option<(String, String, u8)> {
        parse_traceparent(value).map(|p| (p.trace_id, p.span_id, p.trace_flags))
    }

    #[test]
    fn unit__parse_traceparent__accepts_version_00() {
        assert_eq!(
            parent(&format!("00-{TRACE}-{SPAN}-01")),
            Some((TRACE.into(), SPAN.into(), 1))
        );
        // Version 00 flags are one opaque byte, passed on whole.
        assert_eq!(
            parent(&format!("00-{TRACE}-{SPAN}-ff")).map(|p| p.2),
            Some(0xff)
        );
        // Space or tab padding around the field value is not part of it.
        for padded in [
            format!("00-{TRACE}-{SPAN}-01 "),
            format!("00-{TRACE}-{SPAN}-01\t"),
            format!("00-{TRACE}-{SPAN}-01 \t"),
            format!(" 00-{TRACE}-{SPAN}-01"),
        ] {
            assert!(parent(&padded).is_some(), "{padded:?}");
        }
    }

    #[test]
    fn unit__parse_traceparent__rejects_what_the_spec_forbids() {
        let bad = [
            format!("00-{}-{SPAN}-01", TRACE.to_uppercase()),
            format!("00-00000000000000000000000000000000-{SPAN}-01"),
            format!("00-{TRACE}-0000000000000000-01"),
            format!("ff-{TRACE}-{SPAN}-01"),
            format!("00-{TRACE}-{SPAN}-1"),
            format!("00-{TRACE}-{SPAN}-01-tail"),
            format!("00-{TRACE}{SPAN}01"),
            format!("00-{TRACE}-{SPAN}-0x"),
        ];
        for bad in &bad {
            assert_eq!(parse_traceparent(bad), None, "{bad}");
        }
    }

    #[test]
    fn unit__parse_traceparent__reads_a_later_version_and_drops_unknown_flags() {
        assert_eq!(
            parent(&format!("01-{TRACE}-{SPAN}-ff-extra")),
            Some((TRACE.into(), SPAN.into(), 1))
        );
        assert_eq!(
            parent(&format!("01-{TRACE}-{SPAN}-01")),
            Some((TRACE.into(), SPAN.into(), 1))
        );
        assert_eq!(
            parent(&format!("01-{TRACE}-{SPAN}-03-tail")).map(|p| p.2),
            Some(1)
        );
        assert_eq!(
            parent(&format!("01-{TRACE}-{SPAN}-02-tail")).map(|p| p.2),
            Some(0)
        );
        // A later version still needs the full leading 55 characters.
        assert_eq!(parent(&format!("01-{TRACE}-{SPAN}-0-tail")), None);
    }

    #[test]
    fn unit__parse_tracestate__accepts_the_list_grammar() {
        assert_eq!(
            parse_tracestate(&["vendor=value"]).as_deref(),
            Some("vendor=value")
        );
        assert_eq!(
            parse_tracestate(&["rojo=state", "vendor=value"]).as_deref(),
            Some("rojo=state, vendor=value"),
            "header lines combine into one list"
        );
        assert_eq!(
            parse_tracestate(&["a=1,, ,b=2"]).as_deref(),
            Some("a=1,, ,b=2")
        );
        assert_eq!(
            parse_tracestate(&["tenant@sys=v"]).as_deref(),
            Some("tenant@sys=v")
        );
        // A system-id is at most 14 characters; a tenant-id may be long.
        assert_eq!(
            parse_tracestate(&["a@abcdefghijklmn=v"]).as_deref(),
            Some("a@abcdefghijklmn=v")
        );
        assert_eq!(
            parse_tracestate(&["0123456789abcdefghijklmnopqrstuvwxyz@sys=v"]).as_deref(),
            Some("0123456789abcdefghijklmnopqrstuvwxyz@sys=v")
        );
        let longest_key = format!("{}=v", "k".repeat(256));
        assert!(parse_tracestate(&[&longest_key]).is_some());
        // Blanks may precede the final character; 256 bytes is the ceiling.
        let padded = format!(" {}", "v".repeat(255));
        assert!(
            parse_tracestate(&[&format!("k={padded}")]).is_some(),
            "a leading blank is part of the value"
        );
        // Blanks around a member's comma travel with the list (W3C vectors).
        assert_eq!(
            parse_tracestate(&["foo=1 \t , \t bar=2, \t baz=3"]).as_deref(),
            Some("foo=1 \t , \t bar=2, \t baz=3")
        );
        assert_eq!(parse_tracestate(&["foo=1 "]).as_deref(), Some("foo=1"));
        assert_eq!(parse_tracestate(&["foo=1\t"]).as_deref(), Some("foo=1"));
        // Several header fields are combined into one list.
        assert_eq!(
            parse_tracestate(&["foo=1", "bar=2"]).as_deref(),
            Some("foo=1, bar=2")
        );
        assert_eq!(parse_tracestate(&["foo=1", ""]).as_deref(), Some("foo=1,"));
    }

    #[test]
    fn unit__parse_tracestate__rejects_malformed_lists_and_blanks() {
        assert_eq!(parse_tracestate(&[""]), None);
        assert_eq!(parse_tracestate(&[" , "]), None, "only empty members");
        assert_eq!(parse_tracestate(&[]), None);
        // Empty members between real ones stay in the list as received.
        assert_eq!(
            parse_tracestate(&["a=1,, ,b=2"]).as_deref(),
            Some("a=1,, ,b=2")
        );
        // A repeated key is the caller's own list: it travels on as sent.
        assert_eq!(
            parse_tracestate(&["foo=1,foo=1"]).as_deref(),
            Some("foo=1,foo=1")
        );
        assert_eq!(
            parse_tracestate(&["foo=1", "foo=2"]).as_deref(),
            Some("foo=1, foo=2")
        );
        let many = std::iter::repeat_n("a=1", 33)
            .map(String::from)
            .collect::<Vec<_>>();
        let many_refs: Vec<&str> = many.iter().map(String::as_str).collect();
        assert_eq!(parse_tracestate(&many_refs), None, "33 members");
        for bad in [
            "Vendor=value",
            "foo =1",
            "foo.bar=1",
            "foo@=1,bar=2",
            "@foo=1,bar=2",
            "foo@@bar=1,bar=2",
            "t@vvvvvvvvvvvvvvv=1",
            "foo=bar=baz",
            "foo=,bar=3",
            "key",
            "k=a=b",
            "k=a,b",
            "k=",
            "tenant@a@b=v",
            "tenant@abcdefghijklmnopqrstuvwxyz=v",
            "UPPER=value",
            "k=\u{e9}",
        ] {
            assert_eq!(parse_tracestate(&[bad]), None, "{bad:?}");
        }
        let long_value = format!("k={}", "v".repeat(257));
        assert_eq!(parse_tracestate(&[&long_value]), None);
        let long_key = format!("{}=v", "k".repeat(257));
        assert_eq!(parse_tracestate(&[&long_key]), None);
    }

    fn request_headers(traceparent: Option<&str>, tracestate: Option<&str>) -> HeaderMap {
        let mut headers = HeaderMap::new();
        if let Some(value) = traceparent {
            headers.insert("traceparent", HeaderValue::from_str(value).unwrap());
        }
        if let Some(value) = tracestate {
            headers.insert("tracestate", HeaderValue::from_str(value).unwrap());
        }
        headers
    }

    #[test]
    fn unit__propagation__continues_a_valid_context() {
        let headers = request_headers(Some(&format!("00-{TRACE}-{SPAN}-01")), Some("vendor=value"));
        let propagation = Propagation::for_request(&headers);
        assert_eq!(propagation.trace_id, TRACE);
        assert_eq!(propagation.parent_span_id.as_deref(), Some(SPAN));
        assert_eq!(propagation.trace_state.as_deref(), Some("vendor=value"));
        assert_eq!(propagation.trace_flags, 1);
        assert_eq!(propagation.span_id.len(), 16);
        let mut forwarded = headers.clone();
        propagation.rewrite(&mut forwarded);
        assert_eq!(
            forwarded.get("traceparent").and_then(|v| v.to_str().ok()),
            Some(propagation.header().as_str())
        );
        assert_eq!(
            forwarded.get("tracestate").and_then(|v| v.to_str().ok()),
            Some("vendor=value")
        );
        assert_eq!(
            propagation.header(),
            format!("00-{TRACE}-{}-01", propagation.span_id)
        );
    }

    #[test]
    fn unit__propagation__restarts_on_invalid_context_and_drops_state() {
        for bad in [
            format!("00-00000000000000000000000000000000-{SPAN}-01"),
            format!("00-{}-{SPAN}-01", TRACE.to_uppercase()),
        ] {
            let headers = request_headers(Some(&bad), Some("vendor=value"));
            let propagation = Propagation::for_request(&headers);
            assert!(!propagation.inherited(), "{bad}");
            assert_eq!(propagation.parent_span_id, None);
            assert_eq!(propagation.trace_state, None);
            assert_eq!(propagation.trace_flags, 0);
            assert_eq!(propagation.trace_id.len(), 32);
            assert_ne!(propagation.trace_id, TRACE);
            let mut forwarded = headers.clone();
            propagation.rewrite(&mut forwarded);
            assert_eq!(
                forwarded.get("traceparent").and_then(|v| v.to_str().ok()),
                Some(propagation.header().as_str())
            );
            assert!(forwarded.get("tracestate").is_none(), "{bad}");
        }
    }

    #[test]
    fn unit__propagation__two_traceparent_headers_are_no_context() {
        let mut headers = request_headers(Some(&format!("00-{TRACE}-{SPAN}-01")), None);
        headers.append(
            "traceparent",
            HeaderValue::from_str(&format!("00-{TRACE}-0102030405060708-01")).unwrap(),
        );
        assert!(!Propagation::for_request(&headers).inherited());
    }

    #[test]
    fn unit__propagation__plain_request_starts_a_root_span() {
        let propagation = Propagation::for_request(&HeaderMap::new());
        assert!(!propagation.inherited());
        assert_eq!(propagation.trace_id.len(), 32);
        assert_eq!(propagation.trace_flags, 0);
        assert!(propagation.header().ends_with("-00"));
    }

    #[test]
    fn unit__propagation__malformed_state_travels_no_further() {
        let headers = request_headers(Some(&format!("00-{TRACE}-{SPAN}-01")), Some("Bad=v"));
        let propagation = Propagation::for_request(&headers);
        assert!(propagation.inherited(), "the trace itself is fine");
        assert_eq!(propagation.trace_state, None);
        let mut forwarded = headers.clone();
        propagation.rewrite(&mut forwarded);
        assert!(forwarded.get("tracestate").is_none());
    }
}
