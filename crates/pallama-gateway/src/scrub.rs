//! Opt-in PII scrubbing for `why`/`watch` output (`pii_scrub = true`).
//!
//! Redacts the obvious classes that can leak through record details:
//! emails, pallama bearer secrets (`plm_...`) and IPv4 addresses. Hand-
//! scanned — no regex dependency, no allocation beyond the output. Numbers,
//! codes and latencies pass through untouched so `why` stays diagnostic.

use serde_json::Value;

/// Recursively scrub every string field of a JSON value.
#[must_use]
pub fn scrub_value(v: &Value) -> Value {
    match v {
        Value::String(s) => Value::String(scrub_str(s)),
        Value::Array(a) => Value::Array(a.iter().map(scrub_value).collect()),
        Value::Object(o) => Value::Object(
            o.iter()
                .map(|(k, val)| (k.clone(), scrub_value(val)))
                .collect(),
        ),
        other => other.clone(),
    }
}

/// Redact emails, `plm_` secrets and IPv4 addresses; keep everything else
/// byte-identical.
#[must_use]
pub fn scrub_str(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < bytes.len() {
        // pallama key secrets: plm_ + word chars
        if s[i..].starts_with("plm_") {
            let end = s[i..]
                .find(|c: char| !c.is_ascii_alphanumeric() && c != '_' && c != '-')
                .map_or(s.len(), |rel| i + rel);
            out.push_str("plm_[redacted]");
            i = end;
            continue;
        }
        // email-ish: user@domain.tld (scan back from '@' is awkward
        // forward-only; instead detect a run containing exactly one '@'
        // with dots after it)
        if is_word(bytes[i]) {
            let (redacted_email, next) = scan_email(s, i);
            if let Some(next) = next {
                out.push_str(&redacted_email);
                i = next;
                continue;
            }
            // IPv4: d.d.d.d
            if let (ip, Some(next)) = scan_ipv4(s, i) {
                out.push_str(&ip);
                i = next;
                continue;
            }
            let (word, next) = scan_word(s, i);
            out.push_str(&word);
            i = next;
            continue;
        }
        out.push(bytes[i] as char);
        i += 1;
    }
    out
}

fn is_word(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b == b'-'
}

fn scan_word(s: &str, start: usize) -> (String, usize) {
    let end = s[start..]
        .find(|c: char| !is_word(c as u8))
        .map_or(s.len(), |rel| start + rel);
    (s[start..end].to_string(), end)
}

/// Consume `[word@domain.tld]` starting at `start`; None = not an email.
fn scan_email(s: &str, start: usize) -> (String, Option<usize>) {
    let (local, at) = scan_word(s, start);
    if !s[at..].starts_with('@') {
        return (local, None);
    }
    let domain_start = at + 1;
    let bytes = s.as_bytes();
    let mut i = domain_start;
    while i < bytes.len()
        && (bytes[i].is_ascii_alphanumeric()
            || bytes[i] == b'.'
            || bytes[i] == b'-'
            || bytes[i] == b'_')
    {
        i += 1;
    }
    let domain = &s[domain_start..i];
    let shape_ok = domain.contains('.')
        && !domain.starts_with(['.', '-'])
        && !domain.ends_with('.')
        && !domain.contains("..")
        && domain.split('.').next_back().is_some_and(|t| !t.is_empty());
    if !shape_ok {
        return (local, None);
    }
    ("[email redacted]".to_string(), Some(i))
}

/// Consume `d.d.d.d` (each 0..=255) starting at `start`.
fn scan_ipv4(s: &str, start: usize) -> (String, Option<usize>) {
    let mut i = start;
    let mut octets = 0u8;
    loop {
        let (word, next) = scan_word(s, i);
        if word.is_empty() || word.len() > 3 || !word.chars().all(|c| c.is_ascii_digit()) {
            return (String::new(), None);
        }
        if word.parse::<u16>().unwrap_or(999) > 255 {
            return (String::new(), None);
        }
        octets += 1;
        i = next;
        if octets == 4 {
            // Must NOT be followed by another dot (longer number chain).
            if s[i..].starts_with('.') {
                return (String::new(), None);
            }
            return ("[ip redacted]".to_string(), Some(i));
        }
        if !s[i..].starts_with('.') {
            return (String::new(), None);
        }
        i += 1;
    }
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn unit__scrub__emails_secrets_ips() {
        let s = "contact a@b.com or bob.smith@x.org.uk key plm_deadbeef123 from 10.0.0.4 ok";
        let out = scrub_str(s);
        assert!(out.contains("[email redacted]"), "{out}");
        assert_eq!(out.matches("[email redacted]").count(), 2, "{out}");
        assert!(out.contains("plm_[redacted]"), "{out}");
        assert!(!out.contains("deadbeef123"), "{out}");
        assert!(out.contains("[ip redacted]"), "{out}");
    }

    #[test]
    fn unit__scrub__leaves_diagnostics_alone() {
        let s = "prompt 1204 + completion 828 tokens hit the ctx ceiling 16384";
        assert_eq!(scrub_str(s), s);
    }

    #[test]
    fn unit__scrub__version_numbers_not_ips() {
        let s = "engine b10833 version 0.4.0-dev build 10833";
        assert_eq!(scrub_str(s), s);
    }

    #[test]
    fn unit__scrub__json_recursive() {
        let v = json!({"model": "m", "detail": "mail me x@y.io", "nested": {"ip": "192.168.1.10"}});
        let out = scrub_value(&v);
        assert!(out["detail"].as_str().unwrap().contains("[email redacted]"));
        assert!(out["nested"]["ip"]
            .as_str()
            .unwrap()
            .contains("[ip redacted]"));
        assert_eq!(out["model"], "m");
    }
}
