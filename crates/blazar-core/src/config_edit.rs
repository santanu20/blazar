//! Dotted-key config editing shared by the CLI (`blazar config set`)
//! and the gateway's authenticated `/api/config/set|unset` endpoints.
//!
//! Everything here is pure text-to-text: a raw config document in, a
//! candidate document out. The schema oracle is [`Config::from_toml`]
//! (parse + `deny_unknown_fields` + `validate`) - a candidate it
//! accepts is the ONLY thing callers may persist. Routing grammar:
//! dotted keys target table leaves surgically; bare keys route by what
//! they are (root knob, struct-section leaf, section parent, unknown).
//! Model names may themselves contain dots, so `model_overrides.*`
//! keys get merged-split retries decided by the same oracle.

use crate::config::Config;

/// Key part of a `key = value` TOML line (trimmed), None when no `=`.
/// Tolerant of `key="v"`, `key = "v"` and leading indentation.
#[must_use]
pub fn key_before_eq(line: &str) -> Option<&str> {
    let (k, _) = line.split_once('=')?;
    let k = k.trim();
    (!k.is_empty()).then_some(k)
}
/// Does the config schema know this top-level knob? Two sources,
/// unioned: the keys `Config::default()` serializes (every non-Option
/// knob - strings, ints, floats, bools, enums) and parse-only scalar
/// probes against the serde schema (Option knobs serialize as absent
/// while None, so they never appear in the default TOML). Probes go
/// through `toml::from_str`, never `Config::from_toml` - that also
/// VALIDATES, and probe values are placeholders that must not trip
/// value validation. `deny_unknown_fields` is the authority either way.
#[must_use]
pub fn known_config_key(key: &str) -> bool {
    if key.contains('.') {
        // Dotted table-leaf keys probe with the header + one leaf line:
        // `sglang.grammar_backend` parses as `[sglang]\ngrammar_backend = ...`.
        // Any scalar shape the schema accepts proves the path is real.
        return split_table_path(key).is_some_and(|(path, leaf)| {
            let header = table_header(&path);
            ["\"s\"", "0", "0.5", "true", "[]"]
                .iter()
                .any(|v| toml::from_str::<Config>(&format!("{header}\n{leaf} = {v}\n")).is_ok())
        });
    }
    if !key
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        return false; // TOML bare keys only; anything else is a typo
    }
    let defaults = Config::default()
        .to_toml()
        .expect("serializing the built-in default config cannot fail");
    if defaults.lines().any(|l| key_before_eq(l) == Some(key)) {
        return true;
    }
    ["\"s\"", "0", "0.5", "true", "[]"]
        .iter()
        .any(|v| toml::from_str::<Config>(&format!("{key} = {v}\n")).is_ok())
}
/// Does the schema know this key as a ROOT knob? The routing oracle
/// for bare keys: root knobs replace at the root scope only, section
/// leaves route to their table. Unlike `known_config_key` (which also
/// matches leaf names inside sections), the defaults scan here never
/// crosses the first table header.
#[must_use]
pub fn top_level_key_known(key: &str) -> bool {
    if !key
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        return false; // TOML bare keys only; anything else is a typo
    }
    let defaults = Config::default()
        .to_toml()
        .expect("serializing the built-in default config cannot fail");
    let mut in_root_scope = true;
    for line in defaults.lines() {
        if in_root_scope && line.starts_with('[') {
            in_root_scope = false;
        }
        if in_root_scope && key_before_eq(line) == Some(key) {
            return true;
        }
    }
    // Option knobs serialize as absent while None, so they never appear
    // in the default TOML - probe the schema at the root scope instead.
    ["\"s\"", "0", "0.5", "true", "[]"]
        .iter()
        .any(|v| toml::from_str::<Config>(&format!("{key} = {v}\n")).is_ok())
}
/// Top-level section names the schema knows: every table header of the
/// serialized default config, plus headers already present in the
/// user's file (all-Option sections like `[spec_draft]` render only
/// after a leaf is pinned). First segment only - nested headers
/// (`[model_overrides."m".sglang]`) count as their root table.
#[must_use]
pub fn config_section_names(extra_raw: &str) -> Vec<String> {
    let defaults = Config::default()
        .to_toml()
        .expect("serializing the built-in default config cannot fail");
    let mut names: Vec<String> = Vec::new();
    for line in defaults.lines().chain(extra_raw.lines()) {
        if let Some(first) = header_segments(line).and_then(|segs| segs.first().cloned())
            && !names.contains(&first)
        {
            names.push(first);
        }
    }
    names
}
/// Free-form map sections (`[engine_env]`: string -> string) accept ANY
/// key, so a parse probe there proves nothing about leaf names. One
/// probe with a deliberately impossible key tells them apart from
/// struct sections, which deny unknown fields.
#[must_use]
pub fn is_free_form_map_section(section: &str) -> bool {
    toml::from_str::<Config>(&format!("[{section}]\nzzz_probe_leaf = \"s\"\n")).is_ok()
}
/// Which struct sections carry this bare leaf name? A one-line parse
/// probe per section: `[sglang]\n<key> = <scalar>` parses only when the
/// schema really has that leaf (`sglang.grammar_backend`). Map sections
/// are excluded up front - they accept any key and would always match.
#[must_use]
pub fn bare_leaf_sections(key: &str, raw_file: &str) -> Vec<String> {
    if !key
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        return Vec::new();
    }
    config_section_names(raw_file)
        .into_iter()
        .filter(|s| !is_free_form_map_section(s))
        .filter(|s| {
            ["\"s\"", "0", "0.5", "true", "[]"]
                .iter()
                .any(|v| toml::from_str::<Config>(&format!("[{s}]\n{key} = {v}\n")).is_ok())
        })
        .collect()
}
/// Is this bare key the NAME of a table itself (`warm_peg`, `sglang`)?
#[must_use]
pub fn is_section_parent(key: &str, raw_file: &str) -> bool {
    config_section_names(raw_file).iter().any(|s| s == key)
}
/// Leaf names a struct section accepts, parsed from the serde
/// `deny_unknown_fields` teaching text ("unknown field ... expected one
/// of `a`, `b`"). Sections whose leaves never serialize ([sglang] is
/// all-Option) still enumerate this way; map sections and unexpected
/// error shapes fall back to the leaves the default TOML renders.
#[must_use]
pub fn section_leaf_names(section: &str) -> Vec<String> {
    let probe = format!("[{section}]\nzzz_probe_leaf = 0\n");
    if let Err(err) = toml::from_str::<Config>(&probe) {
        let msg = err.to_string();
        if let Some(idx) = msg.find("expected ") {
            let rest = msg[idx + "expected ".len()..]
                .trim_start_matches("one of ")
                .split(" at line")
                .next()
                .unwrap_or("");
            let leaves: Vec<String> = rest
                .split([',', ' '])
                .filter(|t| !t.is_empty() && *t != "or")
                .map(|t| t.trim().trim_matches('`').trim().to_string())
                .filter(|t| !t.is_empty())
                .collect();
            if !leaves.is_empty() {
                return leaves;
            }
        }
    }
    let defaults = Config::default()
        .to_toml()
        .expect("serializing the built-in default config cannot fail");
    let mut in_section = false;
    let mut leaves = Vec::new();
    for line in defaults.lines() {
        if line.starts_with('[') {
            in_section =
                header_segments(line).is_some_and(|segs| segs.len() == 1 && segs[0] == section);
        } else if in_section && let Some(k) = key_before_eq(line) {
            leaves.push(k.to_string());
        }
    }
    leaves
}
/// Canonical `ModelOverride` leaf names, straight from serde's
/// deny-unknown-fields error (the same list a bad pin teaches on
/// parse). `model_overrides` is a map of per-model tables, so the
/// struct-section probes above cannot discover these names - the
/// section probe would scrape serde type words instead (live finding
/// 2026-11-04: "leaves: struct, `ModelOverride`").
#[must_use]
pub fn override_leaf_names() -> Vec<String> {
    let probe = "[model_overrides.m]\nzzz_probe_leaf = 0\n";
    if let Err(err) = toml::from_str::<Config>(probe) {
        let msg = err.to_string();
        if let Some(idx) = msg.find("expected one of ") {
            let rest = msg[idx + "expected one of ".len()..]
                .split(" at line")
                .next()
                .unwrap_or("");
            let leaves: Vec<String> = rest
                .split([',', ' '])
                .filter(|t| !t.is_empty() && *t != "and" && *t != "or")
                .map(|t| t.trim().trim_matches('`').to_string())
                .filter(|t| !t.is_empty())
                .collect();
            if !leaves.is_empty() {
                return leaves;
            }
        }
    }
    Vec::new()
}
/// Pinned leaves under a map-valued section (`model_overrides.<model>`
/// sub-tables): one `section."name".leaf = value` line per pinned leaf,
/// model names quoted so the line copies straight back into
/// `config get`/`set`. Pinless sub-tables are not pins (same contract
/// as `table_pins` for struct sections).
#[must_use]
pub fn map_table_pins(raw: &str, section: &str) -> Vec<String> {
    let mut pins = Vec::new();
    // Model name of the currently open sub-table, None outside one.
    let mut current: Option<String> = None;
    for line in raw.lines() {
        let t = line.trim();
        if t.starts_with('[') && t.ends_with(']') {
            let header = &t[1..t.len() - 1];
            let mut parts = header.split('.');
            let head = parts.next().unwrap_or("");
            let rest: Vec<&str> = parts.collect();
            current = if head == section && !rest.is_empty() {
                Some(rest.join(".").trim().trim_matches('"').to_string())
            } else {
                None
            };
            continue;
        }
        let Some(name) = &current else {
            continue;
        };
        if let Some((k, v)) = t.split_once('=') {
            let (k, v) = (k.trim(), v.trim());
            if !k.is_empty() && !v.is_empty() {
                pins.push(format!("{section}.\"{name}\".{k} = {v}"));
            }
        }
    }
    pins
}
/// Pinned `key = value` lines of a single-segment `[section]` block in
/// the file (blank lines and comments skipped) - the parent-key view
/// for `config get <table>`.
#[must_use]
pub fn table_pins(raw: &str, section: &str) -> Vec<String> {
    let mut in_section = false;
    let mut pins = Vec::new();
    for line in raw.lines() {
        if line.trim_start().starts_with('[') {
            in_section =
                header_segments(line).is_some_and(|segs| segs.len() == 1 && segs[0] == section);
        } else if in_section {
            let t = line.trim();
            if !t.is_empty() && !t.starts_with('#') && key_before_eq(t).is_some() {
                pins.push(t.to_string());
            }
        }
    }
    pins
}
/// Remove a top-level pin (`key = ...` before the first `[table]`
/// header). Table-scoped keys are never touched - the same knob name
/// inside `[model_overrides."<model>"]` is a different setting. Returns
/// the candidate file and the removed line (None when the key carries
/// no top-level pin).
#[must_use]
pub fn remove_top_level_pin(raw: &str, key: &str) -> (String, Option<String>) {
    let mut removed = None;
    let mut in_root_scope = true;
    let mut out: Vec<&str> = Vec::new();
    for line in raw.lines() {
        if in_root_scope && line.starts_with('[') {
            in_root_scope = false;
        }
        if in_root_scope && key_before_eq(line) == Some(key) {
            removed = Some(line.to_string());
        } else {
            out.push(line);
        }
    }
    (out.join("\n") + "\n", removed)
}
/// Split a dotted config key into its table path and leaf field:
/// `model_overrides.qwen.sglang.grammar_backend` resolves to table
/// path `model_overrides`, `qwen`, `sglang` and leaf `grammar_backend`.
/// Bare keys (no dot) return None and keep the top-level flow.
/// Quote-aware split of a dotted path into raw segment ranges:
/// `a."b.c".d` yields the ranges of `a`, `"b.c"`, `d`. Dots inside double
/// quotes are part of the name. Unbalanced quotes -> None.
#[must_use]
pub fn quoted_seg_ranges(s: &str) -> Option<Vec<(usize, usize)>> {
    let bytes = s.as_bytes();
    let mut ranges = Vec::new();
    let mut start = 0usize;
    let mut i = 0usize;
    let mut in_quotes = false;
    while i < bytes.len() {
        match bytes[i] {
            b'"' => in_quotes = !in_quotes,
            b'.' if !in_quotes => {
                ranges.push((start, i));
                start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    if in_quotes {
        return None;
    }
    ranges.push((start, s.len()));
    Some(ranges)
}
#[must_use]
pub fn split_table_path(key: &str) -> Option<(Vec<&str>, &str)> {
    fn clean_seg(s: &str) -> &str {
        s.strip_prefix('"')
            .and_then(|t| t.strip_suffix('"'))
            .unwrap_or(s)
    }
    if !key.contains('.') {
        return None;
    }
    let ranges = quoted_seg_ranges(key)?;
    if ranges.len() < 2 {
        return None;
    }
    let parts: Vec<&str> = ranges[..ranges.len() - 1]
        .iter()
        .map(|&(a, b)| clean_seg(&key[a..b]))
        .collect();
    let (a, b) = ranges[ranges.len() - 1];
    Some((parts, clean_seg(&key[a..b])))
}
/// Alternative splits for `model_overrides` keys whose MODEL NAME itself
/// contains dots (`model_overrides.qwen3-1.7b.engine`). The straight split
/// yields `[model_overrides, qwen3-1, 7b]` + `engine`, which nests one table
/// too deep and the schema rejects (`unknown field 7b`). Each candidate
/// merges the model name across a different run of segments (ascending, so
/// the shortest name wins) and the caller lets `Config::from_toml` decide
/// which split is the real one - no hardcoded field lists.
///
/// Returns `None` unless the key has at least three parts and targets
/// `model_overrides`; the straight split stays authoritative everywhere
/// else.
#[must_use]
pub fn merged_override_splits(key: &str) -> Option<Vec<(Vec<String>, String)>> {
    let parts: Vec<&str> = key.split('.').collect();
    if parts.len() < 3 || parts[0] != "model_overrides" {
        return None;
    }
    let last = parts.len() - 1;
    let candidates = (1..last)
        .map(|j| {
            let name = parts[1..=j].join(".");
            let mut path = vec!["model_overrides".to_string(), name];
            path.extend(parts[j + 1..last].iter().copied().map(String::from));
            (path, parts[last].to_string())
        })
        .collect();
    Some(candidates)
}
/// Normalize a TOML table header into comparable segments:
/// `[model_overrides."qwen-7b".sglang]` becomes `model_overrides`,
/// `qwen-7b`, `sglang`. Array-of-tables headers (`[[...]]`) are a
/// different structure: `None`.
#[must_use]
pub fn header_segments(line: &str) -> Option<Vec<String>> {
    let trimmed = line.trim();
    let body = trimmed.strip_prefix('[')?.strip_suffix(']')?;
    if body.starts_with('[') {
        return None;
    }
    // Quote-aware: a header like [model_overrides."qwen2.5-0.5b".sglang]
    // keeps the dotted model name as ONE segment.
    Some(
        quoted_seg_ranges(body)?
            .into_iter()
            .map(|(a, b)| body[a..b].trim().trim_matches('"').to_string())
            .collect(),
    )
}
/// Emit a table header, quoting segments that are not TOML bare keys
/// (model names with dots/spaces: `[model_overrides."qwen 7b"]`).
#[must_use]
pub fn table_header(path: &[&str]) -> String {
    let joined = path
        .iter()
        .map(|s| {
            let bare = !s.is_empty()
                && s.chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-');
            if bare {
                (*s).to_string()
            } else {
                format!("{s:?}")
            }
        })
        .collect::<Vec<_>>()
        .join(".");
    format!("[{joined}]")
}
/// Set `leaf = stored` inside the table at `path`, creating the header
/// (after the longest existing prefix section, or at EOF) when absent.
/// Every other line survives byte-for-byte; an existing leaf line is
/// replaced in place.
pub fn set_table_key(raw: &str, path: &[&str], leaf: &str, stored: &str) -> String {
    let want: Vec<String> = path.iter().map(std::string::ToString::to_string).collect();
    let lines: Vec<&str> = raw.lines().collect();
    let newline = format!("{leaf} = {stored}");
    if let Some(h) = lines
        .iter()
        .position(|l| header_segments(l).is_some_and(|segs| segs == want))
    {
        // Replace in place, or insert after the last non-empty line of the
        // section (keeps related keys grouped above the next header).
        let end = lines
            .iter()
            .enumerate()
            .skip(h + 1)
            .find(|(_, l)| l.trim_start().starts_with('['))
            .map_or(lines.len(), |(i, _)| i);
        if let Some(leaf_idx) = lines[h + 1..end]
            .iter()
            .position(|l| key_before_eq(l.trim_start()) == Some(leaf))
        {
            let at = h + 1 + leaf_idx;
            let mut out = lines.clone();
            out[at] = &newline;
            return out.join("\n") + "\n";
        }
        let mut insert_at = end;
        while insert_at > h + 1 && lines[insert_at - 1].trim().is_empty() {
            insert_at -= 1;
        }
        let mut out = lines.clone();
        out.insert(insert_at, &newline);
        return out.join("\n") + "\n";
    }
    // Header absent: create it after the longest existing prefix table
    // (so [model_overrides.m.sglang] lands inside m's section footprint),
    // else append at EOF.
    let mut insert_at = lines.len();
    for depth in (1..path.len()).rev() {
        let prefix: Vec<String> = path[..depth]
            .iter()
            .map(std::string::ToString::to_string)
            .collect();
        if let Some(h) = lines
            .iter()
            .position(|l| header_segments(l).is_some_and(|segs| segs == prefix))
        {
            insert_at = lines
                .iter()
                .enumerate()
                .skip(h + 1)
                .find(|(_, l)| l.trim_start().starts_with('['))
                .map_or(lines.len(), |(i, _)| i);
            break;
        }
    }
    let header = table_header(path);
    let mut out = lines.clone();
    out.insert(insert_at, &header);
    out.insert(insert_at + 1, &newline);
    out.join("\n") + "\n"
}
/// Set `key = stored` at the ROOT scope only: replace existing root
/// pins in place, insert above the first table header when new. A root
/// knob must never rewrite the same leaf name pinned inside a section
/// (`port` vs an env var pinned as `[engine_env] port`) - that clobber
/// was the old section-blind whole-file scan's failure mode.
#[must_use]
pub fn set_root_key(raw: &str, key: &str, stored: &str) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut replaced = false;
    let mut in_root_scope = true;
    for line in raw.lines() {
        if in_root_scope && line.starts_with('[') {
            in_root_scope = false;
        }
        // F128: tolerant key match (compact `key="v"`, indented).
        if in_root_scope && key_before_eq(line).is_some_and(|k| k == key) {
            out.push(format!("{key} = {stored}"));
            replaced = true;
        } else {
            out.push(line.to_string());
        }
    }
    if !replaced {
        // A NEW top-level key must go ABOVE the first table header
        // (`[engine_env]`, `[model_overrides.x]`...); appending at the
        // end would nest it inside that table.
        let insert_at = out
            .iter()
            .position(|l| l.starts_with('['))
            .unwrap_or(out.len());
        out.insert(insert_at, format!("{key} = {stored}"));
    }
    out.join("\n") + "\n"
}
/// Read `path.leaf` from the config file (pins only - table knobs have no
/// defaults in the serialized document). Returns the pinned line, if any.
/// The full value span for a top-level key: `toml::to_string_pretty`
/// spreads array values over continuation lines, so the key line may end
/// at an unbalanced `[`. Walk forward until the bracket depth closes and
/// return the joined span - `config get preload` must print the whole
/// list, not a bare `preload = [`. (Config string arrays carry validated
/// model names; bracket characters inside quoted scalars do not occur.)
#[must_use]
pub fn top_level_span(lines: &[&str], key: &str) -> Option<String> {
    let start = lines
        .iter()
        .position(|l| key_before_eq(l).is_some_and(|k| k == key))?;
    let brackets = |l: &str| (l.matches('[').count(), l.matches(']').count());
    let (mut open, mut close) = brackets(lines[start]);
    let mut end = start;
    while open > close {
        end += 1;
        let line = lines.get(end)?;
        let (opened, closed) = brackets(line);
        open += opened;
        close += closed;
    }
    Some(lines[start..=end].join("\n"))
}
pub fn get_table_key(raw: &str, path: &[&str], leaf: &str) -> Option<String> {
    let want: Vec<String> = path.iter().map(std::string::ToString::to_string).collect();
    let lines: Vec<&str> = raw.lines().collect();
    let h = lines
        .iter()
        .position(|l| header_segments(l).is_some_and(|segs| segs == want))?;
    let end = lines
        .iter()
        .enumerate()
        .skip(h + 1)
        .find(|(_, l)| l.trim_start().starts_with('['))
        .map_or(lines.len(), |(i, _)| i);
    lines[h + 1..end]
        .iter()
        .find(|l| key_before_eq(l.trim_start()) == Some(leaf))
        .map(|l| l.trim().to_string())
}
/// Remove a table-scoped pin (`path.leaf`). The table header survives
/// (empty sections are harmless); returns the removed line, if any.
pub fn remove_table_key(raw: &str, path: &[&str], leaf: &str) -> (String, Option<String>) {
    let want: Vec<String> = path.iter().map(std::string::ToString::to_string).collect();
    let lines: Vec<&str> = raw.lines().collect();
    let Some(h) = lines
        .iter()
        .position(|l| header_segments(l).is_some_and(|segs| segs == want))
    else {
        return (raw.to_string(), None);
    };
    let end = lines
        .iter()
        .enumerate()
        .skip(h + 1)
        .find(|(_, l)| l.trim_start().starts_with('['))
        .map_or(lines.len(), |(i, _)| i);
    if let Some(leaf_idx) = lines[h + 1..end]
        .iter()
        .position(|l| key_before_eq(l.trim_start()) == Some(leaf))
    {
        let at = h + 1 + leaf_idx;
        let removed = lines[at].trim().to_string();
        let mut out: Vec<&str> = lines.clone();
        out.remove(at);
        return (out.join("\n") + "\n", Some(removed));
    }
    (raw.to_string(), None)
}

/// Result of routing + validating one `set` against a raw document.
/// Nothing is persisted here; callers own the file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SetOutcome {
    /// Candidate that passed the [`Config::from_toml`] oracle.
    /// `stored` is the TOML spelling that landed in the document
    /// (the quoted-string twin when the raw value was not a scalar).
    Applied { body: String, stored: String },
    /// Every candidate failed the oracle; the file must not change.
    /// `string_hint` carries the quoted twin when the failure smells
    /// like a string-wanted scalar given bare (`0xff` CPU masks).
    Rejected {
        reason: String,
        string_hint: Option<String>,
    },
    /// The key names a table; callers teach the dotted-leaf form.
    TableNotValue { leaves: Vec<String> },
    /// A bare leaf name matching several struct sections.
    Ambiguous { sections: Vec<String> },
}

/// Route `key = value` against `raw` and validate the candidate.
///
/// Mirrors the long-standing `blazar config set` grammar exactly:
/// scalar probe (values that are not already TOML scalars are stored
/// as double-quoted strings), dotted table-leaf surgery, bare-key
/// routing, merged model-name retries, and the quoted-twin hint.
#[must_use]
pub fn set_candidate(raw: &str, key: &str, value: &str) -> SetOutcome {
    // Values that are not already TOML scalars (numbers, bools, quoted
    // strings, arrays) are written as double-quoted strings: setting
    // `child_transport tcp` must not produce `child_transport = tcp`
    // (invalid TOML). A one-line parse probe decides; the full
    // candidate is validated below either way.
    let scalar = toml::from_str::<toml::Table>(&format!("v = {value}\n")).is_ok();
    let stored = if scalar {
        value.to_string()
    } else {
        format!("{value:?}")
    };
    // One closure so the hint-twin path can rebuild the document with
    // a different value spelling through the EXACT same routing.
    let route = |stored: &str| -> Result<String, SetOutcome> {
        let mut candidate = if let Some((tpath, leaf)) = split_table_path(key) {
            set_table_key(raw, &tpath, leaf, stored)
        } else if top_level_key_known(key) {
            set_root_key(raw, key, stored)
        } else {
            let sections = bare_leaf_sections(key, raw);
            if is_section_parent(key, raw) {
                let leaves = section_leaf_names(key);
                return Err(SetOutcome::TableNotValue { leaves });
            }
            match sections.as_slice() {
                [only] => set_table_key(raw, &[only.as_str()], key, stored),
                many if !many.is_empty() => {
                    return Err(SetOutcome::Ambiguous {
                        sections: many.to_vec(),
                    });
                }
                _ => set_root_key(raw, key, stored),
            }
        };
        // A dotted MODEL NAME (`model_overrides.qwen3-1.7b.engine`)
        // makes the straight candidate nest one table too deep. Retry
        // with merged model names - the schema is the oracle, the first
        // candidate it accepts wins, and the final validation below
        // still gates the winner before anything is persisted.
        if Config::from_toml(&candidate).is_err()
            && let Some(candidates) = merged_override_splits(key)
        {
            for (segs, leaf) in &candidates {
                let segs: Vec<&str> = segs.iter().map(String::as_str).collect();
                let alt = set_table_key(raw, &segs, leaf, stored);
                if Config::from_toml(&alt).is_ok() {
                    candidate = alt;
                    break;
                }
            }
        }
        Ok(candidate)
    };
    let candidate = match route(&stored) {
        Ok(c) => c,
        Err(outcome) => return outcome,
    };
    // Validate BEFORE handing back: a bad value/unknown key must never
    // reach a caller that persists.
    match Config::from_toml(&candidate) {
        Ok(_) => SetOutcome::Applied {
            body: candidate,
            stored,
        },
        Err(e) => {
            // A bare scalar the knob wanted as a STRING (hex CPU mask
            // `0xff` parses as a TOML integer) is the classic trap.
            // Verify the quoted twin actually validates through the
            // same routing before teaching it - a hint that would
            // also fail is worse than none.
            let mut string_hint = None;
            if scalar {
                let quoted = format!("{value:?}");
                if let Ok(twin) = route(&quoted)
                    && Config::from_toml(&twin).is_ok()
                {
                    string_hint = Some(quoted);
                }
            }
            SetOutcome::Rejected {
                reason: e.to_string(),
                string_hint,
            }
        }
    }
}

/// Result of routing one `unset` against a raw document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnsetOutcome {
    /// Candidate (oracle-approved) with the pin removed, plus the
    /// removed line for receipts.
    Removed { body: String, removed_line: String },
    /// Key is real but carries no pin - already at the built-in default.
    NotPinned,
    /// Key is not part of the schema.
    UnknownKey,
    /// Bare key names a table; callers teach the dotted-leaf form.
    TableNotValue,
    /// Bare leaf name matching several struct sections.
    Ambiguous { sections: Vec<String> },
    /// Removing the pin produced a document the oracle rejects; the
    /// file must not change.
    Rejected { reason: String },
}

/// Remove `key` from `raw` and validate the candidate - the same
/// routing grammar as [`set_candidate`], mirrored for removal.
pub fn unset_candidate(raw: &str, key: &str) -> UnsetOutcome {
    if let Some((tpath, leaf)) = split_table_path(key) {
        // Dotted MODEL NAME keys may need the merged split before a
        // pin is actually found (see `set_candidate`).
        let merged = merged_override_splits(key);
        let mut removed_pair = remove_table_key(raw, &tpath, leaf);
        if removed_pair.1.is_none()
            && let Some(candidates) = &merged
        {
            for (segs, leaf) in candidates {
                let segs: Vec<&str> = segs.iter().map(String::as_str).collect();
                let pair = remove_table_key(raw, &segs, leaf);
                if pair.1.is_some() {
                    removed_pair = pair;
                    break;
                }
            }
        }
        let (candidate, removed) = removed_pair;
        return match removed {
            Some(old) => match Config::from_toml(&candidate) {
                Ok(_) => UnsetOutcome::Removed {
                    body: candidate,
                    removed_line: old,
                },
                Err(e) => UnsetOutcome::Rejected {
                    reason: e.to_string(),
                },
            },
            None if known_config_key(key) || merged.is_some() => UnsetOutcome::NotPinned,
            None => UnsetOutcome::UnknownKey,
        };
    }
    let (mut candidate, mut removed) = remove_top_level_pin(raw, key);
    if removed.is_none() {
        // No root pin. A bare SECTION leaf (`ttl_secs`) used to fall
        // through to "not pinned" while the file kept the value -
        // resolve it against the sections first, with the same
        // root/table-name/leaf routing as `set`.
        if is_section_parent(key, raw) {
            return UnsetOutcome::TableNotValue;
        }
        match bare_leaf_sections(key, raw).as_slice() {
            [only] => {
                let (cand, rem) = remove_table_key(raw, &[only.as_str()], key);
                candidate = cand;
                removed = rem;
            }
            many if !many.is_empty() => {
                return UnsetOutcome::Ambiguous {
                    sections: many.to_vec(),
                };
            }
            _ => {}
        }
    }
    match removed {
        Some(old) => match Config::from_toml(&candidate) {
            Ok(_) => UnsetOutcome::Removed {
                body: candidate,
                removed_line: old,
            },
            Err(e) => UnsetOutcome::Rejected {
                reason: e.to_string(),
            },
        },
        None if known_config_key(key) => UnsetOutcome::NotPinned,
        None => UnsetOutcome::UnknownKey,
    }
}

/// One settings-group entry for UI/schema browsers: a config section,
/// the leaf knobs it accepts, and whether it is a free-form map (any
/// key goes - leaves are then the known sub-table names, if any).
#[derive(Debug, Clone)]
pub struct SectionSchema {
    pub name: String,
    pub leaves: Vec<String>,
    pub is_map: bool,
}

/// The settable surface of the config: root knobs plus every section
/// the schema knows (unioned with headers already pinned in the user's
/// file - all-Option sections only render after a leaf is pinned).
/// `raw_file` is the current config document.
#[must_use]
pub fn schema_sections(raw_file: &str) -> Vec<SectionSchema> {
    let defaults = Config::default()
        .to_toml()
        .expect("serializing the built-in default config cannot fail");
    let mut root_leaves = Vec::new();
    let mut in_root_scope = true;
    for line in defaults.lines() {
        if in_root_scope && line.starts_with('[') {
            in_root_scope = false;
        }
        if in_root_scope && let Some(k) = key_before_eq(line) {
            root_leaves.push(k.to_string());
        }
    }
    let mut out = vec![SectionSchema {
        name: "root".to_string(),
        leaves: root_leaves,
        is_map: false,
    }];
    for section in config_section_names(raw_file) {
        let is_map = is_free_form_map_section(&section);
        let leaves = if section == "model_overrides" {
            override_leaf_names()
        } else if is_map {
            Vec::new()
        } else {
            section_leaf_names(&section)
        };
        out.push(SectionSchema {
            name: section,
            leaves,
            is_map,
        });
    }
    out
}

/// Every scalar pin in the file as a dotted key, in document order:
/// root knobs bare (`log_level`), table leaves dotted
/// (`sglang.grammar_backend`), dotted headers re-quoted exactly the way
/// the set/unset grammar accepts them (`model_overrides."qwen3:8b".ctx_size`).
/// Array-of-tables sections (`[[keys]]`, `[[models]]`, ...) are skipped:
/// their entries are not dotted-key-addressable - dedicated planes own
/// them. Powers "pinned in file vs built-in default" badges in editors.
#[must_use]
pub fn file_pins(raw: &str) -> Vec<String> {
    let bare = |s: &str| {
        !s.is_empty()
            && s.chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    };
    let dotted_seg = |s: &str| {
        if bare(s) {
            s.to_string()
        } else {
            format!("\"{s}\"")
        }
    };
    let mut pins = Vec::new();
    let mut path: Vec<String> = Vec::new();
    let mut in_array_section = false;
    for line in raw.lines() {
        let t = line.trim();
        if t.starts_with("[[") {
            in_array_section = true;
            path.clear();
        } else if t.starts_with('[') {
            in_array_section = false;
            path = header_segments(line).unwrap_or_default();
        } else if !in_array_section
            && !t.starts_with('#')
            && let Some(k) = key_before_eq(t)
        {
            if path.is_empty() {
                pins.push(k.to_string());
            } else {
                let joined = path
                    .iter()
                    .map(|s| dotted_seg(s))
                    .collect::<Vec<_>>()
                    .join(".");
                pins.push(format!("{joined}.{k}"));
            }
        }
    }
    pins
}

/// The pinned line for `key` (root span or table leaf, with the
/// merged model-name retry), if the file pins one. Powers "current
/// value" displays in editors.
#[must_use]
pub fn get_pin(raw: &str, key: &str) -> Option<String> {
    if let Some((tpath, leaf)) = split_table_path(key) {
        let merged = merged_override_splits(key);
        return get_table_key(raw, &tpath, leaf).or_else(|| {
            merged?.iter().find_map(|(segs, leaf)| {
                let segs: Vec<&str> = segs.iter().map(String::as_str).collect();
                get_table_key(raw, &segs, leaf)
            })
        });
    }
    let lines: Vec<&str> = raw.lines().collect();
    top_level_span(&lines, key)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(non_snake_case)] // suite convention: unit__scenario__expected
    fn unit__known_config_key__defaults_options_and_typos() {
        // Serialized-by-default knobs (every scalar/enum shape).
        for k in [
            "host",
            "port",
            "slots",
            "spec",
            "update_channel",
            "ctx_extend",
            "keys",
        ] {
            assert!(known_config_key(k), "{k} must be a known knob");
        }
        // Option knobs: absent from the default TOML while None - only
        // the parse probes can recognize them.
        for k in [
            "child_auth",
            "log_level",
            "kv_unified",
            "mistralrs_pa_memory_fraction",
        ] {
            assert!(
                known_config_key(k),
                "{k} (Option knob) must be known via probes"
            );
        }
        // Typos and nonsense: bare-key charset gate + schema probes.
        for k in ["slot", "bogus_key", "PORT", "a b", "keys=1"] {
            assert!(!known_config_key(k), "{k:?} must be unknown");
        }
    }
    #[test]
    #[allow(non_snake_case)] // suite convention: unit__scenario__expected
    fn unit__remove_top_level_pin__removes_only_root_scope() {
        // R4 pin: the same knob name at top level and inside a model
        // override table is TWO settings - unset must only touch root.
        let raw = concat!(
            "cache_type = \"f16\"\n",
            "host = \"127.0.0.1\"\n",
            "\n",
            "[model_overrides.\"m\"]\n",
            "cache_type = \"q8_0\"\n",
            "slots = 4\n",
        );
        let (out, removed) = remove_top_level_pin(raw, "cache_type");
        assert_eq!(removed.as_deref(), Some("cache_type = \"f16\""));
        assert!(out.contains("host = \"127.0.0.1\""), "other pins kept");
        assert!(
            out.contains("cache_type = \"q8_0\""),
            "table-scoped pin untouched"
        );
        assert!(out.contains("slots = 4"), "table body kept");
        // The candidate must still be schema-valid TOML.
        assert!(Config::from_toml(&out).is_ok());
    }
    #[test]
    #[allow(non_snake_case)]
    fn unit__top_level_span__joins_multiline_array_and_stops_at_next_key() {
        // `toml::to_string_pretty` layout: the key line opens the array,
        // elements ride continuation lines. The span must join through the
        // closing bracket and NOT swallow the next key.
        let lines = [
            "slots = 8",
            "preload = [",
            "    \"m1\",",
            "    \"m2\",",
            "]",
            "session_bank = true",
        ];
        assert_eq!(
            top_level_span(&lines, "preload").as_deref(),
            Some("preload = [\n    \"m1\",\n    \"m2\",\n]")
        );
        // Plain scalar keys return their single line unchanged.
        assert_eq!(
            top_level_span(&lines, "slots").as_deref(),
            Some("slots = 8")
        );
        assert_eq!(
            top_level_span(&lines, "session_bank").as_deref(),
            Some("session_bank = true")
        );
        // Unknown key -> None (the CLI turns that into its teaching error).
        assert_eq!(top_level_span(&lines, "no_such_knob"), None);
        // An unterminated array cannot be joined whole -> None (fail loud
        // rather than print a truncated value).
        let unterminated = ["preload = [", "    \"m1\","];
        assert_eq!(top_level_span(&unterminated, "preload"), None);
    }
    #[test]
    #[allow(non_snake_case)] // suite convention: unit__scenario__expected
    fn unit__known_config_key__dotted_table_paths() {
        // Real table-leaf knobs across all three engine tables and both
        // scopes (global table + model override nesting).
        for k in [
            "sglang.grammar_backend",
            "sglang.cuda_graph_bs",
            "sglang.stream_interval",
            "mistralrs.max_batch_size",
            "mistralrs.mtp_draft_sampling",
            "mistralrs.device_layers",
            "model_overrides.m.sglang.grammar_backend",
            "model_overrides.m.mistralrs.prefix_cache_n",
        ] {
            assert!(known_config_key(k), "{k} must be a known dotted knob");
        }
        // Plausible-but-fake leaves and tables must be rejected by the
        // schema probes (deny_unknown_fields on the tuning structs).
        for k in [
            "sglang.bogus",
            "mistralrs.bogus",
            "sglangx.grammar_backend",
            "model_overrides.m.sglang.bogus",
            "model_overrides.m.bogus.k",
            "model_overrides.bogus.sglang",
        ] {
            assert!(!known_config_key(k), "{k} must be unknown");
        }
    }
    #[test]
    #[allow(non_snake_case)] // suite convention: unit__scenario__expected
    fn unit__set_table_key__replace_create_and_grouping() {
        // Replace in place inside an existing header, byte-preserving the
        // rest of the section.
        let raw = concat!(
            "host = \"127.0.0.1\"\n",
            "\n",
            "[sglang]\n",
            "stream_interval = 1\n",
            "mem_fraction_static = 0.8\n",
        );
        let out = set_table_key(raw, &["sglang"], "stream_interval", "4");
        assert!(out.contains("stream_interval = 4"), "replaced in place");
        assert!(!out.contains("stream_interval = 1"), "old line gone");
        assert!(out.contains("mem_fraction_static = 0.8"), "sibling kept");
        assert!(out.starts_with("host ="), "root scope untouched");
        assert!(Config::from_toml(&out).is_ok());

        // New leaf in an existing section lands inside it, above the next
        // header (grouped with its siblings, not appended at EOF).
        let raw2 = concat!(
            "[sglang]\n",
            "stream_interval = 1\n",
            "\n",
            "[mistralrs]\n",
            "max_batch_size = 2\n",
        );
        let out2 = set_table_key(raw2, &["sglang"], "grammar_backend", "\"outlines\"");
        let gpos = out2.find("grammar_backend").expect("leaf inserted");
        let mpos = out2.find("[mistralrs]").expect("mistralrs header");
        assert!(gpos < mpos, "leaf grouped inside [sglang]");
        assert!(
            out2.contains("grammar_backend = \"outlines\""),
            "stored verbatim"
        );
        assert!(Config::from_toml(&out2).is_ok());

        // Absent header: created after the longest existing prefix
        // ([model_overrides."qwen 7b"] exists; the .sglang child nests
        // right after its parent section, not at EOF over other tables).
        let raw3 = concat!(
            "[sglang]\n",
            "stream_interval = 1\n",
            "\n",
            "[model_overrides.\"qwen 7b\"]\n",
            "slots = 2\n",
            "\n",
            "[mistralrs]\n",
            "max_batch_size = 2\n",
        );
        let out3 = set_table_key(
            raw3,
            &["model_overrides", "qwen 7b", "sglang"],
            "page_size",
            "32",
        );
        let hpos = out3
            .find("[model_overrides.\"qwen 7b\".sglang]")
            .expect("nested header created, non-bare segment quoted");
        let global_pos = out3.find("[sglang]").expect("global table");
        let mistral_pos = out3.find("[mistralrs]").expect("mistralrs table");
        assert!(hpos > global_pos, "nested after global [sglang]");
        assert!(hpos < mistral_pos, "nested before unrelated [mistralrs]");
        assert!(out3.contains("page_size = 32"), "leaf under new header");
        assert!(Config::from_toml(&out3).is_ok());

        // No prefix at all: header appended at EOF.
        let out4 = set_table_key("host = \"h\"\n", &["sglang"], "page_size", "16");
        assert!(
            out4.trim_end().ends_with("[sglang]\npage_size = 16")
                || out4.contains("\n[sglang]\npage_size = 16\n"),
            "EOF append shape: {out4:?}"
        );
        assert!(Config::from_toml(&out4).is_ok());
    }
    #[test]
    #[allow(non_snake_case)] // suite convention: unit__scenario__expected
    fn unit__get_and_remove_table_key__roundtrip() {
        let raw = concat!(
            "[sglang]\n",
            "stream_interval = 3\n",
            "\n",
            "[model_overrides.\"m\"]\n",
            "slots = 2\n",
        );
        assert_eq!(
            get_table_key(raw, &["sglang"], "stream_interval").as_deref(),
            Some("stream_interval = 3"),
            "get returns the raw pinned line"
        );
        assert_eq!(
            get_table_key(raw, &["model_overrides", "m"], "slots").as_deref(),
            Some("slots = 2"),
            "quoted segment headers match"
        );
        assert!(
            get_table_key(raw, &["sglang"], "page_size").is_none(),
            "unpinned leaf is None"
        );
        assert!(
            get_table_key(raw, &["mistralrs"], "max_batch_size").is_none(),
            "absent table is None"
        );

        let (out, removed) = remove_table_key(raw, &["sglang"], "stream_interval");
        assert_eq!(
            removed.as_deref(),
            Some("stream_interval = 3"),
            "removed line reported for the was: message"
        );
        assert!(
            !out.contains("stream_interval"),
            "leaf gone from the section"
        );
        assert!(out.contains("[sglang]"), "header survives");
        assert!(out.contains("slots = 2"), "other tables untouched");
        assert!(Config::from_toml(&out).is_ok());

        let (out2, removed2) = remove_table_key(raw, &["sglang"], "page_size");
        assert!(removed2.is_none(), "unpinned leaf removes nothing");
        assert_eq!(out2, raw, "byte-identical when nothing to remove");
    }
    #[test]
    #[allow(non_snake_case)] // suite convention: unit__scenario__expected
    fn unit__split_table_path_and_header_segments__grammar() {
        assert!(split_table_path("slots").is_none(), "bare key");
        assert_eq!(
            split_table_path("sglang.page_size"),
            Some((vec!["sglang"], "page_size"))
        );
        assert_eq!(
            split_table_path("model_overrides.m.sglang.page_size"),
            Some((vec!["model_overrides", "m", "sglang"], "page_size"))
        );
        // Quoted segment with dots stays whole; quotes stripped from output.
        assert_eq!(
            split_table_path("model_overrides.\"qwen2.5-0.5b-instruct\".mistralrs.prefix_cache_n"),
            Some((
                vec!["model_overrides", "qwen2.5-0.5b-instruct", "mistralrs"],
                "prefix_cache_n"
            ))
        );
        // Unterminated quoting is a grammar error, not a path.
        assert!(split_table_path("model_overrides.\"qwen2.5.sglang.k").is_none());
        // Header with dots inside the quoted segment stays one segment.
        assert_eq!(
            header_segments("[model_overrides.\"qwen2.5-0.5b-instruct\".mistralrs]").unwrap(),
            vec![
                "model_overrides".to_string(),
                "qwen2.5-0.5b-instruct".to_string(),
                "mistralrs".to_string()
            ]
        );
        assert!(header_segments("[model_overrides.\"qwen2.5.mistralrs]").is_none());
        // Header parsing: quotes stripped, [[array]] refused.
        assert_eq!(
            header_segments("[model_overrides.\"qwen 7b\".sglang]").unwrap(),
            vec!["model_overrides", "qwen 7b", "sglang"]
        );
        assert_eq!(header_segments("[sglang]").unwrap(), vec!["sglang"]);
        assert!(header_segments("[[engine_env]]").is_none(), "array header");
        assert!(
            header_segments("stream_interval = 1").is_none(),
            "not a header"
        );
        // Emission: bare segments bare, others quoted.
        assert_eq!(
            table_header(&["model_overrides", "qwen 7b", "sglang"]),
            "[model_overrides.\"qwen 7b\".sglang]"
        );
        assert_eq!(table_header(&["sglang"]), "[sglang]");
    }
    #[test]
    #[allow(non_snake_case)] // suite convention: unit__scenario__expected
    fn unit__merged_override_splits__candidates_ascending() {
        // A dotted model name yields one candidate per merge point,
        // shortest name first; the schema picks the valid one downstream.
        let cands = merged_override_splits("model_overrides.qwen3-1.7b.engine").unwrap();
        assert_eq!(cands.len(), 2);
        assert_eq!(
            cands[0],
            (
                vec![
                    "model_overrides".to_string(),
                    "qwen3-1".to_string(),
                    "7b".to_string()
                ],
                "engine".to_string()
            )
        );
        assert_eq!(
            cands[1],
            (
                vec!["model_overrides".to_string(), "qwen3-1.7b".to_string()],
                "engine".to_string()
            )
        );
        // Deeper leaves keep the tail path after the merged name.
        let cands = merged_override_splits("model_overrides.qwen3-1.7b.sglang.dtype").unwrap();
        assert_eq!(cands.len(), 3);
        assert_eq!(
            cands[1],
            (
                vec![
                    "model_overrides".to_string(),
                    "qwen3-1.7b".to_string(),
                    "sglang".to_string()
                ],
                "dtype".to_string()
            )
        );
        // Only model_overrides keys qualify, and only with a leaf beyond
        // the name itself.
        assert!(merged_override_splits("sglang.qwen3-1.7b").is_none());
        assert!(merged_override_splits("model_overrides.m").is_none());
        assert!(merged_override_splits("slots").is_none());
    }
    #[test]
    #[allow(non_snake_case)] // suite convention: unit__scenario__expected
    fn unit__config_set__dotted_model_name_resolves_via_schema() {
        let raw = "[sglang]\ngrammar_backend = \"xgrammar\"\n";
        // The straight split nests one table too deep - pins the bug the
        // merged retry exists to fix.
        let straight = set_table_key(
            raw,
            &["model_overrides", "qwen3-1", "7b"],
            "engine",
            "\"sglang-0.5.19\"",
        );
        assert!(
            Config::from_toml(&straight).is_err(),
            "straight split must keep failing until the schema accepts the merged name"
        );
        // The merged candidate quotes the dotted name into ONE header
        // segment and validates.
        let merged = set_table_key(
            raw,
            &["model_overrides", "qwen3-1.7b"],
            "engine",
            "\"sglang-0.5.19\"",
        );
        Config::from_toml(&merged).expect("merged split must validate");
        assert!(
            merged.contains("[model_overrides.\"qwen3-1.7b\"]"),
            "dotted model name must be a single quoted header segment"
        );
        // get/remove round-trip on the merged text.
        assert_eq!(
            get_table_key(&merged, &["model_overrides", "qwen3-1.7b"], "engine").as_deref(),
            Some("engine = \"sglang-0.5.19\"")
        );
        let (after, removed) =
            remove_table_key(&merged, &["model_overrides", "qwen3-1.7b"], "engine");
        assert!(removed.is_some());
        assert!(!after.contains("sglang-0.5.19"), "unset must drop the pin");
        Config::from_toml(&after).expect("config stays valid after unset");
    }
    #[test]
    #[allow(non_snake_case)] // suite convention: unit__scenario__expected
    fn unit__remove_top_level_pin__unpinned_is_none_and_byte_preserving() {
        let raw = "host = \"127.0.0.1\"\n";
        let (out, removed) = remove_top_level_pin(raw, "slots");
        assert!(removed.is_none());
        assert_eq!(out, raw, "nothing pinned - bytes pass through");
        // Commented-out lines are not pins (no '=' before the comment).
        let (out, _) = remove_top_level_pin("# cache_type = \"f16\"\n", "cache_type");
        assert!(out.contains("# cache_type"));
    }
    #[test]
    #[allow(non_snake_case)] // suite convention: unit__scenario__expected
    fn unit__top_level_key_known__root_knobs_yes_section_leaves_no() {
        // Root knobs (slots renders; child_auth is an Option knob the
        // defaults never render but the schema accepts at root scope).
        assert!(top_level_key_known("slots"));
        assert!(top_level_key_known("child_auth"));
        // Section leaves must NOT route as root keys.
        assert!(!top_level_key_known("ttl_secs"));
        assert!(!top_level_key_known("grammar_backend"));
        // Typos and table names are not root knobs.
        assert!(!top_level_key_known("zzz_never_a_knob"));
        assert!(!top_level_key_known("warm_peg"));
    }
    #[test]
    #[allow(non_snake_case)] // suite convention: unit__scenario__expected
    fn unit__bare_leaf_sections__resolves_struct_section_leaves() {
        assert_eq!(
            bare_leaf_sections("ttl_secs", ""),
            vec!["semantic_cache".to_string()],
            "ttl_secs belongs to exactly one section"
        );
        assert_eq!(
            bare_leaf_sections("grammar_backend", ""),
            vec!["sglang".to_string()]
        );
        // Free-form map sections must never match (they accept any key).
        assert_eq!(
            bare_leaf_sections("zzz_never_a_leaf", ""),
            [] as [String; 0]
        );
        // Flat root knobs (`spec_draft_threads` and friends are ROOT
        // fields, not a [spec_draft] table) resolve to no section.
        assert_eq!(
            bare_leaf_sections("spec_draft_threads", ""),
            [] as [String; 0]
        );
    }
    #[test]
    #[allow(non_snake_case)] // suite convention: unit__scenario__expected
    fn unit__config_section_names__defaults_union_file_headers() {
        let names = config_section_names("[stale_custom]\nx = 1\n");
        assert!(names.contains(&"sglang".to_string()));
        assert!(names.contains(&"semantic_cache".to_string()));
        assert!(
            names.contains(&"stale_custom".to_string()),
            "file headers join the defaults set"
        );
        assert!(is_section_parent("warm_peg", ""));
        assert!(!is_section_parent("ttl_secs", ""));
    }
    #[test]
    #[allow(non_snake_case)] // suite convention: unit__scenario__expected
    fn unit__section_leaf_names__schema_enumerates_unrendered_sections() {
        // [sglang] renders empty in defaults (all-Option) - the deny
        // teaching text is the only enumerator.
        let sglang = section_leaf_names("sglang");
        assert!(sglang.contains(&"grammar_backend".to_string()));
        // Rendered sections enumerate too.
        let cache = section_leaf_names("semantic_cache");
        assert!(cache.contains(&"ttl_secs".to_string()));
        assert!(cache.contains(&"enabled".to_string()));
    }
    #[test]
    #[allow(non_snake_case)] // suite convention: unit__scenario__expected
    fn unit__set_root_key__section_pin_with_same_leaf_name_survives() {
        // Schema-valid clobber case: root `port` knob vs an env var
        // literally named `port` pinned under [engine_env].
        let raw = "host = \"127.0.0.1\"\n[engine_env]\nport = \"1\"\n";
        let out = set_root_key(raw, "port", "9000");
        assert!(
            out.contains("[engine_env]\nport = \"1\""),
            "section pin must survive a root-key set of the same name"
        );
        assert!(out.contains("port = 9000"));
        // New root keys land above the first table header, not nested.
        let fresh = set_root_key("[engine_env]\nA = \"b\"\n", "port", "9000");
        assert!(fresh.starts_with("port = 9000\n"));
    }
    #[test]
    #[allow(non_snake_case)] // suite convention: unit__scenario__expected
    fn unit__config_unset__bare_section_leaf_finds_the_real_pin() {
        // BUG-2 pin: `config unset ttl_secs` used to report "not
        // pinned" while the file kept the value - the root scan misses,
        // the section-aware retry must find and remove it.
        let raw = "[semantic_cache]\nthreshold = 0.4\nttl_secs = 2\n";
        let (root_cand, root_removed) = remove_top_level_pin(raw, "ttl_secs");
        assert!(root_removed.is_none(), "root scope never sees section pins");
        assert_eq!(
            bare_leaf_sections("ttl_secs", raw),
            vec!["semantic_cache".to_string()]
        );
        let (cand, removed) = remove_table_key(&root_cand, &["semantic_cache"], "ttl_secs");
        assert_eq!(removed.as_deref(), Some("ttl_secs = 2"));
        assert!(!cand.contains("ttl_secs"));
        assert!(cand.contains("threshold = 0.4"), "siblings survive");
        Config::from_toml(&cand).expect("config stays valid after unset");
    }
    #[test]
    #[allow(non_snake_case)] // suite convention: unit__scenario__expected
    fn unit__table_pins__parent_view_reads_file_pins_only() {
        let raw = "# lead comment\n[semantic_cache]\nenabled = true\n\n# inner\n[sglang]\n";
        let pins = table_pins(raw, "semantic_cache");
        assert_eq!(pins, vec!["enabled = true".to_string()]);
        assert_eq!(table_pins(raw, "sglang"), [] as [String; 0]);
    }
    #[test]
    #[allow(non_snake_case)] // suite convention: unit__scenario__expected
    fn unit__file_pins__dotted_keys_skip_arrays_and_comments() {
        let raw = concat!(
            "# lead comment with a fake key: bogus = 1\n",
            "host = \"127.0.0.1\"\n",
            "log_level = \"debug\"\n",
            "\n",
            "[sglang]\n",
            "grammar_backend = \"outlines\"\n",
            "\n",
            "[engine_env]\n",
            "MY_VAR = \"x\"\n",
            "\n",
            "[model_overrides.\"qwen3:8b\"]\n",
            "ctx_size = 8192\n",
            "\n",
            "[[keys]]\n",
            "name = \"admin\"\n",
            "key = \"sk-live\"\n",
            "\n",
            "[semantic_cache]\n",
            "enabled = true\n",
        );
        assert_eq!(
            file_pins(raw),
            vec![
                "host",
                "log_level",
                "sglang.grammar_backend",
                "engine_env.MY_VAR",
                // dotted header re-quoted exactly as set/unset accepts
                "model_overrides.\"qwen3:8b\".ctx_size",
                // [[keys]] entries skipped: dedicated plane owns them
                "semantic_cache.enabled",
            ]
        );
    }
    #[test]
    #[allow(non_snake_case)] // suite convention: unit__scenario__expected
    fn unit__map_table_pins__subtable_pins_listed_with_model_names() {
        // Live shape: overlays pin model sub-tables whose names routinely
        // carry dots (quoted in the header) or not (plain). The bare
        // `config get model_overrides` must list them leaf-by-leaf.
        let raw = r#"
host = "127.0.0.1"

[model_overrides."gemma3-270m"]
rpc_servers = "127.0.0.1:51001"
ctx = 8192

[model_overrides.foo]
engine = "sglang"
"#;
        let pins = map_table_pins(raw, "model_overrides");
        assert!(
            pins.contains(
                &("model_overrides.\"gemma3-270m\".rpc_servers = \"127.0.0.1:51001\"".to_string())
            ),
            "quoted dotted name: {pins:?}"
        );
        assert!(
            pins.contains(&"model_overrides.\"gemma3-270m\".ctx = 8192".to_string()),
            "second leaf listed: {pins:?}"
        );
        assert!(
            pins.contains(&"model_overrides.\"foo\".engine = \"sglang\"".to_string()),
            "plain name quoted on output for copy-paste safety: {pins:?}"
        );
        assert_eq!(pins.len(), 3, "one line per pinned leaf: {pins:?}");
    }
    #[test]
    #[allow(non_snake_case)] // suite convention: unit__scenario__expected
    fn unit__map_table_pins__absent_section_and_pinless_subtables_empty() {
        assert!(map_table_pins("host = \"127.0.0.1\"\n", "model_overrides").is_empty());
        // A sub-table header with no pinned leaf shows nothing: nothing
        // is pinned, mirroring "no pins" for struct tables.
        assert!(
            map_table_pins("[model_overrides.foo]\n", "model_overrides").is_empty(),
            "pinless sub-table is not a pin"
        );
    }
    #[test]
    #[allow(non_snake_case)] // suite convention: unit__scenario__expected
    fn unit__override_leaf_names__real_fields_not_serde_garbage() {
        // Regression pin: the struct-probe trick scraped serde type words
        // ("struct", "ModelOverride") for the map-valued table and printed
        // them as the leaves hint (live 2026-11-04). The map path must
        // name real ModelOverride knobs instead.
        let leaves = override_leaf_names();
        assert!(leaves.contains(&"rpc_servers".to_string()), "{leaves:?}");
        assert!(
            !leaves.iter().any(|l| l == "struct" || l == "ModelOverride"),
            "serde type tokens must not leak as leaves: {leaves:?}"
        );
    }
}
