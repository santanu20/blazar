//! Task-aware, bounded reasoning verification for adaptive inference.
//!
//! This module deliberately separates verifiable signals from heuristic
//! signals. JSON Schema and simple arithmetic checks are deterministic;
//! answer agreement is a self-consistency signal, not proof of correctness.

use std::cmp::Reverse;
use std::collections::HashMap;

use serde_json::Value;

#[derive(Clone, Debug)]
struct ArithmeticOracle {
    expression: String,
    expected: f64,
}

#[derive(Clone, Debug)]
struct TableOracle {
    label: String,
    value: f64,
}

/// Request-local deterministic verification profile. Only the latest user
/// turn is inspected; general prompt text is never executed.
#[derive(Clone, Debug, Default)]
pub(crate) struct TaskProfile {
    arithmetic: Option<ArithmeticOracle>,
    table: Option<TableOracle>,
}

impl TaskProfile {
    pub(crate) fn from_request(request: &Value) -> Self {
        let latest_user_texts = latest_user_texts(request);
        let arithmetic = latest_user_texts
            .iter()
            .rev()
            .find_map(|text| arithmetic_oracle_from_text(text));
        let table = latest_user_texts
            .iter()
            .rev()
            .find_map(|text| table_oracle_from_text(text));
        Self { arithmetic, table }
    }

    pub(crate) fn math_expected(&self) -> Option<f64> {
        self.arithmetic.as_ref().map(|oracle| oracle.expected)
    }

    pub(crate) fn math_expression(&self) -> Option<&str> {
        self.arithmetic
            .as_ref()
            .map(|oracle| oracle.expression.as_str())
    }
    pub(crate) fn table_label(&self) -> Option<&str> {
        self.table.as_ref().map(|oracle| oracle.label.as_str())
    }

    pub(crate) fn table_value(&self) -> Option<f64> {
        self.table.as_ref().map(|oracle| oracle.value)
    }


    pub(crate) fn math_valid(&self, content: &str) -> Option<bool> {
        let expected = self.math_expected()?;
        Some(
            numeric_answer(content)
                .is_some_and(|actual| numbers_equal(actual, expected))
        )
    }

    pub(crate) fn math_feedback(&self) -> Option<String> {
        let oracle = self.arithmetic.as_ref()?;
        Some(format!(
            "A deterministic arithmetic check evaluated `{}` to {}. Recompute the expression and return the corrected result.",
            oracle.expression,
            format_number(oracle.expected)
        ))
    }

    pub(crate) fn table_valid(&self, content: &str) -> Option<bool> {
        let oracle = self.table.as_ref()?;
        let answer = answer_text(content).unwrap_or_default();
        let label_tokens: Vec<String> = oracle
            .label
            .to_ascii_lowercase()
            .split(|ch: char| !ch.is_ascii_alphanumeric())
            .filter(|token| !token.is_empty())
            .map(str::to_string)
            .collect();
        let answer_tokens: Vec<String> = answer
            .to_ascii_lowercase()
            .split(|ch: char| !ch.is_ascii_alphanumeric())
            .filter(|token| !token.is_empty())
            .map(str::to_string)
            .collect();
        let label_matches = !label_tokens.is_empty()
            && answer_tokens
                .windows(label_tokens.len())
                .any(|window| window == label_tokens.as_slice());
        let value_matches = numeric_answer(&answer)
            .is_some_and(|value| numbers_equal(value, oracle.value));
        Some(label_matches && value_matches)
    }

    pub(crate) fn table_feedback(&self) -> Option<String> {
        let oracle = self.table.as_ref()?;
        Some(format!(
            "A deterministic table check found that {} has the maximum value {}. Return that label and value.",
            oracle.label,
            format_number(oracle.value)
        ))
    }

    pub(crate) fn task_valid(&self, content: &str) -> Option<bool> {
        self.math_valid(content)
            .or_else(|| self.table_valid(content))
    }

    pub(crate) fn task_feedback(&self) -> Option<String> {
        self.math_feedback().or_else(|| self.table_feedback())
    }

    pub(crate) fn has_deterministic_verifier(&self) -> bool {
        self.arithmetic.is_some() || self.table.is_some()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CandidateDecision {
    pub(crate) winner_index: usize,
    pub(crate) agreement: usize,
    pub(crate) candidates: usize,
    pub(crate) answer_hash: Option<u64>,
    pub(crate) vote_tallies: Vec<(u64, usize)>,
    pub(crate) verifier: &'static str,
}

/// Select a candidate using hard contract checks first, a recognized
/// deterministic oracle second, then completed output and answer-level
/// agreement. Consensus is only a tie-breaker / confidence signal; it is
/// never described as factual verification.
pub(crate) fn pick_candidate(
    sentinel: &crate::sentinel::Sentinel,
    candidates: &[(u16, Value)],
    schema: Option<&Value>,
    request: &Value,
) -> CandidateDecision {
    let profile = TaskProfile::from_request(request);
    let answer_keys: Vec<Option<String>> = candidates
        .iter()
        .map(|(_, body)| response_content(body).and_then(normalize_answer))
        .collect();
    let mut counts = HashMap::<String, usize>::new();
    for key in answer_keys.iter().flatten() {
        *counts.entry(key.clone()).or_default() += 1;
    }

    let score = |index: usize| {
        let (status, body) = &candidates[index];
        let http_success = (200..300).contains(status);
        let schema_valid = schema.is_none_or(|schema| {
            response_content(body)
                .and_then(|text| serde_json::from_str::<Value>(text).ok())
                .is_some_and(|instance| sentinel.schema_check(schema, &instance).is_none())
        });
        let task_valid = if profile.has_deterministic_verifier() {
            profile
                .task_valid(&response_content(body).unwrap_or_default())
                .unwrap_or(false)
        } else {
            true
        };
        let finish_reason = body
            .pointer("/choices/0/finish_reason")
            .and_then(Value::as_str);
        let complete = matches!(finish_reason, Some("stop" | "content_filter") | None);
        let non_empty = response_content(body).is_some_and(|text| !text.trim().is_empty());
        let votes = answer_keys[index]
            .as_ref()
            .and_then(|key| counts.get(key))
            .copied()
            .unwrap_or(0);
        let hard_gates = u8::from(schema_valid) + u8::from(task_valid);
        (
            u8::from(http_success),
            hard_gates,
            u8::from(schema_valid),
            u8::from(task_valid),
            u8::from(complete),
            u8::from(non_empty),
            votes,
            Reverse(index),
        )
    };

    let winner_index = (0..candidates.len())
        .max_by_key(|index| score(*index))
        .unwrap_or(0);
    let winning_key = answer_keys.get(winner_index).and_then(Option::as_ref);
    let agreement = winning_key
        .and_then(|key| counts.get(key))
        .copied()
        .unwrap_or(0);
    let verifier = if profile.math_expected().is_some() {
        "math_exact"
    } else if profile.table_label().is_some() {
        "table_lookup_exact"
    } else if schema.is_some() {
        "json_schema+consensus"
    } else {
        "answer_consensus"
    };

    let mut vote_tallies: Vec<(u64, usize)> = counts
        .iter()
        .map(|(answer, count)| (stable_hash(answer), *count))
        .collect();
    vote_tallies.sort_unstable_by_key(|(hash, _)| *hash);

    CandidateDecision {
        winner_index,
        agreement,
        candidates: candidates.len(),
        answer_hash: winning_key.map(|key| stable_hash(key)),
        vote_tallies,
        verifier,
    }
}

/// Extract a final-answer span without treating arbitrary chain-of-thought
/// text as an answer. If no explicit marker is present, the last non-empty
/// line is the bounded fallback.
pub(crate) fn answer_text(content: &str) -> Option<String> {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return None;
    }

    if let Some(start) = trimmed.rfind("\\boxed{") {
        let after = &trimmed[start + "\\boxed{".len()..];
        let mut depth = 1usize;
        for (index, ch) in after.char_indices() {
            match ch {
                '{' => depth = depth.saturating_add(1),
                '}' => {
                    depth = depth.saturating_sub(1);
                    if depth == 0 {
                        return Some(after[..index].trim().to_string());
                    }
                }
                _ => {}
            }
        }
    }

    let lower = trimmed.to_ascii_lowercase();
    for marker in ["####", "final answer:", "answer:", "result:"] {
        if let Some(index) = lower.rfind(marker) {
            let start = index + marker.len();
            let rest = trimmed[start..].trim();
            if !rest.is_empty() {
                return Some(rest.lines().next().unwrap_or(rest).trim().to_string());
            }
        }
    }

    trimmed
        .lines()
        .rev()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .map(str::to_string)
}

pub(crate) fn answer_hash(content: &str) -> Option<u64> {
    answer_text(content)
        .and_then(|answer| normalize_answer(&answer))
        .map(|answer| stable_hash(&answer))
}

fn response_content(body: &Value) -> Option<&str> {
    body.pointer("/choices/0/message/content")
        .and_then(Value::as_str)
}

fn normalize_answer(content: &str) -> Option<String> {
    let answer = answer_text(content)?;
    let answer = answer
        .trim()
        .trim_matches(|ch: char| matches!(ch, '`' | '$' | '.' | '!' | '?' | ';' | ','));
    if answer.is_empty() {
        return None;
    }
    if let Some(number) = parse_expression(answer) {
        return Some(format!("number:{}", format_number(number)));
    }

    let mut normalized = String::new();
    let mut previous_space = true;
    for ch in answer.chars().flat_map(char::to_lowercase) {
        if ch.is_alphanumeric() {
            normalized.push(ch);
            previous_space = false;
        } else if !previous_space {
            normalized.push(' ');
            previous_space = true;
        }
    }
    let normalized = normalized.trim();
    (!normalized.is_empty()).then(|| format!("text:{normalized}"))
}

fn stable_hash(text: &str) -> u64 {
    // FNV-1a is used only as a request-local answer-key fingerprint in a
    // response receipt. It is not an authentication or integrity primitive.
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in text.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn latest_user_texts(request: &Value) -> Vec<&str> {
    let Some(messages) = request.get("messages").and_then(Value::as_array) else {
        return Vec::new();
    };
    let Some(message) = messages
        .iter()
        .rev()
        .find(|message| message.get("role").and_then(Value::as_str) == Some("user"))
    else {
        return Vec::new();
    };
    text_parts(message.get("content"))
}

fn text_parts(value: Option<&Value>) -> Vec<&str> {
    match value {
        Some(Value::String(text)) => vec![text.as_str()],
        Some(Value::Array(items)) => items
            .iter()
            .filter(|item| item.get("type").and_then(Value::as_str) == Some("text"))
            .filter_map(|item| item.get("text").and_then(Value::as_str))
            .collect(),
        _ => Vec::new(),
    }
}

fn arithmetic_oracle_from_text(text: &str) -> Option<ArithmeticOracle> {
    let lower = text.to_ascii_lowercase();
    let cues = ["calculate ", "compute ", "evaluate ", "what is "];
    for cue in cues {
        let Some(cue_index) = lower.find(cue) else {
            continue;
        };
        let tail = &text[cue_index + cue.len()..];
        let Some(start) =
            tail.find(|ch: char| ch.is_ascii_digit() || matches!(ch, '(' | '+' | '-'))
        else {
            continue;
        };
        let expression_tail = &tail[start..];
        let Some(end) = expression_tail
            .char_indices()
            .take_while(|(_, ch)| {
                ch.is_ascii_digit()
                    || ch.is_ascii_whitespace()
                    || matches!(ch, '.' | '+' | '-' | '*' | '/' | '(' | ')' | '^')
            })
            .map(|(index, ch)| index + ch.len_utf8())
            .last()
        else {
            continue;
        };
        let expression = expression_tail[..end].trim().trim_end_matches('.').trim();
        if !expression
            .chars()
            .any(|ch| matches!(ch, '+' | '-' | '*' | '/' | '^'))
        {
            continue;
        }
        let Some(expected) = parse_expression(expression) else {
            continue;
        };
        return Some(ArithmeticOracle {
            expression: expression.to_string(),
            expected,
        });
    }
    None
}

fn table_oracle_from_text(text: &str) -> Option<TableOracle> {
    let lower = text.to_ascii_lowercase();
    if !["highest", "largest", "maximum", "greatest"]
        .iter()
        .any(|cue| lower.contains(cue))
    {
        return None;
    }

    let mut entries = Vec::<(String, f64)>::new();
    for segment in text.split([',', ';', '\\n']) {
        let segment = segment.trim();
        let Some((label, raw_value)) = segment
            .split_once('=')
            .or_else(|| segment.split_once(':'))
        else {
            continue;
        };
        let label = label.trim();
        let raw_value = raw_value.trim().trim_end_matches('.');
        if label.is_empty()
            || label.len() > 64
            || !label.chars().all(|ch| ch.is_ascii_alphabetic() || ch.is_ascii_whitespace() || ch == '_' || ch == '-')
        {
            continue;
        }
        let Some(value_token) = raw_value
            .split_whitespace()
            .next()
            .map(|token| token.trim_end_matches(['.', '!', '?']))
        else {
            continue;
        };
        let Ok(value) = value_token.parse::<f64>() else {
            continue;
        };
        if !value.is_finite() {
            continue;
        }
        entries.push((label.to_string(), value));
        if entries.len() > 32 {
            return None;
        }
    }
    if entries.len() < 2 {
        return None;
    }
    entries.sort_by(|left, right| left.1.total_cmp(&right.1));
    let (label, value) = entries.pop()?;
    if entries.last().is_some_and(|(_, second)| numbers_equal(*second, value)) {
        // A tied maximum is not a unique oracle.
        return None;
    }
    Some(TableOracle { label, value })
}

fn numeric_answer(content: &str) -> Option<f64> {
    let answer = answer_text(content)?;
    let candidate = answer
        .trim()
        .trim_matches(|ch: char| matches!(ch, '`' | '$' | '.' | '!' | '?' | ';' | ','));
    if let Some(number) = parse_expression(candidate) {
        return Some(number);
    }
    let trimmed = candidate.trim_end_matches(|ch: char| matches!(ch, '.' | '!' | '?'));
    let start = trimmed
        .char_indices()
        .rev()
        .find(|(_, ch)| !ch.is_ascii_digit() && !matches!(ch, '.' | '-' | '+'))
        .map_or(0, |(index, ch)| index + ch.len_utf8());
    parse_expression(trimmed[start..].trim())
}

fn numbers_equal(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1.0e-9_f64.max(right.abs() * 1.0e-9)
}

fn format_number(value: f64) -> String {
    if value.fract().abs() < 1.0e-10 {
        format!("{value:.0}")
    } else {
        format!("{value:.10}")
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string()
    }
}

fn parse_expression(input: &str) -> Option<f64> {
    let mut parser = ArithmeticParser {
        bytes: input.as_bytes(),
        position: 0,
    };
    let value = parser.parse_sum()?;
    parser.skip_space();
    (parser.position == parser.bytes.len() && value.is_finite()).then_some(value)
}

struct ArithmeticParser<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl ArithmeticParser<'_> {
    fn skip_space(&mut self) {
        while self
            .bytes
            .get(self.position)
            .is_some_and(u8::is_ascii_whitespace)
        {
            self.position += 1;
        }
    }

    fn consume(&mut self, byte: u8) -> bool {
        self.skip_space();
        if self.bytes.get(self.position) == Some(&byte) {
            self.position += 1;
            true
        } else {
            false
        }
    }

    fn parse_sum(&mut self) -> Option<f64> {
        let mut value = self.parse_product()?;
        loop {
            if self.consume(b'+') {
                value += self.parse_product()?;
            } else if self.consume(b'-') {
                value -= self.parse_product()?;
            } else {
                break;
            }
        }
        value.is_finite().then_some(value)
    }

    fn parse_product(&mut self) -> Option<f64> {
        let mut value = self.parse_unary()?;
        loop {
            if self.consume(b'*') {
                value *= self.parse_unary()?;
            } else if self.consume(b'/') {
                let divisor = self.parse_unary()?;
                if divisor == 0.0 {
                    return None;
                }
                value /= divisor;
            } else {
                break;
            }
        }
        value.is_finite().then_some(value)
    }

    fn parse_unary(&mut self) -> Option<f64> {
        if self.consume(b'+') {
            self.parse_unary()
        } else if self.consume(b'-') {
            Some(-self.parse_unary()?)
        } else {
            self.parse_power()
        }
    }

    fn parse_power(&mut self) -> Option<f64> {
        let base = self.parse_primary()?;
        if self.consume(b'^') {
            let exponent = self.parse_unary()?;
            let result = base.powf(exponent);
            result.is_finite().then_some(result)
        } else {
            Some(base)
        }
    }

    fn parse_primary(&mut self) -> Option<f64> {
        if self.consume(b'(') {
            let value = self.parse_sum()?;
            if !self.consume(b')') {
                return None;
            }
            return Some(value);
        }

        self.skip_space();
        let start = self.position;
        let mut digits = 0usize;
        let mut decimal_points = 0usize;
        while let Some(byte) = self.bytes.get(self.position) {
            if byte.is_ascii_digit() {
                digits += 1;
                self.position += 1;
            } else if *byte == b'.' {
                decimal_points += 1;
                if decimal_points > 1 {
                    return None;
                }
                self.position += 1;
            } else {
                break;
            }
        }
        if digits == 0 {
            return None;
        }
        std::str::from_utf8(&self.bytes[start..self.position])
            .ok()?
            .parse::<f64>()
            .ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn unit__arithmetic_oracle__extracts_and_evaluates_prompt_expression() {
        let request = json!({
            "messages": [{
                "role": "user",
                "content": "Calculate 17 * (23 - 8) + 144 / 12. Return only the number."
            }]
        });
        let profile = TaskProfile::from_request(&request);
        assert_eq!(profile.math_expected(), Some(267.0));
        assert_eq!(profile.math_expression(), Some("17 * (23 - 8) + 144 / 12"));
        assert_eq!(profile.math_valid("267"), Some(true));
        assert_eq!(profile.math_valid("The answer is 268."), Some(false));
    }

    #[test]
    fn unit__task_profile__uses_latest_user_turn_not_old_math_context() {
        let request = json!({
            "messages": [
                {"role": "user", "content": "Calculate 12 * 12. Return only the number."},
                {"role": "assistant", "content": "144"},
                {"role": "user", "content": "What is the capital of France?"}
            ]
        });
        assert_eq!(TaskProfile::from_request(&request).math_expected(), None);
    }

    #[test]
    fn unit__table_oracle__extracts_unique_maximum_and_checks_label_value_pair() {
        let request = json!({
            "messages": [{
                "role": "user",
                "content": "Oak=12, Pine=19, Birch=16. Return the highest label and value."
            }]
        });
        let profile = TaskProfile::from_request(&request);
        assert_eq!(profile.table_valid("Pine=19"), Some(true));
        assert_eq!(profile.table_valid("Birch=19"), Some(false));
        assert_eq!(profile.table_valid("Pine=18"), Some(false));
        assert!(profile.table_feedback().is_some());
    }

    #[test]
    fn unit__table_oracle__parses_last_row_with_trailing_instruction() {
        let request = json!({
            "messages": [{
                "role": "user",
                "content": "Oak=12, Pine=19, Birch=26. Return the highest label and value."
            }]
        });
        let profile = TaskProfile::from_request(&request);
        assert_eq!(profile.table_label(), Some("Birch"));
        assert_eq!(profile.table_value(), Some(26.0));
        assert_eq!(profile.table_valid("Birch=26"), Some(true));
        assert_eq!(profile.table_valid("Pine=19"), Some(false));
    }

    #[test]
    fn unit__pick_candidate__table_oracle_beats_wrong_majority() {
        let sentinel = crate::sentinel::Sentinel::new(false, 0, None);
        let request = json!({
            "messages": [{
                "role": "user",
                "content": "Oak=12, Pine=19, Birch=26. Return the highest label and value."
            }]
        });
        let candidates = vec![
            (200, json!({"choices":[{"message":{"content":"Pine=19"},"finish_reason":"stop"}]})),
            (200, json!({"choices":[{"message":{"content":"Pine=19"},"finish_reason":"stop"}]})),
            (200, json!({"choices":[{"message":{"content":"Birch=26"},"finish_reason":"stop"}]})),
        ];
        let decision = pick_candidate(&sentinel, &candidates, None, &request);
        assert_eq!(decision.winner_index, 2);
        assert_eq!(decision.agreement, 1);
        assert_eq!(decision.verifier, "table_lookup_exact");
    }

    #[test]
    fn unit__table_oracle__rejects_tied_maximum() {
        let request = json!({
            "messages": [{
                "role": "user",
                "content": "Oak=12, Pine=19, Birch=19. Return the highest label and value."
            }]
        });
        assert_eq!(TaskProfile::from_request(&request).table, None);
    }

    #[test]
    fn unit__arithmetic_parser__precedence_parentheses_and_divide_by_zero() {
        assert_eq!(parse_expression("2 + 3 * (8 - 4)"), Some(14.0));
        assert_eq!(parse_expression("2^3 + 1"), Some(9.0));
        assert_eq!(parse_expression("8 / 0"), None);
        assert_eq!(parse_expression("2 + unknown"), None);
    }

    #[test]
    fn unit__answer_text__prefers_boxed_and_final_answer_markers() {
        assert_eq!(answer_text("work\n\\boxed{42}"), Some("42".into()));
        assert_eq!(
            answer_text("derivation\nFinal answer: 17."),
            Some("17.".into())
        );
    }

    #[test]
    fn unit__answer_hash__normalizes_equivalent_numeric_answers() {
        assert_eq!(
            answer_hash("Final answer: 267."),
            answer_hash("\\boxed{267}")
        );
    }

    #[test]
    fn unit__pick_candidate__exact_math_beats_consensus_wrong_answer() {
        let sentinel = crate::sentinel::Sentinel::new(false, 0, None);
        let request = json!({
            "messages": [{
                "role": "user",
                "content": "What is 17 * (23 - 8) + 144 / 12? Return only the number."
            }]
        });
        let candidates = vec![
            (
                200,
                json!({"choices":[{"message":{"content":"268"},"finish_reason":"stop"}]}),
            ),
            (
                200,
                json!({"choices":[{"message":{"content":"268"},"finish_reason":"stop"}]}),
            ),
            (
                200,
                json!({"choices":[{"message":{"content":"267"},"finish_reason":"stop"}]}),
            ),
        ];
        let decision = pick_candidate(&sentinel, &candidates, None, &request);
        assert_eq!(decision.winner_index, 2);
        assert_eq!(decision.agreement, 1);
        assert_eq!(decision.verifier, "math_exact");
    }

    #[test]
    fn unit__pick_candidate__consensus_breaks_general_answer_ties() {
        let sentinel = crate::sentinel::Sentinel::new(false, 0, None);
        let request = json!({"messages":[{"role":"user","content":"What is the capital?"}]});
        let candidates = vec![
            (
                200,
                json!({"choices":[{"message":{"content":"Final answer: Paris"},"finish_reason":"stop"}]}),
            ),
            (
                200,
                json!({"choices":[{"message":{"content":"Final answer: Rome"},"finish_reason":"stop"}]}),
            ),
            (
                200,
                json!({"choices":[{"message":{"content":"Final answer: Paris"},"finish_reason":"stop"}]}),
            ),
        ];
        let decision = pick_candidate(&sentinel, &candidates, None, &request);
        assert_eq!(decision.winner_index, 0);
        assert_eq!(decision.agreement, 2);
        assert_eq!(decision.candidates, 3);
        assert_eq!(decision.verifier, "answer_consensus");
        assert_eq!(decision.vote_tallies.len(), 2);
        assert_eq!(
            decision
                .vote_tallies
                .iter()
                .map(|(_, votes)| votes)
                .sum::<usize>(),
            3
        );
    }
}
