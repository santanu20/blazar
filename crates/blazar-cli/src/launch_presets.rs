//! Agent-CLI launch presets: per-CLI environment knowledge for
//! `blazar launch`, modelled on the behaviour ollama's launcher
//! popularised but kept env-only (no foreign config writes, no
//! auto-install, no network).
//!
//! The table is pure data; [`resolve`] turns a program word plus the
//! daemon base URL into the argv and env rows the child actually
//! needs. [`render_env`] prints the rows with every secret-bearing
//! value redacted so `--dry-run` can show the plan without leaking
//! keys. The generic path (an unknown CLI word) resolves to exactly
//! the four base-URL rows `blazar launch` has always set, so
//! existing users see byte-identical behaviour.

use std::collections::BTreeSet;
use std::path::Path;

/// Which wire protocol a CLI speaks; selects the base-URL rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Protocol {
    /// `Anthropic` messages API (claude code).
    Anthropic,
    /// `OpenAI` chat/completions API (codex, opencode, cline, aider).
    OpenAi,
    /// `Ollama`-native API (dsh).
    Ollama,
}

/// One row of the preset table.
#[derive(Debug)]
pub struct LaunchPreset {
    /// Canonical name (the word users type).
    pub name: &'static str,
    /// Extra words that select this preset.
    pub aliases: &'static [&'static str],
    /// One-line description for `--list`.
    pub description: &'static str,
    /// Wire protocol driving the base-URL env rows.
    pub protocol: Protocol,
    /// Flag the CLI takes for model selection (`--model` for
    /// claude/codex/aider; `None` when the CLI has no model arg).
    pub model_arg: Option<&'static str>,
    /// Env hygiene rows beyond the protocol rows (Claude Code
    /// deferred-MCP and provider pinning set). Pairs are static
    /// strings; per-launch rows (URLs, keys) are added by resolve.
    pub extra_env: &'static [(&'static str, &'static str)],
    /// (command name probed on PATH, install doc URL) for hints.
    pub install: (&'static str, &'static str),
}

/// The preset table. Order is `--list` order.
pub const LAUNCH_PRESETS: &[LaunchPreset] = &[
    LaunchPreset {
        name: "claude",
        aliases: &["claude-code", "claudecode"],
        description: "Claude Code (Anthropic protocol; deferred MCP, provider pinned here)",
        protocol: Protocol::Anthropic,
        model_arg: Some("--model"),
        extra_env: &[
            ("CLAUDE_CODE_PROVIDER_MANAGED_BY_HOST", "1"),
            ("ENABLE_CLAUDEAI_MCP_SERVERS", "false"),
            ("CLAUDE_CODE_USE_BEDROCK", ""),
            ("CLAUDE_CODE_USE_VERTEX", ""),
            ("CLAUDE_CODE_USE_FOUNDRY", ""),
            ("CLAUDE_CODE_USE_ANTHROPIC_AWS", ""),
            ("CLAUDE_CODE_USE_ANTHROPIC_GOOGLE_CLOUD", ""),
            ("CLAUDE_CODE_USE_MANTLE", ""),
            ("CLAUDE_CODE_USE_GATEWAY", ""),
            ("CLAUDE_CODE_ATTRIBUTION_HEADER", "0"),
            ("CLAUDE_CODE_TOTAL_TOKENS_REMINDER", "off"),
            ("DISABLE_ERROR_REPORTING", "1"),
            ("DISABLE_FEEDBACK_COMMAND", "1"),
            ("CLAUDE_CODE_DISABLE_FEEDBACK_SURVEY", "1"),
            ("ANTHROPIC_MODEL", ""),
            ("ANTHROPIC_SMALL_FAST_MODEL", ""),
        ],
        install: ("claude", "https://docs.anthropic.com/en/docs/claude-code"),
    },
    LaunchPreset {
        name: "codex",
        aliases: &[],
        description: "OpenAI Codex CLI (OpenAI protocol)",
        protocol: Protocol::OpenAi,
        model_arg: Some("--model"),
        extra_env: &[],
        install: ("codex", "https://developers.openai.com/codex/cli"),
    },
    LaunchPreset {
        name: "opencode",
        aliases: &[],
        description: "opencode agent CLI (OpenAI protocol; reads its own config)",
        protocol: Protocol::OpenAi,
        model_arg: None,
        extra_env: &[],
        install: ("opencode", "https://opencode.ai"),
    },
    LaunchPreset {
        name: "cline",
        aliases: &["claude-dev"],
        description: "Cline (OpenAI protocol; runs inside editors)",
        protocol: Protocol::OpenAi,
        model_arg: None,
        extra_env: &[],
        install: ("cline", "https://cline.bot"),
    },
    LaunchPreset {
        name: "aider",
        aliases: &["aider-chat"],
        description: "Aider (OpenAI protocol)",
        protocol: Protocol::OpenAi,
        model_arg: Some("--model"),
        extra_env: &[],
        install: ("aider", "https://aider.chat"),
    },
    LaunchPreset {
        name: "dsh",
        aliases: &[],
        description: "dsh (ollama-native API)",
        protocol: Protocol::Ollama,
        model_arg: None,
        extra_env: &[],
        install: ("dsh", "https://github.com/santanu20/dsh"),
    },
];

/// Persisted last launch: what `--restore` replays.
#[derive(Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LaunchState {
    /// Program word as typed.
    pub program: String,
    /// Warmed model, if the last launch had one.
    pub model: Option<String>,
    /// Key NAME (never the secret) when `--key` named one.
    pub key_name: Option<String>,
    /// Args after the program word.
    pub argv_tail: Vec<String>,
}

/// Everything the child exec needs, resolved from the table.
#[derive(Debug, PartialEq)]
pub struct LaunchPlan {
    /// Program word (PATH lookup happens at exec time).
    pub program: String,
    /// Full argv including an injected model arg, if any.
    pub argv: Vec<String>,
    /// Env rows to set, in order; values may carry secrets.
    pub env: Vec<(String, String)>,
    /// Preset matched (None = generic passthrough).
    pub preset: Option<&'static str>,
}

/// Match a program word against the table (names + aliases).
/// Match a program word against the table (names + aliases).
pub fn find_preset(word: &str) -> Option<&'static LaunchPreset> {
    LAUNCH_PRESETS
        .iter()
        .find(|p| p.name == word || p.aliases.contains(&word))
}

/// Resolve the launch plan. `base` is the daemon base URL
/// (`http://127.0.0.1:PORT`); `http_base` the same without the
/// scheme. `key` is the resolved SECRET (None = no key rows).
/// Resolve the launch plan. `base` is the daemon base URL
/// (`http://127.0.0.1:PORT`); `http_base` the same without the
/// scheme. `key` is the resolved SECRET (None = no key rows).
///
/// Every plan carries the four historic base-URL rows so unknown
/// CLIs and presets share one passthrough contract; a preset adds
/// its hygiene rows and (for Claude Code) model env overrides. The
/// argv injects the preset's model flag only when a model is given -
/// a flag without a value would break the child CLI.
pub fn resolve(
    program: &str,
    user_args: &[String],
    base: &str,
    http_base: &str,
    model: Option<&str>,
    key: Option<&str>,
) -> LaunchPlan {
    let preset = find_preset(program);
    let mut env = vec![
        ("OPENAI_BASE_URL".to_string(), base.to_string()),
        ("OPENAI_API_BASE".to_string(), format!("{base}/v1")),
        ("ANTHROPIC_BASE_URL".to_string(), base.to_string()),
        ("OLLAMA_HOST".to_string(), http_base.to_string()),
    ];
    if let Some(p) = preset {
        for (k, v) in p.extra_env {
            env.push((k.to_string(), v.to_string()));
        }
        // Claude Code reads its default model from env; route the
        // warmed model there so it applies even without the flag.
        if let (Protocol::Anthropic, Some(m)) = (p.protocol, model) {
            env.retain(|(k, _)| k != "ANTHROPIC_MODEL" && k != "ANTHROPIC_SMALL_FAST_MODEL");
            env.push(("ANTHROPIC_MODEL".to_string(), m.to_string()));
            env.push(("ANTHROPIC_SMALL_FAST_MODEL".to_string(), m.to_string()));
        }
    }
    if let Some(secret) = key {
        env.push(("OPENAI_API_KEY".to_string(), secret.to_string()));
        env.push(("ANTHROPIC_API_KEY".to_string(), secret.to_string()));
    }
    let mut argv = vec![program.to_string()];
    if let Some((flag, m)) = preset.and_then(|p| p.model_arg).zip(model) {
        argv.push(flag.to_string());
        argv.push(m.to_string());
    }
    argv.extend(user_args.iter().cloned());
    LaunchPlan {
        program: program.to_string(),
        argv,
        env,
        preset: preset.map(|p| p.name),
    }
}

/// Render env rows for display with secret values redacted.
/// Any row whose key contains KEY, TOKEN, SECRET or PASSWORD
/// shows `<redacted>`.
/// Render env rows for display with secret values redacted. A row
/// is secret-bearing when its key ENDS with `_KEY` or `_TOKEN` or
/// contains `SECRET`/`PASSWORD` - a suffix test, not a substring
/// test, so telemetry counters like `CLAUDE_CODE_TOTAL_TOKENS_REMINDER`
/// (value `off`, not a secret) still render verbatim.
pub fn render_env(env: &[(String, String)]) -> Vec<(String, String)> {
    env.iter()
        .map(|(k, v)| {
            let upper = k.to_ascii_uppercase();
            let redacted = upper.ends_with("_KEY")
                || upper.ends_with("_TOKEN")
                || upper.contains("SECRET")
                || upper.contains("PASSWORD");
            (
                k.clone(),
                if redacted {
                    "<redacted>".to_string()
                } else {
                    v.clone()
                },
            )
        })
        .collect()
}

/// Read the last-launch state file; fails loud on corrupt JSON.
/// Read the last-launch state file; fails loud on corrupt JSON.
pub fn read_state(path: &Path) -> std::io::Result<LaunchState> {
    let raw = std::fs::read_to_string(path)?;
    serde_json::from_str(&raw).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

/// Write the last-launch state (best effort: a read-only config dir
/// must not abort a launch).
/// Write the last-launch state (best effort: a read-only config dir
/// must not abort a launch).
pub fn write_state(path: &Path, state: &LaunchState) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let raw = serde_json::to_string_pretty(state)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    std::fs::write(path, raw)
}

/// Pure battery over the table: unique names + aliases, non-empty
/// descriptions, install hints present, every preset resolves with a
/// non-empty env and a protocol base row, generic passthrough stays
/// the four historic rows. Returns the failures (empty = healthy).
/// Pure battery over the table: unique names + aliases, non-empty
/// descriptions, install hints present, every preset resolves with a
/// non-empty env and a protocol base row, generic passthrough stays
/// the four historic rows. Returns the failures (empty = healthy).
pub fn self_test_failures() -> Vec<String> {
    let mut failures = Vec::new();
    let mut seen = BTreeSet::new();
    for p in LAUNCH_PRESETS {
        if !seen.insert(p.name) {
            failures.push(format!("duplicate preset name {}", p.name));
        }
        for a in p.aliases {
            if !seen.insert(*a) {
                failures.push(format!("duplicate alias {a}"));
            }
        }
        if p.description.is_empty() {
            failures.push(format!("{}: empty description", p.name));
        }
        if p.install.0.is_empty() || p.install.1.is_empty() {
            failures.push(format!("{}: missing install hint", p.name));
        }
        let name = p.name.to_string();
        let plan = resolve(
            &name,
            &[],
            "http://127.0.0.1:11435",
            "127.0.0.1:11435",
            None,
            None,
        );
        if plan.preset != Some(p.name) {
            failures.push(format!("{}: does not resolve to itself", p.name));
        }
        if plan.env.is_empty() {
            failures.push(format!("{}: empty env", p.name));
        }
        let expected = match p.protocol {
            Protocol::Anthropic => ("ANTHROPIC_BASE_URL", "http://127.0.0.1:11435"),
            Protocol::OpenAi => ("OPENAI_BASE_URL", "http://127.0.0.1:11435"),
            Protocol::Ollama => ("OLLAMA_HOST", "127.0.0.1:11435"),
        };
        if !plan
            .env
            .iter()
            .any(|(k, v)| (k.as_str(), v.as_str()) == expected)
        {
            failures.push(format!("{}: protocol row {expected:?} missing", p.name));
        }
        if let Some(flag) = p.model_arg {
            let plan = resolve(
                &name,
                &["extra".to_string()],
                "http://127.0.0.1:11435",
                "127.0.0.1:11435",
                Some("m"),
                None,
            );
            if !(plan.argv.contains(&flag.to_string()) && plan.argv.contains(&"m".to_string())) {
                failures.push(format!("{}: model arg not injected", p.name));
            }
        }
    }
    let generic = resolve(
        "zz-unknown-cli",
        &[],
        "http://127.0.0.1:11435",
        "127.0.0.1:11435",
        None,
        None,
    );
    if generic.preset.is_some() {
        failures.push("generic word matched a preset".to_string());
    }
    if generic.env.len() != 4 {
        failures.push(format!(
            "generic passthrough has {} rows, expected the 4 historic rows",
            generic.env.len()
        ));
    }
    failures
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]
    use super::*;

    fn s(x: &str) -> String {
        x.to_string()
    }

    #[test]
    fn unit__find_preset__matches_names_and_aliases() {
        assert_eq!(find_preset("claude").map(|p| p.name), Some("claude"));
        assert_eq!(find_preset("claude-code").map(|p| p.name), Some("claude"));
        assert_eq!(find_preset("aider-chat").map(|p| p.name), Some("aider"));
        assert!(find_preset("some-unknown-cli").is_none());
        assert!(find_preset("gemini").is_none());
    }

    #[test]
    fn unit__preset_table__names_and_aliases_stay_unique() {
        let mut seen = BTreeSet::new();
        for p in LAUNCH_PRESETS {
            assert!(!seen.contains(p.name), "duplicate name {}", p.name);
            seen.insert(p.name);
            for a in p.aliases {
                assert!(!seen.contains(a), "duplicate alias {a}");
                seen.insert(a);
            }
            assert!(!p.description.is_empty());
            assert!(!p.install.0.is_empty() && !p.install.1.is_empty());
        }
    }

    #[test]
    fn unit__resolve__generic_passthrough_is_the_historic_four_rows() {
        let plan = resolve(
            "some-cli",
            &[s("--flag")],
            "http://127.0.0.1:11435",
            "127.0.0.1:11435",
            None,
            None,
        );
        assert_eq!(plan.preset, None);
        assert_eq!(plan.argv, vec![s("some-cli"), s("--flag")]);
        let keys: Vec<&str> = plan.env.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(
            keys,
            vec![
                "OPENAI_BASE_URL",
                "OPENAI_API_BASE",
                "ANTHROPIC_BASE_URL",
                "OLLAMA_HOST",
            ]
        );
        assert_eq!(plan.env[0].1, "http://127.0.0.1:11435");
        assert_eq!(plan.env[1].1, "http://127.0.0.1:11435/v1");
        assert_eq!(plan.env[3].1, "127.0.0.1:11435");
    }

    #[test]
    fn unit__resolve__claude_preset_carries_the_hygiene_rows() {
        let plan = resolve(
            "claude",
            &[],
            "http://127.0.0.1:11435",
            "127.0.0.1:11435",
            None,
            None,
        );
        assert_eq!(plan.preset, Some("claude"));
        assert!(
            plan.env
                .iter()
                .any(|(k, v)| k == "ENABLE_CLAUDEAI_MCP_SERVERS" && v == "false")
        );
        assert!(
            plan.env
                .iter()
                .any(|(k, v)| k == "CLAUDE_CODE_PROVIDER_MANAGED_BY_HOST" && v == "1")
        );
        assert!(plan.env.iter().any(|(k, _)| k == "ANTHROPIC_BASE_URL"));
        // No fabricated auth without --key.
        assert!(!plan.env.iter().any(|(k, _)| k == "ANTHROPIC_AUTH_TOKEN"));
        assert!(!plan.env.iter().any(|(k, _)| k == "ANTHROPIC_API_KEY"));
    }

    #[test]
    fn unit__resolve__model_arg_injection_orders_program_flag_then_user_args() {
        let plan = resolve(
            "claude",
            &[s("--verbose")],
            "http://127.0.0.1:11435",
            "127.0.0.1:11435",
            Some("qwen3.5-9b"),
            None,
        );
        assert_eq!(plan.preset, Some("claude"));
        assert_eq!(
            plan.argv,
            vec![s("claude"), s("--model"), s("qwen3.5-9b"), s("--verbose")]
        );
        // Warm model should ALSO route Claude's model env rows so the
        // CLI defaults to it even without the flag.
        assert!(
            plan.env
                .iter()
                .any(|(k, v)| k == "ANTHROPIC_MODEL" && v == "qwen3.5-9b")
        );
    }

    #[test]
    fn unit__resolve__openai_preset_gets_openai_rows_and_key() {
        let plan = resolve(
            "codex",
            &[],
            "http://127.0.0.1:11435",
            "127.0.0.1:11435",
            None,
            Some("plm_secret123"),
        );
        assert_eq!(plan.preset, Some("codex"));
        assert_eq!(
            plan.argv,
            vec![s("codex")],
            "model-arg flag only injects when a model is given"
        );
        assert!(
            plan.env
                .iter()
                .any(|(k, v)| k == "OPENAI_API_KEY" && v == "plm_secret123")
        );
    }

    #[test]
    fn unit__render_env__redacts_key_rows_only() {
        let rows = vec![
            (s("OPENAI_BASE_URL"), s("http://127.0.0.1:11435")),
            (s("OPENAI_API_KEY"), s("plm_secret123")),
            (s("ANTHROPIC_AUTH_TOKEN"), s("plm_secret123")),
            (s("CLAUDE_CODE_TOTAL_TOKENS_REMINDER"), s("off")),
        ];
        let out = render_env(&rows);
        assert_eq!(out[0].1, "http://127.0.0.1:11435");
        assert_eq!(out[1].1, "<redacted>");
        assert_eq!(out[2].1, "<redacted>");
        // Suffix rule: a token COUNTER row is not a secret channel.
        assert_eq!(out[3].1, "off");
    }

    #[test]
    fn unit__state__round_trips_and_fails_loud_on_corruption() {
        let dir = std::env::temp_dir().join("blazar-launch-state-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("launch-state.json");
        let state = LaunchState {
            program: "claude".into(),
            model: Some("qwen3.5-9b".into()),
            key_name: Some("work".into()),
            argv_tail: vec![s("--verbose")],
        };
        write_state(&path, &state).unwrap();
        assert_eq!(read_state(&path).unwrap(), state);
        std::fs::write(&path, "{not json").unwrap();
        assert!(read_state(&path).is_err(), "corrupt state must fail loud");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn unit__self_test_failures__healthy_table_is_empty() {
        assert!(self_test_failures().is_empty(), "preset table self-test");
    }
}
