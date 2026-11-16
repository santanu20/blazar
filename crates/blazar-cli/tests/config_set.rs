//! `blazar config set` regression pins.
//!
//! Pins two real bugs found 2026-09-05 (smoke-tested live against a config
//! file ending in `[engine_env]`):
//! 1. String values were written unquoted -> invalid TOML -> every later
//!    `config` command failed ("TOML parse error").
//! 2. NEW keys were appended at EOF -> landed INSIDE the last table
//!    (`[engine_env]`) -> type errors / silently wrong section.
//!
//! Isolation: `XDG_CONFIG_HOME` -> tempdir (`BlazarDirs::from_env` honors XDG).

#![allow(non_snake_case)] // suite convention: unit__scenario__expected (§6b)]

use std::path::Path;

use assert_cmd::Command;
use predicates::boolean::PredicateBooleanExt;

fn blazar(cfg_root: &Path) -> Command {
    let mut c = Command::cargo_bin("blazar").unwrap();
    c.env("XDG_CONFIG_HOME", cfg_root);
    c
}

/// A config file shaped like a real user's: top-level keys, then tables.
fn seed(cfg_root: &Path) {
    std::fs::create_dir_all(cfg_root.join("blazar")).unwrap();
    std::fs::write(
        cfg_root.join("blazar/config.toml"),
        "port = 11434\ndefault_ctx = 16384\n\n[engine_env]\nGGML_VK_VISIBLE_DEVICES = \"1\"\n",
    )
    .unwrap();
}

#[test]
fn e2e__config_set_string_value_quoted_and_placed_above_tables() {
    let tmp = tempfile::tempdir().unwrap();
    let cfg_root = tmp.path();
    seed(cfg_root);

    // Before the fix: "cpu_range = 0-15" -> TOML parse error on next load.
    blazar(cfg_root)
        .args(["config", "set", "cpu_range", "0-15"])
        .assert()
        .success()
        .stdout(predicates::str::contains("cpu_range = \"0-15\""));

    let raw = std::fs::read_to_string(cfg_root.join("blazar/config.toml")).unwrap();
    let key_pos = raw.find("cpu_range").expect("key written");
    let table_pos = raw.find("[engine_env]").expect("table present");
    assert!(
        key_pos < table_pos,
        "new top-level key must land ABOVE the first table header, not inside it:\n{raw}"
    );

    // The next config command must still work (file not corrupted).
    blazar(cfg_root)
        .args(["config", "get", "cpu_range"])
        .assert()
        .success()
        .stdout(predicates::str::contains("0-15"));
}

#[test]
fn e2e__config_set_numeric_value_stays_numeric() {
    let tmp = tempfile::tempdir().unwrap();
    let cfg_root = tmp.path();
    seed(cfg_root);

    blazar(cfg_root)
        .args(["config", "set", "poll", "50"])
        .assert()
        .success();

    // Re-set must REPLACE the existing top-level line (not duplicate it
    // above the tables).
    blazar(cfg_root)
        .args(["config", "set", "poll", "0"])
        .assert()
        .success();

    let raw = std::fs::read_to_string(cfg_root.join("blazar/config.toml")).unwrap();
    assert_eq!(
        raw.matches("poll =").count(),
        1,
        "exactly one poll line:\n{raw}"
    );
    assert!(raw.contains("poll = 0"));
    assert!(raw.contains("[engine_env]"));
}

#[test]
fn e2e__config_set_invalid_value_rejected_file_unchanged() {
    let tmp = tempfile::tempdir().unwrap();
    let cfg_root = tmp.path();
    seed(cfg_root);
    let before = std::fs::read_to_string(cfg_root.join("blazar/config.toml")).unwrap();

    blazar(cfg_root)
        .args(["config", "set", "cpu_range", "5-1"])
        .assert()
        .failure()
        .stderr(predicates::str::contains("rejected, file unchanged"));

    let after = std::fs::read_to_string(cfg_root.join("blazar/config.toml")).unwrap();
    assert_eq!(before, after, "rejected set must not mutate the file");
}

#[test]
fn e2e__config_set_scalar_into_string_knob__teaches_quoted_form() {
    let tmp = tempfile::tempdir().unwrap();
    let cfg_root = tmp.path();
    seed(cfg_root);
    let before = std::fs::read_to_string(cfg_root.join("blazar/config.toml")).unwrap();

    // `0xff` parses as a TOML integer, but cpu_mask is a hex STRING knob.
    // The rejection must teach the exact working command, verified by the
    // CLI itself before the hint is shown.
    blazar(cfg_root)
        .args(["config", "set", "cpu_mask", "0xff"])
        .assert()
        .failure()
        .stderr(predicates::str::contains("rejected, file unchanged"))
        .stderr(predicates::str::contains(
            "hint: cpu_mask takes a string - quote the value: blazar config set cpu_mask \"0xff\"",
        ));

    let after = std::fs::read_to_string(cfg_root.join("blazar/config.toml")).unwrap();
    assert_eq!(before, after, "rejected set must not mutate the file");

    // The taught command must actually work (the hint is verified, not
    // guessed): same value, quoted.
    blazar(cfg_root)
        .args(["config", "set", "cpu_mask", "\"0xff\""])
        .assert()
        .success()
        .stdout(predicates::str::contains("cpu_mask = \"0xff\""));
}

#[test]
fn e2e__config_set_invalid_scalar_no_misleading_hint() {
    let tmp = tempfile::tempdir().unwrap();
    let cfg_root = tmp.path();
    seed(cfg_root);

    // An int knob fed an out-of-range int: quoting would NOT fix it, so
    // no quoting hint may appear (plain rejection, same as before).
    blazar(cfg_root)
        .args(["config", "set", "spec_draft_n_min", "2000"])
        .assert()
        .failure()
        .stderr(predicates::str::contains("rejected, file unchanged"))
        .stderr(predicates::str::contains("hint").not());
}
