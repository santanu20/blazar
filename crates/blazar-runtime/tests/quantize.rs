//! quantize orchestration vs the stub llama-quantize: success path,
//! failure path (nonzero exit), and no-output-on-success rejection.

#![allow(non_snake_case)] // unit__scenario__expected naming convention

use blazar_runtime::quantize::{plausible_quant_type, quantize};

// The stub reads BLAZAR_STUB_QUANTIZE_FAIL from the process environment,
// which is global across test threads — serialize the tests that touch it
// (default parallel test execution otherwise races success-vs-failure).
static QUANT_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

// Edition 2024 makes env mutation unsafe (std demands no concurrent
// reader/writer thread). Callers hold QUANT_ENV_LOCK and the stub child
// reads the variable only after spawn — the condition std requires.
fn set_env(key: &str, value: &str) {
    #[expect(unsafe_code)]
    unsafe {
        std::env::set_var(key, value);
    }
}

fn remove_env(key: &str) {
    #[expect(unsafe_code)]
    unsafe {
        std::env::remove_var(key);
    }
}

#[test]
fn unit__quantize_success__copies_and_reports_lines() {
    let _env_guard = QUANT_ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("src.gguf");
    let dst = tmp.path().join("dst.gguf");
    std::fs::write(&src, b"gguf-payload").unwrap();
    let mut lines = Vec::new();
    let out = quantize(
        env!("CARGO_BIN_EXE_stub-llama-quantize"),
        &src,
        &dst,
        "Q4_K_M",
        |l| lines.push(l.to_string()),
    )
    .unwrap();
    assert_eq!(out, dst);
    assert!(dst.exists());
    assert_eq!(std::fs::read(&dst).unwrap(), b"gguf-payload");
    assert!(lines.iter().any(|l| l.contains("quantizing")), "{lines:?}");
    assert!(lines.iter().any(|l| l.contains("success")), "{lines:?}");
}

#[test]
fn unit__quantize_failure__surfaces_stderr_and_no_output() {
    let _env_guard = QUANT_ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("src.gguf");
    let dst = tmp.path().join("dst.gguf");
    std::fs::write(&src, b"gguf-payload").unwrap();
    set_env("BLAZAR_STUB_QUANTIZE_FAIL", "1");
    let err = quantize(
        env!("CARGO_BIN_EXE_stub-llama-quantize"),
        &src,
        &dst,
        "BOGUS",
        |_| {},
    )
    .unwrap_err()
    .to_string();
    remove_env("BLAZAR_STUB_QUANTIZE_FAIL");
    assert!(err.contains("failed"), "{err}");
    assert!(err.contains("unknown quantization type"), "{err}");
    assert!(!dst.exists());
}

#[test]
fn unit__quant_type__shape() {
    assert!(plausible_quant_type("Q4_K_M"));
    assert!(!plausible_quant_type("rm -rf /"));
}
