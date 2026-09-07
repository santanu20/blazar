//! quantize orchestration vs the stub llama-quantize: success path,
//! failure path (nonzero exit), and no-output-on-success rejection.

#![allow(non_snake_case)] // unit__scenario__expected naming convention

use pallama_runtime::quantize::{plausible_quant_type, quantize};

#[test]
fn unit__quantize_success__copies_and_reports_lines() {
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
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("src.gguf");
    let dst = tmp.path().join("dst.gguf");
    std::fs::write(&src, b"gguf-payload").unwrap();
    std::env::set_var("PALLAMA_STUB_QUANTIZE_FAIL", "1");
    let err = quantize(
        env!("CARGO_BIN_EXE_stub-llama-quantize"),
        &src,
        &dst,
        "BOGUS",
        |_| {},
    )
    .unwrap_err()
    .to_string();
    std::env::remove_var("PALLAMA_STUB_QUANTIZE_FAIL");
    assert!(err.contains("failed"), "{err}");
    assert!(err.contains("unknown quantization type"), "{err}");
    assert!(!dst.exists());
}

#[test]
fn unit__quant_type__shape() {
    assert!(plausible_quant_type("Q4_K_M"));
    assert!(!plausible_quant_type("rm -rf /"));
}
