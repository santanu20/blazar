//! sglang engine lane argv tests: tool-call detector derivation from
//! the arch row (upstream `auto` cannot see qwen2-family templates)
//! and the base spawn shape (forced loopback, served-model-name).
#![allow(non_snake_case)] // house test-naming: unit__subject__behavior

use blazar_core::profile::{Endpoint, Profile};
use blazar_core::store::ModelRow;
use blazar_runtime::engine_impl::sglang_argv;

fn model(arch: Option<&str>) -> ModelRow {
    ModelRow {
        name: "m".into(),
        repo: "o/r".into(),
        quant: "Q4_K_M".into(),
        path: "/models/m.d".into(),
        bytes: 1000,
        sha256: None,
        mmproj_path: None,
        components: vec![],
        shards: 1,
        arch: arch.map(str::to_string),
        params: None,
        ctx_train: None,
        pulled_at: 0,
        last_used_at: 0,
    }
}

fn profile() -> Profile {
    Profile {
        argv: Vec::new(),
        warnings: Vec::new(),
        ctx: 0,
        gpu: "auto",
        kv_est_bytes: None,
        ctx_autofit: None,
    }
}

fn argv(arch: Option<&str>) -> Vec<String> {
    sglang_argv(
        &model(arch),
        &profile(),
        &Endpoint::Tcp {
            host: "127.0.0.1".into(),
            port: 41809,
        },
    )
}

fn parser_value(argv: &[String]) -> String {
    let i = argv
        .iter()
        .position(|a| a == "--tool-call-parser")
        .expect("flag present");
    argv[i + 1].clone()
}

#[test]
fn unit__sglang_argv__base_shape_forced_loopback() {
    let argv = argv(Some("llama"));
    let host = argv
        .windows(2)
        .find(|w| w[0] == "--host")
        .map(|w| w[1].clone())
        .expect("host flag");
    assert_eq!(host, "127.0.0.1");
    assert!(
        argv.windows(2)
            .any(|w| w[0] == "--served-model-name" && w[1] == "m")
    );
}

#[test]
fn unit__sglang_argv__qwen2_gguf_arch_maps_to_qwen25() {
    assert_eq!(parser_value(&argv(Some("qwen2"))), "qwen25");
}

#[test]
fn unit__sglang_argv__qwen2_hf_class_maps_to_qwen25() {
    assert_eq!(parser_value(&argv(Some("Qwen2ForCausalLM"))), "qwen25");
}

#[test]
fn unit__sglang_argv__qwen25_instruct_dir_keeps_qwen25() {
    assert_eq!(parser_value(&argv(Some("Qwen2.5ForCausalLM"))), "qwen25");
}

#[test]
fn unit__sglang_argv__qwen3_gguf_arch_maps_to_qwen() {
    assert_eq!(parser_value(&argv(Some("qwen3"))), "qwen");
}

#[test]
fn unit__sglang_argv__qwen3_hf_class_maps_to_qwen() {
    assert_eq!(parser_value(&argv(Some("Qwen3ForCausalLM"))), "qwen");
}

#[test]
fn unit__sglang_argv__qwen35_and_vl_variants_map_to_qwen() {
    assert_eq!(parser_value(&argv(Some("qwen35"))), "qwen");
    assert_eq!(parser_value(&argv(Some("qwen3vl"))), "qwen");
    assert_eq!(
        parser_value(&argv(Some("Qwen3_5ForConditionalGeneration"))),
        "qwen"
    );
}

#[test]
fn unit__sglang_argv__qwen3_coder_maps_to_dedicated_detector() {
    assert_eq!(parser_value(&argv(Some("qwen3coder"))), "qwen3_coder");
    assert_eq!(parser_value(&argv(Some("qwen3_coder"))), "qwen3_coder");
}

#[test]
fn unit__sglang_argv__unknown_llama_none_keep_auto() {
    assert_eq!(parser_value(&argv(Some("llama"))), "auto");
    assert_eq!(parser_value(&argv(None)), "auto");
    assert_eq!(parser_value(&argv(Some("GemmaForCausalLM"))), "auto");
    assert_eq!(parser_value(&argv(Some("qwen"))), "auto");
}

#[test]
fn unit__sglang_argv__tool_parser_emitted_exactly_once() {
    let argv = argv(Some("qwen2"));
    assert_eq!(
        argv.iter().filter(|a| *a == "--tool-call-parser").count(),
        1
    );
}
