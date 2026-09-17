//! Throwaway probe: compile the real qwen3.5-9b profile with the live
//! vulkan census + real mmproj and print slots + warnings.
//! Run: `cargo run --example slots_probe -p pallama-core -- <model.gguf> <mmproj.gguf>`
use pallama_core::{
    gguf::read_metadata_file,
    profile::{compile, ProfileInput, TuningOverrides},
    Config, Endpoint, GpuInfo, Hardware, ModelMeta, ModelOverride,
};
use std::collections::BTreeSet;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let (model, mmproj) = match (args.next(), args.next()) {
        (Some(m), Some(p)) => (m, p),
        _ => {
            eprintln!(
                "usage: cargo run --example slots_probe -p pallama-core -- \
                 <model.gguf> <mmproj.gguf>"
            );
            std::process::exit(2);
        }
    };
    let gguf = read_metadata_file(std::path::Path::new(&model))?;
    println!("train ctx = {:?}", gguf.context_length);
    let hw = Hardware {
        physical_cores: 16,
        total_ram_mib: 13_675,
        gpus: vec![
            GpuInfo {
                name: "Vulkan0".into(),
                description: "Intel(R) Graphics (RPL-S)".into(),
                total_mib: 10_256,
                free_mib: 8_494,
            },
            GpuInfo {
                name: "Vulkan1".into(),
                description: "NVIDIA GeForce RTX 4070 Laptop GPU".into(),
                total_mib: 8_188,
                free_mib: 7_790,
            },
        ],
    };
    let flags: BTreeSet<String> = [
        "--ctx-size",
        "--threads",
        "--gpu-layers",
        "--flash-attn",
        "--cache-type-k",
        "--cache-type-v",
        "--cache-reuse",
        "--kv-unified",
        "-mm",
        "-np",
        "--cache-ram",
        "--slot-save-path",
        "--api-key-file",
        "--metrics",
        "--jinja",
        "--sleep-idle-seconds",
        "--device",
        "--alias",
        "--host",
        "--port",
    ]
    .into_iter()
    .map(String::from)
    .collect();
    let overlay = ModelOverride::default();
    let input = ProfileInput {
        engine_kind: pallama_core::engine_kind::EngineKind::LlamaCpp,
        model_name: "qwen3.5-9b",
        instance_key: "qwen3.5-9b",
        model_path: &model,
        model_bytes: std::fs::metadata(&model)?.len(),
        meta: ModelMeta::Gguf(&gguf),
        hardware: &hw,
        config: &Config::default(),
        overlay: &overlay,
        loras: &[],
        draft_path: None,
        draft_gguf: None,
        mmproj_path: Some(&mmproj),
        engine_tag: "b10809",
        supported_flags: &flags,
        spec_types: &[],
        endpoint: Endpoint::Tcp {
            host: "127.0.0.1".into(),
            port: 1,
        },
        data_dir: "/tmp/pallama-slots-probe",
        cache_hit_rate: None,
        resident_ram_mib: 0,
        device_hint: None,
        mmproj_force: false,
        engine_census: hw.gpus.clone(),
        sibling_devices: Vec::new(),
        auto_tensor_split: None,
    };
    let p = compile(&input, &TuningOverrides::default())?;
    println!(
        "argv np/ctx: {:?}",
        p.argv
            .windows(2)
            .filter(|w| w[0] == "-np" || w[0] == "--ctx-size")
            .collect::<Vec<_>>()
    );
    println!("warnings:");
    for w in &p.warnings {
        println!("  - {w}");
    }
    Ok(())
}
