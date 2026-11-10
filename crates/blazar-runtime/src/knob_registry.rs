//! Runtime half of the emission-truth knob registry: folds the
//! spawn-time argv builders into the compile-time lane unions from
//! [`blazar_core::knob_registry`], and adds the whisper, piper and
//! openaihost lanes (whisper composes its argv at spawn, piper per
//! synthesis call; openaihost's grammar lives in the engine trait,
//! pinned to live emission by the test in this module).
//!
//! llama.cpp and stable-diffusion.cpp children run `profile.argv`
//! verbatim, so the compile-time slice is already their full spawn
//! truth. mistral.rs, `SGLang` and `MLX` translate or extend the profile in
//! their builders; running those builders over the same compiled
//! fixtures here is what makes the registry emission truth instead of
//! source-literal truth. `scripts/first_class_flags.json` is generated
//! from THIS module by `cargo run -p blazar-runtime --example
//! knob_registry`, and `tests/knob_registry.rs` pins the JSON against
//! live emission in both directions.

use std::collections::{BTreeMap, BTreeSet};

use blazar_core::ModelRow;
use blazar_core::knob_registry::{MISTRALRS_SUPPORTED, is_flag_token};
use blazar_core::profile::Endpoint;

use crate::engine_impl::{mistralrs_argv, mlx_argv, sglang_argv};
use crate::piper;
use crate::whisper;

/// First-class flag tokens per lane (compile + spawn-time builders),
/// sorted, deduplicated. Keys are the lane names used by
/// `scripts/first_class_flags.json` and the capability auditor.
#[must_use]
pub fn lane_flags() -> BTreeMap<&'static str, Vec<String>> {
    let mut lanes = blazar_core::knob_registry::lane_flags();
    let profiles = blazar_core::knob_registry::lane_profiles();
    let endpoint = Endpoint::Tcp {
        host: "127.0.0.1".into(),
        port: 12_345,
    };

    // mistral.rs: the builder swaps loader flags by row shape (GGUF file
    // vs HF/safetensors dir), so both rows ride every fixture profile.
    // The manifest is the lane's supported superset - fixture posture,
    // same rule as the compile half.
    let gguf_row = model_row(
        "/models/qwen3.5-9b/qwen3.5-9b-q4_k_m.gguf",
        None,
        Some("/models/qwen3.5-9b/mmproj.gguf"),
    );
    let hf_row = model_row(&hf_dir_fixture(), None, None);
    let manifest: BTreeSet<String> = MISTRALRS_SUPPORTED
        .iter()
        .map(|f| (*f).to_string())
        .collect();
    let mut union: BTreeSet<String> = lanes["mistralrs"].iter().cloned().collect();
    for profile in &profiles["mistralrs"] {
        for row in [&gguf_row, &hf_row] {
            union.extend(
                mistralrs_argv(row, profile, &endpoint, &manifest)
                    .into_iter()
                    .filter(|t| is_flag_token(t)),
            );
        }
    }
    lanes.insert("mistralrs", union.into_iter().collect());

    // SGLang: builder pins transport + tool-call parser + served name.
    let sglang_row = model_row("/models/qwen3.5-9b.d", Some("qwen3"), None);
    let mut union: BTreeSet<String> = lanes["sglang"].iter().cloned().collect();
    for profile in &profiles["sglang"] {
        union.extend(
            sglang_argv(&sglang_row, profile, &endpoint)
                .into_iter()
                .filter(|t| is_flag_token(t)),
        );
    }
    lanes.insert("sglang", union.into_iter().collect());

    // MLX: builder pins transport only.
    let mlx_row = model_row("/models/qwen3.5-9b-mlx.d", None, None);
    let mut union: BTreeSet<String> = lanes["mlx"].iter().cloned().collect();
    for profile in &profiles["mlx"] {
        union.extend(
            mlx_argv(&mlx_row, profile, &endpoint)
                .into_iter()
                .filter(|t| is_flag_token(t)),
        );
    }
    lanes.insert("mlx", union.into_iter().collect());

    // Whisper: composed directly at spawn, no profile compiler.
    lanes.insert("whisper", whisper::knob_surface());

    // Piper: one-shot synthesis spawns, argv composed per call.
    lanes.insert("piper", piper::knob_surface());

    // OpenAiHost: argv grammar is owned by the engine trait itself - the
    // child is registry-wide (no model row pins a listen pair), the
    // supervisor's synthetic row rides an empty profile, and
    // `OpenAiHostEngine::build_argv` composes `serve --host <h>
    // --port <p>`. Pinned to that live emission by the test below.
    lanes.insert("openaihost", vec!["--host".into(), "--port".into()]);

    lanes
}

/// Store row fixture for the spawn-time builders: the fields they read
/// are `path` (loader dialect), `arch` (sglang tool-call parser),
/// `mmproj_path` (mistral.rs `--mmproj`), and `name` (served-model
/// stamp). Everything else is inert filler.
fn model_row(path: &str, arch: Option<&str>, mmproj: Option<&str>) -> ModelRow {
    ModelRow {
        name: "qwen3.5-9b".into(),
        repo: "Qwen/Qwen3.5-9B".into(),
        quant: "Q4_K_M".into(),
        path: path.into(),
        bytes: 5_242_880_000,
        sha256: None,
        mmproj_path: mmproj.map(Into::into),
        components: Vec::new(),
        shards: 1,
        arch: arch.map(Into::into),
        params: Some(9.0),
        ctx_train: Some(40_960),
        pulled_at: 0,
        last_used_at: 0,
    }
}

/// Real (empty) directory: `mistralrs_argv` picks `-m` vs `-f` by
/// `path.is_dir()`, and the HF posture must take the `-m` branch to be
/// represented in the registry.
fn hf_dir_fixture() -> String {
    let dir = std::env::temp_dir().join(format!("blazar-knob-registry-hf-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("mistral.rs HF-dir fixture");
    dir.to_string_lossy().into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Manifest;
    use crate::engine_impl::Engine;

    /// The static `openaihost` lane above must stay identical to what
    /// `OpenAiHostEngine::build_argv` emits for the supervisor's real
    /// spawn shape: the synthetic `_openaihost` row + the empty profile
    /// built at spawn (`supervisor.rs` `spawn_openaihost_instance`) +
    /// a TCP endpoint. If the launch grammar grows, this pin forces the
    /// registry to grow with it (regenerate the committed JSON).
    #[test]
    #[allow(non_snake_case)]
    fn unit__knob_registry__openaihost_lane_matches_live_build_argv_emission() {
        // The manifest is inert to argv composition; minimal JSON walks
        // the same decode path as an installed lane's stored manifest.
        let manifest: Manifest = serde_json::from_str(
            r#"{"tag":"litert-0","build_number":0,"version_raw":"0",
                "devices":[],"flags":[],"spec_types":[],"server_path":"x"}"#,
        )
        .expect("minimal manifest decodes");
        let engine = crate::OpenAiHostEngine::new(manifest);

        // argv grammar ignores the model row (`_model`); inert filler.
        let row = model_row("/models/smollm2-360m.litertlm", None, None);
        let profile = blazar_core::Profile {
            argv: Vec::new(),
            warnings: Vec::new(),
            ctx: 0,
            gpu: "auto",
            vram_mib: 0,
            kv_est_bytes: None,
            ctx_autofit: None,
            capacity: blazar_core::profile::CapacityStatus::RuntimeManaged,
        };
        let endpoint = Endpoint::Tcp {
            host: "127.0.0.1".into(),
            port: 12_345,
        };
        let emitted: BTreeSet<String> = engine
            .build_argv(&row, &profile, &endpoint)
            .into_iter()
            .filter(|t| is_flag_token(t))
            .collect();

        let lane: BTreeSet<String> = lane_flags()["openaihost"].clone().into_iter().collect();
        assert_eq!(
            emitted, lane,
            "openaihost registry lane must equal live build_argv emission"
        );
    }
}
