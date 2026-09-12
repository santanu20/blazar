//! Pallama runtime: async engine management, HF client, process
//! supervision, event bus, bench runner.

/// File name of a llama.cpp tool binary on the current platform
/// (`llama-quantize` vs `llama-quantize.exe`) — F91: every discovery
/// site must go through this, not hand-join the unix name.
#[must_use]
pub fn tool_file_name(base: &str) -> String {
    if cfg!(windows) {
        format!("{base}.exe")
    } else {
        base.to_string()
    }
}

pub mod bench;
pub mod daemon;
pub mod engine;
pub mod engine_impl;
pub mod events;
pub mod hf;
pub mod hf_parallel;
pub mod models;
pub mod probe;
pub mod quantize;
pub mod registry;
pub mod sessionreg;
pub mod supervisor;
pub mod upgrade;
pub mod whisper;

pub use bench::{parse_bench_json, BenchRow, Tuner};
pub use daemon::{process_alive_by_pid, wait_for_shutdown_signal, DaemonLock};
pub use engine::gh::GhClient;
pub use engine::manifest::{probe as probe_manifest, Manifest, Vendor};
pub use engine::{system_vendor_hint, EngineManager, LOCAL_TAG};
pub use engine_impl::{ChildHandle, Engine, LlamaCppEngine, MistralRsEngine};
pub use events::{EventBus, InstanceState, PallamaEvent};
pub use hf::{parse_pull_target, registry_name, PullTarget, Puller};
pub use models::{instance_running, remove_model};
pub use probe::probe_hardware;
pub use supervisor::{EngineRef, PrefixKey, PsRow, SupervisionError, Supervisor, ROUTER_KEY};
